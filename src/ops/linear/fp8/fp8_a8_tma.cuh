#pragma once

// Row-scaled E4M3 weight x row-scaled E4M3 activation GEMM, staged by TMA with a dedicated
// producer warp (upstream PR #167 by MichaelDementii, carried onto the unified A8 template).
//
// Same arithmetic as fp8_a8_mma.cuh: the FP32 accumulator, m16n8k32, the scale application, the
// epilogue and the output policy are unchanged. What differs is the CTA's shape and how it is fed.
// A 256-token x 128-row tile with 64x64 consumer warp tiles issues four times the MMA per
// ldmatrix of the cp.async route's 64x128 tile; one producer lane feeds it through a four-stage
// K=64 mbarrier ring with cp.async.bulk.tensor, instead of the CTA-wide barrier the cp.async K
// loop takes on every stage. Occupancy falls to one CTA of 288 threads per SM, so the route only
// pays at prefill widths; fp8_a8_tma_applies decides where.
//
// Where the width is a whole number of the cp.async route's token tiles the two routes agree byte
// for byte; at other widths the partial tile rounds differently and the Op tests check both
// routes against the host reference.

#include "core/device.h"
#include "core/tma_descriptor_staging.cuh"
#include "ops/common/math.cuh"
#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/vector_output.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_schedule.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

struct alignas(128) Fp8A8TmaDescriptors {
    CUtensorMap a_codes;
    CUtensorMap b_codes;
};

#ifdef _WIN32
// The single Windows descriptor staging for Fp8A8TmaDescriptors, shared by every FP8 A8 TMA
// launch. See core/tma_descriptor_staging.cuh for the invariants its device buffer relies on.
inline TmaDescriptorStaging<Fp8A8TmaDescriptors>& fp8_a8_tma_descriptor_staging() {
    static TmaDescriptorStaging<Fp8A8TmaDescriptors> staging;
    return staging;
}
#endif

template <int BlockTokens, int Stages>
struct Fp8A8TmaSchedule {
    static_assert(BlockTokens == 64 || BlockTokens == 128 || BlockTokens == 256);
    static_assert(Stages >= 2 && Stages <= 8);

    static constexpr int kBlockTokens     = BlockTokens;
    static constexpr int kBlockRows       = 128;
    static constexpr int kBlockK          = 64;
    static constexpr int kStages          = Stages;
    // The routing cost model reads this as achieved occupancy (see fp8_a8_tma_cheaper).
    static constexpr int kMinBlocksPerSm  = 1;
    static constexpr int kWarpsTokens     = 4;
    static constexpr int kWarpsRows       = 2;
    static constexpr int kConsumerWarps   = kWarpsTokens * kWarpsRows;
    static constexpr int kConsumerThreads = kConsumerWarps * 32;
    // One producer warp; only its lane 0 issues TMA. No setmaxnreg register transfer, so the
    // kernel is safe in the relocatable ops archive.
    static constexpr int kProducerThreads = 32;
    static constexpr int kThreads         = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpTokens      = kBlockTokens / kWarpsTokens;
    static constexpr int kWarpRows        = kBlockRows / kWarpsRows;
    static constexpr int kMmaTokens       = kWarpTokens / 16;
    static constexpr int kMmaRows         = kWarpRows / 8;
    static constexpr int kMmaKPerStage    = kBlockK / 32;
    static_assert(kWarpTokens % 16 == 0 && kWarpRows % 8 == 0);
};

// The schedule upstream measured: 256 tokens, four stages of K=64.
using Fp8A8TmaPrefill = Fp8A8TmaSchedule<256, 4>;

// The 64-byte TMA swizzle is a function of the shared address: the segment of a row moves to
// segment XOR ((row / 2) % 4), a pattern that closes after eight 64-byte rows. The mapping in
// fp8_a8_tma_shared_byte is the hardware's only for a tile whose base sits at the start of such a
// 512-byte atom; 128 bytes past one, all 1024 bytes of a tile de-swizzle wrongly. Every tile base
// carries that alignment: the storage, each stage stride, and the allocation itself.
inline constexpr std::size_t kFp8A8TmaSwizzleAtomBytes = 512;

template <class Schedule>
struct alignas(kFp8A8TmaSwizzleAtomBytes) Fp8A8TmaTensorStorage {
    alignas(kFp8A8TmaSwizzleAtomBytes)
        std::uint8_t a_codes[Schedule::kStages][Schedule::kBlockTokens * Schedule::kBlockK];
    alignas(kFp8A8TmaSwizzleAtomBytes)
        std::uint8_t b_codes[Schedule::kStages][Schedule::kBlockRows * Schedule::kBlockK];
    static_assert((Schedule::kBlockTokens * Schedule::kBlockK) % kFp8A8TmaSwizzleAtomBytes == 0,
                  "each activation stage must begin on a swizzle atom");
    static_assert((Schedule::kBlockRows * Schedule::kBlockK) % kFp8A8TmaSwizzleAtomBytes == 0,
                  "each weight stage must begin on a swizzle atom");
};

template <class Schedule>
union alignas(kFp8A8TmaSwizzleAtomBytes) Fp8A8TmaScratch {
    Fp8A8TmaTensorStorage<Schedule> tensors;
    __nv_bfloat16 output[Schedule::kBlockTokens * (Schedule::kBlockRows + 8)];
};

template <class Schedule>
struct alignas(kFp8A8TmaSwizzleAtomBytes) Fp8A8TmaSharedStorage {
    Fp8A8TmaScratch<Schedule> scratch;
    alignas(8) std::uint64_t full[Schedule::kStages];
    alignas(8) std::uint64_t empty[Schedule::kStages];
};

__device__ __forceinline__ int fp8_a8_tma_shared_byte(int row, int logical_byte) {
    return ((logical_byte >> 4) ^ ((row >> 1) & 3)) * 16 + (logical_byte & 15);
}

__device__ __forceinline__ void fp8_a8_tma_load_2d(void* destination,
                                                   const CUtensorMap* descriptor,
                                                   std::int32_t coordinate0,
                                                   std::int32_t coordinate1,
                                                   std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :
                 : "r"(smem_addr(destination)), "l"(descriptor), "r"(coordinate0), "r"(coordinate1),
                   "r"(smem_addr(barrier))
                 : "memory");
}

// Token tiles run fastest, so the CTAs that share a weight tile are resident together and the
// weight matrix streams from DRAM once, as the cp.async route's TokenFast raster does.
__device__ __forceinline__ void fp8_a8_tma_raster_blocks(int& row_tile, int& token_tile) {
    const int token_tiles = static_cast<int>(gridDim.y);
    const int linear =
        static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    token_tile = linear % token_tiles;
    row_tile   = linear / token_tiles;
}

template <class Schedule, class Epilogue, class Output>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a8_tma_kernel(
#ifdef _WIN32
    // MSVC cannot pass the over-aligned CUtensorMap struct by value as a __grid_constant__
    // parameter (C2719), so on Windows the descriptors are pointer-passed from the staging
    // buffer, and the epilogue and output stay ordinary by-value parameters.
    const Fp8A8TmaDescriptors* descriptors_pointer, const Epilogue epilogue, const Output output,
#else
    const __grid_constant__ Fp8A8TmaDescriptors descriptors,
    const __grid_constant__ Epilogue epilogue, const __grid_constant__ Output output,
#endif
    const float* __restrict__ activation_scales, const __nv_bfloat16* __restrict__ weight_scales,
    int tokens, int k) {
#ifdef _WIN32
    const Fp8A8TmaDescriptors& descriptors = *descriptors_pointer;
#endif
    extern __shared__ __align__(kFp8A8TmaSwizzleAtomBytes) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<Fp8A8TmaSharedStorage<Schedule>*>(shared_bytes);
    // The swizzle indexing is correct only on an atom-aligned allocation; stop rather than
    // compute garbage if a toolchain ever places it otherwise.
    if (threadIdx.x == 0 && (smem_addr(shared_bytes) % kFp8A8TmaSwizzleAtomBytes) != 0) __trap();

    int row_tile   = 0;
    int token_tile = 0;
    fp8_a8_tma_raster_blocks(row_tile, token_tile);
    const int token_begin = token_tile * Schedule::kBlockTokens;
    const int row_begin   = row_tile * Schedule::kBlockRows;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Schedule::kStages; ++stage) {
            cta_mbarrier_init(&shared.full[stage], 1);
            cta_mbarrier_init(&shared.empty[stage], Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    const int k_tiles = k / Schedule::kBlockK;

    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
#ifdef _WIN32
            // A staged tensor map was written through the generic proxy; each 128-byte map needs
            // its own acquire for the TMA (tensormap) proxy before its first use.
            acquire_staged_tensor_map(&descriptors.a_codes);
            acquire_staged_tensor_map(&descriptors.b_codes);
#endif
#pragma unroll 1
            for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                const int stage                 = k_tile % Schedule::kStages;
                const std::uint32_t empty_phase = 1U ^ ((k_tile / Schedule::kStages) & 1U);
                cta_mbarrier_wait(&shared.empty[stage], empty_phase);
                constexpr std::uint32_t kTransactionBytes =
                    (Schedule::kBlockTokens + Schedule::kBlockRows) * Schedule::kBlockK;
                cta_mbarrier_arrive_expect_tx(&shared.full[stage], kTransactionBytes);
                auto& tensors = shared.scratch.tensors;
                fp8_a8_tma_load_2d(tensors.a_codes[stage], &descriptors.a_codes,
                                   k_tile * Schedule::kBlockK, token_begin, &shared.full[stage]);
                fp8_a8_tma_load_2d(tensors.b_codes[stage], &descriptors.b_codes,
                                   k_tile * Schedule::kBlockK, row_begin, &shared.full[stage]);
            }
        }
        return;
    }

    auto& tensors             = shared.scratch.tensors;
    const int consumer_thread = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int lane            = consumer_thread & 31;
    const int warp            = consumer_thread >> 5;
    const int warp_token      = warp / Schedule::kWarpsRows;
    const int warp_row        = warp - warp_token * Schedule::kWarpsRows;

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;

    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
#pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage                = k_tile % Schedule::kStages;
        const std::uint32_t full_phase = (k_tile / Schedule::kStages) & 1U;
        cta_mbarrier_wait(&shared.full[stage], full_phase);
#pragma unroll
        for (int k_step = 0; k_step < Schedule::kMmaKPerStage; ++k_step) {
            unsigned a_fragments[Schedule::kMmaTokens][4];
            unsigned b_fragments[Schedule::kMmaRows][2];
#pragma unroll
            for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
                const int row = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
                const auto* address = tensors.a_codes[stage] + row * Schedule::kBlockK +
                                      fp8_a8_tma_shared_byte(row, k_step * 32 + a_column_byte);
                ldmatrix_x4(a_fragments[mma_token][0], a_fragments[mma_token][1],
                            a_fragments[mma_token][2], a_fragments[mma_token][3],
                            smem_addr(address));
            }
#pragma unroll
            for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                const int row       = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
                const auto* address = tensors.b_codes[stage] + row * Schedule::kBlockK +
                                      fp8_a8_tma_shared_byte(row, k_step * 32 + b_column_byte);
                ldmatrix_x2(b_fragments[mma_row][0], b_fragments[mma_row][1], smem_addr(address));
            }
#pragma unroll
            for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
                for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                    mma_fp8_e4m3(
                        accumulators[mma_token][mma_row][0], accumulators[mma_token][mma_row][1],
                        accumulators[mma_token][mma_row][2], accumulators[mma_token][mma_row][3],
                        a_fragments[mma_token][0], a_fragments[mma_token][1],
                        a_fragments[mma_token][2], a_fragments[mma_token][3],
                        b_fragments[mma_row][0], b_fragments[mma_row][1]);
                }
            }
        }
        if (lane == 0) cta_mbarrier_arrive(&shared.empty[stage]);
    }

    // The epilogue reuses the tensor pipeline's storage: every consumer warp must finish its last
    // tensor read before any warp overwrites it. The producer warp has exited, so the consumers
    // synchronise on a named barrier of their own.
    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");

    constexpr int kOutputStride = Schedule::kBlockRows + 8;
    auto* shared_output         = shared.scratch.output;
    const int accumulator_token = lane >> 2;
    const int accumulator_row   = 2 * (lane & 3);
#pragma unroll
    for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
        const int local_token0 =
            warp_token * Schedule::kWarpTokens + mma_token * 16 + accumulator_token;
        const int local_token1 = local_token0 + 8;
        const int token0       = token_begin + local_token0;
        const int token1       = token_begin + local_token1;
        // TMA zero-fills the tokens past the end; they are never stored, and nothing indexed by
        // token (the activation scale, a residual) is read on their behalf.
        const bool valid0             = token0 < tokens;
        const bool valid1             = token1 < tokens;
        const float activation_scale0 = valid0 ? __ldg(activation_scales + token0) : 0.0F;
        const float activation_scale1 = valid1 ? __ldg(activation_scales + token1) : 0.0F;
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            const int local_row0 = warp_row * Schedule::kWarpRows + mma_row * 8 + accumulator_row;
            const int row0       = row_begin + local_row0;
            const float2 weight_scale =
                bf16x2_bits_to_float2(load_ldg<std::uint32_t>(weight_scales + row0));
            float value00 =
                accumulators[mma_token][mma_row][0] * activation_scale0 * weight_scale.x;
            float value01 =
                accumulators[mma_token][mma_row][1] * activation_scale0 * weight_scale.y;
            float value10 =
                accumulators[mma_token][mma_row][2] * activation_scale1 * weight_scale.x;
            float value11 =
                accumulators[mma_token][mma_row][3] * activation_scale1 * weight_scale.y;
            if (valid0) {
                value00 = epilogue.apply(row0, token0, value00);
                value01 = epilogue.apply(row0 + 1, token0, value01);
            }
            if (valid1) {
                value10 = epilogue.apply(row0, token1, value10);
                value11 = epilogue.apply(row0 + 1, token1, value11);
            }
            *reinterpret_cast<__nv_bfloat162*>(shared_output + local_token0 * kOutputStride +
                                               local_row0) = __floats2bfloat162_rn(value00, value01);
            *reinterpret_cast<__nv_bfloat162*>(shared_output + local_token1 * kOutputStride +
                                               local_row0) = __floats2bfloat162_rn(value10, value11);
        }
    }

    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");
    constexpr int kVectorsPerToken = Schedule::kBlockRows / 8;
    constexpr int kOutputVectors   = Schedule::kBlockTokens * kVectorsPerToken;
    for (int task = consumer_thread; task < kOutputVectors; task += Schedule::kConsumerThreads) {
        const int local_token = task / kVectorsPerToken;
        const int token       = token_begin + local_token;
        if (token >= tokens) continue;
        const int row_vector = task - local_token * kVectorsPerToken;
        const uint4 values =
            load_vec<uint4>(shared_output + local_token * kOutputStride + row_vector * 8);
        linear_store_bf16_vector(output, row_begin + row_vector * 8, token, values);
    }
}

inline void fp8_a8_tma_check_driver(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* name = nullptr;
    (void)cuGetErrorName(status, &name);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name != nullptr ? name : "CUDA error"));
}

// A 2-D uint8 map over a K-contiguous [rows, K] matrix with a 64-byte-wide box, the tile the 64B
// swizzle expects.
inline CUtensorMap fp8_a8_tma_map(const std::uint8_t* address, std::uint64_t k, std::uint64_t rows,
                                  std::uint32_t box_rows, const char* operation) {
    CUtensorMap map{};
    const std::uint64_t global_dim[]     = {k, rows};
    const std::uint64_t global_stride[]  = {k};
    const std::uint32_t box_dim[]        = {64, box_rows};
    const std::uint32_t element_stride[] = {1, 1};
    fp8_a8_tma_check_driver(
        cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
                               const_cast<std::uint8_t*>(address), global_dim, global_stride,
                               box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        operation);
    return map;
}

inline int fp8_a8_multiprocessor_count() {
    // One device per process; read once.
    static const int count = [] {
        int device = 0;
        int value  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    }();
    return count;
}

// Routing: the TMA route runs one CTA per SM on 256-token tiles, the cp.async route two per SM on
// its own tiles. Each cost is the waves needed times the tokens one SM carries through a wave; the
// TMA side is scaled by the measured ratio of its per-token cost at equal work (0.936 on RTX
// 5090) and must win by a margin. Below 1024 tokens a 256-token tile is mostly a partial wave, and
// the sweeps behind these constants never favoured the route there. Upstream's measurements are in
// its PR #167 (docs/maintainer/fp8-a8-tma-route.md there).
inline constexpr double kFp8A8TmaWorkRatio   = 0.936;
inline constexpr double kFp8A8TmaMargin      = 0.02;
inline constexpr std::int32_t kFp8A8TmaMinTokens = 1024;
// From 2048 tokens the route runs whenever it can. Measured end to end on an RTX 5090 (prefill,
// NVIDIA NVFP4/FP8 Qwen3.8-27B, 3584-token chunks), taking it at every call site beat the cost
// model at every width from 2048 to 3584, by 0.5 to 2.5 points of prefill throughput: in the
// model a projection's activations arrive L2-resident, which cold-cache operator timings (the
// basis of the ratio above) do not see, and which favours the TMA route most on the 5120-row
// projections. Below 2048 a partial wave decides, and forcing the route lost up to 7 % at 1153
// tokens, so the cost model keeps that range.
inline constexpr std::int32_t kFp8A8TmaAlwaysTokens = 2048;

template <class TmaSchedule, class MmaSchedule>
constexpr bool fp8_a8_tma_cheaper(std::int64_t rows, std::int64_t tokens,
                                  std::int64_t multiprocessors) {
    const std::int64_t tma_blocks = rows / TmaSchedule::kBlockRows *
                                    ((tokens + TmaSchedule::kBlockTokens - 1) /
                                     TmaSchedule::kBlockTokens);
    const std::int64_t mma_blocks = rows / MmaSchedule::kBlockRows *
                                    ((tokens + MmaSchedule::kBlockTokens - 1) /
                                     MmaSchedule::kBlockTokens);
    const std::int64_t tma_slots = TmaSchedule::kMinBlocksPerSm * multiprocessors;
    const std::int64_t mma_slots = MmaSchedule::kMinBlocksPerSm * multiprocessors;
    const std::int64_t tma_waves = (tma_blocks + tma_slots - 1) / tma_slots;
    const std::int64_t mma_waves = (mma_blocks + mma_slots - 1) / mma_slots;
    const double tma = static_cast<double>(tma_waves * TmaSchedule::kMinBlocksPerSm *
                                           TmaSchedule::kBlockTokens) *
                       kFp8A8TmaWorkRatio;
    const double mma =
        static_cast<double>(mma_waves * MmaSchedule::kMinBlocksPerSm * MmaSchedule::kBlockTokens);
    return tma < mma * (1.0 - kFp8A8TmaMargin);
}

// Whether the TMA route replaces MmaSchedule, the cp.async schedule this call would otherwise
// launch, for these operands. max_tokens is a call site's measured ceiling (upstream found the
// 5120x6144 linear_add slower past 4096 tokens than the model predicts).
template <class MmaSchedule, class TmaSchedule = Fp8A8TmaPrefill>
bool fp8_a8_tma_applies(const Fp8A8Operands& p,
                        std::int32_t max_tokens = std::numeric_limits<std::int32_t>::max()) {
    const auto aligned16 = [](const void* v) {
        return v != nullptr && reinterpret_cast<std::uintptr_t>(v) % 16 == 0;
    };
    if (p.tokens < kFp8A8TmaMinTokens || p.tokens > max_tokens) return false;
    if (p.rows % TmaSchedule::kBlockRows || p.k % TmaSchedule::kBlockK ||
        p.k / TmaSchedule::kBlockK < TmaSchedule::kStages)
        return false;
    if (!aligned16(p.x) || !aligned16(p.codes) ||
        reinterpret_cast<std::uintptr_t>(p.scales) % 4 != 0)
        return false;
    if ((p.tokens + TmaSchedule::kBlockTokens - 1) / TmaSchedule::kBlockTokens > 65535)
        return false;
    if (p.tokens >= kFp8A8TmaAlwaysTokens) return true;
    return fp8_a8_tma_cheaper<TmaSchedule, MmaSchedule>(p.rows, p.tokens,
                                                        fp8_a8_multiprocessor_count());
}

template <class Schedule = Fp8A8TmaPrefill, class Output, class Epilogue>
void launch_fp8_a8_tma(const Fp8A8Operands& p, Output output, Epilogue epilogue,
                       cudaStream_t stream) {
    constexpr int kSharedBytes = static_cast<int>(sizeof(Fp8A8TmaSharedStorage<Schedule>));
    static_assert(kSharedBytes <= 99 * 1024);
    constexpr auto kernel = fp8_a8_tma_kernel<Schedule, Epilogue, Output>;
    static const cudaError_t attribute =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    CUDA_CHECK(attribute);

    Fp8A8TmaDescriptors descriptors{};
    descriptors.a_codes = fp8_a8_tma_map(p.x, p.k, static_cast<std::uint64_t>(p.tokens),
                                         Schedule::kBlockTokens, "encode FP8 activation codes TMA");
    descriptors.b_codes = fp8_a8_tma_map(p.codes, p.k, static_cast<std::uint64_t>(p.rows),
                                         Schedule::kBlockRows, "encode FP8 weight codes TMA");
    const dim3 grid(p.rows / Schedule::kBlockRows,
                    (p.tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens);
#ifdef _WIN32
    const Fp8A8TmaDescriptors* staged = fp8_a8_tma_descriptor_staging().stage(descriptors, stream);
    kernel<<<grid, Schedule::kThreads, kSharedBytes, stream>>>(staged, epilogue, output, p.x_scales,
                                                               p.scales, p.tokens, p.k);
#else
    kernel<<<grid, Schedule::kThreads, kSharedBytes, stream>>>(
        descriptors, epilogue, output, p.x_scales, p.scales, p.tokens, p.k);
#endif
    CUDA_CHECK(cudaGetLastError());
}

// Call sites whose widest cp.async schedule is MmaSchedule: launch the TMA route instead where it
// applies, and report whether it did.
template <class MmaSchedule, class Output, class Epilogue>
bool launch_fp8_a8_tma_if_cheaper(const Fp8A8Operands& p, Output output, Epilogue epilogue,
                                  cudaStream_t stream,
                                  std::int32_t max_tokens = std::numeric_limits<std::int32_t>::max()) {
    if (!fp8_a8_tma_applies<MmaSchedule>(p, max_tokens)) return false;
    launch_fp8_a8_tma(p, output, epilogue, stream);
    return true;
}

} // namespace ninfer::ops::detail
