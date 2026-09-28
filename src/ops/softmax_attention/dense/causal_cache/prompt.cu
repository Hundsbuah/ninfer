// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {
static_assert(kPromptWaveRows == CausalPromptI8FastShape<8>::Br);
static_assert(kPromptWaveRows % CausalPromptI8FastShape<4>::Br == 0);
static_assert(kPromptWaveRows % kCausalPromptI8Br == 0);
static_assert(kPromptWaveRows % kCausalPromptBr == 0);

namespace {

// Both fast INT8 variants run one CTA per SM and every CTA of a launch sweeps a similar key range,
// so a launch costs about (waves) x (one CTA's sweep). A four-warp CTA sweeps in about 0.72 of an
// eight-warp CTA's time (measured on RTX 5090 at 64K context) but covers half the rows.
bool causal_attention_prompt_i8_fast_prefers_narrow(std::int32_t tokens, std::int32_t q_heads) {
    static const int multiprocessors = [] {
        int device = 0;
        int count  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    const auto waves = [&](int rows) {
        return div_up(div_up(tokens, rows) * q_heads, multiprocessors);
    };
    constexpr int NarrowCostPercent = 72;
    return waves(CausalPromptI8FastShape<4>::Br) * NarrowCostPercent <
           waves(CausalPromptI8FastShape<8>::Br) * 100;
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_i8_fast_launch_for(const Tensor& q, const Tensor& positions,
                                                float scale, const CacheView& cache,
                                                Metadata metadata, Tensor& out,
                                                cudaStream_t stream) {
    static const cudaError_t attr_wide = cudaFuncSetAttribute(
        causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, 8>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, CausalPromptI8FastShape<8>::SmemBytes);
    CUDA_CHECK(attr_wide);
    static const cudaError_t attr_narrow = cudaFuncSetAttribute(
        causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, 4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, CausalPromptI8FastShape<4>::SmemBytes);
    CUDA_CHECK(attr_narrow);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const auto launch = [&]<int Warps>() {
        using Shape = CausalPromptI8FastShape<Warps>;
        const dim3 grid(static_cast<unsigned>(div_up(tokens, Shape::Br)),
                        static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, Warps>
            <<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache.k_pages.data),
                static_cast<const std::int8_t*>(cache.v_pages.data),
                static_cast<const __half*>(cache.k_scale_pages.data),
                static_cast<const __half*>(cache.v_scale_pages.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    };
    if (causal_attention_prompt_i8_fast_prefers_narrow(tokens, Geometry::QHeads)) {
        launch.template operator()<4>();
    } else {
        launch.template operator()<8>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out, bool fast,
                                                  cudaStream_t stream) {
    if (fast && cache.storage == KvCacheStorage::Int8Group64) {
        causal_attention_prompt_i8_fast_launch_for<Geometry>(q, positions, scale, cache, metadata,
                                                             out, stream);
        return;
    }
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // Both dtype-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(causal_attention_prompt_bf16_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptSmemBytes);
    CUDA_CHECK(attr_bf16);
    static const cudaError_t attr_i8 =
        cudaFuncSetAttribute(causal_attention_prompt_i8_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
    CUDA_CHECK(attr_i8);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (cache.storage == KvCacheStorage::Int8Group64) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        causal_attention_prompt_i8_kernel<Geometry, Metadata>
            <<<attention_grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache_k.data),
                static_cast<const std::int8_t*>(cache_v.data),
                static_cast<const __half*>(cache_k_scale.data),
                static_cast<const __half*>(cache_v_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_bf16_kernel<Geometry, Metadata>
            <<<attention_grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __half*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Split partials may use at most this much workspace; the planner reserves the largest amount any
// prefill width can take, so the budget bounds the Device memory the split costs.
constexpr std::size_t kPromptSplitBudgetBytes  = std::size_t{64} << 20;
constexpr std::int32_t kPromptMinPagesPerSplit = 8;
constexpr std::int32_t kPromptMaxSplits        = 16;
constexpr std::int32_t kPromptSplitCostPercent = 1;

// Time for one fast-kernel CTA to sweep a launch's keys, as a percentage of the eight-warp CTA
// (RTX 5090, 4096 columns over 128K keys); the four-warp CTA takes 72 %.
struct PromptCtaCost {
    std::int32_t warps;
    std::int32_t rows;
    std::int64_t percent;
};

constexpr PromptCtaCost kPromptK8V4Costs[] = {{8, 128, 100}, {4, 64, 72}};

int prompt_multiprocessor_count() {
    static const int multiprocessors = [] {
        int device = 0;
        int count  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    return multiprocessors;
}

std::size_t prompt_split_bytes(std::int32_t q_heads, std::int32_t width, std::int32_t splits) {
    if (splits <= 1) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate_causal_prompt_split_partials(layout, q_heads, width, splits);
    return layout.peak_bytes(1);
}

} // namespace

CausalPromptSplitPlan causal_attention_prompt_split_plan(std::int32_t q_heads, std::int32_t width,
                                                         KvCacheStorage storage,
                                                         CausalAttentionExecutionEnvelope envelope) {
    if (storage != KvCacheStorage::Fp8KeyNvfp4Value || width <= 0) { return {}; }
    if (!envelope.fast_prompt_kernel) { return {.original = true}; }
    // Every CTA of a launch sweeps about the same key range and one CTA fits an SM, so a launch
    // costs about (waves) x (one CTA's sweep); a split CTA sweeps 1/splits of it.
    const std::int32_t multiprocessors = prompt_multiprocessor_count();
    const auto pages                   = static_cast<std::int32_t>(
        (static_cast<std::uint64_t>(envelope.max_visible_keys) + kPagedKVPageSize - 1) /
        kPagedKVPageSize);
    CausalPromptSplitPlan best{};
    std::int64_t best_cost = std::numeric_limits<std::int64_t>::max();
    for (const PromptCtaCost& cta : kPromptK8V4Costs) {
        const std::int32_t ctas = div_up(width, cta.rows) * q_heads;
        const std::int64_t per  = cta.percent;
        for (std::int32_t splits = 1; splits <= kPromptMaxSplits; ++splits) {
            if (splits > 1 &&
                (pages < splits * kPromptMinPagesPerSplit ||
                 prompt_split_bytes(q_heads, width, splits) > kPromptSplitBudgetBytes)) {
                break;
            }
            const std::int64_t waves = div_up(static_cast<std::int64_t>(ctas) * splits,
                                              static_cast<std::int64_t>(multiprocessors));
            // Scaled by kPromptMaxSplits so every split count divides exactly.
            const std::int64_t cost = waves * per * kPromptMaxSplits / splits +
                                      (splits - 1) * kPromptSplitCostPercent * kPromptMaxSplits;
            if (cost < best_cost) {
                best_cost = cost;
                best      = {cta.warps, splits};
            }
        }
    }
    return best;
}

std::size_t causal_attention_prompt_split_workspace_bytes(std::int32_t q_heads, std::int32_t width,
                                                          KvCacheStorage storage,
                                                          CausalAttentionExecutionEnvelope envelope) {
    const CausalPromptSplitPlan plan =
        causal_attention_prompt_split_plan(q_heads, width, storage, envelope);
    return prompt_split_bytes(q_heads, width, plan.splits);
}

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache,
                                              CausalAttentionExecutionEnvelope envelope,
                                              WorkspaceArena& workspace, Tensor& out,
                                              cudaStream_t stream) {
    const bool fast = envelope.fast_prompt_kernel;
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_attention_launch(q, positions, scale, cache, envelope,
                                                      workspace, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(q, positions, scale, cache,
                                                                       metadata, out, fast, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, out, fast, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache,
                                    CausalAttentionExecutionEnvelope envelope,
                                    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const bool fast = envelope.fast_prompt_kernel;
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                            cache, envelope, workspace, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, out, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, out, fast, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, fast, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
