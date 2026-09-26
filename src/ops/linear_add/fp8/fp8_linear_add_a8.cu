#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_epilogue.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Both registered problems have N=5120, so the default 64x128 tile gives 40 CTAs and streams at
// about a third of DRAM bandwidth; measured on RTX 5090 it stays flat near 51 us (K=6144) and
// 137 us (K=17408) up to 128 tokens. Narrow four-stage tiles keep enough weight bytes in flight:
// 32x64 takes 25 and 64 us through 96 tokens, and 64x64 through 128. Beyond 128 tokens the
// default's token tiles fill the device and it is fastest again. Every schedule accumulates each
// output over K in the same order.
using Fp8LinearAddNarrow32 =
    Fp8MmaSchedule<32, 64, 128, 2, 2, 4, 2, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;
using Fp8LinearAddNarrow64 =
    Fp8MmaSchedule<64, 64, 128, 2, 2, 4, 2, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;

template <class Geometry, class Schedule, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    auto* output            = static_cast<__nv_bfloat16*>(residual.data);
    const Fp8AddResidualEpilogue epilogue{output, Geometry::kOutputRows};
    const Fp8ContiguousOutput destination{output, Geometry::kOutputRows};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8AddResidualEpilogue,
                           Fp8ContiguousOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, epilogue, destination);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch_schedule(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                     std::int32_t tokens, cudaStream_t stream) {
    if ((tokens % Schedule::kBlockTokens) == 0) {
        launch_mma<Geometry, Schedule, true>(weight, residual, workspace, tokens, stream);
    } else {
        launch_mma<Geometry, Schedule, false>(weight, residual, workspace, tokens, stream);
    }
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 96) {
        launch_schedule<Geometry, Fp8LinearAddNarrow32>(weight, residual, workspace, tokens,
                                                        stream);
    } else if (tokens <= 128) {
        launch_schedule<Geometry, Fp8LinearAddNarrow64>(weight, residual, workspace, tokens,
                                                        stream);
    } else {
        launch_schedule<Geometry, Fp8A8DefaultSchedule>(weight, residual, workspace, tokens,
                                                        stream);
    }
}

} // namespace

void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N5120K6144:
        launch_problem<Fp8N5120K6144>(weight, residual, scratch, x.ne[1], stream);
        return;
    case Fp8GeometryId::N5120K17408:
        launch_problem<Fp8N5120K17408>(weight, residual, scratch, x.ne[1], stream);
        return;
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N34816K5120:
    case Fp8GeometryId::N248320K5120:
        break;
    }
    throw std::invalid_argument("fp8 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
