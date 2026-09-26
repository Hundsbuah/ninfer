#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_tma.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace, int tokens,
                    cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    const LinearBf16Output output{data, weight.n};
    const LinearResidualAddEpilogue epilogue{{data, weight.n}};
    const auto launch = [&]<class Schedule>() {
        launch_fp8_a8_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
            fp8_a8_operands(weight, workspace, tokens), output, epilogue, stream);
    };
    // N=5120 leaves few CTAs per token tile, so the time is the weight stream per CTA. Narrow
    // four-stage tiles keep more of it in flight: measured on RTX 5090 (cold weights), K=6144 takes
    // 31.1 us from 24 to 64 tokens instead of 35.5-39.5 and K=17408 72.3-74.3 us instead of
    // 84.6-92.4; each output still accumulates over K in the same order.
    using Narrow32 = Fp8A8MmaSchedule<32, 64, 128, 2, 2, 4, 2, Cache::cg, Cache::cg,
                                      Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Narrow64 = Fp8A8MmaSchedule<64, 64, 128, 2, 2, 4, 2, Cache::cg, Cache::cg,
                                      Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    if (tokens <= 96) return launch.template operator()<Narrow32>();
    if (tokens <= 128) return launch.template operator()<Narrow64>();
    // Upstream measured the TMA route behind its cost model past 4096 tokens at K=6144.
    constexpr std::int32_t kTmaMaxTokens =
        Geometry::kInputRows == 6144 ? 4096 : std::numeric_limits<std::int32_t>::max();
    if (launch_fp8_a8_tma_if_cheaper<Fp8A8T64R128K128>(fp8_a8_operands(weight, workspace, tokens),
                                                       output, epilogue, stream, kTmaMaxTokens))
        return;
    launch.template operator()<Fp8A8T64R128K128>();
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
