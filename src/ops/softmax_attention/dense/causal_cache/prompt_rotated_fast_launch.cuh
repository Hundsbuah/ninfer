#pragma once

// ninfer::ops::detail - launch of the rotated-value (NVFP4, K8V4) causal prompt kernel: the
// split plan, its transient partials, the partial kernel and the split merge.

#include "core/arena.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_rotated_fast.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

template <typename Metadata>
const std::int32_t* causal_prompt_rotated_valid_columns(const Metadata& metadata) {
    if constexpr (requires { metadata.valid_columns; }) {
        return metadata.valid_columns;
    } else {
        return nullptr;
    }
}

template <CausalPromptRotatedKey Key, typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_rotated_fast_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const CacheView& cache,
                                                 Metadata metadata, CausalPromptSplitPlan plan,
                                                 WorkspaceArena& workspace, Tensor& out,
                                                 cudaStream_t stream) {
    using KScale      = typename CausalPromptRotatedStage<Key>::KScale;
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    auto scope        = workspace.scope();
    CausalPromptSplitPartials partials{};
    if (plan.splits > 1) {
        partials =
            allocate_causal_prompt_split_partials(workspace, Geometry::QHeads, tokens, plan.splits);
    }
    const auto launch = [&]<int Warps, bool Split>() {
        using Shape = CausalPromptRotatedShape<Key, Warps>;
        auto* kernel =
            causal_attention_prompt_rotated_fast_kernel<Geometry, Metadata, Key, Warps, Split>;
        static const cudaError_t attr = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Shape::SmemBytes);
        CUDA_CHECK(attr);
        const dim3 grid(static_cast<unsigned>(div_up(tokens, Shape::Br)),
                        static_cast<unsigned>(Geometry::QHeads),
                        static_cast<unsigned>(plan.splits));
        kernel<<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const KScale*>(cache.k_scale_pages.data),
            static_cast<const std::uint8_t*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens, static_cast<float*>(partials.rows.data),
            static_cast<float2*>(partials.stats.data));
        CUDA_CHECK(cudaGetLastError());
    };
    const auto dispatch = [&]<bool Split>() {
        if (plan.warps == 4) {
            launch.template operator()<4, Split>();
        } else {
            launch.template operator()<8, Split>();
        }
    };
    if (plan.splits == 1) {
        dispatch.template operator()<false>();
        return;
    }
    dispatch.template operator()<true>();
    constexpr float Log2E = 1.4426950408889634074f;
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(Geometry::QHeads), 1u);
    causal_attention_prompt_rotated_fast_merge_kernel<Geometry, Key>
        <<<grid, kCausalPromptHeadDim, 0, stream>>>(static_cast<const float*>(partials.rows.data),
                                                    static_cast<const float2*>(partials.stats.data),
                                                    causal_prompt_rotated_valid_columns(metadata),
                                                    tokens, plan.splits, scale * Log2E,
                                                    static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
