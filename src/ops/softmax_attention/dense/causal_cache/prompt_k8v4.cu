// ninfer::ops::detail - asymmetric FP8-K/NVFP4-V causal prompt launch ownership.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_k8v4.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_k8v4_fast.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

static_assert(kPromptWaveRows % CausalPromptK8V4FastShape<8>::Br == 0);

template <typename Metadata>
const std::int32_t* causal_prompt_k8v4_valid_columns(const Metadata& metadata) {
    if constexpr (requires { metadata.valid_columns; }) {
        return metadata.valid_columns;
    } else {
        return nullptr;
    }
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_k8v4_original_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  cudaStream_t stream) {
    static const cudaError_t attr = cudaFuncSetAttribute(
        causal_attention_prompt_k8v4_kernel<Geometry, Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptK8V4SmemBytes);
    CUDA_CHECK(attr);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 grid(static_cast<unsigned>(div_up(tokens, kCausalPromptK8V4Br)),
                    static_cast<unsigned>(Geometry::QHeads), 1u);
    causal_attention_prompt_k8v4_kernel<Geometry, Metadata>
        <<<grid, kCausalPromptK8V4Threads, kCausalPromptK8V4SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const std::uint8_t*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_k8v4_fast_launch(const Tensor& q, const Tensor& positions,
                                              float scale, const CacheView& cache,
                                              Metadata metadata, CausalPromptSplitPlan plan,
                                              WorkspaceArena& workspace, Tensor& out,
                                              cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    auto scope        = workspace.scope();
    CausalPromptSplitPartials partials{};
    if (plan.splits > 1) {
        partials =
            allocate_causal_prompt_split_partials(workspace, Geometry::QHeads, tokens, plan.splits);
    }
    const auto launch = [&]<int Warps, bool Split>() {
        using Shape  = CausalPromptK8V4FastShape<Warps>;
        auto* kernel = causal_attention_prompt_k8v4_fast_kernel<Geometry, Metadata, Warps, Split>;
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
            static_cast<const __half*>(cache.k_scale_pages.data),
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
    causal_attention_prompt_k8v4_fast_merge_kernel<Geometry>
        <<<grid, kCausalPromptHeadDim, 0, stream>>>(static_cast<const float*>(partials.rows.data),
                                                    static_cast<const float2*>(partials.stats.data),
                                                    causal_prompt_k8v4_valid_columns(metadata),
                                                    tokens, plan.splits, scale * Log2E,
                                                    static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_k8v4_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                       float scale, const CacheView& cache,
                                                       Metadata metadata,
                                                       CausalPromptSplitPlan plan,
                                                       WorkspaceArena& workspace, Tensor& out,
                                                       cudaStream_t stream) {
    if (plan.original) {
        causal_attention_prompt_k8v4_original_launch<Geometry>(q, positions, scale, cache,
                                                               metadata, out, stream);
        return;
    }
    causal_attention_prompt_k8v4_fast_launch<Geometry>(q, positions, scale, cache, metadata, plan,
                                                       workspace, out, stream);
}

template <typename CacheView, typename Metadata>
void causal_attention_prompt_k8v4_attention_dispatch(const Tensor& q, const Tensor& positions,
                                                     float scale, const CacheView& cache,
                                                     Metadata metadata,
                                                     CausalAttentionExecutionEnvelope envelope,
                                                     WorkspaceArena& workspace, Tensor& out,
                                                     cudaStream_t stream) {
    const CausalPromptSplitPlan plan = causal_attention_prompt_split_plan(
        static_cast<std::int32_t>(q.ne[1]), static_cast<std::int32_t>(q.ne[2]),
        KvCacheStorage::Fp8KeyNvfp4Value, envelope);
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_k8v4_attention_launch_for<CausalD256H24Kv4>(
            q, positions, scale, cache, metadata, plan, workspace, out, stream);
        return;
    }
    causal_attention_prompt_k8v4_attention_launch_for<CausalD256H16Kv2>(
        q, positions, scale, cache, metadata, plan, workspace, out, stream);
}

} // namespace

void causal_attention_prompt_k8v4_attention_launch(const Tensor& q, const Tensor& positions,
                                                   float scale, const PagedKVLayerView& cache,
                                                   CausalAttentionExecutionEnvelope envelope,
                                                   WorkspaceArena& workspace, Tensor& out,
                                                   cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    causal_attention_prompt_k8v4_attention_dispatch(q, positions, scale, cache, metadata, envelope,
                                                    workspace, out, stream);
}

void causal_attention_prompt_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& valid_columns,
                                         const Tensor& table_rows, float scale,
                                         PagedKVBatchLayerView cache,
                                         CausalAttentionExecutionEnvelope envelope,
                                         WorkspaceArena& workspace, Tensor& out,
                                         cudaStream_t stream) {
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        causal_attention_prompt_k8v4_attention_dispatch(q, positions, scale, cache, metadata,
                                                        envelope, workspace, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
