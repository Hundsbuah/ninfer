// ninfer::ops::detail - public-op composition for group-16 NVFP4 causal prompt attention.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_rotated_fast_launch.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

static_assert(kPromptWaveRows %
                  CausalPromptRotatedShape<CausalPromptRotatedKey::Nvfp4Group16, 8>::Br ==
              0);

CausalPromptSplitPlan nvfp4_prompt_plan(const Tensor& q,
                                        CausalAttentionExecutionEnvelope envelope) {
    return causal_attention_prompt_split_plan(static_cast<std::int32_t>(q.ne[1]),
                                              static_cast<std::int32_t>(q.ne[2]),
                                              KvCacheStorage::Nvfp4Group16, envelope);
}

template <typename CacheView, typename Metadata>
void causal_attention_prompt_nvfp4_fast(const Tensor& q, const Tensor& positions, float scale,
                                        const CacheView& cache, Metadata metadata,
                                        CausalPromptSplitPlan plan, WorkspaceArena& workspace,
                                        Tensor& out, cudaStream_t stream) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_rotated_fast_launch<CausalPromptRotatedKey::Nvfp4Group16,
                                                    CausalD256H24Kv4>(
            q, positions, scale, cache, metadata, plan, workspace, out, stream);
        return;
    }
    causal_attention_prompt_rotated_fast_launch<CausalPromptRotatedKey::Nvfp4Group16,
                                                CausalD256H16Kv2>(
        q, positions, scale, cache, metadata, plan, workspace, out, stream);
}

} // namespace

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    CausalAttentionExecutionEnvelope envelope,
                                                    WorkspaceArena& workspace, Tensor& out,
                                                    cudaStream_t stream) {
    const CausalPromptSplitPlan plan = nvfp4_prompt_plan(q, envelope);
    if (plan.original) {
        causal_attention_prompt_nvfp4_kernel_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    causal_attention_prompt_nvfp4_fast(q, positions, scale, cache, metadata, plan, workspace, out,
                                       stream);
}

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache,
                                          CausalAttentionExecutionEnvelope envelope,
                                          WorkspaceArena& workspace, Tensor& out,
                                          cudaStream_t stream) {
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const CausalPromptSplitPlan plan = nvfp4_prompt_plan(q, envelope);
    if (plan.original) {
        causal_attention_prompt_nvfp4_batch_kernel_launch(q, positions, valid_columns, table_rows,
                                                          scale, cache, out, stream);
        return;
    }
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        causal_attention_prompt_nvfp4_fast(q, positions, scale, cache, metadata, plan, workspace,
                                           out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
