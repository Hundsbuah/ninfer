#pragma once

// ninfer::ops::detail - private launch prototypes for causal_softmax_attention policies.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class CausalAttentionRoute { SmallT, ChunkedSmallT, Prompt };

// The widest prompt-route row block (the eight-warp fast INT8 kernel); every prompt kernel's row
// block divides it.
inline constexpr std::int32_t kPromptWaveRows = 128;

struct CausalSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
    // Consecutive width-column chunks computed by one co-scheduled partial launch (K8V4 only).
    // Chunk c covers columns [column_begin + c * width, +width).
    std::int32_t chunks = 1;
};

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size = 1);

CausalAttentionRoute causal_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                                    std::int32_t batch_size, KvCacheStorage storage,
                                                    CausalAttentionExecutionEnvelope envelope);

const char* causal_attention_route_name(CausalAttentionRoute route);

// A non-null gate is applied by the reduce epilogue at the store. The BF16, INT8 and K8V4
// reducers accept one; the caller keeps the standalone multiply for FP8 and NVFP4.
void causal_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     const Tensor& table_rows, float scale,
                                     PagedKVBatchLayerView cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     std::int32_t column_begin, std::int32_t width,
                                     Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                     Tensor& out, cudaStream_t stream, const void* gate = nullptr);

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_small_t_fp8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_fp8_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out,
                                                cudaStream_t stream);

void causal_attention_small_t_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_nvfp4_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  CausalAttentionExecutionEnvelope envelope,
                                                  Tensor& partial_acc, Tensor& partial_m,
                                                  Tensor& partial_l, Tensor& out,
                                                  cudaStream_t stream);

// Appends the chunk's K/V rows inside the partial kernel. A non-null gate is applied by the reduce
// epilogue at the store.
void causal_attention_small_t_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache,
                                          CausalAttentionExecutionEnvelope envelope,
                                          std::int32_t column_begin, std::int32_t width,
                                          Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                          Tensor& out, cudaStream_t stream, const void* gate);

// Attends `chunks` consecutive width-column chunks over an already appended cache in one partial
// launch. The partial tensors hold the chunks back to back, each laid out as for one chunk.
void causal_attention_small_t_k8v4_chunks_launch(
    const Tensor& q, const Tensor& positions, const Tensor& valid_columns, const Tensor& table_rows,
    float scale, PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
    std::int32_t column_begin, std::int32_t width, std::int32_t chunks, Tensor& partial_acc,
    Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream, const void* gate);

void causal_attention_cached_small_t_k8v4_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 CausalAttentionExecutionEnvelope envelope,
                                                 std::int32_t column_begin, std::int32_t width,
                                                 std::int32_t chunks, Tensor& partial_acc,
                                                 Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                                 cudaStream_t stream);

// A K8V4 prompt-route launch: the original kernel, or the fast kernel's warps per CTA and number
// of key splits. More than one split divides every row block's key pages among CTAs and merges
// their FP32 partial rows, so a launch whose row blocks alone would leave SMs idle still fills
// them.
struct CausalPromptSplitPlan {
    std::int32_t warps  = 8;
    std::int32_t splits = 1;
    bool original       = false;
};

// The plan of a single-row prompt launch of `width` columns within `envelope`: K8V4 takes the fast
// kernel when the envelope selects it and the original kernel otherwise. Other storages always
// plan one split.
[[nodiscard]] CausalPromptSplitPlan
causal_attention_prompt_split_plan(std::int32_t q_heads, std::int32_t width, KvCacheStorage storage,
                                   CausalAttentionExecutionEnvelope envelope);

// A split launch's partials: the normalized FP32 row of every (column, query head, split) and its
// (max, sum) statistics.
struct CausalPromptSplitPartials {
    Tensor rows;
    Tensor stats;
};

template <class Allocator>
CausalPromptSplitPartials
allocate_causal_prompt_split_partials(Allocator& workspace, std::int32_t q_heads,
                                      std::int32_t width, std::int32_t splits) {
    return {workspace.alloc(DType::FP32, {256, q_heads, width, splits}),
            workspace.alloc(DType::FP32, {2, q_heads, width, splits})};
}

// Transient bytes of that launch's split partials; zero for one split.
[[nodiscard]] std::size_t
causal_attention_prompt_split_workspace_bytes(std::int32_t q_heads, std::int32_t width,
                                              KvCacheStorage storage,
                                              CausalAttentionExecutionEnvelope envelope);

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache,
                                    CausalAttentionExecutionEnvelope envelope,
                                    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache,
                                              CausalAttentionExecutionEnvelope envelope,
                                              WorkspaceArena& workspace, Tensor& out,
                                              cudaStream_t stream);

void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream);

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream);

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream);

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream);

void causal_attention_prompt_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& valid_columns,
                                         const Tensor& table_rows, float scale,
                                         PagedKVBatchLayerView cache,
                                         CausalAttentionExecutionEnvelope envelope,
                                         WorkspaceArena& workspace, Tensor& out,
                                         cudaStream_t stream);

void causal_attention_prompt_k8v4_attention_launch(const Tensor& q, const Tensor& positions,
                                                   float scale, const PagedKVLayerView& cache,
                                                   CausalAttentionExecutionEnvelope envelope,
                                                   WorkspaceArena& workspace, Tensor& out,
                                                   cudaStream_t stream);

} // namespace ninfer::ops::detail
