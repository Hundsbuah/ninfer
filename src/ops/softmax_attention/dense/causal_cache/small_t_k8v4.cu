// ninfer::ops::detail - asymmetric FP8-K/NVFP4-V split-KV small-T launch ownership.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/softmax_attention/dense/causal_cache/small_t_rotated_launch.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void causal_attention_small_t_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache,
                                          CausalAttentionExecutionEnvelope envelope,
                                          std::int32_t column_begin, std::int32_t width,
                                          Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                          Tensor& out, cudaStream_t stream, const void* gate) {
    causal_attention_small_t_rotated_append<CausalSmallTKey::Fp8Row>(
        q, k, v, positions, valid_columns, table_rows, scale, cache, envelope, column_begin, width,
        partial_acc, partial_m, partial_l, out, stream, gate);
}

void causal_attention_small_t_k8v4_chunks_launch(
    const Tensor& q, const Tensor& positions, const Tensor& valid_columns, const Tensor& table_rows,
    float scale, PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
    std::int32_t column_begin, std::int32_t width, std::int32_t chunks, Tensor& partial_acc,
    Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream, const void* gate) {
    causal_attention_small_t_rotated_chunks<CausalSmallTKey::Fp8Row>(
        q, positions, valid_columns, table_rows, scale, cache, envelope, column_begin, width,
        chunks, partial_acc, partial_m, partial_l, out, stream, gate);
}

void causal_attention_cached_small_t_k8v4_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 CausalAttentionExecutionEnvelope envelope,
                                                 std::int32_t column_begin, std::int32_t width,
                                                 std::int32_t chunks, Tensor& partial_acc,
                                                 Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                                 cudaStream_t stream) {
    causal_attention_small_t_rotated_cached<CausalSmallTKey::Fp8Row>(
        q, positions, scale, cache, envelope, column_begin, width, chunks, partial_acc, partial_m,
        partial_l, out, stream);
}

} // namespace ninfer::ops::detail
