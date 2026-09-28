#pragma once

// ninfer::ops::detail - split-KV small-T launch of the rotated-value (K8V4, NVFP4) kernel: the
// partial kernel's launch profile, its chunk-major partial layout and the gated split reduce.

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_rotated.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Floats of partial_acc owned by one chunk; partial_m and partial_l hold 1/D of that per chunk.
template <typename Geometry>
std::int64_t rotated_chunk_stride(const CausalSmallTInvocation& invocation, std::int32_t splits) {
    return static_cast<std::int64_t>(kCausalHeadDim) * Geometry::QHeads * invocation.width *
           splits * invocation.batch_size;
}

template <CausalSmallTKey Key, typename Geometry, int TokenTile, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_rotated_partial(const Tensor& q, CacheInput input, const Tensor& positions, float scale,
                            PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    using KScale           = typename CausalSmallTKeyFormat<Key>::Scale;
    constexpr int RowCount = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles = (RowCount + 15) / 16;
    const dim3 grid(Geometry::KVHeads * invocation.chunks, splits, invocation.batch_size);
    const auto launch = [&]<int Warps, int MinBlocks, int KeyBlock>() {
        constexpr std::size_t DynamicBytes = causal_small_t_rotated_arena_bytes<Key>(KeyBlock);
        auto kernel = causal_attention_small_t_rotated_tiled_kernel<Key, Geometry, TokenTile, Warps,
                                                                    MinBlocks, KeyBlock, true,
                                                                    MultiBatch, Masked, CacheInput>;
        static const cudaError_t attr = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(DynamicBytes));
        CUDA_CHECK(attr);

        const auto valid_ptr =
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data);
        const auto rows_ptr = invocation.table_rows == nullptr
                                  ? nullptr
                                  : static_cast<const std::int32_t*>(invocation.table_rows->data);
        kernel<<<grid, Warps * 32, DynamicBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(positions.data),
            static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<KScale*>(cache.k_scale_pages.data),
            static_cast<std::uint8_t*>(cache.v_scale_pages.data),
            static_cast<const std::int32_t*>(cache.block_tables.data), valid_ptr, rows_ptr,
            cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
            logical_capacity, scale, static_cast<float*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
            rotated_chunk_stride<Geometry>(invocation, splits));
        CUDA_CHECK(cudaGetLastError());
    };

    // 32-key tiles let two CTAs share an SM, so one CTA's load hides behind the other's math;
    // 64-key tiles fill the SM alone. Measured on RTX 5090 (graph replay over a 240K-key envelope,
    // 8K-128K keys, B=1-8): two 32-key CTAs are 4-33 % faster for one or two row tiles. For three
    // row tiles six warps are 2-13 % faster, except for the 27B geometry at B=2-3, where the
    // resulting grid leaves a partial second wave at long contexts (up to 15 % slower) and the
    // single 64-key CTA stays.
    if constexpr (RowTiles <= 2) {
        launch.template operator()<8, 2, 32>();
    } else if (Geometry::QHeads == 24 && invocation.chunks == 1 &&
               (invocation.batch_size == 2 || invocation.batch_size == 3)) {
        launch.template operator()<12, 1, 64>();
    } else {
        launch.template operator()<6, 2, 32>();
    }
}

template <CausalSmallTKey Key, typename Geometry, bool MultiBatch, bool Masked>
void launch_rotated_reduce(const Tensor& positions, const CausalSmallTInvocation& invocation,
                           std::int32_t splits, const Tensor& partial_acc, const Tensor& partial_m,
                           const Tensor& partial_l, Tensor& out, cudaStream_t stream,
                           const void* gate) {
    constexpr int Block      = 256;
    const std::int64_t acc   = rotated_chunk_stride<Geometry>(invocation, splits);
    const std::int64_t stats = acc / kCausalHeadDim;
    const dim3 grid(Geometry::QHeads, invocation.width * invocation.batch_size);
    for (std::int32_t chunk = 0; chunk < invocation.chunks; ++chunk) {
        const std::int32_t column_begin = invocation.column_begin + chunk * invocation.width;
        const auto launch               = [&]<bool Offset>() {
            causal_attention_small_t_rotated_reduce_output_kernel<Key, Geometry, MultiBatch, Masked,
                                                                  Offset>
                <<<grid, Block, 0, stream>>>(
                    static_cast<const float*>(partial_acc.data) + chunk * acc,
                    static_cast<const float*>(partial_m.data) + chunk * stats,
                    static_cast<const float*>(partial_l.data) + chunk * stats,
                    static_cast<const std::int32_t*>(positions.data),
                    invocation.valid_columns == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                    invocation.width, invocation.full_width, column_begin, invocation.batch_size,
                    splits, static_cast<__nv_bfloat16*>(out.data),
                    static_cast<const __nv_bfloat16*>(gate));
        };
        if (column_begin == 0) {
            launch.template operator()<false>();
        } else {
            launch.template operator()<true>();
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

template <CausalSmallTKey Key, typename Geometry, typename CacheInput>
void causal_attention_small_t_rotated_launch_for(
    const Tensor& q, CacheInput input, const Tensor& positions, float scale,
    PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
    CausalAttentionExecutionEnvelope envelope, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, Tensor& out, cudaStream_t stream, const void* gate) {
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits           = causal_attention_split_capacity(
        Geometry::QHeads, invocation.width, cache.storage, envelope, invocation.batch_size);

    const auto launch_partial = [&]<int Tokens, bool MultiBatch, bool Masked>() {
        launch_rotated_partial<Key, Geometry, Tokens, MultiBatch, Masked>(
            q, input, positions, scale, cache, invocation, logical_capacity, splits, partial_acc,
            partial_m, partial_l, stream);
    };
    const bool masked            = invocation.valid_columns != nullptr;
    const auto dispatch_metadata = [&]<int Tokens>() {
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_partial.template operator()<Tokens, false, true>();
            } else {
                launch_partial.template operator()<Tokens, false, false>();
            }
        } else if (masked) {
            launch_partial.template operator()<Tokens, true, true>();
        } else {
            launch_partial.template operator()<Tokens, true, false>();
        }
    };

    switch (invocation.width) {
    case 1:
        dispatch_metadata.template operator()<1>();
        break;
    case 2:
        dispatch_metadata.template operator()<2>();
        break;
    case 3:
        dispatch_metadata.template operator()<3>();
        break;
    case 4:
        dispatch_metadata.template operator()<4>();
        break;
    case 5:
        dispatch_metadata.template operator()<5>();
        break;
    case 6:
        dispatch_metadata.template operator()<6>();
        break;
    case 7:
        if constexpr (Geometry::QHeads == 24) {
            dispatch_metadata.template operator()<7>();
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    case 8:
        if constexpr (Geometry::QHeads == 24) {
            dispatch_metadata.template operator()<8>();
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    default:
        throw std::invalid_argument("causal_attention_small_t_rotated_launch: unsupported T");
    }

    if (invocation.batch_size == 1) {
        if (masked) {
            launch_rotated_reduce<Key, Geometry, false, true>(positions, invocation, splits,
                                                              partial_acc, partial_m, partial_l,
                                                              out, stream, gate);
        } else {
            launch_rotated_reduce<Key, Geometry, false, false>(positions, invocation, splits,
                                                               partial_acc, partial_m, partial_l,
                                                               out, stream, gate);
        }
    } else if (masked) {
        launch_rotated_reduce<Key, Geometry, true, true>(positions, invocation, splits, partial_acc,
                                                         partial_m, partial_l, out, stream, gate);
    } else {
        launch_rotated_reduce<Key, Geometry, true, false>(
            positions, invocation, splits, partial_acc, partial_m, partial_l, out, stream, gate);
    }
}

template <CausalSmallTKey Key, typename CacheInput>
void causal_attention_small_t_rotated_dispatch(
    const Tensor& q, CacheInput input, const Tensor& positions, float scale,
    PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
    CausalAttentionExecutionEnvelope envelope, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, Tensor& out, cudaStream_t stream, const void* gate) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_rotated_launch_for<Key, CausalD256H24Kv4>(
            q, input, positions, scale, cache, invocation, envelope, partial_acc, partial_m,
            partial_l, out, stream, gate);
        return;
    }
    causal_attention_small_t_rotated_launch_for<Key, CausalD256H16Kv2>(
        q, input, positions, scale, cache, invocation, envelope, partial_acc, partial_m, partial_l,
        out, stream, gate);
}

// The three entry points of one storage: append-and-attend of `width` columns, co-scheduled
// chunks over an appended cache, and cached-only attention.
template <CausalSmallTKey Key>
void causal_attention_small_t_rotated_append(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream,
    const void* gate) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    causal_attention_small_t_rotated_dispatch<Key>(q, input, positions, scale, cache, invocation,
                                                   envelope, partial_acc, partial_m, partial_l, out,
                                                   stream, gate);
}

template <CausalSmallTKey Key>
void causal_attention_small_t_rotated_chunks(
    const Tensor& q, const Tensor& positions, const Tensor& valid_columns, const Tensor& table_rows,
    float scale, PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
    std::int32_t column_begin, std::int32_t width, std::int32_t chunks, Tensor& partial_acc,
    Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream, const void* gate) {
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
        .chunks        = chunks,
    };
    causal_attention_small_t_rotated_dispatch<Key>(q, CausalCachedInput{}, positions, scale, cache,
                                                   invocation, envelope, partial_acc, partial_m,
                                                   partial_l, out, stream, gate);
}

template <CausalSmallTKey Key>
void causal_attention_small_t_rotated_cached(const Tensor& q, const Tensor& positions, float scale,
                                             const PagedKVLayerView& cache,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t column_begin, std::int32_t width,
                                             std::int32_t chunks, Tensor& partial_acc,
                                             Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                             cudaStream_t stream) {
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = 1,
        .chunks        = chunks,
    };
    causal_attention_small_t_rotated_dispatch<Key>(
        q, CausalCachedInput{}, positions, scale, single_row_paged_kv_batch_view(cache), invocation,
        envelope, partial_acc, partial_m, partial_l, out, stream, nullptr);
}

} // namespace ninfer::ops::detail
