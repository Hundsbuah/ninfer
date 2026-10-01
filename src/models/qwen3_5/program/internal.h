#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;
// Abort salvage publishes the live state only when it covers enough committed work that the
// saved rebuild outweighs the checkpoint's retention cost.
inline constexpr std::uint32_t kSalvageMinFrontier       = 1024;

// Prompt-attention kernel family of a prefill pass. Fast selects the storage's fast prompt kernel
// (INT8 or NVFP4 KV); FastPv8 additionally runs the INT8 fast kernel's PV on 8-bit integer Tensor
// Cores (CausalAttentionExecutionEnvelope::fast_prompt_pv8).
enum class PromptAttentionKernel : std::uint8_t { Original, Fast, FastPv8 };

// Rows of 17..64 verification columns keep 16-bit activations in FP8 residual projections at any
// batch size; first/last bound the per-row width, not the round's aggregate columns.
[[nodiscard]] inline bool wide_residual_verification(TextPhase phase, std::int32_t first,
                                                     std::int32_t last) {
    return phase == TextPhase::Verify && first > 16 && first <= last && last <= 64;
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {
using ContractAccess = RuntimeContractAccess;

[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return backend == SpeculativeBackend::DFlash ? main_frontier : 0U;
}

} // namespace ninfer::models::qwen3_5::detail
