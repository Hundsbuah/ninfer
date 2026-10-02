#include "models/qwen3_5/frontend/tool_call_grammar_state.h"

#include <cassert>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {

ToolCallGrammarConstraint::ToolCallGrammarConstraint(std::size_t max_tool_name_length,
                                                     ToolCallSyntaxMode syntax,
                                                     ToolCallIntentPolicy intent)
    : max_tool_name_length_(max_tool_name_length == 0 ? 64 : max_tool_name_length),
      syntax_(syntax), intent_(intent) {}

ToolCallParsePolicy ToolCallGrammarConstraint::parse_policy() const {
    ToolCallParsePolicy policy;
    policy.max_name_length = max_tool_name_length_;
    // The constraint evaluates generated prefixes: a structure cut off at the candidate
    // boundary stays a legal prefix (prefix mode), while a byte that is not a prefix of any
    // valid continuation is definitive. Strict (no tolerant recovery): the model is
    // constrained to the canonical wire syntax, and the declared-name check is a contract
    // concern, not wire syntax.
    policy.tolerant = false;
    policy.enforce_declared_names = false;
    // R6-01: the constraint follows the selected syntax mode (R6-I1): the trigger, the
    // region re-parse, and the production parser must agree on what a top-level entry is.
    // Strict (no tolerant recovery): the model is constrained to the canonical wire syntax,
    // and the declared-name check is a contract concern, not wire syntax.
    policy.syntax = syntax_;
    return policy;
}

// R8-01 (Round 8 §3.8): the single authority for the inactive intent-state transitions,
// shared by ordinary inactive bytes and the post-call trailing replay. It owns the
// formatting whitespace handling, the marker start and continuation, the candidate
// classification, the failed-candidate breaking-'< rescan (F8), the hardened text lock
// (visible content locks the gate at any point, R8-I2/R8-I6), and the complete-marker
// trigger (which seeds the region buffer with the marker plus the remaining bytes).
// Returns true once a complete trigger fired (state.triggered_ set).
bool ToolCallGrammarConstraint::advance_inactive(ToolCallGrammarConstraint& state,
                                                 std::string_view text) {
    std::size_t i = 0;
    while (i < text.size()) {
        const char byte = text[i];
        if (state.marker_prefix_.empty()) {
            if (byte == '<') { state.marker_prefix_.push_back(byte); }
            else if (state.intent_ == ToolCallIntentPolicy::RequireToolAtContentStart &&
                     !is_tool_format_whitespace(byte)) {
                // R8-01: visible content locks the gate at any point (R8-I2: the first
                // latch is not permanent permission).
                state.entry_locked_ = true;
            }
            ++i;
            continue;
        }
        state.marker_prefix_.push_back(byte);
        ToolOpenTag marker = {};
        const ToolMarkerStatus status =
            classify_tool_marker_prefix(state.marker_prefix_, marker, state.syntax_);
        if (status == ToolMarkerStatus::Complete) {
            if (state.intent_ == ToolCallIntentPolicy::RequireToolAtContentStart &&
                state.entry_locked_) {
                // The gate is locked: the complete marker is ordinary content (the parser
                // publishes it as content instead of latching).
                state.marker_prefix_.clear();
                ++i;
                continue;
            }
            state.triggered_ = true;
            state.buffer_ = state.marker_prefix_ + std::string(text.substr(i + 1));
            state.marker_prefix_.clear();
            return true;
        }
        if (status == ToolMarkerStatus::NotMarker) {
            if (state.intent_ == ToolCallIntentPolicy::RequireToolAtContentStart) {
                // R8-01: a failed marker candidate is visible content at any point.
                state.entry_locked_ = true;
            }
            // F8: a breaking '<' starts a fresh candidate (shared split rule).
            const std::size_t rescan_start = failed_marker_candidate_rescan_start(state.marker_prefix_);
            state.marker_prefix_ = rescan_start == std::string_view::npos
                                       ? std::string{}
                                       : std::string(state.marker_prefix_.substr(rescan_start));
        }
        ++i;
    }
    return false;
}

ToolCallConstraintVerdict
ToolCallGrammarConstraint::advance(std::string_view decoded_bytes, ToolCallGrammarConstraint* target) const {
    ToolCallGrammarConstraint state = *this;
    if (decoded_bytes.empty()) {
        if (target != nullptr) { *target = state; }
        return ToolCallConstraintVerdict::Allowed;
    }
    std::size_t offset = 0;
    while (offset < decoded_bytes.size() || state.triggered_) {
        if (state.triggered_) {
            const std::string combined = state.buffer_ + std::string(decoded_bytes.substr(offset));
            state.buffer_ = combined;
            offset = decoded_bytes.size();
            const ToolCallParseProgress progress = parse_tool_call_region(combined, state.parse_policy());
            if (progress.termination == ToolCallRegionTermination::Definitive) {
                // TrailingContent after a complete call is prose, not a break: the region
                // closed earlier in the buffer and the remaining bytes are unconstrained.
                // A structural break inside an open region is a wire violation.
                if (progress.failure == ToolCallParseFailure::TrailingContent) {
                    // R8-01: the region closed earlier in the buffer; the bytes from the
                    // parser's break offset are ordinary trailing content and replay
                    // through the inactive gate: visible content locks the hardened intent,
                    // and a directly consecutive wrapper may retrigger (Round 8 §3.7).
                    state.triggered_ = false;
                    state.buffer_.clear();
                    state.marker_prefix_.clear();
                    (void)advance_inactive(state, combined.substr(progress.break_offset));
                    continue;
                }
                if (target != nullptr) { *target = state; }
                return ToolCallConstraintVerdict::Rejected;
            }
            if (progress.termination == ToolCallRegionTermination::Complete) {
                // R8-01: the region closed cleanly; the hardened gate stays open for a
                // directly consecutive wrapper. The closing tag is not retained as a future
                // marker candidate: a stale closer would flush as a failed candidate and
                // lock the gate (Round 8 §3.6).
                state.triggered_ = false;
                state.buffer_.clear();
                state.marker_prefix_.clear();
                continue;
            }
            // EndOfInput: the candidate ends inside an open structure — a legal prefix.
            break;
        }
        // Inactive bytes: the shared intent-state scanner. Ordinary content and the
        // post-call trailing replay use the same authority (Round 8 §3.8); a complete
        // trigger re-enters the region parse with the seeded buffer.
        const bool marker_complete = advance_inactive(state, decoded_bytes.substr(offset));
        offset = decoded_bytes.size();
        if (!marker_complete) { break; }
    }
    if (target != nullptr) { *target = state; }

    // The candidate ended inside a marker trigger: the grammar cannot decide yet.
    if (!state.triggered_ && !state.marker_prefix_.empty()) {
        ToolOpenTag marker = {};
        if (classify_tool_marker_prefix(state.marker_prefix_, marker,
                                        state.syntax_) == ToolMarkerStatus::NeedMore) {
            return ToolCallConstraintVerdict::NeedMore;
        }
    }
    return ToolCallConstraintVerdict::Allowed;
}

ToolCallConstraintVerdict ToolCallGrammarConstraint::check(std::string_view decoded_bytes) const {
    return advance(decoded_bytes, nullptr);
}

void ToolCallGrammarConstraint::commit(std::string_view decoded_bytes) {
    const ToolCallConstraintVerdict verdict = advance(decoded_bytes, this);
    assert(verdict != ToolCallConstraintVerdict::Rejected &&
           "commit() requires a previously checked candidate; Rejected means the bytes "
           "break the wire syntax and must have been masked out");
}

ToolCallGrammarConstraint ToolCallGrammarConstraint::checkpoint() const { return *this; }

void ToolCallGrammarConstraint::restore(const ToolCallGrammarConstraint& state) { *this = state; }

bool ToolCallGrammarConstraint::active() const noexcept { return triggered_; }

bool ToolCallGrammarConstraint::finished() const noexcept { return !triggered_; }

std::size_t ToolCallGrammarConstraint::observed_bytes() const noexcept {
    return triggered_ ? buffer_.size() : marker_prefix_.size();
}

} // namespace ninfer::models::qwen3_5::frontend
