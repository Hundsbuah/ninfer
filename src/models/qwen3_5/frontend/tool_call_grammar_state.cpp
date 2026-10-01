#include "models/qwen3_5/frontend/tool_call_grammar_state.h"

#include <cassert>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

// The marker-trigger candidate held by the inactive state: the suffix of the observed
// bytes starting at the last '<' (empty while no marker candidate is accumulating). The
// candidate can only still grow into a later wrapper trigger: the parser's retry re-reads
// a failed region at a later <tool_call> wrapper, and the suffix from the last '<' is the
// longest still-open marker candidate.
std::string marker_suffix(const std::string& text) {
    const std::size_t pos = text.rfind('<');
    return pos == std::string::npos ? std::string{} : std::string(text.substr(pos));
}

} // namespace

ToolCallGrammarConstraint::ToolCallGrammarConstraint(std::size_t max_tool_name_length)
    : max_tool_name_length_(max_tool_name_length == 0 ? 64 : max_tool_name_length) {}

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
    return policy;
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
                    state.triggered_ = false;
                    state.marker_prefix_ = marker_suffix(combined);
                    state.buffer_.clear();
                    continue;
                }
                if (target != nullptr) { *target = state; }
                return ToolCallConstraintVerdict::Rejected;
            }
            if (progress.termination == ToolCallRegionTermination::Complete) {
                // The region closed cleanly: the syntax is unconstrained again and a later
                // region may retrigger (the parser accepts consecutive wrappers).
                state.triggered_ = false;
                state.marker_prefix_ = marker_suffix(combined);
                state.buffer_.clear();
                continue;
            }
            // EndOfInput: the candidate ends inside an open structure — a legal prefix.
            break;
        }
        // Inactive: ordinary prose is always legal; only a complete marker trigger
        // constrains. The marker candidate follows the machine's own feed rule
        // (ToolCallStreamParser::feed) via the shared transition (F8): a '<' starts a
        // candidate only when none is held; every further byte is appended and classified
        // by the grammar; a NotMarker classification flushes the failed bytes as prose and
        // a breaking '<' is retained as a fresh candidate start. The first trigger is
        // therefore exactly the parser's latch.
        std::size_t i = offset;
        while (i < decoded_bytes.size()) {
            const char byte = decoded_bytes[i];
            if (state.marker_prefix_.empty()) {
                if (byte == '<') { state.marker_prefix_.push_back(byte); }
                ++i;
                continue;
            }
            state.marker_prefix_.push_back(byte);
            ToolOpenTag marker = {};
            const ToolMarkerStatus status = classify_tool_marker_prefix(state.marker_prefix_, marker);
            if (status == ToolMarkerStatus::Complete) {
                state.triggered_ = true;
                state.buffer_ = state.marker_prefix_ + std::string(decoded_bytes.substr(i + 1));
                state.marker_prefix_.clear();
                offset = decoded_bytes.size();
                break;  // re-enter: the full candidate is now region text
            }
            if (status == ToolMarkerStatus::NotMarker) {
                // F8: a breaking '<' starts a fresh candidate (shared split rule).
                const std::size_t rescan_start = failed_marker_candidate_rescan_start(state.marker_prefix_);
                state.marker_prefix_ = rescan_start == std::string_view::npos
                                          ? std::string{}
                                          : std::string(state.marker_prefix_.substr(rescan_start));
            }
            ++i;
        }
        if (!state.triggered_) { break; }
    }
    if (target != nullptr) { *target = state; }

    // The candidate ended inside a marker trigger: the grammar cannot decide yet.
    if (!state.triggered_ && !state.marker_prefix_.empty()) {
        ToolOpenTag marker = {};
        if (classify_tool_marker_prefix(state.marker_prefix_, marker) == ToolMarkerStatus::NeedMore) {
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
