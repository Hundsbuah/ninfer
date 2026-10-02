#include "models/qwen3_5/frontend/tool_call_grammar_state.h"

#include <cassert>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

// R9-01 (Round 9 §3.4): the exact between-calls whitespace set. Deliberately narrower
// than is_tool_format_whitespace (which also admits form feed and vertical tab as
// harmless pre-trigger formatting): once a strict tool sequence has started, form feed
// and vertical tab are visible content (G3), and only space, tab, CR, and LF are legal.
constexpr bool is_between_calls_whitespace(char byte) noexcept {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

// R9-01 (Round 9 §3.10): the offset just after the last complete top-level region in
// `buffer`, or npos when no complete top-level region ends inside the buffer (the open
// structure then belongs to the still-open first region). A top-level region ends with
// one of the family closer literals; the prefix up to a closer end must parse Complete,
// which excludes closer literals embedded in parameter values (a markup-bearing value
// keeps its own structure open, R3-01). Cost: one prefix parse per closer occurrence —
// bounded like the region re-parse itself; the incremental-state redesign is deferred
// (R3-11 item 4).
std::size_t last_complete_top_level_region_end(const std::string& buffer,
                                               const ToolCallParsePolicy& policy) {
    const std::string_view compatibility_closers[] = {
        tool_close_literal(ToolTagKind::ToolCall),
        tool_close_literal(ToolTagKind::FunctionCalls),
        tool_close_literal(ToolTagKind::Function),
        tool_close_literal(ToolTagKind::Invoke)};
    const std::string_view* closers = compatibility_closers;
    std::size_t count = 4;
    if (policy.syntax == ToolCallSyntaxMode::QwenWrappedNative) {
        static const std::string_view native_closers[] = {
            tool_close_literal(ToolTagKind::ToolCall)};
        closers = native_closers;
        count = 1;
    }
    std::size_t best = 0;
    for (std::size_t c = 0; c < count; ++c) {
        // std::string overloads only: the (const char*, size_t) rfind overload returns
        // npos for a present literal on MSVC 14.51 (probe-verified).
        const std::string closer_str(closers[c]);
        std::size_t pos = buffer.rfind(closer_str);
        while (pos != std::string::npos) {
            const std::size_t end = pos + closers[c].size();
            if (end > best &&
                parse_tool_call_region(buffer.substr(0, end), policy).termination ==
                    ToolCallRegionTermination::Complete) {
                best = end;
            }
            pos = pos == 0 ? std::string::npos : buffer.rfind(closer_str, pos - 1);
        }
    }
    return best == 0 ? std::string::npos : best;
}

}  // namespace

ToolCallGrammarConstraint::ToolCallGrammarConstraint(std::size_t max_tool_name_length,
                                                     ToolCallSyntaxMode syntax,
                                                     ToolCallIntentPolicy intent)
    : max_tool_name_length_(max_tool_name_length == 0 ? 64 : max_tool_name_length),
      syntax_(syntax), intent_(intent), entry_(syntax, intent) {}

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
    policy.syntax = syntax_;
    return policy;
}

// R10 (R10-I1, Round 10 §8.1): the pre-trigger scan delegates to the shared entry
// classifier — the same machine the production pre-latch parser consumes. Fence
// suppression, indented-literal classification, the marker candidate, the failed-candidate
// rescan, and the intent lock all run there, so the constraint's entry eligibility can
// never drift from the parser's. The constraint still owns the higher-level phases:
// PreTrigger -> InRegion on trigger; PreTrigger -> TextLocked when the gate locks
// (R8-01); BetweenCalls stays a separate strict scan (R9-I1, unchanged).
InactiveScanResult
ToolCallGrammarConstraint::advance_pretrigger(ToolCallGrammarConstraint& state,
                                              std::string_view text) {
    const ToolCallEntryScanner::FeedResult r = state.entry_.feed(text);
    if (r.triggered) {
        state.phase_  = ToolConstraintPhase::InRegion;
        // Seed the region buffer with the same prefix the production parser seeds its
        // region with (held formatting whitespace plus the accepted marker) plus the
        // remaining bytes of this candidate; the region re-parse skips leading format
        // whitespace, so the verdicts agree byte for byte.
        state.buffer_ = r.region_prefix + std::string(text.substr(r.consumed));
        return InactiveScanResult::Triggered;
    }
    // Ordinary content: under RequireToolAtContentStart the visible bytes lock the gate
    // (the scanner's intent lock, R8-01); TemplateCompatible stays lazy. The scanner
    // keeps tracking fence/indent state after the lock (the diagnostics stay live), but
    // a locked gate never triggers again.
    state.phase_ = state.entry_.locked() ? ToolConstraintPhase::TextLocked
                                         : ToolConstraintPhase::PreTrigger;
    return InactiveScanResult::Continued;
}

// R9-01 (Round 9 §3.5): the between-calls strict scan. Only ' ' '\t' '\r' '\n', the next
// legal tool marker (shared classifier, §3.6), and EOS are legal. Any visible byte and any
// failed marker candidate return Rejected: the strict parser would demote the completed
// region as TrailingContent, and a constrained sampler must not admit that continuation
// (R9-I1). No failed-candidate rescan: a stale leading '<' is itself illegal suffix
// content.
InactiveScanResult
ToolCallGrammarConstraint::advance_between_calls(ToolCallGrammarConstraint& state,
                                                 std::string_view text) {
    std::size_t i = 0;
    while (i < text.size()) {
        const char byte = text[i];
        if (state.marker_prefix_.empty()) {
            if (is_between_calls_whitespace(byte)) {
                ++i;
                continue;
            }
            if (byte == '<') {
                state.marker_prefix_.push_back(byte);
                ++i;
                continue;
            }
            // R9-I1: visible suffix content after a completed region is illegal.
            return InactiveScanResult::Rejected;
        }
        state.marker_prefix_.push_back(byte);
        ToolOpenTag marker = {};
        const ToolMarkerStatus status =
            classify_tool_marker_prefix(state.marker_prefix_, marker, state.syntax_);
        if (status == ToolMarkerStatus::Complete) {
            state.phase_ = ToolConstraintPhase::InRegion;
            state.buffer_ = state.marker_prefix_ + std::string(text.substr(i + 1));
            state.marker_prefix_.clear();
            return InactiveScanResult::Triggered;
        }
        if (status == ToolMarkerStatus::NotMarker) {
            // R9-I1: a failed marker is visible suffix content (no F8 rescan, §3.5).
            return InactiveScanResult::Rejected;
        }
        // NeedMore: the candidate continues across the boundary.
        ++i;
    }
    return InactiveScanResult::Continued;
}

ToolCallConstraintVerdict
ToolCallGrammarConstraint::advance(std::string_view decoded_bytes,
                                   ToolCallGrammarConstraint* target) const {
    ToolCallGrammarConstraint state = *this;
    if (decoded_bytes.empty()) {
        if (target != nullptr) { *target = state; }
        return ToolCallConstraintVerdict::Allowed;
    }
    std::size_t offset = 0;
    while (offset < decoded_bytes.size() ||
           state.phase_ == ToolConstraintPhase::InRegion) {
        if (state.phase_ == ToolConstraintPhase::InRegion) {
            const std::string combined =
                state.buffer_ + std::string(decoded_bytes.substr(offset));
            state.buffer_ = combined;
            offset = decoded_bytes.size();
            const ToolCallParseProgress progress =
                parse_tool_call_region(combined, state.parse_policy());
            if (progress.termination == ToolCallRegionTermination::Definitive) {
                if (progress.failure == ToolCallParseFailure::TrailingContent) {
                    // R9-01 (Round 9 §3.8): the region closed earlier in the buffer; the
                    // tail replays through the between-calls strict scan — visible suffix
                    // content is illegal once the strict tool sequence has started.
                    state.phase_ = ToolConstraintPhase::BetweenCalls;
                    state.buffer_.clear();
                    state.marker_prefix_.clear();
                    if (advance_between_calls(state,
                                              combined.substr(progress.break_offset)) ==
                        InactiveScanResult::Rejected) {
                        if (target != nullptr) { *target = state; }
                        return ToolCallConstraintVerdict::Rejected;
                    }
                    continue;
                }
                if (target != nullptr) { *target = state; }
                return ToolCallConstraintVerdict::Rejected;
            }
            if (progress.termination == ToolCallRegionTermination::Complete) {
                // R9-01 (Round 9 §3.7): InRegion -> BetweenCalls: a completed tool never
                // returns to PreTrigger. The closing tag is not retained as a future
                // marker candidate: a stale closer would flush as a failed candidate and
                // reject the gate.
                state.phase_ = ToolConstraintPhase::BetweenCalls;
                state.buffer_.clear();
                state.marker_prefix_.clear();
                continue;
            }
            // EndOfInput: the candidate ends inside an open structure — a legal prefix.
            // R9-01 (Round 9 §3.10): the buffer may already contain completed top-level
            // regions followed by a pending tail (the production parse reports the whole
            // sequence as one open region; the cut-stream behavior of finish() is
            // unchanged, §3.17). Derive the tail after the last complete top-level region
            // and move the state to BetweenCalls: a pending partial marker reports
            // NeedMore (EOS is illegal), a complete marker re-triggers the second region,
            // and a visible byte rejects (§3.8/§3.11).
            const std::size_t tail_start =
                last_complete_top_level_region_end(combined, state.parse_policy());
            if (tail_start != std::string::npos && tail_start < combined.size()) {
                state.phase_ = ToolConstraintPhase::BetweenCalls;
                state.buffer_.clear();
                state.marker_prefix_.clear();
                const InactiveScanResult tail_scan =
                    advance_between_calls(state, combined.substr(tail_start));
                if (tail_scan == InactiveScanResult::Rejected) {
                    if (target != nullptr) { *target = state; }
                    return ToolCallConstraintVerdict::Rejected;
                }
                // Triggered: the seeded buffer re-enters the region parse via the main
                // loop; Continued: the loop exits and the pending-candidate check below
                // reports NeedMore for a partial tail marker.
                continue;
            }
            break;
        }
        // Inactive bytes: the phase selects the scanner (Round 9 §3.5): PreTrigger is
        // lazy (TemplateCompatible prose or the hardened pre-trigger gate), BetweenCalls
        // is strict. A complete trigger re-enters the region parse with the seeded
        // buffer; a rejection stops immediately (the state carries the rejected suffix).
        const InactiveScanResult scan =
            state.phase_ == ToolConstraintPhase::BetweenCalls
                ? advance_between_calls(state, decoded_bytes.substr(offset))
                : advance_pretrigger(state, decoded_bytes.substr(offset));
        offset = decoded_bytes.size();
        if (scan == InactiveScanResult::Rejected) {
            if (target != nullptr) { *target = state; }
            return ToolCallConstraintVerdict::Rejected;
        }
        if (scan != InactiveScanResult::Triggered) { break; }
    }
    if (target != nullptr) { *target = state; }

    // The candidate ended inside a marker trigger: the grammar cannot decide yet. The
    // pending candidate is tracked in PreTrigger (the shared entry scanner, R10-I1) and
    // BetweenCalls (the strict scan); TextLocked tracks none (EOS is legal, §3.13).
    if (state.phase_ != ToolConstraintPhase::InRegion &&
        state.phase_ != ToolConstraintPhase::TextLocked) {
        const std::string_view pending =
            state.phase_ == ToolConstraintPhase::BetweenCalls
                ? std::string_view(state.marker_prefix_)
                : state.entry_.pending_candidate();
        if (!pending.empty()) {
            ToolOpenTag marker = {};
            if (classify_tool_marker_prefix(pending, marker,
                                            state.syntax_) == ToolMarkerStatus::NeedMore) {
                return ToolCallConstraintVerdict::NeedMore;
            }
        }
    }
    return ToolCallConstraintVerdict::Allowed;
}

ToolCallConstraintVerdict ToolCallGrammarConstraint::check(std::string_view decoded_bytes) const {
    return advance(decoded_bytes, nullptr);
}

void ToolCallGrammarConstraint::commit(std::string_view decoded_bytes) {
    const ToolCallConstraintVerdict verdict = advance(decoded_bytes, this);
    assert(verdict != ToolCallConstraintVerdict::Rejected);
}

ToolCallGrammarConstraint ToolCallGrammarConstraint::checkpoint() const { return *this; }

void ToolCallGrammarConstraint::restore(const ToolCallGrammarConstraint& state) { *this = state; }

bool ToolCallGrammarConstraint::active() const noexcept {
    return phase_ == ToolConstraintPhase::InRegion;
}

bool ToolCallGrammarConstraint::finished() const noexcept {
    return phase_ != ToolConstraintPhase::InRegion;
}

// R9-01 (Round 9 §3.13): EOS legality — a stream may end only when no open structure
// needs more bytes: an open region and a pending marker candidate both forbid it.
// R10-I1: the pre-trigger candidate lives in the shared entry scanner.
bool ToolCallGrammarConstraint::can_terminate() const noexcept {
    if (phase_ == ToolConstraintPhase::InRegion) { return false; }
    if (phase_ == ToolConstraintPhase::TextLocked) { return true; }
    if (phase_ == ToolConstraintPhase::BetweenCalls) { return marker_prefix_.empty(); }
    return entry_.pending_candidate().empty();
}

std::size_t ToolCallGrammarConstraint::observed_bytes() const noexcept {
    switch (phase_) {
    case ToolConstraintPhase::InRegion:     return buffer_.size();
    case ToolConstraintPhase::BetweenCalls: return marker_prefix_.size();
    case ToolConstraintPhase::TextLocked:   return 0;
    case ToolConstraintPhase::PreTrigger:   return entry_.pending_candidate().size();
    }
    return 0;
}

} // namespace ninfer::models::qwen3_5::frontend