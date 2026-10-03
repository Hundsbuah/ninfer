#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "models/qwen3_5/frontend/tool_call_entry_scan.h"
#include "models/qwen3_5/frontend/tool_call_grammar.h"
#include "ninfer/types.h"


namespace ninfer::models::qwen3_5::frontend {

// Status of an incremental tool-region parse. One-shot parsing feeds the whole text and
// finishes once; streaming feeds chunks of any size. The result is independent of the chunk
// partition: every byte sequence yields the same status, region and content as the unchunked
// text.
enum class ToolCallStreamStatus : std::uint8_t {
    Complete,  // the region parsed fully (or a tolerant truncation retained complete-enough calls)
    Invalid,   // the region is definitely broken
    NeedMore,  // the stream ended mid-region; more bytes could still complete a valid parse
};

// Failure classification for a region that is not accepted as complete. The parse entry maps
// these 1:1 onto ToolCallParseFallbackReason.
enum class ToolCallParseFailure : std::uint8_t {
    None,
    MalformedStructure,
    InvalidToolName,
    UndeclaredTool,
    TrailingContent,
    TruncatedTail,
    // R3-08: a committed call of a declared tool contains a synthetic argument, or two or more
    // Stage-2 bases complete the region with unbalanced boundaries (multi-base ambiguity).
    AmbiguousStructure,
};

// Parser-owned AST. The stream parser never builds response objects; the parse entry applies
// the tool contract's normalization to this structure.
struct ParsedParameter {
    std::string name;
    std::string value;
};

struct ParsedFunctionCall {
    std::string name;
    std::vector<ParsedParameter> parameters;
};

struct ParsedToolRegion {
    std::vector<ParsedFunctionCall> calls;
    // R3-08: every parameter occurrence is retained (a repeated name is ambiguous for a
    // declared tool); the legacy last-wins merge and its count move to the parse entry.
};

// The region's outer wrapper (F5): an explicit kind instead of the old ambiguous boolean
// pair. A wrapper open is legal only from None; the matching close returns to None. A second
// wrapper open before the first closed is a definitive structural break.
enum class ToolWrapperKind : std::uint8_t {
    None,
    ToolCall,
    FunctionCalls,
};
// Policy inputs for name handling. The structure grammar itself stays policy-free; this is
// where declared-tool identity and the tolerant mode enter.
struct ToolCallParsePolicy {
    std::size_t max_name_length = 64;
    bool tolerant               = false;
    bool enforce_declared_names = false;
    bool (*declared_check)(const void* contract, std::string_view name) = nullptr;
    const void* contract = nullptr;
    // R3-01: Stage-2 plausibility predicate. Must return false exactly when the named tool
    // has an unambiguous declared schema with at least one property and the named parameter
    // is not among them; true otherwise (unknown tool, conflicting contract, empty schema)
    // so the rule does not apply. Used by the consistent completer to skip a value boundary
    // whose continuation opens an undeclared or repeated parameter of the current call.
    bool (*parameter_plausible)(const void* contract, std::string_view tool_name,
                                std::string_view parameter_name) = nullptr;
    // Round 4: the global Stage-2 work budget of one finish() call (charged per glue
    // transition, candidate, and candidate-walk '<' visit across all bases). Zero selects the
    // default max(100000, 4 x region bytes); a positive value overrides it (tests pin the
    // fail-closed behavior with a tiny budget).
    std::uint64_t stage2_step_budget = 0;
    // R5-07: the top-level entry syntax (ToolCallSyntaxMode). Native latches only the wrapped
    // "<tool_call>" entry; compatibility keeps the legacy bare/function_calls top-level forms.
    // The product boundary (OutputOptions) defaults to QwenWrappedNative for the Qwen3.8
    // production path; this internal default keeps the parser entry API at the historical
    // compatibility behavior for existing callers.
    ToolCallSyntaxMode syntax = ToolCallSyntaxMode::Compatibility;
    // R5-06: the ambiguous-byte protocol policy (ToolCallAmbiguityPolicy). The product
    // boundary (OutputOptions) defaults to FailClosed; this internal default keeps the
    // historical payload-fidelity behavior for existing callers.
    ToolCallAmbiguityPolicy ambiguity = ToolCallAmbiguityPolicy::PayloadFidelity;
    // R6-05: the tool-call intent policy (ToolCallIntentPolicy). TemplateCompatible keeps
    // the upstream Qwen behavior (a region may latch after any content); RequireToolAtContentStart
    // locks the turn to text once any visible (non-formatting-whitespace) Content byte has
    // been committed before a latch (reasoning-channel bytes never reach this scanner).
    ToolCallIntentPolicy intent = ToolCallIntentPolicy::TemplateCompatible;
};
// Objective parse outcome of one tool region (P3.1): what was safely recognized, where the
// input ended, which state was complete and which was incomplete. It carries no policy
// decision; strict and tolerant both consume the same progress, and a separate recovery
// policy decides what may be committed.
enum class ToolCallRegionTermination : std::uint8_t {
    Complete,     // clean end: the top level reached the input end (after format whitespace)
                  // with no wrapper open — wrapper balance is part of the outcome (R2-I2)
    EndOfInput,   // the input ended mid-region (open value, open function, open wrapper, ...)
    Definitive,   // a structural break, or trailing content after complete calls
};

struct ToolCallParseProgress {
    // Fully complete calls: every parameter value has its closing tag.
    std::vector<ParsedFunctionCall> calls;
    // The current (incomplete) call: its name (once parsed) and its closed parameters only.
    // A cut value never contributes parameter bytes here.
    ParsedFunctionCall open_call;
    bool open_value_open  = false; // the open call's current parameter value is unclosed
    ToolCallRegionTermination termination = ToolCallRegionTermination::Complete;
    ToolCallParseFailure failure = ToolCallParseFailure::None; // only when Definitive
    // A break the wire format does not forgive (an empty <function_calls> wrapper): no call
    // may be committed, even by tolerant recovery.
    bool unrecoverable_break = false;
    // A definitive break: the absolute offset (in the full text passed to finish()) of the
    // first byte that proved the break. Absolute since Round-4 §2.1 (the retry chain consumes
    // the full text, so the offset needs no per-region translation). The recovery retry may
    // re-enter at or after this byte (the bytes before it were consumed as the failed region's
    // structure or payload). Zero for a non-definitive outcome.
    std::size_t break_offset = 0;
    // The wrapper state at the break (or at the input end for EndOfInput). A break that
    // leaves a wrapper open is never eligible for a recovery retry: the failed wrapper owns
    // the remaining ambiguous bytes (R2-I1); the retry search runs only from a proven
    // top-level scope.
    ToolWrapperKind wrapper_at_break = ToolWrapperKind::None;
    // R3-02: the break happened directly after an open wrapper literal, on a byte that does
    // not attempt a function opener (no function header was tried inside the wrapper). Such
    // prose owns no payload scope: the retry may re-read the region from the break offset,
    // unlike a wrapper that failed while parsing a function header (R2-I1 still applies).
    bool prose_after_wrapper = false;
    // Stage-2 internal: the machine stopped at the start of a parameter value, waiting for
    // the sequential value chooser to select its boundary (the state in `open_call`/the caller's
    // RegionState carries the value start).
    bool stage2_at_value = false;
};

// P3.5: the explicit, pure recovery decision. Integrity over availability: a call is
// committed only when its function close has been consumed (it is then already a complete
// call in `calls`): a missing function close never makes the call executable, whatever the
// finish reason (F2/I2).
enum class ToolCallRecoveryDecision : std::uint8_t {
    Reject,      // the region falls back to text
    CommitCalls, // commit the complete calls only
};

struct ToolCallRecoveryResult {
    ToolCallRecoveryDecision decision = ToolCallRecoveryDecision::Reject;
    // The reason the caller records (None when clean, TruncatedTail for a tolerant
    // truncation, the precise failure class otherwise).
    ToolCallParseFailure diagnostic = ToolCallParseFailure::None;
};

struct ToolCallRecoveryPolicy {
    bool tolerant = false;
    // The finish reason is decision-relevant (R3-03): a natural stop (StopToken, or None for
    // one-shot callers whose end is unknown) allows committing a complete call before a
    // definitive tail; a deliberate cut (StopString/OutputLimit/ContextCapacity/Cancelled)
    // forbids it. A budget cut never makes an open parameter value safe: an open value is
    // never committed, whatever the reason (F2/I2, pinned by test).
    FinishReason finish_reason = FinishReason::None;
    // R3-03: the tail after the definitive break (region-relative, from break_offset) already
    // contains a `</parameter>` or `</param>` closer. When the stream was cut, that later
    // closer could have extended a committed parameter value, so the committed calls are not
    // safe to commit. Precomputed by the caller (pure decision, no byte access here).
    bool tail_has_value_closer = false;
};

[[nodiscard]] constexpr ToolCallRecoveryResult
decide_tool_call_recovery(const ToolCallParseProgress& progress,
                          const ToolCallRecoveryPolicy& policy) noexcept {
    ToolCallRecoveryResult result;
    if (progress.termination == ToolCallRegionTermination::Complete) {
        result.decision = ToolCallRecoveryDecision::CommitCalls;
        return result;
    }
    if (!policy.tolerant) {
        result.diagnostic = progress.termination == ToolCallRegionTermination::EndOfInput
                                ? ToolCallParseFailure::MalformedStructure
                                : progress.failure;
        return result;
    }
    if (progress.unrecoverable_break) {
        result.diagnostic = progress.failure;
        return result;
    }
    // Tolerant, EndOfInput (R3-03, unchanged): a function-closed call commits, open value or
    // not (an open value never contributes parameter bytes); a region with no complete call
    // is rejected.
    if (progress.termination == ToolCallRegionTermination::EndOfInput) {
        if (!progress.calls.empty()) {
            result.decision   = ToolCallRecoveryDecision::CommitCalls;
            result.diagnostic = ToolCallParseFailure::TruncatedTail;
        } else {
            result.diagnostic = ToolCallParseFailure::TruncatedTail;
        }
        return result;
    }
    // Tolerant, Definitive (R3-03): a committed call with a parameter value is committed only
    // when the stream ended naturally and the tail after the break carries no value closer —
    // otherwise the later value could have extended the committed one. A call without
    // parameters cannot be extended and commits unchanged.
    bool parameterized = false;
    for (const ParsedFunctionCall& call : progress.calls) {
        if (!call.parameters.empty()) { parameterized = true; break; }
    }
    const bool natural_stop =
        policy.finish_reason == FinishReason::StopToken || policy.finish_reason == FinishReason::None;
    if (parameterized && (!natural_stop || policy.tail_has_value_closer)) {
        result.diagnostic = progress.failure;
        return result;
    }
    if (!progress.calls.empty()) {
        result.decision   = ToolCallRecoveryDecision::CommitCalls;
        result.diagnostic = ToolCallParseFailure::TruncatedTail;
    } else {
        result.diagnostic = progress.failure;
    }
    return result;
}


struct ToolCallStreamResult {
    ToolCallStreamStatus status = ToolCallStreamStatus::Invalid;
    ToolCallParseFailure failure = ToolCallParseFailure::MalformedStructure;
    ParsedToolRegion region;
    bool truncated_tail = false;  // tolerant: the stream cut the tail, retained calls kept
    // Bytes that could not be assigned to the structured region: on Invalid these are the
    // verbatim region text (published as content); on Complete these are the bytes between the
    // first marker and the accepted region (rtrimmed at the parse entry).
    std::string tail;
    bool marker_seen = false;
    // Round 4: deterministic work counters (not wall-clock): Stage-2 work units charged over
    // all bases (glue transitions + candidates + candidate-walk '<' visits, from the global
    // WorkBudget) and pre-latch bytes re-fed by the NotMarker rescan. Tests pin the bounds
    // without timing.
    std::uint64_t stage2_steps = 0;
    std::uint64_t rescan_steps = 0;
    // Round 4 (N-07): the global Stage-2 work budget was exhausted; fail closed (nothing from
    // Stage 2 is accepted; the region falls back to the Stage-1/Stage-3 result).
    bool parse_budget_exhausted = false;
    // R3-06/R10: fence and completion diagnostics for the request/operational log.
    bool markup_tolerant_completion = false;
    std::uint32_t fenced_markers_suppressed = 0;
    // R10-03: complete top-level markers suppressed because their '<' began on an indented
    // literal line (visual column >= 4 outside a fence) — pre-latch entries and recovery
    // retry/rebase candidates (R11-I3) alike, never fenced markers.
    std::uint32_t indented_markers_suppressed = 0;
    bool ended_in_unclosed_fence = false;
};

// Stateful, incremental parser for the Qwen tool-call wire syntax. One instance parses one
// output stream. feed() publishes ordinary content as soon as it can no longer become part of
// a tool marker — the decision comes from the wire grammar (classify_tool_marker_prefix), not
// from a separate marker list — and buffers the tool region. A line-oriented fence tracker
// (R2-I6/CR6) keeps recognized final-content code fences out of the marker candidate machine:
// fence bytes are ordinary content, and a tool marker inside a fence never latches.
// finish() runs the region state machine over the complete region bytes and reports the parsed
// region or the precise failure.
//
// Region states (byte-driven; the machine is deterministic on the complete bytes, so the
// result is independent of any chunk partition):
//   marker latch: wrapper literal or complete function/invoke opener
//   ExpectFunction -> FunctionHeader (grammar; tolerant recovery for malformed openers and for
//                     a dropped '>' after the name)
//   FunctionBody -> ParameterHeader -> ParameterValue (opaque byte range; the matching
//   outer close is a candidate boundary, classified by its continuation) -> FunctionBody
//   function close -> ExpectWrapperClose (tool_call wrapper) | Top (bare / function_calls)
//   Top -> next call | wrapper close | end | trailing

// The strict region parse (single source of truth for in-region legality): parses `text` as a
// complete tool region and reports where it ended. Independent of any recovery policy;
// the Phase-4 grammar-constraint core consumes it directly.
[[nodiscard]] ToolCallParseProgress parse_tool_call_region(std::string_view text,
                                                           const ToolCallParsePolicy& policy);

// R3-06: deterministic fence scan over one pre-latch byte stream and one latched region.
// Counts the complete top-level markers a recognized code fence suppressed (in fence bytes,
// pre-latch and region) and reports whether the pre-latch stream ended inside an unclosed
// fence. The marker count is a shadow scan: the same classify_tool_marker_prefix transition
// over fence bytes, never latching.
struct FenceDiagnostics {
    std::uint32_t suppressed_markers = 0;
    bool ended_in_unclosed_fence = false;
};
[[nodiscard]] FenceDiagnostics compute_fence_diagnostics(std::string_view pre_latch,
                                                         std::string_view region,
                                                         ToolCallSyntaxMode syntax);

class ToolCallStreamParser {
public:
    explicit ToolCallStreamParser(ToolCallParsePolicy policy);

    // Consume a chunk of assistant bytes. Returns the bytes that can no longer become part of a
    // tool marker and may be published as ordinary content.
    std::string feed(std::string_view chunk);

    // End the stream and report the parse of the tool region seen so far. `finish_reason`
    // signals why the stream ended: the recovery policy records it, but a budget cut never
    // makes an open parameter value safe. A rejected region may be re-read at a later
    // top-level marker, but never inside a failed wrapper's still-unclosed scope (R2-I1).
    [[nodiscard]] ToolCallStreamResult finish(FinishReason finish_reason = FinishReason::None) const;

    // All bytes determined to be ordinary content so far (the published prefix of the accepted
    // region). The parse entry combines this with result.tail.
    [[nodiscard]] std::string content_prefix() const { return content_; }

    [[nodiscard]] bool marker_seen() const { return marker_seen_; }

    // True once the feed latched a top-level marker; afterwards every fed byte extends
    // the region. The decoder reads the latched region and its pre-marker held bytes
    // through these accessors (it owns no separate marker scan).
    [[nodiscard]] bool latched() const noexcept { return latched_; }
    [[nodiscard]] std::string_view latched_region() const noexcept { return region_; }
    // The bytes held back after the published content: the whitespace prefix plus the
    // pending marker candidate (both empty after a latch).
    [[nodiscard]] std::string held_tail() const { return entry_.held_tail(); }

    // R10 (R10-I1): the shared pre-trigger entry classifier (tool_call_entry_scan.h) owns
    // the pre-latch fence/indentation/marker/intent state; the parser owns the latched
    // region and the content stream.
    [[nodiscard]] const ToolCallEntryScanner& entry() const noexcept { return entry_; }

    ToolCallParsePolicy policy_;
    ToolCallEntryScanner entry_;
    std::string content_; // all bytes determined to be ordinary content
    bool latched_   = false;
    std::string region_;
    bool marker_seen_ = false;
};

} // namespace ninfer::models::qwen3_5::frontend
