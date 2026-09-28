#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

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
    std::uint32_t duplicate_parameters_repaired = 0;
};

// Policy inputs for name handling. The structure grammar itself stays policy-free; this is
// where declared-tool identity and the tolerant mode enter.
struct ToolCallParsePolicy {
    std::size_t max_name_length = 64;
    bool tolerant               = false;
    // Evaluate `text` as a generated prefix instead of a complete region: a structure cut
    // off at the slice end (an in-progress opener, wrapper, or close literal) reports
    // EndOfInput rather than a definitive break. A byte that is not a prefix of any valid
    // continuation is still definitive. The Phase-4 grammar-constraint core uses this mode.
    bool prefix = false;
    bool enforce_declared_names = false;
    bool (*declared_check)(const void* contract, std::string_view name) = nullptr;
    const void* contract = nullptr;
};
// Objective parse outcome of one tool region (P3.1): what was safely recognized, where the
// input ended, which state was complete and which was incomplete. It carries no policy
// decision; strict and tolerant both consume the same progress, and a separate recovery
// policy decides what may be committed.
enum class ToolCallRegionTermination : std::uint8_t {
    Complete,     // clean end: wrapper closed, function_calls closed, or top-level EOF after a
                  // complete call
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
    std::uint32_t duplicate_parameters_repaired = 0;
    // A break the wire format does not forgive (an empty <function_calls> wrapper): no call
    // may be committed, even by tolerant recovery.
    bool unrecoverable_break = false;
};

// P3.5: the explicit, pure recovery decision. Integrity over availability: a call is
// committed only when all of its argument bytes are unambiguously closed.
enum class ToolCallRecoveryDecision : std::uint8_t {
    Reject,                 // the region falls back to text
    CommitCalls,            // commit the complete calls only
    CommitCallsAndOpenCall, // complete calls plus the open call: its argument bytes are closed
};

struct ToolCallRecoveryResult {
    ToolCallRecoveryDecision decision = ToolCallRecoveryDecision::Reject;
    // The reason the caller records (None when clean, TruncatedTail for a tolerant
    // truncation, the precise failure class otherwise).
    ToolCallParseFailure diagnostic = ToolCallParseFailure::None;
};

struct ToolCallRecoveryPolicy {
    bool tolerant = false;
    // FinishReason::OutputLimit / ContextCapacity is available to the decision as a signal,
    // but a budget cut does not make an open parameter value safe.
    FinishReason finish_reason = FinishReason::None;
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
    // The open call is committable exactly when it kept at least one closed parameter and no
    // value is still open: a name-only truncation carries no arguments, and a cut value must
    // never execute (write/edit/bash/delete all take their payload from a string value).
    const bool open_commit = !progress.open_value_open && !progress.open_call.parameters.empty();
    if (progress.calls.empty() && !open_commit) {
        result.diagnostic = progress.termination == ToolCallRegionTermination::EndOfInput
                                ? ToolCallParseFailure::TruncatedTail
                                : progress.failure;
        return result;
    }
    result.decision = open_commit ? ToolCallRecoveryDecision::CommitCallsAndOpenCall
                                  : ToolCallRecoveryDecision::CommitCalls;
    result.diagnostic = ToolCallParseFailure::TruncatedTail;
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
};

// Stateful, incremental parser for the Qwen tool-call wire syntax. One instance parses one
// output stream. feed() publishes ordinary content as soon as it can no longer become part of
// a tool marker — the decision comes from the wire grammar (classify_tool_marker_prefix), not
// from a separate marker list — and buffers the tool region. finish() runs the region state
// machine over the complete region bytes and reports the parsed region or the precise failure.
//
// Region states (byte-driven; the machine is deterministic on the complete bytes, so the
// result is independent of any chunk partition):
//   marker latch: wrapper literal or complete function/invoke opener
//   ExpectFunction -> FunctionHeader (grammar; tolerant recovery for malformed openers and for
//                     a dropped '>' after the name)
//   FunctionBody -> ParameterHeader -> ParameterValue (nested-opener depth scan with the
//                   grammar's next-token lookahead after a quoted closer) -> FunctionBody
//   function close -> ExpectWrapperClose (tool_call wrapper) | Top (bare / function_calls)
//   Top -> next call | wrapper close | end | trailing

// The strict region parse (single source of truth for in-region legality): parses `text` as a
// complete tool region and reports where it ended. Independent of any recovery policy;
// the Phase-4 grammar-constraint core consumes it directly.
[[nodiscard]] ToolCallParseProgress parse_tool_call_region(std::string_view text,
                                                           const ToolCallParsePolicy& policy);

class ToolCallStreamParser {
public:
    explicit ToolCallStreamParser(ToolCallParsePolicy policy);

    // Consume a chunk of assistant bytes. Returns the bytes that can no longer become part of a
    // tool marker and may be published as ordinary content.
    std::string feed(std::string_view chunk);

    // End the stream and report the parse of the tool region seen so far. `finish_reason`
    // signals why the stream ended: the recovery policy records it, but a budget cut never
    // makes an open parameter value safe. Retries move only to a later <tool_call> wrapper:
    // markup nested inside a failed region must not re-read a truncated call.
    [[nodiscard]] ToolCallStreamResult finish(FinishReason finish_reason = FinishReason::None) const;

    // All bytes determined to be ordinary content so far (the published prefix of the accepted
    // region). The parse entry combines this with result.tail.
    [[nodiscard]] std::string content_prefix() const { return content_; }

    [[nodiscard]] bool marker_seen() const { return marker_seen_; }

    void publish(std::string_view bytes, std::string& visible);
    void latch(std::string_view marker);

    ToolCallParsePolicy policy_;
    std::string content_;       // all bytes determined to be ordinary content
    std::string published_;     // the subset of content_ already returned by feed()
    std::string pending_ws_;    // whitespace since the last published byte (held: may precede a marker)
    std::string marker_prefix_; // held bytes that may become a top-level marker
    bool latched_    = false;
    std::string region_;
    bool marker_seen_ = false;
};

} // namespace ninfer::models::qwen3_5::frontend
