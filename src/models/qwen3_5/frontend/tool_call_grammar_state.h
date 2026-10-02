#pragma once

#include "models/qwen3_5/frontend/tool_call_entry_scan.h"
#include "models/qwen3_5/frontend/tool_call_stream.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

// Phase-4 CPU core: the wire-syntax constraint state for grammar-constrained tool decoding.
// Pure CPU, deterministic, CUDA-independent. It takes the grammar state so far plus one
// candidate token's decoded bytes and decides whether those bytes are a legal continuation
// of the wire syntax.
//
// The state is byte-driven by the Phase-1 wire grammar, which stays the single source of
// truth: `classify_tool_marker_prefix` decides what a trigger is, and `parse_tool_call_region`
// decides in-region legality. No second hand-written wire grammar exists.
//
// While no complete marker trigger has been seen, the syntax is unconstrained: ordinary
// prose and reasoning are always legal (a candidate that ends inside a marker trigger
// reports NeedMore rather than a decision). A complete trigger activates the region
// machine, which stays constraining until the region closes cleanly; afterwards the
// the suffix is constrained: visible content after a completed region is illegal, only
// formatting whitespace, the next legal tool entry, and EOS remain legal (R9-01).
enum class ToolCallConstraintVerdict : std::uint8_t {
    Allowed,   // the bytes are a legal continuation (the state advances; the region may close)
    Rejected,  // the bytes break the wire syntax where a legal continuation existed before
    NeedMore,  // the bytes end inside a marker trigger; the grammar cannot decide yet (legal
               // so far: treat as Allowed when masking, the state stays pending)
};

// R9-01 (Round 9 §3.4): the explicit phase of the constraint state machine. It replaces
// the old `triggered_` + `entry_locked_` boolean pair: pre-trigger and post-call
// between-regions are different states (R9-I2), and TextLocked is the hardened
// pre-trigger lock (R8-01).
enum class ToolConstraintPhase : std::uint8_t {
    PreTrigger,    // no tool has begun (lazy: prose may precede the first entry)
    TextLocked,    // hardened mode committed visible text before a tool entry; later
                   // bytes stay ordinary text and later markers never trigger
    InRegion,      // parse_tool_call_region() is validating a triggered tool sequence
    BetweenCalls,  // at least one strict tool region completed; only formatting
                   // whitespace (' ' '\t' '\r' '\n'), the next legal tool marker, and
                   // EOS are legal — visible suffix content is rejected (R9-I1)
};

// R9-01: the outcome of one inactive-phase scan over a byte run.
enum class InactiveScanResult : std::uint8_t {
    Continued,  // the bytes were consumed; no trigger, no rejection
    Triggered,  // a complete marker fired (phase_ is InRegion, buffer_ seeded)
    Rejected,   // a strict-scan illegal byte (between-calls visible suffix content)
};
class ToolCallGrammarConstraint {
public:
    // `max_tool_name_length` mirrors the model's tool-name limit for the strict structural
    // check; zero selects the 64-byte default.
    // R6-01: `syntax` is the top-level entry syntax this constraint state shares with the
    // production parser (R6-I1: one syntax mode means one syntax mode). The trigger, the
    // marker classification, and the pending-prefix decision all use this mode, so the
    // CPU constraint core can never latch an entry the parser would treat as prose (or
    // vice versa). The default keeps the historical compatibility entry set for existing
    // callers.
    // R7-02/R8-01: `intent` mirrors the parser's tool-entry intent policy (R7-I5: the
    // constraint must eventually agree with the parser on both syntax and intent). Under
    // RequireToolAtContentStart the trigger is a content-start gate, not a bare
    // marker detector: visible non-whitespace content locks the gate at any point before
    // or after a completed call (the first latch is not permanent permission, R8-I2), a
    // complete marker under a locked gate is ordinary content, and only formatting
    // whitespace plus a directly consecutive wrapper keeps the tool-sequence eligibility
    // open (R8-I4). Trailing content after a closed region replays through the gate from
    // the parser's break offset (R8-I3).
    explicit ToolCallGrammarConstraint(std::size_t max_tool_name_length = 64,
                                       ToolCallSyntaxMode syntax =
                                           ToolCallSyntaxMode::Compatibility,
                                       ToolCallIntentPolicy intent =
                                           ToolCallIntentPolicy::TemplateCompatible);

    // Decide the legality of one candidate token's decoded bytes (the wire grammar is
    // byte-oriented and performs no further character-set validation, so UTF-8 multi-byte
    // characters split across token boundaries are handled by construction). The state is
    // not modified. A candidate that would break an open region is Rejected; a candidate
    // that ends inside an open structure is Allowed (a legal prefix); a candidate that
    // ends inside a marker trigger is NeedMore.
    [[nodiscard]] ToolCallConstraintVerdict check(std::string_view decoded_bytes) const;

    // Advance the state with previously checked bytes (Allowed or NeedMore). Committing a
    // Rejected candidate is a programmer error; debug builds assert on it.
    void commit(std::string_view decoded_bytes);

    // Speculative support: the state is value-semantic. checkpoint() captures the state
    // before a draft round; when the target rejects part of the draft, restore() discards
    // the draft's progression and the accepted prefix plus the correction token commits
    // from the checkpointed state. Committing a draft's bytes without a checkpoint commits
    // them for good.
    [[nodiscard]] ToolCallGrammarConstraint checkpoint() const;
    void restore(const ToolCallGrammarConstraint& state);

    // True while a tool region is open (the trigger fired, the region has not closed).
    // While inactive the constraint imposes no restriction.
    [[nodiscard]] bool active() const noexcept;
    // True while no region is open: initially (nothing has triggered) and after a region
    // closed (a new region may retrigger).
    [[nodiscard]] bool finished() const noexcept;
    // R9-01 (Round 9 §3.13): true when the input stream may legally end here (an EOS/EOT
    // candidate is admissible): no open region and no pending marker candidate. A
    // constrained decoder masks the EOS/EOT candidate while this is false instead of
    // feeding EOS as decoded bytes.
    [[nodiscard]] bool can_terminate() const noexcept;
    // Diagnostics only: the open region's byte count while active, otherwise the length
    // of the pending marker candidate (not a count of all bytes observed).
    [[nodiscard]] std::size_t observed_bytes() const noexcept;

private:
    ToolCallParsePolicy parse_policy() const;
    // Advance the internal state with one candidate; `target` receives the new state
    // (commit) while the member state stays untouched (check) — implemented by running
    // the byte rules on a copy of the state words.
    [[nodiscard]] ToolCallConstraintVerdict
    advance(std::string_view decoded_bytes, ToolCallGrammarConstraint* target) const;
    // R9-01 (Round 9 §3.5): the pre-trigger lazy scan. Under TemplateCompatible any byte
    // is ordinary prose; under RequireToolAtContentStart the first visible non-whitespace
    // byte or a failed marker candidate transitions the phase to TextLocked (R8-01), and
    // a complete marker under a locked gate is ordinary content (never a trigger). A
    // complete marker in PreTrigger seeds the region buffer and returns Triggered.
    [[nodiscard]] static InactiveScanResult
    advance_pretrigger(ToolCallGrammarConstraint& state, std::string_view text);
    // R9-01 (Round 9 §3.5): the between-calls strict scan. Only ' ' '\t' '\r' '\n', the
    // next legal tool marker (shared classifier, §3.6), and EOS are legal; any visible
    // byte and any failed marker candidate return Rejected — the strict parser would
    // demote the completed region as TrailingContent, and a constrained sampler must not
    // admit that continuation (R9-I1). No failed-candidate rescan: a stale leading '<'
    // is itself illegal suffix content.
    [[nodiscard]] static InactiveScanResult
    advance_between_calls(ToolCallGrammarConstraint& state, std::string_view text);
    std::size_t max_tool_name_length_;
    // R6-01: the selected top-level entry syntax (shared by the marker trigger, the
    // pending-prefix classification, and the region re-parse policy). Value-semantic:
    // checkpoint()/restore() carry it with the state words.
    ToolCallSyntaxMode syntax_;
    // R7-02: the tool-entry intent policy shared with the parser (R7-I5). Value-semantic:
    // checkpoint()/restore() carry it with the state words.
    ToolCallIntentPolicy intent_ = ToolCallIntentPolicy::TemplateCompatible;
    // Bytes of the open region since the trigger (empty while inactive).
    std::string buffer_;
    // R10 (R10-I1): the shared pre-trigger entry classifier — the same machine the
    // production pre-latch parser consumes. It owns the pre-trigger fence state, the
    // indented-literal state, the marker candidate, the held whitespace, and the intent
    // lock, so the constraint can never latch an entry the parser treats as prose (or
    // vice versa). BetweenCalls keeps its own strict marker scan (R9-I1: post-call is
    // deliberately stricter than the lazy pre-trigger; no fence escape hatch).
    // Value-semantic: checkpoint()/restore() carry it with the state words.
    ToolCallEntryScanner entry_;
    // The accumulating marker-trigger candidate of the between-calls strict scan (starts
    // with '<'; may contain further '<' bytes inside a quoted header value). Never tracked
    // in TextLocked (later bytes are ordinary text, R9-I2) or across a completed region
    // (a stale closer would flush as a failed candidate and reject the gate). The
    // pre-trigger candidate lives in the shared entry scanner (R10-I1).
    std::string marker_prefix_;
    // R9-01: the explicit phase (replaces the old triggered_ + entry_locked_ booleans).
    ToolConstraintPhase phase_ = ToolConstraintPhase::PreTrigger;
};


} // namespace ninfer::models::qwen3_5::frontend
