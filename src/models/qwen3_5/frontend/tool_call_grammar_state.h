#pragma once

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
// constraint is inactive again and a later region may retrigger.
enum class ToolCallConstraintVerdict : std::uint8_t {
    Allowed,   // the bytes are a legal continuation (the state advances; the region may close)
    Rejected,  // the bytes break the wire syntax where a legal continuation existed before
    NeedMore,  // the bytes end inside a marker trigger; the grammar cannot decide yet (legal
               // so far: treat as Allowed when masking, the state stays pending)
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
    // The accumulating marker-trigger candidate while inactive (starts with '<'; may
    // contain further '<' bytes inside a quoted header value).
    std::string marker_prefix_;
    bool triggered_ = false;  // a complete marker fired; the region machine is live
    // R7-02/R8-01: the content-start gate is locked (visible non-formatting-whitespace
    // content appeared at any point, before or after a completed call). While locked under
    // RequireToolAtContentStart a complete marker is ordinary content and never triggers.
    bool entry_locked_ = false;
};


} // namespace ninfer::models::qwen3_5::frontend
