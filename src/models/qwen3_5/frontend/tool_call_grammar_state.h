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
    explicit ToolCallGrammarConstraint(std::size_t max_tool_name_length = 64);

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
    // True once the current region closed cleanly (a new region may retrigger).
    [[nodiscard]] bool finished() const noexcept;
    // Bytes observed since construction or the last region close (diagnostics).
    [[nodiscard]] std::size_t observed_bytes() const noexcept;

private:
    ToolCallParsePolicy parse_policy() const;
    // Advance the internal state with one candidate; `target` receives the new state
    // (commit) while the member state stays untouched (check) — implemented by running
    // the byte rules on a copy of the state words.
    [[nodiscard]] ToolCallConstraintVerdict
    advance(std::string_view decoded_bytes, ToolCallGrammarConstraint* target) const;
    std::size_t max_tool_name_length_;
    // Bytes of the open region since the trigger (empty while inactive).
    std::string buffer_;
    // The accumulating marker-trigger candidate (starts with '<') while inactive.
    std::string marker_prefix_;
    bool triggered_ = false;  // a complete marker fired; the region machine is live
};

// The strict region parse used by the constraint (and by the machine): one source of truth
// for in-region legality, independent of any recovery policy.
[[nodiscard]] ToolCallParseProgress parse_tool_call_region(std::string_view text,
                                                           const ToolCallParsePolicy& policy);

} // namespace ninfer::models::qwen3_5::frontend
