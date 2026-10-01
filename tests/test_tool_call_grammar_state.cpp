#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "models/qwen3_5/frontend/tool_call_grammar_state.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"

namespace {

using ninfer::models::qwen3_5::frontend::ToolCallConstraintVerdict;
using ninfer::models::qwen3_5::frontend::ToolCallGrammarConstraint;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Drives the constraint with one decoded token at a time and verifies the expected verdict.
// A Rejected candidate is never committed (the masking contract: it must have been excluded).
struct Harness {
    ToolCallGrammarConstraint constraint;

    void expect(ToolCallConstraintVerdict expected, std::string_view bytes, const char* what) {
        const ToolCallConstraintVerdict verdict = constraint.check(bytes);
        if (verdict != expected) {
            std::fprintf(stderr, "FAIL: %s: expected %d, got %d (bytes \"%.*s\")\n", what,
                         static_cast<int>(expected), static_cast<int>(verdict),
                         static_cast<int>(bytes.size()), bytes.data());
            ++failures;
        }
        if (verdict != ToolCallConstraintVerdict::Rejected) { constraint.commit(bytes); }
    }

    void allowed(std::string_view bytes, const char* what) {
        expect(ToolCallConstraintVerdict::Allowed, bytes, what);
    }

    void rejected(std::string_view bytes, const char* what) {
        expect(ToolCallConstraintVerdict::Rejected, bytes, what);
    }

    void need_more(std::string_view bytes, const char* what) {
        expect(ToolCallConstraintVerdict::NeedMore, bytes, what);
    }
};

// P4.12: a valid Qwen tool call is accepted end to end; the state activates on the trigger
// and finishes at the wrapper close.
void test_valid_tool_call_accepted() {
    Harness h;
    check(h.constraint.active() == false, "valid call: initial inactive");
    h.allowed("<tool_call>", "valid call: trigger");
    check(h.constraint.active(), "valid call: active after trigger");
    h.allowed("<function=read>", "valid call: function open");
    h.allowed("<parameter=path>", "valid call: parameter open");
    h.allowed("/tmp/a.txt", "valid call: parameter value");
    h.allowed("</parameter>", "valid call: parameter close");
    h.allowed("</function>", "valid call: function close");
    h.allowed("</tool_call>", "valid call: wrapper close");
    check(h.constraint.active() == false, "valid call: inactive after wrapper close");
    check(h.constraint.finished(), "valid call: finished after wrapper close");
}

// P4.12: an invalid function prefix is rejected once a byte breaks the header. The name
// value itself may still grow (it is value-scanned); the break is forced by whitespace
// followed by a byte that is neither '>' nor a value character.
void test_invalid_function_prefix_rejected() {
    Harness h;
    h.allowed("<tool_call><function=mem", "invalid prefix: the name value is still open");
    h.rejected(" x", "invalid prefix: whitespace then a non-terminator byte breaks the header");
}

// P4.12: a missing required delimiter cannot advance the state (strict wire syntax): the
// header name must terminate with '>' (optionally after format whitespace).
void test_missing_delimiter_cannot_advance() {
    Harness h;
    h.allowed("<tool_call><function=read", "missing delimiter: open header is a legal prefix");
    h.rejected(" x", "missing delimiter: the name must terminate with '>' after whitespace");
}

// P4.12: a partial marker returns NeedMore (the grammar cannot decide yet); prose never
// rejects.
void test_partial_marker_returns_need_more() {
    Harness h;
    h.need_more("<tool", "partial marker: <tool");
    h.need_more("_call", "partial marker: completes the keyword but not the tag");
    check(h.constraint.active() == false, "partial marker: still inactive after NeedMore");
    h.allowed(">", "partial marker: the closing '>' activates the region");
    check(h.constraint.active(), "partial marker: active after the full trigger");

    Harness prose;
    prose.allowed("reasoning with < and <tool_caller and << noise", "prose: never rejects");
    check(prose.constraint.active() == false, "prose: still inactive");
    prose.need_more("<in", "prose: a partial invoke trigger is pending, not rejected");
    // A non-marker candidate clears instead of rejecting.
    Harness clear_candidate;
    clear_candidate.allowed("<xyz", "prose: a non-marker candidate clears");
    check(clear_candidate.constraint.active() == false, "prose: cleared candidate stays inactive");
}

// P4.12: parameters may appear in any order; the constraint is wire-syntax only and does
// not encode the schema.
void test_optional_parameter_order() {
    Harness h;
    h.allowed("<function=write>", "parameter order: function open");
    h.allowed("<parameter=content>", "parameter order: second parameter first");
    h.allowed("body", "parameter order: value");
    h.allowed("</parameter>", "parameter order: close");
    h.allowed("<parameter=extra>", "parameter order: first parameter later");
    h.allowed("1", "parameter order: value");
    h.allowed("</parameter></function>", "parameter order: close both");
    check(h.constraint.finished(), "parameter order: the bare region completed");
}

// P4.12: required parameters are a contract concern, not wire syntax: a name-only region
// stays open, and a structural byte right after the complete header breaks it.
void test_required_parameter_handling() {
    Harness h;
    h.allowed("<function=read>", "required: function open with no parameters");
    h.allowed("</function>", "required: a bare close is wire-legal");
    check(h.constraint.active() == false, "required: the region closed");

    Harness open;
    open.allowed("<function=read>", "required: open, expecting parameter or close");
    open.rejected("@", "required: a structural byte right after the header breaks the body");
}

// P4.12: non-string JSON arguments (booleans, numbers, null) are arbitrary value bytes.
void test_non_string_json_arguments() {
    Harness h;
    h.allowed("<function=cfg>", "json args: function open");
    h.allowed("<parameter=flag>", "json args: parameter open");
    h.allowed("true", "json args: boolean value");
    h.allowed("</parameter>", "json args: parameter close");
    h.allowed("<parameter=depth>", "json args: parameter open");
    h.allowed("123", "json args: number value");
    h.allowed("</parameter>", "json args: parameter close");
    h.allowed("<parameter=opt>", "json args: parameter open");
    h.allowed("null", "json args: null value");
    h.allowed("</parameter></function>", "json args: close both");
}

// P4.12: string arguments: long values with markup and quoted closer pairs stay legal;
// the value commits at the real closer (P3.10 semantics shared from the parser).
void test_string_argument_handling() {
    Harness h;
    h.allowed("<function=write>", "string args: function open");
    h.allowed("<parameter=content>", "string args: parameter open");
    h.allowed("line 1\n</parameter>", "string args: a quoted closer pair inside a value");
    h.allowed("<function=nested></function>", "string args: markup inside a value");
    h.allowed("line 3", "string args: more value");
    h.allowed("</parameter>", "string args: the parameter close candidate");
    h.allowed("</function>", "string args: the function close commits value and region");
    check(h.constraint.finished(), "string args: the region closed");
}

// P4.12: multiple tool calls retrigger after a wrapper close.
void test_multiple_tool_calls() {
    Harness h;
    h.allowed("<tool_call><function=a></function></tool_call>", "multi: first region");
    check(h.constraint.active() == false, "multi: inactive between regions");
    h.allowed(" prose between regions ", "multi: prose after a region");
    h.allowed("<tool_call><function=b><parameter=x></parameter></function></tool_call>",
              "multi: second region");
    check(h.constraint.finished(), "multi: finished after the second region");
}
// F5: <function_calls> contains a sequence of calls: after a function close the constraint
// accepts the next call (the strict region parse, its shared legality source, completes the
// sequence) and finishes at the wrapper close.
void test_function_calls_call_sequence_accepted() {
    Harness h;
    h.allowed("<function_calls>", "F5 sequence: wrapper trigger");
    h.allowed("<function=read>", "F5 sequence: first call open");
    h.allowed("<parameter=path>", "F5 sequence: first parameter open");
    h.allowed("a", "F5 sequence: first value");
    h.allowed("</parameter>", "F5 sequence: first parameter close");
    h.allowed("</function>", "F5 sequence: first function close");
    h.allowed("<function=read>", "F5 sequence: second call open after the function close");
    h.allowed("<parameter=path>", "F5 sequence: second parameter open");
    h.allowed("b", "F5 sequence: second value");
    h.allowed("</parameter>", "F5 sequence: second parameter close");
    h.allowed("</function>", "F5 sequence: second function close");
    h.allowed("</function_calls>", "F5 sequence: wrapper close completes the region");
    check(h.constraint.finished(), "F5 sequence: the region completed");
}


// P4.12: reasoning and prose before the trigger are never constrained.
void test_reasoning_before_trigger() {
    Harness h;
    h.allowed("", "reasoning: close think tag is prose");
    h.allowed("now: ", "reasoning: prose before the trigger");
    check(h.constraint.active() == false, "reasoning: inactive until the trigger");
    h.allowed("<tool_call>", "reasoning: the trigger activates");
    check(h.constraint.active(), "reasoning: active after the trigger");
}

// P4.12: the grammar completes at the correct boundary; subsequent prose is free.
void test_grammar_completes_at_correct_boundary() {
    Harness h;
    h.allowed("<tool_call><function=a><parameter=x>v</parameter></function>",
              "boundary: open region");
    check(h.constraint.active(), "boundary: active before the wrapper close");
    h.allowed("</tool_call>", "boundary: the wrapper close completes the region");
    check(h.constraint.finished(), "boundary: finished at the boundary");
    h.allowed("follow-up prose with < markers <tool_caller and </function> inside",
              "boundary: prose after completion is free");
    check(h.constraint.finished(), "boundary: still finished after prose");
}

// P4.12: UTF-8 multi-byte characters split across token boundaries (byte-oriented syntax).
void test_utf8_split_across_tokens() {
    Harness h;
    h.allowed("<function=read>", "utf8: function open");
    h.allowed("<parameter=path>", "utf8: parameter open");
    h.allowed("\xc3", "utf8: first byte of a two-byte character");
    h.allowed("\xa9", "utf8: second byte of the character");
    h.allowed("</parameter></function>", "utf8: close both");
}

// P4.12 / P4.8: checkpoint and restore — the state is value-semantic.
void test_checkpoint_restore() {
    const ToolCallGrammarConstraint fresh;
    Harness h;
    h.allowed("<tool_call><function=read>", "checkpoint: region open");
    h.allowed("<parameter=path>", "checkpoint: parameter open");

    const ToolCallGrammarConstraint before_draft = h.constraint.checkpoint();
    // Draft: a legal value prefix.
    h.allowed("value", "checkpoint: draft commits legally");
    check(h.constraint.observed_bytes() > before_draft.observed_bytes(),
          "checkpoint: the draft advanced the state");

    // The target rejects the draft suffix: roll back and commit the correction instead.
    h.constraint.restore(before_draft);
    check(h.constraint.check("</parameter>") == ToolCallConstraintVerdict::Allowed,
          "checkpoint: the checkpointed state accepts the closer");
    h.constraint.commit("</parameter>");
    h.constraint.commit("</function>");
    h.constraint.commit("</tool_call>");
    check(h.constraint.finished(), "checkpoint: the corrected region completed");

    // Restore to the fresh state: behavior must match the fresh instance.
    h.constraint.restore(fresh);
    check(h.constraint.check("<tool") == ToolCallConstraintVerdict::NeedMore,
          "checkpoint: restored state matches fresh behavior");
}

// P4.12: rollback of a rejected draft — the rejected candidate never commits; the
// corrected token commits from the checkpointed state. The draft suffix that breaks the
// region is a structural byte right after the complete header.
void test_rejected_draft_rollback() {
    Harness h;
    h.allowed("<tool_call><function=read>", "draft rollback: region open");

    const ToolCallGrammarConstraint checkpoint = h.constraint.checkpoint();
    // The draft proposes a structural byte where a parameter or the function close is
    // expected — the constraint rejects it, so it never commits.
    check(h.constraint.check("x") == ToolCallConstraintVerdict::Rejected,
          "draft rollback: the breaking draft suffix is rejected");
    // The draft is rejected: restore and take the correction path instead.
    h.constraint.restore(checkpoint);
    h.allowed("</function>", "draft rollback: the correction closes the function");
    h.allowed("</tool_call>", "draft rollback: the wrapper closes the region");
    check(h.constraint.finished(), "draft rollback: finished after the correction");
}

// P4.12: the state progression of an accepted draft prefix (committed without rollback).
void test_accepted_draft_prefix() {
    Harness h;
    h.allowed("<tool_call><function=read>", "draft prefix: region open");
    h.allowed("<parameter=path>", "draft prefix: parameter open");
    h.allowed("/etc/", "draft prefix: the draft prefix commits for good");
    h.allowed("hostname", "draft prefix: the target accepts the continuation");
    h.allowed("</parameter></function></tool_call>", "draft prefix: region completes");
    check(h.constraint.finished(), "draft prefix: finished");
}

// P4.12 / 7.3: truncation at every byte boundary — a valid region must never reject under
// any tokenization of its bytes, up to the point where the region completes.
void test_all_byte_splits_of_valid_region() {
    const std::string_view region =
        "<tool_call><function=write><parameter=path>/tmp/out.txt</parameter>"
        "<parameter=content>line one\nline two with <markup></parameter></function></tool_call>";
    for (std::size_t split = 0; split <= region.size(); ++split) {
        ToolCallGrammarConstraint constraint;
        bool legal = true;
        bool saw_finish = false;
        std::size_t pos = 0;
        for (; pos < region.size() && legal && !saw_finish; ++pos) {
            const bool was_active = constraint.active();
            const std::string_view first = region.substr(pos, 1);
            if (constraint.check(first) == ToolCallConstraintVerdict::Rejected) { legal = false; break; }
            constraint.commit(first);
            if (was_active && constraint.finished()) { saw_finish = true; break; }
            // At the split boundary the remainder travels as one long candidate token:
            // this exercises both the short and the long candidate path per split.
            const std::size_t rest = region.size() - pos - 1;
            if (split > 0 && pos + 1 == split && rest > 0) {
                const std::string_view second = region.substr(pos + 1, rest);
                if (constraint.check(second) == ToolCallConstraintVerdict::Rejected) { legal = false; break; }
                constraint.commit(second);
                if (was_active && constraint.finished()) { saw_finish = true; }
            }
        }
        if (!legal) {
            std::fprintf(stderr, "FAIL: all byte splits: rejection at split %zu (pos %zu)\n", split, pos);
            ++failures;
            break;
        }
    }
    ToolCallGrammarConstraint whole;
    whole.commit(region);
    check(whole.finished(), "all byte splits: the whole region completes");
}

// F8: a marker candidate broken by a second '<' publishes the failed candidate bytes and
// retains the breaking '<' as a fresh marker candidate: the inner <tool_call> triggers at
// the machine's latch byte (shared marker transition rule).
void test_marker_breaking_angle_restarts_candidate() {
    Harness h;
    h.need_more("<function<", "F8 restart: the failing function prefix publishes, '<' restarts");
    check(h.constraint.active() == false, "F8 restart: no region before the inner marker");
    h.allowed("tool_call>", "F8 restart: the inner wrapper completes the fresh candidate");
    check(h.constraint.active(), "F8 restart: the inner wrapper triggered the region");
    h.allowed("<function=read>", "F8 restart: function open inside the region");
    h.allowed("<parameter=path>", "F8 restart: parameter open");
    h.allowed("/x", "F8 restart: value");
    h.allowed("</parameter>", "F8 restart: parameter close");
    h.allowed("</function>", "F8 restart: function close");
    h.allowed("</tool_call>", "F8 restart: wrapper close completes the region");
    check(h.constraint.finished(), "F8 restart: the region completed");
}

// Review (M1): a '<' inside a quoted header value keeps the marker candidate open (the old
// reset rule discarded it and treated the whole header as prose). At the header close the
// constraint triggers at the machine's latch byte and rejects there: the name 'a<b' violates
// the tool-name grammar — a definitive structural break the final parser also rejects.
void test_quoted_angle_bracket_in_header_keeps_candidate() {
    Harness h;
    h.need_more("<function name=\"a<b\"", "quoted '<': the closed value keeps the candidate");
    h.rejected(">", "quoted '<': the latch byte breaks the region definitively");
}

// F7/F8/I5: the stream parser latch and the grammar-constraint lazy trigger agree on the
// trigger byte for every corpus line (both consumers step the same marker transition rule).
void test_marker_stream_and_constraint_latch_equivalence() {
    // The quoted-angle line ("<function name=\"a<b\">") is excluded: its marker is
    // syntactically complete at the latch byte, but the region breaks there (invalid name),
    // so the constraint rejects the latch byte while the stream latches syntactically — a
    // different predicate, pinned by test_quoted_angle_bracket_in_header_keeps_candidate.
    const std::vector<std::string> corpus = {{"<function<tool_call>"},
                                              {"abc <tool_"},
                                              {"abc <tool_call>"},
                                              {"abc <<tool_call>"},
                                              {"prose <function= x\n<tool_call>"},
                                              {"<tool"},
                                              {"<function"},
                                              {"x <parameter=content> y"}};
    for (const std::string& text : corpus) {
        int stream_latch = -1;
        ninfer::models::qwen3_5::frontend::ToolCallStreamParser machine(
            ninfer::models::qwen3_5::frontend::ToolCallParsePolicy{});
        for (std::size_t i = 0; i < text.size(); ++i) {
            machine.feed(text.substr(i, 1));
            if (stream_latch < 0 && machine.latched()) { stream_latch = static_cast<int>(i); }
        }
        int constraint_trigger = -1;
        ToolCallGrammarConstraint constraint;
        for (std::size_t i = 0; i < text.size(); ++i) {
            const std::string_view byte = text.substr(i, 1);
            const ToolCallConstraintVerdict verdict = constraint.check(byte);
            if (verdict != ToolCallConstraintVerdict::Rejected) { constraint.commit(byte); }
            if (constraint_trigger < 0 && constraint.active()) {
                constraint_trigger = static_cast<int>(i);
            }
        }
        check(stream_latch == constraint_trigger,
              ("latch equivalence diverged for \"" + text + "\"").c_str());
    }
}


// R6-01 (Round 6 §3.6): the constraint CPU core follows the selected syntax mode. Native:
// only the wrapped <tool_call> entry triggers; the compatibility-only top-level entries
// (bare function/invoke, function_calls) are ordinary prose. Compatibility: all historical
// top-level markers retain their baseline trigger behavior.
void test_r6_constraint_syntax_mode_matrix() {
    using Mode = ninfer::ToolCallSyntaxMode;
    // Native mode: the complete wrapper triggers.
    {
        ToolCallGrammarConstraint constraint(64, Mode::QwenWrappedNative);
        const ToolCallConstraintVerdict verdict = constraint.check("<tool_call>");
        check(verdict == ToolCallConstraintVerdict::Allowed,
              "r6-01 native: complete wrapper is a legal trigger prefix");
        check(!constraint.active(), "r6-01 native: check leaves state untouched");
        constraint.commit("<tool_call>");
        check(constraint.active(), "r6-01 native: active after the complete wrapper");
    }
    // Native mode: every proper prefix of the wrapper stays NeedMore and never triggers.
    {
        const std::string_view full = "<tool_call>";
        for (std::size_t length = 1; length < full.size(); ++length) {
            ToolCallGrammarConstraint constraint(64, Mode::QwenWrappedNative);
            const ToolCallConstraintVerdict verdict = constraint.check(full.substr(0, length));
            check(verdict == ToolCallConstraintVerdict::NeedMore,
                  "r6-01 native: wrapper prefix held as NeedMore");
            check(!constraint.active(), "r6-01 native: wrapper prefix never triggers");
        }
    }
    // Native mode: compatibility-only entries are unconstrained content, never active.
    {
        for (const std::string_view entry :
             {std::string_view("<function=read>"), std::string_view("<invoke=read>"),
              std::string_view("<function_calls>")}) {
            ToolCallGrammarConstraint constraint(64, Mode::QwenWrappedNative);
            const ToolCallConstraintVerdict verdict = constraint.check(entry);
            check(verdict == ToolCallConstraintVerdict::Allowed,
                  "r6-01 native: compatibility-only entry is unconstrained content");
            check(!constraint.active(), "r6-01 native: compatibility-only entry never activates");
            constraint.commit(entry);
            check(!constraint.active(), "r6-01 native: still inactive after commit");
        }
    }
    // Compatibility mode: every historical top-level marker retains the baseline verdicts.
    {
        for (const std::string_view entry :
             {std::string_view("<tool_call>"), std::string_view("<function_calls>"),
              std::string_view("<function=read>"), std::string_view("<invoke=read>")}) {
            ToolCallGrammarConstraint constraint(64, Mode::Compatibility);
            const ToolCallConstraintVerdict verdict = constraint.check(entry);
            check(verdict == ToolCallConstraintVerdict::Allowed,
                  "r6-01 compat: historical marker remains a legal trigger");
            constraint.commit(entry);
            check(constraint.active(), "r6-01 compat: historical marker activates");
        }
    }
}

// R6-01 (Round 6 §3.7): for each top-level entry, the parser's entry decision and the
// constraint's trigger decision must agree in the same syntax mode (one syntax mode means
// one syntax mode, R6-I1). CPU only: the one-shot parse is the shared incremental machine.
void test_r6_constraint_parser_entry_cross_check() {
    using Mode = ninfer::ToolCallSyntaxMode;
    const ninfer::models::qwen3_5::frontend::ToolCallOutputContract contract;
    struct Entry {
        std::string_view marker;
        std::string_view region;
    };
    const Entry entries[] = {
        {"<tool_call>", "<tool_call>\n<function=read>\n</function>\n</tool_call>"},
        {"<function_calls>",
         "<function_calls>\n<function=read>\n</function>\n</function_calls>"},
        {"<function=read>", "<function=read>\n</function>"},
        {"<invoke=read>", "<invoke=read>\n</invoke>"},
    };
    for (const Mode mode : {Mode::QwenWrappedNative, Mode::Compatibility}) {
        for (const Entry& entry : entries) {
            const auto parsed = ninfer::models::qwen3_5::frontend::parse_qwen_tool_call_output(
                entry.region, 64, contract, false, ninfer::FinishReason::StopToken, mode,
                ninfer::ToolCallAmbiguityPolicy::PayloadFidelity);
            ToolCallGrammarConstraint constraint(64, mode);
            constraint.commit(entry.marker);
            check(parsed.is_tool_call_response == constraint.active(),
                  "r6-01 cross-check: parser and constraint entry decisions agree per mode");
        }
    }
}

// R6-01 (Round 6 §3.5): checkpoint/restore is value-semantic and must carry the syntax mode
// with the state words: a restored native instance still treats a bare <function=...> as
// ordinary prose (a future restore that copies buffer state but not policy state is caught).
void test_r6_constraint_checkpoint_restore_preserves_syntax() {
    using Mode = ninfer::ToolCallSyntaxMode;
    ToolCallGrammarConstraint constraint(64, Mode::QwenWrappedNative);
    constraint.commit("<tool_call>\n<function=read>\n");
    check(constraint.active(), "r6-01 restore: active before checkpoint");
    const ToolCallGrammarConstraint checkpoint = constraint.checkpoint();
    constraint.commit("</function>\n</tool_call>");
    check(!constraint.active(), "r6-01 restore: region closed");
    constraint.restore(checkpoint);
    check(constraint.active(), "r6-01 restore: active after restore");
    constraint.commit("</function>\n</tool_call>\n");
    check(!constraint.active(), "r6-01 restore: region re-closed after restore");
    const ToolCallConstraintVerdict verdict = constraint.check("<function=read>");
    check(verdict == ToolCallConstraintVerdict::Allowed,
          "r6-01 restore: bare function stays content under the restored native mode");
    check(!constraint.active(),
          "r6-01 restore: restored native mode does not latch a bare function");
}
} // namespace

int main() {
    test_valid_tool_call_accepted();
    test_invalid_function_prefix_rejected();
    test_missing_delimiter_cannot_advance();
    test_partial_marker_returns_need_more();
    test_optional_parameter_order();
    test_required_parameter_handling();
    test_non_string_json_arguments();
    test_string_argument_handling();
    test_multiple_tool_calls();
    test_function_calls_call_sequence_accepted();
    test_reasoning_before_trigger();
    test_grammar_completes_at_correct_boundary();
    test_utf8_split_across_tokens();
    test_checkpoint_restore();
    test_rejected_draft_rollback();
    test_accepted_draft_prefix();
    test_all_byte_splits_of_valid_region();
    test_marker_breaking_angle_restarts_candidate();
    test_marker_stream_and_constraint_latch_equivalence();
    test_quoted_angle_bracket_in_header_keeps_candidate();
    test_r6_constraint_syntax_mode_matrix();
    test_r6_constraint_parser_entry_cross_check();
    test_r6_constraint_checkpoint_restore_preserves_syntax();
    if (failures == 0) { std::puts("tool_call_grammar_state tests: all passed"); }
    return failures == 0 ? 0 : 1;
}
