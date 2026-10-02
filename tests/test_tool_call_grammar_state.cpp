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

// P4.12: multiple tool calls retrigger after a wrapper close. R9-01: visible prose
// between the regions is rejected suffix content; only formatting whitespace and a
// directly consecutive wrapper stay legal.
void test_multiple_tool_calls() {
    Harness h;
    h.allowed("<tool_call><function=a></function></tool_call>", "multi: first region");
    check(h.constraint.active() == false, "multi: inactive between regions");
    h.allowed(" \n", "multi: whitespace between regions stays legal");
    h.rejected(" prose between regions ", "multi: prose after a region is rejected (R9-01)");
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

// P4.12 / R9-01: the grammar completes at the correct boundary; subsequent visible
// content is rejected suffix content (the constrained sampler cannot admit it).
void test_grammar_completes_at_correct_boundary() {
    Harness h;
    h.allowed("<tool_call><function=a><parameter=x>v</parameter></function>",
              "boundary: open region");
    check(h.constraint.active(), "boundary: active before the wrapper close");
    h.allowed("</tool_call>", "boundary: the wrapper close completes the region");
    check(h.constraint.finished(), "boundary: finished at the boundary");
    h.rejected("follow-up prose with < markers <tool_caller and </function> inside",
               "boundary: prose after completion is rejected (R9-01)");
    check(h.constraint.finished(), "boundary: still finished after the rejected prose");
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
                break;  // R9-01: the remainder is fully committed; re-reading committed bytes would re-enter the strict between-calls state
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
        check(stream_latch == constraint_trigger, "latch equivalence diverged");
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
                std::string(entry.region), 64, contract, false, ninfer::FinishReason::StopToken, mode,
                ninfer::ToolCallAmbiguityPolicy::PayloadFidelity);
            ToolCallGrammarConstraint constraint(64, mode);
            constraint.commit(entry.marker);
            check(parsed.is_tool_call_response == constraint.active(),
                  "r6-01 cross-check: parser and constraint entry decisions agree per mode");
        }
    }
}

// R6-01 (Round 6 §3.5): checkpoint/restore is value-semantic and must carry the syntax mode
// with the state words: a restored native instance still rejects a bare <function=...> as
// non-wrapped suffix content after a closed region (a restore that copies buffer state but
// not policy state is caught; R9-01 §3.15 sharpens "stays content" to "is rejected").
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
    check(verdict == ToolCallConstraintVerdict::Rejected,
          "r6-01 restore: bare function after a closed native region is rejected suffix content");
    check(!constraint.active(),
          "r6-01 restore: the rejected bare function does not latch a region");
}

// R7 (Round 7 §5.7): the tool-entry intent policy is shared between constraint and
// parser. Under RequireToolAtContentStart the constraint's trigger is a content-start
// gate, not a bare marker detector: visible non-whitespace content before the first
// entry point locks the gate, a complete marker under a locked gate is ordinary content,
// and only formatting whitespace between a closed region and the next wrapper keeps the
// consecutive-wrapper eligibility open. TemplateCompatible keeps the historical entry
// semantics (prose + marker may still trigger).
void test_r7_intent_content_start_gate() {
    using Mode = ninfer::ToolCallSyntaxMode;
    using Intent = ninfer::ToolCallIntentPolicy;
    const char* marker = "<tool_call>";
    const char* full_call =
        "<tool_call><function=bash><parameter=command>echo hi</parameter></function></tool_call>";

    // 1. TemplateCompatible baseline: prose + marker may still trigger (historical).
    {
        ToolCallGrammarConstraint constraint;
        constraint.commit("Example: ");
        constraint.commit(marker);
        check(constraint.active(), "r7 tc: prose + marker still triggers the region");
    }

    // 2. SOC: prose + marker is content, never a trigger.
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit("Example: ");
        constraint.commit(marker);
        check(!constraint.active(), "r7 soc: prose + marker is content, no trigger");
        check(constraint.finished(), "r7 soc: the gate stays closed after prose");
    }

    // 3. SOC: formatting whitespace + marker still triggers (whitespace is not content).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit("\n \t\n");
        constraint.commit(marker);
        check(constraint.active(), "r7 soc: whitespace + marker triggers");
    }

    // 4. SOC: a failed marker candidate is visible content and locks the gate; a later
    //    complete marker then stays content.
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit("<tool_x>");
        constraint.commit(marker);
        check(!constraint.active(), "r7 soc: failed candidate locks the gate");
    }

    // 5. SOC: consecutive wrappers keep eligibility (formatting whitespace only between
    //    the close and the next marker).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit(full_call);
        check(!constraint.active(), "r7 soc: first wrapper closed");
        constraint.commit(std::string("\n") + marker);
        check(constraint.active(), "r7 soc: consecutive wrapper re-triggers after whitespace");
    }

    // 6. SOC: visible prose after a completed call is rejected (R9-01: the R8 gate lock
    //    becomes a hard rejection); a rejected candidate never commits.
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit(full_call);
        check(!constraint.active(), "r8 soc 6: first wrapper closed");
        check(constraint.check("\nor so.\n") == ToolCallConstraintVerdict::Rejected,
              "r9 soc 6: prose after a completed call is rejected");
        check(!constraint.active(), "r9 soc 6: the rejected suffix leaves the state inactive");
    }

    // 7. Checkpoint: a speculative draft that emitted prose must not permanently lock
    //    the restored state.
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        const ToolCallGrammarConstraint gate_open = constraint.checkpoint();
        constraint.commit("visible prose");
        check(constraint.finished(), "r7 soc 7: prose locks the draft state");
        constraint.commit(marker);
        check(!constraint.active(), "r7 soc 7: the locked draft does not trigger");
        constraint.restore(gate_open);
        constraint.commit(marker);
        check(constraint.active(), "r7 soc 7: the restored gate-open state triggers");
    }

    // 8. Syntax x intent cross product: the gate is orthogonal to the syntax mode.
    {
        // (native, TC): prose + marker may trigger.
        ToolCallGrammarConstraint native_tc(64, Mode::QwenWrappedNative);
        native_tc.commit("Example: ");
        native_tc.commit(marker);
        check(native_tc.active(), "r7 xprod: (native, tc) prose + marker triggers");
        // (native, SOC): prose + marker is content.
        ToolCallGrammarConstraint native_soc(64, Mode::QwenWrappedNative, Intent::RequireToolAtContentStart);
        native_soc.commit("Example: ");
        native_soc.commit(marker);
        check(!native_soc.active(), "r7 xprod: (native, soc) prose + marker is content");
        // (compat, TC): a function_calls wrapper after prose may trigger.
        ToolCallGrammarConstraint compat_tc;
        compat_tc.commit("Example: ");
        compat_tc.commit("<function_calls>");
        check(compat_tc.active(), "r7 xprod: (compat, tc) wrapper after prose triggers");
        // (compat, SOC): the same wrapper is content.
        ToolCallGrammarConstraint compat_soc(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        compat_soc.commit("Example: ");
        compat_soc.commit("<function_calls>");
        check(!compat_soc.active(), "r7 xprod: (compat, soc) wrapper after prose is content");
    }
}

// R8-01 (Round 8 §3): post-call hardened intent parity. The first latch is not permanent
// permission (R8-I2): visible content after a completed call locks the gate, formatting
// whitespace and a directly consecutive wrapper keep eligibility (R8-I4), and the state
// stays value-semantic (R8-I5). TemplateCompatible keeps the historical semantics.
void test_r8_constraint_post_call_gate() {
    using Mode = ninfer::ToolCallSyntaxMode;
    using Intent = ninfer::ToolCallIntentPolicy;
    const char* marker = "<tool_call>";
    const char* call1 =
        "<tool_call><function=bash><parameter=command>echo hi</parameter></function></tool_call>";
    const char* call2 =
        "<tool_call><function=bash><parameter=command>echo again</parameter></function></tool_call>";

    // A: prose + CALL — the pre-latch lock (the R7-02 baseline control).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit("prose ");
        constraint.commit(marker);
        check(!constraint.active(), "r8 A: prose before the first call is content");
    }

    // B: CALL1 alone — the region closes and the gate stays open for the next entry.
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit(call1);
        check(!constraint.active(), "r8 B: the first region closes");
        constraint.commit(marker);
        check(constraint.active(), "r8 B: the gate is open after a closed region");
    }

    // E (split): CALL1 + prose + partial CALL2 — R9-01: the visible prose between the
    // calls rejects the whole candidate (a rejected candidate never commits; Round 9
    // §3.8/§3.11).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit(call1);
        check(!constraint.active(), "r8 E: first region closed");
        check(constraint.check(std::string("\nvisible prose\n<tool_") + "call>") ==
                  ToolCallConstraintVerdict::Rejected,
              "r9 E: the prose suffix rejects the whole candidate");
        check(!constraint.active(), "r9 E: the rejected candidate leaves the state inactive");
    }

    // J (same commit): CALL1 + prose + CALL2 in one chunk — no token boundary may hide
    // the visible content between the calls; R9-01: the chunk rejects (Round 8 §3.10,
    // Round 9 §3.8).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        check(constraint.check(std::string(call1) + "\nvisible prose\n" + call2) ==
                  ToolCallConstraintVerdict::Rejected,
              "r9 J: the visible content between the calls rejects the chunk");
        check(!constraint.active(), "r9 J: the rejected chunk leaves the state inactive");
    }

    // I (partial failed marker): CALL1 + \n<tool_ then x> — R9-01: the partial second
    // marker is a legal prefix (NeedMore, not an open region); the failed continuation
    // rejects; a rejected candidate never commits, so the pending marker survives for
    // the good continuation (Round 8 §3.15, Round 9 §3.11).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit(std::string(call1) + "\n<tool_");
        check(constraint.check("x>") == ToolCallConstraintVerdict::Rejected,
              "r9 I: the failed second marker is rejected suffix content");
        check(constraint.check("call>") == ToolCallConstraintVerdict::Allowed,
              "r9 I: the good continuation completes the second entry");
        constraint.commit("call>");
        check(constraint.active(), "r9 I: the second region is active");
    }

    // H (control, partial immediate CALL2): no visible content between the calls — the
    // pending second marker keeps the chunk a legal prefix (R9-01: the state is
    // BetweenCalls with a pending marker, not an open region; Round 8 §3.14).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit(std::string(call1) + "\n<tool_");
        check(!constraint.active(), "r9 H: the partial CALL2 is a pending marker, not an open region");
        constraint.commit("call>");
        check(constraint.active(), "r9 H: the second wrapper resumes and triggers");
    }

    // M (checkpoint): the post-call state is value-semantic — a draft that emitted prose
    // is rejected (the draft never commits it); the restore recovers the clean post-call
    // state, where the next wrapper still triggers (Round 9 §3.14).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit(call1);
        const ToolCallGrammarConstraint cp = constraint.checkpoint();
        check(constraint.check("visible prose") == ToolCallConstraintVerdict::Rejected,
              "r9 M: a draft that emits prose after a call is rejected");
        constraint.restore(cp);
        constraint.commit(std::string("\n") + marker);
        check(constraint.active(), "r8 M: the restore keeps the post-call gate open");
    }

    // R8-01b (marker-suffix skip): CALL1 + prose + partial CALL2 in one commit — R9-01:
    // the TrailingContent tail replays through the strict between-calls scan and the
    // prose rejects the chunk; the rejected chunk never commits (Round 8 §3.5/§3.7,
    // Round 9 §3.8).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        check(constraint.check(std::string(call1) + "\nvisible prose\n<tool_") ==
                  ToolCallConstraintVerdict::Rejected,
              "r9 01b: the single-chunk prose suffix rejects the chunk");
        check(!constraint.active(), "r9 01b: the rejected chunk leaves the state inactive");
    }

    // TC control: post-call strictness is orthogonal to the intent policy (R9-I3) — even
    // under TemplateCompatible the visible suffix rejects; only whitespace + a marker is
    // legal after a completed call (Round 9 §3.12).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::TemplateCompatible);
        check(constraint.check(std::string(call1) + "\nvisible prose\n" + call2) ==
                  ToolCallConstraintVerdict::Rejected,
              "r9 TC: post-call prose rejects even under TemplateCompatible");
        check(!constraint.active(), "r9 TC: the rejected chunk leaves the state inactive");
    }
}

// R8-01 (Round 8 §6/§7/§8): the post-call chunk/checkpoint matrix. Every fixture runs
// whole-chunk and bytewise with an agreeing final state; consecutive wrappers keep
// eligibility, visible content (including form feed) locks, and checkpoint/restore
// recovers the exact post-call gate including a partial second marker.
void test_r8_post_call_chunk_checkpoint_matrix() {
    using Mode = ninfer::ToolCallSyntaxMode;
    using Intent = ninfer::ToolCallIntentPolicy;
    const char* marker = "<tool_call>";
    const char* call1 =
        "<tool_call><function=bash><parameter=command>echo hi</parameter></function></tool_call>";
    const char* call2 =
        "<tool_call><function=bash><parameter=command>echo again</parameter></function></tool_call>";

    // C/D/K: no gap, mixed-whitespace gap, and CRLF gap — the second wrapper stays
    // eligible (R8-I4), and whole-chunk vs bytewise commits agree (Round 8 §3.12/§7).
    for (const std::string& gap : {std::string{}, std::string("\n\t \r\n"), std::string("\r\n")}) {
        const std::string text = std::string(call1) + gap + call2;
        ToolCallGrammarConstraint whole(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        whole.commit(text);
        check(!whole.active(), "r8 matrix C/D/K: the whole-chunk region ends closed");
        ToolCallGrammarConstraint bytewise(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        for (std::size_t i = 0; i < text.size(); ++i) { bytewise.commit(text.substr(i, 1)); }
        check(bytewise.active() == whole.active() && bytewise.finished() == whole.finished(),
              "r8 matrix C/D/K: the bytewise final state agrees with the whole chunk");
        ToolCallGrammarConstraint split(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        split.commit(std::string(call1) + gap);
        split.commit(marker);
        check(split.active(), "r8 matrix C/D/K: the second wrapper marker stays eligible");
    }

    // E/L: visible prose and form feed after the first call reject in both whole-chunk
    // and bytewise form (R9-01: a rejected candidate never commits; form feed is visible
    // content, not format whitespace; Round 8 §3.21, Round 9 §3.8).
    for (const std::string& suffix : {std::string("\nvisible prose\n"), std::string("\f")}) {
        const std::string text = std::string(call1) + suffix + call2;
        ToolCallGrammarConstraint whole(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        check(whole.check(text) == ToolCallConstraintVerdict::Rejected,
              "r9 matrix E/L: visible content after the first call rejects the chunk");
        ToolCallGrammarConstraint bytewise(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        for (std::size_t i = 0; i < std::string(call1).size(); ++i) { bytewise.commit(text.substr(i, 1)); }
        check(bytewise.check(text.substr(std::string(call1).size())) ==
                  ToolCallConstraintVerdict::Rejected,
              "r9 matrix E/L: the bytewise form rejects at the visible suffix");
    }

    // N (checkpoint inside a partial second marker): the failed draft rejects (a rejected
    // candidate never commits); the restore recovers the exact pending-marker state so the
    // marker can still complete (Round 8 §3.16, Round 9 §3.14).
    {
        ToolCallGrammarConstraint constraint(64, Mode::Compatibility, Intent::RequireToolAtContentStart);
        constraint.commit(std::string(call1) + "\n<tool_");
        const ToolCallGrammarConstraint cp = constraint.checkpoint();
        check(constraint.check("x>") == ToolCallConstraintVerdict::Rejected,
              "r9 N: the failed continuation is rejected");
        constraint.restore(cp);
        constraint.commit("call>");
        check(constraint.active(), "r9 N: the restore keeps the partial marker completable");
    }

    // Syntax cross: the post-call strictness is orthogonal to the entry syntax (Round 8
    // §3.19, Round 9 §3.15).
    {
        const std::string text = std::string(call1) + "\nvisible prose\n" + call2;
        ToolCallGrammarConstraint native(64, Mode::QwenWrappedNative, Intent::RequireToolAtContentStart);
        check(native.check(text) == ToolCallConstraintVerdict::Rejected,
              "r9 xprod: (native, soc) post-call prose rejects");
        ToolCallGrammarConstraint native_ws(64, Mode::QwenWrappedNative, Intent::RequireToolAtContentStart);
        native_ws.commit(std::string(call1) + "\n");
        native_ws.commit(marker);
        check(native_ws.active(), "r8 xprod: (native, soc) post-call whitespace stays open");
    }
}

// R9-01: once a strict tool sequence has started, the grammar constraint must reject the
// visible suffix content that the strict final parser would demote as TrailingContent (a
// complete call followed by non-whitespace parses to zero structured calls). Lazy
// pre-trigger semantics are unchanged: prose before the first call never rejects.
void test_r9_constraint_post_call_suffix() {
    using Mode   = ninfer::ToolCallSyntaxMode;
    using Intent = ninfer::ToolCallIntentPolicy;
    const Mode syntax = Mode::QwenWrappedNative;
    const std::string call_a =
        "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>";
    const std::string call_b =
        "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>";
    const std::string marker = "<tool_call>";
    const std::string open_a =
        "<tool_call><function=weather><parameter=city>Paris</parameter></function>";

    // G1: a complete call is a legal continuation and leaves the state inactive.
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        check(constraint.check(call_a) == ToolCallConstraintVerdict::Allowed && !constraint.active(),
              "r9 G1: a complete call is a legal continuation");
    }
    // G2: a formatting-whitespace suffix after a completed call stays legal.
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        constraint.commit(call_a);
        check(constraint.check("\n\t \r\n") == ToolCallConstraintVerdict::Allowed &&
                  !constraint.active(),
              "r9 G2: a whitespace suffix after a call stays allowed");
    }
    // G3: visible prose after a completed call is rejected (the strict parser would demote
    // the complete call to TrailingContent).
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        constraint.commit(call_a);
        check(constraint.check("\nDone") == ToolCallConstraintVerdict::Rejected,
              "r9 G3: visible prose after a call is rejected");
    }
    // G4: a form-feed suffix is rejected (visible, not formatting whitespace).
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        constraint.commit(call_a);
        check(constraint.check("\f") == ToolCallConstraintVerdict::Rejected,
              "r9 G4: a form-feed suffix after a call is rejected");
    }
    // G5: a directly consecutive second call is legal and closes cleanly.
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        constraint.commit(call_a);
        constraint.commit(call_b);
        check(constraint.finished() && !constraint.active(),
              "r9 G5: a directly consecutive second call is allowed");
    }
    // G6: a whitespace-separated second call is legal and closes cleanly.
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        constraint.commit(call_a);
        constraint.commit("\n");
        constraint.commit(call_b);
        check(constraint.finished() && !constraint.active(),
              "r9 G6: a whitespace-separated second call is allowed");
    }
    // G7: a partial second marker is a legal prefix (NeedMore; EOS is illegal).
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        check(constraint.check(call_a + "\n<tool_") == ToolCallConstraintVerdict::NeedMore,
              "r9 G7: a partial second marker needs more bytes");
    }
    // G8: a failed second marker is rejected (illegal suffix content, not a text lock).
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        constraint.commit(call_a);
        check(constraint.check("\n<tool_x>") == ToolCallConstraintVerdict::Rejected,
              "r9 G8: a failed second marker is rejected");
    }
    // G9: a same-token close plus prose candidate is rejected atomically and is never
    // committed (a constrained sampler cannot accept only the legal prefix of a token).
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        constraint.commit(open_a);
        check(constraint.active() &&
                  constraint.check("</tool_call>\nDone") == ToolCallConstraintVerdict::Rejected,
              "r9 G9: a same-token close plus prose is rejected atomically");
    }
    // G10: a same-token close plus next-marker candidate stays legal and reactivates the
    // second region.
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        constraint.commit(open_a);
        check(constraint.check("</tool_call>\n<tool_call>") ==
                  ToolCallConstraintVerdict::Allowed,
              "r9 G10: a same-token close plus next marker is legal");
        constraint.commit("</tool_call>\n<tool_call>");
        check(constraint.active(),
              "r9 G10: the committed candidate reactivates the second region");
    }
    // G11: TemplateCompatible still allows a preamble-prose first call to trigger.
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::TemplateCompatible);
        constraint.commit("Sure, let me check.\n");
        constraint.commit(marker);
        check(constraint.active(), "r9 G11: the preamble-prose first call still triggers");
    }
    // G12: hardened pre-trigger locking is unchanged: the marker never triggers.
    {
        ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        constraint.commit("Sure, let me check.\n");
        constraint.commit(marker);
        check(!constraint.active(),
              "r9 G12: a locked preamble keeps the marker from triggering");
    }
}

// R9-01 §3.13: the explicit EOS legality predicate. EOS/EOT is an admissible candidate
// only when no open structure needs more bytes: a clean preamble or a clean post-call
// state, a locked pre-trigger gate — but never inside an open region or a pending
// marker candidate. Runs native × compatibility.
void test_r9_constraint_termination_predicate() {
    using Mode   = ninfer::ToolCallSyntaxMode;
    using Intent = ninfer::ToolCallIntentPolicy;
    const char* call =
        "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>";

    for (const Mode syntax : {Mode::QwenWrappedNative, Mode::Compatibility}) {
        // Fresh: a clean preamble may end.
        {
            ToolCallGrammarConstraint constraint(64, syntax);
            check(constraint.can_terminate(), "r9 term: a fresh preamble may terminate");
        }
        // Pending marker candidate, then an open region: both forbid termination.
        {
            ToolCallGrammarConstraint constraint(64, syntax);
            constraint.commit("<tool_");
            check(!constraint.can_terminate(), "r9 term: a pending marker forbids termination");
            constraint.commit("call><function=wea");
            check(!constraint.can_terminate(), "r9 term: an open region forbids termination");
        }
        // Clean post-call: a closed sequence (with whitespace) may end; a post-call
        // partial marker forbids termination.
        {
            ToolCallGrammarConstraint constraint(64, syntax);
            constraint.commit(call);
            check(constraint.can_terminate(), "r9 term: a closed call may terminate");
            constraint.commit("\n");
            check(constraint.can_terminate(), "r9 term: a whitespace-closed sequence may terminate");
            constraint.commit("<tool_");
            check(!constraint.can_terminate(), "r9 term: a post-call partial marker forbids termination");
        }
        // SOC: a locked preamble may end (the gate decision is final, the stream just
        // carries ordinary content).
        {
            ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
            constraint.commit("preamble");
            check(constraint.can_terminate(), "r9 term: a locked preamble may terminate");
        }
    }
}


// R10-02 (Round 10 §10): cross-equivalence. The production pre-latch parser and the
// grammar-constraint pre-trigger scan share the entry classifier (R10-I1); the latch /
// InRegion-entry decision must agree on the same byte index for every corpus line and
// every syntax/intent quadrant.
void test_r10_constraint_parser_entry_cross_equivalence() {
    using Mode = ninfer::ToolCallSyntaxMode;
    using Intent = ninfer::ToolCallIntentPolicy;
    const std::string call =
        "<tool_call>\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n</tool_call>";
    const std::string bare =
        "<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>";
    const auto indent_lines = [](const std::string& text, const std::string& prefix) {
        std::string out;
        std::size_t start = 0;
        while (true) {
            const std::size_t end = text.find('\n', start);
            const std::size_t len = (end == std::string::npos) ? text.size() - start : end - start;
            if (len > 0) { out += prefix; out.append(text, start, len); }
            if (end == std::string::npos) { break; }
            out += '\n';
            start = end + 1;
        }
        return out;
    };
    // A genuine call; B/C indented but still eligible (columns 1 and 3); D/E indented
    // literal (column >= 4); F/G fenced (closed/unclosed); H prose + call; I a failed
    // marker candidate + call; J a whitespace-only indented line + call; K a partial
    // marker; L the compatibility bare entry; M the same bare entry indented.
    const std::vector<std::string> corpus = {
        call,
        " " + call,
        "   " + call,
        indent_lines(call, "    "),
        indent_lines(call, "\t"),
        "```xml\n" + call + "\n```",
        "```xml\n" + call,
        "prose " + call,
        "abc <foo bar> " + call,
        "    \n" + call,
        "<tool_",
        bare,
        indent_lines(bare, "    "),
    };
    const auto parser_latch = [](const std::string& text, Mode syntax, Intent intent) {
        ninfer::models::qwen3_5::frontend::ToolCallParsePolicy policy;
        policy.max_name_length = 64;
        policy.syntax          = syntax;
        policy.intent          = intent;
        ninfer::models::qwen3_5::frontend::ToolCallStreamParser machine(policy);
        int latch = -1;
        for (std::size_t i = 0; i < text.size(); ++i) {
            machine.feed(text.substr(i, 1));
            if (latch < 0 && machine.latched()) { latch = static_cast<int>(i); }
        }
        return latch;
    };
    const auto constraint_latch = [](const std::string& text, Mode syntax, Intent intent) {
        ToolCallGrammarConstraint constraint(64, syntax, intent);
        int latch = -1;
        for (std::size_t i = 0; i < text.size(); ++i) {
            const std::string_view byte = text.substr(i, 1);
            const ToolCallConstraintVerdict verdict = constraint.check(byte);
            if (verdict != ToolCallConstraintVerdict::Rejected) { constraint.commit(byte); }
            if (latch < 0 && constraint.active()) { latch = static_cast<int>(i); }
        }
        return latch;
    };
    for (const Mode syntax : {Mode::QwenWrappedNative, Mode::Compatibility}) {
        for (const Intent intent :
             {Intent::TemplateCompatible, Intent::RequireToolAtContentStart}) {
            for (const std::string& text : corpus) {
                check(parser_latch(text, syntax, intent) == constraint_latch(text, syntax, intent),
                      "r10 cross: parser and constraint entry bytes diverged");
            }
        }
    }
    // Absolute pins: the genuine call latches in every quadrant; the 4-space-indented
    // copy never latches in any quadrant (both consumers agree on "no entry").
    for (const Mode syntax : {Mode::QwenWrappedNative, Mode::Compatibility}) {
        for (const Intent intent :
             {Intent::TemplateCompatible, Intent::RequireToolAtContentStart}) {
            check(parser_latch(call, syntax, intent) != -1,
                  "r10 cross: the genuine call never latched");
            check(constraint_latch(call, syntax, intent) != -1,
                  "r10 cross: the genuine call never triggered the constraint");
            check(parser_latch(indent_lines(call, "    "), syntax, intent) == -1 &&
                      constraint_latch(indent_lines(call, "    "), syntax, intent) == -1,
                  "r10 cross: the indented copy latched");
        }
    }
}

// R10-01 at the constraint level: an indented complete wrapper is inert (never activates),
// and under RequireToolAtContentStart its visible bytes lock the gate so a later column-0
// call stays text; under TemplateCompatible the later call still triggers.
void test_r10_indented_entry_suppression() {
    using Mode = ninfer::ToolCallSyntaxMode;
    using Intent = ninfer::ToolCallIntentPolicy;
    const std::string indented_wrapper = "    <tool_call>";
    const std::string call =
        "<tool_call>\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n</tool_call>";
    for (const Mode syntax : {Mode::QwenWrappedNative, Mode::Compatibility}) {
        {
            ToolCallGrammarConstraint constraint(64, syntax, Intent::TemplateCompatible);
            constraint.commit(std::string(indented_wrapper));
            check(!constraint.active(), "r10 tc: the indented wrapper never activates");
            constraint.commit("\n");
            constraint.commit("<tool_call>");
            check(constraint.active(),
                  "r10 tc: the genuine call after an indented wrapper activates");
        }
        {
            ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
            constraint.commit(std::string(indented_wrapper));
            check(!constraint.active(), "r10 soc: the indented wrapper never activates");
            check(constraint.can_terminate(),
                  "r10 soc: the locked preamble has no open structure and may terminate");
            constraint.commit("\n");
            constraint.commit("<tool_call>");
            check(!constraint.active(),
                  "r10 soc: the locked gate keeps the later column-0 call text");
        }
    }
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
    test_r7_intent_content_start_gate();
    test_r8_constraint_post_call_gate();
    test_r8_post_call_chunk_checkpoint_matrix();
    test_r9_constraint_post_call_suffix();
    test_r9_constraint_termination_predicate();
    test_r10_constraint_parser_entry_cross_equivalence();
    test_r10_indented_entry_suppression();
    if (failures == 0) { std::puts("tool_call_grammar_state tests: all passed"); }
    return failures == 0 ? 0 : 1;
}
