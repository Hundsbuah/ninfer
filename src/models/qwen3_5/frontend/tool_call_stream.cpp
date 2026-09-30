#include <algorithm>

#include "models/qwen3_5/frontend/tool_call_stream.h"

namespace ninfer::models::qwen3_5::frontend {
namespace {

constexpr std::string_view kImStartMarker = "|im_start|>";

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view literal) noexcept {
    return pos <= text.size() && text.substr(pos, literal.size()) == literal;
}
// True when the bytes from pos form a non-empty proper prefix of `literal` (an in-progress
// write of that literal, cut off at the slice end).
bool is_strict_prefix_of(std::string_view literal, std::string_view text, std::size_t pos) noexcept {
    const std::size_t len = text.size() - pos;
    return len > 0 && len < literal.size() && literal.compare(0, len, text.substr(pos, len)) == 0;
}


std::size_t skip_ws(std::string_view text, std::size_t pos) noexcept {
    while (pos < text.size() && is_tool_format_whitespace(text[pos])) { ++pos; }
    return pos;
}

struct RegionState {
    enum class Mode : std::uint8_t {
        ExpectFunction,    // after a <tool_call> / <function_calls> literal
        FunctionHeader,    // scanning the function/invoke opener
        FunctionBody,      // between parameters; expecting a parameter or the function close
        ParameterHeader,   // scanning the parameter opener
        ParameterValue,    // inside the parameter value (opaque byte range)
        ExpectWrapperClose,// after a function close inside a <tool_call> wrapper
        Top,               // top level of the region: next call, wrapper close, end, trailing
    };

    Mode mode = Mode::ExpectFunction;
    std::size_t pos = 0;
    ToolTagKind fn_family  = ToolTagKind::Function;
    ToolTagKind param_family = ToolTagKind::Parameter;
    // The outer wrapper state (F5): None at the top level, one kind while inside a wrapper.
    // A second wrapper open before the first closed is a definitive structural break; the
    // wrapper's close returns the state to None.
    ToolWrapperKind wrapper = ToolWrapperKind::None;
    // A call completed inside a <function_calls> wrapper: the wire format requires the
    // wrapper to contain at least one call (an empty wrapper fails the whole region).
    bool function_calls_had_call = false;
    ParsedFunctionCall current;
    ParsedToolRegion region;
    // A definitive invalid that must not be retained even in tolerant mode (an empty
    // function_calls wrapper fails the whole region, as the wire format requires the wrapper
    // to contain at least one call).
    bool no_retain = false;
    std::string param_name; // the current parameter's name (set in ParameterHeader)
    // Parameter value scan: the value is an opaque byte range (F1/I1); only the matching
    // outer close is a candidate boundary, decided by its continuation (no nesting depth).
    std::size_t value_begin = 0;
};


// What may legally follow a consumed function/invoke close, per the surrounding wrapper
// state (F4). The answer decides whether the close was a structural boundary or quoted
// payload: inside <tool_call> only the wrapper close (or its in-progress prefix at the
// generation end) may follow; at the top level and inside <function_calls> the next call,
// the wrapper close, or the end of generation continues the region. Anything else proves
// the close was part of the payload.
enum class FunctionCloseContinuation : std::uint8_t {
    Legal,    // the close is a structural boundary
    Payload,  // the following bytes prove the close is quoted payload
};

FunctionCloseContinuation classify_function_close_continuation(std::string_view text,
                                                               std::size_t after,
                                                               ToolWrapperKind wrapper) noexcept {
    const std::size_t at = skip_ws(text, after);
    if (at >= text.size()) { return FunctionCloseContinuation::Legal; }
    if (wrapper == ToolWrapperKind::ToolCall) {
        if (starts_with_at(text, at, tool_close_literal(ToolTagKind::ToolCall))) {
            return FunctionCloseContinuation::Legal;
        }
        return is_strict_prefix_of(tool_close_literal(ToolTagKind::ToolCall), text, at)
                   ? FunctionCloseContinuation::Legal
                   : FunctionCloseContinuation::Payload;
    }
    if (wrapper == ToolWrapperKind::FunctionCalls &&
        starts_with_at(text, at, tool_close_literal(ToolTagKind::FunctionCalls))) {
        return FunctionCloseContinuation::Legal;
    }
    // R2-I4 (CR4): the continuation after a function close at top level (or inside
    // <function_calls>) is exactly the top-level entry the region's Top state accepts —
    // classified by the shared classifier, so a complete next entry and its partial
    // prefixes agree on the previous call's boundary. A wrapper open inside an open
    // wrapper proves the close was payload (a nesting break, F5); only at top level
    // (wrapper == None) is a complete wrapper open a legal next entry.
    const TopLevelEntryInfo entry = classify_top_level_entry(text, at);
    if (entry.kind == TopLevelEntry::NeedMore) {
        // A next entry in progress at the slice end (F4/I3).
        return FunctionCloseContinuation::Legal;
    }
    if (entry.kind == TopLevelEntry::Entry) {
        if (is_tool_function_kind(entry.tag)) {
            // The next call.
            return FunctionCloseContinuation::Legal;
        }
        return wrapper == ToolWrapperKind::None ? FunctionCloseContinuation::Legal
                                                : FunctionCloseContinuation::Payload;
    }
    if (is_strict_prefix_of(tool_close_literal(ToolTagKind::FunctionCalls), text, at)) {
        // An in-progress </function_calls> close at the slice end (F4/I3).
        return FunctionCloseContinuation::Legal;
    }
    return FunctionCloseContinuation::Payload;
}


// The tri-state continuation classification of a candidate outer parameter close (F1/F3/F4).
// A candidate closer is a structural boundary when the following bytes are the grammar's
// next structural token; anything else keeps the closer inside the payload. End of input
// right after the closer keeps the boundary provisional: the parameter may be provisionally
// closed, but the function remains open and therefore non-executable.
enum class CloseContinuation : std::uint8_t {
    Invalid,   // the following bytes prove the closer is payload
    Complete,  // the following bytes prove a structural boundary
    NeedMore,  // the input ends right after the closer: provisional boundary
};

CloseContinuation classify_close_continuation(std::string_view text, std::size_t after,
                                              ToolTagKind fn_family,
                                              ToolWrapperKind wrapper) noexcept {
    const std::size_t at = skip_ws(text, after);
    if (at >= text.size()) { return CloseContinuation::NeedMore; }
    ToolOpenTag opener = {};
    const ToolHeaderStatus opener_status = parse_tool_parameter_open(text.substr(at), opener);
    if (opener_status == ToolHeaderStatus::Complete) { return CloseContinuation::Complete; }
    if (opener_status == ToolHeaderStatus::NeedMore) {
        // A cut-off parameter opener at the generation end (F4/I3): the preceding close stays
        // closed; the incomplete next token is a truncation, not payload.
        return CloseContinuation::Complete;
    }
    if (starts_with_at(text, at, tool_close_literal(fn_family))) {
        return classify_function_close_continuation(
                   text, at + tool_close_literal(fn_family).size(), wrapper) ==
                   FunctionCloseContinuation::Legal
                   ? CloseContinuation::Complete
                   : CloseContinuation::Invalid;
    }
    if (is_strict_prefix_of(tool_close_literal(fn_family), text, at)) {
        // A cut-off function close at the generation end (F4/I3).
        return CloseContinuation::Complete;
    }
    return CloseContinuation::Invalid;
}

// Deterministic parse of a complete tool-region slice. The slice always starts at a marker
// (the latched marker or a <tool_call> wrapper found by the retry loop). Every transition is
// byte-driven, so the outcome is independent of how the bytes were chunked upstream.
ToolCallParseProgress parse_region(std::string_view text, const ToolCallParsePolicy& policy) {
    RegionState s;
    ToolCallParseProgress out;

    auto invalid = [&](ToolCallParseFailure f, std::size_t break_at) {
        // A definitive structural break (or trailing content after complete calls). The
        // progress records the complete calls and the open call; the recovery policy decides
        // retention, never the parse. `break_at` is the first byte that proved the break: a
        // recovery retry may re-read at or after it, but never inside the failed structure.
        out.termination                   = ToolCallRegionTermination::Definitive;
        out.failure                       = f;
        out.break_offset                  = break_at;
        out.wrapper_at_break              = s.wrapper;
        out.open_call                     = std::move(s.current);
        out.unrecoverable_break           = s.no_retain;
        out.calls                         = std::move(s.region.calls);
        out.duplicate_parameters_repaired = s.region.duplicate_parameters_repaired;
    };

    auto truncated = [&](bool value_open, std::size_t end_at) {
        // The input ended mid-region. Strict mode maps this to a structural failure; tolerant
        // recovery may commit complete calls. An open call is never committed (F2/I2): its
        // function close was not consumed.
        out.termination                   = ToolCallRegionTermination::EndOfInput;
        out.break_offset                  = end_at;
        out.wrapper_at_break              = s.wrapper;
        out.open_call                     = std::move(s.current);
        out.open_value_open               = value_open;
        out.calls                         = std::move(s.region.calls);
        out.duplicate_parameters_repaired = s.region.duplicate_parameters_repaired;
    };

    auto complete = [&]() {
        out.termination                   = ToolCallRegionTermination::Complete;
        out.calls                         = std::move(s.region.calls);
        out.duplicate_parameters_repaired = s.region.duplicate_parameters_repaired;
    };

    auto complete_call = [&]() {
        s.region.calls.push_back(std::move(s.current));
        s.current = {};
        if (s.wrapper == ToolWrapperKind::FunctionCalls) { s.function_calls_had_call = true; }
    };

    for (;;) {
        switch (s.mode) {
        case RegionState::Mode::ExpectFunction:
        case RegionState::Mode::Top: {
            const std::size_t i = skip_ws(text, s.pos);
            if (i == text.size()) {
                // R2-I2 (CR2): Complete implies wrapper balance. A wrapper open at the input
                // end is an objective truncation (a compatibility omission, not a clean
                // completion): the recovery policy decides whether the function-closed calls
                // are retained. ExpectFunction (a wrapper literal with no function yet) also
                // stays a truncation.
                if (s.mode == RegionState::Mode::Top && s.wrapper == ToolWrapperKind::None) {
                    complete();
                    return out;
                }
                // A wrapper literal was consumed but no function followed.
                truncated(false, i);
                return out;
            }
            if (s.wrapper == ToolWrapperKind::FunctionCalls &&
                starts_with_at(text, i, tool_close_literal(ToolTagKind::FunctionCalls))) {
                s.pos = i + tool_close_literal(ToolTagKind::FunctionCalls).size();
                if (!s.function_calls_had_call) {
                    // An empty <function_calls> wrapper is a wire violation that fails the
                    // whole region (no recovery commits anything from it).
                    s.no_retain = true;
                    invalid(ToolCallParseFailure::MalformedStructure, i);
                    return out;
                }
                // R2-I3 (CR3): consuming the wrapper close does not consume the rest of the
                // output: the wrapper closes and the state returns to Top, which validates the
                // remainder (EOF, a next region, a partial marker, or trailing content).
                s.wrapper = ToolWrapperKind::None;
                continue;
            }
            if (s.wrapper == ToolWrapperKind::FunctionCalls &&
                is_strict_prefix_of(tool_close_literal(ToolTagKind::FunctionCalls), text, i)) {
                // An in-progress </function_calls> close at the slice end: the generated
                // prefix may still complete (F4: an objective truncation fact).
                truncated(false, i);
                return out;
            }
            // R2-I4 (CR4): top-level entries are classified by the shared classifier, the
            // same authority the function-close continuation lookahead consumes, so complete
            // and partial next entries agree on the previous call's boundary.
            const TopLevelEntryInfo entry = classify_top_level_entry(text, i);
            if (entry.kind == TopLevelEntry::Entry &&
                (entry.tag == ToolTagKind::ToolCall || entry.tag == ToolTagKind::FunctionCalls)) {
                if (s.wrapper != ToolWrapperKind::None) {
                    // A second wrapper open before the first closed: unbalanced nesting is a
                    // definitive structural break (F5/I4). The nested wrapper never parses and
                    // its calls are never re-read as recovery entries; the recovery policy
                    // still decides about complete calls the outer region already committed
                    // (F2 retention).
                    invalid(ToolCallParseFailure::MalformedStructure, i);
                    return out;
                }
                s.pos     = i + entry.opener.consumed;
                s.wrapper = (entry.tag == ToolTagKind::ToolCall) ? ToolWrapperKind::ToolCall
                                                                 : ToolWrapperKind::FunctionCalls;
                s.mode    = RegionState::Mode::ExpectFunction;
                continue;
            }
            if (entry.kind == TopLevelEntry::Entry) {
                // A complete function/invoke opener.
                s.pos  = i;
                s.mode = RegionState::Mode::FunctionHeader;
                continue;
            }
            if (entry.kind == TopLevelEntry::NeedMore) {
                // An in-progress top-level entry (wrapper literal or opener) at the slice end
                // (F4).
                truncated(false, i);
                return out;
            }
            // NoEntry: a broken opener or a non-marker byte. Under tolerant mode a broken
            // opener (a dropped '<' or keyword) also enters the header state, where the
            // recovery rules run; strict mode invalidates below with the same diagnostics the
            // old batch parser produced.
            if (policy.tolerant) {
                s.pos  = i;
                s.mode = RegionState::Mode::FunctionHeader;
                continue;
            }
            // A byte that is not a marker: trailing text after complete calls, or a broken
            // region start. A definitive structural outcome: strict mode rejects with the
            // reason.
            if (s.region.calls.empty()) { invalid(ToolCallParseFailure::MalformedStructure, i); }
            else { invalid(ToolCallParseFailure::TrailingContent, i); }
            return out;
        }
        case RegionState::Mode::FunctionHeader: {
            const std::size_t i = s.pos;
            ToolOpenTag opener = {};
            ToolHeaderStatus st = parse_tool_function_open(text.substr(i), opener);
            bool from_boundary  = false;
            std::size_t base    = i + 1 + (opener.kind == ToolTagKind::Invoke ? 6 : 8);
            if (st == ToolHeaderStatus::NoMatch && policy.tolerant) {
                // Recover a malformed opener: a dropped or doubled leading '<', a leaked ChatML
                // turn marker, or a dropped keyword. The repair only moves where the header
                // starts; the canonical header grammar still decides what the header accepts.
                std::size_t scan = i;
                while (scan < text.size() && text[scan] == '<') { ++scan; }
                if (starts_with_at(text, scan, kImStartMarker)) { scan += kImStartMarker.size(); }
                if (starts_with_at(text, scan, "function")) {
                    opener.kind = ToolTagKind::Function;
                } else if (starts_with_at(text, scan, "invoke")) {
                    opener.kind = ToolTagKind::Invoke;
                } else {
                    invalid(ToolCallParseFailure::MalformedStructure, i);
                    return out;
                }
                scan += (opener.kind == ToolTagKind::Function ? 8 : 6);
                if (scan < text.size() && (text[scan] == '=' || is_tool_format_whitespace(text[scan]))) {
                    base    = scan;
                    from_boundary = true;
                    st      = parse_tool_header_after_keyword(text.substr(scan), opener.kind, opener);
                } else {
                    invalid(ToolCallParseFailure::MalformedStructure, i);
                    return out;
                }
            }
            std::string_view name;
            bool ws_boundary = false;
            if (st == ToolHeaderStatus::Complete) {
                name = opener.name;
                s.pos = from_boundary ? base + opener.consumed : i + opener.consumed;
            } else if (policy.tolerant) {
                // The model sometimes drops the '>' after the function name (for example a name
                // followed directly by a newline and a parameter tag). Recover by scanning the
                // identifier run and accepting it when format whitespace separates it from the
                // next '<' or the end of region.
                if (base <= text.size()) {
                    const std::string_view body = text.substr(base);
                    std::size_t scan            = 0;
                    while (scan < body.size() && body[scan] == '=') { ++scan; }
                    const std::size_t ident_begin = scan;
                    while (scan < body.size() && scan - ident_begin < policy.max_name_length &&
                           is_tool_name_char(body[scan])) {
                        ++scan;
                    }
                    if (scan > ident_begin && scan < body.size() &&
                        is_tool_format_whitespace(body[scan])) {
                        std::size_t after = scan;
                        while (after < body.size() && is_tool_format_whitespace(body[after])) {
                            ++after;
                        }
                        if (after >= body.size() || body[after] == '<') {
                            name        = body.substr(ident_begin, scan - ident_begin);
                            ws_boundary = true;
                            s.pos       = base + scan;
                        }
                    }
                }
            }
            if (!ws_boundary && st != ToolHeaderStatus::Complete) {
                // NeedMore at the slice end is a cut-off header (more bytes could complete it);
                // Invalid or a failed recovery is a definitive break.
                if (st == ToolHeaderStatus::NeedMore) {
                    truncated(false, i);
                } else {
                    invalid(ToolCallParseFailure::MalformedStructure, i);
                }
                return out;
            }
            if (!is_valid_tool_name(name, policy.max_name_length)) {
                invalid(ToolCallParseFailure::InvalidToolName, i);
                return out;
            }
            // R2-I5 (CR5): the declared-tool identity policy is independent of syntax
            // tolerance: tolerant mode repairs syntax damage, it does not let the model
            // invent function identities. The identity check applies in strict and tolerant
            // alike whenever the contract enforces declared names.
            if (policy.enforce_declared_names &&
                (policy.declared_check == nullptr ||
                 !policy.declared_check(policy.contract, name))) {
                invalid(ToolCallParseFailure::UndeclaredTool, i);
                return out;
            }
            s.current.name  = std::string(name);
            s.fn_family     = opener.kind;
            s.mode          = RegionState::Mode::FunctionBody;
            continue;
        }
        case RegionState::Mode::FunctionBody: {
            const std::size_t i = skip_ws(text, s.pos);
            if (i == text.size()) {
                // The region ends after the last parameter (or the name): a missing function
                // close is a truncation, not a malformed structure. The function stays open
                // and therefore non-executable (F2/I2); the recovery policy commits only the
                // complete calls.
                truncated(false, i);
                return out;
            }
            if (starts_with_at(text, i, tool_close_literal(s.fn_family))) {
                // The matching function close is consumed here: the call is structurally
                // complete from this byte on (F2/I2). The wrapper close, when present, only
                // closes the wrapper; it completes no call.
                s.pos = i + tool_close_literal(s.fn_family).size();
                complete_call();
                // A <tool_call> wrapper must close next (ExpectWrapperClose). A
                // <function_calls> wrapper holds a sequence of calls (F5): the next token
                // may be another call, the wrapper close, or the region end — Top handles
                // all three and rejects a nested wrapper open.
                s.mode = (s.wrapper == ToolWrapperKind::None ||
                          s.wrapper == ToolWrapperKind::FunctionCalls)
                            ? RegionState::Mode::Top
                            : RegionState::Mode::ExpectWrapperClose;
                continue;
            }
            ToolOpenTag opener = {};
            const ToolHeaderStatus param_status = parse_tool_parameter_open(text.substr(i), opener);
            if (param_status == ToolHeaderStatus::Complete) {
                s.pos  = i;
                s.mode = RegionState::Mode::ParameterHeader;
                continue;
            }
            if (param_status == ToolHeaderStatus::NeedMore ||
                is_strict_prefix_of(tool_close_literal(s.fn_family), text, i)) {
                // An in-progress parameter opener or function close at the slice end (F4).
                truncated(false, i);
                return out;
            }
            invalid(ToolCallParseFailure::MalformedStructure, i);
            return out;
        }
        case RegionState::Mode::ParameterHeader: {
            const std::size_t i = s.pos;
            ToolOpenTag opener = {};
            const ToolHeaderStatus st = parse_tool_parameter_open(text.substr(i), opener);
            if (st != ToolHeaderStatus::Complete || opener.name.empty()) {
                // A broken or cut-off parameter header fails the call even in tolerant mode:
                // there is no value to retain, and the current call is dropped with it.
                invalid(ToolCallParseFailure::MalformedStructure, i);
                return out;
            }
            s.param_family = opener.kind;
            s.param_name   = std::string(opener.name);
            s.value_begin  = i + opener.consumed;
            s.pos          = s.value_begin;
            s.mode         = RegionState::Mode::ParameterValue;
            continue;
        }
        case RegionState::Mode::ParameterValue: {
            // A parameter value is an opaque byte range (I1/F1): literal openers inside the
            // payload are ordinary bytes, and only the matching outer close is a candidate
            // boundary. No nesting depth is tracked: a candidate is a boundary exactly when
            // its continuation is the grammar's next structural token.
            const std::string_view required_close = tool_close_literal(s.param_family);
            std::size_t scan = s.value_begin;
            for (;;) {
                const std::size_t candidate = text.find(required_close, scan);
                if (candidate == std::string_view::npos) {
                    // The region ends inside the value: a cut parameter value is never
                    // committed — its bytes may still grow into a different value. The
                    // progress records the open value; the recovery policy decides (it never
                    // commits a call whose value is still open, whatever the finish reason).
                    truncated(true, s.value_begin);
                    return out;
                }
                const CloseContinuation continuation = classify_close_continuation(
                    text, candidate + required_close.size(), s.fn_family, s.wrapper);
                if (continuation == CloseContinuation::Invalid) {
                    // A quoted closer (echoed markup, command text): keep scanning.
                    scan = candidate + required_close.size();
                    continue;
                }
                // Complete or NeedMore: the parameter boundary is retained (I3). A NeedMore
                // boundary leaves the function open, so the call stays non-executable (F3).
                auto& current = s.current;
                const auto existing = std::find_if(current.parameters.begin(),
                                                   current.parameters.end(),
                                                   [&](const ParsedParameter& candidate) {
                                                       return candidate.name == s.param_name;
                                                   });
                if (existing != current.parameters.end()) {
                    // Last occurrence wins, as it would in JSON object syntax, rather than
                    // discarding an otherwise well-formed call.
                    existing->value = std::string(text.substr(s.value_begin, candidate - s.value_begin));
                    ++s.region.duplicate_parameters_repaired;
                } else {
                    current.parameters.push_back(
                        ParsedParameter{.name = s.param_name,
                                        .value = std::string(text.substr(s.value_begin, candidate - s.value_begin))});
                }
                s.pos  = candidate + required_close.size();
                s.mode = RegionState::Mode::FunctionBody;
                break;
            }
            // The parameter closed: re-dispatch the region state machine at the new mode
            // (the inner scan loop is done; the outer loop re-enters the switch).
            continue;
        }
        case RegionState::Mode::ExpectWrapperClose: {
            // Only the <tool_call> wrapper reaches this state: after a function close
            // inside <function_calls> the machine returns to Top, where the call sequence
            // may continue (F5).
            const std::string_view close_literal = tool_close_literal(ToolTagKind::ToolCall);
            const std::size_t i = skip_ws(text, s.pos);
            if (i == text.size()) {
                // A function-closed call may be followed by the end of its budget before the
                // closing wrapper tag: a truncated tail. The complete call was committed at
                // the function close; tolerant recovery may retain it (F2).
                truncated(false, i);
                return out;
            }
            if (starts_with_at(text, i, close_literal)) {
                // The wrapper close closes the wrapper only: the call was already committed
                // at its function close (F2). A second complete_call() here is exactly the
                // bug the recovery integrity fix removes.
                s.pos     = i + close_literal.size();
                s.wrapper = ToolWrapperKind::None;
                s.mode    = RegionState::Mode::Top;
                continue;
            }
            if (is_strict_prefix_of(close_literal, text, i)) {
                // An in-progress wrapper close at the slice end: a truncated tail (F4).
                truncated(false, i);
                return out;
            }
            invalid(ToolCallParseFailure::MalformedStructure, i);
            return out;
        }
    }
    }
}

} // namespace
ToolCallParseProgress parse_tool_call_region(std::string_view text, const ToolCallParsePolicy& policy) {
    return parse_region(text, policy);
}

ToolCallStreamParser::ToolCallStreamParser(ToolCallParsePolicy policy) : policy_(policy) {}
ToolCallStreamParser::FenceTracker::Verdict
ToolCallStreamParser::FenceTracker::consume(char byte) noexcept {
    if (byte == '\n') {
        if (in_fence_ && close_pending_ && !opener_line_) {
            // A close line (run of the opener character of at least the opener length,
            // followed only by whitespace) ends the fence at its newline.
            in_fence_   = false;
            fence_char_ = '\0';
            fence_len_  = 0;
        }
        close_pending_ = false;
        opener_line_   = false;
        at_line_start_ = true;
        run_char_      = '\0';
        run_len_       = 0;
        indent_        = 0;
        return in_fence_ ? Verdict::Content : Verdict::Pass;
    }
    if (in_fence_) {
        // Every byte inside a fence is ordinary content until a valid close line ends the
        // fence. Track the close candidate at line start: up to 3 spaces, then a run of the
        // opener character reaching the opener length; any other byte cancels it (the line
        // is content, the fence stays open). The opener line itself never closes the fence.
        if (at_line_start_) {
            if (byte == ' ') {
                if (indent_ < 3) { ++indent_; }
                else { at_line_start_ = false; }
            } else if (byte == fence_char_) {
                ++run_len_;
                if (opener_line_ && at_line_start_ && indent_ <= 3) {
                    // The opener line's run is still growing: the fence length is the full
                    // run (CommonMark), not the length at the opening threshold.
                    fence_len_ = run_len_;
                } else if (!opener_line_ && indent_ <= 3 && run_len_ >= fence_len_) {
                    close_pending_ = true;
                }
            } else {
                at_line_start_ = false;
                close_pending_ = false;
            }
        } else if (!is_tool_format_whitespace(byte)) {
            // A non-whitespace byte after the close run: the line is not a close; further
            // fence characters on the line are not a new run.
            at_line_start_  = false;
            close_pending_ = false;
        }
        return Verdict::Content;
    }
    if (!at_line_start_) { return Verdict::Pass; }
    if (byte == ' ') {
        // Whitespace stays with the marker machine (a pre-latch whitespace run may still
        // belong to the region's leading part); only the fence run is structural here.
        if (indent_ < 3) { ++indent_; }
        else { at_line_start_ = false; }
        return Verdict::Pass;
    }
    if (byte == '`' || byte == '~') {
        if (indent_ <= 3 && (run_char_ == '\0' || run_char_ == byte)) {
            run_char_ = byte;
            ++run_len_;
            if (run_len_ >= 3) {
                // The run opened the fence at its third character; the rest of the line is
                // the info string.
                in_fence_    = true;
                fence_char_  = byte;
                fence_len_   = run_len_;
                opener_line_ = true;
            }
            return Verdict::Content;
        }
        run_char_      = '\0';
        at_line_start_ = false;
        return Verdict::Pass;
    }
    // A non-whitespace, non-fence byte at line start: the line is ordinary content.
    at_line_start_ = false;
    return Verdict::Pass;
}


void ToolCallStreamParser::publish(std::string_view bytes, std::string& visible) {
    content_.append(bytes);
    visible.append(bytes);
}

void ToolCallStreamParser::latch(std::string_view marker) {
    latched_    = true;
    marker_seen_ = true;
    // The whitespace held before the marker is kept at the start of the region: the accepted
    // path rtrims it away from the content, the verbatim path keeps it, matching one-shot.
    region_ = std::move(pending_ws_);
    pending_ws_.clear();
    region_.append(marker);
    marker_prefix_.clear();
}

std::string ToolCallStreamParser::feed(std::string_view chunk) {
    std::string visible;
    if (chunk.empty()) { return visible; }
    if (latched_) {
        region_.append(chunk);
        return visible;
    }
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        const char byte = chunk[i];
        if (fence_.consume(byte) == FenceTracker::Verdict::Content) {
            // R2-I6 (CR6): a fence byte is ordinary content and cannot continue a top-level
            // marker (a marker is single-line and the newline that bounds a fence line has
            // already broken any held candidate). Publish the held bytes as content.
            publish(pending_ws_, visible);
            pending_ws_.clear();
            publish(marker_prefix_, visible);
            marker_prefix_.clear();
            publish(std::string_view(&byte, 1), visible);
            continue;
        }
        if (!marker_prefix_.empty()) {
            marker_prefix_.push_back(byte);
            ToolOpenTag marker = {};
            const ToolMarkerStatus state = classify_tool_marker_prefix(marker_prefix_, marker);
            if (state == ToolMarkerStatus::Complete) {
                latch(marker_prefix_);
                region_.append(chunk.substr(i + 1));
                return visible;
            }
            if (state == ToolMarkerStatus::NotMarker) {
                // F8: a breaking '<' starts a fresh candidate; the failed bytes before it
                // are published as ordinary content, the '<' itself is retained.
                const std::size_t retained = failed_marker_candidate_retained(marker_prefix_);
                publish(pending_ws_, visible);
                publish(marker_prefix_.substr(0, marker_prefix_.size() - retained), visible);
                pending_ws_.clear();
                marker_prefix_.resize(retained);
            }
            continue;
        }
        if (byte == '<') {
            marker_prefix_.push_back(byte);
        } else if (is_tool_format_whitespace(byte)) {
            pending_ws_.push_back(byte);
        } else {
            publish(pending_ws_, visible);
            pending_ws_.clear();
            publish(std::string_view(&byte, 1), visible);
        }
    }
    return visible;
}

ToolCallStreamResult ToolCallStreamParser::finish(FinishReason finish_reason) const {
    ToolCallStreamResult result;
    result.marker_seen = marker_seen_;
    if (!latched_) {
        // No marker anywhere: the whole stream is ordinary content.
        result.status  = ToolCallStreamStatus::Invalid;
        result.failure = ToolCallParseFailure::MalformedStructure;
        result.tail    = content_ + pending_ws_ + marker_prefix_;
        return result;
    }
    // Recovery retry (R2-I1/CR1, F7): a rejected region may be re-read at a later top-level
    // marker — but only from a proven top-level scope. A break that leaves a wrapper still
    // open is never eligible for retry: the wire bytes cannot prove exit from the failed
    // wrapper (the payload syntax has no escaping, and the break left the structure
    // unbalanced), so the failed wrapper owns the remaining ambiguous bytes and the whole
    // region falls back to content. Non-execution over recovery (R2-I7).
    std::size_t base = 0;
    ToolCallParseFailure first_failure = ToolCallParseFailure::MalformedStructure;
    const ToolCallRecoveryPolicy recovery_policy{policy_.tolerant, finish_reason};
    for (;;) {
        ToolCallParseProgress progress = parse_region(region_.substr(base), policy_);
        const ToolCallRecoveryResult recovery =
            decide_tool_call_recovery(progress, recovery_policy);
        if (recovery.decision != ToolCallRecoveryDecision::Reject) {
            result.status = ToolCallStreamStatus::Complete;
            result.failure = recovery.diagnostic;
            result.region.calls = std::move(progress.calls);
            result.region.duplicate_parameters_repaired = progress.duplicate_parameters_repaired;
            result.truncated_tail =
                progress.termination != ToolCallRegionTermination::Complete;
            result.tail = region_.substr(0, base);
            return result;
        }
        if (base == 0) { first_failure = recovery.diagnostic; }
        if (progress.wrapper_at_break != ToolWrapperKind::None) {
            // R2-I1: a failed wrapper owns its still-unclosed scope. No marker located inside
            // it (a nested wrapper, a quoted literal, or a truncation tail) may become a
            // recovery entry; the entire latched region stays content.
            result.status  = ToolCallStreamStatus::Invalid;
            result.failure = first_failure;
            result.tail    = region_;
            return result;
        }
        // The retried slice starts with the entry marker (a latched or retry-found complete
        // marker). A definitive break at the marker's own header must not re-read that same
        // marker; the search starts strictly after the failed marker's opener bytes, or at
        // the break byte when the break proved itself deeper in the structure.
        ToolOpenTag entry_marker = {};
        const ToolMarkerStatus entry_status =
            classify_tool_marker_prefix(region_.substr(base), entry_marker);
        const std::size_t entry_end =
            base + (entry_status == ToolMarkerStatus::Complete ? entry_marker.consumed : 1);
        std::size_t search_from =
            progress.termination == ToolCallRegionTermination::Definitive
                ? std::max(base + progress.break_offset, entry_end)
                : base + 1;
        const std::size_t next = find_tool_marker(region_, search_from);
        if (next == std::string_view::npos || next == base) {
            // No later recovery entry. `next == base` guards a header that fails at its own
            // start: re-reading the same marker cannot change the outcome.
            result.status  = ToolCallStreamStatus::Invalid;
            result.failure = first_failure;
            result.tail    = region_;
            return result;
        }
        base = next;
    }
}

} // namespace ninfer::models::qwen3_5::frontend
