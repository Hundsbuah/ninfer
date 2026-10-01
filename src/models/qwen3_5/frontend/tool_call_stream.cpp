#include <algorithm>
#include <unordered_map>
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

// Stage-2 boundary mode of the region machine: Greedy keeps the Stage-1 rule (the first
// candidate closer whose one-token continuation is legal ends the value); Consistent runs
// the R3-01 candidate selection (balanced pass, lazy pass, plausibility, canonical framing).
enum class BoundaryMode : std::uint8_t {
    Greedy,
    Consistent,
};
struct ConsistentCompleter;

// The region machine with a selectable value-boundary policy and an entry machine state.
// `cc` (non-null only for Consistent mode) charges the deterministic step budget, memoizes
// proven-Definitive states, and selects the value boundaries.
ToolCallParseProgress parse_region_at(std::string_view text, const ToolCallParsePolicy& policy,
                                      RegionState s, BoundaryMode mode, ConsistentCompleter* cc);

// Deterministic parse of a complete tool-region slice (Stage 1). The slice always starts at a
// marker (the latched marker or a <tool_call> wrapper found by the retry loop). Every
// transition is byte-driven, so the outcome is independent of how the bytes were chunked
// upstream.
ToolCallParseProgress parse_region(std::string_view text, const ToolCallParsePolicy& policy) {
    return parse_region_at(text, policy, RegionState{}, BoundaryMode::Greedy, nullptr);
}

// Stage 2 — ConsistentCompleter (R3-01): the only mechanism that makes a truncated region
// complete. It runs only when Stage 1 accepted nothing, on every admissible base of the
// Stage-1 retry chain, and parses the canonical grammar (tolerant header repairs disabled in
// the Consistent machine): the glue transitions are the Stage-1 machine itself, and only the
// parameter-value boundary selection differs.
//
//   pass 1 (balanced): the value-family closers not consumed by a nested same-family opener
//                      inside the value (depth counting, same family only);
//   pass 2 (lazy):     every closer of the value's family in order, only when pass 1 produced
//                      no candidate whose path stood.
// A candidate is skipped when its one-token continuation is Invalid, when its continuation
// opens a repeated or undeclared parameter of the current call (plausibility), or when a
// viable candidate of the same value was already contradicted and the candidate is not
// preceded by '\n' (canonical framing: the real value close is serialized as
// '\n</parameter>'). A candidate stands when its path ends EndOfInput (the region result is
// EndOfInput); it is abandoned only when the path ends Definitive. When every candidate is
// contradicted, the value is open to the end (EndOfInput, never Complete).
//
// Bounded by construction (R3-I8/R3-14): a hard recursion depth bound fails closed, a
// deterministic step budget bounds candidate/glue work, and the memo key is an O(1) hash of
// the machine state (position, mode, families, wrapper, call and parameter identity) — never
// a string concatenation.
struct ConsistentCompleter {
    static constexpr std::uint32_t kMaxStage2Depth = 64;

    std::string_view text;
    ToolCallParsePolicy policy;
    std::uint64_t budget;
    std::uint64_t steps = 0;
    std::uint32_t lazy_choices = 0; // pass-2 boundaries used on the standing path
    std::uint32_t depth = 0;
    bool exhausted = false;
    std::unordered_map<std::uint64_t, bool> memo; // state hash -> the path proved Definitive

    // Charge one candidate/glue step; true when the budget is exhausted (fail closed).
    [[nodiscard]] bool charge() noexcept {
        if (exhausted) { return true; }
        if (++steps > budget) { exhausted = true; return true; }
        return false;
    }

    [[nodiscard]] static std::uint64_t state_hash(const RegionState& s) noexcept {
        std::uint64_t h = 0x9e3779b97f4a7c15ULL;
        auto mix = [&h](std::uint64_t v) {
            h ^= v;
            h = (h << 27) | (h >> 37);
            h *= 0x9e3779b97f4a7c15ULL;
        };
        auto mix_str = [&mix](std::string_view v) {
            std::uint64_t f = 0xcbf29ce484222325ULL;
            for (const char c : v) { f ^= static_cast<std::uint8_t>(c); f *= 0x100000001b3ULL; }
            mix(f);
        };
        mix(s.pos);
        mix(static_cast<std::uint64_t>(s.mode));
        mix(static_cast<std::uint64_t>(s.fn_family));
        mix(static_cast<std::uint64_t>(s.param_family));
        mix(static_cast<std::uint64_t>(s.wrapper));
        mix(static_cast<std::uint64_t>(s.function_calls_had_call));
        mix(static_cast<std::uint64_t>(s.no_retain));
        mix(static_cast<std::uint64_t>(s.region.calls.empty()));
        mix_str(s.current.name);
        for (const ParsedParameter& p : s.current.parameters) { mix_str(p.name); }
        mix_str(s.param_name);
        return h;
    }

    // The Stage-2 value-boundary selection for a machine state in ParameterValue mode.
    // Returns the standing candidate's path outcome (Complete or EndOfInput) or the
    // EndOfInput outcome with the value open to the end; never a Definitive outcome.
    ToolCallParseProgress select_value(RegionState& s) noexcept {
        const std::string_view required_close = tool_close_literal(s.param_family);
        // Candidate census: every closer occurrence and the balanced subset (depth 0, same
        // family only). A complete same-family opener inside the value consumes the next
        // closer; a partial opener at the value end cannot decide depth yet and is ignored.
        std::vector<std::size_t> all;
        std::vector<std::size_t> balanced;
        {
            std::size_t depth = 0;
            for (std::size_t i = s.value_begin; i < text.size();) {
                if (text[i] != '<') { ++i; continue; }
                if (starts_with_at(text, i, required_close)) {
                    if (depth == 0) { balanced.push_back(i); }
                    else { --depth; }
                    all.push_back(i);
                    i += required_close.size();
                    continue;
                }
                ToolOpenTag opener = {};
                if (parse_tool_parameter_open(text.substr(i), opener) == ToolHeaderStatus::Complete) {
                    ++depth;
                    i += opener.consumed;
                    continue;
                }
                ++i;
            }
        }
        auto try_list = [&](const std::vector<std::size_t>& list, bool lazy) {
            bool skipped_viable = false;
            for (const std::size_t c : list) {
                if (charge()) { exhausted = true; }
                if (exhausted) { break; }
                if (skipped_viable && (c == 0 || text[c - 1] != '\n')) { continue; }
                const std::size_t after = c + required_close.size();
                if (classify_close_continuation(text, after, s.fn_family, s.wrapper) ==
                    CloseContinuation::Invalid) {
                    continue;
                }
                // Plausibility: the continuation opens the next parameter of the same call.
                // Its name must not repeat a name of the current call (including the value
                // being closed) and, when the tool has an unambiguous non-empty declared
                // schema, must be declared (the predicate returns false exactly in that case).
                const std::size_t at = skip_ws(text, after);
                if (at < text.size()) {
                    ToolOpenTag next = {};
                    if (parse_tool_parameter_open(text.substr(at), next) == ToolHeaderStatus::Complete) {
                        const std::string_view m = next.name;
                        bool bad = m == s.param_name;
                        if (!bad) {
                            for (const ParsedParameter& p : s.current.parameters) {
                                if (p.name == m) { bad = true; break; }
                            }
                        }
                        if (!bad && policy.parameter_plausible != nullptr &&
                            !policy.parameter_plausible(policy.contract, s.current.name, m)) {
                            bad = true;
                        }
                        if (bad) { continue; }
                    }
                }
                const std::size_t saved_parameters = s.current.parameters.size();
                const std::size_t saved_calls      = s.region.calls.size();
                const std::uint32_t saved_lazy     = lazy_choices;
                s.current.parameters.push_back(
                    ParsedParameter{.name = s.param_name,
                                    .value = std::string(text.substr(s.value_begin, c - s.value_begin))});
                RegionState after_state = s;
                after_state.pos  = after;
                after_state.mode = RegionState::Mode::FunctionBody;
                const std::uint64_t key = state_hash(after_state);
                ToolCallParseProgress r;
                const auto found = memo.find(key);
                if (found != memo.end()) {
                    r.termination = ToolCallRegionTermination::Definitive; // memoized abandon
                } else if (depth + 1 >= kMaxStage2Depth) {
                    // R3-I8: the hard recursion bound fails closed.
                    exhausted = true;
                    s.current.parameters.resize(saved_parameters);
                    s.region.calls.resize(saved_calls);
                    lazy_choices = saved_lazy;
                    break;
                } else {
                    ++depth;
                    r = parse_region_at(text, policy, after_state, BoundaryMode::Consistent, this);
                    --depth;
                    if (r.termination == ToolCallRegionTermination::Definitive) {
                        memo.emplace(key, true);
                    }
                }
                if (r.termination != ToolCallRegionTermination::Definitive) {
                    if (lazy) { ++lazy_choices; }
                    return r;
                }
                // Abandoned: roll the machine state back to before the candidate.
                s.current.parameters.resize(saved_parameters);
                s.region.calls.resize(saved_calls);
                lazy_choices = saved_lazy;
                skipped_viable = true;
            }
            return ToolCallParseProgress{.termination = ToolCallRegionTermination::Definitive};
        };
        auto r = try_list(balanced, false);
        if (r.termination != ToolCallRegionTermination::Definitive) { return r; }
        r = try_list(all, true);
        if (r.termination != ToolCallRegionTermination::Definitive) { return r; }
        // Every candidate contradicted (or the budget/depth bound fired): the value is open
        // to the end. Never Complete.
        ToolCallParseProgress open;
        open.termination      = ToolCallRegionTermination::EndOfInput;
        open.break_offset     = s.value_begin;
        open.wrapper_at_break = s.wrapper;
        open.open_call        = std::move(s.current);
        open.open_value_open  = true;
        open.calls            = std::move(s.region.calls);
        return open;
    }
};

ToolCallParseProgress parse_region_at(std::string_view text, const ToolCallParsePolicy& policy,
                                      RegionState s, BoundaryMode mode,
                                      ConsistentCompleter* cc) {
    // R3-01: Stage 2 parses the canonical grammar — tolerant header repairs and the tolerant
    // NoEntry repair entry are Stage-1 mechanisms and do not run in the Consistent parse.
    ToolCallParsePolicy effective = policy;
    if (mode == BoundaryMode::Consistent) { effective.tolerant = false; }
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
    };

    auto complete = [&]() {
        out.termination                   = ToolCallRegionTermination::Complete;
        out.calls                         = std::move(s.region.calls);
    };

    auto complete_call = [&]() {
        s.region.calls.push_back(std::move(s.current));
        s.current = {};
        if (s.wrapper == ToolWrapperKind::FunctionCalls) { s.function_calls_had_call = true; }
    };

    for (;;) {
        // R3-14: one glue step per machine transition in the Stage-2 parse; the budget fails
        // closed (the base is rejected as a definitive break, never completed).
        if (mode == BoundaryMode::Consistent && cc->charge()) {
            out.termination                   = ToolCallRegionTermination::Definitive;
            out.failure                       = ToolCallParseFailure::MalformedStructure;
            out.break_offset                  = s.pos;
            out.wrapper_at_break              = s.wrapper;
            out.open_call                     = std::move(s.current);
            out.calls                         = std::move(s.region.calls);
            return out;
        }
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
            // NoEntry: a broken opener or a non-marker byte. R3-02 (R3-I3): a break directly
            // after an open wrapper literal, on a byte that does not attempt a function
            // opener (no function header was tried inside the wrapper), leaves no payload
            // scope: the prose owns nothing, and the retry may re-read the region from the
            // break offset. A failed wrapper still owns its scope when the break happened
            // while parsing a function header (R2-I1); the prose flag is set before the
            // tolerant repair and skips it.
            if (s.mode == RegionState::Mode::ExpectFunction && s.wrapper != ToolWrapperKind::None &&
                entry.kind == TopLevelEntry::NoEntry) {
                std::size_t j = i;
                if (starts_with_at(text, j, kImStartMarker)) { j += kImStartMarker.size(); }
                const bool attempts_opener = j < text.size() &&
                    (starts_with_at(text, j, "function") || starts_with_at(text, j, "invoke"));
                if (text[i] != '<' && !attempts_opener) {
                    out.prose_after_wrapper = true;
                    invalid(ToolCallParseFailure::MalformedStructure, i);
                    return out;
                }
            }
            // Under tolerant mode a broken opener (a dropped '<' or keyword) also enters the
            // header state, where the recovery rules run; strict mode invalidates below with
            // the same diagnostics the old batch parser produced.
            if (effective.tolerant) {
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
            if (st == ToolHeaderStatus::NoMatch && effective.tolerant) {
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
                    // R3-03: a definitive top-level break after complete calls is trailing
                    // content, same class as strict (the repair could not find a function
                    // opener to attribute the break to).
                    invalid(s.region.calls.empty() ? ToolCallParseFailure::MalformedStructure
                                                   : ToolCallParseFailure::TrailingContent,
                            i);
                    return out;
                }
                scan += (opener.kind == ToolTagKind::Function ? 8 : 6);
                if (scan < text.size() && (text[scan] == '=' || is_tool_format_whitespace(text[scan]))) {
                    base    = scan;
                    from_boundary = true;
                    st      = parse_tool_header_after_keyword(text.substr(scan), opener.kind, opener);
                } else {
                    invalid(s.region.calls.empty() ? ToolCallParseFailure::MalformedStructure
                                                   : ToolCallParseFailure::TrailingContent,
                            i);
                    return out;
                }
            }
            std::string_view name;
            bool ws_boundary = false;
            if (st == ToolHeaderStatus::Complete) {
                name = opener.name;
                s.pos = from_boundary ? base + opener.consumed : i + opener.consumed;
            } else if (effective.tolerant) {
                // The model sometimes drops the '>' after the function name (for example a name
                // followed directly by a newline and a parameter tag). Recover by scanning the
                // identifier run and accepting it when format whitespace separates it from the
                // next '<' or the end of region.
                if (base <= text.size()) {
                    const std::string_view body = text.substr(base);
                    std::size_t scan            = 0;
                    while (scan < body.size() && body[scan] == '=') { ++scan; }
                    const std::size_t ident_begin = scan;
                    while (scan < body.size() && scan - ident_begin < effective.max_name_length &&
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
                // Invalid or a failed recovery is a definitive break (R3-03: trailing content
                // after complete calls, same class as strict).
                if (st == ToolHeaderStatus::NeedMore) {
                    truncated(false, i);
                } else {
                    invalid(s.region.calls.empty() ? ToolCallParseFailure::MalformedStructure
                                                   : ToolCallParseFailure::TrailingContent,
                            i);
                }
                return out;
            }
            if (!is_valid_tool_name(name, effective.max_name_length)) {
                invalid(ToolCallParseFailure::InvalidToolName, i);
                return out;
            }
            // R2-I5 (CR5): the declared-tool identity policy is independent of syntax
            // tolerance: tolerant mode repairs syntax damage, it does not let the model
            // invent function identities. The identity check applies in strict and tolerant
            // alike whenever the contract enforces declared names.
            if (effective.enforce_declared_names &&
                (effective.declared_check == nullptr ||
                 !effective.declared_check(effective.contract, name))) {
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
            if (mode == BoundaryMode::Consistent) {
                // R3-01: the Stage-2 candidate selection owns the boundary; the machine
                // returns the standing path's outcome (or the open-value EndOfInput).
                out = cc->select_value(s);
                out.stage2_lazy_boundary_used = cc->lazy_choices > 0;
                return out;
            }
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
                // R3-08: every parameter occurrence is retained; a repeated name is ambiguous
                // for a declared tool, and the legacy last-wins merge (with its count) happens
                // at the parse entry, only for tools the contract does not declare uniquely.
                s.current.parameters.push_back(
                    ParsedParameter{.name = s.param_name,
                                    .value = std::string(text.substr(s.value_begin, candidate - s.value_begin))});
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
        // R3-06: a close line is a run of the opener character of at least the opener length,
        // followed only by format whitespace (CR, spaces, tabs) up to the line end — CRLF and
        // LF-only framing both keep the close valid. The opener line never closes the fence.
        if (in_fence_ && !opener_line_ && close_ok_ &&
            (phase_ == Phase::LineRun || phase_ == Phase::LineTail) && run_len_ >= fence_len_) {
            in_fence_  = false;
            fence_len_ = 0;
        }
        opener_line_ = false;
        close_ok_    = false;
        run_len_     = 0;
        indent_      = 0;
        phase_       = Phase::LineIndent;
        return in_fence_ ? Verdict::Content : Verdict::Pass;
    }
    if (!in_fence_) {
        // Outside a fence: only a run of >= 3 '`' or '~' at line start (indent <= 3) is fence
        // structure; everything else is left to the marker machine.
        if (phase_ == Phase::LineIndent) {
            if (byte == ' ') {
                ++indent_;
                if (indent_ > 3) { phase_ = Phase::LineBody; }
                return Verdict::Pass;
            }
            if (byte == '`' || byte == '~') {
                phase_    = Phase::LineRun;
                run_char_ = byte;
                run_len_  = 1;
                return Verdict::Content;
            }
            phase_ = Phase::LineBody;
            return Verdict::Pass;
        }
        if (phase_ == Phase::LineRun) {
            if (byte == run_char_) {
                ++run_len_;
                if (run_len_ == 3) {
                    // The run opens the fence at its third character; the rest of the line is
                    // the info string, and further run characters extend the fence length.
                    in_fence_     = true;
                    fence_char_   = byte;
                    fence_len_    = 3;
                    fence_indent_ = indent_;
                    opener_line_  = true;
                }
                return Verdict::Content;
            }
            phase_ = Phase::LineBody;
            return Verdict::Pass;
        }
        return Verdict::Pass; // LineBody
    }
    // Inside a fence.
    if (opener_line_) {
        if (phase_ == Phase::LineRun) {
            if (byte == fence_char_) { ++fence_len_; return Verdict::Content; }
            // The run ended: the info string starts (the cancel rule only applies from the
            // info string on, where a backtick means inline code, not a longer run).
            phase_ = Phase::OpenerTail;
            return Verdict::Content;
        }
        if (fence_char_ == '`' && byte == '`') {
            // A backtick in a backtick opener's info string: the line is inline code, not a
            // fence (CommonMark). Cancel the opener; the line is ordinary content from here.
            in_fence_    = false;
            fence_len_   = 0;
            opener_line_ = false;
            phase_       = Phase::LineBody;
            return Verdict::Pass;
        }
        return Verdict::Content; // the info string
    }
    // Other lines inside a fence (always content; no nested fences).
    if (phase_ == Phase::LineIndent) {
        if (byte == ' ') {
            ++indent_;
            if (indent_ > fence_indent_ + 3) { phase_ = Phase::LineBody; }
            return Verdict::Content;
        }
        if (byte == fence_char_) {
            phase_    = Phase::LineRun;
            run_char_ = byte;
            run_len_  = 1;
            close_ok_ = true;
            return Verdict::Content;
        }
        phase_ = Phase::LineBody;
        return Verdict::Content;
    }
    if (phase_ == Phase::LineRun) {
        if (byte == run_char_) { ++run_len_; return Verdict::Content; }
        if (is_tool_format_whitespace(byte)) { phase_ = Phase::LineTail; }
        else { close_ok_ = false; phase_ = Phase::LineBody; }
        return Verdict::Content;
    }
    if (phase_ == Phase::LineTail) {
        if (!is_tool_format_whitespace(byte)) { close_ok_ = false; phase_ = Phase::LineBody; }
        return Verdict::Content;
    }
    return Verdict::Content; // LineBody
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

bool ToolCallStreamParser::marker_byte(char byte, std::string& visible) noexcept {
    if (!marker_prefix_.empty()) {
        marker_prefix_.push_back(byte);
        ToolOpenTag marker = {};
        const ToolMarkerStatus state = classify_tool_marker_prefix(marker_prefix_, marker);
        if (state == ToolMarkerStatus::Complete) {
            latch(marker_prefix_);
            return true;
        }
        if (state == ToolMarkerStatus::NotMarker) {
            // R3-05 (generalized F8): publish the failed candidate's head up to the next '<'
            // (the head's trailing format whitespace stays held in pending_ws_, R3-07, so
            // streaming output equals one-shot output), and re-feed the remaining bytes
            // through this same transition. They already passed the fence tracker, so no
            // fence re-scan is needed here.
            std::string failed = std::move(marker_prefix_);
            marker_prefix_.clear();
            publish(pending_ws_, visible);
            pending_ws_.clear();
            const std::size_t next = failed_marker_candidate_rescan_start(failed);
            const std::string_view head =
                std::string_view(failed).substr(0, next == std::string_view::npos ? failed.size() : next);
            std::size_t keep = head.size();
            while (keep > 0 && is_tool_format_whitespace(head[keep - 1])) { --keep; }
            publish(head.substr(0, keep), visible);
            if (keep < head.size()) { pending_ws_.assign(head.substr(keep)); }
            if (next != std::string_view::npos) {
                const std::string_view rest = std::string_view(failed).substr(next);
                rescan_steps_ += rest.size(); // R3-14: deterministic work counter
                for (std::size_t j = 0; j < rest.size(); ++j) {
                    if (marker_byte(rest[j], visible)) {
                        region_.append(rest.substr(j + 1));
                        return true;
                    }
                }
            }
        }
        return false;
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
    return false;
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
        if (marker_byte(byte, visible)) {
            region_.append(chunk.substr(i + 1));
            return visible;
        }
    }
    return visible;
}
namespace {

// R3-06: the shadow marker scan: the same classify_tool_marker_prefix transition over fence
// bytes, never latching — it counts the complete top-level markers a fence suppressed.
struct ShadowMarkerScan {
    std::string candidate;
    std::uint32_t complete = 0;
    void consume(char byte) {
        if (candidate.empty()) {
            if (byte == '<') { candidate.push_back(byte); }
            return;
        }
        candidate.push_back(byte);
        ToolOpenTag marker = {};
        const ToolMarkerStatus state = classify_tool_marker_prefix(candidate, marker);
        if (state == ToolMarkerStatus::Complete) {
            ++complete;
            candidate.clear();
        } else if (state == ToolMarkerStatus::NotMarker) {
            const std::size_t next = candidate.find('<', 1);
            candidate = (next == std::string_view::npos) ? std::string{} : candidate.substr(next);
        }
    }
};

} // namespace

std::vector<char> fence_mask(std::string_view region) {
    std::vector<char> mask(region.size(), 0);
    ToolCallStreamParser::FenceTracker tracker;
    for (std::size_t i = 0; i < region.size(); ++i) {
        if (tracker.consume(region[i]) == ToolCallStreamParser::FenceTracker::Verdict::Content) {
            mask[i] = 1;
        }
    }
    return mask;
}

FenceDiagnostics compute_fence_diagnostics(std::string_view pre_latch,
                                           std::string_view region) {
    FenceDiagnostics diagnostics;
    auto scan = [&](std::string_view bytes) {
        ToolCallStreamParser::FenceTracker tracker;
        ShadowMarkerScan shadow;
        for (const char byte : bytes) {
            if (tracker.consume(byte) == ToolCallStreamParser::FenceTracker::Verdict::Content) {
                shadow.consume(byte);
            }
        }
        if (tracker.open()) { diagnostics.ended_in_unclosed_fence = true; }
        diagnostics.suppressed_markers += shadow.complete;
    };
    scan(pre_latch);
    scan(region);
    return diagnostics;
}
ToolCallStreamResult ToolCallStreamParser::finish(FinishReason finish_reason) const {
    ToolCallStreamResult result;
    result.marker_seen  = marker_seen_;
    result.rescan_steps = rescan_steps_;
    // R3-06: deterministic fence scan over the pre-latch bytes and the latched region (the
    // same bytes in one-shot and streaming): suppressed markers and the unclosed-fence flag.
    const std::string pre_latch = content_ + pending_ws_ + marker_prefix_;
    const FenceDiagnostics fence_diag =
        compute_fence_diagnostics(pre_latch, latched_ ? region_ : std::string_view{});
    result.fenced_markers_suppressed = fence_diag.suppressed_markers;
    result.ended_in_unclosed_fence   = fence_diag.ended_in_unclosed_fence;
    if (!latched_) {
        // No marker anywhere: the whole stream is ordinary content (R3-09: the entry reports
        // default diagnostics; the fence fields stay visible).
        result.status  = ToolCallStreamStatus::Invalid;
        result.failure = ToolCallParseFailure::MalformedStructure;
        result.tail    = std::move(pre_latch);
        return result;
    }
    // R3-06: the retry search skips markers inside a recognized fence (the fence state is
    // computed over the region bytes, independent of how they were chunked).
    const std::vector<char> fenced = fence_mask(region_);
    const bool natural_stop =
        finish_reason == FinishReason::StopToken || finish_reason == FinishReason::None;

    // Stage 1 (R3-02/R3-04/R3-06): the greedy parse + retry chain. Only a clean Complete is
    // accepted here; every other attempt feeds Stage 2 (consistent completion) and Stage 3
    // (tolerant recovery) in chain order.
    struct Attempt {
        std::size_t base;
        ToolCallParseProgress progress;
    };
    constexpr std::size_t kMaxChainAttempts = 256; // R3-04: the bounded chain
    std::vector<Attempt> chain;
    std::size_t base = 0;
    for (std::size_t attempts = 0; attempts < kMaxChainAttempts; ++attempts) {
        ToolCallParseProgress progress = parse_region(std::string_view(region_).substr(base), policy_);
        if (progress.termination == ToolCallRegionTermination::Complete) {
            // Stage 1 accepts: the result is final (R3-I2), subject only to R3-08 at the
            // parse entry.
            result.status        = ToolCallStreamStatus::Complete;
            result.failure       = ToolCallParseFailure::None;
            result.region.calls  = std::move(progress.calls);
            result.truncated_tail = false;
            result.tail          = region_.substr(0, base);
            return result;
        }
        chain.push_back(Attempt{base, std::move(progress)});
        const Attempt& last = chain.back();
        // R3-02: a prose break directly after an open wrapper owns no scope (the retry may
        // re-read from the break offset). R2-I1/R3-04: a wrapper that failed while parsing
        // a function header owns its scope; an open value keeps its payload after a cut
        // (the value bytes may be the real truncated call) but not after a natural stop.
        const bool owned = last.progress.wrapper_at_break != ToolWrapperKind::None &&
                           !last.progress.prose_after_wrapper;
        const bool open_value =
            last.progress.termination == ToolCallRegionTermination::EndOfInput &&
            last.progress.open_value_open;
        if (owned || (open_value && !natural_stop)) { break; }
        // The retried slice starts with the entry marker (a latched or retry-found complete
        // marker). A definitive break at the marker's own header must not re-read that same
        // marker; the search starts strictly after the failed marker's opener bytes, or at
        // the break byte when the break proved itself deeper in the structure.
        ToolOpenTag entry_marker = {};
        const ToolMarkerStatus entry_status =
            classify_tool_marker_prefix(region_.substr(base), entry_marker);
        const std::size_t entry_end =
            base + (entry_status == ToolMarkerStatus::Complete ? entry_marker.consumed : 1);
        // R3-04: never base + 1 — EndOfInput retries from the break offset (the open
        // value's start when a value is open, the input-end position otherwise).
        const std::size_t from = std::max(base + last.progress.break_offset, entry_end);
        std::size_t next = find_tool_marker(region_, from);
        while (next != std::string_view::npos && fenced[next]) {
            next = find_tool_marker(region_, next + 1); // R3-06: skip fenced markers
        }
        if (next == std::string_view::npos || next <= base) { break; }
        base = next;
    }

    // Stage 2 (R3-01): consistent completion on every admissible base of the Stage-1 chain.
    // The canonical parse (tolerant repairs off) reuses the Stage-1 machine with the
    // ConsistentCompleter value-boundary selection.
    struct Stage2Result {
        std::size_t base;
        ToolCallParseProgress progress;
    };
    std::vector<Stage2Result> stage2;
    for (const Attempt& attempt : chain) {
        // R3-14: without a parameter closer literal there is no consistent value boundary,
        // so the consistent parse degenerates to the greedy one; skip it (steps stay 0).
        if (region_.find(tool_close_literal(ToolTagKind::Parameter), attempt.base) ==
            std::string_view::npos) {
            continue;
        }
        ConsistentCompleter cc;
        cc.text   = std::string_view(region_).substr(attempt.base);
        cc.policy = policy_;
        cc.budget = policy_.stage2_step_budget != 0
                        ? policy_.stage2_step_budget
                        : std::max<std::uint64_t>(200000, 32 * cc.text.size());
        const ToolCallParseProgress r =
            parse_region_at(cc.text, cc.policy, RegionState{}, BoundaryMode::Consistent, &cc);
        result.stage2_steps += cc.steps;
        stage2.push_back(Stage2Result{attempt.base, r});
    }
    {
        // R3-01 selection: the earliest base whose chosen path used only pass-1 (balanced)
        // boundaries; else the only completed base; two or more completed bases without a
        // balanced one reject the region as ambiguous (Stage 3 does not run).
        std::size_t balanced_index = std::size_t(-1);
        std::size_t only_index     = std::size_t(-1);
        std::size_t complete_count = 0;
        for (std::size_t i = 0; i < stage2.size(); ++i) {
            if (stage2[i].progress.termination != ToolCallRegionTermination::Complete) { continue; }
            ++complete_count;
            if (balanced_index == std::size_t(-1) && !stage2[i].progress.stage2_lazy_boundary_used) {
                balanced_index = i;
            }
            only_index = i;
        }
        if (complete_count >= 2 && balanced_index == std::size_t(-1)) {
            result.status  = ToolCallStreamStatus::Invalid;
            result.failure = ToolCallParseFailure::AmbiguousStructure;
            result.tail    = region_;
            return result;
        }
        if (complete_count == 1 || balanced_index != std::size_t(-1)) {
            // Stage 2 accepts: the region was resolved by consistent completion (R3-I7).
            const std::size_t pick = balanced_index != std::size_t(-1) ? balanced_index : only_index;
            result.status  = ToolCallStreamStatus::Complete;
            result.failure = ToolCallParseFailure::None;
            result.region.calls = std::move(stage2[pick].progress.calls);
            result.truncated_tail             = false;
            result.markup_tolerant_completion = true;
            result.tail          = region_.substr(0, stage2[pick].base);
            return result;
        }
    }

    // Stage 3 (R3-03): tolerant recovery over the Stage-1 chain, only when Stage 1 and
    // Stage 2 accepted nothing (and Stage 2 did not return AmbiguousStructure). Commit the
    // first attempt the recovery decision accepts. The value-closer check is precomputed per
    // attempt: the tail from the break offset must carry no `</param>`/`</parameter>`
    // literal (a wrapper or function closer does not count: it cannot extend a value).
    for (const Attempt& attempt : chain) {
        const ToolCallRecoveryPolicy decision_policy{
            policy_.tolerant,
            finish_reason,
            region_.substr(attempt.base + attempt.progress.break_offset).find("</param>") !=
                std::string_view::npos,
        };
        const ToolCallRecoveryResult recovery =
            decide_tool_call_recovery(attempt.progress, decision_policy);
        if (recovery.decision == ToolCallRecoveryDecision::CommitCalls) {
            result.status        = ToolCallStreamStatus::Complete;
            result.failure       = recovery.diagnostic;
            result.region.calls  = attempt.progress.calls;
            result.truncated_tail = true;
            result.tail          = region_.substr(0, attempt.base);
            return result;
        }
    }

    // Nothing committed: the strict failure class of the first attempt (R3-03 item 6):
    // Definitive keeps its reason, EndOfInput is TruncatedTail in tolerant mode and
    // MalformedStructure in strict mode (as today).
    const ToolCallParseProgress& first = chain.front().progress;
    result.status  = ToolCallStreamStatus::Invalid;
    result.failure = first.termination == ToolCallRegionTermination::Definitive
                         ? first.failure
                         : policy_.tolerant ? ToolCallParseFailure::TruncatedTail
                                            : ToolCallParseFailure::MalformedStructure;
    result.tail    = region_;
    return result;
}


} // namespace ninfer::models::qwen3_5::frontend
