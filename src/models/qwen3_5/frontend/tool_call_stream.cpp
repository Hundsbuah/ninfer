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
        ParameterValue,    // inside the parameter value (nested-opener depth scan)
        ExpectWrapperClose,// after a function close inside a <tool_call> wrapper
        Top,               // top level of the region: next call, wrapper close, end, trailing
    };

    Mode mode = Mode::ExpectFunction;
    std::size_t pos = 0;
    ToolTagKind fn_family = ToolTagKind::Function;
    ToolTagKind param_family = ToolTagKind::Parameter;
    bool wrapper_close_expected = false;
    bool inside_function_calls = false;
    bool had_calls = false;
    ParsedFunctionCall current;
    ParsedToolRegion region;
    // A definitive invalid that must not be retained even in tolerant mode (an empty
    // function_calls wrapper fails the whole region, as the wire format requires the wrapper
    // to contain at least one call).
    bool no_retain = false;
    std::string param_name; // the current parameter's name (set in ParameterHeader)
    // Parameter value scan
    std::size_t value_begin = 0;
    std::uint8_t depth = 1;
    std::size_t next_close = std::string::npos;
};


// After a candidate closer the grammar continues with another parameter, the current
// function's closer, or (for an output cut short) the region end. A closer followed by
// anything else is text the value quotes, such as a command that echoes tool markup.
// Inside a <tool_call> wrapper the function close must itself be followed by the wrapper
// close (or the region end): a new function/invoke opener there is structurally impossible,
// so the candidate was a quoted closer and the value keeps scanning (P3.10: a string value
// must never be committed at a quoted boundary).
bool is_close_continuation(std::string_view text, std::size_t after, ToolTagKind fn_family,
                           bool wrapper_close_expected) noexcept {
    const std::size_t at = skip_ws(text, after);
    if (at >= text.size()) { return true; }
    ToolOpenTag opener = {};
    if (parse_tool_parameter_open(text.substr(at), opener) == ToolHeaderStatus::Complete) {
        return true;
    }
    if (starts_with_at(text, at, tool_close_literal(fn_family))) {
        if (!wrapper_close_expected) { return true; }
        const std::size_t after_fn =
            skip_ws(text, at + tool_close_literal(fn_family).size());
        if (after_fn >= text.size()) { return true; }
        return starts_with_at(text, after_fn, tool_close_literal(ToolTagKind::ToolCall));
    }
    return false;
}

// Deterministic parse of a complete tool-region slice. The slice always starts at a marker
// (the latched marker or a <tool_call> wrapper found by the retry loop). Every transition is
// byte-driven, so the outcome is independent of how the bytes were chunked upstream.
ToolCallParseProgress parse_region(std::string_view text, const ToolCallParsePolicy& policy) {
    RegionState s;
    ToolCallParseProgress out;

    auto invalid = [&](ToolCallParseFailure f) {
        // A definitive structural break (or trailing content after complete calls). The
        // progress records the complete calls and the open call; the recovery policy decides
        // retention, never the parse.
        out.termination                   = ToolCallRegionTermination::Definitive;
        out.failure                       = f;
        out.open_call                     = std::move(s.current);
        out.unrecoverable_break           = s.no_retain;
        out.calls                         = std::move(s.region.calls);
        out.duplicate_parameters_repaired = s.region.duplicate_parameters_repaired;
    };

    auto truncated = [&](bool value_open) {
        // The input ended mid-region. Strict mode maps this to a structural failure; tolerant
        // recovery may commit complete calls, and the open call only when no value of it is
        // still open.
        out.termination                   = ToolCallRegionTermination::EndOfInput;
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
        if (s.inside_function_calls) { s.had_calls = true; }
    };

    for (;;) {
        switch (s.mode) {
        case RegionState::Mode::ExpectFunction:
        case RegionState::Mode::Top: {
            const bool top = s.mode == RegionState::Mode::Top;
            s.mode = RegionState::Mode::Top;
            const std::size_t i = skip_ws(text, s.pos);
            if (i == text.size()) {
                if (top) {
                    complete();
                    return out;
                }
                // A wrapper literal was consumed but no function followed.
                truncated(false);
                return out;
            }
            if (s.inside_function_calls &&
                starts_with_at(text, i, tool_close_literal(ToolTagKind::FunctionCalls))) {
                s.pos = i + tool_close_literal(ToolTagKind::FunctionCalls).size();
                if (!s.had_calls) {
                    s.no_retain = true;
                    invalid(ToolCallParseFailure::MalformedStructure);
                    return out;
                }
                s.inside_function_calls = false;
                complete();
                return out;
            }
            if (policy.prefix && s.inside_function_calls &&
                is_strict_prefix_of(tool_close_literal(ToolTagKind::FunctionCalls), text, i)) {
                // An in-progress </function_calls> close at the slice end.
                truncated(false);
                return out;
            }
            if (starts_with_at(text, i, tool_open_literal(ToolTagKind::ToolCall))) {
                s.pos                  = i + tool_open_literal(ToolTagKind::ToolCall).size();
                s.wrapper_close_expected = true;
                s.mode = RegionState::Mode::ExpectFunction;
                continue;
            }
            if (starts_with_at(text, i, tool_open_literal(ToolTagKind::FunctionCalls))) {
                s.pos                 = i + tool_open_literal(ToolTagKind::FunctionCalls).size();
                s.inside_function_calls = true;
                s.mode = RegionState::Mode::ExpectFunction;
                continue;
            }
            ToolOpenTag opener_probe = {};
            const ToolHeaderStatus opener_status = parse_tool_function_open(text.substr(i), opener_probe);
            // A complete opener is a function. Under tolerant mode a broken opener
            // ("<function=memory\n...", a dropped '<' or keyword) also enters the header state,
            // where the recovery rules run; strict mode invalidates it below with the same
            // diagnostics the old batch parser produced.
            if (opener_status == ToolHeaderStatus::Complete ||
                (policy.tolerant && opener_status != ToolHeaderStatus::Complete)) {
                s.pos  = i;
                s.mode = RegionState::Mode::FunctionHeader;
                continue;
            }
            if (policy.prefix &&
                (opener_status == ToolHeaderStatus::NeedMore ||
                 is_strict_prefix_of(tool_open_literal(ToolTagKind::ToolCall), text, i) ||
                 is_strict_prefix_of(tool_open_literal(ToolTagKind::FunctionCalls), text, i))) {
                // An in-progress opener or wrapper literal at the slice end: the generated
                // prefix may still complete into a valid structure.
                truncated(false);
                return out;
            }
            // A byte that is not a marker: trailing text after complete calls, or a broken
            // region start. A definitive structural outcome: tolerant recovery commits the
            // complete calls (recording the truncation); strict mode rejects with the reason.
            if (s.region.calls.empty()) { invalid(ToolCallParseFailure::MalformedStructure); }
            else { invalid(ToolCallParseFailure::TrailingContent); }
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
                    invalid(ToolCallParseFailure::MalformedStructure);
                    return out;
                }
                scan += (opener.kind == ToolTagKind::Function ? 8 : 6);
                if (scan < text.size() && (text[scan] == '=' || is_tool_format_whitespace(text[scan]))) {
                    base    = scan;
                    from_boundary = true;
                    st      = parse_tool_header_after_keyword(text.substr(scan), opener.kind, opener);
                } else {
                    invalid(ToolCallParseFailure::MalformedStructure);
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
                    truncated(false);
                } else {
                    invalid(ToolCallParseFailure::MalformedStructure);
                }
                return out;
            }
            if (!is_valid_tool_name(name, policy.max_name_length)) {
                invalid(ToolCallParseFailure::InvalidToolName);
                return out;
            }
            // Strict mode rejects a name outside the declared tool set. Tolerant mode keeps an
            // otherwise well-formed call structured and leaves the identity judgment to the
            // consumer: leaking the raw region to content would turn a valid call into prose.
            if (!policy.tolerant && policy.enforce_declared_names &&
                (policy.declared_check == nullptr ||
                 !policy.declared_check(policy.contract, name))) {
                invalid(ToolCallParseFailure::UndeclaredTool);
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
                // close is a truncation, not a malformed structure.
                truncated(false);
                return out;
            }
            if (starts_with_at(text, i, tool_close_literal(s.fn_family))) {
                s.pos = i + tool_close_literal(s.fn_family).size();
                if (s.wrapper_close_expected) {
                    s.mode = RegionState::Mode::ExpectWrapperClose;
                } else {
                    complete_call();
                    s.mode = RegionState::Mode::Top;
                }
                continue;
            }
            ToolOpenTag opener = {};
            const ToolHeaderStatus param_status = parse_tool_parameter_open(text.substr(i), opener);
            if (param_status == ToolHeaderStatus::Complete) {
                s.pos  = i;
                s.mode = RegionState::Mode::ParameterHeader;
                continue;
            }
            if (policy.prefix &&
                (param_status == ToolHeaderStatus::NeedMore ||
                 is_strict_prefix_of(tool_close_literal(s.fn_family), text, i))) {
                // An in-progress parameter opener or function close at the slice end.
                truncated(false);
                return out;
            }
            invalid(ToolCallParseFailure::MalformedStructure);
            return out;
        }
        case RegionState::Mode::ParameterHeader: {
            const std::size_t i = s.pos;
            ToolOpenTag opener = {};
            const ToolHeaderStatus st = parse_tool_parameter_open(text.substr(i), opener);
            if (st != ToolHeaderStatus::Complete || opener.name.empty()) {
                // A broken or cut-off parameter header fails the call even in tolerant mode:
                // there is no value to retain, and the current call is dropped with it.
                invalid(ToolCallParseFailure::MalformedStructure);
                return out;
            }
            s.param_family = opener.kind;
            s.param_name   = std::string(opener.name);
            s.value_begin  = i + opener.consumed;
            s.depth        = 1;
            s.next_close   = text.find(tool_close_literal(opener.kind), s.value_begin);
            s.pos          = s.value_begin;
            s.mode         = RegionState::Mode::ParameterValue;
            continue;
        }
        case RegionState::Mode::ParameterValue: {
            const std::string_view required_close = tool_close_literal(s.param_family);
            std::size_t scan       = s.value_begin;
            std::size_t next_close = s.next_close;
            std::size_t value_end  = 0;
            std::size_t close_len  = 0;
            bool found             = false;
            while (scan < text.size()) {
                if (scan == next_close) {
                    // Closers of parameters the value quotes whole always balance their
                    // openers; only the outermost one must be followed by the grammar's next
                    // token.
                    if (s.depth != 1 ||
                        is_close_continuation(text, scan + required_close.size(), s.fn_family,
                                               s.wrapper_close_expected)) {
                        --s.depth;
                    }
                    if (s.depth == 0) {
                        value_end = scan;
                        close_len = required_close.size();
                        found     = true;
                        break;
                    }
                    scan       += required_close.size();
                    next_close  = text.find(required_close, scan);
                    continue;
                }
                if (text[scan] == '<') {
                    ToolOpenTag nested = {};
                    if (parse_tool_parameter_open(text.substr(scan), nested) ==
                        ToolHeaderStatus::Complete) {
                        ++s.depth;
                        scan = scan + nested.consumed;
                        continue;
                    }
                }
                ++scan;
            }
            auto commit_parameter = [&](std::string_view value) {
                // Last occurrence wins, as it would in JSON object syntax, rather than
                // discarding an otherwise well-formed call.
                const auto existing = std::find_if(s.current.parameters.begin(),
                                                   s.current.parameters.end(),
                                                   [&](const ParsedParameter& candidate) {
                                                       return candidate.name == s.param_name;
                                                   });
                if (existing != s.current.parameters.end()) {
                    existing->value = std::string(value);
                    ++s.region.duplicate_parameters_repaired;
                } else {
                    s.current.parameters.push_back(
                        ParsedParameter{.name = s.param_name, .value = std::string(value)});
                }
            };
            if (!found) {
                // The region ends inside the value: a cut parameter value is never committed —
                // its bytes may still grow into a different value. The progress records the
                // open value; the recovery policy decides (it never commits a call whose value
                // is still open, whatever the finish reason was).
                truncated(true);
                return out;
            }
            commit_parameter(text.substr(s.value_begin, value_end - s.value_begin));
            s.pos  = value_end + close_len;
            s.mode = RegionState::Mode::FunctionBody;
            continue;
        }
        case RegionState::Mode::ExpectWrapperClose: {
            const std::size_t i = skip_ws(text, s.pos);
            if (i == text.size()) {
                // A complete call may be followed by the end of its budget before the closing
                // wrapper tag.
                truncated(false);
                return out;
            }
            if (starts_with_at(text, i, tool_close_literal(ToolTagKind::ToolCall))) {
                s.pos = i + tool_close_literal(ToolTagKind::ToolCall).size();
                complete_call();
                s.wrapper_close_expected = false;
                s.mode = RegionState::Mode::Top;
                continue;
            }
            if (policy.prefix && is_strict_prefix_of(tool_close_literal(ToolTagKind::ToolCall), text, i)) {
                // An in-progress </tool_call> close at the slice end.
                truncated(false);
                return out;
            }
            invalid(ToolCallParseFailure::MalformedStructure);
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
                publish(pending_ws_, visible);
                publish(marker_prefix_, visible);
                pending_ws_.clear();
                marker_prefix_.clear();
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
    // Retries move only to a later <tool_call> wrapper: the markup nested inside a failed
    // region (its <function=...> or <invoke>) must not re-read a truncated call.
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
            if (recovery.decision == ToolCallRecoveryDecision::CommitCallsAndOpenCall) {
                result.region.calls.push_back(std::move(progress.open_call));
            }
            result.truncated_tail =
                progress.termination != ToolCallRegionTermination::Complete;
            result.tail = region_.substr(0, base);
            return result;
        }
        if (base == 0) { first_failure = recovery.diagnostic; }
        const std::size_t next = region_.find(tool_open_literal(ToolTagKind::ToolCall), base + 1);
        if (next == std::string::npos) {
            result.status  = ToolCallStreamStatus::Invalid;
            result.failure = first_failure;
            result.tail    = region_;
            return result;
        }
        base = next;
    }
}

} // namespace ninfer::models::qwen3_5::frontend
