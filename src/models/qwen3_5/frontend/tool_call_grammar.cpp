#include "models/qwen3_5/frontend/tool_call_grammar.h"

namespace ninfer::models::qwen3_5::frontend {
namespace {

constexpr std::string_view keyword_of(ToolTagKind kind) noexcept {
    switch (kind) {
        case ToolTagKind::Function:  return "function";
        case ToolTagKind::Invoke:    return "invoke";
        case ToolTagKind::Parameter: return "parameter";
        case ToolTagKind::Param:     return "param";
        default:                     return {};
    }
}

void skip_ws(std::string_view text, std::size_t& pos) noexcept {
    while (pos < text.size() && is_tool_format_whitespace(text[pos])) { ++pos; }
}

bool is_literal_prefix(std::string_view text, std::string_view literal) noexcept {
    if (text.size() > literal.size()) { return false; }
    return literal.substr(0, text.size()) == text;
}

// Parse a value starting after the '=' of a short form or attribute. Returns the value and
// advances `pos` past it; NeedMore when the input ends inside a quoted value, Invalid for an
// empty unquoted value.
ToolHeaderStatus parse_value(std::string_view text, std::size_t& pos, std::string_view& value)
    noexcept {
    skip_ws(text, pos);
    if (pos >= text.size()) { return ToolHeaderStatus::NeedMore; }
    if (text[pos] == '"' || text[pos] == '\'') {
        const char quote = text[pos];
        ++pos;
        const std::size_t end = text.find(quote, pos);
        if (end == std::string_view::npos) { return ToolHeaderStatus::NeedMore; }
        value = text.substr(pos, end - pos);
        pos   = end + 1;
        return ToolHeaderStatus::Complete;
    }
    if (is_tool_format_whitespace(text[pos]) || text[pos] == '>') {
        return ToolHeaderStatus::Invalid;
    }
    const std::size_t begin = pos;
    while (pos < text.size() && !is_tool_format_whitespace(text[pos]) && text[pos] != '>' &&
           text[pos] != '/') {
        ++pos;
    }
    value = text.substr(begin, pos - begin);
    return ToolHeaderStatus::Complete;
}

// Parse the header body starting right after the tag keyword. `text[0]` is the boundary byte:
// '=' (short form), format whitespace or '>' (attribute list / bare opener). `out.consumed`
// counts from the start of `text` through the terminating '>'.
ToolHeaderStatus parse_header_after_keyword(std::string_view text, ToolTagKind kind,
                                            ToolOpenTag& out) noexcept {
    out.kind     = kind;
    out.name     = {};
    out.consumed = 0;
    if (text.empty()) { return ToolHeaderStatus::NeedMore; }
    std::size_t pos = 0;
    if (text[pos] != '=' && !is_tool_format_whitespace(text[pos]) && text[pos] != '>') {
        // The keyword runs on into other text ("<functionx", "<parameter" as a `param` tag,
        // "<function_calls>" as a function tag): not an opener of this family.
        return ToolHeaderStatus::NoMatch;
    }

    if (text[pos] == '=') {
        ++pos;
        std::string_view value = {};
        const ToolHeaderStatus status = parse_value(text, pos, value);
        if (status == ToolHeaderStatus::NeedMore) { return ToolHeaderStatus::NeedMore; }
        if (status == ToolHeaderStatus::Invalid) { return ToolHeaderStatus::Invalid; }
        skip_ws(text, pos);
        if (pos >= text.size()) { return ToolHeaderStatus::NeedMore; }
        if (text[pos] != '>') { return ToolHeaderStatus::Invalid; }
        out.name     = value;
        out.consumed = pos + 1;
        return ToolHeaderStatus::Complete;
    }

    bool saw_name     = false;
    std::string_view name = {};
    for (;;) {
        skip_ws(text, pos);
        if (pos >= text.size()) { return ToolHeaderStatus::NeedMore; }
        if (text[pos] == '>') {
            out.name     = saw_name ? name : std::string_view{};
            out.consumed = pos + 1;
            return ToolHeaderStatus::Complete;
        }
        if (!is_tool_name_char(text[pos])) { return ToolHeaderStatus::Invalid; }
        const std::size_t attr_begin = pos;
        while (pos < text.size() && is_tool_name_char(text[pos])) { ++pos; }
        const std::string_view attr_name = text.substr(attr_begin, pos - attr_begin);
        skip_ws(text, pos);
        if (pos >= text.size()) { return ToolHeaderStatus::NeedMore; }
        if (text[pos] != '=') { return ToolHeaderStatus::Invalid; }
        ++pos;
        std::string_view value = {};
        const ToolHeaderStatus status = parse_value(text, pos, value);
        if (status != ToolHeaderStatus::Complete) { return status; }
        if (!saw_name && attr_name == "name") {
            saw_name = true;
            name     = value;
        }
        if (pos >= text.size()) { return ToolHeaderStatus::NeedMore; }
        if (text[pos] != '>' && !is_tool_format_whitespace(text[pos])) {
            // An attribute value must be followed by whitespace or the header end; a name char
            // here ("name=\"a\"b=\"x\"") means the separator is missing.
            return ToolHeaderStatus::Invalid;
        }
    }
}

// Parse a tag opener at `text[0] == '<'` for one family: "<" + keyword + header body.
ToolHeaderStatus parse_open_body(std::string_view text, ToolTagKind kind, ToolOpenTag& out)
    noexcept {
    const std::string_view keyword = keyword_of(kind);
    if (text.empty() || text[0] != '<') { return ToolHeaderStatus::NoMatch; }
    if (text.size() <= keyword.size() + 1) {
        // "<" plus a strict prefix of the keyword (including the keyword alone): still
        // growable into a valid opener.
        for (std::size_t i = 0; i < text.size(); ++i) {
            const char expected = i == 0 ? '<' : keyword[i - 1];
            if (text[i] != expected) { return ToolHeaderStatus::NoMatch; }
        }
        return ToolHeaderStatus::NeedMore;
    }
    for (std::size_t i = 0; i < keyword.size(); ++i) {
        if (text[1 + i] != keyword[i]) { return ToolHeaderStatus::NoMatch; }
    }
    const ToolHeaderStatus status =
        parse_header_after_keyword(text.substr(1 + keyword.size()), kind, out);
    if (status == ToolHeaderStatus::Complete) { out.consumed += 1 + keyword.size(); }
    return status;
}

} // namespace

ToolHeaderStatus parse_tool_open_header(std::string_view text, ToolTagKind kind, ToolOpenTag& out)
    noexcept {
    const std::string_view literal = tool_open_literal(kind);
    if (!literal.empty()) {
        if (text.size() >= literal.size()) {
            if (text.substr(0, literal.size()) == literal) {
                out.kind     = kind;
                out.name     = {};
                out.consumed = literal.size();
                return ToolHeaderStatus::Complete;
            }
            return ToolHeaderStatus::NoMatch;
        }
        if (is_literal_prefix(text, literal)) { return ToolHeaderStatus::NeedMore; }
        return ToolHeaderStatus::NoMatch;
    }
    if (!is_tool_function_kind(kind) && !is_tool_parameter_kind(kind)) {
        return ToolHeaderStatus::NoMatch;
    }
    return parse_open_body(text, kind, out);
}

ToolHeaderStatus parse_tool_header_after_keyword(std::string_view text, ToolTagKind kind,
                                                 ToolOpenTag& out) noexcept {
    if (!is_tool_function_kind(kind) && !is_tool_parameter_kind(kind)) {
        return ToolHeaderStatus::NoMatch;
    }
    return parse_header_after_keyword(text, kind, out);
}

ToolHeaderStatus parse_tool_function_open(std::string_view text, ToolOpenTag& out) noexcept {
    const ToolHeaderStatus status =
        parse_tool_open_header(text, ToolTagKind::Function, out);
    if (status != ToolHeaderStatus::NoMatch) { return status; }
    return parse_tool_open_header(text, ToolTagKind::Invoke, out);
}

ToolHeaderStatus parse_tool_parameter_open(std::string_view text, ToolOpenTag& out) noexcept {
    const ToolHeaderStatus status =
        parse_tool_open_header(text, ToolTagKind::Parameter, out);
    if (status != ToolHeaderStatus::NoMatch) { return status; }
    return parse_tool_open_header(text, ToolTagKind::Param, out);
}

ToolMarkerStatus classify_tool_marker_prefix(std::string_view text, ToolOpenTag& out) noexcept {
    const std::string_view tool_call      = tool_open_literal(ToolTagKind::ToolCall);
    const std::string_view function_calls = tool_open_literal(ToolTagKind::FunctionCalls);
    if (!text.empty() && text[0] == '<') {
        if (text.size() >= tool_call.size()) {
            if (text.substr(0, tool_call.size()) == tool_call) {
                out.kind     = ToolTagKind::ToolCall;
                out.consumed = tool_call.size();
                return ToolMarkerStatus::Complete;
            }
        } else if (is_literal_prefix(text, tool_call)) {
            return ToolMarkerStatus::NeedMore;
        }
        if (text.size() >= function_calls.size()) {
            if (text.substr(0, function_calls.size()) == function_calls) {
                out.kind     = ToolTagKind::FunctionCalls;
                out.consumed = function_calls.size();
                return ToolMarkerStatus::Complete;
            }
        } else if (is_literal_prefix(text, function_calls)) {
            return ToolMarkerStatus::NeedMore;
        }
    }
    const ToolHeaderStatus function = parse_tool_function_open(text, out);
    if (function == ToolHeaderStatus::Complete) { return ToolMarkerStatus::Complete; }
    if (function == ToolHeaderStatus::NeedMore) { return ToolMarkerStatus::NeedMore; }
    return ToolMarkerStatus::NotMarker;
}

std::size_t find_tool_marker(std::string_view text, std::size_t search_from) noexcept {
    for (std::size_t index = search_from; index < text.size(); ++index) {
        if (text[index] != '<') { continue; }
        if (index + 1 < text.size()) {
            const char next = text[index + 1];
            if (next != 't' && next != 'f' && next != 'i') { continue; }
        }
        ToolOpenTag marker = {};
        if (classify_tool_marker_prefix(text.substr(index), marker) ==
            ToolMarkerStatus::Complete) {
            return index;
        }
    }
    return std::string_view::npos;
}

} // namespace ninfer::models::qwen3_5::frontend
