#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

// Single declarative authority for the Qwen tool wire syntax. The one-shot parser, the streaming
// decoder, marker discovery, and the value boundary lookahead all take their tag rules from here,
// so no consumer can accept bytes another one rejects. Policy decisions (declared-tool identity,
// tolerant recovery) stay out of this grammar; it answers only what the wire bytes say.
//
// Grammar (bytes, no escaping): format whitespace is space, tab, CR, LF.
//
//   wrapper      := "<tool_call>" | "<function_calls>"          (exact literal)
//   closer       := "</tool_call>" | "</function_calls>" | "</function>" | "</invoke>"
//                  | "</parameter>" | "</param>"                (exact literal, family-matched)
//   opener       := "<" keyword header ">"   keyword := function | invoke | parameter | param
//   header       := short | attribute attribute* | ""            (empty header is a bare opener)
//   short        := "=" value
//   attribute    := attr-name ws* "=" ws* value
//   attr-name    := name-char+
//   value        := quoted | unquoted
//   quoted       := '"' any* '"' | "'" any* "'"                  (no escape; ">" may appear)
//   unquoted     := char+ excluding format whitespace, ">", "/"  (empty is invalid)
//
// After an attribute value or the short value, only format whitespace or ">" may follow.
// A quoted value ends only at the matching quote; the first unquoted ">" ends the header.

enum class ToolTagKind : std::uint8_t {
    ToolCall,
    FunctionCalls,
    Function,
    Invoke,
    Parameter,
    Param,
};

enum class ToolHeaderStatus : std::uint8_t {
    NoMatch,   // the bytes at the position are not an opener of this family
    NeedMore,  // syntactically incomplete so far; more bytes could still complete a valid opener
    Complete,  // a syntactically complete opener (the name may be empty for a bare opener)
    Invalid,   // the header is definitely broken
};

enum class ToolMarkerStatus : std::uint8_t {
    NotMarker,  // cannot become a top-level tool-region marker; release as content
    NeedMore,   // still a possible marker prefix; hold the bytes
    Complete,   // a complete top-level marker (wrapper literal or opener header)
};

struct ToolOpenTag {
    ToolTagKind kind = ToolTagKind::Function;
    std::string_view name;   // name attribute / short value; empty for a bare opener
    std::size_t consumed     = 0; // bytes from the tag's '<' through the terminating '>'
};

[[nodiscard]] constexpr bool is_tool_format_whitespace(char byte) noexcept {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

[[nodiscard]] constexpr bool is_tool_name_char(char byte) noexcept {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
           (byte >= '0' && byte <= '9') || byte == '_' || byte == '-';
}

// Syntax-level name validation for function names. Whether the name is declared is a policy
// question answered by the tool contract, not here.
[[nodiscard]] constexpr bool is_valid_tool_name(std::string_view name, std::size_t max_name_length)
    noexcept {
    if (name.empty() || name.size() > max_name_length) { return false; }
    for (const char byte : name) {
        if (!is_tool_name_char(byte)) { return false; }
    }
    return true;
}

// Exact opener literal for a family. Wrapper families are literal-only; the tag families carry a
// header (see ToolHeaderStatus).
[[nodiscard]] constexpr std::string_view tool_open_literal(ToolTagKind kind) noexcept {
    switch (kind) {
        case ToolTagKind::ToolCall:      return "<tool_call>";
        case ToolTagKind::FunctionCalls: return "<function_calls>";
        default:                         return {};
    }
}

// Exact closer literal for a family; family must match the opener's family.
[[nodiscard]] constexpr std::string_view tool_close_literal(ToolTagKind kind) noexcept {
    switch (kind) {
        case ToolTagKind::ToolCall:      return "</tool_call>";
        case ToolTagKind::FunctionCalls: return "</function_calls>";
        case ToolTagKind::Function:      return "</function>";
        case ToolTagKind::Invoke:        return "</invoke>";
        case ToolTagKind::Parameter:     return "</parameter>";
        case ToolTagKind::Param:         return "</param>";
    }
    return {};
}

[[nodiscard]] constexpr bool is_tool_function_kind(ToolTagKind kind) noexcept {
    return kind == ToolTagKind::Function || kind == ToolTagKind::Invoke;
}

[[nodiscard]] constexpr bool is_tool_parameter_kind(ToolTagKind kind) noexcept {
    return kind == ToolTagKind::Parameter || kind == ToolTagKind::Param;
}

// F8 marker progression: a held marker candidate that the grammar classified as
// definitively NotMarker is split at its breaking (last) byte. A breaking '<' becomes the
// start of a fresh candidate; the failed bytes before it are ordinary content. Any other
// breaking byte flushes the whole candidate. The grammar decision is final: a '<' that the
// header grammar still accepts (a quoted value) keeps the candidate NeedMore and never
// reaches this split.
[[nodiscard]] constexpr std::size_t
failed_marker_candidate_retained(std::string_view candidate) noexcept {
    return candidate.size() >= 2 && candidate.back() == '<' ? 1 : 0;
}

// Parse the opener at `text[0] == '<'` for one family. Wrapper families match the exact literal
// (a strict prefix is NeedMore). Tag families parse the header; a quoted value may contain '>'
// and any other byte, an unterminated quote at the end of the input is NeedMore, and an empty
// unquoted value or missing attribute separator is Invalid. The name may be empty (bare opener).
[[nodiscard]] ToolHeaderStatus
parse_tool_open_header(std::string_view text, ToolTagKind kind, ToolOpenTag& out) noexcept;

// Parse a tag header body starting right after the tag keyword (no leading '<'). Used by the
// tolerant recovery path, which repairs only where the header starts and then defers to the
// canonical header rules. `out.consumed` counts from the start of `text`.
[[nodiscard]] ToolHeaderStatus
parse_tool_header_after_keyword(std::string_view text, ToolTagKind kind, ToolOpenTag& out)
    noexcept;

// Parse a function opener that may use either the `function` or the `invoke` family.
[[nodiscard]] ToolHeaderStatus parse_tool_function_open(std::string_view text, ToolOpenTag& out)
    noexcept;

// Parse a parameter opener that may use either the `parameter` or the `param` family.
[[nodiscard]] ToolHeaderStatus parse_tool_parameter_open(std::string_view text, ToolOpenTag& out)
    noexcept;

// Classify a top-level tool-region marker prefix. `text` starts at the marker's '<'.
// Complete: the full `<tool_call>` / `<function_calls>` literal or a syntactically complete
// function/invoke opener header (bare openers count; name validity is the region parser's job).
[[nodiscard]] ToolMarkerStatus
classify_tool_marker_prefix(std::string_view text, ToolOpenTag& out) noexcept;

// First position at or after `search_from` at which a complete top-level marker starts,
// or npos. With `wrapper_only`, only the wrapper literals qualify: recovery entries after
// a failed region that broke with a wrapper open (F7 entry-marker policy).
[[nodiscard]] std::size_t find_tool_marker(std::string_view text, std::size_t search_from = 0,
                                           bool wrapper_only = false)
    noexcept;

// R2-I4 (CR4): the single top-level entry classification. The region parser's Top state and the
// function-close continuation lookahead both consume this, so a complete next entry and its
// partial prefixes agree on the previous call's structural boundary. Wrapper-aware legality (a
// wrapper open inside an open wrapper is a nesting break, not an entry) is the caller's scope
// rule, not the entry classification.
enum class TopLevelEntry : std::uint8_t {
    End,       // the input ends (after format whitespace)
    NoEntry,   // a non-whitespace byte that is not a legal top-level entry
    NeedMore,  // a strict prefix of a legal top-level entry at the input end
    Entry,     // a complete legal top-level entry (the family in `out`)
};

struct TopLevelEntryInfo {
    TopLevelEntry kind = TopLevelEntry::NoEntry;
    ToolTagKind tag = ToolTagKind::Function;
    ToolOpenTag opener; // the parsed entry (function/invoke: name and consumed bytes)
};

// What a byte sequence at a top-level position is: a wrapper literal, a complete function/
// invoke opener, a strict prefix of either at the input end, the input end itself, or nothing.
[[nodiscard]] TopLevelEntryInfo classify_top_level_entry(std::string_view text,
                                                         std::size_t at) noexcept;

} // namespace ninfer::models::qwen3_5::frontend
