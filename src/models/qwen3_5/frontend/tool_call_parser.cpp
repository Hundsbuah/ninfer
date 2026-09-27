#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

using Json                = nlohmann::json;
using Contract            = ToolCallOutputContract;
using FallbackReason      = ToolCallParseFallbackReason;
using NormalizationPolicy = Contract::NormalizationPolicy;
using SchemaType          = Contract::SchemaType;
using TypeSet             = Contract::TypeSet;

constexpr std::string_view kToolOpen      = "<tool_call>";
constexpr std::string_view kToolClose     = "</tool_call>";

struct RawParameter {
    std::string_view name;
    std::string_view value;
};

struct RawToolCall {
    std::string_view name;
    std::vector<RawParameter> parameters;
};

enum class FunctionContainer : std::uint8_t {
    ToolCall,
    FunctionCalls,
    TopLevel,
};

enum class JsonValueKind : std::uint8_t {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Object,
    Array,
};

enum class ParameterNormalization : std::uint8_t {
    Emitted,
    Omitted,
    SchemaMismatch,
};

struct NormalizedParameter {
    ParameterNormalization disposition = ParameterNormalization::Emitted;
    std::string json_value;
};

constexpr bool is_format_whitespace(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

constexpr bool is_ascii_digit(char byte) { return byte >= '0' && byte <= '9'; }

constexpr bool is_ascii_alphanumeric(char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || is_ascii_digit(byte);
}

std::string_view trim_format_whitespace(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && is_format_whitespace(text[begin])) { ++begin; }
    std::size_t end = text.size();
    while (end > begin && is_format_whitespace(text[end - 1])) { --end; }
    return text.substr(begin, end - begin);
}

std::string rtrim_format_whitespace(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && is_format_whitespace(text[end - 1])) { --end; }
    return std::string(text.substr(0, end));
}

void skip_format_whitespace(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && is_format_whitespace(text[pos])) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

std::string_view unquote(std::string_view str) {
    str = trim_format_whitespace(str);
    if (str.size() >= 2) {
        if ((str.front() == '"' && str.back() == '"') ||
            (str.front() == '\'' && str.back() == '\'')) {
            return trim_format_whitespace(str.substr(1, str.size() - 2));
        }
    }
    return str;
}

// Quote-aware scan for the byte that terminates an opening tag: the first '>' outside a
// single- or double-quoted attribute or direct-name value. A quote is closed only by the
// same quote; there is no escape syntax (the wire format has none). Returns false when no
// unquoted '>' exists or EOF is reached with a quote still open - both are invalid headers.
bool find_tag_end(std::string_view text, std::size_t begin, std::size_t& tag_end) {
    char quote = '\0';
    for (std::size_t i = begin; i < text.size(); ++i) {
        const char byte = text[i];
        if (quote != '\0') {
            if (byte == quote) { quote = '\0'; }
            continue;
        }
        if (byte == '\'' || byte == '"') {
            quote = byte;
            continue;
        }
        if (byte == '>') {
            tag_end = i;
            return true;
        }
    }
    return false;
}

constexpr bool is_attr_name_start(char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || byte == '_';
}

constexpr bool is_attr_name_rest(char byte) {
    return is_attr_name_start(byte) || is_ascii_digit(byte) || byte == '.' || byte == ':' ||
           byte == '-';
}

enum class HeaderSyntax : std::uint8_t {
    DirectName,
    Attributes,
};

struct NameHeaderPayload {
    std::string_view name;
    HeaderSyntax syntax;
};

// Validates the complete payload between an opener prefix and its quote-aware terminating
// '>'. Every payload byte must belong to the syntax: a name that is found is not a header
// that is valid. The two accepted forms:
//
//   direct:   "=name" - the name may be quoted ("='name'", '="name"'), where the quote must
//             enclose the whole remainder; an unquoted name is the whole remainder, so a
//             declared unusual name ("=a<b") keeps working, and a fake opener that scans
//             through value data to a later ">" (="X\nif (a") yields a name that is not a
//             plausible identifier.
//   attrs:    whitespace-separated attributes "name="value""; values may be single- or
//             double-quoted (a quoted value may contain '>') or unquoted (up to whitespace);
//             an attribute without '=' or trailing junk makes the whole header invalid; the
//             first name= attribute supplies the name and must be non-empty.
bool parse_name_header_payload(std::string_view payload, NameHeaderPayload& parsed) {
    payload = trim_format_whitespace(payload);
    if (payload.starts_with('=')) {
        const std::string_view rest = payload.substr(1);
        if (rest.empty()) { return false; }
        if (rest.front() == '"' || rest.front() == '\'') {
            const char quote = rest.front();
            if (rest.size() < 3 || rest.back() != quote) { return false; }
            const std::string_view name =
                trim_format_whitespace(rest.substr(1, rest.size() - 2));
            if (name.empty()) { return false; }
            parsed.name   = name;
            parsed.syntax = HeaderSyntax::DirectName;
            return true;
        }
        parsed.name   = rest;
        parsed.syntax = HeaderSyntax::DirectName;
        return true;
    }

    std::string_view name;
    bool has_name = false;
    std::size_t i  = 0;
    while (i < payload.size()) {
        while (i < payload.size() && is_format_whitespace(payload[i])) { ++i; }
        if (i >= payload.size()) { break; }
        const std::size_t attr_begin = i;
        while (i < payload.size() && payload[i] != '=' && !is_format_whitespace(payload[i])) {
            ++i;
        }
        const std::string_view attr_name = payload.substr(attr_begin, i - attr_begin);
        if (attr_begin >= i || !is_attr_name_start(payload[attr_begin]) ||
            std::any_of(attr_name.begin() + 1, attr_name.end(),
                        [](char byte) { return !is_attr_name_rest(byte); })) {
            return false;
        }
        if (i >= payload.size() || payload[i] != '=') { return false; }
        ++i;
        skip_format_whitespace(payload, i);
        std::string_view value;
        if (i < payload.size() && (payload[i] == '"' || payload[i] == '\'')) {
            const char quote   = payload[i];
            const std::size_t q_start = i + 1;
            i = q_start;
            while (i < payload.size() && payload[i] != quote) { ++i; }
            if (i >= payload.size()) { return false; }
            value = payload.substr(q_start, i - q_start);
            ++i;
        } else {
            const std::size_t v_begin = i;
            while (i < payload.size() && !is_format_whitespace(payload[i])) { ++i; }
            value = payload.substr(v_begin, i - v_begin);
        }
        if (attr_name == "name" && !has_name) {
            if (value.empty()) { return false; }
            name     = value;
            has_name = true;
        }
    }
    if (!has_name) { return false; }
    parsed.name   = name;
    parsed.syntax = HeaderSyntax::Attributes;
    return true;
}

// The canonical parameter opener grammar shared by parameter parsing and structural
// boundary detection: "<parameter" or "<param", then a byte that delimits the prefix
// ('=', format whitespace, or '>'), then a header payload terminated by the quote-aware
// tag end and fully validated by parse_name_header_payload(). The byte after the prefix
// must delimit it: a longer identifier such as "<parameterX>" is not a parameter opener.
// Because the payload is validated as a whole, a fake opener that finds its ">" in value
// data (an attribute header carrying trailing source code, a direct name carrying markup)
// is rejected by this one grammar - so the two parser stages can never classify the same
// bytes differently.
struct ParameterHeader {
    std::size_t header_begin; // first byte after the opener prefix
    std::size_t tag_end;      // index of the header's quote-aware terminating '>'
    std::string_view name;    // the validated parameter name; empty when the header yields none
    std::string_view close_tag;
};

bool parse_parameter_header(std::string_view text, std::size_t pos, ParameterHeader& header) {
    std::size_t header_begin = 0;
    std::string_view close_tag;
    if (starts_with_at(text, pos, "<parameter")) {
        header_begin = pos + 10;
        close_tag    = "</parameter>";
    } else if (starts_with_at(text, pos, "<param")) {
        header_begin = pos + 6;
        close_tag    = "</param>";
    } else {
        return false;
    }
    if (header_begin >= text.size()) { return false; }
    const char separator = text[header_begin];
    if (separator != '=' && separator != '>' && !is_format_whitespace(separator)) {
        return false;
    }
    std::size_t tag_end = 0;
    if (!find_tag_end(text, header_begin, tag_end) || tag_end == header_begin) {
        return false;
    }
    NameHeaderPayload payload;
    if (!parse_name_header_payload(text.substr(header_begin, tag_end - header_begin), payload)) {
        return false;
    }
    header.header_begin = header_begin;
    header.tag_end      = tag_end;
    header.name         = payload.name;
    header.close_tag    = close_tag;
    return true;
}

bool is_param_open_at(std::string_view text, std::size_t pos, std::size_t& tag_end) {
    ParameterHeader header;
    if (!parse_parameter_header(text, pos, header)) { return false; }
    tag_end = header.tag_end;
    return true;
}

bool is_param_close_at(std::string_view text, std::size_t pos, std::size_t& tag_len) {
    if (starts_with_at(text, pos, "</parameter>")) {
        tag_len = 12;
        return true;
    }
    if (starts_with_at(text, pos, "</param>")) {
        tag_len = 8;
        return true;
    }
    return false;
}

enum class FunctionTagKind : std::uint8_t {
    Function,
    Invoke,
};

struct FunctionHeader {
    std::size_t header_begin; // first byte after the opener prefix
    std::size_t tag_end;      // index of the header's quote-aware terminating '>'
    std::string_view name;    // the function name; empty when the header yields none
    std::string_view close_tag;
    FunctionTagKind kind;
    bool recovered_missing_gt; // true when the missing-'> recovery supplied the tag end
};

// The payload stage shared by function parsing and structural boundary detection, mirroring
// parse_parameter_header(): the quote-aware tag end, then full payload validation by
// parse_name_header_payload(). A name that is found is not a header that is valid. The
// missing-'> recovery (tolerant mode) fires before the payload parse and supplies both the
// tag end and the name: the model sometimes drops the '>' after the function name (for
// example a name followed directly by a newline and a parameter tag), so it recovers by
// scanning the identifier run and accepting it when format whitespace separates it from the
// next '<' or end of region. A header that fails here fails in every stage alike.
bool parse_function_payload(std::string_view text, std::size_t header_begin,
                            std::size_t max_name_length, bool tolerant,
                            std::string_view close_tag, FunctionTagKind kind,
                            FunctionHeader& header) {
    std::size_t tag_end = 0;
    const bool found_gt = find_tag_end(text, header_begin, tag_end);
    header.header_begin = header_begin;
    header.close_tag    = close_tag;
    header.kind         = kind;
    header.recovered_missing_gt = false;
    header.name           = std::string_view{};
    header.tag_end        = tag_end;
    if (tolerant) {
        std::size_t scan = header_begin;
        while (scan < text.size() && text[scan] == '=') { ++scan; }
        const std::size_t ident_begin = scan;
        while (scan < text.size() && scan - header_begin < max_name_length) {
            const char byte = text[scan];
            if (!is_ascii_alphanumeric(byte) && byte != '_' && byte != '-') { break; }
            ++scan;
        }
        if (scan > ident_begin && scan < text.size() && is_format_whitespace(text[scan]) &&
            (!found_gt || scan < tag_end)) {
            std::size_t after = scan;
            while (after < text.size() && is_format_whitespace(text[after])) { ++after; }
            if (after >= text.size() || text[after] == '<') {
                header.tag_end               = scan;
                header.recovered_missing_gt  = true;
                header.name                  = text.substr(ident_begin, scan - ident_begin);
                return true;
            }
        }
    }
    if (!found_gt || tag_end == header_begin) { return false; }
    NameHeaderPayload payload;
    if (!parse_name_header_payload(text.substr(header_begin, tag_end - header_begin), payload)) {
        return false;
    }
    header.name = payload.name;
    return true;
}

// The strict opener boundary: the byte after "<function" or "<invoke>" must delimit the
// prefix ('=', format whitespace, or '>'). True when the text opens a function tag at pos;
// reports the prefix end, the matching close tag, and the tag kind.
bool function_opener_prefix_at(std::string_view text, std::size_t pos, std::size_t& prefix_end,
                               std::string_view& close_tag, FunctionTagKind& kind) {
    if (starts_with_at(text, pos, "<function")) {
        prefix_end = pos + 9;
        close_tag  = "</function>";
        kind       = FunctionTagKind::Function;
    } else if (starts_with_at(text, pos, "<invoke")) {
        prefix_end = pos + 7;
        close_tag  = "</invoke>";
        kind       = FunctionTagKind::Invoke;
    } else {
        return false;
    }
    if (prefix_end >= text.size()) { return false; }
    const char separator = text[prefix_end];
    return separator == '=' || separator == '>' || is_format_whitespace(separator);
}

// The canonical function opener grammar shared by function parsing and structural boundary
// detection: "<function" or "<invoke", then the strict opener boundary, then the validated
// payload of parse_function_payload(). A longer identifier such as "<functionbash>" is not
// a function opener, so the two parser stages can never classify the same bytes differently.
bool parse_function_header(std::string_view text, std::size_t pos, std::size_t max_name_length,
                           bool tolerant, FunctionHeader& header) {
    std::size_t prefix_end = 0;
    std::string_view close_tag;
    FunctionTagKind kind;
    if (!function_opener_prefix_at(text, pos, prefix_end, close_tag, kind)) { return false; }
    return parse_function_payload(text, prefix_end, max_name_length, tolerant, close_tag, kind,
                                  header);
}

static constexpr std::string_view kToolMarkers[] = {
    "<tool_call>",
    "<function_calls>",
    "<function=",
    "<function ",
    "<function>",
    "<function=\"",
    "<function=\'",
    "<invoke=",
    "<invoke ",
    "<invoke>",
    "<invoke=\"",
    "<invoke=\'",
};

bool is_prefix_of_any_marker(std::string_view prefix) {
    for (const auto& marker : kToolMarkers) {
        if (marker.starts_with(prefix)) { return true; }
    }
    return false;
}

bool matches_any_marker(std::string_view text) {
    for (const auto& marker : kToolMarkers) {
        if (text.starts_with(marker)) { return true; }
    }
    return false;
}

std::size_t find_first_tool_marker(std::string_view text) {
    std::size_t earliest = std::string_view::npos;
    for (const auto& marker : kToolMarkers) {
        std::size_t idx = text.find(marker);
        if (idx != std::string_view::npos && (earliest == std::string_view::npos || idx < earliest)) {
            earliest = idx;
        }
    }
    return earliest;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    return std::all_of(name.begin(), name.end(), [](char byte) {
        return is_ascii_alphanumeric(byte) || byte == '_' || byte == '-';
    });
}

constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

constexpr bool admits_type(TypeSet types, SchemaType type) {
    return (types.bits & type_bit(type)) != 0;
}

bool schema_type(std::string_view name, SchemaType& type) {
    if (name == "null") {
        type = SchemaType::Null;
    } else if (name == "boolean") {
        type = SchemaType::Boolean;
    } else if (name == "integer") {
        type = SchemaType::Integer;
    } else if (name == "number") {
        type = SchemaType::Number;
    } else if (name == "string") {
        type = SchemaType::String;
    } else if (name == "object") {
        type = SchemaType::Object;
    } else if (name == "array") {
        type = SchemaType::Array;
    } else {
        return false;
    }
    return true;
}

bool compile_direct_types(const Json& type_definition, TypeSet& types) {
    types = {};
    if (type_definition.is_string()) {
        SchemaType type;
        if (!schema_type(type_definition.get_ref<const std::string&>(), type)) { return false; }
        types.bits = type_bit(type);
        return true;
    }
    if (!type_definition.is_array() || type_definition.empty()) { return false; }
    for (const Json& member : type_definition) {
        if (!member.is_string()) { return false; }
        SchemaType type;
        if (!schema_type(member.get_ref<const std::string&>(), type)) { return false; }
        types.bits |= type_bit(type);
    }
    return types.bits != 0;
}

bool compile_schema_types(const Json& schema, TypeSet& types) {
    if (!schema.is_object()) { return false; }
    const auto direct = schema.find("type");
    if (direct != schema.end()) { return compile_direct_types(*direct, types); }

    const auto any_of     = schema.find("anyOf");
    const auto one_of     = schema.find("oneOf");
    const bool has_any_of = any_of != schema.end();
    const bool has_one_of = one_of != schema.end();
    if (has_any_of == has_one_of) { return false; }

    const Json& alternatives = has_any_of ? *any_of : *one_of;
    if (!alternatives.is_array() || alternatives.empty()) { return false; }

    TypeSet combined;
    for (const Json& alternative : alternatives) {
        TypeSet branch;
        if (!compile_schema_types(alternative, branch)) { return false; }
        combined.bits |= branch.bits;
    }
    if (combined.bits == 0) { return false; }
    types = combined;
    return true;
}

Contract::Tool compile_tool_contract(const Json& definition) {
    Contract::Tool contract;
    if (!definition.is_object()) { return contract; }
    const auto function = definition.find("function");
    if (function == definition.end() || !function->is_object()) { return contract; }
    const auto name = function->find("name");
    if (name == function->end() || !name->is_string()) { return contract; }
    contract.name = name->get<std::string>();

    const auto schema = function->find("parameters");
    if (schema == function->end() || !schema->is_object()) { return contract; }
    const auto properties = schema->find("properties");
    if (properties == schema->end() || !properties->is_object()) { return contract; }

    contract.parameters.reserve(properties->size());
    for (const auto& [parameter_name, property] : properties->items()) {
        Contract::Parameter parameter;
        parameter.name = parameter_name;
        if (compile_schema_types(property, parameter.types)) {
            parameter.policy = NormalizationPolicy::DeclaredTypes;
        }
        contract.parameters.push_back(std::move(parameter));
    }
    return contract;
}

bool same_contract(const Contract::Tool& lhs, const Contract::Tool& rhs) {
    if (lhs.parameters.size() != rhs.parameters.size()) { return false; }
    for (std::size_t i = 0; i < lhs.parameters.size(); ++i) {
        const Contract::Parameter& left  = lhs.parameters[i];
        const Contract::Parameter& right = rhs.parameters[i];
        if (left.name != right.name || left.policy != right.policy ||
            left.types.bits != right.types.bits) {
            return false;
        }
    }
    return true;
}

void append_tool_contract(Contract& contracts, const Json& definition) {
    Contract::Tool compiled = compile_tool_contract(definition);
    if (compiled.name.empty()) { return; }
    const auto existing =
        std::find_if(contracts.tools.begin(), contracts.tools.end(),
                     [&](const auto& tool) { return tool.name == compiled.name; });
    if (existing == contracts.tools.end()) {
        contracts.tools.push_back(std::move(compiled));
        return;
    }
    if (existing->unambiguous && !same_contract(*existing, compiled)) {
        existing->parameters.clear();
        existing->unambiguous = false;
    }
}

const Contract::Tool* find_tool_contract(const Contract& contract, std::string_view tool_name) {
    const auto tool =
        std::find_if(contract.tools.begin(), contract.tools.end(),
                     [&](const auto& candidate) { return candidate.name == tool_name; });
    return tool == contract.tools.end() ? nullptr : &*tool;
}

const Contract::Parameter* find_parameter_contract(const Contract::Tool& tool,
                                                   std::string_view parameter_name) {
    const auto parameter =
        std::find_if(tool.parameters.begin(), tool.parameters.end(),
                     [&](const auto& candidate) { return candidate.name == parameter_name; });
    return parameter == tool.parameters.end() ? nullptr : &*parameter;
}

bool is_declared_parameter(const Contract& contract, std::string_view tool_name,
                           std::string_view parameter_name) {
    const Contract::Tool* tool = find_tool_contract(contract, tool_name);
    if (tool == nullptr || !tool->unambiguous) { return false; }
    return find_parameter_contract(*tool, parameter_name) != nullptr;
}

bool is_declared_string_parameter(const Contract& contract, std::string_view tool_name,
                                  std::string_view parameter_name) {
    const Contract::Tool* tool = find_tool_contract(contract, tool_name);
    if (tool == nullptr || !tool->unambiguous) { return false; }

    const Contract::Parameter* parameter = find_parameter_contract(*tool, parameter_name);
    return parameter != nullptr &&
           parameter->policy == NormalizationPolicy::DeclaredTypes &&
           admits_type(parameter->types, SchemaType::String);
}

// Fallback trust policy for undeclared candidate siblings. After a literal close, a
// following parameter opener is structural only when the current tool contract declares
// its name (trusted by definition). An undeclared name is trusted only as a fallback,
// and only while it is still a plausible ordinary parameter identifier - the form the
// official chat templates emit as "<parameter=<name>". A name carrying markup, format
// whitespace, control bytes, or other non-identifier bytes can only be the product of a
// fake opener that found its first ">" in value data, so it terminates nothing.
bool is_ordinary_parameter_name(std::string_view name) {
    if (name.empty()) { return false; }
    for (const char byte : name) {
        const unsigned char c = static_cast<unsigned char>(byte);
        const bool identifier =
            (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
            c == '_' || c == '.' || c == '-' || c == ':';
        if (!identifier) { return false; }
    }
    return true;
}

std::string_view remove_parameter_framing_newlines(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end   = text.size();
    if (text.starts_with("\r\n")) {
        begin = 2;
    } else if (text.starts_with('\n')) {
        begin = 1;
    }
    if (end >= begin + 2 && text.substr(end - 2, 2) == "\r\n") {
        end -= 2;
    } else if (end > begin && text[end - 1] == '\n') {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool ascii_case_equal(std::string_view text, std::string_view lowercase) {
    if (text.size() != lowercase.size()) { return false; }
    for (std::size_t i = 0; i < text.size(); ++i) {
        char byte = text[i];
        if (byte >= 'A' && byte <= 'Z') { byte = static_cast<char>(byte + ('a' - 'A')); }
        if (byte != lowercase[i]) { return false; }
    }
    return true;
}

bool json_number_is_integer(std::string_view number) {
    std::size_t pos = number.starts_with('-') ? 1 : 0;
    if (pos >= number.size()) { return false; }

    const std::size_t integer_begin = pos;
    while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
    const std::size_t integer_end = pos;

    std::size_t fraction_begin = pos;
    std::size_t fraction_end   = pos;
    if (pos < number.size() && number[pos] == '.') {
        fraction_begin = ++pos;
        while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
        fraction_end = pos;
    }

    bool exponent_negative     = false;
    std::size_t exponent_value = 0;
    if (pos < number.size() && (number[pos] == 'e' || number[pos] == 'E')) {
        ++pos;
        if (pos < number.size() && (number[pos] == '+' || number[pos] == '-')) {
            exponent_negative = number[pos] == '-';
            ++pos;
        }
        const std::size_t cap = number.size();
        while (pos < number.size() && is_ascii_digit(number[pos])) {
            const std::size_t digit = static_cast<std::size_t>(number[pos] - '0');
            if (exponent_value != cap) {
                if (exponent_value > cap / 10 || (exponent_value == cap / 10 && digit > cap % 10)) {
                    exponent_value = cap;
                } else {
                    exponent_value = exponent_value * 10 + digit;
                }
            }
            ++pos;
        }
    }
    if (integer_begin == integer_end || pos != number.size()) { return false; }

    bool coefficient_is_zero   = true;
    std::size_t trailing_zeros = 0;
    const auto observe_digit   = [&](char digit) {
        if (digit == '0') {
            ++trailing_zeros;
        } else {
            coefficient_is_zero = false;
            trailing_zeros      = 0;
        }
    };
    for (std::size_t i = integer_begin; i < integer_end; ++i) { observe_digit(number[i]); }
    for (std::size_t i = fraction_begin; i < fraction_end; ++i) { observe_digit(number[i]); }
    if (coefficient_is_zero) { return true; }

    const std::size_t fraction_digits = fraction_end - fraction_begin;
    if (!exponent_negative) {
        if (exponent_value >= fraction_digits) { return true; }
        return fraction_digits - exponent_value <= trailing_zeros;
    }
    if (exponent_value > trailing_zeros) { return false; }
    return fraction_digits <= trailing_zeros - exponent_value;
}

bool classify_json_value(std::string_view value, JsonValueKind& kind) {
    if (value.empty() || !Json::accept(value.begin(), value.end())) { return false; }
    switch (value.front()) {
    case 'n':
        kind = JsonValueKind::Null;
        return true;
    case 't':
    case 'f':
        kind = JsonValueKind::Boolean;
        return true;
    case '"':
        kind = JsonValueKind::String;
        return true;
    case '{':
        kind = JsonValueKind::Object;
        return true;
    case '[':
        kind = JsonValueKind::Array;
        return true;
    default:
        if (value.front() == '-' || is_ascii_digit(value.front())) {
            kind = json_number_is_integer(value) ? JsonValueKind::Integer : JsonValueKind::Number;
            return true;
        }
        return false;
    }
}

bool admits_value(TypeSet types, JsonValueKind kind) {
    switch (kind) {
    case JsonValueKind::Null:
        return admits_type(types, SchemaType::Null);
    case JsonValueKind::Boolean:
        return admits_type(types, SchemaType::Boolean);
    case JsonValueKind::Integer:
        return admits_type(types, SchemaType::Integer) || admits_type(types, SchemaType::Number);
    case JsonValueKind::Number:
        return admits_type(types, SchemaType::Number);
    case JsonValueKind::String:
        return admits_type(types, SchemaType::String);
    case JsonValueKind::Object:
        return admits_type(types, SchemaType::Object);
    case JsonValueKind::Array:
        return admits_type(types, SchemaType::Array);
    }
    return false;
}

std::string encode_json_string(std::string_view value) { return Json(std::string(value)).dump(); }

NormalizedParameter normalize_declared_parameter(std::string_view encoded_value, TypeSet types) {
    const std::string_view framed = remove_parameter_framing_newlines(encoded_value);
    if (admits_type(types, SchemaType::String)) {
        return {.json_value = encode_json_string(framed)};
    }

    const std::string_view value = trim_format_whitespace(framed);
    if (value.empty()) { return {.disposition = ParameterNormalization::Omitted}; }

    JsonValueKind kind;
    if (classify_json_value(value, kind)) {
        return {.disposition = admits_value(types, kind) ? ParameterNormalization::Emitted
                                                         : ParameterNormalization::SchemaMismatch,
                .json_value  = std::string(value)};
    }

    if (admits_type(types, SchemaType::Boolean)) {
        if (ascii_case_equal(value, "true")) { return {.json_value = "true"}; }
        if (ascii_case_equal(value, "false")) { return {.json_value = "false"}; }
    }
    return {.disposition = ParameterNormalization::SchemaMismatch,
            .json_value  = encode_json_string(framed)};
}

NormalizedParameter normalize_parameter(std::string_view encoded_value,
                                        const Contract::Parameter* parameter) {
    if (parameter != nullptr && parameter->policy == NormalizationPolicy::DeclaredTypes) {
        return normalize_declared_parameter(encoded_value, parameter->types);
    }

    const std::string_view value = trim_format_whitespace(encoded_value);
    if (Json::accept(value.begin(), value.end())) { return {.json_value = std::string(value)}; }
    return {.json_value = encode_json_string(value)};
}

class QwenToolRegionParser {
public:
    QwenToolRegionParser(std::string_view text, std::size_t max_name_length,
                         const Contract& contract, bool tolerant)
        : text_(text), max_name_length_(max_name_length), contract_(contract),
          tolerant_(tolerant) {}

    [[nodiscard]] std::uint32_t duplicate_parameters_repaired() const noexcept {
        return duplicate_parameters_repaired_;
    }

    FallbackReason parse(std::vector<RawToolCall>& calls) {
        std::size_t pos = 0;
        for (;;) {
            skip_format_whitespace(text_, pos);
            if (pos == text_.size()) {
                return calls.empty() ? FallbackReason::MalformedStructure : FallbackReason::None;
            }
            if (starts_with_at(text_, pos, "<tool_call>")) {
                RawToolCall call;
                const FallbackReason terminal = finish_call(calls, call, parse_tool_call(pos, call));
                if (terminal != FallbackReason::None) { return terminal; }
            } else if (starts_with_at(text_, pos, "<function_calls>")) {
                pos += 16;
                bool had_calls = false;
                for (;;) {
                    skip_format_whitespace(text_, pos);
                    if (consume(pos, "</function_calls>")) { break; }
                    if (pos == text_.size()) {
                        // Tolerant: an unclosed wrapper after one or more complete calls is a
                        // truncation; the strict parser keeps the hard structural failure.
                        if (tolerant_ && had_calls) { return FallbackReason::TruncatedTail; }
                        return FallbackReason::MalformedStructure;
                    }
                    RawToolCall call;
                    const FallbackReason terminal = finish_call(
                        calls, call, parse_function(pos, call, FunctionContainer::FunctionCalls));
                    if (terminal != FallbackReason::None) { return terminal; }
                    had_calls = true;
                }
                if (!had_calls) { return FallbackReason::MalformedStructure; }
            } else if (starts_with_at(text_, pos, "<function") || starts_with_at(text_, pos, "<invoke")) {
                RawToolCall call;
                const FallbackReason terminal = finish_call(
                    calls, call, parse_function(pos, call, FunctionContainer::TopLevel));
                if (terminal != FallbackReason::None) { return terminal; }
            } else {
                // Tolerant: a trailing suffix after one or more complete calls is discarded
                // rather than failing the whole output.
                if (tolerant_ && !calls.empty()) { return FallbackReason::TruncatedTail; }
                return calls.empty() ? FallbackReason::MalformedStructure
                                     : FallbackReason::TrailingContent;
            }
        }
    }

private:
    bool consume(std::size_t& pos, std::string_view token) const {
        if (!starts_with_at(text_, pos, token)) { return false; }
        pos += token.size();
        return true;
    }

    // True when nothing but trailing whitespace remains after `pos` in the tool region.
    bool at_region_end(std::size_t pos) const {
        std::size_t at = pos;
        skip_format_whitespace(text_, at);
        return at == text_.size();
    }

    // Applies tolerant recovery to a sub-parse result and reports whether parse() should
    // terminate. In tolerant mode, a malformed suffix after one or more complete calls is
    // discarded, and a single truncated final call whose name and at least one parameter are
    // complete is retained; the recovered calls are never demoted to text. The strict parser
    // returns the raw failure unchanged.
    FallbackReason finish_call(std::vector<RawToolCall>& calls, RawToolCall& call,
                               FallbackReason failure) {
        if (failure == FallbackReason::None) {
            calls.push_back(std::move(call));
            return FallbackReason::None;
        }
        if (tolerant_ && !calls.empty()) { return FallbackReason::TruncatedTail; }
        if (tolerant_ && failure == FallbackReason::TruncatedTail && calls.empty() &&
            !call.parameters.empty()) {
            calls.push_back(std::move(call));
            return FallbackReason::TruncatedTail;
        }
        return failure;
    }

    FallbackReason parse_tool_call(std::size_t& pos, RawToolCall& call) {
        if (!consume(pos, "<tool_call>")) { return FallbackReason::MalformedStructure; }
        skip_format_whitespace(text_, pos);
        const FallbackReason failure =
            parse_function(pos, call, FunctionContainer::ToolCall);
        if (failure != FallbackReason::None) { return failure; }
        skip_format_whitespace(text_, pos);
        if (consume(pos, "</tool_call>")) { return FallbackReason::None; }
        // Tolerant: a complete call may be followed by explanatory text, or the model may have
        // stopped at the end of its budget before the closing tag. parse() resolves the terminal
        // flag; the strict parser keeps the hard structural failure.
        return tolerant_ ? FallbackReason::TruncatedTail : FallbackReason::MalformedStructure;
    }

    FallbackReason parse_function(std::size_t& pos, RawToolCall& call,
                                  FunctionContainer container) {
        FunctionHeader header;
        std::string_view fn_close;
        FunctionTagKind kind;
        std::size_t prefix_end = 0;
        if (function_opener_prefix_at(text_, pos, prefix_end, fn_close, kind)) {
            // The same canonical opener grammar the structural boundary detection uses: a byte
            // sequence that is not a function opener here is not one there either.
            if (!parse_function_payload(text_, prefix_end, max_name_length_, tolerant_, fn_close,
                                        kind, header)) {
                return FallbackReason::InvalidToolName;
            }
        } else if (tolerant_) {
            // Recover a malformed function opener: a dropped or doubled leading '<', a leaked
            // ChatML turn marker, or a dropped 'function'/'invoke' keyword, followed by '=' or a
            // name. A form that yields no valid name is rejected by the payload stage below, so
            // prose after a marker cannot pass.
            std::size_t scan = pos;
            while (scan < text_.size() && text_[scan] == '<') { ++scan; }
            if (starts_with_at(text_, scan, "|im_start|>")) { scan += 11; }
            std::size_t kw_len = 0;
            if (starts_with_at(text_, scan, "function")) { kw_len = 8; }
            else if (starts_with_at(text_, scan, "invoke")) { kw_len = 6; }
            else { return FallbackReason::MalformedStructure; }
            scan += kw_len;
            if (scan >= text_.size() || (text_[scan] != '=' && !is_format_whitespace(text_[scan]))) {
                return FallbackReason::MalformedStructure;
            }
            if (text_[scan] == '=') { ++scan; }
            prefix_end = scan;
            fn_close   = kw_len == 8 ? "</function>" : "</invoke>";
            kind       = kw_len == 8 ? FunctionTagKind::Function : FunctionTagKind::Invoke;
            if (!parse_function_payload(text_, prefix_end, max_name_length_, tolerant_, fn_close,
                                        kind, header)) {
                return FallbackReason::InvalidToolName;
            }
        } else {
            return FallbackReason::MalformedStructure;
        }
        call.name = header.name;
        if (!valid_function_name(call.name, max_name_length_)) {
            return FallbackReason::InvalidToolName;
        }
        // Strict mode rejects a name outside the declared tool set. Tolerant mode keeps an
        // otherwise well-formed call structured and leaves the identity judgment to the consumer:
        // leaking the raw region to content would turn a valid call into prose.
        if (!tolerant_ && contract_.enforce_declared_names &&
            find_tool_contract(contract_, call.name) == nullptr) {
            return FallbackReason::UndeclaredTool;
        }
        pos = header.recovered_missing_gt ? header.tag_end : header.tag_end + 1;

        for (;;) {
            skip_format_whitespace(text_, pos);
            if (consume(pos, fn_close)) {
                return FallbackReason::None;
            }
            // Tolerant: the region is exhausted after the last complete parameter, so a missing
            // function close is a truncation, not a malformed structure. parse() retains the
            // recovered parameters; the strict parser still requires the closing tag.
            if (tolerant_ && at_region_end(pos)) {
                return FallbackReason::TruncatedTail;
            }
            const FallbackReason failure = parse_parameter(pos, call, fn_close, container);
            if (failure != FallbackReason::None) { return failure; }
        }
    }

    FallbackReason parse_parameter(std::size_t& pos, RawToolCall& call,
                                   std::string_view fn_close, FunctionContainer container) {
        // The same canonical opener grammar the structural boundary detection uses: a byte
        // sequence that is not a parameter opener here is not one there either.
        ParameterHeader header;
        if (!parse_parameter_header(text_, pos, header)) {
            return FallbackReason::MalformedStructure;
        }
        if (header.name.empty()) {
            return FallbackReason::MalformedStructure;
        }
        const std::string_view name = header.name;

        const bool opaque_string =
            is_declared_string_parameter(contract_, call.name, name);
        const std::size_t value_begin = header.tag_end + 1;
        std::size_t value_end         = 0;
        std::size_t close_len         = 0;
        if (!find_parameter_close(value_begin, value_end, close_len, header.close_tag, fn_close,
                                  container, opaque_string, call.name)) {
            if (tolerant_) {
                // Tolerant: the region ends before the closing tag, so the output budget cut the
                // parameter value. Keep the value up to the cut (last occurrence wins, as with a
                // complete parameter) and flag the tail; the strict parser keeps the hard
                // structural failure.
                const std::string_view partial = text_.substr(value_begin, text_.size() - value_begin);
                const auto existing = std::find_if(call.parameters.begin(), call.parameters.end(),
                                                   [&](const RawParameter& candidate) { return candidate.name == name; });
                if (existing != call.parameters.end()) {
                    existing->value = partial;
                    ++duplicate_parameters_repaired_;
                } else {
                    call.parameters.push_back(RawParameter{.name = name, .value = partial});
                }
                pos = text_.size();
                return FallbackReason::TruncatedTail;
            }
            return FallbackReason::MalformedStructure;
        }
        const std::string_view value = text_.substr(value_begin, value_end - value_begin);

        const auto existing =
            std::find_if(call.parameters.begin(), call.parameters.end(),
                         [&](const RawParameter& candidate) { return candidate.name == name; });

        // Last occurrence wins, as it would in JSON object syntax, rather than discarding an
        // otherwise well-formed call.
        if (existing != call.parameters.end()) {
            existing->value = value;
            ++duplicate_parameters_repaired_;
        } else {
            call.parameters.push_back(RawParameter{.name = name, .value = value});
        }
        pos = value_end + close_len;
        return FallbackReason::None;
    }

    // The same header grammar parse_function() consumes, applied to a boundary candidate:
    // a function opener that does not parse as a header here does not parse in the consumer
    // either. Tolerant mode accepts what the tolerant consumer accepts (undeclared names,
    // the missing-'> recovery); strict mode adds name validity and the declared-tool policy.
    bool has_valid_function_header(std::size_t pos, FunctionHeader& header) const {
        if (!parse_function_header(text_, pos, max_name_length_, tolerant_, header)) {
            return false;
        }
        if (!tolerant_) {
            if (!valid_function_name(header.name, max_name_length_)) { return false; }
            if (contract_.enforce_declared_names && find_tool_contract(contract_, header.name) ==
                                                     nullptr) {
                return false;
            }
        }
        return true;
    }

    // A valid continuation after a completed function in this container. The ToolCall
    // container additionally requires that what follows the wrapper close is EOF, a valid
    // top-level construct, or prose carrying no further "</tool_call>": a later wrapper
    // close that the prose does not belong to is the boundary of a fake call embedded in
    // value data, and accepting it would shadow a later, more complete close.
    bool has_structural_function_successor(std::size_t pos,
                                           FunctionContainer container) const {
        skip_format_whitespace(text_, pos);
        switch (container) {
        case FunctionContainer::ToolCall: {
            if (!starts_with_at(text_, pos, kToolClose)) { return false; }
            const std::size_t after = pos + kToolClose.size();
            std::size_t at = after;
            skip_format_whitespace(text_, at);
            if (at == text_.size()) { return true; }
            if (starts_with_at(text_, at, "<tool_call>") ||
                starts_with_at(text_, at, "<function_calls>")) {
                return true;
            }
            FunctionHeader header;
            if (has_valid_function_header(at, header)) { return true; }
            return text_.find(kToolClose, after) == std::string_view::npos;
        }
        case FunctionContainer::FunctionCalls: {
            if (starts_with_at(text_, pos, "</function_calls>")) { return true; }
            FunctionHeader header;
            return has_valid_function_header(pos, header);
        }
        case FunctionContainer::TopLevel: {
            if (pos == text_.size()) { return true; }
            if (starts_with_at(text_, pos, "<tool_call>") ||
                starts_with_at(text_, pos, "<function_calls>")) {
                return true;
            }
            FunctionHeader header;
            return has_valid_function_header(pos, header);
        }
        }
        return false;
    }

    bool has_structural_parameter_successor(std::size_t pos, std::string_view fn_close,
                                            FunctionContainer container, bool relaxed_tolerant,
                                            std::string_view fn_name) const {
        skip_format_whitespace(text_, pos);
        if (starts_with_at(text_, pos, fn_close)) {
            if (has_structural_function_successor(pos + fn_close.size(), container)) {
                return true;
            }
            if (!relaxed_tolerant) { return false; }
            // Tolerant: the close is still a boundary when the region ends after the function
            // close or only tokens remain that the container discards (a stray parameter open,
            // plain prose). A marker the container does not expect there (a
            // function/invoke/tool_call/function_calls open inside a wrapper) is the closing
            // boundary of an embedded fake call in opaque data; accepting it would shadow a
            // later, more complete close.
            std::size_t after = pos + fn_close.size();
            skip_format_whitespace(text_, after);
            if (after == text_.size()) { return true; }
            std::size_t tag_end = 0;
            if (is_param_open_at(text_, after, tag_end)) { return true; }
            return !matches_any_marker(text_.substr(after));
        }

        // A complete last parameter at EOF is useful only to tolerant recovery, where
        // parse_function() will classify the missing function close as a truncated tail.
        if (relaxed_tolerant && pos == text_.size()) { return true; }

        // The same canonical opener grammar parse_parameter() consumes: both stages must
        // classify a byte sequence as a parameter opener or as none alike.
        ParameterHeader header;
        if (!parse_parameter_header(text_, pos, header)) { return false; }
        if (header.name.empty()) { return false; }
        // A genuine Qwen parameter header is "<parameter=<name>": the official chat
        // templates emit the declared name verbatim between "=" and the first ">". A
        // candidate whose name the current unambiguous tool contract does not declare
        // therefore establishes a structural boundary only while the name is still a
        // plausible ordinary parameter identifier. A name that is not - embedded
        // markup, format whitespace, control bytes - may only be the product of a fake
        // opener scanning through value data to a later ">" (a comparison operator, a
        // shell redirection, ...), so the preceding literal close remains value data.
        // Declared names are not subject to the plausibility grammar: NInfer does not
        // restrict declared names to a grammar that excludes markup, so the contract
        // alone decides that case.
        return is_declared_parameter(contract_, fn_name, header.name) ||
               is_ordinary_parameter_name(header.name);
    }

    bool find_opaque_parameter_close(std::size_t value_begin, std::size_t& value_end,
                                     std::size_t& close_len, std::string_view required_close,
                                     std::string_view fn_close, FunctionContainer container,
                                     bool relaxed_tolerant, std::string_view fn_name) const {
        std::size_t scan = value_begin;
        for (;;) {
            const std::size_t candidate = text_.find(required_close, scan);
            if (candidate == std::string_view::npos) { return false; }
            const std::size_t after = candidate + required_close.size();
            if (has_structural_parameter_successor(after, fn_close, container, relaxed_tolerant,
                                                   fn_name)) {
                value_end = candidate;
                close_len = required_close.size();
                return true;
            }
            scan = after;
        }
    }

    bool find_parameter_close(std::size_t value_begin, std::size_t& value_end,
                              std::size_t& close_len, std::string_view required_close,
                              std::string_view fn_close, FunctionContainer container,
                              bool opaque_string, std::string_view fn_name) const {
        if (opaque_string) {
            // Declared string parameters are opaque data. Parameter elements are siblings in the
            // Qwen grammar, not recursively nested elements, so a literal "<parameter...>" inside
            // source code must not increase structural depth. Conversely, a literal
            // "</parameter>" is data unless what follows can continue the enclosing function.
            //
            // The wire format is still intrinsically ambiguous if user data itself contains an
            // exact complete closing boundary: no delimiter parser can distinguish that byte
            // sequence without escaping or length-prefixing, so the value ends at the first
            // such boundary and the following bytes may then be parsed as further parameters
            // or calls without any fallback reason. Restricting this recovery to
            // schema-declared strings avoids changing legacy/non-string parsing.
            // Prefer a fully structural boundary. Only when that is impossible does tolerant mode
            // relax the suffix check: a function close still counts when the region ends or only
            // discarded tokens follow it, but never when a wrapper-foreign marker follows.
            if (find_opaque_parameter_close(value_begin, value_end, close_len, required_close,
                                            fn_close, container,
                                            /*relaxed_tolerant*/ false, fn_name)) {
                return true;
            }
            return tolerant_ &&
                   find_opaque_parameter_close(value_begin, value_end, close_len, required_close,
                                               fn_close, container,
                                               /*relaxed_tolerant*/ true, fn_name);
        }

        // Legacy and non-string values keep the historical balanced-marker behavior.
        std::size_t depth = 1;
        std::size_t scan  = value_begin;
        while (scan < text_.size()) {
            if (starts_with_at(text_, scan, required_close)) {
                --depth;
                if (depth == 0) {
                    value_end = scan;
                    close_len = required_close.size();
                    return true;
                }
                scan += required_close.size();
                continue;
            }
            std::size_t open_tag_end = 0;
            if (is_param_open_at(text_, scan, open_tag_end)) {
                ++depth;
                scan = open_tag_end + 1;
                continue;
            }
            ++scan;
        }
        return false;
    }

    std::string_view text_;
    std::size_t max_name_length_;
    const Contract& contract_;
    std::uint32_t duplicate_parameters_repaired_ = 0;
    bool tolerant_ = false;
};

GeneratedToolCall normalize_raw_tool_call(const RawToolCall& raw, const Contract& contract,
                                          ToolCallParseDiagnostics& diagnostics) {
    const Contract::Tool* tool = find_tool_contract(contract, raw.name);
    if (tool != nullptr && !tool->unambiguous) { tool = nullptr; }

    std::string arguments = "{";
    bool first            = true;
    for (const RawParameter& raw_parameter : raw.parameters) {
        const Contract::Parameter* parameter =
            tool == nullptr ? nullptr : find_parameter_contract(*tool, raw_parameter.name);
        NormalizedParameter normalized = normalize_parameter(raw_parameter.value, parameter);
        if (tool != nullptr && parameter == nullptr) {
            normalized.disposition = ParameterNormalization::SchemaMismatch;
        }
        if (normalized.disposition == ParameterNormalization::Omitted) {
            ++diagnostics.empty_arguments_omitted;
            continue;
        }
        if (normalized.disposition == ParameterNormalization::SchemaMismatch) {
            ++diagnostics.schema_mismatch_arguments;
        }

        if (!first) { arguments.push_back(','); }
        first = false;
        arguments += encode_json_string(raw_parameter.name);
        arguments.push_back(':');
        arguments += normalized.json_value;
    }
    arguments.push_back('}');

    return GeneratedToolCall{.name = std::string(raw.name), .arguments_json = std::move(arguments)};
}

ParsedToolCallOutput fallback(const std::string& text, ToolCallParseDiagnostics diagnostics = {}) {
    ParsedToolCallOutput out;
    out.content     = text;
    out.diagnostics = diagnostics;
    return out;
}

} // namespace

std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled) {
    if (!enabled) { return {}; }
    auto contract                    = std::make_shared<ToolCallOutputContract>();
    contract->enforce_declared_names = true;
    contract->tools.reserve(tool_jsons.size());
    for (const std::string& tool_json : tool_jsons) {
        const Json definition = Json::parse(tool_json, nullptr, false);
        if (!definition.is_discarded()) { append_tool_contract(*contract, definition); }
    }
    return contract;
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolCallOutputContract& contract,
                                                 bool tolerant) {
    const std::string_view source(text);
    std::size_t candidate = find_first_tool_marker(source);
    if (candidate == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.diagnostics.marker_seen = true;

    // Generated prose can quote a tool-call marker before the real turn. Try the first marker, then
    // each later `<tool_call>` wrapper, and accept the first region that parses; earlier markers
    // stay ordinary content. A truncated tail that still kept a complete call (tolerant mode) is a
    // recovered region, not a failure.
    std::vector<RawToolCall> raw_calls;
    std::size_t accepted                   = std::string::npos;
    FallbackReason accepted_reason         = FallbackReason::None;
    std::uint32_t duplicate_repairs        = 0;
    FallbackReason first_failure           = FallbackReason::MalformedStructure;
    bool first_failure_recorded            = false;
    while (candidate != std::string::npos) {
        std::vector<RawToolCall> calls;
        QwenToolRegionParser parser(source.substr(candidate), max_tool_name_length, contract,
                                    tolerant);
        const FallbackReason failure = parser.parse(calls);
        if (failure == FallbackReason::None ||
            (failure == FallbackReason::TruncatedTail && !calls.empty())) {
            accepted          = candidate;
            accepted_reason   = failure;
            duplicate_repairs = parser.duplicate_parameters_repaired();
            raw_calls         = std::move(calls);
            break;
        }
        if (!first_failure_recorded) {
            first_failure          = failure;
            first_failure_recorded = true;
        }
        // Retries move only to a later `<tool_call>` wrapper: the markup nested inside a failed
        // region (its `<function=...>` or `<invoke>`) must not re-read a truncated call.
        candidate = text.find(kToolOpen, candidate + 1);
    }
    if (accepted == std::string::npos) {
        // No region parsed. A truncated tail that kept no call carries no arguments either, so
        // the response is returned as text with the first region's reason recorded.
        out.diagnostics.fallback_reason = first_failure;
        return fallback(text, out.diagnostics);
    }
    // A recovered truncated tail keeps its reason for transparency without demoting the output.
    out.diagnostics.fallback_reason = accepted_reason;

    out.content = rtrim_format_whitespace(source.substr(0, accepted));
    out.tool_calls.reserve(raw_calls.size());
    for (const RawToolCall& raw : raw_calls) {
        out.tool_calls.push_back(normalize_raw_tool_call(raw, contract, out.diagnostics));
    }

    out.diagnostics.duplicate_parameters_repaired = duplicate_repairs;
    out.diagnostics.structured_call_count = static_cast<std::uint32_t>(out.tool_calls.size());
    out.is_tool_call_response             = true;
    return out;
}

ToolCallOutputDecoder::ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                                             std::size_t max_tool_name_length, bool tolerant)
    : contract_(std::move(contract)), max_tool_name_length_(max_tool_name_length),
      tolerant_(tolerant) {}

std::string ToolCallOutputDecoder::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    if (text.empty()) { return {}; }
    if (!contract_) { return std::string(text); }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    std::string visible;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char byte = text[index];
        if (!pending_tag_.empty()) {
            pending_tag_.push_back(byte);
            if (matches_any_marker(pending_tag_)) {
                tool_region_ = std::move(trailing_whitespace_);
                trailing_whitespace_.clear();
                tool_region_.append(pending_tag_);
                pending_tag_.clear();
                tool_region_.append(text.substr(index + 1));
                saw_tool_marker_ = true;
                break;
            }
            if (!is_prefix_of_any_marker(pending_tag_)) {
                visible.append(trailing_whitespace_);
                trailing_whitespace_.clear();
                visible.append(pending_tag_);
                pending_tag_.clear();
            }
            continue;
        }

        if (byte == '<') {
            pending_tag_.push_back(byte);
        } else if (is_format_whitespace(byte)) {
            trailing_whitespace_.push_back(byte);
        } else {
            visible.append(trailing_whitespace_);
            trailing_whitespace_.clear();
            visible.push_back(byte);
        }
    }
    return visible;
}

ToolCallOutputDecoder::Terminal ToolCallOutputDecoder::finish() {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    finished_ = true;
    if (!contract_) { return {}; }

    ParsedToolCallOutput parsed =
        parse_qwen_tool_call_output(tool_region_, max_tool_name_length_, *contract_, tolerant_);
    if (saw_tool_marker_ && parsed.is_tool_call_response) {
        // The parser reports the held bytes before the accepted structured region, which are the
        // bytes after an earlier quoted marker that this decoder has not published yet.
        std::string content = std::move(parsed.content);
        trailing_whitespace_.clear();
        tool_region_.clear();
        pending_tag_.clear();
        return Terminal{.content     = std::move(content),
                        .tool_calls  = std::move(parsed.tool_calls),
                        .diagnostics = parsed.diagnostics};
    }

    std::string tail = std::move(trailing_whitespace_);
    tail.append(pending_tag_);
    pending_tag_.clear();
    tail += tool_region_;
    tool_region_.clear();
    return Terminal{
        .content = std::move(tail), .tool_calls = {}, .diagnostics = parsed.diagnostics};
}

} // namespace ninfer::models::qwen3_5::frontend
