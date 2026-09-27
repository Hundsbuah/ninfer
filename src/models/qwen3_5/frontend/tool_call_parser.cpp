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

constexpr std::string_view kToolOpen        = "<tool_call>";
constexpr std::string_view kToolClose       = "</tool_call>";
constexpr std::string_view kFunctionCallsOpen  = "<function_calls>";
constexpr std::string_view kFunctionCallsClose = "</function_calls>";

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

// ---------------------------------------------------------------------------
// Byte utilities
// ---------------------------------------------------------------------------

constexpr bool is_format_whitespace(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

constexpr bool is_ascii_digit(char byte) { return byte >= '0' && byte <= '9'; }

constexpr bool is_ascii_alphanumeric(char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || is_ascii_digit(byte);
}

// The opener boundary delimiter: the byte right after a "<function", "<invoke", "<parameter"
// or "<param" prefix that proves the prefix is not a longer identifier.
constexpr bool is_opener_delimiter(char byte) {
    return byte == '=' || byte == '>' || is_format_whitespace(byte);
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

// ---------------------------------------------------------------------------
// Tag header recognition (shared by tokenization, parsing, and lookahead)
// ---------------------------------------------------------------------------

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
//             first name= attribute supplies the name and must be non-empty. Whitespace may
//             separate an attribute name from its '=' and an attribute value from the next
//             attribute; a quoted value may be empty, an unquoted value may not.
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
        skip_format_whitespace(payload, i);
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
            if (value.empty()) { return false; }
        }
        if (attr_name == "name" && !has_name) {
            if (value.empty()) { return false; }
            name     = value;
            has_name = true;
        }
        if (i < payload.size() && !is_format_whitespace(payload[i])) { return false; }
    }
    if (!has_name) { return false; }
    parsed.name   = name;
    parsed.syntax = HeaderSyntax::Attributes;
    return true;
}

enum class FunctionTagKind : std::uint8_t {
    Function,
    Invoke,
};

// The strict opener boundary: the byte after "<function" or "<invoke" must delimit the
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
    return is_opener_delimiter(text[prefix_end]);
}

// Tolerant malformed-opener recovery: a dropped or doubled leading '<', a leaked ChatML
// turn marker, or a dropped 'function'/'invoke' keyword, followed by '=' or a name. True
// when the recovered prefix is plausible enough to enter the payload stage; a form that
// yields no valid name is rejected by the payload stage, so prose after a marker cannot
// pass.
bool recover_function_prefix(std::string_view text, std::size_t pos, std::size_t& prefix_end,
                             std::string_view& close_tag, FunctionTagKind& kind) {
    std::size_t scan = pos;
    while (scan < text.size() && text[scan] == '<') { ++scan; }
    if (starts_with_at(text, scan, "|im_start|>")) { scan += 11; }
    std::size_t kw_len = 0;
    if (starts_with_at(text, scan, "function")) {
        kw_len = 8;
    } else if (starts_with_at(text, scan, "invoke")) {
        kw_len = 6;
    } else {
        return false;
    }
    scan += kw_len;
    if (scan >= text.size() || (text[scan] != '=' && !is_format_whitespace(text[scan]))) {
        return false;
    }
    if (text[scan] == '=') { ++scan; }
    prefix_end = scan;
    close_tag  = kw_len == 8 ? "</function>" : "</invoke>";
    kind       = kw_len == 8 ? FunctionTagKind::Function : FunctionTagKind::Invoke;
    return true;
}

// One recognized tool marker in the token stream. begin/end span the whole opener (a
// recovered function opener ends at the whitespace that terminates its name); name is the
// validated header name; close_kind is the matching close marker for openers.
struct Marker {
    enum class Kind : std::uint8_t {
        ToolCallOpen,
        FunctionCallsOpen,
        FunctionOpen,
        InvokeOpen,
        ParameterOpen,
        ToolCallClose,
        FunctionCallsClose,
        FunctionClose,
        InvokeClose,
        ParameterClose,
        ParamClose,
    };
    Kind kind              = Kind::ToolCallClose;
    Kind close_kind        = Kind::ToolCallClose;
    std::size_t begin      = 0;
    std::size_t end        = 0;
    std::string_view name;
    bool recovered         = false;
};

constexpr bool is_region_open_kind(Marker::Kind kind) {
    return kind == Marker::Kind::ToolCallOpen || kind == Marker::Kind::FunctionCallsOpen ||
           kind == Marker::Kind::FunctionOpen || kind == Marker::Kind::InvokeOpen;
}

// The canonical parameter opener grammar: "<parameter" or "<param", then a byte that
// delimits the prefix, then a header payload terminated by the quote-aware tag end and
// fully validated by parse_name_header_payload(). The byte after the prefix must delimit
// it: a longer identifier such as "<parameterX>" is not a parameter opener. Because the
// payload is validated as a whole, a fake opener that finds its ">" in value data is
// rejected by this one grammar - so tokenization, parsing, and boundary detection can
// never classify the same bytes differently.
bool parse_parameter_open(std::string_view text, std::size_t pos, Marker& marker) {
    std::size_t header_begin = 0;
    Marker::Kind close_kind   = Marker::Kind::ParameterClose;
    if (starts_with_at(text, pos, "<parameter")) {
        header_begin = pos + 10;
    } else if (starts_with_at(text, pos, "<param")) {
        header_begin = pos + 6;
        close_kind   = Marker::Kind::ParamClose;
    } else {
        return false;
    }
    if (header_begin >= text.size()) { return false; }
    if (!is_opener_delimiter(text[header_begin])) { return false; }
    std::size_t tag_end = 0;
    if (!find_tag_end(text, header_begin, tag_end) || tag_end == header_begin) {
        return false;
    }
    NameHeaderPayload payload;
    if (!parse_name_header_payload(text.substr(header_begin, tag_end - header_begin), payload)) {
        return false;
    }
    marker.kind       = Marker::Kind::ParameterOpen;
    marker.close_kind = close_kind;
    marker.begin      = pos;
    marker.end        = tag_end + 1;
    marker.name       = payload.name;
    return true;
}

// The canonical function opener grammar: "<function" or "<invoke", then the strict opener
// boundary (or, in tolerant mode, the recovered malformed form), then the validated
// payload. The missing-'> recovery (tolerant mode) fires before the payload parse and
// supplies both the tag end and the name: the model sometimes drops the '>' after the
// function name (for example a name followed directly by a newline and a parameter tag),
// so it recovers by scanning the identifier run and accepting it when format whitespace
// separates it from the next '<' or end of region. A header that fails here fails in
// every stage alike: no function name is validated here, so name validity stays a policy
// decision of the consumer and of the boundary lookahead alike.
bool parse_function_open(std::string_view text, std::size_t pos, std::size_t max_name_length,
                         bool tolerant, Marker& marker) {
    std::size_t prefix_end = 0;
    std::string_view close_tag;
    FunctionTagKind kind = FunctionTagKind::Function;
    if (!function_opener_prefix_at(text, pos, prefix_end, close_tag, kind)) {
        if (!tolerant || !recover_function_prefix(text, pos, prefix_end, close_tag, kind)) {
            return false;
        }
    }
    std::size_t tag_end = 0;
    const bool found_gt = find_tag_end(text, prefix_end, tag_end);
    if (tolerant) {
        std::size_t scan = prefix_end;
        while (scan < text.size() && text[scan] == '=') { ++scan; }
        const std::size_t ident_begin = scan;
        while (scan < text.size() && scan - prefix_end < max_name_length) {
            const char byte = text[scan];
            if (!is_ascii_alphanumeric(byte) && byte != '_' && byte != '-') { break; }
            ++scan;
        }
        if (scan > ident_begin && scan < text.size() && is_format_whitespace(text[scan]) &&
            (!found_gt || scan < tag_end)) {
            std::size_t after = scan;
            while (after < text.size() && is_format_whitespace(text[after])) { ++after; }
            if (after >= text.size() || text[after] == '<') {
                marker.kind      = kind == FunctionTagKind::Function ? Marker::Kind::FunctionOpen
                                                                     : Marker::Kind::InvokeOpen;
                marker.close_kind = marker.kind;
                marker.begin     = pos;
                marker.end       = scan;
                marker.name      = text.substr(ident_begin, scan - ident_begin);
                marker.recovered = true;
                return true;
            }
        }
    }
    if (!found_gt || tag_end == prefix_end) { return false; }
    NameHeaderPayload payload;
    if (!parse_name_header_payload(text.substr(prefix_end, tag_end - prefix_end), payload)) {
        return false;
    }
    marker.kind      = kind == FunctionTagKind::Function ? Marker::Kind::FunctionOpen
                                                         : Marker::Kind::InvokeOpen;
    marker.close_kind = marker.kind;
    marker.begin     = pos;
    marker.end       = tag_end + 1;
    marker.name      = payload.name;
    return true;
}

bool close_marker_at(std::string_view text, std::size_t pos, Marker& marker) {
    if (starts_with_at(text, pos, kToolClose)) {
        marker.kind = Marker::Kind::ToolCallClose;
        marker.begin = pos;
        marker.end   = pos + kToolClose.size();
        return true;
    }
    if (starts_with_at(text, pos, kFunctionCallsClose)) {
        marker.kind = Marker::Kind::FunctionCallsClose;
        marker.begin = pos;
        marker.end   = pos + kFunctionCallsClose.size();
        return true;
    }
    if (starts_with_at(text, pos, "</parameter>")) {
        marker.kind = Marker::Kind::ParameterClose;
        marker.begin = pos;
        marker.end   = pos + 12;
        return true;
    }
    if (starts_with_at(text, pos, "</param>")) {
        marker.kind = Marker::Kind::ParamClose;
        marker.begin = pos;
        marker.end   = pos + 8;
        return true;
    }
    if (starts_with_at(text, pos, "</function>")) {
        marker.kind = Marker::Kind::FunctionClose;
        marker.begin = pos;
        marker.end   = pos + 11;
        return true;
    }
    if (starts_with_at(text, pos, "</invoke>")) {
        marker.kind = Marker::Kind::InvokeClose;
        marker.begin = pos;
        marker.end   = pos + 9;
        return true;
    }
    return false;
}

// One region-open candidate: a wrapper open, or the function/invoke opener prefix with a
// delimiting byte. This is the single definition of a top-level tool-call marker, shared
// by one-shot candidate discovery and the streaming decoder.
bool is_region_open_at(std::string_view text, std::size_t pos) {
    if (starts_with_at(text, pos, kToolOpen)) { return true; }
    if (starts_with_at(text, pos, kFunctionCallsOpen)) { return true; }
    std::size_t prefix_end = 0;
    std::string_view close_tag;
    FunctionTagKind kind;
    return function_opener_prefix_at(text, pos, prefix_end, close_tag, kind);
}

std::size_t find_first_region_open(std::string_view text) {
    for (std::size_t pos = 0; pos < text.size(); ++pos) {
        if (is_region_open_at(text, pos)) { return pos; }
    }
    return std::string_view::npos;
}

// Tokenize a tool region: one pass over the text, emitting every recognized marker and
// advancing past each (a marker's header bytes are consumed opaquely, so a close marker
// inside a fake header is never a structural token). Wrapper opens are literal tokens; a
// function open token starts exactly where the shared recognition succeeds, so a dispatch
// or successor position that does not align with a token is a failed attempt there - the
// bytes after it (a doubled '<', ...) are not re-recognized at that position.
void tokenize_markers(std::string_view text, std::size_t max_name_length, bool tolerant,
                      std::vector<Marker>& markers) {
    std::size_t pos = 0;
    while (pos < text.size()) {
        Marker marker;
        if (starts_with_at(text, pos, kToolOpen)) {
            marker.kind    = Marker::Kind::ToolCallOpen;
            marker.begin   = pos;
            marker.end     = pos + kToolOpen.size();
            markers.push_back(marker);
            pos = marker.end;
            continue;
        }
        if (starts_with_at(text, pos, kFunctionCallsOpen)) {
            marker.kind    = Marker::Kind::FunctionCallsOpen;
            marker.begin   = pos;
            marker.end     = pos + kFunctionCallsOpen.size();
            markers.push_back(marker);
            pos = marker.end;
            continue;
        }
        if (close_marker_at(text, pos, marker)) {
            markers.push_back(marker);
            pos = marker.end;
            continue;
        }
        if (parse_parameter_open(text, pos, marker)) {
            markers.push_back(marker);
            pos = marker.end;
            continue;
        }
        if (parse_function_open(text, pos, max_name_length, tolerant, marker)) {
            markers.push_back(marker);
            pos = marker.end;
            continue;
        }
        ++pos;
    }
}

// The same marker grammar the one-shot parser discovers: a pending byte run is a marker
// when it equals a wrapper open, or extends an opener prefix by one delimiting byte.
bool marker_confirmed(std::string_view pending) {
    if (pending == kToolOpen || pending == kFunctionCallsOpen) { return true; }
    for (const std::string_view root : {std::string_view("<function"), std::string_view("<invoke")}) {
        if (pending.size() == root.size() + 1 && pending.starts_with(root) &&
            is_opener_delimiter(pending[root.size()])) {
            return true;
        }
    }
    return false;
}

// A pending byte run may still become a confirmed marker.
bool marker_possible(std::string_view pending) {
    if (kToolOpen.starts_with(pending)) { return true; }
    if (kFunctionCallsOpen.starts_with(pending)) { return true; }
    for (const std::string_view root : {std::string_view("<function"), std::string_view("<invoke")}) {
        if (root.starts_with(pending)) { return true; }
        if (pending.size() == root.size() + 1 && pending.starts_with(root) &&
            is_opener_delimiter(pending[root.size()])) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Name policy
// ---------------------------------------------------------------------------

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    return std::all_of(name.begin(), name.end(), [](char byte) {
        return is_ascii_alphanumeric(byte) || byte == '_' || byte == '-';
    });
}

// ---------------------------------------------------------------------------
// Schema and normalization utilities
// ---------------------------------------------------------------------------

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

// Encodes a raw byte run as a JSON string literal. Valid UTF-8 is dumped by nlohmann
// unchanged; an invalid sequence (for example a multibyte character cut by the output
// budget) is encoded as \u00XX escapes - one per byte, so the bytes stay recoverable - and
// model-derived bytes can never throw out of normalization.
std::string encode_json_string(std::string_view value) {
    try {
        return Json(std::string(value)).dump();
    } catch (const Json::type_error&) {
        const char hex_digits[] = "0123456789abcdef";
        std::string encoded;
        encoded.reserve(value.size() * 2 + 2);
        encoded.push_back('"');
        for (const char byte : value) {
            const unsigned c = static_cast<unsigned char>(byte);
            if (c == '"') {
                encoded += "\\\"";
            } else if (c == '\\') {
                encoded += "\\\\";
            } else if (c == '\b') {
                encoded += "\\b";
            } else if (c == '\f') {
                encoded += "\\f";
            } else if (c == '\n') {
                encoded += "\\n";
            } else if (c == '\r') {
                encoded += "\\r";
            } else if (c == '\t') {
                encoded += "\\t";
            } else if (c < 0x20 || c >= 0x80) {
                encoded += "\\u";
                encoded += hex_digits[(c >> 12) & 0xF];
                encoded += hex_digits[(c >> 8) & 0xF];
                encoded += hex_digits[(c >> 4) & 0xF];
                encoded += hex_digits[c & 0xF];
            } else {
                encoded.push_back(static_cast<char>(c));
            }
        }
        encoded.push_back('"');
        return encoded;
    }
}

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

// ---------------------------------------------------------------------------
// Region parsing over the marker token stream
// ---------------------------------------------------------------------------

class QwenToolRegionParser {
public:
    QwenToolRegionParser(std::string_view text, std::size_t max_name_length,
                         const Contract& contract, bool tolerant)
        : text_(text), max_name_length_(max_name_length), contract_(contract),
          tolerant_(tolerant) {
        tokenize_markers(text_, max_name_length_, tolerant_, markers_);
    }

    [[nodiscard]] std::uint32_t duplicate_parameters_repaired() const noexcept {
        return duplicate_parameters_repaired_;
    }

    FallbackReason parse(std::vector<RawToolCall>& calls) {
        std::size_t idx = 0;
        while (idx < markers_.size()) {
            // Any byte run before a marker that is not a region open, and a marker that is
            // not a region open, are ordinary text at dispatch level: a function prefix
            // there is a failed function attempt (its reason is reported), anything else is
            // trailing content.
            const std::size_t from = idx > 0 ? markers_[idx - 1].end : 0;
            if (has_content_between(from, markers_[idx].begin) ||
                !is_region_open_kind(markers_[idx].kind)) {
                return dispatch_gap_failure(calls, from, markers_[idx].begin);
            }
            switch (markers_[idx].kind) {
            case Marker::Kind::ToolCallOpen: {
                RawToolCall call;
                const FallbackReason terminal = finish_call(calls, call, parse_tool_call(idx, call));
                if (terminal != FallbackReason::None) { return terminal; }
                break;
            }
            case Marker::Kind::FunctionCallsOpen: {
                const FallbackReason terminal = parse_function_calls(idx, calls);
                if (terminal != FallbackReason::None) { return terminal; }
                break;
            }
            case Marker::Kind::FunctionOpen:
            case Marker::Kind::InvokeOpen: {
                RawToolCall call;
                const FallbackReason terminal =
                    finish_call(calls, call, parse_function(idx, call, FunctionContainer::TopLevel));
                if (terminal != FallbackReason::None) { return terminal; }
                break;
            }
            }
        }
        const std::size_t last = markers_.empty() ? 0 : markers_.back().end;
        if (has_content_between(last, text_.size())) {
            return dispatch_gap_failure(calls, last, text_.size());
        }
        return calls.empty() ? FallbackReason::MalformedStructure : FallbackReason::None;
    }

private:
    bool has_content_between(std::size_t a, std::size_t b) const {
        if (a >= b) { return false; }
        return !trim_format_whitespace(text_.substr(a, b - a)).empty();
    }

    std::size_t first_non_whitespace(std::size_t a, std::size_t b) const {
        std::size_t at = a;
        while (at < b && is_format_whitespace(text_[at])) { ++at; }
        return at;
    }

    // The same opener grammar parse_function_open() consumes, applied to a boundary
    // candidate: an opener that does not parse here does not parse in the consumer
    // either. The only policy overlay is the declared-tool identity in strict mode;
    // name validity itself is mode-independent.
    bool function_name_policy(std::string_view name) const {
        if (!valid_function_name(name, max_name_length_)) { return false; }
        if (!tolerant_ && contract_.enforce_declared_names &&
            find_tool_contract(contract_, name) == nullptr) {
            return false;
        }
        return true;
    }

    // A genuine Qwen parameter header is "<parameter=<name>": the official chat
    // templates emit the declared name verbatim between "=" and the first ">". A
    // candidate whose name the current unambiguous tool contract does not declare
    // therefore establishes a structural boundary only while the name is still a
    // plausible ordinary parameter identifier.
    bool parameter_sibling_policy(std::string_view fn_name, std::string_view name) const {
        return is_declared_parameter(contract_, fn_name, name) ||
               is_ordinary_parameter_name(name);
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

    // A failed function attempt at a position where a function open is expected and none
    // was tokenized: the prefix boundary (or the tolerant recovery) reached the payload
    // stage and found no valid header - InvalidToolName in both modes - or the bytes do
    // not even open a function tag - MalformedStructure.
    FallbackReason function_entry_failure(std::size_t at) const {
        std::size_t prefix_end = 0;
        std::string_view close_tag;
        FunctionTagKind kind;
        if (function_opener_prefix_at(text_, at, prefix_end, close_tag, kind)) {
            return FallbackReason::InvalidToolName;
        }
        if (tolerant_ && recover_function_prefix(text_, at, prefix_end, close_tag, kind)) {
            return FallbackReason::InvalidToolName;
        }
        return FallbackReason::MalformedStructure;
    }

    // The expected-function contexts (inside a wrapper, inside a function_calls container)
    // call the consumer on any position: a failed attempt there is a structural failure,
    // never trailing text.
    FallbackReason expected_function_failure(const std::vector<RawToolCall>& calls,
                                             std::size_t from, std::size_t to) const {
        const FallbackReason failure = function_entry_failure(first_non_whitespace(from, to));
        if (tolerant_ && !calls.empty()) { return FallbackReason::TruncatedTail; }
        return failure;
    }


    // Top-level trailing text: tolerant discards a suffix after one or more complete
    // calls; a function prefix that opens no tag keeps the hard structural failure.
    FallbackReason dispatch_gap_failure(const std::vector<RawToolCall>& calls, std::size_t from,
                                        std::size_t to) const {
        const std::size_t at = first_non_whitespace(from, to);
        if (at < to && (starts_with_at(text_, at, "<function") ||
                        starts_with_at(text_, at, "<invoke"))) {
            const FallbackReason failure = function_entry_failure(at);
            if (tolerant_ && !calls.empty()) { return FallbackReason::TruncatedTail; }
            return failure;
        }
        return trailing_failure(calls);
    }

    FallbackReason trailing_failure(const std::vector<RawToolCall>& calls) const {
        if (tolerant_ && !calls.empty()) { return FallbackReason::TruncatedTail; }
        return calls.empty() ? FallbackReason::MalformedStructure
                             : FallbackReason::TrailingContent;
    }

    FallbackReason parse_tool_call(std::size_t& idx, RawToolCall& call) {
        // markers_[idx] is the wrapper open.
        const std::size_t open_end = markers_[idx].end;
        if (idx + 1 >= markers_.size()) {
            // The region ends inside the function opener: report the attempt at the first
            // non-whitespace byte after the wrapper, like every other expected position.
            return function_entry_failure(first_non_whitespace(open_end, text_.size()));
        }
        // The expected function position is the first non-whitespace byte after the wrapper
        // open. Only a function open token aligned there is the attempt the single-position
        // consumer would make: a token that starts later (a doubled '<' before a valid
        // opener) is not it, and the attempt at the expected position is reported as such.
        const std::size_t expected = first_non_whitespace(open_end, markers_[idx + 1].begin);
        if (expected != markers_[idx + 1].begin ||
            (markers_[idx + 1].kind != Marker::Kind::FunctionOpen &&
             markers_[idx + 1].kind != Marker::Kind::InvokeOpen)) {
            return function_entry_failure(expected);
        }
        std::size_t fn_idx = idx + 1;
        const FallbackReason failure =
            parse_function(fn_idx, call, FunctionContainer::ToolCall);
        idx = fn_idx;
        if (failure != FallbackReason::None) { return failure; }
        // idx now points past the function close.
        if (idx < markers_.size() && markers_[idx].kind == Marker::Kind::ToolCallClose &&
            !has_content_between(markers_[idx - 1].end, markers_[idx].begin)) {
            ++idx;
            return FallbackReason::None;
        }
        // Tolerant: a complete call may be followed by explanatory text, or the model may
        // have stopped at the end of its budget before the closing tag. parse() applies the
        // call-level recovery policy; the strict parser keeps the hard structural failure.
        return tolerant_ ? FallbackReason::TruncatedTail : FallbackReason::MalformedStructure;
    }

    FallbackReason parse_function_calls(std::size_t& idx, std::vector<RawToolCall>& calls) {
        // markers_[idx] is the container open.
        bool had_calls = false;
        std::size_t i = idx + 1;
        while (i < markers_.size()) {
            if (has_content_between(markers_[i - 1].end, markers_[i].begin) ||
                (markers_[i].kind != Marker::Kind::FunctionCallsClose &&
                 markers_[i].kind != Marker::Kind::FunctionOpen &&
                 markers_[i].kind != Marker::Kind::InvokeOpen)) {
                return expected_function_failure(calls, markers_[i - 1].end,
                                                 markers_[i].begin);
            }
            if (markers_[i].kind == Marker::Kind::FunctionCallsClose) {
                idx = i + 1;
                // An empty container is not a tool-call region.
                return had_calls ? FallbackReason::None : FallbackReason::MalformedStructure;
            }
            RawToolCall call;
            const FallbackReason terminal =
                finish_call(calls, call, parse_function(i, call, FunctionContainer::FunctionCalls));
            if (terminal != FallbackReason::None) { return terminal; }
            had_calls = true;
        }
        // The region is exhausted inside the container: an unclosed wrapper after one or
        // more complete calls is a truncation in tolerant mode; the strict parser keeps
        // the hard structural failure.
        if (tolerant_ && had_calls) { return FallbackReason::TruncatedTail; }
        return FallbackReason::MalformedStructure;
    }

    FallbackReason parse_function(std::size_t& idx, RawToolCall& call,
                                  FunctionContainer container) {
        const Marker& open = markers_[idx];
        call.name = open.name;
        if (!valid_function_name(call.name, max_name_length_)) {
            return FallbackReason::InvalidToolName;
        }
        // Strict mode rejects a name outside the declared tool set. Tolerant mode keeps an
        // otherwise well-formed call structured and leaves the identity judgment to the
        // consumer: leaking the raw region to content would turn a valid call into prose.
        if (!tolerant_ && contract_.enforce_declared_names &&
            find_tool_contract(contract_, call.name) == nullptr) {
            return FallbackReason::UndeclaredTool;
        }
        const Marker::Kind fn_close_kind =
            open.kind == Marker::Kind::FunctionOpen ? Marker::Kind::FunctionClose
                                                    : Marker::Kind::InvokeClose;
        for (std::size_t i = idx + 1; ; ) {
            if (i >= markers_.size()) {
                // Tolerant: a missing function close is a truncation only when the region truly
                // ends after the last complete parameter; trailing prose after it is a
                // structural failure in both modes.
                if (tolerant_ && !has_content_between(markers_[i - 1].end, text_.size())) {
                    return FallbackReason::TruncatedTail;
                }
                return FallbackReason::MalformedStructure;
            }
            if (has_content_between(markers_[i - 1].end, markers_[i].begin)) {
                return FallbackReason::MalformedStructure;
            }
            if (markers_[i].kind == fn_close_kind) {
                idx = i + 1;
                return FallbackReason::None;
            }
            if (markers_[i].kind != Marker::Kind::ParameterOpen) {
                return FallbackReason::MalformedStructure;
            }
            const FallbackReason failure = parse_parameter(i, call, fn_close_kind, container);
            if (failure != FallbackReason::None) { return failure; }
        }
    }

    FallbackReason parse_parameter(std::size_t& idx, RawToolCall& call, Marker::Kind fn_close_kind,
                                   FunctionContainer container) {
        const Marker& open = markers_[idx];
        const std::string_view name = open.name;
        const bool opaque_string    = is_declared_string_parameter(contract_, call.name, name);
        const std::size_t value_begin = open.end;
        std::size_t value_end = 0;
        std::size_t close_idx = 0;
        if (!find_parameter_close(idx, value_end, close_idx, open.close_kind, fn_close_kind,
                                  container, opaque_string, call.name)) {
            if (tolerant_) {
                // Tolerant: the region ends before the closing tag, so the output budget cut
                // the parameter value. Keep the value up to the cut (last occurrence wins, as
                // with a complete parameter) and flag the tail.
                const std::string_view partial =
                    text_.substr(value_begin, text_.size() - value_begin);
                const auto existing =
                    std::find_if(call.parameters.begin(), call.parameters.end(),
                                 [&](const RawParameter& candidate) { return candidate.name == name; });
                if (existing != call.parameters.end()) {
                    existing->value = partial;
                    ++duplicate_parameters_repaired_;
                } else {
                    call.parameters.push_back(RawParameter{.name = name, .value = partial});
                }
                idx = markers_.size();
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
        idx = close_idx + 1;
        return FallbackReason::None;
    }

    // A valid continuation after a completed function in this container. Every check is
    // positional: the successor is examined at the first non-whitespace byte after the
    // function close, so only a token that begins exactly there is seen; bytes between the
    // close and a later token are prose at that position (a doubled '<' before a valid
    // opener is one such run). The ToolCall container additionally requires that what
    // follows the wrapper close is EOF, a valid top-level construct, or prose carrying no
    // further "</tool_call>": a later wrapper close that the prose does not belong to is
    // the boundary of a fake call embedded in value data, and accepting it would shadow a
    // later, more complete close.
    bool function_successor(std::size_t k, FunctionContainer container) const {
        if (k + 1 >= markers_.size()) {
            // TopLevel: a region that ends after the function close is a successor; the
            // wrapper containers still require their own closing tag.
            return container == FunctionContainer::TopLevel &&
                   !has_content_between(markers_[k].end, text_.size());
        }
        const std::size_t expected = first_non_whitespace(markers_[k].end, markers_[k + 1].begin);
        const bool aligned = expected == markers_[k + 1].begin;
        switch (container) {
        case FunctionContainer::ToolCall: {
            if (!aligned || markers_[k + 1].kind != Marker::Kind::ToolCallClose) {
                return false;
            }
            if (k + 2 >= markers_.size()) { return true; }
            const std::size_t expected2 =
                first_non_whitespace(markers_[k + 1].end, markers_[k + 2].begin);
            if (expected2 == markers_[k + 2].begin) {
                const Marker::Kind nxt = markers_[k + 2].kind;
                if (nxt == Marker::Kind::ToolCallOpen || nxt == Marker::Kind::FunctionCallsOpen) {
                    return true;
                }
                if ((nxt == Marker::Kind::FunctionOpen || nxt == Marker::Kind::InvokeOpen) &&
                    function_name_policy(markers_[k + 2].name)) {
                    return true;
                }
            }
            for (std::size_t i = k + 2; i < markers_.size(); ++i) {
                if (markers_[i].kind == Marker::Kind::ToolCallClose) { return false; }
            }
            return true;
        }
        case FunctionContainer::FunctionCalls: {
            if (!aligned) { return false; }
            if (markers_[k + 1].kind == Marker::Kind::FunctionCallsClose) { return true; }
            return (markers_[k + 1].kind == Marker::Kind::FunctionOpen ||
                    markers_[k + 1].kind == Marker::Kind::InvokeOpen) &&
                   function_name_policy(markers_[k + 1].name);
        }
        case FunctionContainer::TopLevel: {
            if (!aligned) { return false; }
            const Marker::Kind nxt = markers_[k + 1].kind;
            if (nxt == Marker::Kind::ToolCallOpen || nxt == Marker::Kind::FunctionCallsOpen) {
                return true;
            }
            return (nxt == Marker::Kind::FunctionOpen || nxt == Marker::Kind::InvokeOpen) &&
                   function_name_policy(markers_[k + 1].name);
        }
        }
        return false;
    }

    // Marker k is a literal parameter close of the required spelling. It ends the value
    // only when what follows continues the enclosing call: a genuine sibling, or the
    // function close with a structural successor - in both cases a token aligned at the
    // first non-whitespace byte after this close. In the tolerant relaxation pass a
    // function close still counts when the region ends after it or only tokens remain
    // that the container discards (a stray parameter open, plain prose); a marker the
    // container does not expect there (a wrapper-foreign open) is the closing boundary of
    // an embedded fake call in opaque data and never a sibling boundary.
    bool parameter_close_structural(std::size_t k, FunctionContainer container, bool relaxed,
                                    Marker::Kind fn_close_kind,
                                    std::string_view fn_name) const {
        if (k + 1 >= markers_.size()) {
            // A complete last parameter at EOF is useful only to tolerant recovery, where
            // parse_function() classifies the missing function close as a truncated tail.
            return relaxed;
        }
        const std::size_t expected = first_non_whitespace(markers_[k].end, markers_[k + 1].begin);
        if (expected != markers_[k + 1].begin) {
            // Prose between this close and the next marker: the positional check at the
            // expected position sees neither a close tag nor a parameter header.
            return false;
        }
        const Marker& next = markers_[k + 1];
        if (next.kind == fn_close_kind) {
            if (function_successor(k + 1, container)) { return true; }
            if (!relaxed) { return false; }
            if (k + 2 >= markers_.size()) { return true; }
            const std::size_t expected2 =
                first_non_whitespace(markers_[k + 1].end, markers_[k + 2].begin);
            if (expected2 != markers_[k + 2].begin) { return true; }
            const Marker::Kind nxt2 = markers_[k + 2].kind;
            if (nxt2 == Marker::Kind::ParameterOpen) { return true; }
            return !is_region_open_kind(nxt2);
        }
        if (next.kind == Marker::Kind::ParameterOpen) {
            return parameter_sibling_policy(fn_name, next.name);
        }
        return false;
    }

    bool find_opaque_parameter_close(std::size_t idx, std::size_t& value_end,
                                     std::size_t& close_idx, Marker::Kind required_close,
                                     Marker::Kind fn_close_kind, FunctionContainer container,
                                     bool relaxed, std::string_view fn_name) const {
        for (std::size_t k = idx + 1; k < markers_.size(); ++k) {
            if (markers_[k].kind != required_close) { continue; }
            if (parameter_close_structural(k, container, relaxed, fn_close_kind, fn_name)) {
                value_end = markers_[k].begin;
                close_idx = k;
                return true;
            }
        }
        return false;
    }

    bool find_parameter_close(std::size_t idx, std::size_t& value_end, std::size_t& close_idx,
                              Marker::Kind required_close, Marker::Kind fn_close_kind,
                              FunctionContainer container, bool opaque_string,
                              std::string_view fn_name) const {
        if (opaque_string) {
            // Declared string parameters are opaque data. Parameter elements are siblings in
            // the Qwen grammar, not recursively nested elements, so a literal "<parameter...>"
            // inside source code must not increase structural depth. Conversely, a literal
            // "</parameter>" is data unless what follows can continue the enclosing function.
            //
            // The wire format is still intrinsically ambiguous if user data itself contains an
            // exact complete closing boundary: no delimiter parser can distinguish that byte
            // sequence without escaping or length-prefixing, so the value ends at the first
            // such boundary and the following bytes may then be parsed as further parameters
            // or calls without any fallback reason. Restricting this recovery to
            // schema-declared strings avoids changing legacy/non-string parsing.
            //
            // Prefer a fully structural boundary. Only when that is impossible does tolerant
            // mode relax the suffix check.
            if (find_opaque_parameter_close(idx, value_end, close_idx, required_close,
                                            fn_close_kind, container, /*relaxed*/ false,
                                            fn_name)) {
                return true;
            }
            return tolerant_ && find_opaque_parameter_close(idx, value_end, close_idx,
                                                            required_close, fn_close_kind,
                                                            container, /*relaxed*/ true, fn_name);
        }

        // Legacy and non-string values keep the historical balanced-marker behavior.
        std::size_t depth = 1;
        for (std::size_t k = idx + 1; k < markers_.size(); ++k) {
            if (markers_[k].kind == Marker::Kind::ParameterOpen) {
                ++depth;
                continue;
            }
            if (markers_[k].kind == required_close) {
                --depth;
                if (depth == 0) {
                    value_end = markers_[k].begin;
                    close_idx = k;
                    return true;
                }
            }
        }
        return false;
    }

    std::string_view text_;
    std::size_t max_name_length_;
    const Contract& contract_;
    std::vector<Marker> markers_;
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
    std::size_t candidate = find_first_region_open(source);
    if (candidate == std::string_view::npos) { return fallback(text); }

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
            if (marker_confirmed(pending_tag_)) {
                tool_region_ = std::move(trailing_whitespace_);
                trailing_whitespace_.clear();
                tool_region_.append(pending_tag_);
                pending_tag_.clear();
                tool_region_.append(text.substr(index + 1));
                saw_tool_marker_ = true;
                break;
            }
            if (!marker_possible(pending_tag_)) {
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
