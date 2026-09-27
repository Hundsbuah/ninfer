#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <compare>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

using Json                = nlohmann::json;
using Contract            = ToolCallOutputContract;
using FallbackReason      = ToolCallParseFallbackReason;
using NormalizationPolicy = Contract::NormalizationPolicy;
using SchemaType          = Contract::SchemaType;
using TypeSet             = Contract::TypeSet;

constexpr std::size_t kNpos                = std::string_view::npos;
constexpr std::string_view kToolOpen       = "<tool_call>";
constexpr std::string_view kToolClose      = "</tool_call>";
constexpr std::string_view kContainerOpen  = "<function_calls>";
constexpr std::string_view kContainerClose = "</function_calls>";
constexpr std::string_view kFunctionClose  = "</function>";
constexpr std::string_view kInvokeClose    = "</invoke>";
constexpr std::string_view kParameterClose = "</parameter>";
constexpr std::string_view kParamClose     = "</param>";
// A tag header is one short line; a longer run is text. This also bounds every header scan.
constexpr std::size_t kMaxTagHeaderBytes = 256;

struct RawParameter {
    std::string_view name;
    std::string_view value;
};

struct RawToolCall {
    std::string_view name;
    std::vector<RawParameter> parameters;
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

std::string_view extract_name_from_tag_header(std::string_view header) {
    header = trim_format_whitespace(header);
    if (header.starts_with('=')) {
        return unquote(header.substr(1));
    }
    for (std::size_t i = 0; i < header.size();) {
        if (is_format_whitespace(header[i])) {
            ++i;
            continue;
        }
        const std::size_t attr_begin = i;
        while (i < header.size() && header[i] != '=' && !is_format_whitespace(header[i]) && header[i] != '>') {
            ++i;
        }
        const std::string_view attr_name = header.substr(attr_begin, i - attr_begin);
        skip_format_whitespace(header, i);
        if (i < header.size() && header[i] == '=') {
            ++i;
            skip_format_whitespace(header, i);
            if (i >= header.size()) break;
            std::string_view val;
            if (header[i] == '"' || header[i] == '\'') {
                const char q = header[i];
                const std::size_t q_start = i + 1;
                const std::size_t q_end = header.find(q, q_start);
                if (q_end != std::string_view::npos) {
                    val = header.substr(q_start, q_end - q_start);
                    i = q_end + 1;
                } else {
                    val = header.substr(q_start);
                    i = header.size();
                }
            } else {
                const std::size_t val_begin = i;
                while (i < header.size() && !is_format_whitespace(header[i]) && header[i] != '/' && header[i] != '>') {
                    ++i;
                }
                val = header.substr(val_begin, i - val_begin);
            }
            if (attr_name == "name") {
                return val;
            }
        }
    }
    return unquote(header);
}

// Position of the '>' that ends a tag header, or npos. The header must open with '=' or a space,
// so `<parameterX>` or `<functionbash>` is text, and it stays on one line without a nested '<'.
std::size_t tag_header_end(std::string_view text, std::size_t header_begin) {
    if (header_begin >= text.size()) { return kNpos; }
    const char delimiter = text[header_begin];
    if (delimiter != '=' && delimiter != ' ' && delimiter != '\t') { return kNpos; }
    const std::size_t limit = std::min(text.size(), header_begin + kMaxTagHeaderBytes);
    for (std::size_t pos = header_begin + 1; pos < limit; ++pos) {
        const char byte = text[pos];
        if (byte == '>') { return pos; }
        if (byte == '<' || byte == '\n' || byte == '\r') { return kNpos; }
    }
    return kNpos;
}

// Undeclared parameter names are accepted in identifier form only; declared names are accepted
// verbatim. Anything else inside a tag header (a regex class, `...`, a quoted phrase) is text.
bool plain_parameter_name(std::string_view name) {
    if (name.empty()) { return false; }
    const char lead = name.front();
    if (is_ascii_digit(lead) || (!is_ascii_alphanumeric(lead) && lead != '_' && lead != '$' &&
                                 lead != '@' && static_cast<unsigned char>(lead) < 0x80U)) {
        return false;
    }
    return std::all_of(name.begin(), name.end(), [](char byte) {
        return static_cast<unsigned char>(byte) >= 0x80U || is_ascii_alphanumeric(byte) ||
               byte == '_' || byte == '-' || byte == '.' || byte == '$' || byte == '@' ||
               byte == ':';
    });
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

// Region grammar:
//
//   region    := item (ws item)* ws EOF
//   item      := "<tool_call>" ws function ws "</tool_call>"
//              | "<function_calls>" ws function (ws function)* ws "</function_calls>"
//              | function
//   function  := function-open ws (parameter ws)* function-close
//   parameter := parameter-open VALUE parameter-close
//
// Qwen writes argument text without a delimiter escape, so VALUE may hold any bytes, including
// complete or partial tool-call markup. A parameter close is therefore a boundary candidate only
// where a sibling parameter, the function close or the region end follows it, and a reading must
// consume the whole region. Among the readings that do, the parser keeps the one whose values hold
// the least unmatched markup: line-framed tags, the serialization the chat template writes for
// real structure, count before inline ones, and a remaining tie keeps the earliest close.

enum class Wrapper : std::uint8_t { ToolCall, Container, Bare };

constexpr std::size_t kWrapperCount = 3;
// Where a value's call ends depends only on its function close form and its wrapper.
constexpr std::size_t kContextCount = 2 * kWrapperCount;

constexpr std::uint8_t context_of(bool invoke, Wrapper wrapper) {
    return static_cast<std::uint8_t>((invoke ? kWrapperCount : 0U) +
                                     static_cast<std::size_t>(wrapper));
}

constexpr bool context_invoke(std::uint8_t context) { return context >= kWrapperCount; }

constexpr Wrapper context_wrapper(std::uint8_t context) {
    return static_cast<Wrapper>(context % kWrapperCount);
}

enum class TagFamily : std::uint8_t { ToolCall, Container, Function, Parameter };

constexpr std::size_t kTagFamilyCount = 4;

struct ParameterOpen {
    std::size_t value_begin = 0;
    std::string_view name;
    bool short_form = false;
};

struct FunctionOpen {
    std::size_t body_begin = 0;
    std::string_view name;
    bool invoke = false;
};

struct FunctionOpenResult {
    std::optional<FunctionOpen> open;
    FallbackReason failure = FallbackReason::MalformedStructure;
};

struct MarkupTag {
    std::size_t begin = 0;
    // Index of the boundary candidate a parameter close forms, or npos when nothing that can
    // continue a call follows it.
    std::size_t candidate = kNpos;
    TagFamily family      = TagFamily::Parameter;
    bool closing          = false;
    bool short_form       = false;
    bool framed           = false;
};

struct BoundaryCandidate {
    enum class Kind : std::uint8_t { Sibling, FunctionEnd, RegionEnd };
    Kind kind         = Kind::RegionEnd;
    std::size_t begin = 0;
    ParameterOpen sibling;
    std::size_t function_end = 0;
    bool invoke              = false;
};

// Lexicographic plausibility of a reading. Recovery terms exist only in tolerant salvage.
struct ReadingCost {
    std::uint32_t discarded_suffix = 0;
    std::uint32_t missing_close    = 0;
    std::uint32_t framed_markup    = 0;
    std::uint32_t inline_markup    = 0;

    friend constexpr auto operator<=>(const ReadingCost&, const ReadingCost&) = default;
};

constexpr ReadingCost operator+(ReadingCost lhs, const ReadingCost& rhs) {
    lhs.discarded_suffix += rhs.discarded_suffix;
    lhs.missing_close += rhs.missing_close;
    lhs.framed_markup += rhs.framed_markup;
    lhs.inline_markup += rhs.inline_markup;
    return lhs;
}

struct Reading {
    bool valid = false;
    ReadingCost cost;
    FallbackReason failure = FallbackReason::MalformedStructure;
};

constexpr Reading accepted(ReadingCost cost = {}) { return Reading{.valid = true, .cost = cost}; }

constexpr Reading rejected(FallbackReason failure) { return Reading{.failure = failure}; }

struct ValueChoice {
    Reading reading;
    std::size_t candidate = kNpos;
};

using ValueChoices = std::array<ValueChoice, kContextCount>;

// Tags inside one value that no other tag of the same family and framing closes or opens.
class MarkupBalance {
public:
    void add(const MarkupTag& tag) noexcept {
        const std::size_t slot = slot_of(tag.family, tag.framed);
        if (!tag.closing) {
            ++open_[slot];
        } else if (open_[slot] != 0) {
            --open_[slot];
        } else {
            ++stray_[slot];
        }
    }

    [[nodiscard]] std::uint32_t unmatched(bool framed) const noexcept {
        std::uint32_t total = 0;
        for (std::size_t family = 0; family < kTagFamilyCount; ++family) {
            const std::size_t slot = slot_of(static_cast<TagFamily>(family), framed);
            total += open_[slot] + stray_[slot];
        }
        return total;
    }

    // Closes without an opener only accumulate as a value grows, so this bounds every longer value.
    [[nodiscard]] std::uint32_t stray(bool framed) const noexcept {
        std::uint32_t total = 0;
        for (std::size_t family = 0; family < kTagFamilyCount; ++family) {
            total += stray_[slot_of(static_cast<TagFamily>(family), framed)];
        }
        return total;
    }

private:
    static constexpr std::size_t slot_of(TagFamily family, bool framed) noexcept {
        return static_cast<std::size_t>(family) * 2U + (framed ? 1U : 0U);
    }

    std::array<std::uint32_t, 2 * kTagFamilyCount> open_{};
    std::array<std::uint32_t, 2 * kTagFamilyCount> stray_{};
};

struct IgnoreCalls {
    void open(std::string_view) noexcept {}

    void close() noexcept {}

    void drop() noexcept {}
};

// Positions of the `<tool_call>` wrappers a reading holds inside its parameter values.
class NestedWrapperRecorder {
public:
    explicit NestedWrapperRecorder(std::string_view text) noexcept : text_(text) {}

    void open(std::string_view) noexcept { call_begin_ = positions_.size(); }

    void parameter(std::string_view, std::string_view value) {
        const auto base = static_cast<std::size_t>(value.data() - text_.data());
        for (std::size_t pos = value.find(kToolOpen); pos != kNpos;
             pos             = value.find(kToolOpen, pos + 1)) {
            positions_.push_back(base + pos);
        }
    }

    void close() noexcept {}

    void drop() noexcept { positions_.resize(call_begin_); }

    [[nodiscard]] std::vector<std::size_t> take_positions() noexcept {
        return std::move(positions_);
    }

private:
    std::string_view text_;
    std::vector<std::size_t> positions_;
    std::size_t call_begin_ = 0;
};

class CallRecorder {
public:
    void open(std::string_view name) {
        call_         = RawToolCall{.name = name};
        call_repairs_ = 0;
    }

    void parameter(std::string_view name, std::string_view value) {
        // Last occurrence wins, as it would in JSON object syntax, rather than discarding an
        // otherwise well-formed call.
        const auto existing =
            std::find_if(call_->parameters.begin(), call_->parameters.end(),
                         [&](const RawParameter& candidate) { return candidate.name == name; });
        if (existing != call_->parameters.end()) {
            existing->value = value;
            ++call_repairs_;
        } else {
            call_->parameters.push_back(RawParameter{.name = name, .value = value});
        }
    }

    void close() {
        calls_.push_back(std::move(*call_));
        call_.reset();
        duplicate_repairs_ += call_repairs_;
        call_repairs_ = 0;
    }

    void drop() noexcept {
        call_.reset();
        call_repairs_ = 0;
    }

    [[nodiscard]] std::vector<RawToolCall> take_calls() noexcept { return std::move(calls_); }

    [[nodiscard]] std::uint32_t duplicate_repairs() const noexcept { return duplicate_repairs_; }

private:
    std::optional<RawToolCall> call_;
    std::vector<RawToolCall> calls_;
    std::uint32_t call_repairs_      = 0;
    std::uint32_t duplicate_repairs_ = 0;
};

class QwenToolRegionParser {
public:
    struct Calls {
        std::vector<RawToolCall> calls;
        std::uint32_t duplicate_parameters_repaired = 0;
    };

    struct NestedWrapper {
        std::size_t begin = 0;
        // Unmatched markup between the reading's start and this wrapper, read as prose.
        ReadingCost prose;
    };

    QwenToolRegionParser(std::string_view text, std::size_t max_name_length,
                         const Contract& contract, bool tolerant)
        : text_(text), max_name_length_(max_name_length), contract_(contract), tolerant_(tolerant) {
        scan_markup();
    }

    // Best reading of the region that starts at `start`. A salvage reading (tolerant mode only)
    // drops a malformed suffix after a complete call or accepts closing tags cut at the region
    // end, and only for calls whose values carry no stray line-framed markup.
    Reading region(std::size_t start, bool salvage) {
        Tables& tables = prepared(salvage);
        IgnoreCalls ignore;
        return resolve(tables, walk(start, Phase::Item, Wrapper::Bare, salvage, ignore), salvage);
    }

    // Start of the first parameter value the region opens, or npos when it fails before one.
    [[nodiscard]] std::size_t first_value(std::size_t start) const {
        IgnoreCalls ignore;
        const Step step = walk(start, Phase::Item, Wrapper::Bare, false, ignore);
        return step.kind == Step::Kind::Value ? step.parameter.value_begin : kNpos;
    }

    // Materializes the reading region() selected for the same start and mode.
    Calls build(std::size_t start, bool salvage) {
        CallRecorder recorder;
        replay(start, salvage, recorder);
        return Calls{.calls                         = recorder.take_calls(),
                     .duplicate_parameters_repaired = recorder.duplicate_repairs()};
    }

    // `<tool_call>` wrappers inside the parameter values of the exact reading from `start`, each
    // with the unmatched markup of the text from `prose_begin` up to it.
    std::vector<NestedWrapper> nested_wrappers(std::size_t start, std::size_t prose_begin) {
        NestedWrapperRecorder recorder(text_);
        replay(start, false, recorder);
        std::vector<NestedWrapper> wrappers;
        MarkupBalance prose;
        auto tag = std::lower_bound(
            tags_.begin(), tags_.end(), prose_begin,
            [](const MarkupTag& markup, std::size_t pos) { return markup.begin < pos; });
        for (const std::size_t begin : recorder.take_positions()) {
            for (; tag != tags_.end() && tag->begin < begin; ++tag) { prose.add(*tag); }
            wrappers.push_back(NestedWrapper{.begin = begin,
                                             .prose = {.framed_markup = prose.unmatched(true),
                                                       .inline_markup = prose.unmatched(false)}});
        }
        return wrappers;
    }

private:
    enum class Phase : std::uint8_t { Item, Function, AfterFunction, AfterItem };

    struct Step {
        enum class Kind : std::uint8_t { End, Value };
        Kind kind = Kind::End;
        Reading reading;
        ParameterOpen parameter;
        std::uint8_t context    = 0;
        bool discard_on_failure = false;
    };

    struct Tables {
        bool prepared = false;
        std::vector<std::array<Reading, kContextCount>> tails;
        // Highest candidate index whose tail is valid in each context, or npos.
        std::array<std::size_t, kContextCount> last_valid_tail{};
        std::unordered_map<std::size_t, ValueChoices> values;
    };

    // Replays the reading region() selected for the same start and mode into `recorder`.
    template <typename Recorder>
    void replay(std::size_t start, bool salvage, Recorder& recorder) {
        Tables& tables = prepared(salvage);
        Step step      = walk(start, Phase::Item, Wrapper::Bare, salvage, recorder);
        while (step.kind == Step::Kind::Value) {
            ParameterOpen parameter  = step.parameter;
            const ValueChoice* value = &value_choices(tables, parameter, salvage)[step.context];
            if (!value->reading.valid) {
                recorder.drop();
                return;
            }
            for (;;) {
                const BoundaryCandidate& boundary = candidates_[value->candidate];
                recorder.parameter(
                    parameter.name,
                    text_.substr(parameter.value_begin, boundary.begin - parameter.value_begin));
                if (boundary.kind != BoundaryCandidate::Kind::Sibling) { break; }
                parameter = boundary.sibling;
                value     = &value_choices(tables, parameter, salvage)[step.context];
            }
            recorder.close();
            const BoundaryCandidate& last = candidates_[value->candidate];
            if (last.kind == BoundaryCandidate::Kind::RegionEnd) { return; }
            step = walk(last.function_end, Phase::AfterFunction, context_wrapper(step.context),
                        salvage, recorder);
        }
    }

    bool consume(std::size_t& pos, std::string_view token) const {
        if (!starts_with_at(text_, pos, token)) { return false; }
        pos += token.size();
        return true;
    }

    [[nodiscard]] FallbackReason cut_reason() const noexcept {
        return tolerant_ ? FallbackReason::TruncatedTail : FallbackReason::MalformedStructure;
    }

    [[nodiscard]] bool valid_parameter_name(std::string_view name) const {
        if (plain_parameter_name(name)) { return true; }
        if (name.empty()) { return false; }
        return std::any_of(contract_.tools.begin(), contract_.tools.end(), [&](const auto& tool) {
            return find_parameter_contract(tool, name) != nullptr;
        });
    }

    [[nodiscard]] std::optional<ParameterOpen> parameter_open_at(std::size_t pos) const {
        std::size_t header_begin = 0;
        bool short_form          = false;
        if (starts_with_at(text_, pos, "<parameter")) {
            header_begin = pos + 10;
        } else if (starts_with_at(text_, pos, "<param")) {
            header_begin = pos + 6;
            short_form   = true;
        } else {
            return std::nullopt;
        }
        const std::size_t header_end = tag_header_end(text_, header_begin);
        if (header_end == kNpos) { return std::nullopt; }
        const std::string_view name =
            extract_name_from_tag_header(text_.substr(header_begin, header_end - header_begin));
        if (!valid_parameter_name(name)) { return std::nullopt; }
        return ParameterOpen{.value_begin = header_end + 1, .name = name, .short_form = short_form};
    }

    // Function opener in strict syntax, recognized only to weigh literal markup inside values.
    [[nodiscard]] std::size_t function_tag_end(std::size_t pos) const {
        std::size_t header_begin = 0;
        if (starts_with_at(text_, pos, "<function")) {
            header_begin = pos + 9;
        } else if (starts_with_at(text_, pos, "<invoke")) {
            header_begin = pos + 7;
        } else {
            return kNpos;
        }
        const std::size_t header_end = tag_header_end(text_, header_begin);
        if (header_end == kNpos ||
            !valid_function_name(
                extract_name_from_tag_header(text_.substr(header_begin, header_end - header_begin)),
                max_name_length_)) {
            return kNpos;
        }
        return header_end + 1;
    }

    [[nodiscard]] FunctionOpenResult function_open_at(std::size_t pos) const {
        std::size_t header_begin = 0;
        bool invoke              = false;
        if (starts_with_at(text_, pos, "<function")) {
            header_begin = pos + 9;
        } else if (starts_with_at(text_, pos, "<invoke")) {
            header_begin = pos + 7;
            invoke       = true;
        } else if (tolerant_) {
            // Recover a malformed function opener: a dropped or doubled leading '<' or a leaked
            // ChatML turn marker before the keyword. A form that yields no valid name is rejected
            // below, so prose after a marker cannot pass.
            std::size_t scan = pos;
            while (scan < text_.size() && text_[scan] == '<') { ++scan; }
            if (starts_with_at(text_, scan, "|im_start|>")) { scan += 11; }
            if (starts_with_at(text_, scan, "function")) {
                scan += 8;
            } else if (starts_with_at(text_, scan, "invoke")) {
                scan += 6;
                invoke = true;
            } else {
                return {};
            }
            header_begin = scan;
        } else {
            return {};
        }

        std::size_t tag_end = tag_header_end(text_, header_begin);
        bool ws_boundary    = false;
        // Tolerant: the model sometimes drops the '>' after the function name (for example a name
        // followed directly by a newline and a parameter tag). Recover by scanning the identifier
        // run and accepting it when format whitespace separates it from the next '<' or end of
        // region.
        if (tolerant_) {
            std::size_t scan = header_begin;
            while (scan < text_.size() && text_[scan] == '=') { ++scan; }
            const std::size_t ident_begin = scan;
            while (scan < text_.size() && scan - header_begin < max_name_length_) {
                const char byte = text_[scan];
                if (!is_ascii_alphanumeric(byte) && byte != '_' && byte != '-') { break; }
                ++scan;
            }
            if (scan > ident_begin && scan < text_.size() && is_format_whitespace(text_[scan]) &&
                (tag_end == kNpos || scan < tag_end)) {
                std::size_t after = scan;
                skip_format_whitespace(text_, after);
                if (after >= text_.size() || text_[after] == '<') {
                    tag_end     = scan;
                    ws_boundary = true;
                }
            }
        }
        if (tag_end == kNpos || tag_end == header_begin) {
            return {.failure = FallbackReason::InvalidToolName};
        }
        const std::string_view name =
            extract_name_from_tag_header(text_.substr(header_begin, tag_end - header_begin));
        if (!valid_function_name(name, max_name_length_)) {
            return {.failure = FallbackReason::InvalidToolName};
        }
        // Strict mode rejects a name outside the declared tool set. Tolerant mode keeps an
        // otherwise well-formed call structured and leaves the identity judgment to the consumer:
        // leaking the raw region to content would turn a valid call into prose.
        if (!tolerant_ && contract_.enforce_declared_names &&
            find_tool_contract(contract_, name) == nullptr) {
            return {.failure = FallbackReason::UndeclaredTool};
        }
        return {.open = FunctionOpen{.body_begin = ws_boundary ? tag_end : tag_end + 1,
                                     .name       = name,
                                     .invoke     = invoke}};
    }

    // The template writes every structural tag on a line of its own.
    [[nodiscard]] bool line_framed(std::size_t begin, std::size_t end) const noexcept {
        const bool line_start = begin == 0 || text_[begin - 1] == '\n';
        const bool line_end =
            end == text_.size() || text_[end] == '\n' ||
            (text_[end] == '\r' && (end + 1 == text_.size() || text_[end + 1] == '\n'));
        return line_start && line_end;
    }

    std::size_t add_candidate(std::size_t begin, std::size_t end) {
        std::size_t next = end;
        skip_format_whitespace(text_, next);
        BoundaryCandidate candidate{.begin = begin};
        if (next == text_.size()) {
            candidate.kind = BoundaryCandidate::Kind::RegionEnd;
        } else if (const std::optional<ParameterOpen> sibling = parameter_open_at(next)) {
            candidate.kind    = BoundaryCandidate::Kind::Sibling;
            candidate.sibling = *sibling;
        } else if (starts_with_at(text_, next, kFunctionClose)) {
            candidate.kind         = BoundaryCandidate::Kind::FunctionEnd;
            candidate.function_end = next + kFunctionClose.size();
        } else if (starts_with_at(text_, next, kInvokeClose)) {
            candidate.kind         = BoundaryCandidate::Kind::FunctionEnd;
            candidate.function_end = next + kInvokeClose.size();
            candidate.invoke       = true;
        } else {
            return kNpos;
        }
        candidates_.push_back(candidate);
        return candidates_.size() - 1;
    }

    void scan_markup() {
        std::size_t pos = text_.find('<');
        while (pos != kNpos) {
            MarkupTag tag{.begin = pos};
            std::size_t end  = kNpos;
            const auto fixed = [&](std::string_view marker, TagFamily family, bool closing,
                                   bool short_form = false) {
                if (!starts_with_at(text_, pos, marker)) { return false; }
                tag.family     = family;
                tag.closing    = closing;
                tag.short_form = short_form;
                end            = pos + marker.size();
                return true;
            };
            if (fixed(kToolClose, TagFamily::ToolCall, true) ||
                fixed(kToolOpen, TagFamily::ToolCall, false) ||
                fixed(kContainerClose, TagFamily::Container, true) ||
                fixed(kContainerOpen, TagFamily::Container, false) ||
                fixed(kFunctionClose, TagFamily::Function, true) ||
                fixed(kInvokeClose, TagFamily::Function, true) ||
                fixed(kParameterClose, TagFamily::Parameter, true) ||
                fixed(kParamClose, TagFamily::Parameter, true, true)) {
            } else if (const std::optional<ParameterOpen> parameter = parameter_open_at(pos)) {
                tag.family     = TagFamily::Parameter;
                tag.short_form = parameter->short_form;
                end            = parameter->value_begin;
            } else if (const std::size_t body = function_tag_end(pos); body != kNpos) {
                tag.family = TagFamily::Function;
                end        = body;
            } else {
                pos = text_.find('<', pos + 1);
                continue;
            }
            tag.framed = line_framed(pos, end);
            if (tag.family == TagFamily::Parameter && tag.closing) {
                tag.candidate = add_candidate(pos, end);
            }
            tags_.push_back(tag);
            pos = text_.find('<', end);
        }
    }

    // Walks the deterministic structure between two parameter values: wrapper tags, parameterless
    // calls and the next function opener. It stops at the next value, whose close is a choice, or
    // at the end of the reading.
    template <typename Recorder>
    Step walk(std::size_t pos, Phase phase, Wrapper wrapper, bool salvage,
              Recorder& recorder) const {
        bool after_call = phase == Phase::AfterFunction || phase == Phase::AfterItem;
        const auto stop = [&](FallbackReason failure) {
            if (salvage && after_call) {
                recorder.drop();
                return Step{.reading = accepted({.discarded_suffix = 1})};
            }
            return Step{.reading = rejected(failure)};
        };
        for (;;) {
            switch (phase) {
            case Phase::AfterItem:
                skip_format_whitespace(text_, pos);
                if (pos == text_.size()) { return Step{.reading = accepted()}; }
                phase = Phase::Item;
                break;
            case Phase::Item:
                if (consume(pos, kToolOpen)) {
                    wrapper = Wrapper::ToolCall;
                } else if (consume(pos, kContainerOpen)) {
                    wrapper = Wrapper::Container;
                } else if (starts_with_at(text_, pos, "<function") ||
                           starts_with_at(text_, pos, "<invoke")) {
                    wrapper = Wrapper::Bare;
                } else {
                    return stop(after_call ? FallbackReason::TrailingContent
                                           : FallbackReason::MalformedStructure);
                }
                skip_format_whitespace(text_, pos);
                phase = Phase::Function;
                break;
            case Phase::Function: {
                const FunctionOpenResult opened = function_open_at(pos);
                if (!opened.open) { return stop(opened.failure); }
                recorder.open(opened.open->name);
                pos = opened.open->body_begin;
                skip_format_whitespace(text_, pos);
                if (consume(pos, opened.open->invoke ? kInvokeClose : kFunctionClose)) {
                    recorder.close();
                    after_call = true;
                    phase      = Phase::AfterFunction;
                    break;
                }
                if (const std::optional<ParameterOpen> parameter = parameter_open_at(pos)) {
                    return Step{.kind               = Step::Kind::Value,
                                .parameter          = *parameter,
                                .context            = context_of(opened.open->invoke, wrapper),
                                .discard_on_failure = salvage && after_call};
                }
                return stop(pos == text_.size() ? cut_reason()
                                                : FallbackReason::MalformedStructure);
            }
            case Phase::AfterFunction:
                after_call = true;
                if (wrapper == Wrapper::Bare) {
                    phase = Phase::AfterItem;
                    break;
                }
                skip_format_whitespace(text_, pos);
                if (consume(pos, wrapper == Wrapper::ToolCall ? kToolClose : kContainerClose)) {
                    phase = Phase::AfterItem;
                    break;
                }
                if (pos == text_.size()) {
                    return Step{.reading = salvage ? accepted({.missing_close = 1})
                                                   : rejected(cut_reason())};
                }
                if (wrapper == Wrapper::Container) {
                    phase = Phase::Function;
                    break;
                }
                return stop(cut_reason());
            }
        }
    }

    Reading resolve(Tables& tables, const Step& step, bool salvage) {
        if (step.kind == Step::Kind::End) { return step.reading; }
        const Reading& value = value_choices(tables, step.parameter, salvage)[step.context].reading;
        if (!value.valid && step.discard_on_failure) { return accepted({.discarded_suffix = 1}); }
        return value;
    }

    // Best close for a value in every context. It reads only the tails of later candidates, which
    // prepared() fills from the end of the text backward, so no evaluation recurses.
    const ValueChoices& value_choices(Tables& tables, const ParameterOpen& parameter,
                                      bool salvage) {
        if (const auto found = tables.values.find(parameter.value_begin);
            found != tables.values.end()) {
            return found->second;
        }
        ValueChoices choices{};
        std::array<bool, kContextCount> explained{};
        // A context is settled once no later close can improve it: every longer value keeps at
        // least the stray closes seen so far (ties keep the earlier close), or no later candidate
        // has a valid tail.
        std::array<bool, kContextCount> settled{};
        bool candidate_seen = false;
        MarkupBalance balance;
        const auto first =
            std::lower_bound(tags_.begin(), tags_.end(), parameter.value_begin,
                             [](const MarkupTag& tag, std::size_t pos) { return tag.begin < pos; });
        for (auto tag = first; tag != tags_.end(); ++tag) {
            if (tag->candidate != kNpos && tag->short_form == parameter.short_form) {
                candidate_seen = true;
                const ReadingCost value_cost{.framed_markup = balance.unmatched(true),
                                             .inline_markup = balance.unmatched(false)};
                const ReadingCost later_bound{.framed_markup = balance.stray(true),
                                              .inline_markup = balance.stray(false)};
                const bool admissible = !salvage || value_cost.framed_markup == 0;
                bool open_context     = false;
                for (std::size_t context = 0; context < kContextCount; ++context) {
                    if (settled[context]) { continue; }
                    ValueChoice& choice = choices[context];
                    const Reading& tail = tables.tails[tag->candidate][context];
                    if (admissible && tail.valid) {
                        const ReadingCost total = value_cost + tail.cost;
                        if (!choice.reading.valid || total < choice.reading.cost) {
                            choice = ValueChoice{.reading   = accepted(total),
                                                 .candidate = tag->candidate};
                        }
                    } else if (!choice.reading.valid && !explained[context]) {
                        choice.reading.failure =
                            tail.valid ? FallbackReason::MalformedStructure : tail.failure;
                        explained[context] = true;
                    }
                    const std::size_t last = tables.last_valid_tail[context];
                    settled[context] =
                        (choice.reading.valid && choice.reading.cost <= later_bound) ||
                        last == kNpos || last <= tag->candidate;
                    open_context = open_context || !settled[context];
                }
                if (!open_context) { break; }
            }
            balance.add(*tag);
            // Salvage admits only values free of line-framed markup, and a stray close stays
            // unmatched in every longer value.
            if (salvage && balance.stray(true) != 0) { break; }
        }
        if (!candidate_seen) {
            for (ValueChoice& choice : choices) { choice.reading.failure = cut_reason(); }
        }
        return tables.values.emplace(parameter.value_begin, choices).first->second;
    }

    Reading tail(Tables& tables, const BoundaryCandidate& candidate, std::uint8_t context,
                 bool salvage) {
        switch (candidate.kind) {
        case BoundaryCandidate::Kind::Sibling:
            return value_choices(tables, candidate.sibling, salvage)[context].reading;
        case BoundaryCandidate::Kind::FunctionEnd: {
            if (candidate.invoke != context_invoke(context)) {
                return rejected(FallbackReason::MalformedStructure);
            }
            IgnoreCalls ignore;
            return resolve(tables,
                           walk(candidate.function_end, Phase::AfterFunction,
                                context_wrapper(context), salvage, ignore),
                           salvage);
        }
        case BoundaryCandidate::Kind::RegionEnd:
            return salvage ? accepted({.missing_close = 1}) : rejected(cut_reason());
        }
        return rejected(FallbackReason::MalformedStructure);
    }

    Tables& prepared(bool salvage) {
        Tables& tables = salvage ? salvage_ : exact_;
        if (tables.prepared) { return tables; }
        tables.tails.assign(candidates_.size(), {});
        tables.last_valid_tail.fill(kNpos);
        for (std::size_t index = candidates_.size(); index-- != 0;) {
            for (std::uint8_t context = 0; context < kContextCount; ++context) {
                tables.tails[index][context] = tail(tables, candidates_[index], context, salvage);
            }
            for (std::size_t context = 0; context < kContextCount; ++context) {
                if (tables.tails[index][context].valid &&
                    tables.last_valid_tail[context] == kNpos) {
                    tables.last_valid_tail[context] = index;
                }
            }
        }
        tables.prepared = true;
        return tables;
    }

    std::string_view text_;
    std::size_t max_name_length_;
    const Contract& contract_;
    bool tolerant_ = false;
    std::vector<MarkupTag> tags_;
    std::vector<BoundaryCandidate> candidates_;
    Tables exact_;
    Tables salvage_;
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
    const std::size_t first_marker = find_first_tool_marker(source);
    if (first_marker == kNpos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.diagnostics.marker_seen = true;

    // Generated prose can quote a tool-call marker before the real turn. Try the first marker, then
    // each later `<tool_call>` wrapper, and accept the first region that parses; earlier markers
    // stay ordinary content. Retries move only to a later wrapper, and only past a region that
    // failed before its first value or whose leading calls still close cleanly: after a value that
    // never closes, the following markup may be that value's content, and a nested example must
    // not be re-read as the call. Every exact reading is preferred over any tolerant salvage.
    QwenToolRegionParser parser(source, max_tool_name_length, contract, tolerant);
    const auto next_region = [&](std::size_t region) {
        const bool self_contained =
            parser.first_value(region) == kNpos || parser.region(region, true).valid;
        return self_contained ? source.find(kToolOpen, region + 1) : kNpos;
    };
    std::optional<FallbackReason> first_failure;
    std::size_t accepted = kNpos;
    bool salvaged        = false;
    for (std::size_t region = first_marker; region != kNpos; region = next_region(region)) {
        const Reading reading = parser.region(region, false);
        if (reading.valid) {
            accepted = region;
            break;
        }
        if (!first_failure) { first_failure = reading.failure; }
    }
    // A value of the accepted reading can hold a later wrapper whose own reading also parses: the
    // value then swallowed the real call after a call or opener that prose quoted. Each such
    // wrapper is weighed with the text before it read as prose. The first one with strictly less
    // unmatched markup in total replaces the reading, whose own values are examined in turn, so a
    // tie never lets a nested example displace the call holding it.
    if (accepted != kNpos) {
        const std::size_t prose_begin = accepted;
        ReadingCost best              = parser.region(accepted, false).cost;
        for (bool moved = best != ReadingCost{}; moved;) {
            moved = false;
            for (const auto& nested : parser.nested_wrappers(accepted, prose_begin)) {
                const Reading reading = parser.region(nested.begin, false);
                if (reading.valid && nested.prose + reading.cost < best) {
                    best     = nested.prose + reading.cost;
                    accepted = nested.begin;
                    moved    = true;
                    break;
                }
            }
        }
    }
    if (accepted == kNpos && tolerant) {
        for (std::size_t region = first_marker; region != kNpos; region = next_region(region)) {
            if (parser.region(region, true).valid) {
                accepted = region;
                salvaged = true;
                break;
            }
        }
    }
    if (accepted == kNpos) {
        // No region parsed, so the response is returned as text with the first region's reason.
        out.diagnostics.fallback_reason =
            first_failure.value_or(FallbackReason::MalformedStructure);
        return fallback(text, out.diagnostics);
    }
    // A salvaged region keeps its reason for transparency without demoting the output.
    out.diagnostics.fallback_reason =
        salvaged ? FallbackReason::TruncatedTail : FallbackReason::None;

    const QwenToolRegionParser::Calls raw = parser.build(accepted, salvaged);
    out.content = rtrim_format_whitespace(source.substr(0, accepted));
    out.tool_calls.reserve(raw.calls.size());
    for (const RawToolCall& call : raw.calls) {
        out.tool_calls.push_back(normalize_raw_tool_call(call, contract, out.diagnostics));
    }

    out.diagnostics.duplicate_parameters_repaired = raw.duplicate_parameters_repaired;
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
