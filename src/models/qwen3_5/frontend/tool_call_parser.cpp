#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "models/qwen3_5/frontend/tool_call_grammar.h"
#include "models/qwen3_5/frontend/tool_call_stream.h"

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
    return is_tool_format_whitespace(byte);
}

constexpr bool is_ascii_digit(char byte) { return byte >= '0' && byte <= '9'; }


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

GeneratedToolCall normalize_parsed_tool_call(const ParsedFunctionCall& raw, const Contract& contract,
                                          ToolCallParseDiagnostics& diagnostics) {
    const Contract::Tool* tool = find_tool_contract(contract, raw.name);
    if (tool != nullptr && !tool->unambiguous) { tool = nullptr; }

    std::string arguments = "{";
    bool first            = true;
    for (const ParsedParameter& raw_parameter : raw.parameters) {
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

bool declared_tool_name_check(const void* contract, std::string_view name) {
    return find_tool_contract(*static_cast<const Contract*>(contract), name) != nullptr;
}

// R3-01 plausibility: a parameter boundary is implausible exactly when the named tool has an
// unambiguous declared schema with at least one property and the name is not declared.
bool declared_parameter_plausible(const void* contract, std::string_view tool_name,
                                  std::string_view param_name) {
    const Contract& c = *static_cast<const Contract*>(contract);
    const Contract::Tool* tool = find_tool_contract(c, tool_name);
    if (tool == nullptr || !tool->unambiguous || tool->parameters.empty()) { return true; }
    return std::any_of(tool->parameters.begin(), tool->parameters.end(),
                       [&](const Contract::Parameter& p) { return p.name == param_name; });
}

ToolCallParseFallbackReason to_fallback_reason(ToolCallParseFailure failure) {
    switch (failure) {
        case ToolCallParseFailure::None:               return ToolCallParseFallbackReason::None;
        case ToolCallParseFailure::MalformedStructure: return ToolCallParseFallbackReason::MalformedStructure;
        case ToolCallParseFailure::InvalidToolName:    return ToolCallParseFallbackReason::InvalidToolName;
        case ToolCallParseFailure::UndeclaredTool:     return ToolCallParseFallbackReason::UndeclaredTool;
        case ToolCallParseFailure::TrailingContent:    return ToolCallParseFallbackReason::TrailingContent;
        case ToolCallParseFailure::TruncatedTail:      return ToolCallParseFallbackReason::TruncatedTail;
        case ToolCallParseFailure::AmbiguousStructure: return ToolCallParseFallbackReason::AmbiguousStructure;
    }
    return ToolCallParseFallbackReason::MalformedStructure;
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

MaterializedToolCallResult materialize_tool_call_result(const ToolCallStreamResult& result,
                                                        const ToolCallOutputContract& contract) {
    MaterializedToolCallResult materialized;
    if (!result.marker_seen) {
        // R3-09: no marker was ever seen: default diagnostics (fallback None, marker_seen
        // false) — the fence fields stay visible for the operational log.
        ToolCallParseDiagnostics diagnostics;
        diagnostics.fenced_markers_suppressed = result.fenced_markers_suppressed;
        diagnostics.indented_markers_suppressed = result.indented_markers_suppressed;
        diagnostics.ended_in_unclosed_fence   = result.ended_in_unclosed_fence;
        materialized.diagnostics = std::move(diagnostics);
        return materialized;
    }
    const ToolCallParseFallbackReason failure = to_fallback_reason(result.failure);
    if (result.status != ToolCallStreamStatus::Complete) {
        // No region parsed. A truncated tail that kept no call carries no arguments either, so
        // the response is returned as text with the first region's reason recorded.
        ToolCallParseDiagnostics diagnostics;
        diagnostics.marker_seen     = true;
        diagnostics.fallback_reason = failure;
        diagnostics.fenced_markers_suppressed = result.fenced_markers_suppressed;
        diagnostics.indented_markers_suppressed = result.indented_markers_suppressed;
        diagnostics.ended_in_unclosed_fence   = result.ended_in_unclosed_fence;
        diagnostics.parse_budget_exhausted      = result.parse_budget_exhausted;
        materialized.diagnostics = std::move(diagnostics);
        return materialized;
    }
    // R2-I5 (CR5) defense in depth: identity is a property of the output, not a side effect
    // of one parse branch. When the contract enforces declared names, no call may leave the
    // frontend with a name outside the declared set — the state-machine branch is the primary
    // enforcement, this check is the output boundary (a graceful fallback, never a crash).
    if (contract.enforce_declared_names) {
        for (const ParsedFunctionCall& call : result.region.calls) {
            if (find_tool_contract(contract, call.name) == nullptr) {
                ToolCallParseDiagnostics diagnostics;
                diagnostics.marker_seen     = true;
                diagnostics.fallback_reason = ToolCallParseFallbackReason::UndeclaredTool;
                diagnostics.fenced_markers_suppressed = result.fenced_markers_suppressed;
                diagnostics.indented_markers_suppressed = result.indented_markers_suppressed;
                diagnostics.ended_in_unclosed_fence   = result.ended_in_unclosed_fence;
                diagnostics.parse_budget_exhausted = result.parse_budget_exhausted;
                materialized.diagnostics = std::move(diagnostics);
                return materialized;
            }
        }
    }

    // R3-08: the single output boundary for parameter occurrences (after a stage accepted
    // the region, before normalization). A declared tool with an unambiguous non-empty
    // schema: a repeated parameter name, or a non-first parameter outside the declared
    // schema, makes the region ambiguous — it is returned verbatim as text. Undeclared
    // tools keep the legacy JSON-object rule (last wins), and its count is kept.
    std::vector<ParsedFunctionCall> calls = result.region.calls;
    bool ambiguous_region = false;
    std::uint32_t legacy_repairs = 0;
    for (ParsedFunctionCall& call : calls) {
        bool duplicate = false;
        for (std::size_t i = 0; i < call.parameters.size() && !duplicate; ++i) {
            for (std::size_t j = i + 1; j < call.parameters.size(); ++j) {
                if (call.parameters[i].name == call.parameters[j].name) {
                    duplicate = true;
                    break;
                }
            }
        }
        const Contract::Tool* tool = find_tool_contract(contract, call.name);
        if (tool != nullptr) {
            if (duplicate) { ambiguous_region = true; break; }
            if (tool->unambiguous && !tool->parameters.empty()) {
                for (std::size_t i = 1; i < call.parameters.size() && !ambiguous_region; ++i) {
                    const std::string_view name = call.parameters[i].name;
                    if (!std::any_of(tool->parameters.begin(), tool->parameters.end(),
                                     [&](const Contract::Parameter& p) { return p.name == name; })) {
                        ambiguous_region = true;
                    }
                }
            }
        } else if (duplicate) {
            // Legacy JSON-object rule for tools the contract does not declare: last wins.
            std::vector<ParsedParameter> merged;
            merged.reserve(call.parameters.size());
            for (const ParsedParameter& parameter : call.parameters) {
                const auto existing =
                    std::find_if(merged.begin(), merged.end(),
                                 [&](const ParsedParameter& p) { return p.name == parameter.name; });
                if (existing == merged.end()) { merged.push_back(parameter); }
                else { *existing = parameter; ++legacy_repairs; }
            }
            call.parameters = std::move(merged);
        }
    }
    if (ambiguous_region) {
        ToolCallParseDiagnostics diagnostics;
        diagnostics.marker_seen     = true;
        diagnostics.fallback_reason = FallbackReason::AmbiguousStructure;
        diagnostics.fenced_markers_suppressed = result.fenced_markers_suppressed; // N-06 item 3
        diagnostics.indented_markers_suppressed = result.indented_markers_suppressed; // R10-03
        diagnostics.ended_in_unclosed_fence   = result.ended_in_unclosed_fence;     // N-06 item 3
        diagnostics.parse_budget_exhausted = result.parse_budget_exhausted;
        materialized.diagnostics = std::move(diagnostics);
        return materialized;
    }

    materialized.accepted = true;
    // A recovered truncated tail keeps its reason for transparency without demoting the output.
    materialized.diagnostics.marker_seen     = true;
    materialized.diagnostics.fallback_reason = failure;
    materialized.diagnostics.markup_tolerant_completion = result.markup_tolerant_completion;
    materialized.diagnostics.fenced_markers_suppressed  = result.fenced_markers_suppressed;
    materialized.diagnostics.indented_markers_suppressed = result.indented_markers_suppressed;
    materialized.diagnostics.ended_in_unclosed_fence    = result.ended_in_unclosed_fence;
    materialized.diagnostics.parse_budget_exhausted   = result.parse_budget_exhausted;
    // R13-03: tolerant mode turned output the strict parser would return as text into
    // structured calls. Only tolerant paths can set the truncated-tail reason on an
    // accepted region or the per-call repaired flag, so the expression is false in strict
    // mode by construction. An undeclared tool is never accepted (R2-I5), so it can never
    // make the flag true — a deliberate difference from the pre-merge parser.
    materialized.diagnostics.tolerant_recovered =
        failure == ToolCallParseFallbackReason::TruncatedTail ||
        std::any_of(calls.begin(), calls.end(),
                    [](const ParsedFunctionCall& c) { return c.repaired; });

    materialized.calls.reserve(calls.size());
    for (const ParsedFunctionCall& call : calls) {
        materialized.calls.push_back(
            normalize_parsed_tool_call(call, contract, materialized.diagnostics));
    }

    materialized.diagnostics.duplicate_parameters_repaired = legacy_repairs;
    materialized.diagnostics.structured_call_count         =
        static_cast<std::uint32_t>(materialized.calls.size());
    return materialized;
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolCallOutputContract& contract,
                                                 bool tolerant,
                                                 FinishReason finish_reason,
                                                 ToolCallSyntaxMode syntax,
                                                 ToolCallAmbiguityPolicy ambiguity,
                                                 ToolCallIntentPolicy intent) {
    ToolCallParsePolicy policy;
    policy.max_name_length        = max_tool_name_length;
    policy.tolerant               = tolerant;
    policy.enforce_declared_names = contract.enforce_declared_names;
    policy.declared_check         = declared_tool_name_check;
    policy.contract               = &contract;
    policy.parameter_plausible    = declared_parameter_plausible;
    policy.syntax                 = syntax;
    policy.ambiguity              = ambiguity;
    policy.intent                 = intent;

    // One-shot and streaming share the same incremental parser: this feeds the whole output
    // and finishes once; the streaming decoder feeds chunks of the same machine.
    ToolCallStreamParser machine(policy);
    machine.feed(text);
    const ToolCallStreamResult result = machine.finish(finish_reason);
    // R11-03: one shared contract-aware materialization for one-shot and streaming.
    const MaterializedToolCallResult materialized =
        materialize_tool_call_result(result, contract);
    if (!materialized.accepted) {
        // Fallback: the complete original text is returned verbatim (one-shot has published
        // nothing yet).
        return fallback(text, materialized.diagnostics);
    }
    ParsedToolCallOutput out;
    out.diagnostics = materialized.diagnostics;
    // Generated prose can quote a tool-call marker before the real turn. Bytes before the
    // accepted region (prose plus any failed earlier region) stay ordinary content.
    out.content = rtrim_format_whitespace(machine.content_prefix() + result.tail);
    out.tool_calls = std::move(materialized.calls);
    out.is_tool_call_response = true;
    return out;
}

ToolCallOutputDecoder::ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                                             std::size_t max_tool_name_length, bool tolerant,
                                             ToolCallSyntaxMode syntax,
                                             ToolCallAmbiguityPolicy ambiguity,
                                             ToolCallIntentPolicy intent)
    : contract_(std::move(contract)), max_tool_name_length_(max_tool_name_length),
      tolerant_(tolerant), syntax_(syntax), ambiguity_(ambiguity), intent_(intent) {
    // R11-03 (R11-I6): the live machine carries the full contract-aware policy. The terminal
    // path finalizes this same machine — there is no second, contextless region re-parse —
    // so the entry classification and the terminal materialization run on one machine over
    // the same text. An empty contract disables contract-awareness (null pointer).
    ToolCallParsePolicy machine_policy;
    machine_policy.max_name_length        = max_tool_name_length_;
    machine_policy.tolerant               = tolerant_;
    machine_policy.enforce_declared_names = contract_ && contract_->enforce_declared_names;
    machine_policy.declared_check         = declared_tool_name_check;
    machine_policy.contract               = contract_ ? contract_.get() : nullptr;
    machine_policy.parameter_plausible    = declared_parameter_plausible;
    machine_policy.syntax                 = syntax_;
    machine_policy.ambiguity              = ambiguity_;
    machine_policy.intent                 = intent_;
    machine_ = ToolCallStreamParser(machine_policy);
}

std::string ToolCallOutputDecoder::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    if (text.empty()) { return {}; }
    if (!contract_) { return std::string(text); }
    return machine_.feed(text);
}

ToolCallOutputDecoder::Terminal ToolCallOutputDecoder::finish(FinishReason finish_reason) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    finished_ = true;
    if (!contract_) { return {}; }

    // R11-03 (R11-I9): finalize the live machine. It already classified every entry over the
    // full physical-line context (pre-latch bytes included) and carries the contract-aware
    // policy; there is no second region-only re-parse. The shared materialization applies the
    // exact one-shot boundary (declared-name defense in depth, the R3-08 ambiguity rule,
    // schema-aware normalization), so one-shot and streaming outcomes cannot drift.
    const ToolCallStreamResult result = machine_.finish(finish_reason);
    const MaterializedToolCallResult materialized =
        materialize_tool_call_result(result, *contract_);

    if (materialized.accepted && machine_.latched()) {
        // The machine's tail is the held text between the first latched marker and the
        // accepted region — the only bytes the decoder has not published yet (everything
        // before the first latch already left through feed()).
        return Terminal{.content     = rtrim_format_whitespace(result.tail),
                        .tool_calls  = std::move(materialized.calls),
                        .diagnostics = materialized.diagnostics};
    }

    // Fallback: publish the held bytes verbatim. Latched-but-unparsable — or latched but
    // rejected by the materialization boundary (the ambiguity rule) after Stage 1 accepted:
    // the whole latched region. result.tail is the bytes before the accepted base and is
    // empty when Stage 1 accepted at base zero, so it must not stand in for the region.
    // Never latched: only the held candidate tail (the pre-latch content was already
    // published through feed()).
    const std::string tail =
        machine_.latched() ? std::string(machine_.latched_region()) : machine_.held_tail();
    return Terminal{.content = std::move(tail), .tool_calls = {},
                    .diagnostics = materialized.diagnostics};
}

} // namespace ninfer::models::qwen3_5::frontend
