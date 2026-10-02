#include "models/qwen3_5/frontend/tool_call_stream.h"


#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "models/qwen3_5/frontend/tool_call_grammar_state.h"
#include <nlohmann/json.hpp>

#include <initializer_list>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace fi = ninfer::models::qwen3_5::frontend;

const fi::ToolCallOutputContract kLegacyContract;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

std::string tool_definition(const std::string& tool_name, Json properties,
                            Json required = Json::array()) {
    Json parameters{{"type", "object"}, {"properties", std::move(properties)}};
    if (!required.empty()) { parameters["required"] = std::move(required); }
    return Json{{"type", "function"},
                {"function", Json{{"name", tool_name}, {"parameters", std::move(parameters)}}}}
        .dump();
}

std::shared_ptr<const fi::ToolCallOutputContract>
contract_from_definitions(const std::vector<std::string>& definitions) {
    return fi::build_tool_call_output_contract(
        std::span<const std::string>(definitions.data(), definitions.size()), true);
}

std::shared_ptr<const fi::ToolCallOutputContract> output_contract_for(const std::string& tool_name,
                                                                      Json properties) {
    const std::vector<std::string> definitions = {
        tool_definition(tool_name, std::move(properties))};
    return contract_from_definitions(definitions);
}

fi::ToolCallOutputContract contract_for(const std::string& tool_name, Json properties) {
    return *output_contract_for(tool_name, std::move(properties));
}

std::string
tool_call(std::string_view tool_name,
          std::initializer_list<std::pair<std::string_view, std::string_view>> parameters = {}) {
    std::string text = "<tool_call>\n<function=";
    text.append(tool_name);
    text += ">\n";
    for (const auto& [name, value] : parameters) {
        text += "<parameter=";
        text.append(name);
        text += ">\n";
        text.append(value);
        text += "\n</parameter>\n";
    }
    text += "</function>\n</tool_call>";
    return text;
}

int check_rejected(const std::string& text, const fi::ToolCallOutputContract& contract,
                   ninfer::ToolCallParseFallbackReason reason, std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
    return check(!parsed.is_tool_call_response && parsed.content == text &&
                     parsed.tool_calls.empty() && parsed.diagnostics.marker_seen &&
                     parsed.diagnostics.fallback_reason == reason,
                 std::string(message));
}

int check_parameter_schema_mismatch(const fi::ToolCallOutputContract& contract,
                                    std::string_view parameter_name, std::string_view value,
                                    std::string_view expected_json_value,
                                    std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{parameter_name, value}}), 64, contract);
    const std::string expected_arguments = "{" + Json(std::string(parameter_name)).dump() + ":" +
                                           std::string(expected_json_value) + "}";
    return check(
        parsed.is_tool_call_response && parsed.content.empty() && parsed.tool_calls.size() == 1 &&
            parsed.tool_calls.front().arguments_json == expected_arguments &&
            parsed.diagnostics.marker_seen && parsed.diagnostics.structured_call_count == 1 &&
            parsed.diagnostics.schema_mismatch_arguments == 1 &&
            parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
        std::string(message));
}

int test_basic_legacy_parsing() {
    const auto parsed = fi::parse_qwen_tool_call_output("Calling weather.\n"
                                                        "<tool_call>\n"
                                                        "<function=get_weather>\n"
                                                        "<parameter=city>\nParis\n</parameter>\n"
                                                        "<parameter=days>\n2\n</parameter>\n"
                                                        "</function>\n"
                                                        "</tool_call>",
                                                        64, kLegacyContract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "legacy call was not parsed");
    failures += check(parsed.content == "Calling weather.", "content prefix was not trimmed");
    failures += check(parsed.tool_calls.size() == 1, "legacy call count changed");
    if (parsed.tool_calls.size() != 1) { return failures; }
    failures += check(parsed.tool_calls.front().name == "get_weather", "function name changed");
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("city") == "Paris", "legacy string inference changed");
    failures += check(args.at("days") == 2, "legacy JSON inference changed");
    return failures;
}

int test_multiple_calls() {
    const std::string text = tool_call("first", {{"payload", "{\"ok\":true,\"items\":[1,2]}"}}) +
                             "\n" + tool_call("second", {{"value", "plain text"}});
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2,
                      "multiple complete calls were not parsed");
    if (parsed.tool_calls.size() != 2) { return failures; }
    const Json first  = Json::parse(parsed.tool_calls[0].arguments_json);
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures +=
        check(first.at("payload").at("ok") == true && first.at("payload").at("items").at(1) == 2,
              "legacy object value changed");
    failures += check(second.at("value") == "plain text", "legacy plain text value changed");
    return failures;
}

int test_declared_strings_preserve_text() {
    const auto contract =
        contract_for("TaskUpdate",
                     Json{{"taskId", Json{{"type", "string"}}},
                          {"content", Json{{"type", "string"}}},
                          {"truthy", Json{{"type", "string"}}},
                          {"nullish", Json{{"type", "string"}}},
                          {"quoted", Json{{"type", "string"}}},
                          {"windows", Json{{"type", "string"}}},
                          {"string_or_number", Json{{"type", Json::array({"number", "string"})}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=TaskUpdate>\n"
                                        "<parameter=taskId>\n1\n</parameter>\n"
                                        "<parameter=content>\n  {\"x\":1}\n\n</parameter>\n"
                                        "<parameter=truthy>\ntrue\n</parameter>\n"
                                        "<parameter=nullish>\nnull\n</parameter>\n"
                                        "<parameter=quoted>\n\"literal\"\n</parameter>\n"
                                        "<parameter=windows>\r\n  value  \r\n</parameter>\n"
                                        "<parameter=string_or_number>\n7\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        128, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "declared string call was rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("taskId") == "1", "numeric-shaped string was promoted");
    failures +=
        check(args.at("content") == "  {\"x\":1}\n", "string whitespace or content changed");
    failures += check(args.at("truthy") == "true" && args.at("nullish") == "null",
                      "boolean/null-shaped string was promoted");
    failures +=
        check(args.at("quoted") == "\"literal\"", "quoted string was reinterpreted as JSON");
    failures += check(args.at("windows") == "  value  ", "CRLF framing changed string content");
    failures +=
        check(args.at("string_or_number") == "7", "string-admitting union did not preserve text");
    return failures;
}

int test_string_values_preserve_embedded_tool_markup() {
    const auto contract       = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                          {"timeout", Json{{"type", "integer"}}}});
    const std::string command = "python3 - <<'PY'\n"
                                "import re\n"
                                "pattern = r'<parameter=edits>\\n(.*?)\\n</parameter>'\n"
                                "print(pattern)\n"
                                "PY";
    const std::string text    = tool_call("bash", {{"command", command}, {"timeout", "30"}});
    const auto parsed         = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "balanced parameter markup inside a string broke the tool call");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("command") == command,
                      "embedded parameter markup was removed from the string value");
    failures +=
        check(args.at("timeout") == 30, "sibling parameter after embedded markup was not parsed");

    const std::string nested_markup =
        "literal closes: </function> and </tool_call>\n"
        "<function=fake>body</function>\n"
        "<tool_call>body</tool_call>\n"
        "<parameter=outer>before<parameter=inner>value</parameter>after</parameter>";
    const std::string nested_text = tool_call("bash", {{"command", nested_markup}});
    const auto nested             = fi::parse_qwen_tool_call_output(nested_text, 64, contract);
    failures += check(nested.is_tool_call_response && nested.tool_calls.size() == 1,
                      "nested tool markup inside a string broke outer structure");
    if (nested.tool_calls.size() == 1) {
        const Json nested_args = Json::parse(nested.tool_calls.front().arguments_json);
        failures += check(nested_args.at("command") == nested_markup,
                          "nested function/tool/parameter markup was not preserved exactly");
    }
    return failures;
}

int test_parameter_delimiters_in_values() {
    using Reason        = ninfer::ToolCallParseFallbackReason;
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                    {"timeout", Json{{"type", "integer"}}}});
    int failures        = 0;

    // A closer the grammar cannot continue from is text the value quotes: the call is kept whole,
    // whether the quote is followed by a sibling parameter or ends the call.
    for (const std::string command :
         {std::string("echo '</parameter>'"), std::string("echo '</parameter>' && ls\n</param>")}) {
        for (const bool sibling : {false, true}) {
            const std::string text =
                sibling ? tool_call("bash", {{"command", command}, {"timeout", "30"}})
                        : tool_call("bash", {{"command", command}});
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                              "a value quoting a parameter closer lost its tool call");
            if (parsed.tool_calls.size() != 1) { continue; }
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.at("command") == command,
                              "a quoted parameter closer was not kept in the value");
            failures += check(!sibling || args.at("timeout") == 30,
                              "the parameter after a quoted closer was not parsed");
        }
    }

    // Markup the grammar could continue from used to fall back as ambiguous. Inside a
    // <tool_call> wrapper the function close must be followed by the wrapper close (P3.10),
    // so a closer pair followed by anything else is text the value quotes: the value keeps
    // scanning and the call is parsed with the markup preserved exactly.
    const std::string quoted_pair =
        tool_call("bash", {{"command", "echo '</parameter></function>'"}});
    const auto pair_parsed = fi::parse_qwen_tool_call_output(quoted_pair, 64, contract);
    failures += check(pair_parsed.is_tool_call_response && pair_parsed.tool_calls.size() == 1,
                      "a quoted closer pair inside a wrapper was rejected as ambiguous");
    if (pair_parsed.tool_calls.size() == 1) {
        const Json pair_args = Json::parse(pair_parsed.tool_calls.front().arguments_json);
        failures += check(pair_args.at("command") == "echo '</parameter></function>'",
                          "a quoted closer pair was not preserved in the value");
    }

    // Tolerant truncation: a value the budget cut before its closer is never committed, and
    // a value closed without the function close is not executable (F2/I2): the open
    // function is never committed, whatever its argument bytes show.
    const std::string open  = std::string("<") + "parameter=command>\n";
    const std::string close = std::string("</") + "parameter>";
    const std::string head  = "<tool_call>\n<function=bash>\n" + open;
    const auto closed_cut =
        fi::parse_qwen_tool_call_output(head + "ls\n" + close + "\n", 64, contract, true);
    const auto quoted_cut =
        fi::parse_qwen_tool_call_output(head + "echo '" + close + "' more", 64, contract, true);
    failures += check(!closed_cut.is_tool_call_response && closed_cut.tool_calls.empty() &&
                          closed_cut.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "a tolerant cut call with a closed value was executed without its function close");
    failures += check(!quoted_cut.is_tool_call_response && quoted_cut.tool_calls.empty() &&
                          quoted_cut.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "a tolerant cut value was committed instead of falling back to text");
    return failures;
}

int test_declared_json_types() {
    const auto contract = contract_for(
        "configure", Json{{"count", Json{{"type", "integer"}}},
                          {"total", Json{{"type", "number"}}},
                          {"ratio", Json{{"type", "number"}}},
                          {"enabled", Json{{"type", "boolean"}}},
                          {"payload", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"unset", Json{{"type", "null"}}},
                          {"optional", Json{{"type", Json::array({"integer", "null"})}}}});
    const std::string text = tool_call("configure", {{"count", "7"},
                                                     {"total", "8"},
                                                     {"ratio", "1.5"},
                                                     {"enabled", "true"},
                                                     {"payload", "{\"x\":1}"},
                                                     {"items", "[\"a\",2]"},
                                                     {"unset", "null"},
                                                     {"optional", "null"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "valid declared JSON values were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("count") == 7, "integer was not decoded");
    failures += check(args.at("total") == 8, "integer did not satisfy number");
    failures += check(args.at("ratio") == 1.5, "fractional number was not decoded");
    failures += check(args.at("enabled") == true, "JSON boolean was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "object was not decoded");
    failures +=
        check(args.at("items").is_array() && args.at("items").at(1) == 2, "array was not decoded");
    failures += check(args.at("unset").is_null() && args.at("optional").is_null(),
                      "declared null was not decoded");
    return failures;
}

int test_boolean_boundary() {
    const auto contract =
        contract_for("configure", Json{{"lower_true", Json{{"type", "boolean"}}},
                                       {"title_true", Json{{"type", "boolean"}}},
                                       {"upper_true", Json{{"type", "boolean"}}},
                                       {"mixed_false", Json{{"type", "boolean"}}},
                                       {"spaced_true", Json{{"type", "boolean"}}},
                                       {"windows_false", Json{{"type", "boolean"}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=lower_true>\ntrue\n</parameter>\n"
                                        "<parameter=title_true>\nTrue\n</parameter>\n"
                                        "<parameter=upper_true>\nTRUE\n</parameter>\n"
                                        "<parameter=mixed_false>\nfAlSe\n</parameter>\n"
                                        "<parameter=spaced_true>\n \tTrUe \n</parameter>\n"
                                        "<parameter=windows_false>\r\nFaLsE\r\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "case-insensitive booleans were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("lower_true") == true && args.at("title_true") == true &&
                          args.at("upper_true") == true && args.at("spaced_true") == true,
                      "true variants were not canonicalized");
    failures += check(args.at("mixed_false") == false && args.at("windows_false") == false,
                      "false variants were not canonicalized");

    const auto one_flag = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    failures += check_parameter_schema_mismatch(one_flag, "flag", "1", "1",
                                                "integer boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "0", "0",
                                                "zero boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "\"true\"", "\"true\"",
                                                "string boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "yes", "\"yes\"",
                                                "plain boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "None", "\"None\"",
                                                "Python null mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "null", "null",
                                                "null boolean mismatch was not structured");
    return failures;
}

int test_exact_integer_boundary() {
    const auto integer_contract =
        contract_for("configure", Json{{"decimal", Json{{"type", "integer"}}},
                                       {"exponent", Json{{"type", "integer"}}},
                                       {"scaled", Json{{"type", "integer"}}},
                                       {"negative_zero", Json{{"type", "integer"}}},
                                       {"large", Json{{"type", "integer"}}}});
    const std::string valid = tool_call("configure", {{"decimal", "7.0"},
                                                      {"exponent", "1e2"},
                                                      {"scaled", "100e-2"},
                                                      {"negative_zero", "-0.0"},
                                                      {"large", "9007199254740992.0"}});
    const auto parsed       = fi::parse_qwen_tool_call_output(valid, 64, integer_contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "mathematically integral JSON numbers were rejected");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().arguments_json ==
                              "{\"decimal\":7.0,\"exponent\":1e2,\"scaled\":100e-2,"
                              "\"negative_zero\":-0.0,\"large\":9007199254740992.0}",
                          "integer JSON lexemes were rewritten");
    }

    const auto one_integer = contract_for("configure", Json{{"value", Json{{"type", "integer"}}}});
    failures += check_parameter_schema_mismatch(one_integer, "value", "7.5", "7.5",
                                                "fractional integer mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_integer, "value", "1e-1", "1e-1",
                                                "fractional exponent mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        one_integer, "value", "9007199254740992.5", "9007199254740992.5",
        "large fractional integer mismatch lost its exact lexeme");

    const auto one_number = contract_for("configure", Json{{"value", Json{{"type", "number"}}}});
    const std::string large_fraction = tool_call("configure", {{"value", "9007199254740992.5"}});
    const auto number_parsed = fi::parse_qwen_tool_call_output(large_fraction, 64, one_number);
    failures += check(number_parsed.is_tool_call_response && number_parsed.tool_calls.size() == 1 &&
                          number_parsed.tool_calls.front().arguments_json ==
                              "{\"value\":9007199254740992.5}",
                      "valid number was rejected or lost its original precision");
    return failures;
}

int test_composed_schema_types() {
    const auto contract = contract_for(
        "configure",
        Json{{"flag",
              Json{{"anyOf", Json::array({Json{{"type", "boolean"}}, Json{{"type", "null"}}})}}},
             {"unset",
              Json{{"oneOf", Json::array({Json{{"type", "null"}}, Json{{"type", "boolean"}}})}}},
             {"count",
              Json{{"anyOf", Json::array({Json{{"type", "integer"}}, Json{{"type", "null"}}})}}},
             {"nested",
              Json{{"anyOf", Json::array({Json{{"oneOf", Json::array({Json{{"type", "boolean"}},
                                                                      Json{{"type", "null"}}})}},
                                          Json{{"type", "integer"}}})}}},
             {"string_or_number",
              Json{{"oneOf", Json::array({Json{{"type", "string"}}, Json{{"type", "number"}}})}}}});
    const std::string text = tool_call("configure", {{"flag", "False"},
                                                     {"unset", "null"},
                                                     {"count", "7.0"},
                                                     {"nested", "TRUE"},
                                                     {"string_or_number", "7"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "explicit anyOf/oneOf primitive union was rejected");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("flag") == false && args.at("unset").is_null(),
                          "nullable boolean composition was decoded incorrectly");
        failures += check(args.at("count") == 7.0 && args.at("nested") == true,
                          "nested primitive composition was decoded incorrectly");
        failures += check(args.at("string_or_number") == "7",
                          "string-admitting composition did not preserve text");
    }
    failures += check_parameter_schema_mismatch(
        contract, "count", "7.5", "7.5",
        "fractional anyOf integer/null mismatch was not structured");
    return failures;
}

int test_empty_declared_non_string_is_omitted() {
    const Json properties = {
        {"file_path", Json{{"type", "string"}}},
        {"new_string", Json{{"type", "string"}}},
        {"old_string", Json{{"type", "string"}}},
        {"replace_all", Json{{"type", "boolean"}}},
    };
    const std::string text = "I need one more check.\n\n"
                             "<tool_call>\n"
                             "<function=Edit>\n"
                             "<parameter=file_path>\n/tmp/probe.cpp\n</parameter>\n"
                             "<parameter=new_string>\n"
                             "std::map<std::uint32_t, int> counts;\n"
                             "</parameter>\n"
                             "<parameter=old_string>\nold line\n</parameter>\n"
                             "<parameter=replace_all>\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>";

    const std::vector<std::string> definitions = {tool_definition(
        "Edit", properties, Json::array({"file_path", "new_string", "old_string"}))};
    const auto contract                        = contract_from_definitions(definitions);
    const auto parsed = fi::parse_qwen_tool_call_output(text, 128, *contract);

    int failures = 0;
    failures +=
        check(parsed.is_tool_call_response && parsed.content == "I need one more check." &&
                  parsed.tool_calls.size() == 1 && parsed.diagnostics.marker_seen &&
                  parsed.diagnostics.structured_call_count == 1 &&
                  parsed.diagnostics.empty_arguments_omitted == 1 &&
                  parsed.diagnostics.schema_mismatch_arguments == 0 &&
                  parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
              "empty optional boolean demoted a complete Edit call to text");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 3 && args.at("file_path") == "/tmp/probe.cpp" &&
                              args.at("new_string") == "std::map<std::uint32_t, int> counts;" &&
                              args.at("old_string") == "old line" && !args.contains("replace_all"),
                          "empty optional boolean was not omitted from Edit arguments");
    }

    bool every_split_matches = true;
    for (std::size_t split = 0; split <= text.size(); ++split) {
        fi::ToolCallOutputDecoder decoder(contract, 128);
        std::string visible = decoder.feed(std::string_view(text).substr(0, split));
        visible += decoder.feed(std::string_view(text).substr(split));
        auto terminal = decoder.finish(ninfer::FinishReason::None);
        if (visible != "I need one more check." || !terminal.content.empty() ||
            terminal.tool_calls.size() != 1 ||
            terminal.tool_calls.front().arguments_json !=
                parsed.tool_calls.front().arguments_json ||
            terminal.diagnostics != parsed.diagnostics) {
            every_split_matches = false;
            break;
        }
    }
    failures += check(every_split_matches,
                      "incremental Edit parsing depends on the transport chunk boundary");

    fi::ToolCallOutputDecoder bytewise(contract, 128);
    std::string bytewise_visible;
    for (const char byte : text) { bytewise_visible += bytewise.feed(std::string_view(&byte, 1)); }
    auto bytewise_terminal = bytewise.finish(ninfer::FinishReason::None);
    failures +=
        check(bytewise_visible == "I need one more check." && bytewise_terminal.content.empty() &&
                  bytewise_terminal.tool_calls.size() == 1 &&
                  bytewise_terminal.diagnostics == parsed.diagnostics,
              "bytewise Edit parsing changed the terminal tool-call semantics");

    const auto string_contract =
        contract_for("configure", Json{{"label", Json{{"type", "string"}}}});
    const auto empty_string = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"label", ""}}), 64, string_contract);
    failures +=
        check(empty_string.is_tool_call_response && empty_string.tool_calls.size() == 1 &&
                  Json::parse(empty_string.tool_calls.front().arguments_json).at("label") == "",
              "empty declared string was incorrectly omitted");
    return failures;
}

int test_schema_mismatches_remain_structured() {
    const auto contract =
        contract_for("configure", Json{{"integer_value", Json{{"type", "integer"}}},
                                       {"number_value", Json{{"type", "number"}}},
                                       {"boolean_value", Json{{"type", "boolean"}}},
                                       {"object_value", Json{{"type", "object"}}},
                                       {"array_value", Json{{"type", "array"}}},
                                       {"null_value", Json{{"type", "null"}}}});

    int failures = 0;
    failures += check_parameter_schema_mismatch(contract, "number_value", "\"1\"", "\"1\"",
                                                "string number mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "boolean_value", "[]", "[]",
                                                "array boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "object_value", "[]", "[]",
                                                "array object mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "array_value", "{}", "{}",
                                                "object array mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "null_value", "false", "false",
                                                "boolean null mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        contract, "object_value", "{'x': True}", "\"{'x': True}\"",
        "Python object mismatch was not preserved for client validation");
    failures += check_parameter_schema_mismatch(
        contract, "array_value", "['a', None]", "\"['a', None]\"",
        "Python array mismatch was not preserved for client validation");
    return failures;
}

int test_unsupported_schema_uses_legacy_policy() {
    const auto contract = contract_for(
        "configure",
        Json{{"missing_type", Json::object()},
             {"alias", Json{{"type", "int"}}},
             {"invalid_type_array", Json{{"type", Json::array({"integer", "int"})}}},
             {"partial_anyof", Json{{"anyOf", Json::array({Json{{"type", "integer"}},
                                                           Json{{"enum", Json::array({1, 2})}}})}}},
             {"mixed_composition", Json{{"anyOf", Json::array({Json{{"type", "boolean"}}})},
                                        {"oneOf", Json::array({Json{{"type", "null"}}})}}}});
    // R3-08: the first parameter position is outside the non-first undeclared rule, so the
    // legacy inference still applies there and is emitted with its schema_mismatch count.
    const std::string text = tool_call("configure", {{"undeclared", "{\"x\":1}"},
                                                     {"missing_type", "7"},
                                                     {"alias", "8"},
                                                     {"invalid_type_array", "9"},
                                                     {"partial_anyof", "7.5"},
                                                     {"mixed_composition", "True"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.diagnostics.schema_mismatch_arguments == 1,
                      "first-position undeclared parameter lost legacy inference");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("alias") == 8 &&
                          args.at("invalid_type_array") == 9,
                      "legacy numeric inference changed");
    failures += check(args.at("partial_anyof") == 7.5 && args.at("mixed_composition") == "True",
                      "unsupported composition was partially inferred");
    failures +=
        check(args.at("undeclared").at("x") == 1, "undeclared parameter legacy inference changed");

    // A non-first undeclared parameter outside the declared schema is ambiguous: the
    // region is returned verbatim (R3-08).
    const std::string non_first_undeclared = tool_call(
        "configure", {{"alias", "8"}, {"undeclared", "{\"x\":1}"}});
    const auto parsed_ambiguous =
        fi::parse_qwen_tool_call_output(non_first_undeclared, 64, contract);
    failures += check(!parsed_ambiguous.is_tool_call_response &&
                          parsed_ambiguous.content == non_first_undeclared &&
                          parsed_ambiguous.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::AmbiguousStructure,
                      "non-first undeclared parameter was not rejected as ambiguous");
    return failures;
}

int test_strict_structure_and_active_tool_set() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    const std::string malformed = "<tool_call>\n<function=configure>\n";
    failures +=
        check_rejected(malformed, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                       "missing structural tags were accepted");

    const std::string suffix = tool_call("configure", {{"value", "x"}}) + "\nextra answer";
    failures +=
        check_rejected(suffix, contract, ninfer::ToolCallParseFallbackReason::TrailingContent,
                       "non-whitespace suffix was accepted");

    const std::string missing_parameter_close =
        "<tool_call>\n<function=configure>\n<parameter=value>\nx\n"
        "</function>\n</tool_call>";
    failures += check_rejected(missing_parameter_close, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "missing parameter close was repaired");

    const std::string unknown_tool = tool_call("other", {{"value", "x"}});
    failures +=
        check_rejected(unknown_tool, contract, ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                       "undeclared tool name was accepted");

    const std::string invalid_name = tool_call("bad.name", {{"value", "x"}});
    failures += check_rejected(invalid_name, kLegacyContract,
                               ninfer::ToolCallParseFallbackReason::InvalidToolName,
                               "invalid function-name character was accepted");
    return failures;
}

int test_name_limits_and_non_strict_omissions() {
    const std::string name(128, 'a');
    const std::string text          = tool_call(name);
    const auto anthropic            = fi::parse_qwen_tool_call_output(text, 128, kLegacyContract);
    const auto openai               = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);
    const std::string too_long_text = tool_call(std::string(129, 'a'));
    const auto too_long = fi::parse_qwen_tool_call_output(too_long_text, 128, kLegacyContract);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1,
                      "128-character Anthropic tool name was rejected");
    failures += check(!openai.is_tool_call_response, "128-character OpenAI tool name was accepted");
    failures +=
        check(!too_long.is_tool_call_response, "129-character Anthropic tool name was accepted");

    const std::string definition = tool_definition(
        "optional", Json{{"value", Json{{"type", "string"}}}}, Json::array({"value"}));
    const std::vector<std::string> definitions = {definition};
    const auto contract                        = contract_from_definitions(definitions);
    const auto omitted = fi::parse_qwen_tool_call_output(tool_call("optional"), 64, *contract);
    failures += check(omitted.is_tool_call_response && omitted.tool_calls.size() == 1 &&
                          omitted.tool_calls.front().arguments_json == "{}",
                      "non-strict parser enforced required parameters");
    return failures;
}

int test_conflicting_duplicate_tool_contracts_use_legacy_normalization() {
    const std::string integer_definition =
        tool_definition("configure", Json{{"value", Json{{"type", "integer"}}}});
    const std::string string_definition =
        tool_definition("configure", Json{{"value", Json{{"type", "string"}}}});

    const std::vector<std::string> identical_definitions = {integer_definition, integer_definition};
    const auto identical = contract_from_definitions(identical_definitions);
    const auto accepted =
        fi::parse_qwen_tool_call_output(tool_call("configure", {{"value", "7"}}), 64, *identical);

    const std::vector<std::string> conflicting_definitions = {integer_definition,
                                                              string_definition};
    const auto conflicting      = contract_from_definitions(conflicting_definitions);
    const std::string ambiguous = tool_call("configure", {{"value", "7"}});
    const auto ambiguous_parsed = fi::parse_qwen_tool_call_output(ambiguous, 64, *conflicting);

    int failures = 0;
    failures += check(accepted.is_tool_call_response && accepted.tool_calls.size() == 1,
                      "identical duplicate tool contracts became ambiguous");
    failures +=
        check(ambiguous_parsed.is_tool_call_response && ambiguous_parsed.tool_calls.size() == 1 &&
                  ambiguous_parsed.tool_calls.front().arguments_json == "{\"value\":7}" &&
                  ambiguous_parsed.diagnostics.schema_mismatch_arguments == 0,
              "conflicting duplicate tool contracts did not use legacy normalization");
    return failures;
}

int test_all_or_nothing_structural_commit() {
    const auto contract    = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    const std::string text = tool_call("configure", {{"flag", "true"}}) +
                             "\n<tool_call>\n<function=configure>\n<parameter=flag>\nfalse\n";
    return check_rejected(text, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                          "partially valid tool-call region was partially committed");
}

int test_quoted_marker_before_real_call() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<function=command>\\nprintf broken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text = "explaining " + quoted + " then the real turn\n" +
                             tool_call("bash", {{"command", "echo ok"}});
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    // R3-02: a quoted broken wrapper no longer keeps the real turn out of reach. The region
    // breaks inside the quoted scope, but the real marker sits after the quoted wrapper's
    // own close — a proven top-level position — so the retry commits the real call and the
    // quoted example plus the prose remain ordinary content.
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.tool_calls.front().name == "bash" &&
                          parsed.content == "explaining " + quoted + " then the real turn" &&
                          parsed.diagnostics.marker_seen &&
                          parsed.diagnostics.structured_call_count == 1 &&
                          parsed.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::None,
                      "the quoted broken wrapper still hides the real call after its close");
    return failures;
}

int test_later_candidate_must_consume_the_end() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<parameter=command>\\nbroken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text =
        quoted + "\n" + tool_call("bash", {{"command", "echo ok"}}) + "\nstill explaining";
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty() &&
                          parsed.content == text && parsed.diagnostics.marker_seen &&
                          parsed.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::MalformedStructure,
                      "a quoted marker before a non-terminal call was partially committed");
    return failures;
}

int test_incremental_quoted_marker_preserves_bytes() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text = "explaining " + quoted + " then the real turn\n" +
                             tool_call("bash", {{"command", "echo ok"}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 5;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += decoder.feed(std::string_view(text).substr(offset, kChunk));
    }
    auto terminal = decoder.finish(ninfer::FinishReason::None);

    int failures = 0;
    failures += check(terminal.tool_calls.size() == 1 &&
                          terminal.tool_calls.front().name == "bash",
                      "the real call after a quoted broken wrapper was not committed (R3-02)");
    failures += check(visible + terminal.content == "explaining " + quoted + " then the real turn",
                      "incremental quoted marker lost or duplicated bytes");
    failures += check(terminal.diagnostics.marker_seen &&
                          terminal.diagnostics.structured_call_count == 1 &&
                          terminal.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::None,
                      "incremental quoted marker changed terminal diagnostics");
    return failures;
}

int test_incremental_valid_and_boolean() {
    fi::ToolCallOutputDecoder legacy(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string visible;
    visible += legacy.feed("Calling weather.  \n<tool_");
    visible += legacy.feed("call>\n<function=get_weather>");
    visible += legacy.feed("\n</function>\n</tool_call>");
    auto legacy_terminal = legacy.finish(ninfer::FinishReason::None);
    visible += legacy_terminal.content;

    auto bool_contract =
        output_contract_for("configure", Json{{"enabled", Json{{"type", "boolean"}}}});
    fi::ToolCallOutputDecoder boolean(std::move(bool_contract), 64);
    std::string boolean_visible;
    boolean_visible += boolean.feed("<tool_call>\n<function=configure>\n<parameter=enabled>\nT");
    boolean_visible += boolean.feed("r");
    boolean_visible += boolean.feed("ue\n</parameter>\n</function>\n</tool_call>");
    auto boolean_terminal = boolean.finish(ninfer::FinishReason::None);

    int failures = 0;
    failures += check(visible == "Calling weather." && legacy_terminal.tool_calls.size() == 1,
                      "incremental valid call was not committed");
    failures += check(boolean_visible.empty() && boolean_terminal.content.empty() &&
                          boolean_terminal.tool_calls.size() == 1,
                      "incremental boolean call leaked as content");
    if (boolean_terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(boolean_terminal.tool_calls.front().arguments_json);
        failures += check(args.at("enabled") == true,
                          "split case-insensitive boolean was not canonicalized");
    }
    return failures;
}

int test_incremental_fallback_preserves_bytes() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    fi::ToolCallOutputDecoder malformed(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    auto malformed_terminal = malformed.finish(ninfer::FinishReason::None);
    restored += malformed_terminal.content;

    fi::ToolCallOutputDecoder ordinary(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string ordinary_text;
    ordinary_text += ordinary.feed("ordinary text  ");
    ordinary_text += ordinary.finish(ninfer::FinishReason::None).content;

    const std::string partial_original = "  <tool_x then <tool_";
    fi::ToolCallOutputDecoder partial(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string partial_restored;
    partial_restored += partial.feed("  <too");
    partial_restored += partial.feed("l_x then <tool_");
    partial_restored += partial.finish(ninfer::FinishReason::None).content;

    int failures = 0;
    failures += check(restored == original && malformed_terminal.diagnostics.marker_seen &&
                          malformed_terminal.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::MalformedStructure,
                      "malformed incremental call lost raw bytes or fallback diagnostics");
    failures += check(ordinary_text == "ordinary text  ",
                      "ordinary incremental output lost trailing whitespace");
    failures +=
        check(partial_restored == partial_original, "partial marker mismatch lost raw bytes");
    return failures;
}

int test_incremental_embedded_parameter_markup() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string command = "pattern='<parameter=inner>value</parameter>'\n"
                                "printf '%s' \"$pattern\"";
    const std::string text    = tool_call("bash", {{"command", command}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 7;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += decoder.feed(std::string_view(text).substr(offset, kChunk));
    }
    auto terminal = decoder.finish(ninfer::FinishReason::None);

    int failures = 0;
    failures +=
        check(visible.empty() && terminal.content.empty() && terminal.tool_calls.size() == 1,
              "chunked embedded parameter markup was not committed as a tool call");
    if (terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(terminal.tool_calls.front().arguments_json);
        failures += check(args.at("command") == command,
                          "chunked embedded parameter markup changed string bytes");
    }
    return failures;
}

int test_claude_code_xml_markup_variants() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    int failures = 0;

    const std::string standard_xml =
        "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Initial setup\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_standard = fi::parse_qwen_tool_call_output(standard_xml, 128, contract);
    failures += check(parsed_standard.is_tool_call_response &&
                          parsed_standard.tool_calls.size() == 1 &&
                          parsed_standard.tool_calls.front().name == "TaskCreate",
                      "function name attribute syntax was not parsed");
    if (parsed_standard.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed_standard.tool_calls.front().arguments_json);
        failures += check(args.at("description") == "Initial setup",
                          "function name attribute argument changed");
    }

    const std::string invoke_xml =
        "<tool_call>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Create tasks\n</parameter>\n</invoke>\n</tool_call>";
    const auto parsed_invoke = fi::parse_qwen_tool_call_output(invoke_xml, 128, contract);
    failures += check(parsed_invoke.is_tool_call_response && parsed_invoke.tool_calls.size() == 1 &&
                          parsed_invoke.tool_calls.front().name == "TaskCreate",
                      "invoke tag syntax was not parsed");

    const std::string function_calls_xml =
        "<function_calls>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Function calls container\n</parameter>\n</invoke>\n</function_calls>";
    const auto parsed_function_calls =
        fi::parse_qwen_tool_call_output(function_calls_xml, 128, contract);
    failures += check(parsed_function_calls.is_tool_call_response &&
                          parsed_function_calls.tool_calls.size() == 1 &&
                          parsed_function_calls.tool_calls.front().name == "TaskCreate",
                      "function_calls container syntax was not parsed");

    const std::string standalone_invoke =
        "Plan is ready:\n<invoke name=\"TaskCreate\">\n<param name=\"description\">\n"
        "Standalone invoke\n</param>\n</invoke>";
    const auto parsed_standalone = fi::parse_qwen_tool_call_output(standalone_invoke, 128, contract);
    failures += check(parsed_standalone.is_tool_call_response &&
                          parsed_standalone.content == "Plan is ready:" &&
                          parsed_standalone.tool_calls.size() == 1 &&
                          parsed_standalone.tool_calls.front().name == "TaskCreate",
                      "standalone invoke after plan was not parsed");

    return failures;
}

int test_duplicate_parameters_keep_last_value() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    // R3-08: a declared parameter written twice makes the region ambiguous (the values
    // cannot be reconciled without guessing which occurrence is authoritative): the region
    // is returned verbatim as text with ambiguous_structure, in strict and tolerant alike.
    const std::string identical_dup =
        "<tool_call>\n<function=configure>\n<parameter=value>\nfirst\n</parameter>\n"
        "<parameter=value>\nfirst\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_identical = fi::parse_qwen_tool_call_output(identical_dup, 64, contract);
    failures += check(!parsed_identical.is_tool_call_response &&
                          parsed_identical.content == identical_dup &&
                          parsed_identical.tool_calls.empty() &&
                          parsed_identical.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::AmbiguousStructure,
                      "identical duplicate parameter was not rejected as ambiguous");

    // A conflicting repeat is ambiguous in the same way: last-wins is no longer a repair.
    const std::string conflicting_dup =
        "<tool_call>\n<function=configure>\n<parameter=value>\nfirst\n</parameter>\n"
        "<parameter=value>\nsecond\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_conflicting = fi::parse_qwen_tool_call_output(conflicting_dup, 64, contract);
    failures += check(!parsed_conflicting.is_tool_call_response &&
                          parsed_conflicting.content == conflicting_dup &&
                          parsed_conflicting.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::AmbiguousStructure,
                      "conflicting duplicate parameter was not rejected as ambiguous");

    // The legacy JSON-object rule (last wins, with the repair count) now applies only to
    // tools the contract does not declare; the public entry enforces the same
    // declared-name predicate in the state machine (CR5) that R3-08's lookup uses, so
    // that branch is defensive (spec R3-08) and pinned by the stream-level tests below
    // only when a contract can be built without declared-name enforcement.
    return failures;
}

int test_attribute_token_boundary() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    int failures = 0;

    const std::string text =
        "<tool_call>\n<function filename=\"x\" name=\"TaskCreate\">\n"
        "<parameter filename=\"ignored\" name=\"description\">\nCreate task\n</parameter>\n"
        "</function>\n</tool_call>";
    const auto parsed = fi::parse_qwen_tool_call_output(text, 128, contract);
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.tool_calls.front().name == "TaskCreate",
                      "attribute token boundary failed to extract correct name");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("description") == "Create task",
                          "parameter attribute token boundary failed");
    }
    return failures;
}

int test_mismatched_closing_tags_rejected() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    int failures = 0;

    const std::string fn_invoke_mismatch =
        "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</parameter>\n</invoke>\n</tool_call>";
    failures += check_rejected(fn_invoke_mismatch, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "function opening with invoke closing tag was accepted");

    const std::string invoke_fn_mismatch =
        "<tool_call>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</parameter>\n</function>\n</tool_call>";
    failures += check_rejected(invoke_fn_mismatch, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "invoke opening with function closing tag was accepted");

    const std::string param_mismatch =
        "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</param>\n</function>\n</tool_call>";
    failures += check_rejected(param_mismatch, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "parameter opening with param closing tag was accepted");

    const std::string param_open_mismatch =
        "<tool_call>\n<function name=\"TaskCreate\">\n<param name=\"description\">\n"
        "Value\n</parameter>\n</function>\n</tool_call>";
    failures += check_rejected(param_open_mismatch, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "param opening with parameter closing tag was accepted");

    return failures;
}

int test_claude_code_plan_and_task_create_exact_repro() {
    const std::string task_create_def = tool_definition(
        "TaskCreate",
        Json{{"description", Json{{"type", "string"}}},
             {"task_type", Json{{"type", "string"}}},
             {"priority", Json{{"type", "integer"}}}});
    const std::string task_update_def = tool_definition(
        "TaskUpdate",
        Json{{"taskId", Json{{"type", "string"}}}, {"status", Json{{"type", "string"}}}});
    const auto contract = contract_from_definitions({task_create_def, task_update_def});

    const std::string full_response =
        "I have analyzed the repository requirements. Here is the implementation plan:\n\n"
        "### Plan\n"
        "1. Inspect existing CUDA kernels in `src/ops/softmax_attention/`\n"
        "2. Add test coverage for long context attention splits\n"
        "3. Update frontend tool call decoder\n\n"
        "Let me create the first task in the tracking system now:\n\n"
        "<tool_call>\n"
        "<function name=\"TaskCreate\">\n"
        "<parameter name=\"description\">\n"
        "Implement split-KV page-safety and bounded loops\n"
        "</parameter>\n"
        "<parameter name=\"task_type\">\n"
        "feature\n"
        "</parameter>\n"
        "<parameter name=\"priority\">\n"
        "1\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";

    const auto parsed = fi::parse_qwen_tool_call_output(full_response, 128, *contract);
    int failures      = 0;
    failures += check(parsed.is_tool_call_response,
                      "Claude Code Plan + TaskCreate failed to parse as tool call");
    failures += check(parsed.tool_calls.size() == 1, "tool call count != 1");
    failures += check(parsed.content.starts_with("I have analyzed"), "plan content prefix lost");
    failures += check(parsed.content.ends_with("tracking system now:"), "plan content tail lost");

    if (parsed.tool_calls.size() == 1) {
        const auto& call = parsed.tool_calls.front();
        failures += check(call.name == "TaskCreate", "tool name != TaskCreate");
        const Json args = Json::parse(call.arguments_json);
        failures += check(args.at("description") == "Implement split-KV page-safety and bounded loops",
                          "TaskCreate description argument changed");
        failures += check(args.at("task_type") == "feature", "TaskCreate task_type argument changed");
        failures += check(args.at("priority") == 1, "TaskCreate priority argument changed");
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Regression tests for the bugfix specification
// (NInfer_new_parser_design_bugfix_implementation.md, findings F1-F8). These pin the safety
// invariants: opaque parameter payload (I1), non-execution of open functions (I2), boundary
// retention across partial next tokens (I3), explicit wrapper balance (I4), shared marker
// progression (I5), and non-permissive recovery (I6).
// ---------------------------------------------------------------------------

// Exact round trip of `payload` as the declared string value of `parameter` in one complete
// well-formed call: strict and tolerant one-shot plus fixed-size chunked streaming must all
// produce the identical structured result with the value bytes unchanged.
int check_payload_round_trip(const fi::ToolCallOutputContract& contract, std::string_view tool_name,
                             std::string_view parameter, std::string_view payload,
                             std::string_view what) {
    int failures = 0;
    const std::string text = tool_call(tool_name, {{parameter, payload}});
    for (const bool tolerant : {false, true}) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.content.empty(),
                          std::string("payload round trip lost the call: ") + std::string(what));
        if (parsed.tool_calls.size() == 1) {
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.at(std::string(parameter)).get<std::string>() == std::string(payload),
                              std::string("payload round trip changed the value: ") +
                                  std::string(what));
        }
    }
    auto contract_ptr = std::make_shared<fi::ToolCallOutputContract>(contract);
    for (const std::size_t chunk : {1, 2, 3, 5, 7}) {
        fi::ToolCallOutputDecoder decoder(contract_ptr, 64);
        std::string visible;
        for (std::size_t offset = 0; offset < text.size(); offset += chunk) {
            visible += decoder.feed(std::string_view(text).substr(offset, chunk));
        }
        auto terminal = decoder.finish(ninfer::FinishReason::None);
        failures += check(visible.empty() && terminal.content.empty() &&
                              terminal.tool_calls.size() == 1 &&
                              Json::parse(terminal.tool_calls.front().arguments_json)
                                      .at(std::string(parameter)).get<std::string>() == std::string(payload),
                          std::string("chunked payload round trip diverged: ") + std::string(what));
    }
    return failures;
}

// F1: a literal parameter opener inside a value must not change nesting; the value is an
// opaque byte range.
int test_literal_parameter_opener_in_payload_does_not_nest() {
    const auto contract = contract_for("write", Json{{"content", Json{{"type", "string"}}}});
    int failures = 0;
    failures += check_payload_round_trip(
        contract, "write", "content", "const x = \"<parameter=fake>\";",
        "C literal <parameter=...> in payload");
    failures += check_payload_round_trip(contract, "write", "content", "A <parameter=fake> B",
                                         "literal <parameter=...> in payload");
    return failures;
}

int test_literal_param_opener_in_payload_does_not_nest() {
    const auto contract = contract_for("write", Json{{"content", Json{{"type", "string"}}}});
    return check_payload_round_trip(contract, "write", "content", "A <param=fake> B",
                                    "literal <param=...> in payload");
}

// F1: real coding payloads carrying tool-like markup must round-trip byte for byte.
int test_coding_payloads_round_trip() {
    const auto contract = contract_for("write", Json{{"content", Json{{"type", "string"}}}});
    int failures = 0;
    const std::vector<std::pair<const char*, std::string>> payloads = {{
        {"C/C++ source string",
         "const char* s = \"<parameter=edits>\";\nprintf(\"%s</parameter>\", s);\n"},
        {"TypeScript template",
         "const tpl = `<parameter=inner>${x}</parameter>`;\nconst close = '</parameter>';"},
        {"XML/HTML snippet", "<root>\n  <parameter name=\"x\">v</parameter>\n  <tool_call/>\n</root>"},
        {"shell here-doc", "cat <<'EOF'\n<parameter=edits>\nline\n</parameter>\nEOF\necho done"},
        {"JSON string with markup", "{\"pattern\": \"<parameter=x>\", \"close\": \"</parameter>\"}"},
        {"multi-line patch",
         "--- a/f.c\n+++ b/f.c\n@@ -1,3 +1,3 @@\n-<parameter=old>\n+<parameter=new>\n</parameter>\n context line"},
    }};
    for (const auto& [label, payload] : payloads) {
        failures += check_payload_round_trip(contract, "write", "content", payload, label);
    }
    return failures;
}
// 18.4: the spec's deterministic adversarial payload corpus embedded into a declared string
// argument. Ordinary prefix/suffix text keeps every entry an unambiguous payload (I1): the
// exact value round-trips in both modes and in streaming. R3-01 (section 7.2): entries that
// end with the outer closer literal or a closer triple round-trip too — Stage 2 resolves
// the value boundary, so the old exclusion no longer applies.
int test_payload_adversarial_corpus_round_trip() {
    const auto contract = output_contract_for("write", Json{{"content", Json{{"type", "string"}}}});
    int failures = 0;
    const std::vector<std::string> corpus = {{"<"}, {">"}, {"</parameter>"}, {"<parameter=x>"},
                                             {"<param=x>"},   {"</param>"},     {"</function>"},
                                             {"</invoke>"},  {"<tool_call>"},   {"</tool_call>"},
                                             {"<function_calls>"}, {"</function_calls>"},
                                             {"<function=fake>"},  {"<invoke=fake>"},
                                             {"<function name=\"fake\">"},
                                             {"<parameter name=\"x\">"}};
    for (const std::string& entry : corpus) {
        for (const char* suffix : {"x", "x\n"}) {
            failures += check_payload_round_trip(
                *contract, "write", "content", std::string_view{"x " + entry + suffix},
                ("corpus entry as payload: " + entry).c_str());
        }
        // The entry leads the value (no prefix): the continuation after a candidate closer is
        // still ordinary text, so the closer stays payload and the value round-trips.
        failures += check_payload_round_trip(
            *contract, "write", "content", std::string_view{entry + " x"},
            ("entry at payload start: " + entry).c_str());
        // CRLF/tab around a repeated sequence keeps the boundary decision deterministic.
        failures += check_payload_round_trip(
            *contract, "write", "content",
            std::string_view{"\r\n" + entry + "\t" + entry + " x"},
            ("CRLF/tab repeated entry: " + entry).c_str());
    }
    // R3-01 section 7.2: the excluded class — payloads ending in the outer closer literal
    // or a closer triple — now round-trips: the consistent parse ends the value at the
    // candidate whose continuation is legal.
    for (const std::string closer :
         {"</parameter>", "</param>", "</parameter>\n</parameter>\n</parameter>"}) {
        for (const char* prefix : {"x ", "x\n"}) {
            failures += check_payload_round_trip(
                *contract, "write", "content",
                std::string_view{std::string(prefix) + closer},
                ("payload ending in closer literal: " + closer).c_str());
        }
    }
    return failures;
}


// F1: an inner opener without an inner closer; the final closer is the outer close.
int test_parameter_value_with_inner_opener_only_closes_at_outer() {
    const auto contract = contract_for("write", Json{{"content", Json{{"type", "string"}}}});
    return check_payload_round_trip(contract, "write", "content",
                                    "<parameter=inner>\nliteral without inner closer",
                                    "inner opener without inner closer");
}

// F6: mixed parameter families inside a value stay opaque (both directions).
int test_mixed_parameter_families_inside_payload_are_opaque() {
    const auto contract = contract_for("write", Json{{"content", Json{{"type", "string"}}}});
    int failures = 0;
    failures += check_payload_round_trip(contract, "write", "content", "A <param=x>B</param> C",
                                         "param family inside a parameter value");
    const std::string text = "<tool_call>\n<function=write>\n<param=content>\n"
                             "A <parameter=x>B</parameter> C\n</param>\n</function>\n</tool_call>";
    for (const bool tolerant : {false, true}) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "parameter family inside a param value broke the call");
        if (parsed.tool_calls.size() == 1) {
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.at("content") == "A <parameter=x>B</parameter> C",
                              "parameter family inside a param value changed the bytes");
        }
    }
    return failures;
}

// F2: a function whose closing tag was never observed is never executed, whatever the finish
// reason (StopToken / StopString / OutputLimit / ContextCapacity / Cancelled) and whatever the wrapper form.
int test_tolerant_never_commits_open_function() {
    using FinishReason = ninfer::FinishReason;
    using Reason       = ninfer::ToolCallParseFallbackReason;
    const auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    int failures        = 0;
    const std::string cut_tool_call =
        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>";
    const std::string cut_invoke =
        "<tool_call>\n<invoke=bash>\n<parameter=command>\nls\n</parameter>";
    const std::string cut_bare   = "<function=bash>\n<parameter=command>\nls\n</parameter>";
    for (const std::string& region : {cut_tool_call, cut_invoke, cut_bare}) {
        for (const FinishReason reason : {ninfer::FinishReason::StopToken, ninfer::FinishReason::StopString,
                                          ninfer::FinishReason::OutputLimit,
                                          ninfer::FinishReason::ContextCapacity,
                                          ninfer::FinishReason::Cancelled}) {
            const auto tolerant =
                fi::parse_qwen_tool_call_output(region, 64, *contract, true, reason);
            failures += check(!tolerant.is_tool_call_response && tolerant.tool_calls.empty(),
                              "tolerant executed a function whose close was never observed");
            failures += check(tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
                              "open-function truncation lost the truncated-tail diagnostic");
            const auto strict = fi::parse_qwen_tool_call_output(region, 64, *contract, false, reason);
            failures += check(!strict.is_tool_call_response && strict.tool_calls.empty(),
                              "strict executed a function whose close was never observed");
            failures += check(strict.diagnostics.fallback_reason == Reason::MalformedStructure,
                              "strict open-function truncation lost the structural diagnostic");
        }
    }
    return failures;
}

// F2: a function-closed call whose wrapper close is missing remains recoverable in tolerant
// mode (TruncatedTail); strict still rejects. All three wrapper forms.
int test_tolerant_commits_function_closed_missing_wrapper() {
    using Reason = ninfer::ToolCallParseFallbackReason;
    const auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    int failures = 0;
    const std::string tool_call_cut =
        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>";
    const std::string function_calls_cut =
        "<function_calls>\n<invoke=bash>\n<parameter=command>\nls\n</parameter>\n</invoke>";
    const std::string bare_complete =
        "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>";
    // The tool_call wrapper: a missing close at the region end is a truncated tail — tolerant
    // retains the function-closed call with a diagnostic, strict rejects.
    const auto tool_tolerant =
        fi::parse_qwen_tool_call_output(tool_call_cut, 64, *contract, true);
    failures += check(
        tool_tolerant.is_tool_call_response && tool_tolerant.tool_calls.size() == 1 &&
            tool_tolerant.tool_calls.front().name == "bash" &&
            Json::parse(tool_tolerant.tool_calls.front().arguments_json).at("command") == "ls",
        "tolerant did not retain the function-closed call with a missing tool_call wrapper");
    failures += check(tool_tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "missing tool_call wrapper close was not flagged");
    const auto tool_strict = fi::parse_qwen_tool_call_output(tool_call_cut, 64, *contract);
    failures += check(!tool_strict.is_tool_call_response && tool_strict.tool_calls.empty(),
                      "strict retained a call with a missing tool_call wrapper close");
    // R2-I2/CR2: a wrapper open at the input end is an objective truncation, never a clean
    // completion. Strict rejects (MalformedStructure); tolerant retains the closed call
    // with the truncation diagnostic.
    {
        const auto fc_strict = fi::parse_qwen_tool_call_output(function_calls_cut, 64, *contract);
        failures += check(
            !fc_strict.is_tool_call_response && fc_strict.tool_calls.empty() &&
                fc_strict.diagnostics.fallback_reason == Reason::MalformedStructure,
            "a missing function_calls close completed cleanly in strict mode");
        const auto fc_tolerant =
            fi::parse_qwen_tool_call_output(function_calls_cut, 64, *contract, true);
        failures += check(
            fc_tolerant.is_tool_call_response && fc_tolerant.tool_calls.size() == 1 &&
                fc_tolerant.tool_calls.front().name == "bash" &&
                Json::parse(fc_tolerant.tool_calls.front().arguments_json).at("command") ==
                    "ls" &&
                fc_tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
            "a missing function_calls close dropped the tolerant call sequence");
    }
    // A bare function close at the region end is a clean end: committed without a diagnostic.
    const auto bare = fi::parse_qwen_tool_call_output(bare_complete, 64, *contract, true);
    failures += check(bare.is_tool_call_response && bare.tool_calls.size() == 1 &&
                          bare.diagnostics.fallback_reason == Reason::None,
                      "bare function close at region end was not a clean completion");
    return failures;
}

// F3: an EOF </parameter> that may be literal payload never executes the open function; the
// region falls back to verbatim content.
int test_eof_parameter_closer_never_executes_open_function() {
    using FinishReason = ninfer::FinishReason;
    int failures       = 0;
    const std::vector<std::pair<const char*, const char*>> fixtures = {{
        {"bash.command", "<tool_call>\n<function=bash>\n<parameter=command>\necho 'literal </parameter>"},
        {"write.content", "<tool_call>\n<function=write>\n<parameter=content>\necho 'literal </parameter>"},
        {"edit.new_string", "<tool_call>\n<function=edit>\n<parameter=new_string>\necho 'literal </parameter>"},
    }};
    for (const auto& [label, text] : fixtures) {
        for (const FinishReason reason : {ninfer::FinishReason::StopToken, ninfer::FinishReason::StopString,
                                          ninfer::FinishReason::OutputLimit,
                                          ninfer::FinishReason::ContextCapacity,
                                          ninfer::FinishReason::Cancelled}) {
            const auto strict =
                fi::parse_qwen_tool_call_output(text, 64, kLegacyContract, false, reason);
            failures += check(
                !strict.is_tool_call_response && strict.tool_calls.empty() && strict.content == text,
                std::string("strict executed the EOF parameter-closer call: ") + std::string(label));
            const auto tolerant = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract, true, reason);
            failures += check(
                !tolerant.is_tool_call_response && tolerant.tool_calls.empty() &&
                    tolerant.content == text,
                std::string("tolerant executed the EOF parameter-closer call: ") + std::string(label));
        }
    }
    return failures;
}

// F4: a partial next structural token after a complete parameter close must not reopen the
// previous value: the close is retained and the truncated token is an objective EndOfInput.
int test_partial_next_parameter_does_not_reopen_previous_value() {
    int failures = 0;
    const std::vector<std::pair<const char*, std::string>> fixtures = {{
        {"partial parameter opener",
         "<tool_call>\n<function=read>\n<parameter=path>\nfoo\n</parameter>\n<paramet"},
        {"partial function close",
         "<tool_call>\n<function=read>\n<parameter=path>\nfoo\n</parameter>\n</funct"},
        {"partial wrapper close",
         "<tool_call>\n<function=read>\n<parameter=path>\nfoo\n</parameter>\n</function>\n</tool_"},
    }};
    for (const auto& [label, text] : fixtures) {
        fi::ToolCallParsePolicy policy;
        const auto progress = fi::parse_tool_call_region(text, policy);
        failures += check(
            progress.termination == fi::ToolCallRegionTermination::EndOfInput,
            std::string("partial next token was not an objective truncation: ") +
                std::string(label));
        failures += check(
            !progress.open_value_open,
            std::string("partial next token reopened the previous value: ") + std::string(label));
        // The "partial wrapper close" fixture consumed the function close: the call is
        // committed there (F2) and no longer open. The other two fixtures leave the
        // function open with its closed parameter retained.
        if (label == "partial wrapper close") {
            failures += check(
                progress.calls.size() == 1 && progress.calls.front().name == "read" &&
                    progress.calls.front().parameters.size() == 1 &&
                    progress.calls.front().parameters.front().name == "path" &&
                    progress.calls.front().parameters.front().value == "\nfoo\n",
                std::string("committed call lost its closed parameter: ") + std::string(label));
        } else {
            failures += check(
                progress.open_call.name == "read" && progress.open_call.parameters.size() == 1 &&
                    progress.open_call.parameters.front().name == "path" &&
                    progress.open_call.parameters.front().value == "\nfoo\n",
                std::string("closed previous parameter was not retained: ") + std::string(label));
        }
    }
    return failures;
}

// F2/F4/I3/I5: the canonical call cut at every byte. Safety: no cut before the complete
// matching function close executes the current function in tolerant mode. Recovery: a cut
// after the function close may retain the complete call. One-shot and streaming (every chunk
// size) agree at every cut.
int test_all_delimiter_byte_cuts_preserve_previous_boundary() {
    using FinishReason = ninfer::FinishReason;
    using Reason       = ninfer::ToolCallParseFallbackReason;
    const auto contract =
        output_contract_for("write", Json{{"path", Json{{"type", "string"}}},
                                          {"content", Json{{"type", "string"}}}});
    const std::string text = "<tool_call>\n<function=write>\n"
                             "<parameter=path>\na.txt\n</parameter>\n"
                             "<parameter=content>\nhello\n</parameter>\n"
                             "</function>\n</tool_call>";
    const std::size_t fn_close_end   = text.find("</function>") + 11;
    const std::size_t first_close_end = text.find("</parameter>") + 12;
    const std::string expected_args  = R"({"path":"a.txt","content":"hello"})";
    int failures                     = 0;
    for (std::size_t cut = 0; cut <= text.size(); ++cut) {
        const std::string cut_text(text.substr(0, cut));
        // Strict: only the complete region commits.
        const auto strict = fi::parse_qwen_tool_call_output(cut_text, 64, *contract);
        if (cut < text.size()) {
            failures += check(!strict.is_tool_call_response && strict.tool_calls.empty(),
                              "strict executed a truncated canonical call");
        } else {
            failures += check(
                strict.is_tool_call_response && strict.tool_calls.size() == 1 &&
                    strict.tool_calls.front().arguments_json == expected_args,
                "strict did not commit the complete canonical call");
        }
        // Tolerant, every finish reason.
        for (const FinishReason reason : {ninfer::FinishReason::StopToken, ninfer::FinishReason::StopString,
                                          ninfer::FinishReason::OutputLimit,
                                          ninfer::FinishReason::ContextCapacity,
                                          ninfer::FinishReason::Cancelled}) {
            const auto tolerant =
                fi::parse_qwen_tool_call_output(cut_text, 64, *contract, true, reason);
            if (cut < fn_close_end) {
                failures += check(!tolerant.is_tool_call_response && tolerant.tool_calls.empty(),
                                  "tolerant executed the current function before its complete close");
            } else if (cut < text.size()) {
                failures += check(
                    tolerant.is_tool_call_response && tolerant.tool_calls.size() == 1 &&
                        tolerant.tool_calls.front().name == "write" &&
                        tolerant.tool_calls.front().arguments_json == expected_args &&
                        tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
                    "tolerant did not retain the function-closed canonical call");
            } else {
                failures += check(
                    tolerant.is_tool_call_response && tolerant.tool_calls.size() == 1 &&
                        tolerant.tool_calls.front().arguments_json == expected_args &&
                        tolerant.diagnostics.fallback_reason == Reason::None,
                    "tolerant lost the complete canonical call");
            }
        }
        // The closed first value must never be reopened or truncated, at any cut after it
        // closed: wherever it is recorded (open call or committed call) it keeps its bytes.
        fi::ToolCallParsePolicy policy;
        const auto progress = fi::parse_tool_call_region(cut_text, policy);
        if (cut > first_close_end) {
            const fi::ParsedFunctionCall* call = nullptr;
            if (!progress.open_call.name.empty() && progress.open_call.name == "write") {
                call = &progress.open_call;
            } else if (!progress.calls.empty()) {
                call = &progress.calls.front();
            }
            if (call != nullptr) {
                failures += check(
                    call->parameters.size() >= 1 && call->parameters.front().name == "path" &&
                        call->parameters.front().value == "\na.txt\n",
                    "the closed previous value was reopened or truncated at a later cut");
            }
        }
        // Streaming (every chunk size) must reproduce the one-shot tolerant result exactly.
        auto contract_ptr = std::make_shared<fi::ToolCallOutputContract>(*contract);
        for (const std::size_t chunk : {1, 2, 3, 5, 7}) {
            fi::ToolCallOutputDecoder decoder(contract_ptr, 64, true);
            std::string streamed;
            for (std::size_t offset = 0; offset < cut_text.size(); offset += chunk) {
                streamed += decoder.feed(std::string_view(cut_text).substr(offset, chunk));
            }
            auto terminal = decoder.finish(ninfer::FinishReason::StopToken);
            const auto one_shot = fi::parse_qwen_tool_call_output(
                cut_text, 64, *contract, true, ninfer::FinishReason::StopToken);
            failures += check(
                terminal.tool_calls.size() == one_shot.tool_calls.size() &&
                    terminal.content == one_shot.content &&
                    terminal.diagnostics == one_shot.diagnostics &&
                    std::equal(
                        one_shot.tool_calls.begin(), one_shot.tool_calls.end(),
                        terminal.tool_calls.begin(),
                        [](const auto& a, const auto& b) {
                            return a.name == b.name && a.arguments_json == b.arguments_json;
                        }),
                "streaming byte-split diverged from one-shot at a cut");
        }
    }
    return failures;
}

// F4: the same every-byte property on the other wire families (param/invoke/function_calls).
int test_all_delimiter_byte_cuts_other_families() {
    using FinishReason = ninfer::FinishReason;
    using Reason       = ninfer::ToolCallParseFallbackReason;
    const auto contract = output_contract_for("read", Json{{"path", Json{{"type", "string"}}}});
    const std::string text = "<function_calls>\n<invoke=read>\n"
                             "<param=path>\nfoo\n</param>\n"
                             "</invoke>\n</function_calls>";
    const std::size_t fn_close_end  = text.find("</invoke>") + 9;
    const std::size_t wrapper_start = text.find("</function_calls>");
    int failures = 0;
    for (std::size_t cut = 0; cut <= text.size(); ++cut) {
        const std::string cut_text(text.substr(0, cut));
        const auto tolerant = fi::parse_qwen_tool_call_output(
            cut_text, 64, *contract, true, ninfer::FinishReason::OutputLimit);
        if (cut < fn_close_end) {
            failures += check(!tolerant.is_tool_call_response && tolerant.tool_calls.empty(),
                              "tolerant executed an invoke before its complete close");
        } else if (cut <= wrapper_start) {
            // R2-I2/CR2: the region ends after the last invoke close but the wrapper close
            // is still missing: an objective truncation — tolerant retains the closed call
            // with the truncation diagnostic, never a clean completion.
            failures += check(
                tolerant.is_tool_call_response && tolerant.tool_calls.size() == 1 &&
                    tolerant.tool_calls.front().name == "read" &&
                    tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
                ("other families: tolerant cut at " + std::to_string(cut)).c_str());
        } else {
            failures += check(tolerant.is_tool_call_response && tolerant.tool_calls.size() == 1,
                              "the complete function_calls region was not committed");
        }
    }
    return failures;
}

// F5: wrapper nesting and balance are explicit; more opens than closes never completes, in
// strict or tolerant. Sequential balanced wrappers still complete.
int test_nested_and_cross_nested_wrappers_rejected() {
    using Reason = ninfer::ToolCallParseFallbackReason;
    const auto contract = output_contract_for("read", Json{{"path", Json{{"type", "string"}}}});
    int failures = 0;
    const std::string nested_tool_call = "<tool_call>\n<tool_call>\n"
                                         "<function=read>\n<parameter=path>x</parameter>\n"
                                         "</function>\n</tool_call>";
    const std::string nested_function_calls = "<function_calls>\n<function_calls>\n"
                                              "<function=read>\n<parameter=path>x</parameter>\n"
                                              "</function>\n</function_calls>";
    const std::string cross_tool_to_calls = "<tool_call>\n<function_calls>\n"
                                            "<function=read>\n<parameter=path>x</parameter>\n"
                                            "</function>\n</function_calls>";
    const std::string cross_calls_to_tool = "<function_calls>\n<tool_call>\n"
                                            "<function=read>\n<parameter=path>x</parameter>\n"
                                            "</function>\n</tool_call>";
    for (const std::string& text : {nested_tool_call, nested_function_calls,
                                    cross_tool_to_calls, cross_calls_to_tool}) {
        const auto strict = fi::parse_qwen_tool_call_output(text, 64, *contract);
        failures += check(
            !strict.is_tool_call_response && strict.tool_calls.empty() &&
                strict.diagnostics.fallback_reason == Reason::MalformedStructure,
            "unsupported wrapper nesting completed strict");
        const auto tolerant = fi::parse_qwen_tool_call_output(text, 64, *contract, true);
        failures += check(!tolerant.is_tool_call_response && tolerant.tool_calls.empty(),
                          "unsupported wrapper nesting was committed tolerant");
    }
    // Stray wrapper closer after a complete region: unbalanced, never silently absorbed.
    const std::string stray = tool_call("read", {{"path", "x"}}) + "\n</tool_call>";
    const auto stray_strict = fi::parse_qwen_tool_call_output(stray, 64, *contract);
    failures += check(
        !stray_strict.is_tool_call_response &&
            stray_strict.diagnostics.fallback_reason == Reason::TrailingContent,
        "stray wrapper closer was balanced against nothing (strict)");
    const auto stray_tolerant = fi::parse_qwen_tool_call_output(stray, 64, *contract, true);
    failures += check(
        stray_tolerant.is_tool_call_response && stray_tolerant.tool_calls.size() == 1 &&
            stray_tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
        "stray wrapper closer dropped the complete call (tolerant)");
    // Too many wrapper closers.
    const std::string many = tool_call("read", {{"path", "x"}}) + "\n</tool_call>\n</tool_call>";
    failures += check(
        !fi::parse_qwen_tool_call_output(many, 64, *contract).is_tool_call_response,
        "too many wrapper closers were accepted");
    // Missing outer close after a complete inner structure: F2 semantics (see the F2 tests).
    const std::string missing_close =
        "<tool_call>\n<function=read>\n<parameter=path>x</parameter>\n</function>";
    failures += check(
        !fi::parse_qwen_tool_call_output(missing_close, 64, *contract).is_tool_call_response,
        "missing outer close completed strict");
    // Multiple sequential valid wrappers at top level still complete.
    const std::string two =
        tool_call("read", {{"path", "x"}}) + "\n" + tool_call("read", {{"path", "y"}});
    const auto two_parsed = fi::parse_qwen_tool_call_output(two, 64, *contract);
    failures += check(two_parsed.is_tool_call_response && two_parsed.tool_calls.size() == 2,
                      "sequential balanced wrappers did not complete");
    return failures;
}

// F7: the recovery retry enters only at a later top-level wrapper. A function nested inside a
// failed region never becomes a new executable call, and a compatibility bare marker after a
// failed region is not re-read (it stays content).
int test_recovery_retry_entry_policy() {
    const auto contract = output_contract_for(
        "bash", Json{{"command", Json{{"type", "string"}}}, {"x", Json{{"type", "string"}}}});
    int failures = 0;
    // A malformed wrapper containing a nested function: never recover the nested call.
    const std::string nested_fake = "<tool_call>\n<function=broken\n"
                                    "<function=fake>\n<parameter=x>\n1\n</parameter>\n</function>";
    const auto nested = fi::parse_qwen_tool_call_output(nested_fake, 64, *contract, true);
    failures += check(!nested.is_tool_call_response && nested.tool_calls.empty(),
                      "a function nested inside a failed region became executable");
    // A compatibility bare function before a real wrapper: the turn recovers at the wrapper.
    const std::string bare_before = "<function=read>\n<parameter=path>\ncut\n" +
                                    tool_call("bash", {{"command", "echo ok"}});
    const auto bare = fi::parse_qwen_tool_call_output(bare_before, 64, *contract, true);
    failures += check(bare.is_tool_call_response && bare.tool_calls.size() == 1 &&
                          bare.tool_calls.front().name == "bash",
                      "a failed bare region before a real wrapper was not recovered at the wrapper");
    failures += check(bare.content == "<function=read>\n<parameter=path>\ncut",
                      "the failed bare region was not retained as content");
    // A failed region followed by a supported compatibility (bare) marker: the bare marker is
    // not a recovery entry; the region stays content.
    const std::string compat_after = "<tool_call>\n<function=broken\n"
                                     "<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>";
    const auto compat = fi::parse_qwen_tool_call_output(compat_after, 64, *contract, true);
    failures += check(!compat.is_tool_call_response && compat.tool_calls.empty() &&
                          compat.content == compat_after,
                      "a compatibility marker after a failed region was re-read by the retry");
    // Multiple false markers before the terminal real call.
    const std::string false_markers = "<function=false1>\n<function=false2>\n" +
                                      tool_call("bash", {{"command", "echo real"}});
    const auto false_parsed = fi::parse_qwen_tool_call_output(false_markers, 64, *contract, true);
    failures += check(false_parsed.is_tool_call_response && false_parsed.tool_calls.size() == 1 &&
                          false_parsed.tool_calls.front().name == "bash",
                      "multiple false markers before the real call were not recovered");
    return failures;
}

// F8: a marker candidate broken by a second '<' publishes the failed candidate bytes and
// retains the breaking '<' as a fresh candidate: the inner <tool_call> latches.
int test_marker_breaking_angle_restarts_candidate() {
    auto contract = output_contract_for("read", Json{{"path", Json{{"type", "string"}}}});
    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible = decoder.feed("prefix <function<");
    visible += decoder.feed(
        "tool_call>\n<function=read>\n<parameter=path>x</parameter>\n</function>\n</tool_call>");
    auto terminal = decoder.finish(ninfer::FinishReason::None);
    int failures = 0;
    failures += check(visible == "prefix <function",
                      "the failed marker candidate was not published without the breaking '<'");
    failures += check(terminal.tool_calls.size() == 1 && terminal.tool_calls.front().name == "read",
                      "the marker after a breaking '<' was not latched");
    if (terminal.tool_calls.size() == 1) {
        failures += check(terminal.tool_calls.front().arguments_json == R"({"path":"x"})",
                          "the latched region after a breaking '<' lost its arguments");
    }
    return failures;
}

// ---------------------------------------------------------------------------
// Round-2 regression tests (NInfer_new_parser_design_round2_bugfix_implementation.md,
// findings CR1-CR6). These pin the Round-2 invariants: a failed wrapper owns its still
// unclosed scope (R2-I1), Complete implies wrapper balance (R2-I2), a wrapper close
// returns to Top and the remainder is validated (R2-I3), complete and partial next
// top-level entries agree on the previous call's boundary (R2-I4), declared-tool
// identity holds in strict and tolerant alike (R2-I5), and fenced final content never
// latches a tool marker (R2-I6).
// ---------------------------------------------------------------------------

// Shared Round-2 assertion runner: one-shot (the given mode, every finish reason the
// parser API accepts) plus streamed (1/2/3/5/7-byte chunks, the same mode). The streaming
// terminal must equal the one-shot parse exactly (visible + terminal content, calls,
// diagnostics), and the one-shot result must match the expected structured outcome or the
// verbatim fallback. Fixtures must not end in format whitespace (the fallback is verbatim).
int check_round2_region(const std::string& text, const fi::ToolCallOutputContract& contract,
                        bool tolerant, std::vector<std::string> expected_names,
                        ninfer::ToolCallParseFallbackReason expected_reason,
                        std::string_view message) {
    using FinishReason = ninfer::FinishReason;
    int failures = 0;
    for (const FinishReason reason : {ninfer::FinishReason::StopToken, ninfer::FinishReason::StopString,
                                      ninfer::FinishReason::OutputLimit, ninfer::FinishReason::ContextCapacity,
                                      ninfer::FinishReason::Cancelled}) {
        const auto parsed =
            fi::parse_qwen_tool_call_output(text, 64, contract, tolerant, reason);
        failures += check(parsed.is_tool_call_response == !expected_names.empty(),
                          (std::string("response flag: ") + std::string(message)).c_str());
        failures += check(parsed.tool_calls.size() == expected_names.size(),
                          (std::string("call count: ") + std::string(message)).c_str());
        for (std::size_t i = 0; i < parsed.tool_calls.size() && i < expected_names.size(); ++i) {
            failures += check(parsed.tool_calls[i].name == expected_names[i],
                              (std::string("call name: ") + std::string(message)).c_str());
        }
        failures += check(parsed.diagnostics.fallback_reason == expected_reason,
                          (std::string("fallback reason: ") + std::string(message)).c_str());
        if (expected_names.empty()) {
            failures += check(parsed.content == text,
                              (std::string("verbatim fallback content: ") + std::string(message))
                                  .c_str());
        }
        // Streaming (every chunk size) must reproduce the one-shot result exactly.
        auto contract_ptr = std::make_shared<fi::ToolCallOutputContract>(contract);
        for (const std::size_t chunk : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5},
                                       std::size_t{7}}) {
            fi::ToolCallOutputDecoder decoder(contract_ptr, 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < text.size(); offset += chunk) {
                visible += decoder.feed(std::string_view(text).substr(offset, chunk));
            }
            auto terminal = decoder.finish(reason);
            failures += check(
                terminal.tool_calls.size() == parsed.tool_calls.size() &&
                    visible + terminal.content == parsed.content,
                (std::string("streaming diverged from one-shot: ") + std::string(message))
                    .c_str());
            for (std::size_t i = 0; i < terminal.tool_calls.size(); ++i) {
                failures += check(terminal.tool_calls[i].name == parsed.tool_calls[i].name &&
                                      terminal.tool_calls[i].arguments_json ==
                                          parsed.tool_calls[i].arguments_json,
                                  (std::string("streaming call diverged: ") + std::string(message))
                                      .c_str());
            }
        }
    }
    return failures;
}
 // Round-3 per-reason assertion runner: like check_round2_region, but the expected outcome
 // may differ between the natural stop (StopToken) and the cut reasons (StopString,
 // OutputLimit, ContextCapacity, Cancelled) — R3-03 splits tolerant commits by finish
 // reason.
 struct Round3Outcome {
     std::vector<std::string> names;
     ninfer::ToolCallParseFallbackReason reason;
 };
 int check_round3_region(const std::string& text, const fi::ToolCallOutputContract& contract,
                         bool tolerant, const Round3Outcome& natural, const Round3Outcome& cut,
                         std::string_view message) {
     using FinishReason = ninfer::FinishReason;
     int failures = 0;
     for (const FinishReason reason : {ninfer::FinishReason::StopToken, ninfer::FinishReason::StopString,
                                       ninfer::FinishReason::OutputLimit, ninfer::FinishReason::ContextCapacity,
                                       ninfer::FinishReason::Cancelled}) {
         const Round3Outcome& expected = reason == ninfer::FinishReason::StopToken ? natural : cut;
         const auto parsed =
             fi::parse_qwen_tool_call_output(text, 64, contract, tolerant, reason);
         failures += check(parsed.is_tool_call_response == !expected.names.empty(),
                           (std::string("response flag: ") + std::string(message)).c_str());
         failures += check(parsed.tool_calls.size() == expected.names.size(),
                           (std::string("call count: ") + std::string(message)).c_str());
         for (std::size_t i = 0; i < parsed.tool_calls.size() && i < expected.names.size(); ++i) {
             failures += check(parsed.tool_calls[i].name == expected.names[i],
                               (std::string("call name: ") + std::string(message)).c_str());
         }
         failures += check(parsed.diagnostics.fallback_reason == expected.reason,
                           (std::string("fallback reason: ") + std::string(message)).c_str());
         if (expected.names.empty()) {
             failures += check(parsed.content == text,
                               (std::string("verbatim fallback content: ") +
                                std::string(message))
                                   .c_str());
         }
         auto contract_ptr = std::make_shared<fi::ToolCallOutputContract>(contract);
         for (const std::size_t chunk : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                                         std::size_t{5},  std::size_t{7}}) {
             fi::ToolCallOutputDecoder decoder(contract_ptr, 64, tolerant);
             std::string visible;
             for (std::size_t offset = 0; offset < text.size(); offset += chunk) {
                 visible += decoder.feed(std::string_view(text).substr(offset, chunk));
             }
             auto terminal = decoder.finish(reason);
             failures += check(terminal.tool_calls.size() == parsed.tool_calls.size() &&
                                   visible + terminal.content == parsed.content &&
                                   terminal.diagnostics == parsed.diagnostics,
                               (std::string("streaming diverged from one-shot: ") +
                                std::string(message))
                                   .c_str());
         }
     }
     return failures;
 }


// CR1/R2-I1: a failed wrapper owns its still-unclosed scope. No marker located inside it
// may become a recovery entry; the entire region falls back to content in strict and
// tolerant alike (non-execution over recovery when the wire bytes cannot prove scope exit).
int test_failed_open_wrapper_never_recovers_nested_wrapper() {
    const auto contract = contract_from_definitions({
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    int failures = 0;
    // The spec's exact reproducer: invalid outer name, nested valid tool_call.
    const std::string nested_tool_call =
        "<tool_call>\n"
        "<function=bad.name>\n"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\necho SHOULD_NOT_EXECUTE\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    failures += check_round2_region(
        nested_tool_call, *contract, false, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 nested tool_call (strict)");
    failures += check_round2_region(
        nested_tool_call, *contract, true, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 nested tool_call (tolerant)");
    // Nested <function_calls> instead of <tool_call>.
    const std::string nested_function_calls =
        "<tool_call>\n"
        "<function=bad.name>\n"
        "<function_calls>\n"
        "<function=bash>\n"
        "<parameter=command>\necho SHOULD_NOT_EXECUTE\n</parameter>\n"
        "</function>\n"
        "</function_calls>";
    failures += check_round2_region(
        nested_function_calls, *contract, false, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 nested function_calls (strict)");
    failures += check_round2_region(
        nested_function_calls, *contract, true, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 nested function_calls (tolerant)");
    // A failed <function_calls> wrapper with a nested <tool_call>.
    const std::string calls_wrapper =
        "<function_calls>\n"
        "<function=bad.name>\n"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\necho SHOULD_NOT_EXECUTE\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    failures += check_round2_region(
        calls_wrapper, *contract, false, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 function_calls->tool_call (strict)");
    failures += check_round2_region(
        calls_wrapper, *contract, true, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 function_calls->tool_call (tolerant)");
    // An invalid header instead of an invalid name.
    const std::string invalid_header =
        "<tool_call>\n"
        "<function bad.name>\n"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\necho SHOULD_NOT_EXECUTE\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    failures += check_round2_region(
        invalid_header, *contract, false, {},
        ninfer::ToolCallParseFallbackReason::MalformedStructure, "CR1 invalid outer header (strict)");
    failures += check_round2_region(
        invalid_header, *contract, true, {},
        ninfer::ToolCallParseFallbackReason::MalformedStructure, "CR1 invalid outer header (tolerant)");
    // A complete outer header whose body breaks on a nested marker before the slice end:
    // the break is a definitive structural failure with the wrapper open (scope owned, no retry).
    const std::string truncated_header =
        "<tool_call>\n"
        "<function=bash\n"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\necho SHOULD_NOT_EXECUTE\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    failures += check_round2_region(
        truncated_header, *contract, false, {},
        ninfer::ToolCallParseFallbackReason::MalformedStructure, "CR1 truncated outer header (strict)");
    failures += check_round2_region(
        truncated_header, *contract, true, {},
        ninfer::ToolCallParseFallbackReason::MalformedStructure, "CR1 truncated outer header (tolerant)");
    // A nested valid wrapper after arbitrary whitespace/prose.
    const std::string prose_then_nested =
        "<tool_call>\n"
        "<function=bad.name>\n\n"
        "some prose that is not markup\n"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\necho SHOULD_NOT_EXECUTE\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    failures += check_round2_region(
        prose_then_nested, *contract, false, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 prose before nested wrapper (strict)");
    failures += check_round2_region(
        prose_then_nested, *contract, true, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 prose before nested wrapper (tolerant)");
    // Two nested wrapper candidates.
    const std::string two_nested =
        "<tool_call>\n"
        "<function=bad.name>\n"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\necho A\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<function_calls>\n"
        "<function=bash>\n"
        "<parameter=command>\necho B\n</parameter>\n"
        "</function>\n"
        "</function_calls>";
    failures += check_round2_region(
        two_nested, *contract, false, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 two nested candidates (strict)");
    failures += check_round2_region(
        two_nested, *contract, true, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 two nested candidates (tolerant)");
    // A valid-looking nested wrapper followed by a genuine top-level wrapper after the
    // failed wrapper's visible close: the scope exit is not provable from the wire bytes
    // (the failed function broke the structure), so non-execution wins over availability.
    const std::string proven_boundary =
        "<tool_call>\n"
        "<function=bad.name>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\necho AFTER_PROVEN_BOUNDARY\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    failures += check_round2_region(
        proven_boundary, *contract, false, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 call after visible close (strict)");
    failures += check_round2_region(
        proven_boundary, *contract, true, {},
        ninfer::ToolCallParseFallbackReason::InvalidToolName, "CR1 call after visible close (tolerant)");
    return failures;
}

// CR1: recovery restarts only after a proven top-level scope. A failed bare region (no
// wrapper open at the break) before a real wrapper is still recovered at the wrapper (F7);
// a failed wrapper scope is never re-entered (the previous test).
int test_recovery_only_restarts_after_proven_scope() {
    const auto contract = contract_from_definitions({
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    int failures = 0;
    // A failed bare function (no wrapper) before a real wrapper: the break leaves no
    // wrapper open, so the top-level retry may restart at the wrapper.
    const std::string bare_failed =
        "<function=bad.name>\n"
        "<parameter=command>\ncut\n" +
        tool_call("bash", {{"command", "echo OK"}});
    failures += check_round2_region(bare_failed, *contract, false, {"bash"},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR1 bare failed region recovered at wrapper (strict)");
    failures += check_round2_region(bare_failed, *contract, true, {"bash"},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR1 bare failed region recovered at wrapper (tolerant)");
    // Every single split of the CR1 reproducer must agree with the one-shot result (the
    // failed wrapper's scope never exposes the nested call, at any chunk boundary).
    const std::string nested =
        "<tool_call>\n"
        "<function=bad.name>\n"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\necho SHOULD_NOT_EXECUTE\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    const auto one_shot = fi::parse_qwen_tool_call_output(nested, 64, *contract);
    bool every_split_matches = one_shot.tool_calls.empty() && !one_shot.is_tool_call_response;
    for (std::size_t split = 0; split <= nested.size() && every_split_matches; ++split) {
        fi::ToolCallOutputDecoder decoder(
            std::make_shared<fi::ToolCallOutputContract>(*contract), 64);
        std::string visible = decoder.feed(std::string_view(nested).substr(0, split));
        visible += decoder.feed(std::string_view(nested).substr(split));
        auto terminal = decoder.finish(ninfer::FinishReason::None);
        every_split_matches = terminal.tool_calls.empty() && !terminal.content.empty() &&
                              visible + terminal.content == one_shot.content &&
                              terminal.diagnostics == one_shot.diagnostics;
    }
    failures += check(every_split_matches, "CR1 every-split streaming diverged from one-shot");
    return failures;
}

// CR2/R2-I2: an open <function_calls> at the region end is an objective truncation, never
// a clean completion. Strict rejects the region; tolerant may retain the function-closed
// calls with TruncatedTail. One and N calls, every finish reason (via the shared runner).
int test_function_calls_missing_wrapper_close() {
    const auto contract = contract_from_definitions({
        tool_definition("read", Json{{"path", Json{{"type", "string"}}}})});
    int failures = 0;
    const std::string one_open =
        "<function_calls>\n<function=read>\n<parameter=path>a</parameter>\n</function>";
    const std::string two_open =
        one_open + "\n<function=read>\n<parameter=path>b</parameter>\n</function>";
    failures += check_round2_region(one_open, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR2 one open call (strict)");
    failures += check_round2_region(one_open, *contract, true, {"read"},
                                    ninfer::ToolCallParseFallbackReason::TruncatedTail,
                                    "CR2 one open call (tolerant)");
    failures += check_round2_region(two_open, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR2 two open calls (strict)");
    failures += check_round2_region(two_open, *contract, true, {"read", "read"},
                                    ninfer::ToolCallParseFallbackReason::TruncatedTail,
                                    "CR2 two open calls (tolerant)");
    return failures;
}

// CR3/R2-I3: consuming </function_calls> does not consume the rest of the output: the
// state returns to Top, which validates the remainder (EOF, a next region, a partial
// marker, or trailing content).
int test_function_calls_close_then_remainder() {
    const auto contract = contract_from_definitions({
        tool_definition("read", Json{{"path", Json{{"type", "string"}}}}),
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    int failures = 0;
    const std::string closed =
        "<function_calls>\n<function=read>\n<parameter=path>a</parameter>\n</function>\n"
        "</function_calls>";
    const std::string trailing = closed + "\nEXTRA";
    const std::string whitespace = closed + "\n  ";
    const std::string next_region = closed + "\n" + tool_call("bash", {{"command", "echo ok"}});
    const std::string partial_marker = closed + "\n<tool_c";
    failures += check_round2_region(trailing, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::TrailingContent,
                                    "CR3 trailing prose (strict)");
    failures += check_round3_region(
        trailing, *contract, true,
        Round3Outcome{{"read"}, ninfer::ToolCallParseFallbackReason::TruncatedTail},
        Round3Outcome{{}, ninfer::ToolCallParseFallbackReason::TrailingContent},
        "CR3 trailing prose (tolerant)");
    failures += check_round2_region(whitespace, *contract, false, {"read"},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR3 trailing whitespace (strict)");
    failures += check_round2_region(whitespace, *contract, true, {"read"},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR3 trailing whitespace (tolerant)");
    failures += check_round2_region(next_region, *contract, false, {"read", "bash"},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR3 next valid region (strict)");
    failures += check_round2_region(next_region, *contract, true, {"read", "bash"},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR3 next valid region (tolerant)");
    failures += check_round2_region(partial_marker, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR3 partial marker (strict)");
    failures += check_round2_region(partial_marker, *contract, true, {"read"},
                                    ninfer::ToolCallParseFallbackReason::TruncatedTail,
                                    "CR3 partial marker (tolerant)");
    return failures;
}

// CR4/R2-I4: a complete next top-level entry must not make the previous function's close
// look like quoted payload. Every supported entry form composes with every other.
int test_entry_forms_compose_consistently() {
    const auto contract = contract_from_definitions({
        tool_definition("read", Json{{"path", Json{{"type", "string"}}}}),
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    int failures = 0;
    const std::string bare_read =
        "<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>";
    const std::string invoke_read =
        "<invoke=read>\n<parameter=path>\na.txt\n</parameter>\n</invoke>";
    const std::string bare_bash =
        "<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>";
    const std::string tc_bash = tool_call("bash", {{"command", "echo ok"}});
    const std::string fc_bash =
        "<function_calls>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n"
        "</function_calls>";
    for (const std::string& previous : {bare_read, invoke_read}) {
        failures += check_round2_region(previous + "\n" + tc_bash, *contract, false,
                                        {"read", "bash"}, ninfer::ToolCallParseFallbackReason::None,
                                        "CR4 bare->tool_call (strict)");
        failures += check_round2_region(previous + "\n" + tc_bash, *contract, true,
                                        {"read", "bash"}, ninfer::ToolCallParseFallbackReason::None,
                                        "CR4 bare->tool_call (tolerant)");
        failures += check_round2_region(previous + "\n" + fc_bash, *contract, false,
                                        {"read", "bash"}, ninfer::ToolCallParseFallbackReason::None,
                                        "CR4 bare->function_calls (strict)");
        failures += check_round2_region(previous + "\n" + fc_bash, *contract, true,
                                        {"read", "bash"}, ninfer::ToolCallParseFallbackReason::None,
                                        "CR4 bare->function_calls (tolerant)");
        failures += check_round2_region(previous + "\n" + bare_bash, *contract, false,
                                        {"read", "bash"}, ninfer::ToolCallParseFallbackReason::None,
                                        "CR4 bare->bare (strict)");
        failures += check_round2_region(previous + "\n" + bare_bash, *contract, true,
                                        {"read", "bash"}, ninfer::ToolCallParseFallbackReason::None,
                                        "CR4 bare->bare (tolerant)");
    }
    return failures;
}

// CR4: the previous parameter close is a structural boundary whether the next entry is
// complete or a cut prefix: at every cut after the function close, the read call is
// retained (tolerant) or the region stays a single clean rejection (strict); no cut may
// reopen the previous value as payload.
int test_complete_and_partial_next_entry_have_same_boundary_result() {
    const auto contract = contract_from_definitions({
        tool_definition("read", Json{{"path", Json{{"type", "string"}}}}),
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    const std::string text = "<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n"
                             "<tool_call>\n"
                             "<function=bash>\n"
                             "<parameter=command>\necho ok\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>";
    const std::size_t fn_close_end = text.find("</function>") + 11;
    int failures = 0;
    // A clean end right after the first function close: committed in both modes without a
    // diagnostic (the function close is the executability boundary).
    {
        const std::string after_first = text.substr(0, fn_close_end + 1);
        for (const bool mode : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(after_first, 64, *contract, mode);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                                  parsed.tool_calls.front().name == "read" &&
                                  parsed.diagnostics.fallback_reason ==
                                      ninfer::ToolCallParseFallbackReason::None,
                              (std::string("CR4 boundary: clean end after first close (") +
                               (mode ? "tolerant)" : "strict)"))
                                  .c_str());
        }
    }
    for (std::size_t cut = fn_close_end + 2; cut <= text.size(); ++cut) {
        const std::string cut_text(text.substr(0, cut));
        if (cut == text.size()) {
            failures += check_round2_region(cut_text, *contract, false, {"read", "bash"},
                                            ninfer::ToolCallParseFallbackReason::None,
                                            "CR4 boundary: complete text (strict)");
            failures += check_round2_region(cut_text, *contract, true, {"read", "bash"},
                                            ninfer::ToolCallParseFallbackReason::None,
                                            "CR4 boundary: complete text (tolerant)");
            continue;
        }
        // Strict is all-or-nothing: every cut before the complete text is a single clean
        // rejection (EndOfInput break -> MalformedStructure); the committed read call is not
        // executable without the complete region.
        const auto strict = fi::parse_qwen_tool_call_output(cut_text, 64, *contract);
        failures += check(!strict.is_tool_call_response && strict.tool_calls.empty() &&
                              strict.diagnostics.fallback_reason ==
                                  ninfer::ToolCallParseFallbackReason::MalformedStructure,
                          (std::string("CR4 boundary: strict cut ") + std::to_string(cut)).c_str());
        // Tolerant retains the committed read call at every cut (TruncatedTail); a cut that
        // completes the bash call retains that call as well. Exact value bytes.
        const auto tolerant = fi::parse_qwen_tool_call_output(cut_text, 64, *contract, true);
        failures += check(tolerant.is_tool_call_response && tolerant.tool_calls.size() >= 1 &&
                              tolerant.tool_calls.size() <= 2 &&
                              tolerant.tool_calls.front().name == "read" &&
                              Json::parse(tolerant.tool_calls.front().arguments_json)
                                  .at("path")
                                  .get<std::string>() ==
                                  "a.txt" &&
                              tolerant.diagnostics.fallback_reason ==
                                  ninfer::ToolCallParseFallbackReason::TruncatedTail,
                          (std::string("CR4 boundary: tolerant cut ") + std::to_string(cut)).c_str());
    }
    return failures;
}

// CR5/R2-I5: syntax tolerance never disables the declared-tool identity policy. With
// enforce_declared_names active, no undeclared name is executable in strict or tolerant,
// and the identity policy is identical in both modes.
int test_tolerant_rejects_undeclared_tools() {
    const auto contract = contract_from_definitions({
        tool_definition("read", Json{{"path", Json{{"type", "string"}}}})});
    int failures = 0;
    // The spec's exact reproducer: a syntactically valid undeclared name inside a wrapper.
    const std::string undeclared =
        "<tool_call>\n<function=example_function_name>\n</function>\n</tool_call>";
    failures += check_round2_region(undeclared, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                                    "CR5 undeclared single (strict)");
    failures += check_round2_region(undeclared, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                                    "CR5 undeclared single (tolerant)");
    // A valid first call followed by a later undeclared call: strict rejects the whole
    // region; tolerant retains only the earlier already-complete declared call.
    const std::string mixed = tool_call("read", {{"path", "a"}}) + "\n" +
                              "<tool_call>\n<function=not_declared>\n</function>\n</tool_call>";
    failures += check_round2_region(mixed, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                                    "CR5 valid then undeclared (strict)");
    failures += check_round3_region(
        mixed, *contract, true,
        Round3Outcome{{"read"}, ninfer::ToolCallParseFallbackReason::TruncatedTail},
        Round3Outcome{{}, ninfer::ToolCallParseFallbackReason::UndeclaredTool},
        "CR5 valid then undeclared (tolerant)");
    // A bare undeclared function (no wrapper).
    const std::string bare_undeclared =
        "<function=unknown>\n<parameter=x>\nv\n</parameter>\n</function>";
    failures += check_round2_region(bare_undeclared, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                                    "CR5 bare undeclared (strict)");
    failures += check_round2_region(bare_undeclared, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                                    "CR5 bare undeclared (tolerant)");
    return failures;
}

// CR5: the declared-name matrix of the spec (declared: read, bash, write). Declared names
// pass with a valid structure; undeclared syntactically valid names are never emitted;
// a syntactically invalid name is InvalidToolName; the policy does not change with
// tolerance.
int test_strict_and_tolerant_share_declared_name_policy() {
    const auto contract = contract_from_definitions({
        tool_definition("read", Json{{"path", Json{{"type", "string"}}}}),
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}}}),
        tool_definition("write", Json{{"content", Json{{"type", "string"}}}})});
    int failures = 0;
    const struct {
        const char* name;
        bool declared;
        bool syntactically_valid;
    } matrix[] = {{"read", true, true},
                  {"bash", true, true},
                  {"write", true, true},
                  {"function_name", false, true},
                  {"example_function_name", false, true},
                  {"unknown", false, true},
                  {"bad.name", false, false}};
    for (const auto& entry : matrix) {
        const std::string text = tool_call(entry.name);
        const auto strict = fi::parse_qwen_tool_call_output(text, 64, *contract);
        const auto tolerant = fi::parse_qwen_tool_call_output(text, 64, *contract, true);
        const char* context = entry.name;
        failures += check(strict.is_tool_call_response == entry.declared &&
                              tolerant.is_tool_call_response == entry.declared &&
                              strict.tool_calls.size() == (entry.declared ? 1 : 0) &&
                              tolerant.tool_calls.size() == (entry.declared ? 1 : 0),
                          (std::string("CR5 name acceptance diverged: ") + context).c_str());
        const auto expected_reason = entry.declared
                                         ? ninfer::ToolCallParseFallbackReason::None
                                         : entry.syntactically_valid
                                               ? ninfer::ToolCallParseFallbackReason::UndeclaredTool
                                               : ninfer::ToolCallParseFallbackReason::InvalidToolName;
        failures += check(strict.diagnostics.fallback_reason == expected_reason &&
                              tolerant.diagnostics.fallback_reason == expected_reason,
                          (std::string("CR5 name reason diverged: ") + context).c_str());
    }
    return failures;
}

// CR5 defense in depth: with the declared-name policy active, no generated call may leave
// the frontend with a name absent from the contract. The invariant is checked on every
// output the public API can produce (one-shot and streaming, both modes, the full name
// matrix) — the API boundary, not the state-machine branch.
int test_output_normalization_cannot_emit_out_of_set_name() {
    const auto contract = contract_from_definitions({
        tool_definition("read", Json{{"path", Json{{"type", "string"}}}})});
    int failures = 0;
    for (const char* name : {"read", "function_name", "example_function_name",
                             "unknown", "bad.name"}) {
        const std::string text = tool_call(name);
        for (const bool tolerant : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, *contract, tolerant);
            for (const auto& call : parsed.tool_calls) {
                failures += check(call.name == "read",
                                  (std::string("CR5 out-of-set name emitted: ") + name).c_str());
            }
            fi::ToolCallOutputDecoder decoder(
                std::make_shared<fi::ToolCallOutputContract>(*contract), 64, tolerant);
            std::string visible;
            for (std::size_t i = 0; i < text.size(); ++i) {
                visible += decoder.feed(std::string_view(text).substr(i, 1));
            }
            for (const auto& call : decoder.finish(ninfer::FinishReason::None).tool_calls) {
                failures += check(call.name == "read",
                                  (std::string("CR5 streamed out-of-set name emitted: ") + name)
                                      .c_str());
            }
        }
    }
    return failures;
}

// CR5: the chat template's system prompt quotes <function=example_function_name> as the
// format example. A model that echoes the example or a bare placeholder name must not
// produce an executable call.
int test_placeholder_function_name_is_not_executable() {
    const auto contract = contract_from_definitions({
        tool_definition("read", Json{{"path", Json{{"type", "string"}}}})});
    int failures = 0;
    failures += check_round2_region(tool_call("function_name"), *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                                    "CR5 placeholder name (strict)");
    failures += check_round2_region(tool_call("function_name"), *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                                    "CR5 placeholder name (tolerant)");
    failures += check_round2_region(tool_call("example_function_name"), *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                                    "CR5 template example name (strict)");
    failures += check_round2_region(tool_call("example_function_name"), *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                                    "CR5 template example name (tolerant)");
    return failures;
}

// CR6/R2-I6: syntactically valid tool markup inside a recognized final-content fenced
// code block is ordinary content and never latches. Outside the fence, marker scanning
// resumes; an unclosed fence stays open through EOF.
int test_fenced_content_never_latches() {
    const auto contract = contract_from_definitions({
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    int failures = 0;
    // N-09 f: the fence suppressed the marker, so the fence fields are visible. The closed-
    // fence fixtures report ended_in_unclosed_fence false; the unclosed one reports true.
    auto check_fence = [&](const std::string& text, bool expect_unclosed, const char* label) {
        const auto o = fi::parse_qwen_tool_call_output(text, 64, *contract);
        return check(o.diagnostics.fenced_markers_suppressed >= 1 &&
                        o.diagnostics.ended_in_unclosed_fence == expect_unclosed,
                     std::string(label));
    };
    const std::string example_block =
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\necho example\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    const std::string real_call = tool_call("bash", {{"command", "echo real"}});
    // Backtick fence with an info string.
    const std::string backtick = "Example:\n\n```xml\n" + example_block + "\n```\n";
    failures += check_round2_region(backtick, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 backtick fence (strict)");
    failures += check_round2_region(backtick, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 backtick fence (tolerant)");
    failures += check_fence(backtick, false, "CR6 backtick fence: markers suppressed, fence closed");
    // Tilde fence without an info string.
    const std::string tilde = "~~~~\n" + example_block + "\n~~~~\n";
    failures += check_round2_region(tilde, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 tilde fence (strict)");
    failures += check_round2_region(tilde, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 tilde fence (tolerant)");
    failures += check_fence(tilde, false, "CR6 tilde fence: markers suppressed, fence closed");
    // An unclosed fence stays open through EOF: the marker inside is content.
    const std::string unclosed = "```xml\n" + example_block + "\n";
    failures += check_round2_region(unclosed, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 unclosed fence (strict)");
    failures += check_round2_region(unclosed, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 unclosed fence (tolerant)");
    failures += check_fence(unclosed, true, "CR6 unclosed fence: markers suppressed, fence open");
    // A real call immediately after the closing fence still executes; the fenced example
    // stays content.
    const std::string after_fence = "```xml\n" + example_block + "\n```\n" + real_call;
    failures += check_round2_region(after_fence, *contract, false, {"bash"},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 real call after closed fence (strict)");
    failures += check_round2_region(after_fence, *contract, true, {"bash"},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 real call after closed fence (tolerant)");
    failures += check_fence(after_fence, false, "CR6 after closed fence: markers suppressed, fence closed");
    // A longer fence contains shorter backtick lines: the inner fences are content.
    const std::string nested_fence = "````\n```\n" + example_block + "\n```\n````\n";
    failures += check_round2_region(nested_fence, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 longer fence contains shorter (strict)");
    failures += check_round2_region(nested_fence, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 longer fence contains shorter (tolerant)");
    failures += check_fence(nested_fence, false, "CR6 nested fence: markers suppressed, fence closed");
    // The opening fence line's info string may carry a marker: it is content.
    const std::string info_string = "```xml <tool_call>\n" + example_block + "\n```\n";
    failures += check_round2_region(info_string, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 fence info string (strict)");
    failures += check_round2_region(info_string, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 fence info string (tolerant)");
    failures += check_fence(info_string, false, "CR6 fence info string: markers suppressed, fence closed");
    // CRLF line endings.
    const std::string crlf = "Example:\r\n\r\n```xml\r\n" + example_block + "\r\n```\r\n";
    failures += check_round2_region(crlf, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 CRLF fence (strict)");
    failures += check_round2_region(crlf, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 CRLF fence (tolerant)");
    failures += check_fence(crlf, false, "CR6 CRLF fence: markers suppressed, fence closed");
    // A complete call before a fence: the terminal policy decides (strict rejects the
    // trailing fence block, tolerant retains the call only at a natural stop; a cut
    // finish reason leaves nothing committable behind the definitive break, R3-03).
    const std::string before_fence = real_call + "\nExample:\n\n```\ncontent\n```\n";
    failures += check_round2_region(before_fence, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::TrailingContent,
                                    "CR6 call before fence (strict)");
    failures += check_round3_region(
        before_fence, *contract, true,
        Round3Outcome{{"bash"}, ninfer::ToolCallParseFallbackReason::TruncatedTail},
        Round3Outcome{{}, ninfer::ToolCallParseFallbackReason::TrailingContent},
        "CR6 call before fence (tolerant)");
    return failures;
}

// CR6/R2-I6 streaming invariance: the fence guard is a deterministic per-byte automaton,
// so the result must be independent of the chunk partition. Every single split of a
// fence plus real-call text must agree with the one-shot parse.
int test_fence_split_at_every_byte() {
    const auto contract = contract_from_definitions({
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    const std::string text = "Example:\n\n```xml\n"
                             "<tool_call>\n"
                             "<function=bash>\n"
                             "<parameter=command>\necho example\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "```\n"
                             "<tool_call>\n"
                             "<function=bash>\n"
                             "<parameter=command>\necho real\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>";
    const auto one_shot = fi::parse_qwen_tool_call_output(text, 64, *contract);
    int failures = 0;
    failures += check(one_shot.is_tool_call_response && one_shot.tool_calls.size() == 1 &&
                          one_shot.tool_calls.front().name == "bash",
                      "CR6 every split: the one-shot parse lost the real call");
    for (std::size_t split = 0; split <= text.size(); ++split) {
        fi::ToolCallOutputDecoder decoder(
            std::make_shared<fi::ToolCallOutputContract>(*contract), 64);
        std::string visible = decoder.feed(std::string_view(text).substr(0, split));
        visible += decoder.feed(std::string_view(text).substr(split));
        auto terminal = decoder.finish(ninfer::FinishReason::None);
        failures += check(
            terminal.tool_calls.size() == one_shot.tool_calls.size() &&
                visible + terminal.content == one_shot.content &&
                terminal.diagnostics == one_shot.diagnostics &&
                (terminal.tool_calls.empty() ||
                 terminal.tool_calls.front().arguments_json ==
                     one_shot.tool_calls.front().arguments_json),
            (std::string("CR6 every split: split ") + std::to_string(split) +
             " diverged from one-shot")
                .c_str());
    }
    return failures;
}


// ===== Round 10 (R10-01/02/03): indented literal lines =====

std::string r10_indent_lines(const std::string& text, const std::string& prefix) {
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
}

fi::ToolCallParsePolicy r10_policy(ninfer::ToolCallSyntaxMode syntax,
                                   ninfer::ToolCallIntentPolicy intent) {
    fi::ToolCallParsePolicy policy;
    policy.max_name_length = 64;
    policy.syntax          = syntax;
    policy.intent          = intent;
    return policy;
}

// Runs one machine over `text` (byte-wise when split == 1, one-shot otherwise) and returns
// the finish result plus the published visible bytes.
fi::ToolCallStreamResult r10_machine_run(const fi::ToolCallParsePolicy& policy,
                                         const std::string& text, std::size_t split,
                                         std::string* published = nullptr) {
    fi::ToolCallStreamParser machine(policy);
    std::string visible;
    if (split == 1) {
        for (std::size_t i = 0; i < text.size(); ++i) {
            visible += machine.feed(std::string_view(text.data() + i, 1));
        }
    } else {
        visible += machine.feed(std::string_view(text));
    }
    if (published != nullptr) { *published = visible; }
    return machine.finish(ninfer::FinishReason::None);
}

int test_r10_indented_wrapper_is_literal() {
    // R10-01 / §9.1: an indented wrapper (every non-empty line + 4 spaces) is ordinary
    // content: 0 calls, byte-identical, marker_seen false, indented diagnostic set.
    int failures = 0;
    const std::string call     = tool_call("bash", {{"command", "echo example"}});
    const std::string indented = r10_indent_lines(call, "    ");
    const std::string tabbed   = r10_indent_lines(call, "\t"); // §9.2
    for (const std::string& fixture : {indented, tabbed}) {
        for (const auto syntax :
             {ninfer::ToolCallSyntaxMode::QwenWrappedNative, ninfer::ToolCallSyntaxMode::Compatibility}) {
            for (const auto intent :
                 {ninfer::ToolCallIntentPolicy::TemplateCompatible,
                  ninfer::ToolCallIntentPolicy::RequireToolAtContentStart}) {
                for (const std::size_t split : {std::size_t{1}, fixture.size()}) {
                    std::string published;
                    const auto res = r10_machine_run(r10_policy(syntax, intent), fixture, split, &published);
                    failures += check(res.marker_seen == false, "R10 indented wrapper latched a marker");
                    failures += check(res.indented_markers_suppressed >= 1,
                                      "R10 indented wrapper missing the suppression diagnostic");
                    failures += check(res.fenced_markers_suppressed == 0,
                                      "R10 indented wrapper double-counted as fenced");
                    failures += check(res.ended_in_unclosed_fence == false,
                                      "R10 indented wrapper reported an unclosed fence");
                    failures += check(res.status == fi::ToolCallStreamStatus::Invalid &&
                                         res.failure == fi::ToolCallParseFailure::MalformedStructure,
                                      "R10 indented wrapper was not plain content");
                    failures += check(res.tail == fixture,
                                      "R10 indented content was not byte-identical");
                }
            }
        }
    }
    // Entry level: both strict and tolerant, both intent policies, reason None.
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    for (const auto syntax :
         {ninfer::ToolCallSyntaxMode::QwenWrappedNative, ninfer::ToolCallSyntaxMode::Compatibility}) {
        for (const auto intent :
             {ninfer::ToolCallIntentPolicy::TemplateCompatible,
              ninfer::ToolCallIntentPolicy::RequireToolAtContentStart}) {
            for (const bool tolerant : {false, true}) {
                const auto parsed =
                    fi::parse_qwen_tool_call_output(indented, 64, contract, tolerant,
                                                    ninfer::FinishReason::None, syntax,
                                                    ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
                                                    intent);
                failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                                  "R10 indented wrapper produced a call");
                failures += check(parsed.content == indented,
                                  "R10 indented wrapper content was not byte-identical");
                failures += check(parsed.diagnostics.fallback_reason ==
                                      ninfer::ToolCallParseFallbackReason::None,
                                  "R10 indented wrapper reported a parse failure");
                failures += check(parsed.diagnostics.indented_markers_suppressed >= 1,
                                  "R10 indented wrapper diagnostic missing at the entry");
            }
        }
    }
    return failures;
}

int test_r10_mixed_indentation_suppressed() {
    // §9.3: TAB/space mixes placing '<' at visual column >= 4 are all literal.
    int failures = 0;
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::vector<std::string> fixtures = {" \t<tool_call>",
                                               "  \t<tool_call>",
                                               "   \t<tool_call>",
                                               "\t <tool_call>"};
    for (const std::string& fixture : fixtures) {
        const auto res = r10_machine_run(r10_policy(ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                                    ninfer::ToolCallIntentPolicy::TemplateCompatible),
                                         fixture, 1);
        failures += check(res.marker_seen == false,
                          std::string("R10 mixed-indent fixture latched: ") + fixture);
        failures += check(res.indented_markers_suppressed >= 1,
                          std::string("R10 mixed-indent fixture not counted: ") + fixture);
        const auto parsed = fi::parse_qwen_tool_call_output(
            fixture, 64, contract, false, ninfer::FinishReason::None,
            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
            ninfer::ToolCallIntentPolicy::TemplateCompatible);
        failures += check(parsed.tool_calls.empty() && parsed.content == fixture,
                          "R10 mixed-indent fixture was not plain content");
        failures += check(parsed.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::None,
                          "R10 mixed-indent fixture reported a parse failure");
    }
    return failures;
}

int test_r10_three_space_control_still_latches() {
    // §9.4: three spaces is not an indented literal: the genuine call still executes.
    int failures = 0;
    const std::string call = r10_indent_lines(tool_call("bash", {{"command", "echo example"}}), "   ");
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    for (const auto syntax :
         {ninfer::ToolCallSyntaxMode::QwenWrappedNative, ninfer::ToolCallSyntaxMode::Compatibility}) {
        const auto parsed =
            fi::parse_qwen_tool_call_output(call, 64, contract, false, ninfer::FinishReason::None,
                                            syntax, ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
                                            ninfer::ToolCallIntentPolicy::TemplateCompatible);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().name == "bash",
                          "R10 3-space genuine call did not latch");
        failures += check(parsed.diagnostics.indented_markers_suppressed == 0,
                          "R10 3-space control was counted as indented");
    }
    return failures;
}

int test_r10_blank_indented_line_does_not_lock() {
    // §9.5: a whitespace-only indented line carries no visible byte and does not lock the
    // intent gate: the following genuine call executes even under SOC.
    int failures = 0;
    const std::string fixture = "    \n" + tool_call("bash", {{"command", "echo example"}});
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    for (const auto intent :
         {ninfer::ToolCallIntentPolicy::TemplateCompatible,
          ninfer::ToolCallIntentPolicy::RequireToolAtContentStart}) {
        const auto parsed =
            fi::parse_qwen_tool_call_output(fixture, 64, contract, false,
                                            ninfer::FinishReason::None,
                                            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                            ninfer::ToolCallAmbiguityPolicy::PayloadFidelity, intent);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "R10 blank indented line locked the intent gate");
        failures += check(parsed.diagnostics.indented_markers_suppressed == 0,
                          "R10 blank indented line counted a marker");
    }
    return failures;
}

int test_r10_indented_example_then_real_call() {
    // §9.6: an indented example followed by a genuine call — TemplateCompatible still
    // executes the call (the example is content); RequireToolAtContentStart locks after
    // the example's visible bytes and returns everything as text.
    int failures = 0;
    const std::string example =
        r10_indent_lines(tool_call("bash", {{"command", "echo example"}}), "    ");
    const std::string call    = tool_call("bash", {{"command", "echo real"}});
    const std::string fixture = example + "\n" + call;
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    {
        const auto parsed =
            fi::parse_qwen_tool_call_output(fixture, 64, contract, false,
                                            ninfer::FinishReason::None,
                                            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                            ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
                                            ninfer::ToolCallIntentPolicy::TemplateCompatible);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().arguments_json ==
                                  Json{{"command", "echo real"}}.dump(),
                          "R10 TC: the real call after an indented example did not execute");
        failures += check(parsed.content == example,
                          "R10 TC: the indented example was not returned as content");
    }
    {
        const auto parsed =
            fi::parse_qwen_tool_call_output(fixture, 64, contract, false,
                                            ninfer::FinishReason::None,
                                            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                            ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
                                            ninfer::ToolCallIntentPolicy::RequireToolAtContentStart);
        failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                          "R10 SOC: the real call after an indented example executed");
        failures += check(parsed.content == fixture,
                          "R10 SOC: the indented-example input was not plain content");
        failures += check(parsed.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::None,
                          "R10 SOC: the gate lock reported a parse failure");
        failures += check(parsed.diagnostics.indented_markers_suppressed >= 1,
                          "R10 SOC: the indented example was not counted");
    }
    return failures;
}

int test_r10_fenced_control_unaffected() {
    // §9.7: a marker inside a recognized fence is fence-suppressed, not indented-suppressed,
    // even when the fenced lines are indented.
    int failures = 0;
    const std::string fenced = "```xml\n    <tool_call>\n    <function=bash>\n    </function>\n"
                               "    </tool_call>\n```";
    const auto res = r10_machine_run(
        r10_policy(ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                   ninfer::ToolCallIntentPolicy::TemplateCompatible),
        fenced, 1);
    failures += check(res.fenced_markers_suppressed > 0,
                      "R10 fence control: the fenced marker was not fence-suppressed");
    failures += check(res.indented_markers_suppressed == 0,
                      "R10 fence control: the fenced marker was double-counted as indented");
    failures += check(res.marker_seen == false, "R10 fence control: a fence byte latched");
    return failures;
}

int test_r10_compat_indented_bare_function_suppressed() {
    // §9.8: Compatibility mode suppresses the indented bare <function=> entry family too.
    int failures = 0;
    const std::string bare = r10_indent_lines(
        "<function=bash>\n<parameter=command>\necho example\n</parameter>\n</function>", "    ");
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const auto parsed = fi::parse_qwen_tool_call_output(
        bare, 64, contract, false, ninfer::FinishReason::None,
        ninfer::ToolCallSyntaxMode::Compatibility,
        ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
        ninfer::ToolCallIntentPolicy::TemplateCompatible);
    failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                      "R10 compat indented bare function executed");
    failures += check(parsed.content == bare, "R10 compat indented bare function content changed");
    failures += check(parsed.diagnostics.fallback_reason ==
                          ninfer::ToolCallParseFallbackReason::None,
                      "R10 compat indented bare function reported a parse failure");
    failures += check(parsed.diagnostics.indented_markers_suppressed >= 1,
                      "R10 compat indented bare function missing the diagnostic");
    return failures;
}

int test_r10_native_bare_function_unchanged() {
    // §9.9: a bare <function=> entry at column 0 under QwenWrappedNative was already
    // non-executable; Round 10 must not change that.
    int failures = 0;
    const std::string bare =
        "<function=bash>\n<parameter=command>\necho example\n</parameter>\n</function>";
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const auto parsed = fi::parse_qwen_tool_call_output(
        bare, 64, contract, false, ninfer::FinishReason::None,
        ninfer::ToolCallSyntaxMode::QwenWrappedNative,
        ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
        ninfer::ToolCallIntentPolicy::TemplateCompatible);
    failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                      "R10 native bare function became executable");
    failures += check(parsed.content == bare, "R10 native bare function content changed");
    return failures;
}

int test_r10_indented_payload_preserved() {
    // §9.10: an indented markup line inside a parameter value of a genuine call is
    // byte-exact — the literal guard is pre-latch only.
    int failures = 0;
    const std::string fixture = "<tool_call>\n<function=write>\n<parameter=content>\n    "
                                "<tool_call>\n    example only\n    </tool_call>\n</parameter>\n"
                                "</function>\n</tool_call>";
    const std::string expected_value = "    <tool_call>\n    example only\n    </tool_call>";
    const auto contract = contract_for("write", Json{{"content", Json{{"type", "string"}}}});
    for (const auto syntax :
         {ninfer::ToolCallSyntaxMode::QwenWrappedNative, ninfer::ToolCallSyntaxMode::Compatibility}) {
        const auto parsed =
            fi::parse_qwen_tool_call_output(fixture, 64, contract, false,
                                            ninfer::FinishReason::None, syntax,
                                            ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
                                            ninfer::ToolCallIntentPolicy::TemplateCompatible);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().name == "write" &&
                              parsed.tool_calls.front().arguments_json ==
                                  Json{{"content", expected_value}}.dump(),
                          "R10 payload: the indented markup line was not byte-exact");
    }
    return failures;
}

fi::ToolCallOutputDecoder::Terminal r10_decoder_terminal(
    std::shared_ptr<const fi::ToolCallOutputContract> contract, const std::string& text,
    std::size_t split, ninfer::ToolCallSyntaxMode syntax,
    ninfer::ToolCallIntentPolicy intent) {
    fi::ToolCallOutputDecoder decoder(contract, 64, false, syntax,
                                      ninfer::ToolCallAmbiguityPolicy::PayloadFidelity, intent);
    std::string discarded;
    if (split == text.size()) {
        discarded += decoder.feed(std::string_view(text));
    } else if (split != 0) {
        discarded += decoder.feed(std::string_view(text).substr(0, split));
        discarded += decoder.feed(std::string_view(text).substr(split));
    } else {
        discarded += decoder.feed(std::string_view(text).substr(0, 0));
        discarded += decoder.feed(std::string_view(text).substr(0));
    }
    (void)discarded;
    return decoder.finish(ninfer::FinishReason::None);
}

bool r10_terminal_equal(const fi::ToolCallOutputDecoder::Terminal& a,
                        const fi::ToolCallOutputDecoder::Terminal& b) {
    if (a.content != b.content || a.tool_calls.size() != b.tool_calls.size()) { return false; }
    for (std::size_t i = 0; i < a.tool_calls.size(); ++i) {
        if (a.tool_calls[i].name != b.tool_calls[i].name ||
            a.tool_calls[i].arguments_json != b.tool_calls[i].arguments_json) {
            return false;
        }
    }
    const auto& da = a.diagnostics;
    const auto& db = b.diagnostics;
    return da.fallback_reason == db.fallback_reason && da.marker_seen == db.marker_seen &&
           da.fenced_markers_suppressed == db.fenced_markers_suppressed &&
           da.indented_markers_suppressed == db.indented_markers_suppressed &&
           da.ended_in_unclosed_fence == db.ended_in_unclosed_fence;
}

int test_r10_every_byte_split() {
    // §9.11: every split point of the new representative fixtures must equal one-shot.
    int failures = 0;
    const auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string native_example =
        r10_indent_lines(tool_call("bash", {{"command", "echo example"}}), "    ");
    const std::string tab_example = r10_indent_lines(tool_call("bash", {{"command", "echo example"}}), "\t");
    const std::string mixed_example = " \t<tool_call>";
    const std::string three_space_call =
        r10_indent_lines(tool_call("bash", {{"command", "echo real"}}), "   ");
    const std::string example_then_call = native_example + "\n" +
                                           tool_call("bash", {{"command", "echo real"}});
    const std::vector<std::pair<std::string, std::size_t>> fixtures = {
        {native_example, 0}, {tab_example, 0}, {mixed_example, 0},
        {three_space_call, 1}, {example_then_call, 1},
    };
    for (const auto& [text, expected_calls] : fixtures) {
        const auto reference =
            r10_decoder_terminal(contract, text, text.size(),
                                 ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                 ninfer::ToolCallIntentPolicy::TemplateCompatible);
        failures += check(reference.tool_calls.size() == expected_calls,
                          std::string("R10 split corpus: unexpected one-shot call count for ") + text);
        for (std::size_t split = 0; split <= text.size(); ++split) {
            const auto streamed =
                r10_decoder_terminal(contract, text, split,
                                     ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                     ninfer::ToolCallIntentPolicy::TemplateCompatible);
            failures += check(r10_terminal_equal(streamed, reference),
                              (std::string("R10 split corpus: split ") + std::to_string(split) +
                               " diverged from one-shot for ") + text);
        }
    }
    return failures;
}

int test_r10_multi_boundary_splits() {
    // §9.12: explicit splits inside indentation, the TAB transition, the marker prefix,
    // CRLF framing, the fence opener, and a failed marker candidate.
    int failures = 0;
    const auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const auto run = [&](std::string_view first, std::string_view second,
                         bool expect_call) -> int {
        const std::string text = std::string(first) + std::string(second);
        fi::ToolCallOutputDecoder decoder(contract, 64, false,
                                          ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                          ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
                                          ninfer::ToolCallIntentPolicy::TemplateCompatible);
        std::string discarded;
        discarded += decoder.feed(first);
        discarded += decoder.feed(second);
        (void)discarded;
        const auto terminal = decoder.finish(ninfer::FinishReason::None);
        return check(terminal.tool_calls.size() == (expect_call ? 1 : 0),
                     "R10 multi-boundary split gave the wrong call count");
    };
    const std::string call = tool_call("bash", {{"command", "echo example"}});
    // Split inside the indentation (after two of the four spaces).
    const std::string indented_call = r10_indent_lines(call, "    ");
    failures += run(indented_call.substr(0, 2), indented_call.substr(2), false);
    // The TAB transition.
    failures += run(" \t", "<tool_call>", false);
    // Split inside the suppressed marker prefix.
    failures += run("\t<tool_", "call>", false);
    // CRLF framing of an indented wrapper.
    {
        const std::string crlf = r10_indent_lines(
            std::string("<tool_call>\n<function=bash>\n</function>\n</tool_call>"), "    ");
        std::string crlf_text;
        for (char byte : crlf) { crlf_text += byte; if (byte == '\n') { crlf_text += '\r'; } }
        // One-shot and split: the full content is the published visible bytes plus the
        // held tail the terminal carries.
        fi::ToolCallOutputDecoder ref_decoder(contract, 64, false,
                                              ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                              ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
                                              ninfer::ToolCallIntentPolicy::TemplateCompatible);
        std::string visible = ref_decoder.feed(std::string_view(crlf_text));
        const auto reference = ref_decoder.finish(ninfer::FinishReason::None);
        fi::ToolCallOutputDecoder split_decoder(contract, 64, false,
                                                ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                                ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
                                                ninfer::ToolCallIntentPolicy::TemplateCompatible);
        std::string visible2;
        visible2 += split_decoder.feed(crlf_text.substr(0, 5));
        visible2 += split_decoder.feed(crlf_text.substr(5));
        const auto streamed = split_decoder.finish(ninfer::FinishReason::None);
        failures += check(reference.tool_calls.empty() && streamed.tool_calls.empty(),
                          "R10 CRLF indented wrapper latched");
        failures += check(r10_terminal_equal(streamed, reference),
                          "R10 CRLF indented wrapper split diverged");
        failures += check(visible + reference.content == crlf_text &&
                              visible2 + streamed.content == crlf_text,
                          "R10 CRLF indented wrapper content was not byte-identical");
        failures += check(reference.diagnostics.indented_markers_suppressed >= 1,
                          "R10 CRLF indented wrapper missing the diagnostic");
    }
    // Split inside the fence opener.
    {
        const std::string fence = "```xml\n    <tool_call>\n    </tool_call>\n```";
        const auto reference = r10_decoder_terminal(
            contract, fence, fence.size(), ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallIntentPolicy::TemplateCompatible);
        fi::ToolCallOutputDecoder decoder(contract, 64, false,
                                          ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                          ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
                                          ninfer::ToolCallIntentPolicy::TemplateCompatible);
        std::string discarded;
        discarded += decoder.feed("``");
        discarded += decoder.feed(fence.substr(2));
        (void)discarded;
        const auto streamed = decoder.finish(ninfer::FinishReason::None);
        failures += check(r10_terminal_equal(streamed, reference),
                          "R10 fence-opener split diverged");
        failures += check(reference.diagnostics.fenced_markers_suppressed > 0 &&
                              reference.diagnostics.indented_markers_suppressed == 0,
                          "R10 fence-opener split misclassified the suppression");
    }
    // A failed marker candidate on an indented line, then a genuine call.
    {
        const std::string text = "    <toolx not a marker>\n" + call;
        fi::ToolCallOutputDecoder decoder(contract, 64, false,
                                          ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                          ninfer::ToolCallAmbiguityPolicy::PayloadFidelity,
                                          ninfer::ToolCallIntentPolicy::TemplateCompatible);
        std::string visible;
        visible += decoder.feed(text.substr(0, 1));
        visible += decoder.feed(text.substr(1));
        const auto terminal = decoder.finish(ninfer::FinishReason::None);
        failures += check(terminal.tool_calls.size() == 1 &&
                              terminal.tool_calls.front().name == "bash",
                          "R10 failed-candidate line blocked the genuine call");
        failures += check(visible + terminal.content == "    <toolx not a marker>",
                          "R10 failed-candidate literal line was not content");
    }
    return failures;
}

} // namespace

int test_duplicate_parameter_keeps_last_value() {
    // R3-08 flip: a declared parameter written twice is ambiguous_structure, not a
    // last-wins repair (identical or conflicting values alike).
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;
    const std::string duplicate = tool_call("configure", {{"value", "first"}, {"value", "second"}});
    const auto parsed = fi::parse_qwen_tool_call_output(duplicate, 64, contract);

    failures += check(!parsed.is_tool_call_response,
                      "duplicate declared parameter was still accepted");
    failures += check(parsed.content == duplicate,
                      "duplicate parameter region was not returned verbatim");
    failures += check(parsed.tool_calls.empty(), "duplicate parameter still yielded a call");
    failures += check(parsed.diagnostics.fallback_reason ==
                          ninfer::ToolCallParseFallbackReason::AmbiguousStructure,
                      "duplicate parameter did not report ambiguous_structure");
    return failures;
}

int test_tolerant_recovery() {
    using Reason = ninfer::ToolCallParseFallbackReason;
    int failures = 0;
    const std::string suffixed = tool_call("configure", {{"value", "x"}}) + "\nextra answer";
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    const auto tolerant = fi::parse_qwen_tool_call_output(suffixed, 64, contract, true);
    failures += check(tolerant.is_tool_call_response, "tolerant suffix was not recovered");
    failures += check(tolerant.tool_calls.size() == 1 && tolerant.tool_calls.front().name == "configure",
                      "tolerant suffix lost the recovered call");
    failures += check(tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "tolerant suffix was not flagged as a truncated tail");
    const auto strict = fi::parse_qwen_tool_call_output(suffixed, 64, contract);
    failures += check(!strict.is_tool_call_response, "strict suffix was recovered instead of text");
    failures += check(strict.tool_calls.empty(), "strict suffix retained the recovered call");
    failures += check(strict.diagnostics.fallback_reason == Reason::TrailingContent,
                      "strict suffix was not flagged as trailing content");

    const std::string two = tool_call("first", {{"value", "a"}}) + "\n" + "<tool_call>\n<function=broken>";
    const auto tolerant_two = fi::parse_qwen_tool_call_output(two, 64, kLegacyContract, true);
    failures += check(tolerant_two.is_tool_call_response, "tolerant multi-call was not recovered");
    failures += check(tolerant_two.tool_calls.size() == 1 && tolerant_two.tool_calls.front().name == "first",
                      "tolerant multi-call kept the malformed second call");
    failures += check(tolerant_two.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "tolerant multi-call was not flagged as a truncated tail");
    const auto strict_two = fi::parse_qwen_tool_call_output(two, 64, kLegacyContract);
    failures += check(!strict_two.is_tool_call_response, "strict multi-call kept the recovered call");

    const auto decoder_contract =
        output_contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    fi::ToolCallOutputDecoder decoder(decoder_contract, 64, /*tolerant*/ true);
    std::string visible;
    constexpr std::size_t kChunk = 7;
    for (std::size_t offset = 0; offset < suffixed.size(); offset += kChunk) {
        visible += decoder.feed(std::string_view(suffixed).substr(offset, kChunk));
    }
    auto terminal = decoder.finish(ninfer::FinishReason::None);
    failures += check(visible.empty() && terminal.content.empty(),
                      "tolerant increment leaked recovered bytes to visible content");
    failures += check(terminal.tool_calls.size() == 1,
                      "tolerant increment did not commit the recovered call");
    failures += check(terminal.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "tolerant increment lost the truncated-tail diagnostic");
    return failures;
}

int test_tolerant_truncated_final_call() {
    using Reason = ninfer::ToolCallParseFallbackReason;
    const auto contract =
        output_contract_for("delete_file", Json{{"filePath", Json{{"type", "string"}}}});
    const std::string open_tag = std::string("<") + "parameter=filePath>\n";
    const std::string close_tag = std::string("</") + "parameter>";
    const std::string truncated_tool_close =
        "Now let me verify.\n"
        "<tool_call>\n"
        "<function=delete_file>\n"
        + open_tag + "/tmp/out.js\n" + close_tag + "\n" + "</function>";
    const std::string truncated_function_close = "Now let me verify.\n"
                                                 "<tool_call>\n"
                                                 "<function=delete_file>\n"
                                                 + open_tag + "/tmp/out.js\n" + close_tag;
    const std::vector<std::pair<const char*, std::string>> cases = {
        {"missing tool close", truncated_tool_close},
        {"missing function close", truncated_function_close}};
    int failures = 0;
    for (const auto& [label, text] : cases) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, *contract, /*tolerant*/ true);
        if (label == "missing tool close") {
            // A function-closed call is committed at its function close; a cut wrapper close
            // is a truncated tail that tolerant recovery retains (F2).
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                                  parsed.tool_calls.front().name == "delete_file",
                              std::string("tolerant did not recover ") + label);
            if (parsed.tool_calls.size() == 1) {
                const Json arguments = Json::parse(parsed.tool_calls.front().arguments_json);
                failures += check(arguments == Json{{"filePath", "/tmp/out.js"}},
                                  std::string("tolerant lost arguments for ") + label);
            }
        } else {
            // The function close was not consumed: the call is not executable (F2/I2) and
            // tolerant recovery commits nothing from the open function.
            failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                              std::string("tolerant executed a call for ") + label);
        }
        failures += check(parsed.diagnostics.fallback_reason == Reason::TruncatedTail,
                          std::string("tolerant did not flag ") + label + " as truncated tail");
        const auto strict = fi::parse_qwen_tool_call_output(text, 64, *contract);
        failures += check(!strict.is_tool_call_response && strict.tool_calls.empty(),
                          std::string("strict recovered a truncated final call: ") + label);
    }
    return failures;
}

int test_tolerant_missing_function_close_bracket() {
    using Reason = ninfer::ToolCallParseFallbackReason;
    const auto contract = output_contract_for(
        "memory",
        Json{{"command", Json{{"type", "string"}}}, {"path", Json{{"type", "string"}}}});
    const std::string open_cmd = std::string("<") + "parameter=command>\n";
    const std::string open_path = std::string("<") + "parameter=path>\n";
    const std::string close_tag = std::string("</") + "parameter>";
    const std::string text =
        "Let me check my memory file first.\n"
        "<tool_call>\n"
        "<function=memory\n"
        + open_cmd + "str_replace\n" + close_tag + "\n"
        + open_path + "/memories/repo/notes.md\n" + close_tag + "\n"
        "</function>\n"
        "</tool_call>";
    int failures = 0;
    const auto tolerant = fi::parse_qwen_tool_call_output(text, 64, *contract, /*tolerant*/ true);
    failures += check(tolerant.is_tool_call_response,
                      "tolerant mode did not recover a missing closing bracket after the function name");
    failures += check(tolerant.tool_calls.size() == 1, "tolerant mode recovered wrong call count");
    if (tolerant.tool_calls.size() == 1) {
        const auto& call = tolerant.tool_calls.front();
        failures += check(call.name == "memory", "recovered call lost the function name");
        const Json args = Json::parse(call.arguments_json);
        failures += check(args.at("command").get<std::string>() == "str_replace",
                          "recovered call lost command arg");
        failures += check(args.at("path").get<std::string>() == "/memories/repo/notes.md",
                          "recovered call lost path arg");
    }
    failures += check(tolerant.diagnostics.fallback_reason == Reason::None,
                      "tolerant recovery of a missing bracket reported a spurious fallback reason");
    const auto strict = fi::parse_qwen_tool_call_output(text, 64, *contract);
    failures += check(!strict.is_tool_call_response,
                      "strict mode recovered a call with a missing closing bracket after the function name");
    failures += check(strict.tool_calls.empty(),
                      "strict mode retained an invalid tool name call");
    return failures;
}

int test_tolerant_undeclared_and_value_cut() {
    using Reason = ninfer::ToolCallParseFallbackReason;
    const auto contract =
        output_contract_for("delete_file", Json{{"filePath", Json{{"type", "string"}}}});
    int failures = 0;
    const std::string open_tag = std::string("<") + "parameter=filePath>\n";
    const std::string close_tag = std::string("</") + "parameter>";
    const std::string undeclared = "<tool_call>\n"
                                   "<function=not_a_declared_tool>\n"
                                   + open_tag + "/tmp/out.js\n" + close_tag + "\n"
                                   "</function>\n"
                                   "</tool_call>";
    // R2-I5/CR5: tolerant mode repairs syntax damage; it does not accept undeclared
    // identities. The undeclared call is never emitted in either mode.
    const auto tolerant = fi::parse_qwen_tool_call_output(undeclared, 64, *contract, true);
    failures += check(!tolerant.is_tool_call_response && tolerant.tool_calls.empty() &&
                          tolerant.content == undeclared && tolerant.diagnostics.marker_seen &&
                          tolerant.diagnostics.fallback_reason == Reason::UndeclaredTool,
                      "tolerant mode emitted a syntactically valid undeclared call (CR5)");
    const auto strict = fi::parse_qwen_tool_call_output(undeclared, 64, *contract);
    failures += check(!strict.is_tool_call_response &&
                          strict.diagnostics.fallback_reason == Reason::UndeclaredTool,
                      "strict mode did not reject the undeclared-name call");

    // A value cut before its closer is never committed: its bytes may still grow into a
    // different value (P3.4). The region falls back to text with the truncated-tail reason,
    // whatever the finish reason was.
    const std::string value_cut = "Now\n"
                                  "<tool_call>\n"
                                  "<function=delete_file>\n"
                                  + open_tag + "/tmp/out";
    const auto cut_tolerant = fi::parse_qwen_tool_call_output(value_cut, 64, *contract, true);
    failures += check(!cut_tolerant.is_tool_call_response && cut_tolerant.tool_calls.empty(),
                      "tolerant mode committed a cut string value");
    failures += check(cut_tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "tolerant value-cut was not flagged as a truncated tail");
    const auto cut_strict = fi::parse_qwen_tool_call_output(value_cut, 64, *contract);
    failures += check(!cut_strict.is_tool_call_response &&
                          cut_strict.diagnostics.fallback_reason == Reason::MalformedStructure,
                      "strict mode did not reject the value-cut call");

    // A call cut after the name but before any parameter completed carries no arguments, so
    // even tolerant mode returns the region as text (with the reason recorded).
    const std::string name_only = "<tool_call>\n"
                                  "<function=delete_file>\n";
    const auto name_tolerant = fi::parse_qwen_tool_call_output(name_only, 64, *contract, true);
    failures += check(!name_tolerant.is_tool_call_response && name_tolerant.tool_calls.empty(),
                      "tolerant mode kept a zero-parameter truncated call");
    failures += check(name_tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "zero-parameter truncation was not flagged as a truncated tail");
    const auto name_strict = fi::parse_qwen_tool_call_output(name_only, 64, *contract);
    failures += check(!name_strict.is_tool_call_response &&
                          name_strict.diagnostics.fallback_reason == Reason::MalformedStructure,
                      "strict mode misclassified a zero-parameter truncated call");
    return failures;
}

int test_grammar_header_forms_one_shot() {
    const auto contract = contract_for("write", Json{{"content", Json{{"type", "string"}}}});
    int failures = 0;
    for (const char* header : {"<function name=\"write\">", "<function name = \"write\">",
                               "<function\tname=\"write\">", "<function\nname=\"write\">",
                               "<function name='write'>"}) {
        const std::string text = std::string("<tool_call>\n") + header +
                                 "\n<parameter=content>\nhi\n</parameter>\n</function>\n</tool_call>";
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls[0].name == "write" &&
                              parsed.tool_calls[0].arguments_json == "{\"content\":\"hi\"}",
                          (std::string("attribute-form header parses: ") + header).c_str());
    }
    const std::string quoted =
        "<tool_call>\n<function name=\"write\">\n<parameter filename=\"a>b\" name=\"content\">\n"
        "x\n</parameter>\n</function>\n</tool_call>";
    const auto quoted_parsed = fi::parse_qwen_tool_call_output(quoted, 64, contract);
    failures += check(quoted_parsed.is_tool_call_response &&
                          quoted_parsed.tool_calls.size() == 1 &&
                          quoted_parsed.tool_calls[0].arguments_json == "{\"content\":\"x\"}",
                      "quoted > does not break the header");

    const std::vector<std::string> broken = {
        "<tool_call>\n<function name=\"write\"junk=\"x\">\n<parameter=content>\nhi\n</parameter>\n"
        "</function>\n</tool_call>",
        "<tool_call>\n<function name=>\n<parameter=content>\nhi\n</parameter>\n</function>\n"
        "</tool_call>",
        "<tool_call>\n<function name=\"write>\n<parameter=content>\nhi\n</parameter>\n</function>\n"
        "</tool_call>",
    };
    for (const auto& text : broken) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
        failures += check(!parsed.is_tool_call_response && parsed.content == text &&
                              parsed.diagnostics.fallback_reason ==
                                  ninfer::ToolCallParseFallbackReason::MalformedStructure,
                          "broken header falls back to verbatim text");
    }
    return failures;
}

int test_streaming_recognizes_grammar_markers() {
    int failures = 0;
    const std::string long_name = "very-long-tool-name-0123456789";
    const auto contract =
        contract_for(long_name.c_str(), Json{{"subject", Json{{"type", "string"}}}});
    const std::vector<std::string> regions = {
        std::string("<function name=\"") + long_name +
        "\">\n<parameter=subject>\ndo it\n</parameter>\n</function>",
        std::string("<function=") + long_name +
        ">\n<parameter=subject>\ndo it\n</parameter>\n</function>",
        std::string("<invoke name=\"") + long_name +
        "\">\n<parameter=subject>\ndo it\n</parameter>\n</invoke>",
    };
    for (const auto& region : regions) {
        const std::string text = "Creating task. " + region;
        fi::ToolCallOutputDecoder decoder(
            std::make_shared<fi::ToolCallOutputContract>(contract), 64, false);
        std::string visible;
        for (std::size_t i = 0; i < text.size(); ++i) {
            visible += decoder.feed(text.substr(i, 1));
        }
        auto terminal = decoder.finish(ninfer::FinishReason::None);
        failures += check(visible == "Creating task." && terminal.content.empty() &&
                              terminal.tool_calls.size() == 1 &&
                              terminal.tool_calls.front().name == long_name &&
                              terminal.tool_calls.front().arguments_json ==
                                  "{\"subject\":\"do it\"}",
                          "grammar marker recognized while streaming");
    }
    return failures;
}

int test_recovery_policy_phase3() {
    using Reason = ninfer::ToolCallParseFallbackReason;
    using ninfer::FinishReason;
    using fi::ToolCallParseFailure;
    using fi::ToolCallParseProgress;
    using fi::ToolCallRecoveryDecision;
    using fi::ToolCallRecoveryPolicy;
    using fi::ToolCallRegionTermination;
    int failures = 0;

    // P3.5: the recovery decision is a pure function of the objective progress; strict and
    // tolerant consume the same progress and differ only in this decision.
    {
        ToolCallParseProgress progress;
        progress.calls.push_back(fi::ParsedFunctionCall{
            .name       = "read",
            .parameters = {fi::ParsedParameter{.name = "path", .value = "foo.cpp"}}});
        const ToolCallRecoveryPolicy strict;
        const ToolCallRecoveryPolicy tolerant{.tolerant = true};
        failures += check(fi::decide_tool_call_recovery(progress, strict).decision ==
                              ToolCallRecoveryDecision::CommitCalls,
                          "complete progress was not committed by strict");
        failures += check(fi::decide_tool_call_recovery(progress, tolerant).diagnostic ==
                              ToolCallParseFailure::None,
                          "complete progress recorded a diagnostic");

        // Input ended after a complete call whose closing tags were cut: strict maps it to a
        // structural failure. Tolerant recovery also never commits the open call (F2/I2): its
        // function close was not consumed, so its arguments are not executable, whatever the
        // argument bytes show.
        ToolCallParseProgress cut   = progress;
        cut.termination             = ToolCallRegionTermination::EndOfInput;
        cut.calls                  = {};
        cut.open_call              = fi::ParsedFunctionCall{
            .name       = "read",
            .parameters = {fi::ParsedParameter{.name = "path", .value = "foo.cpp"}}};
        failures += check(fi::decide_tool_call_recovery(cut, strict).decision ==
                              ToolCallRecoveryDecision::Reject,
                          "strict committed a truncated call");
        failures += check(fi::decide_tool_call_recovery(cut, strict).diagnostic ==
                              ToolCallParseFailure::MalformedStructure,
                          "strict truncation lost the structural failure class");
        failures += check(fi::decide_tool_call_recovery(cut, tolerant).decision ==
                              ToolCallRecoveryDecision::Reject,
                          "tolerant committed a call whose function close was not consumed");
        failures += check(fi::decide_tool_call_recovery(cut, tolerant).diagnostic ==
                              ToolCallParseFailure::TruncatedTail,
                          "tolerant truncation lost the truncated-tail diagnostic");

        // An open value is never committable, whatever the finish reason was.
        cut.open_value_open = true;
        for (const FinishReason reason : {ninfer::FinishReason::StopToken, ninfer::FinishReason::StopString,
                                          ninfer::FinishReason::OutputLimit,
                                          ninfer::FinishReason::ContextCapacity,
                                          ninfer::FinishReason::Cancelled}) {
            const ToolCallRecoveryPolicy budget{.tolerant = true, .finish_reason = reason};
            failures += check(fi::decide_tool_call_recovery(cut, budget).decision ==
                                  ToolCallRecoveryDecision::Reject,
                              "an open value was committed under a budget finish reason");
        }

        // A name-only truncation carries no arguments and is never committed.
        ToolCallParseProgress name_only = cut;
        name_only.open_value_open       = false;
        name_only.open_call             = {};
        failures += check(fi::decide_tool_call_recovery(name_only, tolerant).decision ==
                              ToolCallRecoveryDecision::Reject,
                          "a name-only truncation was committed");
        failures += check(fi::decide_tool_call_recovery(name_only, tolerant).diagnostic ==
                              ToolCallParseFailure::TruncatedTail,
                          "a name-only truncation lost the truncated-tail diagnostic");

        // Trailing content after complete calls: tolerant keeps the calls, strict rejects.
        ToolCallParseProgress trailing = progress;
        trailing.termination           = ToolCallRegionTermination::Definitive;
        trailing.failure               = ToolCallParseFailure::TrailingContent;
        failures += check(fi::decide_tool_call_recovery(trailing, strict).decision ==
                              ToolCallRecoveryDecision::Reject,
                          "strict committed calls with trailing content");
        failures += check(fi::decide_tool_call_recovery(trailing, strict).diagnostic ==
                              ToolCallParseFailure::TrailingContent,
                          "strict trailing content lost its reason class");
        failures += check(fi::decide_tool_call_recovery(trailing, tolerant).decision ==
                              ToolCallRecoveryDecision::CommitCalls,
                          "tolerant dropped calls before trailing content");
        failures += check(fi::decide_tool_call_recovery(trailing, tolerant).diagnostic ==
                              ToolCallParseFailure::TruncatedTail,
                          "tolerant trailing content was not recorded as a truncated tail");

        // A broken later call keeps the complete earlier calls under tolerant policy.
        ToolCallParseProgress later_broken = progress;
        later_broken.termination           = ToolCallRegionTermination::Definitive;
        later_broken.failure               = ToolCallParseFailure::MalformedStructure;
        failures += check(fi::decide_tool_call_recovery(later_broken, tolerant).decision ==
                              ToolCallRecoveryDecision::CommitCalls,
                          "tolerant dropped complete calls before a broken later call");
        failures += check(fi::decide_tool_call_recovery(later_broken, strict).decision ==
                              ToolCallRecoveryDecision::Reject,
                          "strict kept calls before a broken later call");

        // The empty function_calls wrapper is the one break no policy forgives.
        ToolCallParseProgress empty_wrapper = progress;
        empty_wrapper.termination           = ToolCallRegionTermination::Definitive;
        empty_wrapper.failure               = ToolCallParseFailure::MalformedStructure;
        empty_wrapper.unrecoverable_break   = true;
        failures += check(fi::decide_tool_call_recovery(empty_wrapper, tolerant).decision ==
                              ToolCallRecoveryDecision::Reject,
                          "an empty function_calls wrapper was committed");
    }

    // P3.9: the one-shot entry and the streaming decoder agree on every rule, and an output
    // budget cut (P3.6) changes nothing about what is safe to commit.
    const auto contract =
        output_contract_for("read", Json{{"path", Json{{"type", "string"}}}});
    const std::string open  = std::string("<") + "parameter=path>\n";
    const std::string close = std::string("</") + "parameter>";
    const std::string complete =
        "<tool_call>\n<function=read>\n" + open + "foo.cpp\n" + close + "\n";
    const std::string missing_wrapper = complete + "</function>";
    const std::string missing_function = complete;
    const std::string value_cut = complete + std::string("<") + "parameter=extra>\npartial";
    const std::string trailing   = complete + "</function>\n</tool_call>\nextra answer";
    auto stream = [&](std::string_view text, bool tolerant, FinishReason reason,
                      fi::ToolCallOutputDecoder::Terminal& terminal,
                      const std::shared_ptr<const fi::ToolCallOutputContract>& decoder_contract) {
        fi::ToolCallOutputDecoder decoder(decoder_contract, 64, tolerant);
        std::string visible;
        for (std::size_t offset = 0; offset < text.size(); offset += 7) {
            visible += decoder.feed(std::string_view(text).substr(offset, 7));
        }
        terminal = decoder.finish(reason);
        return visible;
    };
    for (const FinishReason reason : {ninfer::FinishReason::StopToken, ninfer::FinishReason::StopString,
                                      ninfer::FinishReason::OutputLimit,
                                      ninfer::FinishReason::ContextCapacity,
                                      ninfer::FinishReason::Cancelled}) {
        const auto one_shot = fi::parse_qwen_tool_call_output(missing_wrapper, 64, *contract,
                                                              true, reason);
        failures += check(one_shot.is_tool_call_response && one_shot.tool_calls.size() == 1 &&
                              one_shot.tool_calls.front().name == "read" &&
                              one_shot.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "missing wrapper close was not recovered");
        auto terminal = fi::ToolCallOutputDecoder::Terminal{};
        const std::string visible = stream(missing_wrapper, true, reason, terminal, contract);
        failures += check(visible.empty() && terminal.content.empty() &&
                              terminal.tool_calls.size() == 1 &&
                              terminal.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "streaming recovery of a missing wrapper close diverged from one-shot");
        const auto strict = fi::parse_qwen_tool_call_output(missing_wrapper, 64, *contract);
        failures += check(!strict.is_tool_call_response &&
                              strict.diagnostics.fallback_reason == Reason::MalformedStructure,
                          "strict recovered a missing wrapper close");

        const auto closed = fi::parse_qwen_tool_call_output(missing_function, 64, *contract,
                                                            true, reason);
        failures += check(!closed.is_tool_call_response && closed.tool_calls.empty() &&
                              closed.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "an open function was committed without its function close");

        const auto cut = fi::parse_qwen_tool_call_output(value_cut, 64, *contract, true, reason);
        failures += check(!cut.is_tool_call_response && cut.tool_calls.empty() &&
                              cut.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "a cut string value was committed");
        auto cut_terminal = fi::ToolCallOutputDecoder::Terminal{};
        stream(value_cut, true, reason, cut_terminal, contract);
        failures += check(cut_terminal.tool_calls.empty() &&
                              cut_terminal.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "streaming committed a cut string value");

        const auto tail = fi::parse_qwen_tool_call_output(trailing, 64, *contract, true, reason);
        if (reason == ninfer::FinishReason::StopToken) {
            failures += check(tail.is_tool_call_response && tail.tool_calls.size() == 1 &&
                                  tail.diagnostics.fallback_reason == Reason::TruncatedTail,
                              "complete calls before trailing prose were not recovered at a natural stop");
        } else {
            failures += check(!tail.is_tool_call_response && tail.tool_calls.empty() &&
                                  tail.diagnostics.fallback_reason == Reason::TrailingContent,
                              "tolerant committed trailing prose under a cut finish reason");
        }
        const auto tail_strict = fi::parse_qwen_tool_call_output(trailing, 64, *contract);
        failures += check(!tail_strict.is_tool_call_response &&
                              tail_strict.diagnostics.fallback_reason == Reason::TrailingContent,
                          "strict recovered calls with trailing prose");
    }

    // P3.10: markup that quotes the closing tags inside a declared string argument must not
    // commit the string early; the region is broken and falls back to text. Inside a
    // <tool_call> wrapper a function/invoke close must be followed by the wrapper close, so
    // the quoted closer pair is text and the value keeps scanning.
    const auto write_contract = output_contract_for(
        "write", Json{{"content", Json{{"type", "string"}}},
                      {"command", Json{{"type", "string"}}}});
    const std::string function_fixture =
        "<tool_call>\n<function=write>\n<parameter=content>\n"
        "A</parameter></function><function=bad.name><parameter=command>echo hi";
    const std::string invoke_fixture =
        "<tool_call>\n<invoke=write>\n<parameter=content>\n"
        "A</parameter></invoke><invoke=bad.name><parameter=command>echo hi";
    for (const std::string& fixture : {function_fixture, invoke_fixture}) {
        const auto tolerant = fi::parse_qwen_tool_call_output(fixture, 64, *write_contract, true);
        failures += check(!tolerant.is_tool_call_response && tolerant.tool_calls.empty() &&
                              tolerant.content == fixture &&
                              tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "a quoted closer committed a string value early (tolerant)");
        const auto strict = fi::parse_qwen_tool_call_output(fixture, 64, *write_contract);
        failures += check(!strict.is_tool_call_response && strict.tool_calls.empty() &&
                              strict.diagnostics.fallback_reason == Reason::MalformedStructure,
                          "a quoted closer committed a string value early (strict)");
        auto terminal = fi::ToolCallOutputDecoder::Terminal{};
        stream(fixture, true, ninfer::FinishReason::StopToken, terminal, write_contract);
        failures += check(terminal.tool_calls.empty() && terminal.content == fixture &&
                              terminal.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "streaming committed a string value at a quoted closer");
    }
    return failures;
}

// F5: <function_calls> contains a sequence of calls (legacy wire contract): after each
// function close the region may continue with another call, the wrapper close, or the region
// end. A nested wrapper open remains a definitive break.
int test_function_calls_holds_call_sequence() {
    using Reason = ninfer::ToolCallParseFallbackReason;
    const auto contract = output_contract_for("read", Json{{"path", Json{{"type", "string"}}}});
    int failures = 0;
    const std::string two_calls =
        "<function_calls>\n<function=read>\n<parameter=path>a</parameter>\n"
        "</function>\n<function=read>\n<parameter=path>b</parameter>\n"
        "</function>\n</function_calls>";
    // The complete sequence with the wrapper close: both calls commit without a diagnostic.
    for (const bool tolerant : {true, false}) {
        const auto parsed = fi::parse_qwen_tool_call_output(two_calls, 64, *contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2 &&
                              parsed.tool_calls.front().name == "read" &&
                              parsed.tool_calls.back().name == "read" &&
                              parsed.diagnostics.fallback_reason == Reason::None,
                          "a complete <function_calls> call sequence was not committed");
    }
    // R2-I2/CR2: a wrapper open at the input end is an objective truncation, never a clean
    // completion. Strict rejects (MalformedStructure); tolerant retains the closed calls
    // with the truncation diagnostic.
    const std::string two_calls_open =
        "<function_calls>\n<function=read>\n<parameter=path>a</parameter>\n"
        "</function>\n<function=read>\n<parameter=path>b</parameter>\n"
        "</function>";
    {
        const auto strict_open = fi::parse_qwen_tool_call_output(two_calls_open, 64, *contract);
        failures += check(
            !strict_open.is_tool_call_response && strict_open.tool_calls.empty() &&
                strict_open.diagnostics.fallback_reason == Reason::MalformedStructure,
            "a missing function_calls close completed cleanly in strict mode");
        const auto two_tolerant =
            fi::parse_qwen_tool_call_output(two_calls_open, 64, *contract, true);
        // CR2: an open function_calls wrapper at EOF is a truncation, not a clean end: the
        // closed calls are retained with the TruncatedTail diagnostic in tolerant mode.
        failures += check(two_tolerant.is_tool_call_response && two_tolerant.tool_calls.size() == 2 &&
                              two_tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "a missing function_calls close demoted the tolerant call sequence");
    }
    const std::string second_cut =
        "<function_calls>\n<function=read>\n<parameter=path>a</parameter>\n"
        "</function>\n<function=read>\n<parameter=path=c";
    const auto strict_cut = fi::parse_qwen_tool_call_output(second_cut, 64, *contract);
    failures +=
        check(!strict_cut.is_tool_call_response &&
                  strict_cut.diagnostics.fallback_reason == Reason::MalformedStructure,
              "strict retained a call inside a broken <function_calls> sequence");
    const auto tolerant_cut = fi::parse_qwen_tool_call_output(second_cut, 64, *contract, true);
    failures += check(tolerant_cut.is_tool_call_response && tolerant_cut.tool_calls.size() == 1 &&
                          tolerant_cut.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "tolerant did not retain the complete first call of a cut sequence");
    // A nested wrapper open at the top level of <function_calls> (after a committed function
    // close with no open value) is a definitive break: strict rejects the region; tolerant
    // applies the general retention rule to the complete pre-break call.
    const std::string nested =
        "<function_calls>\n<function=read>\n</function>\n"
        "<tool_call>\n<function=read>\n<parameter=path=b</parameter>\n"
        "</function>\n</tool_call>";
    const auto nested_strict = fi::parse_qwen_tool_call_output(nested, 64, *contract);
    failures += check(
        !nested_strict.is_tool_call_response &&
            nested_strict.diagnostics.fallback_reason == Reason::MalformedStructure,
        "a nested wrapper inside <function_calls> was not rejected (strict)");
    const auto nested_tolerant = fi::parse_qwen_tool_call_output(nested, 64, *contract, true);
    failures += check(
        nested_tolerant.is_tool_call_response && nested_tolerant.tool_calls.size() == 1 &&
            nested_tolerant.tool_calls.front().name == "read" &&
            nested_tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
        "the pre-break call was lost by tolerant nesting recovery");
    if (nested_tolerant.tool_calls.size() == 1) {
        failures += check(nested_tolerant.tool_calls.front().arguments_json.find("b") ==
                              std::string::npos,
                          "the nested wrapper's call was committed instead of the pre-break "
                          "call");
    }
    // A pre-break call whose parameter close is followed by the nested open is never a
    // boundary (the function_calls continuation after a function close allows only the
    // wrapper close or the region end): the value swallows the nested structure, the call
    // stays open, and both modes reject the region verbatim.
    const std::string nested_value =
        "<function_calls>\n<function=read>\n<parameter=path>a</parameter>\n"
        "</function>\n<tool_call>\n<function=read>\n<parameter=path=b</parameter>\n"
        "</function>\n</tool_call>";
    for (const bool tolerant : {true, false}) {
        const auto parsed = fi::parse_qwen_tool_call_output(nested_value, 64, *contract, tolerant);
        failures += check(
            !parsed.is_tool_call_response && parsed.tool_calls.empty() &&
                parsed.content == nested_value,
            "a value-swallowed nested region was not rejected verbatim");
    }
    // The streaming machine agrees with the one-shot parse on the complete sequence.
    for (const std::size_t chunk : {std::size_t{1}, std::size_t{3}, std::size_t{7}}) {
        fi::ToolCallOutputDecoder decoder(contract, 64);
        std::string visible;
        for (std::size_t at = 0; at < two_calls.size();) {
            const std::size_t take = std::min(chunk, two_calls.size() - at);
            visible += decoder.feed(two_calls.substr(at, take));
            at += take;
        }
        const auto terminal = decoder.finish(ninfer::FinishReason::None);
        failures += check(terminal.tool_calls.size() == 2 && visible.empty() &&
                              terminal.content.empty() &&
                              terminal.diagnostics.fallback_reason == Reason::None,
                          "the streaming <function_calls> sequence diverged from the one-shot "
                          "parse");
    }
    return failures;
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char ch : text) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += ch;
        }
    }
    return out;
}

// Round-3 §7 reproducer corpus (Appendix A): the R3-01 markup payload, the R3-04 retry
// ownership matrix, the R3-08 synthetic-argument matrix, the R3-05/R3-06 header and fence
// cases, and the R3-09 no-marker diagnostic.
int test_round3_spec_corpus() {
    using ninfer::FinishReason;
    using Reason = ninfer::ToolCallParseFallbackReason;
    const std::vector<std::string> defs = {
        tool_definition("write",
                        Json{{"path", Json{{"type", "string"}}},
                             {"content", Json{{"type", "string"}}}}),
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}}}),
        tool_definition("edit", Json{{"path", Json{{"type", "string"}}},
                                     {"old_string", Json{{"type", "string"}}},
                                     {"new_string", Json{{"type", "string"}}}}),
        tool_definition("search", Json{{"a/b", Json{{"type", "string"}}}}),
        tool_definition("read", Json{{"path", Json{{"type", "string"}}}})};
    const auto contract = contract_from_definitions(defs);
    const fi::ToolCallOutputContract& c = *contract;
    int failures = 0;
    const std::string E  = tool_call("bash", {{"command", "rm -rf x"}});
    const std::string Ex = tool_call("bash", {{"command", "ls"}});

    auto one_call = [&](const char* label, const fi::ParsedToolCallOutput& parsed,
                        const std::string& name, const std::string& args) {
        return check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                         parsed.tool_calls.front().name == name &&
                         parsed.tool_calls.front().arguments_json == args &&
                         parsed.diagnostics.fallback_reason == Reason::None,
                     label);
    };
    auto as_text = [&](const char* label, const fi::ParsedToolCallOutput& parsed,
                       const std::string& text, Reason reason) {
        return check(!parsed.is_tool_call_response && parsed.content == text &&
                         parsed.tool_calls.empty() &&
                         parsed.diagnostics.fallback_reason == reason,
                     label);
    };
    // R3-07 / section 7: the streamed parse must agree with the one-shot entry (same
    // verdict, calls, reason, and published content under the documented pre-region rtrim
    // rule). Fixtures under 512 bytes run every split point; larger fixtures run the
    // chunk sizes 1/2/3/5/7.
    auto stream_equals_one_shot = [&](const char* label, const std::string& text,
                                     bool tolerant, bool every_split) {
        const auto one = fi::parse_qwen_tool_call_output(text, 64, c, tolerant,
                                                         ninfer::FinishReason::StopToken);
        auto compare = [&](const std::vector<std::size_t>& points) -> bool {
            fi::ToolCallOutputDecoder dec(contract, 64, tolerant);
            std::string_view sv = text;
            std::size_t from = 0;
            std::string visible;
            for (const std::size_t pt : points) {
                visible += dec.feed(sv.substr(from, pt - from));
                from = pt;
            }
            if (from < text.size()) { visible += dec.feed(sv.substr(from)); }
            const auto term = dec.finish(ninfer::FinishReason::StopToken);
            std::string total = visible + term.content;
            if (one.is_tool_call_response) {
                while (!total.empty() && fi::is_tool_format_whitespace(total.back())) {
                    total.pop_back();
                }
            }
            bool same_calls = one.tool_calls.size() == term.tool_calls.size();
            for (std::size_t i = 0; same_calls && i < one.tool_calls.size(); ++i) {
                same_calls = one.tool_calls[i].name == term.tool_calls[i].name &&
                             one.tool_calls[i].arguments_json ==
                                 term.tool_calls[i].arguments_json;
            }
            return total == one.content &&
                   one.is_tool_call_response == !term.tool_calls.empty() && same_calls &&
                   one.diagnostics.fallback_reason == term.diagnostics.fallback_reason;
        };
        if (every_split) {
            for (std::size_t k = 0; k <= text.size(); ++k) {
                if (!compare({k})) {
                    std::printf("fail %s: streaming split %zu diverges from one-shot\n", label, k);
                    return 1;
                }
            }
        } else {
            for (const int chunk : {1, 2, 3, 5, 7}) {
                std::vector<std::size_t> points;
                for (std::size_t k = 0; k < text.size(); k += std::size_t(chunk)) {
                    points.push_back(k + std::size_t(chunk));
                }
                if (!compare(points)) {
                    std::printf("fail %s: chunk %d streaming diverges from one-shot\n", label, chunk);
                    return 1;
                }
            }
        }
        return 0;
    };

    // R3-01: a quoted example that closes its own structure stays byte-exact.
    const std::string inner = tool_call("write", {{"path", "x"}, {"content", "hello"}});
    const std::string p_h_text =
        tool_call("write", {{"path", "d.md"}, {"content", inner}});
    const std::string p_h_args =
        "{\"path\":\"d.md\",\"content\":\"" + json_escape(inner) + "\"}";
    {
        const auto p_h = fi::parse_qwen_tool_call_output(p_h_text, 64, c);
        failures += one_call("R3 P-H: the quoted example is the exact content", p_h, "write", p_h_args);
        failures += check(p_h.diagnostics.markup_tolerant_completion,
                          "R3 P-H: resolved by the Stage-2 consistent completion");
    }
    {
        const std::string r4_text = tool_call("write", {{"path", "d.md"},
                                                        {"content", "A:\n" + Ex + "\nB:\n" + Ex + "\nEnd."}});
        const std::string r4_args =
            "{\"path\":\"d.md\",\"content\":\"" + json_escape("A:\n" + Ex + "\nB:\n" + Ex + "\nEnd.") + "\"}";
        const auto r4 = fi::parse_qwen_tool_call_output(r4_text, 64, c);
        failures += one_call("R3 R4: two embedded examples stay byte-exact", r4, "write", r4_args);
        for (const int examples : {50, 200, 500, 2000}) {
            std::string big;
            for (int i = 0; i < examples; ++i) { big += Ex + "\n"; }
            const std::string scale_text = tool_call("write", {{"path", "d.md"}, {"content", big}});
            const auto scale = fi::parse_qwen_tool_call_output(scale_text, 64, c);
            failures += check(scale.is_tool_call_response && scale.tool_calls.size() == 1 &&
                                  scale.tool_calls.front().name == "write" &&
                                  scale.tool_calls.front().arguments_json ==
                                      "{\"path\":\"d.md\",\"content\":\"" + json_escape(big) + "\"}",
                              ("R3 scale: " + std::to_string(examples) +
                               " embedded examples round-trip byte-exact")
                                  .c_str());
            failures += stream_equals_one_shot("R3 scale streaming", scale_text, false, false);
        }
        std::string triples = "<function=write>\n";
        for (int i = 0; i < 2000; ++i) {
            triples += "<parameter=p>\nv\n</parameter>\n</parameter>\n</parameter>\n";
        }
        const auto triples_parsed = fi::parse_qwen_tool_call_output(triples, 64, c);
        failures += as_text("R3 scale: 2000 closer triples stay text", triples_parsed, triples,
                            Reason::MalformedStructure);
        failures += stream_equals_one_shot("R3 closer triples streaming", triples, false, false);
    }
    {
        // TAIL: the value itself ends with the closer literal (LF and CRLF continuations).
        for (const char* line : {"\n", "\r\n"}) {
            const std::string value = std::string("x") + line + "</parameter>";
            const std::string text = tool_call("write", {{"path", "d.md"}, {"content", value}});
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, c);
            failures += one_call(line == "\n" ? "R3 TAIL LF" : "R3 TAIL CRLF", parsed, "write",
                                 std::string("{\"path\":\"d.md\",\"content\":\"") +
                                     json_escape(value) + "\"}");
        }
    }
    // R2: the first region is prose before the real turn; the retry finds the real one.
    {
        const std::string r2_text = "Example:\n" + Ex + "\nNow writing.\n" +
                                   tool_call("write", {{"path", "d.md"}, {"content", "Doc\n" + Ex + "\nmore"}});
        const std::string r2_args =
            "{\"path\":\"d.md\",\"content\":\"" + json_escape("Doc\n" + Ex + "\nmore") + "\"}";
        const auto r2 = fi::parse_qwen_tool_call_output(r2_text, 64, c);
        failures += one_call("R3 R2: the real turn after a quoted example", r2, "write", r2_args);
    }
    // R6: a complete call plus a trailing closer echo is text.
    {
        const std::string r6_text = tool_call("bash", {{"command", "echo real"}}) +
                                   "\nNote: calls end with </parameter>\n</function>\n</tool_call>";
        const auto r6 = fi::parse_qwen_tool_call_output(r6_text, 64, c);
        failures += as_text("R3 R6: the trailing closer echo keeps the region text",
                            r6, r6_text, Reason::TrailingContent);
    }
    // R3-04: the natural-stop retry re-reads the quoted example; the cut never does.
    {
        const std::string p_a_text =
            "<function=write>\n<parameter=path>\nd.md\n</parameter>\n<parameter=content>\nExample:\n" + E;
        const std::string p_a_args = "{\"command\":\"rm -rf x\"}";
        for (const bool tolerant : {false, true}) {
            const auto st = fi::parse_qwen_tool_call_output(p_a_text, 64, c, tolerant,
                                                            ninfer::FinishReason::StopToken);
            failures += one_call(tolerant ? "R3 P-A StopToken tolerant: the example call"
                                          : "R3 P-A StopToken: the example call",
                                 st, "bash", p_a_args);
            const auto ol = fi::parse_qwen_tool_call_output(p_a_text, 64, c, tolerant,
                                                            ninfer::FinishReason::OutputLimit);
            failures += as_text(tolerant ? "R3 P-A OutputLimit tolerant: text" : "R3 P-A OutputLimit: text",
                                ol, p_a_text,
                                tolerant ? Reason::TruncatedTail : Reason::MalformedStructure);
        }
    }
    // P-A3: an embedded example inside a closed value never executes.
    {
        const std::string p_a3_text = "<function=write>\n<parameter=content>\nEx:\n" + E +
                                      "\nDone.\n</parameter>\n";
        for (const bool tolerant : {false, true}) {
            for (const auto reason : {ninfer::FinishReason::StopToken, ninfer::FinishReason::OutputLimit}) {
                const auto parsed = fi::parse_qwen_tool_call_output(p_a3_text, 64, c, tolerant, reason);
                failures += as_text("R3 P-A3: the closed value keeps the example inert",
                                    parsed, p_a3_text,
                                    tolerant ? Reason::TruncatedTail : Reason::MalformedStructure);
            }
        }
    }
    // R3-01 section 7.1: the S1 markup payload family (byte-exact content).
    const std::string EX = tool_call("read", {{"path", "foo.cpp"}});
    {
        const std::string s1 = tool_call("write", {{"path", "docs/x.md"},
                                                   {"content", "# Example\n" + EX + "\nDone."}});
        const auto parsed = fi::parse_qwen_tool_call_output(s1, 64, c);
        failures += one_call("R3 S1: the embedded example is the exact content", parsed, "write",
                             std::string("{\"path\":\"docs/x.md\",\"content\":\"") +
                                 json_escape("# Example\n" + EX + "\nDone.") + "\"}");
        failures += stream_equals_one_shot("R3 S1 streaming", s1, false, true);
        failures += stream_equals_one_shot("R3 S1 streaming tolerant", s1, true, true);
    }
    {
        const std::string s1c = tool_call("write", {{"path", "docs/x.md"},
                                                    {"content", "Example:\n" + EX}});
        const auto parsed = fi::parse_qwen_tool_call_output(s1c, 64, c);
        failures += one_call("R3 S1c: an example ending the value", parsed, "write",
                             std::string("{\"path\":\"docs/x.md\",\"content\":\"") +
                                 json_escape("Example:\n" + EX) + "\"}");
        failures += stream_equals_one_shot("R3 S1c streaming", s1c, false, true);
    }
    {
        const std::string s1d = tool_call("write", {{"path", "docs/x.md"},
                                                    {"content", "Text\n```xml\n" + EX + "\n```\nMore text"}});
        const auto parsed = fi::parse_qwen_tool_call_output(s1d, 64, c);
        failures += one_call("R3 S1d: the fenced example is content", parsed, "write",
                             std::string("{\"path\":\"docs/x.md\",\"content\":\"") +
                                 json_escape("Text\n```xml\n" + EX + "\n```\nMore text") + "\"}");
        failures += stream_equals_one_shot("R3 S1d streaming", s1d, false, true);
    }
    {
        const std::string old = std::string("R\"(") + EX + ")\"";
        const std::string s1e = tool_call("edit", {{"path", "d.md"}, {"old_string", old}});
        const auto parsed = fi::parse_qwen_tool_call_output(s1e, 64, c);
        failures += one_call("R3 S1e: the raw-string literal example is content", parsed, "edit",
                             std::string("{\"path\":\"d.md\",\"old_string\":\"") +
                                 json_escape(old) + "\"}");
        failures += stream_equals_one_shot("R3 S1e streaming", s1e, false, true);
    }
    {
        const std::string value = "Close with:\n</parameter>\n</function>\n</tool_call>\nthen stop.";
        const std::string s1f = tool_call("write", {{"path", "d.md"}, {"content", value}});
        const auto parsed = fi::parse_qwen_tool_call_output(s1f, 64, c);
        failures += one_call("R3 S1f: the closer lines inside the value are content", parsed, "write",
                             std::string("{\"path\":\"d.md\",\"content\":\"") +
                                 json_escape(value) + "\"}");
        const auto tol = fi::parse_qwen_tool_call_output(s1f, 64, c, true);
        failures += one_call("R3 S1f tolerant: the closer lines stay content", tol, "write",
                             std::string("{\"path\":\"d.md\",\"content\":\"") +
                                 json_escape(value) + "\"}");
        failures += stream_equals_one_shot("R3 S1f streaming", s1f, false, true);
    }
    {
        const std::string s1g = tool_call("write", {{"path", "d.md"},
                                                    {"content", tool_call("edit", {{"path", "a"}, {"old_string", "b"}})}});
        const auto parsed = fi::parse_qwen_tool_call_output(s1g, 64, c);
        failures += one_call("R3 S1g: the complete inner call is content", parsed, "write",
                             std::string("{\"path\":\"d.md\",\"content\":\"") +
                                 json_escape(tool_call("edit", {{"path", "a"}, {"old_string", "b"}})) + "\"}");
        failures += stream_equals_one_shot("R3 S1g streaming", s1g, false, true);
    }
    // S1-short: the R3-01 class in the short parameter family. The Stage-2 work-bound skip
    // must check both closer families: a short-only region carries no '</parameter>' literal,
    // and skipping the consistent completion there rejects a region with a unique consistent
    // parse (before the fix: text, TrailingContent).
    {
        const std::string ex_short =
            "<function=bash>\n<param=command>\nrm -rf x\n</param>\n</function>\n</tool_call>";
        const std::string s1_short =
            std::string("<tool_call>\n<function=write>\n<param=path>\ndocs/x.md\n</param>\n<param=content>\n") +
            "# Example\n" + ex_short + "\nDone.\n</param>\n</function>\n</tool_call>";
        const std::string content = "# Example\n" + ex_short + "\nDone.";
        const auto parsed = fi::parse_qwen_tool_call_output(s1_short, 64, c);
        failures += one_call("R3 S1-short: the short-family embedded example is the exact content",
                             parsed, "write",
                             std::string("{\"path\":\"docs/x.md\",\"content\":\"") +
                                 json_escape(content) + "\"}");
        failures += check(parsed.diagnostics.markup_tolerant_completion,
                          "R3 S1-short: the Stage-2 acceptance flag is set");
        failures += stream_equals_one_shot("R3 S1-short streaming", s1_short, false, true);
    }

    // S8: the here-doc that carries the closer lines commits in all modes (the real closer
    // terminates the value); the cut variant after the final EOF line is the R3-03 residual.
    const std::string s8_value = "cat <<EOF\n</parameter>\n</function>\n</tool_call>\nEOF";
    const std::string s8 = tool_call("bash", {{"command", s8_value}});
    const std::string s8_args = std::string("{\"command\":\"") + json_escape(s8_value) + "\"}";
    {
        const auto parsed = fi::parse_qwen_tool_call_output(s8, 64, c);
        failures += one_call("R3 S8: the here-doc closer lines are the exact command", parsed,
                             "bash", s8_args);
        const auto tol = fi::parse_qwen_tool_call_output(s8, 64, c, true);
        failures += one_call("R3 S8 tolerant: the here-doc commits", tol, "bash", s8_args);
        failures += stream_equals_one_shot("R3 S8 streaming", s8, false, true);
    }
    {
        // Cut after the final EOF line (before the real closer): the natural stop commits
        // the truncated command (documented residual, R3-03); a cut reason commits nothing.
        const std::string s8_cut = "<tool_call>\n<function=bash>\n<parameter=command>\ncat <<EOF\n"
                                   "</parameter>\n</function>\n</tool_call>\nEOF";
        const auto strict = fi::parse_qwen_tool_call_output(s8_cut, 64, c);
        failures += as_text("R3 S8-cut strict: text", strict, s8_cut, Reason::TrailingContent);
        const auto tol_stop = fi::parse_qwen_tool_call_output(s8_cut, 64, c, true,
                                                              ninfer::FinishReason::StopToken);
        failures += check(tol_stop.is_tool_call_response && tol_stop.tool_calls.size() == 1 &&
                              tol_stop.tool_calls.front().name == "bash" &&
                              tol_stop.tool_calls.front().arguments_json ==
                                  "{\"command\":\"cat <<EOF\"}" &&
                              tol_stop.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "R3 S8-cut tolerant StopToken: the truncated command commits (residual)");
        const auto tol_limit = fi::parse_qwen_tool_call_output(s8_cut, 64, c, true,
                                                               ninfer::FinishReason::OutputLimit);
        failures += as_text("R3 S8-cut tolerant OutputLimit: no commit", tol_limit, s8_cut,
                            Reason::TrailingContent);
        failures += stream_equals_one_shot("R3 S8-cut streaming", s8_cut, true, true);
    }
    // P-C: two unfenced examples inside one value commit in strict and tolerant alike.
    {
        const std::string value = "Ex:\n" + E + "\n" + E + "\nDone.";
        const std::string p_c = tool_call("write", {{"path", "d.md"}, {"content", value}});
        const std::string p_c_args = std::string("{\"path\":\"d.md\",\"content\":\"") +
                                     json_escape(value) + "\"}";
        const auto parsed = fi::parse_qwen_tool_call_output(p_c, 64, c);
        failures += one_call("R3 P-C: both examples are the exact content", parsed, "write",
                             p_c_args);
        const auto tol = fi::parse_qwen_tool_call_output(p_c, 64, c, true);
        failures += one_call("R3 P-C tolerant: both examples are the exact content", tol,
                             "write", p_c_args);
        failures += stream_equals_one_shot("R3 P-C streaming", p_c, false, true);
    }
    // R5: two fenced examples inside one value commit (byte-exact).
    {
        const std::string value = "A:\n```\n" + Ex + "\n```\nB:\n```\n" + Ex + "\n```\nEnd.";
        const std::string r5 = tool_call("write", {{"path", "d.md"}, {"content", value}});
        const auto parsed = fi::parse_qwen_tool_call_output(r5, 64, c);
        failures += one_call("R3 R5: the two fenced examples are the exact content", parsed,
                             "write",
                             std::string("{\"path\":\"d.md\",\"content\":\"") +
                                 json_escape(value) + "\"}");
        failures += stream_equals_one_shot("R3 R5 streaming", r5, false, true);
    }
    // R1-inline (inline closer note): the example is followed by a note line that carries an
    // inline </parameter> (not a standalone closer line). The canonical-framing rule rejects
    // an inline closer, so this is not the Round-3 section-9 phantom acceptance (see
    // test_round4_r1_residual_pinned for the real R1 residual); the observed verdict is pinned
    // so a future change is intentional.
    {
        const std::string r1 = "Example:\n" + tool_call("read", {{"path", "ex.txt"}}) +
                               "\nNote: </parameter>\n</function>\n</tool_call>";
        const auto strict = fi::parse_qwen_tool_call_output(r1, 64, c);
        failures += as_text("R3 R1-inline strict: the inline closer note keeps the region text", strict,
                            r1, Reason::TrailingContent);
        const auto tol_stop = fi::parse_qwen_tool_call_output(r1, 64, c, true,
                                                              ninfer::FinishReason::StopToken);
        failures += as_text("R3 R1-inline tolerant StopToken: the value closer in the tail blocks the commit",
                            tol_stop, r1, Reason::TrailingContent);
        failures += stream_equals_one_shot("R3 R1 streaming", r1, true, true);
    }
    // S2b: the Round-2 baseline quoted-fixture fixture (R3-02).
    {
        const std::string s2b =
            "explaining <tool_call>\\n<function=shell>\\n<function=command>\\nprintf broken\\n"
            "</parameter>\\n</function>\\n</tool_call> then the real turn\\n" +
            tool_call("bash", {{"command", "echo ok"}});
        const auto parsed = fi::parse_qwen_tool_call_output(s2b, 64, c);
        failures += one_call("R3 S2b: the baseline quoted fixture does not demote the real turn",
                             parsed, "bash", "{\"command\":\"echo ok\"}");
        failures += stream_equals_one_shot("R3 S2b streaming", s2b, false, true);
    }
    // Q1/Q2 (probe4): an attribute-form marker cut inside a quoted name rescan-continues
    // to the real call (R3-05).
    {
        const std::string q1 = std::string("Attr form: <function name=\"x\n") +
                               tool_call("bash", {{"command", "ls"}});
        const auto parsed = fi::parse_qwen_tool_call_output(q1, 64, c);
        failures += one_call("R3 Q1: the cut quoted attribute does not latch", parsed, "bash",
                             "{\"command\":\"ls\"}");
        failures += stream_equals_one_shot("R3 Q1 streaming", q1, false, true);
        const std::string q2 = std::string("The <invoke name='it\n") +
                               tool_call("bash", {{"command", "ls"}});
        const auto parsed2 = fi::parse_qwen_tool_call_output(q2, 64, c);
        failures += one_call("R3 Q2: the cut single-quoted attribute does not latch", parsed2,
                             "bash", "{\"command\":\"ls\"}");
        failures += stream_equals_one_shot("R3 Q2 streaming", q2, false, true);
    }
    // P-A2: a bare write with a path and a closed content value containing the example
    // commits with the exact content (R3-04/R3-01: the consistent parse resolves the
    // value boundary, so the cut is immaterial).
    {
        const std::string p_a2 =
            "<function=write>\n<parameter=path>\nd.md\n</parameter>\n<parameter=content>\nEx:\n" +
            E + "\nDone.\n</parameter>\n</function>\n";
        const std::string p_a2_args =
            std::string("{\"path\":\"d.md\",\"content\":\"") +
            json_escape("Ex:\n" + E + "\nDone.") + "\"}";
        for (const bool tolerant : {false, true}) {
            for (const auto reason : {ninfer::FinishReason::StopToken, ninfer::FinishReason::OutputLimit}) {
                const auto parsed = fi::parse_qwen_tool_call_output(p_a2, 64, c, tolerant, reason);
                failures += one_call("R3 P-A2: the closed value with the example commits",
                                     parsed, "write", p_a2_args);
            }
        }
        failures += stream_equals_one_shot("R3 P-A2 streaming", p_a2, false, true);
    }
    // F1: a CRLF-fenced example never latches; the call after the fence commits (R3-06).
    {
        const std::string f1 = std::string("```xml\r\n") + E + "\r\n```\r\n" +
                               tool_call("read", {{"path", "a"}});
        const auto parsed = fi::parse_qwen_tool_call_output(f1, 64, c);
        failures += one_call("R3 F1: the CRLF fence does not latch", parsed, "read",
                             "{\"path\":\"a\"}");
        failures += stream_equals_one_shot("R3 F1 streaming", f1, false, true);
    }
    // Fence-aware retry: a bare opener before a fenced marker commits nothing (R3-06/R3-04):
    // the retry skips the fenced marker, so no later entry exists to commit.
    {
        const std::string faretry = "Use <function=read> x\n```xml\n" + EX + "\n```\n";
        const auto strict = fi::parse_qwen_tool_call_output(faretry, 64, c);
        failures += as_text("R3 fence retry strict: the fenced marker is not re-read", strict,
                            faretry, Reason::MalformedStructure);
        const auto tol = fi::parse_qwen_tool_call_output(faretry, 64, c, true);
        failures += as_text("R3 fence retry tolerant: the fenced marker is not re-read", tol,
                            faretry, Reason::MalformedStructure);
        failures += stream_equals_one_shot("R3 fence retry streaming", faretry, true, true);
    }
    // Trailing prose carrying a value closer in the tail commits nothing, even at a
    // natural stop (R3-03 item 3: the tail may be the last value's remainder). A stray
    // wrapper closer keeps the old tolerant commit (the stray-closer test above pins it).
    {
        const std::string tailc = tool_call("read", {{"path", "a"}}) + "\nSee </parameter> note";
        const auto strict = fi::parse_qwen_tool_call_output(tailc, 64, c);
        failures += as_text("R3 tail value-closer strict: text", strict, tailc,
                            Reason::TrailingContent);
        const auto tol_stop = fi::parse_qwen_tool_call_output(tailc, 64, c, true,
                                                              ninfer::FinishReason::StopToken);
        failures += as_text(
            "R3 tail value-closer tolerant StopToken: no commit despite the natural stop",
            tol_stop, tailc, Reason::TrailingContent);
        const auto tol_limit = fi::parse_qwen_tool_call_output(tailc, 64, c, true,
                                                               ninfer::FinishReason::OutputLimit);
        failures += as_text("R3 tail value-closer tolerant OutputLimit: no commit", tol_limit,
                            tailc, Reason::TrailingContent);
        failures += stream_equals_one_shot("R3 tail value-closer streaming", tailc, true, true);
    }
    // R3-04 section 7.5: the F7 bare_before fixture with a cut finish reason is rejected
    // (the open value's payload after a cut is not committable).
    {
        const std::string bare_before = std::string("<function=read>\n<parameter=path>\ncut\n") +
                                        tool_call("bash", {{"command", "echo ok"}});
        const auto parsed = fi::parse_qwen_tool_call_output(bare_before, 64, c, true,
                                                            ninfer::FinishReason::OutputLimit);
        failures += as_text("R3 F7 cut: bare_before with a cut reason is rejected", parsed,
                            bare_before, Reason::TruncatedTail);
    }
    // R3-08: synthetic arguments make a declared-tool call ambiguous; the legacy contract
    // keeps last-value semantics.
    {
        const std::string e2_text = "<tool_call>\n<function=write>\n<parameter=path>\nd.md\n</parameter>\n"
                                    "<parameter=content>\nIntro\n</parameter>\n<parameter=content>\ntail\n</parameter>\n"
                                    "</function>\n</tool_call>";
        for (const bool tolerant : {false, true}) {
            const auto e2 = fi::parse_qwen_tool_call_output(e2_text, 64, c, tolerant);
            failures += as_text(tolerant ? "R3 E2 tolerant: duplicate name is ambiguous"
                                         : "R3 E2: duplicate name is ambiguous",
                                e2, e2_text, Reason::AmbiguousStructure);
        }
        const auto e2_legacy = fi::parse_qwen_tool_call_output(e2_text, 64, kLegacyContract);
        failures += one_call("R3 E2 legacy: the legacy contract keeps last value",
                             e2_legacy, "write", "{\"path\":\"d.md\",\"content\":\"tail\"}");
        failures += check(e2_legacy.diagnostics.duplicate_parameters_repaired == 1,
                          "R3 E2 legacy: the duplicate is counted");
        const std::string e1_text = "<tool_call>\n<function=edit>\n<parameter=path>\nd.md\n</parameter>\n"
                                    "<parameter=old_string>\nx\n</parameter>\n<parameter=content>\nhello\n</parameter>\n"
                                    "<parameter=new_string>\nNEW\n</parameter>\n</function>\n</tool_call>";
        const auto e1 = fi::parse_qwen_tool_call_output(e1_text, 64, c);
        failures += as_text("R3 E1: a non-declared name outside the first parameter is ambiguous",
                            e1, e1_text, Reason::AmbiguousStructure);
        const std::string e3_text = "<tool_call>\n<function=edit>\n<parameter=path>\nd.md\n</parameter>\n"
                                    "<parameter=old_string>\nx\n</parameter>\n<parameter=mode>\nhello\n</parameter>\n"
                                    "<parameter=new_string>\nNEW\n</parameter>\n</function>\n</tool_call>";
        const auto e3 = fi::parse_qwen_tool_call_output(e3_text, 64, c);
        failures += as_text("R3 E3: a placeholder name instead of the declared one is ambiguous",
                            e3, e3_text, Reason::AmbiguousStructure);
    }
    // R3-05: a quoted '>' does not break the header; the '/' stays part of the parameter name.
    {
        const std::string s5b_text = "<tool_call>\n<function=search>\n<parameter=a/b>\nx\n</parameter>\n"
                                     "</function>\n</tool_call>";
        const auto s5b = fi::parse_qwen_tool_call_output(s5b_text, 64, c);
        failures += one_call("R3 S5b: the '/' keeps the declared parameter name",
                             s5b, "search", "{\"a/b\":\"x\"}");
        const std::string p_b_text =
            tool_call("bash", {{"command", "grep '</parameter><parameter name=\"' file"}});
        const auto p_b = fi::parse_qwen_tool_call_output(p_b_text, 64, c);
        failures += one_call("R3 P-B: the echoed markup stays the exact command",
                             p_b, "bash",
                             "{\"command\":\"grep '</parameter><parameter name=\\\"' file\"}");
    }
    // R3-09: an output without any marker records the reason none.
    {
        const auto s4 = fi::parse_qwen_tool_call_output("Just an answer.", 64, c);
        failures += as_text("R3 S4: no marker records the reason none", s4, "Just an answer.",
                            Reason::None);
    }
    // R3-02: a quoted broken wrapper no longer hides the real turn.
    {
        const std::string s2_text = "I will emit a <tool_call> block now.\n" +
                                   tool_call("write", {{"path", "d.md"}, {"content", "Doc"}});
        const auto s2 = fi::parse_qwen_tool_call_output(s2_text, 64, c);
        failures += one_call("R3 S2: the real call after quoted prose", s2, "write",
                             "{\"path\":\"d.md\",\"content\":\"Doc\"}");
        const std::string p_d_text = "For example:\n" + tool_call("read", {{"path", "ex.txt"}}) +
                                     "\nNow the real one.\n" +
                                     tool_call("bash", {{"command", "echo real"}});
        for (const bool tolerant : {false, true}) {
            const auto p_d = fi::parse_qwen_tool_call_output(p_d_text, 64, c, tolerant);
            failures += one_call(tolerant ? "R3 P-D tolerant: the later complete call" : "R3 P-D strict: the later complete call",
                                 p_d, "bash", "{\"command\":\"echo real\"}");
        }
    }
    // R3-06: fence variants that must not suppress the following real marker.
    {
        const std::string f2_text = "```x``` is inline code\n" +
                                   tool_call("write", {{"path", "d.md"}, {"content", "Doc"}});
        const auto f2 = fi::parse_qwen_tool_call_output(f2_text, 64, c);
        failures += one_call("R3 F2: inline backticks are not a fence", f2, "write",
                             "{\"path\":\"d.md\",\"content\":\"Doc\"}");
        const std::string f3_text = "- step:\n  ```bash\n  make\n    ```\nNow calling.\n" +
                                   tool_call("write", {{"path", "d.md"}, {"content", "Doc"}});
        const auto f3 = fi::parse_qwen_tool_call_output(f3_text, 64, c);
        failures += one_call("R3 F3: a list-indented closed fence releases the marker",
                             f3, "write", "{\"path\":\"d.md\",\"content\":\"Doc\"}");
        const std::string s3_text = "Here is code:\n```python\nprint(1)\n\n" +
                                   tool_call("write", {{"path", "d.md"}, {"content", "Doc"}});
        const auto s3 = fi::parse_qwen_tool_call_output(s3_text, 64, c);
        failures += as_text("R3 S3: an unclosed fence keeps the region text",
                            s3, s3_text, Reason::None);
        failures += check(s3.diagnostics.ended_in_unclosed_fence,
                          "R3 S3: the unclosed fence is diagnosed");
        failures += check(s3.diagnostics.fenced_markers_suppressed == 2,
                          "R3 S3: the fenced markers are counted as suppressed");
    }
    return failures;
}

// R3-07: streaming equivalence fuzz over the §10.3 fragment corpus — one-shot parsing and
// streamed parsing (full feed and byte-at-a-time) must agree on every text.
int test_round3_streaming_equivalence_fuzz() {
    using ninfer::FinishReason;
    // Round-3 section-10.3 fragment corpus; the same tail rule as before.
    const std::vector<std::string> frags = {
        "<tool_call>\n", "</tool_call>\n", "<function=write>\n", "<function=bash>\n", "</function>\n",
        "<parameter=path>\n", "<parameter=content>\n", "<parameter=command>\n", "</parameter>\n",
        "text ", "x\n", "```\n", "```xml\n", "~~~\n", "  ```\n", "<", ">", "\"", "'", "\r\n",
        "<function name=\"", "<invoke=bash>", "</invoke>", "<param=command>", "</param>",
        "<function_calls>\n", "</function_calls>\n", "echo '</parameter>'\n", "<tool_c", "<function",
        "prose. ", "\n"};
    const std::string tail =
        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";
    const std::vector<std::string> defs = {
        tool_definition("write",
                        Json{{"path", Json{{"type", "string"}}},
                             {"content", Json{{"type", "string"}}}}),
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})};
    const auto contract = contract_from_definitions(defs);
    const fi::ToolCallOutputContract& c = *contract;
    std::mt19937 rng(20260930);
    int failures = 0;
    // Compare the streamed result against the one-shot exactly (no rtrim): the content, the
    // calls, and the full diagnostics must be identical.
    auto stream_matches = [&](const char* partition, const std::string& visible,
                              const fi::ToolCallOutputDecoder::Terminal& term,
                              const fi::ParsedToolCallOutput& one, int t, bool tolerant,
                              FinishReason reason) {
        const std::string total = visible + term.content;
        bool match = total == one.content && one.tool_calls.size() == term.tool_calls.size() &&
                     term.diagnostics == one.diagnostics;
        std::string first;
        for (std::size_t i = 0; match && i < one.tool_calls.size(); ++i) {
            match = one.tool_calls[i].name == term.tool_calls[i].name &&
                    one.tool_calls[i].arguments_json == term.tool_calls[i].arguments_json;
            if (!match) { first = "call " + std::to_string(i); }
        }
        if (!match) {
            std::cout << "FAIL fuzz " << t << " " << partition << " tolerant=" << (int)tolerant
                      << " reason=" << (reason == ninfer::FinishReason::StopToken ? "stop" : "limit")
                      << (first.empty() ? "" : " (mismatch at " + first + ")") << std::endl;
        }
        return match ? 0 : 1;
    };
    for (int t = 0; t < 2000; ++t) {
        std::string text;
        const int count = 2 + int(rng() % 14);
        for (int i = 0; i < count; ++i) { text += frags[rng() % frags.size()]; }
        if (rng() % 3 == 0) { text += tail; }
        for (const bool tolerant : {false, true}) {
            for (const FinishReason reason : {ninfer::FinishReason::StopToken, ninfer::FinishReason::OutputLimit}) {
                const auto one = fi::parse_qwen_tool_call_output(text, 64, c, tolerant, reason);
                // whole text
                {
                    fi::ToolCallOutputDecoder dec(contract, 64, tolerant);
                    const std::string visible = dec.feed(text);
                    const auto term = dec.finish(reason);
                    failures += stream_matches("whole", visible, term, one, t, tolerant, reason);
                }
                // byte-wise
                {
                    fi::ToolCallOutputDecoder dec(contract, 64, tolerant);
                    std::string visible;
                    for (std::size_t i = 0; i < text.size(); ++i) {
                        visible += dec.feed(text.substr(i, 1));
                    }
                    const auto term = dec.finish(reason);
                    failures += stream_matches("byte", visible, term, one, t, tolerant, reason);
                }
                // random chunk sizes 1..7 seeded with the text index
                {
                    std::mt19937 chunk_rng(t);
                    fi::ToolCallOutputDecoder dec(contract, 64, tolerant);
                    std::string visible;
                    std::size_t offset = 0;
                    while (offset < text.size()) {
                        const std::size_t n =
                            std::min<std::size_t>(1 + chunk_rng() % 7, text.size() - offset);
                        visible += dec.feed(text.substr(offset, n));
                        offset += n;
                    }
                    const auto term = dec.finish(reason);
                    failures += stream_matches("chunk", visible, term, one, t, tolerant, reason);
                }
            }
        }
    }
    return failures;
}

// R3-14: deterministic work bounds (step counters, not wall-clock): 4000 bare open regions
// leave Stage 2 uncharged; a 40 KB unterminated quoted header bounds the pre-latch rescan;
// 2000 closer triples bound the Stage-2 steps.
int test_round3_work_bounds() {
    using ninfer::FinishReason;
    fi::ToolCallParsePolicy policy;
    policy.max_name_length = 64;
    int failures = 0;
    {
        std::string text;
        text.reserve(4000 * 34);
        for (int i = 0; i < 4000; ++i) { text += "<function=write>\n<parameter=content>\n"; }
        fi::ToolCallStreamParser machine(policy);
        machine.feed(std::string_view(text));
        const auto term = machine.finish(ninfer::FinishReason::StopToken);
        failures += check(term.status == fi::ToolCallStreamStatus::Invalid && term.stage2_steps == 0,
                          "R3-14: 4000 bare open regions reject with Stage 2 uncharged");
    }
    {
        std::string header = "<parameter name=\"";
        header.append(40 * 1024, 'x');
        fi::ToolCallStreamParser machine(policy);
        machine.feed(std::string_view(header));
        const auto term = machine.finish(ninfer::FinishReason::OutputLimit);
        failures += check(term.rescan_steps < 512,
                          "R3-14: a 40 KB unterminated quoted header bounds the rescan");
    }
    {
        std::string text = "<function=write>\n";
        for (int i = 0; i < 2000; ++i) {
            text += "<parameter=p>\nv\n</parameter>\n</parameter>\n</parameter>\n";
        }
        fi::ToolCallStreamParser machine(policy);
        machine.feed(std::string_view(text));
        const auto term = machine.finish(ninfer::FinishReason::OutputLimit);
        failures += check(term.stage2_steps <= std::max<std::uint64_t>(100000, 4 * text.size()) &&
                              !term.parse_budget_exhausted,
                          "R4 N-01/N-07: 2000 closer triples bound the Stage-2 work units");
    }
    {
        // An exhausted Stage-2 step budget fails closed to Invalid (section 7.11): the
        // region is rejected instead of running unbounded consistent parses. S1 needs 14
        // steps; a budget of 8 cannot finish it.
        const std::string s1 = tool_call("write", {{"path", "docs/x.md"},
                                                   {"content", "# Example\n" +
                                                                  tool_call("read", {{"path", "foo.cpp"}}) +
                                                                  "\nDone."}});
        fi::ToolCallParsePolicy small;
        small.max_name_length = 64;
        small.stage2_step_budget = 8;
        fi::ToolCallStreamParser machine(small);
        machine.feed(std::string_view(s1));
        const auto term = machine.finish(ninfer::FinishReason::StopToken);
        failures += check(term.status == fi::ToolCallStreamStatus::Invalid &&
                              term.region.calls.empty() && term.parse_budget_exhausted,
                          "R4 N-07: an exhausted Stage-2 budget fails closed");
    }
    return failures;
}

// N-04: there is no depth or chain bound on the number of values on one Stage-2 path. A
// call with 70 parameters whose last value needs Stage 2 completes with all 70 parameters;
// the removed depth constant of 64 would have rejected it.
int test_round4_stage2_many_values() {
    using ninfer::FinishReason;
    Json properties = Json::object();
    for (int i = 0; i < 70; ++i) { properties["p" + std::to_string(i)] = Json{{"type", "string"}}; }
    const auto contract = contract_for("multi", std::move(properties));
    std::string text = "<tool_call>\n<function=multi>\n";
    for (int i = 0; i < 69; ++i) {
        text += "<parameter=p" + std::to_string(i) + ">\nv" + std::to_string(i) + "\n</parameter>\n";
    }
    // The last value needs Stage 2: it embeds a complete tool call with its own closer, which
    // the greedy one-token rule mis-selects.
    text += "<parameter=p69>\n# Example\n" + tool_call("read", {{"path", "foo.cpp"}}) + "\nDone.\n</parameter>\n";
    text += "</function>\n</tool_call>";
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, false,
                                                        ninfer::FinishReason::StopToken);
    int ok = parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
             parsed.tool_calls[0].name == "multi" &&
             parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None;
    if (ok) {
        const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
        if (!args.is_object()) { ok = false; }
        else {
            for (int i = 0; i < 70 && ok; ++i) { ok = args.contains("p" + std::to_string(i)); }
        }
    }
    return check(ok, "N-04: a 70-parameter call whose last value needs Stage 2 completes");
}

// N-01, N-07: the global Stage-2 work budget bounds the deterministic work (stage2_steps is
// the charged unit count) and exhaustion fails closed with parse_budget_exhausted.
int test_round4_work_bounds() {
    using ninfer::FinishReason;
    using ninfer::ToolCallParseFallbackReason;
    fi::ToolCallParsePolicy policy;
    policy.max_name_length = 64;
    int failures = 0;
    const auto limit_of = [](const std::string& t) {
        return std::max<std::uint64_t>(100000, 4 * t.size());
    };
    {
        // Repetition with trailing prose: 300 complete calls plus prose after the last one.
        std::string text;
        for (int i = 0; i < 300; ++i) {
            text += "Example " + std::to_string(i) + ":\n" +
                    tool_call("bash", {{"command", "ls -la"}}) + "\n";
        }
        text += "That is all.";
        const std::uint64_t limit = limit_of(text);
        fi::ToolCallStreamParser machine(policy);
        machine.feed(std::string_view(text));
        const auto term = machine.finish(ninfer::FinishReason::StopToken);
        failures += check(term.status == fi::ToolCallStreamStatus::Invalid &&
                              term.stage2_steps <= limit + 1,
                          "N-01: repetition with trailing prose is Invalid within the work bound");
    }
    {
        // Closer triples inside one write value: Stage 2 resolves it; the work stays bounded
        // and the budget is not exhausted.
        std::string content;
        for (int i = 0; i < 2000; ++i) { content += "<parameter=p>\nv\n</parameter>\n</parameter>\n</parameter>\n"; }
        const std::string text = tool_call("write", {{"path", "d.md"}, {"content", content}});
        const std::uint64_t limit = limit_of(text);
        fi::ToolCallStreamParser machine(policy);
        machine.feed(std::string_view(text));
        const auto term = machine.finish(ninfer::FinishReason::StopToken);
        failures += check(term.status == fi::ToolCallStreamStatus::Complete &&
                              !term.parse_budget_exhausted && term.stage2_steps <= limit + 1,
                          "N-01: closer triples complete within the work bound without exhaustion");
    }
    {
        // Tiny budget: a Stage-2 budget of 8 cannot finish the Round-3 S1 example, so the
        // region fails closed with parse_budget_exhausted and no calls.
        const std::string s1 = tool_call("write", {{"path", "docs/x.md"},
                                                   {"content", "# Example\n" +
                                                                  tool_call("read", {{"path", "foo.cpp"}}) +
                                                                  "\nDone."}});
        fi::ToolCallParsePolicy small;
        small.max_name_length = 64;
        small.stage2_step_budget = 8;
        fi::ToolCallStreamParser machine(small);
        machine.feed(std::string_view(s1));
        const auto term = machine.finish(ninfer::FinishReason::StopToken);
        failures += check(term.status == fi::ToolCallStreamStatus::Invalid &&
                              term.region.calls.empty() && term.parse_budget_exhausted,
                          "N-07: a tiny Stage-2 budget fails closed with parse_budget_exhausted");
    }
    return failures;
}
// Section-7 streaming guard: the streamed decode (chunk sizes 1/2/3/5/7, every split for
// fixtures below 512 bytes) must equal the one-shot entry (content under the documented
// pre-region rtrim rule, calls, arguments, fallback reason).
int stream_matches_one_shot(const std::string& text,
                            const std::shared_ptr<const fi::ToolCallOutputContract>& contract,
                            bool tolerant, ninfer::FinishReason reason, const char* label) {
    const auto one = fi::parse_qwen_tool_call_output(text, 64, *contract, tolerant, reason);
    auto compare = [&](const std::vector<std::size_t>& points) -> bool {
        fi::ToolCallOutputDecoder dec(contract, 64, tolerant);
        std::string_view sv = text;
        std::size_t from = 0;
        std::string visible;
        for (const std::size_t pt : points) { visible += dec.feed(sv.substr(from, pt - from)); from = pt; }
        if (from < text.size()) { visible += dec.feed(sv.substr(from)); }
        const auto term = dec.finish(reason);
        std::string total = visible + term.content;
        if (one.is_tool_call_response) {
            while (!total.empty() && fi::is_tool_format_whitespace(total.back())) { total.pop_back(); }
        }
        bool same_calls = one.tool_calls.size() == term.tool_calls.size();
        for (std::size_t i = 0; same_calls && i < one.tool_calls.size(); ++i) {
            same_calls = one.tool_calls[i].name == term.tool_calls[i].name &&
                         one.tool_calls[i].arguments_json == term.tool_calls[i].arguments_json;
        }
        return total == one.content && same_calls &&
               one.diagnostics.fallback_reason == term.diagnostics.fallback_reason;
    };
    if (text.size() < 512) {
        for (std::size_t k = 0; k <= text.size(); ++k) {
            if (!compare({k})) { std::printf("fail %s: streaming split %zu diverges\n", label, k); return 1; }
        }
    } else {
        for (const int chunk : {1, 2, 3, 5, 7}) {
            std::vector<std::size_t> points;
            for (std::size_t k = 0; k < text.size(); k += std::size_t(chunk)) { points.push_back(k); }
            if (!compare(points)) { std::printf("fail %s: streaming chunk %d diverges\n", label, chunk); return 1; }
        }
    }
    return 0;
}

// N-02: a Stage-2 base whose values use the `<param>`/`</param>` family must not be skipped
// (the old skip tested only `</parameter>`). The reproducer completes via Stage 2.
int test_round4_param_family_stage2() {
    using ninfer::FinishReason;
    using ninfer::ToolCallParseFallbackReason;
    const std::string text =
        "<tool_call>\n<function=write>\n<param=path>\ndocs/x.md\n</param>\n<param=content>\n# Example\n"
        "<tool_call>\n<function=read>\n<param=path>\nfoo.cpp\n</param>\n</function>\n</tool_call>\nDone.\n"
        "</param>\n</function>\n</tool_call>";
    int failures = 0;
    Json write_props = Json::object();
    write_props["path"]    = Json{{"type", "string"}};
    write_props["content"] = Json{{"type", "string"}};
    Json read_props = Json::object();
    read_props["path"] = Json{{"type", "string"}};
    const std::vector<std::string> definitions = {tool_definition("write", write_props),
                                                  tool_definition("read", read_props)};
    const auto shared = contract_from_definitions(definitions);
    for (const bool tolerant : {false, true}) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, *shared, tolerant,
                                                            ninfer::FinishReason::StopToken);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls[0].name == "write" &&
                              parsed.diagnostics.markup_tolerant_completion &&
                              parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
                          "N-02: a <param>-family write completes via Stage 2");
    }
    failures += stream_matches_one_shot(text, shared, false, ninfer::FinishReason::StopToken, "N-02 streaming");
    return failures;
}

// N-05: the balance census counts only same-family openers. A cross-family `<param>` opener
// inside a `<parameter>` value is ordinary bytes, not a nesting opener.
int test_round4_same_family_balance() {
    using ninfer::FinishReason;
    using ninfer::ToolCallParseFallbackReason;
    Json write_props = Json::object();
    write_props["path"]    = Json{{"type", "string"}};
    write_props["content"] = Json{{"type", "string"}};
    Json bash_props = Json::object();
    bash_props["command"] = Json{{"type", "string"}};
    const std::vector<std::string> definitions = {tool_definition("write", write_props),
                                                  tool_definition("bash", bash_props)};
    const auto shared = contract_from_definitions(definitions);
    const std::string ex      = tool_call("bash", {{"command", "ls"}});
    const std::string content = "Doc <param=x> marker\n" + ex + "\nmore";
    const std::string text    = "Example:\n" + ex + "\nNow writing.\n" +
                                tool_call("write", {{"path", "d.md"}, {"content", content}});
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, *shared, false,
                                                        ninfer::FinishReason::StopToken);
    int failures = check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                             parsed.tool_calls[0].name == "write" &&
                             parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
                         "N-05: a cross-family <param> opener stays content (write completes)");
    if (parsed.is_tool_call_response && parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
        failures += check(args["content"].get<std::string>() == content,
                          "N-05: the write content is byte-exact (cross-family opener is content)");
    }
    failures += stream_matches_one_shot(text, shared, false, ninfer::FinishReason::StopToken, "N-05 streaming");
    return failures;
}

int test_round4_fence_diagnostics_scope() {
    using ninfer::FinishReason;
    using ninfer::ToolCallParseFallbackReason;
    Json write_props = Json::object();
    write_props["path"]    = Json{{"type", "string"}};
    write_props["content"] = Json{{"type", "string"}};
    Json read_props = Json::object();
    read_props["path"] = Json{{"type", "string"}};
    Json bash_props = Json::object();
    bash_props["command"] = Json{{"type", "string"}};
    const std::vector<std::string> definitions = {tool_definition("write", write_props),
                                                  tool_definition("read", read_props),
                                                  tool_definition("bash", bash_props)};
    const auto shared = contract_from_definitions(definitions);
    int failures = 0;
    // N6d: a closed fence before a call -> accepted, both fence fields 0/false.
    {
        const std::string text = "Run this:\n```bash\nls\n```\n" + tool_call("bash", {{"command", "ls"}});
        const auto o = fi::parse_qwen_tool_call_output(text, 64, *shared, false, ninfer::FinishReason::StopToken);
        failures += check(o.is_tool_call_response && o.tool_calls.size() == 1 &&
                              o.tool_calls[0].name == "bash" &&
                              o.diagnostics.fenced_markers_suppressed == 0 &&
                              !o.diagnostics.ended_in_unclosed_fence,
                          "N6d: a closed fence before a call is accepted, fence 0/false");
        failures += stream_matches_one_shot(text, shared, false, ninfer::FinishReason::StopToken, "N6d streaming");
    }
    // N6a: a write content with a python fence -> accepted, fence 0/false.
    {
        const std::string text = tool_call("write", {{"path", "a.md"}, {"content", "```python\nprint(1)"}});
        const auto o = fi::parse_qwen_tool_call_output(text, 64, *shared, false, ninfer::FinishReason::StopToken);
        failures += check(o.is_tool_call_response && o.tool_calls.size() == 1 &&
                              o.tool_calls[0].name == "write" &&
                              o.diagnostics.fenced_markers_suppressed == 0 &&
                              !o.diagnostics.ended_in_unclosed_fence,
                          "N6a: an accepted write with a python fence, fence 0/false");
        failures += stream_matches_one_shot(text, shared, false, ninfer::FinishReason::StopToken, "N6a streaming");
    }
    // N6b: a write content with a fenced read example -> accepted, fence 0/false.
    {
        const std::string content = "Ex:\n```xml\n" + tool_call("read", {{"path", "foo.cpp"}}) + "\n```\nmore";
        const std::string text = tool_call("write", {{"path", "a.md"}, {"content", content}});
        const auto o = fi::parse_qwen_tool_call_output(text, 64, *shared, false, ninfer::FinishReason::StopToken);
        failures += check(o.is_tool_call_response && o.tool_calls.size() == 1 &&
                              o.tool_calls[0].name == "write" &&
                              o.diagnostics.fenced_markers_suppressed == 0 &&
                              !o.diagnostics.ended_in_unclosed_fence,
                          "N6b: an accepted write with a fenced example, fence 0/false");
        failures += stream_matches_one_shot(text, shared, false, ninfer::FinishReason::StopToken, "N6b streaming");
    }
    // N6c: rejected, strict, OutputLimit -> text, malformed_structure, fence 2/true.
    {
        const std::string text = "<function=read>\n<parameter=path>\nx\n```xml\n" +
                                 tool_call("bash", {{"command", "ls"}}) + "\n";
        const auto o = fi::parse_qwen_tool_call_output(text, 64, *shared, false, ninfer::FinishReason::OutputLimit);
        failures += check(!o.is_tool_call_response && o.tool_calls.empty() &&
                              o.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::MalformedStructure &&
                              o.diagnostics.fenced_markers_suppressed == 2 &&
                              o.diagnostics.ended_in_unclosed_fence,
                          "N6c: a rejected region with an unclosed fence, fence 2/true");
        failures += stream_matches_one_shot(text, shared, false, ninfer::FinishReason::OutputLimit, "N6c streaming");
    }
    // Round-3 S3 (no latch): unchanged (fence 2/true).
    {
        const std::string text = "Here is code:\n```python\nprint(1)\n\n" +
                                 tool_call("write", {{"path", "d.md"}, {"content", "Doc"}});
        const auto o = fi::parse_qwen_tool_call_output(text, 64, *shared, false, ninfer::FinishReason::StopToken);
        failures += check(!o.is_tool_call_response && o.tool_calls.empty() &&
                              o.diagnostics.ended_in_unclosed_fence &&
                              o.diagnostics.fenced_markers_suppressed == 2,
                          "R3 S3: the no-latch unclosed fence is unchanged (fence 2/true)");
    }
    // N-08 reproducer: reports ended_in_unclosed_fence == false.
    {
        const std::string text = std::string("```\nx\n```  \r\n") + tool_call("bash", {{"command", "ls"}});
        const auto o = fi::parse_qwen_tool_call_output(text, 64, *shared, false, ninfer::FinishReason::StopToken);
        failures += check(o.is_tool_call_response && !o.diagnostics.ended_in_unclosed_fence,
                          "N-08: the N-08 reproducer reports ended_in_unclosed_fence == false");
    }
    return failures;
}

int test_round4_r1_residual_pinned() {
    using ninfer::FinishReason;
    Json bash_props = Json::object();
    bash_props["command"] = Json{{"type", "string"}};
    const std::vector<std::string> definitions = {tool_definition("bash", bash_props)};
    const auto shared = contract_from_definitions(definitions);
    // Documented Round-3 section-9 residual R1 (phantom acceptance): an unfenced complete
    // example is followed by prose ending with the canonical closer lines. The Stage-1
    // value-swallowing candidate commits the example and swallows the closer lines and the
    // trailing prose into the bash command. This is a documented residual, not a bug; any
    // change to this verdict must be deliberate.
    const std::string text = "Example:\n" + tool_call("bash", {{"command", "ls"}}) +
                             "\nThen close with\n</parameter>\n</function>\n</tool_call>";
    const std::string expected_args =
        std::string("{\"command\":\"ls\\n</parameter>\\n</function>\\n</tool_call>\\nThen close with\"}");
    int failures = 0;
    const auto strict = fi::parse_qwen_tool_call_output(text, 64, *shared, false,
                                                        ninfer::FinishReason::StopToken);
    failures += check(strict.is_tool_call_response && strict.tool_calls.size() == 1 &&
                          strict.tool_calls[0].name == "bash" &&
                          strict.tool_calls[0].arguments_json == expected_args &&
                          strict.content == "Example:" && strict.diagnostics.markup_tolerant_completion,
                      "R4 R1 residual: strict commits the phantom bash call under "
                      "PayloadFidelity (documented)");
    const auto tol = fi::parse_qwen_tool_call_output(text, 64, *shared, true,
                                                     ninfer::FinishReason::StopToken);
    failures += check(tol.is_tool_call_response && tol.tool_calls.size() == 1 &&
                          tol.tool_calls[0].name == "bash" &&
                          tol.tool_calls[0].arguments_json == expected_args &&
                          tol.content == "Example:" && tol.diagnostics.markup_tolerant_completion,
                      "R4 R1 residual: tolerant StopToken commits the phantom bash call under "
                      "PayloadFidelity (documented)");
    failures += stream_matches_one_shot(text, shared, false, ninfer::FinishReason::StopToken,
                                        "R4 R1 residual streaming");
    // R5-06: the FailClosed ambiguity policy refuses this documented ambiguity class: the
    // same bytes are either a complete call at the early closer (the rest is trailing
    // content) or a value extended to the later closer. Both are structurally plausible, so
    // the region is returned as text with ambiguous_structure — strict and tolerant alike.
    const auto fail_closed = fi::parse_qwen_tool_call_output(text, 64, *shared, false,
                                                             ninfer::FinishReason::StopToken,
                                                             ninfer::ToolCallSyntaxMode::Compatibility,
                                                             ninfer::ToolCallAmbiguityPolicy::FailClosed);
    failures += check(!fail_closed.is_tool_call_response && fail_closed.tool_calls.empty() &&
                          fail_closed.content == text &&
                          fail_closed.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::AmbiguousStructure,
                      "R5-06 FailClosed: the R1 phantom-acceptance class is refused as ambiguous");
    const auto fail_closed_tol =
        fi::parse_qwen_tool_call_output(text, 64, *shared, true, ninfer::FinishReason::StopToken,
                                        ninfer::ToolCallSyntaxMode::Compatibility,
                                        ninfer::ToolCallAmbiguityPolicy::FailClosed);
    failures += check(!fail_closed_tol.is_tool_call_response &&
                          fail_closed_tol.tool_calls.empty() &&
                          fail_closed_tol.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::AmbiguousStructure,
                      "R5-06 FailClosed: tolerant mode does not recover the ambiguous region");
    return failures;
}


// R5-07: the top-level entry syntax is an explicit policy. Native (the production Qwen3.8
// default) latches only the wrapped <tool_call> entry; compatibility keeps the legacy
// bare-function, invoke and function_calls top-level forms. The canonical wrapped call
// parses identically in both modes.
int test_r5_syntax_mode_native_vs_compatibility() {
    int failures = 0;
    const fi::ToolCallOutputContract contract = contract_for("read", Json{
                                                        {"path", Json{{"type", "string"}}},
                                                        {"command", Json{{"type", "string"}}}});
    const std::vector<std::pair<std::string, std::string>> compat_entries = {
        {"<function=read>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n",
         "{\"path\":\"/tmp/x\"}"},
        {"<invoke=read>\n<parameter=path>\n/tmp/x\n</parameter>\n</invoke>\n",
         "{\"path\":\"/tmp/x\"}"},
        {"<function_calls>\n<function=read>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>"
         "\n</function_calls>\n",
         "{\"path\":\"/tmp/x\"}"},
    };
    for (const auto& [text, expected_args] : compat_entries) {
        const auto native = fi::parse_qwen_tool_call_output(
            text, 64, contract, false, ninfer::FinishReason::None, ninfer::ToolCallSyntaxMode::QwenWrappedNative);
        failures += check(
            !native.is_tool_call_response && native.tool_calls.empty() &&
                native.diagnostics.marker_seen == false &&
                native.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
            "R5-07 native: a compatibility-only entry never latches as a tool region");
        failures += check(native.content == text,
                          "R5-07 native: the entry returns verbatim as text content");
        const auto compat = fi::parse_qwen_tool_call_output(
            text, 64, contract, false, ninfer::FinishReason::None, ninfer::ToolCallSyntaxMode::Compatibility);
        failures += check(compat.is_tool_call_response && compat.tool_calls.size() == 1 &&
                              compat.tool_calls.front().name == "read" &&
                              compat.tool_calls.front().arguments_json == expected_args,
                          "R5-07 compatibility: the legacy entry keeps its structured call");
    }

    // The product boundary (OutputOptions) defaults to the native syntax for the Qwen3.8
    // production path; this low-level entry keeps the historical compatibility entry set.
    failures += check(ninfer::OutputOptions{}.tool_call_syntax ==
                          ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                      "R5-07: the production OutputOptions default is the native syntax");
    const auto defaulted = fi::parse_qwen_tool_call_output(
        "<function=read>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n", 64, contract);
    failures += check(defaulted.is_tool_call_response && defaulted.tool_calls.size() == 1,
                      "R5-07: the low-level entry default keeps the compatibility entry set");

    // The canonical wrapped Qwen3.8 call is identical in both modes.
    const std::string wrapped = "<tool_call>\n<function=read>\n<parameter=command>\ncat /tmp/x\n"
                                "</parameter>\n</function>\n</tool_call>\n";
    const auto native = fi::parse_qwen_tool_call_output(
        wrapped, 64, contract, false, ninfer::FinishReason::None, ninfer::ToolCallSyntaxMode::QwenWrappedNative);
    const auto compat = fi::parse_qwen_tool_call_output(
        wrapped, 64, contract, false, ninfer::FinishReason::None, ninfer::ToolCallSyntaxMode::Compatibility);
    failures += check(native.is_tool_call_response && native.tool_calls.size() == 1 &&
                          native.tool_calls.front().name == "read" &&
                          native.tool_calls.front().arguments_json ==
                              "{\"command\":\"cat /tmp/x\"}",
                      "R5-07 native: the canonical wrapped call parses");
    failures += check(compat.is_tool_call_response && compat.tool_calls.size() == 1 &&
                          compat.tool_calls.front().name == "read" &&
                          compat.tool_calls.front().arguments_json ==
                              "{\"command\":\"cat /tmp/x\"}" &&
                          native.content == compat.content &&
                          native.diagnostics == compat.diagnostics,
                      "R5-07: the canonical wrapped call is identical in both modes");
    return failures;
}


// R5-06: two write.content payloads and their policy verdicts.
//
// (1) A balanced nested example — the embedded example carries its own <parameter> opener,
// so its closer is consumed by nesting depth and only the outer close is a viable boundary.
// The region is unambiguous: both policies commit the outer write call with the content
// preserved byte-exact. FailClosed must not over-refuse well-formed nested payloads.
//
// (2) An unbalanced early close chain followed by prose and the outer close chain — the R1
// ambiguity class inside a write payload. Both boundaries are structurally viable (a
// complete call at the early close, or the later close as payload). PayloadFidelity commits
// the legacy last-close-wins value; FailClosed refuses the region as ambiguous_structure
// and no call may execute.
int test_r5_ambiguity_policy_write_payload() {
    Json write_props = Json::object();
    write_props["content"] = Json{{"type", "string"}};
    const std::vector<std::string> definitions = {tool_definition("write", write_props)};
    const auto shared = contract_from_definitions(definitions);
    int failures = 0;

    // (1) balanced nested example: unambiguous under both policies.
    const std::string content = "Syntax example:\n" + tool_call("write", {{"content", "ls"}}) +
                                "\nThen run the command.";
    const std::string text = tool_call("write", {{"content", content}});
    const auto fidelity = fi::parse_qwen_tool_call_output(text, 64, *shared, false,
                                                          ninfer::FinishReason::StopToken);
    failures += check(fidelity.is_tool_call_response && fidelity.tool_calls.size() == 1 &&
                          fidelity.tool_calls[0].name == "write" &&
                          fidelity.tool_calls[0].arguments_json ==
                              Json{{"content", content}}.dump(),
                      "R5-06 PayloadFidelity: the balanced embedded example stays inside the "
                      "content byte-exact");
    const auto balanced_closed = fi::parse_qwen_tool_call_output(
        text, 64, *shared, false, ninfer::FinishReason::StopToken,
        ninfer::ToolCallSyntaxMode::Compatibility,
        ninfer::ToolCallAmbiguityPolicy::FailClosed);
    failures += check(balanced_closed.is_tool_call_response &&
                          balanced_closed.tool_calls.size() == 1 &&
                          balanced_closed.tool_calls[0].name == "write" &&
                          balanced_closed.tool_calls[0].arguments_json ==
                              Json{{"content", content}}.dump(),
                      "R5-06 FailClosed: a balanced nested example is not ambiguous and must "
                      "not be refused");

    // (2) the R1 ambiguity class inside a write payload: an early complete close chain,
    // prose, then the outer close chain. Two structurally viable boundaries.
    const std::string r1_content =
        "X\n</parameter>\n</function>\n</tool_call>\nThen close with";
    const std::string r1_text = tool_call("write", {{"content", r1_content}});
    const auto r1_fidelity = fi::parse_qwen_tool_call_output(r1_text, 64, *shared, false,
                                                             ninfer::FinishReason::StopToken);
    failures += check(r1_fidelity.is_tool_call_response && r1_fidelity.tool_calls.size() == 1 &&
                          r1_fidelity.tool_calls[0].name == "write" &&
                          r1_fidelity.tool_calls[0].arguments_json ==
                              Json{{"content", r1_content}}.dump(),
                      "R5-06 PayloadFidelity: the ambiguous write payload commits the legacy "
                      "last-close-wins value");
    const auto r1_closed = fi::parse_qwen_tool_call_output(
        r1_text, 64, *shared, false, ninfer::FinishReason::StopToken,
        ninfer::ToolCallSyntaxMode::Compatibility,
        ninfer::ToolCallAmbiguityPolicy::FailClosed);
    failures += check(!r1_closed.is_tool_call_response && r1_closed.tool_calls.empty() &&
                          r1_closed.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::AmbiguousStructure,
                      "R5-06 FailClosed: the ambiguous write payload is refused as text, no "
                      "call executes");
    return failures;
}
// R6-04 (Round 6 §6): pinned semantic quotation residual — a documented residual, NOT a global
// safety proof. A complete declared canonical unfenced <tool_call> at EOF that is
// byte-identical to a genuine call is still committed under the production policies
// (QwenWrappedNative + FailClosed + strict): the bytes carry no contradictory structure
// (no fence, no competing boundary, no undeclared name, no malformed tail), so a byte parser
// cannot infer whether the preceding prose is explanatory or the natural-language preamble
// the Qwen3.8 template explicitly allows before a function call. FailClosed is not invoked:
// a clean Stage-1 completion never reaches Stage 2. This test makes the residual
// executable/test-visible so documentation cannot later claim FailClosed "eliminates
// phantom calls" while these fixtures keep committing.
int test_r6_complete_unfenced_in_set_example_residual() {
    const std::shared_ptr<const fi::ToolCallOutputContract> contract = contract_from_definitions(
        {tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    const std::string call = tool_call("bash", {{"command", "echo example"}});
    int failures = 0;

    const auto parse_production = [&](const std::string& text) {
        return fi::parse_qwen_tool_call_output(
            text, 64, *contract, false, ninfer::FinishReason::StopToken,
            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed);
    };

    // Fixture A: example label + complete call + EOF — structured (documented residual).
    {
        const auto parsed = parse_production("Example:\n" + call);
        failures += check(parsed.is_tool_call_response, "R6-04 A: prose + complete example commits (residual)");
        failures += check(parsed.tool_calls.size() == 1, "R6-04 A: one structured bash call");
        if (parsed.tool_calls.size() == 1) {
            failures += check(parsed.tool_calls.front().name == "bash", "R6-04 A: call name");
            failures += check(Json::parse(parsed.tool_calls.front().arguments_json).at("command") ==
                                  "echo example",
                              "R6-04 A: argument preserved");
        }
        failures += check(parsed.content == "Example:", "R6-04 A: prose preserved as content");
    }
    // Fixture B: explanatory prose + complete call + EOF — structured.
    {
        const auto parsed = parse_production("Here is the syntax:\n" + call);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "R6-04 B: prose + complete example commits (residual)");
        failures += check(parsed.content == "Here is the syntax:", "R6-04 B: prose preserved");
    }
    // Fixture C: complete call only (positive control; genuinely indistinguishable) — structured.
    {
        const auto parsed = parse_production(call);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "R6-04 C: example-only output commits (indistinguishable)");
        failures += check(parsed.content.empty(), "R6-04 C: no content");
    }
    // Fixture D: fenced version of the same call — text / no calls (the fence guard covers
    // this sub-class; the residue is the UNFENCED class).
    {
        const std::string fenced = "```xml\n" + call + "\n```\n";
        const auto parsed = parse_production(fenced);
        failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                          "R6-04 D: fenced example stays text");
        failures += check(parsed.content == fenced, "R6-04 D: fenced bytes preserved verbatim");
        failures += check(!parsed.diagnostics.marker_seen, "R6-04 D: no marker latch inside fence");
    }
    // §12.3: undeclared complete call — non-executable (declared-tool enforcement).
    {
        const std::string undeclared = tool_call("other", {{"payload", "x"}});
        const auto parsed = parse_production(undeclared);
        failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                          "R6-04 E: undeclared complete call non-executable");
        failures += check(
            parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::UndeclaredTool,
            "R6-04 E: undeclared reason recorded");
    }
    // §12.3: compatibility-only bare entry under Native — non-executable (R5-07 entry set).
    {
        const std::string bare = "<function=bash>\n<parameter=command>\necho example\n</parameter>\n"
                                "</function>";
        const auto parsed = parse_production(bare);
        failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                          "R6-04 F: compat-only bare entry stays text under Native");
    }
    return failures;
}
// R6-05 (Round 6 §7/§12.4/§13): the optional tool-call intent policy. TemplateCompatible
// keeps the upstream Qwen behavior (a tool region may latch after any content);
// RequireToolAtContentStart locks the turn to text once any visible
// (non-formatting-whitespace) Content byte commits. The gate lives in the pre-latch machine,
// so one-shot and streaming share it (the bytewise decoder leg below proves the equality).
// This is a mitigation: the example-only byte-identical case stays executable under both
// modes (pinned by test_r6_complete_unfenced_in_set_example_residual).
int test_r6_intent_policy_matrix() {
    const auto contract = contract_from_definitions(
        {tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    const std::string call = tool_call("bash", {{"command", "echo example"}});
    int failures = 0;

    const auto parse_intent = [&](const std::string& text, ninfer::ToolCallIntentPolicy intent) {
        return fi::parse_qwen_tool_call_output(
            text, 64, *contract, false, ninfer::FinishReason::StopToken,
            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, intent);
    };
    const auto streaming_intent = [&](const std::string& text, ninfer::ToolCallIntentPolicy intent) {
        fi::ToolCallOutputDecoder decoder(
            contract, 64, false, ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, intent);
        std::string visible;
        for (const char byte : text) { visible += decoder.feed(std::string_view(&byte, 1)); }
        const auto terminal = decoder.finish(ninfer::FinishReason::StopToken);
        return fi::ParsedToolCallOutput{.is_tool_call_response = !terminal.tool_calls.empty(),
                                    .content              = visible + terminal.content,
                                    .tool_calls           = terminal.tool_calls,
                                    .diagnostics          = terminal.diagnostics};
    };
    const std::array<ninfer::ToolCallIntentPolicy, 2> both_intents = {
        ninfer::ToolCallIntentPolicy::TemplateCompatible,
        ninfer::ToolCallIntentPolicy::RequireToolAtContentStart};

    // Case A — plain intended call: both modes → call.
    for (const auto intent : both_intents) {
        const auto parsed = parse_intent(call, intent);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "R6-05 A: plain call commits under both intent modes");
    }
    // Case B — whitespace before call: both → call (formatting whitespace never locks;
    // the marker stays at visual column <= 3, so the Round-10 literal guard does not
    // apply — a tab that lands the marker on column 4 is literal, §9.3).
    {
        const std::string text = "\n   " + call;
        for (const auto intent : both_intents) {
            const auto parsed = parse_intent(text, intent);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                              "R6-05 B: whitespace-prefixed call commits under both modes");
            failures += check(parsed.content.empty(), "R6-05 B: no residual content");
        }
    }
    // Case C — prose before call: TemplateCompatible → call; RequireToolAtContentStart → text.
    {
        const std::string text = "I'll inspect it.\n" + call;
        const auto tc = parse_intent(text, ninfer::ToolCallIntentPolicy::TemplateCompatible);
        const auto soc = parse_intent(text, ninfer::ToolCallIntentPolicy::RequireToolAtContentStart);
        failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 1,
                          "R6-05 C: TemplateCompatible keeps the prose-prefixed call");
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R6-05 C: the hardened mode locks the turn to text");
        failures += check(soc.content == text, "R6-05 C: the locked bytes return verbatim");
        failures += check(!soc.diagnostics.marker_seen,
                          "R6-05 C: the locked marker never latches");
    }
    // Case D — explicit example prose: TemplateCompatible → call (documented residual);
    // RequireToolAtContentStart → text / 0 calls.
    {
        const std::string text = "Example:\n" + call;
        const auto tc = parse_intent(text, ninfer::ToolCallIntentPolicy::TemplateCompatible);
        const auto soc = parse_intent(text, ninfer::ToolCallIntentPolicy::RequireToolAtContentStart);
        failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 1,
                          "R6-05 D: TemplateCompatible keeps the example (residual)");
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R6-05 D: the hardened mode rejects the example prose class");
    }
    // Case E — marker-like failed prefix, then a real call: the hardened mode stays text;
    // a failed marker-like prefix cannot bypass the gate.
    {
        const std::string text = "<tool_x>\n" + call;
        const auto soc = parse_intent(text, ninfer::ToolCallIntentPolicy::RequireToolAtContentStart);
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R6-05 E: a failed marker candidate closes the entry gate");
        const auto tc = parse_intent(text, ninfer::ToolCallIntentPolicy::TemplateCompatible);
        failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 1,
                          "R6-05 E: TemplateCompatible still latches the later marker");
    }
    // Case F — fenced example: 0 calls in both modes.
    {
        const std::string text = "```xml\n" + call + "\n```\n";
        for (const auto intent : both_intents) {
            const auto parsed = parse_intent(text, intent);
            failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                              "R6-05 F: the fenced example stays text under both modes");
        }
    }
    // §13B — leading HTML comment: the hardened mode stays text (no HTML special-case).
    {
        const std::string text = "<!-- example -->\n" + call;
        const auto soc = parse_intent(text, ninfer::ToolCallIntentPolicy::RequireToolAtContentStart);
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R6-05 13B: a leading comment locks the turn to text");
    }
    // §13C — unicode prose: the first UTF-8 non-whitespace byte locks text.
    {
        const std::string text = "Beispiel:\n" + call;
        const auto soc = parse_intent(text, ninfer::ToolCallIntentPolicy::RequireToolAtContentStart);
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R6-05 13C: unicode prose locks the turn to text");
    }
    // §7.9 — multiple consecutive valid calls at content start remain unchanged.
    {
        const std::string text = call + "\n" + tool_call("bash", {{"command", "pwd"}});
        for (const auto intent : both_intents) {
            const auto parsed = parse_intent(text, intent);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2,
                              "R6-05 7.9: consecutive wrappers keep their calls");
        }
    }
    // §12.5 — streaming invariance: bytewise streaming equals one-shot for the
    // intent-sensitive fixtures under both modes.
    {
        const std::vector<std::string> fixtures = {
            "\n  \t" + call,
            "I'll inspect it.\n" + call,
            "Example:\n" + call,
            "<tool_x>\n" + call,
            "```xml\n" + call + "\n```\n",
        };
        for (const auto& text : fixtures) {
            for (const auto intent : both_intents) {
                const auto one_shot = parse_intent(text, intent);
                const auto streamed = streaming_intent(text, intent);
                failures += check(
                    streamed.content == one_shot.content &&
                        streamed.tool_calls.size() == one_shot.tool_calls.size(),
                    "R6-05 12.5: bytewise streaming equals one-shot under the intent policy");
            }
        }
    }
    return failures;
}

// R7-01 (Round 7 §3): the R6-05 hardened intent gate only protects the initial scanner
// latch; the Stage-1 retry chain (and the Stage 2/3 bases it feeds) can still accept a
// later top-level marker after bytes that become visible Content. R7-I1/R7-I2: under
// RequireToolAtContentStart only the initially latched entry may execute; a later accepted
// base whose prefix becomes Content is forbidden. TemplateCompatible keeps the historical
// later-retry behavior unchanged.
int test_r7_hardened_intent_blocks_later_retry_base() {
    using Intent = ninfer::ToolCallIntentPolicy;
    const auto contract = contract_from_definitions(
        {tool_definition("read", Json{{"path", Json{{"type", "string"}}}}),
         tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    const std::string R = tool_call("read", {{"path", "example.txt"}});
    const std::string B = tool_call("bash", {{"command", "echo REAL"}});
    int failures = 0;

    const auto parse_intent = [&](const std::string& text, Intent intent) {
        return fi::parse_qwen_tool_call_output(
            text, 64, *contract, false, ninfer::FinishReason::StopToken,
            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, intent);
    };
    const auto streaming_intent = [&](const std::string& text, Intent intent, std::size_t chunk) {
        fi::ToolCallOutputDecoder decoder(
            contract, 64, false, ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, intent);
        std::string visible;
        for (std::size_t i = 0; i < text.size(); i += chunk) {
            visible += decoder.feed(text.substr(i, std::min(chunk, text.size() - i)));
        }
        const auto terminal = decoder.finish(ninfer::FinishReason::StopToken);
        return fi::ParsedToolCallOutput{.is_tool_call_response = !terminal.tool_calls.empty(),
                                        .content              = visible + terminal.content,
                                        .tool_calls           = terminal.tool_calls,
                                        .diagnostics          = terminal.diagnostics};
    };

    // C1: valid first call + prose + valid later call. The later base must not execute in
    // hardened mode; TemplateCompatible preserves the historical later-call behavior.
    {
        const std::string text = R + "\nNow the real action:\n" + B;
        const auto tc = parse_intent(text, Intent::TemplateCompatible);
        failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 1 &&
                              tc.tool_calls.front().name == "bash",
                          "R7 C1: TemplateCompatible preserves the later-call baseline");
        const auto soc = parse_intent(text, Intent::RequireToolAtContentStart);
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R7 C1: hardened mode never executes the later retry base");
        failures += check(soc.content == text,
                          "R7 C1: the whole generated bytes return verbatim");
        // The latched region (complete call + prose + complete call) is structurally
        // ambiguous as one structured suffix: Stage 2 refuses the competing completions
        // under the FailClosed ambiguity policy, so the region falls to verbatim text with
        // the region-wide AmbiguousStructure class (the current exact first-attempt class).
        failures += check(soc.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::AmbiguousStructure,
                          "R7 C1: the ambiguous region falls to text under FailClosed");
        failures += check(soc.diagnostics.marker_seen,
                          "R7 C1: the initially latched marker is still reported");
    }
    // C2: prose immediately after an open wrapper (the historical R3-02 prose_after_wrapper
    // retry): valid under TemplateCompatible, forbidden in hardened mode.
    {
        const std::string text = "<tool_call>\nordinary prose\n" + B;
        const auto tc = parse_intent(text, Intent::TemplateCompatible);
        failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 1 &&
                              tc.tool_calls.front().name == "bash",
                          "R7 C2: TemplateCompatible preserves the prose-after-wrapper retry");
        const auto soc = parse_intent(text, Intent::RequireToolAtContentStart);
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R7 C2: hardened mode rejects the prose-after-wrapper rebase");
        failures += check(soc.content == text, "R7 C2: the region returns verbatim");
    }
    // C3: a latched malformed wrapper region (closed) followed by a valid later call.
    {
        const std::string text = "<tool_call>\nordinary prose\n</tool_call>\n" + B;
        const auto tc = parse_intent(text, Intent::TemplateCompatible);
        failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 1 &&
                              tc.tool_calls.front().name == "bash",
                          "R7 C3: TemplateCompatible preserves the later-call retry");
        const auto soc = parse_intent(text, Intent::RequireToolAtContentStart);
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R7 C3: hardened mode never executes after a failed initial region");
        failures += check(soc.content == text, "R7 C3: the region returns verbatim");
    }
    // C7: leading formatting whitespace before the first call (held pre-latch whitespace):
    // the initial entry is still the only executable entry in hardened mode (R7-I1), and
    // streaming equals one-shot across chunk sizes.
    {
        const std::string text = "\n  " + R + "\nNow the real action:\n" + B;
        const auto tc = parse_intent(text, Intent::TemplateCompatible);
        failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 1 &&
                              tc.tool_calls.front().name == "bash",
                          "R7 C7: TemplateCompatible preserves the whitespace-prefixed baseline");
        const auto soc = parse_intent(text, Intent::RequireToolAtContentStart);
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R7 C7: held whitespace does not open a later retry base");
        failures += check(soc.content == text, "R7 C7: the region returns verbatim");
        for (const std::size_t chunk : {1, 2, 5, 7, 4096}) {
            const auto streamed = streaming_intent(text, Intent::RequireToolAtContentStart, chunk);
            failures += check(streamed.tool_calls.empty() && streamed.content == text,
                              "R7 C7: streaming equals one-shot at the hardened cutoff");
        }
    }
    return failures;
}

// R7-01b (Round 7 §4): the streaming decoder's terminal re-parse must carry the exact same
// intent policy as the one-shot entry and the live machine. A dropped policy parameter made
// the decoder silently fall back to TemplateCompatible after a latch.
int test_r7_decoder_intent_parity() {
    using Intent = ninfer::ToolCallIntentPolicy;
    const auto contract = contract_from_definitions(
        {tool_definition("read", Json{{"path", Json{{"type", "string"}}}}),
         tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    const std::string R = tool_call("read", {{"path", "example.txt"}});
    const std::string B = tool_call("bash", {{"command", "echo REAL"}});
    int failures = 0;

    const std::string retry_text    = R + "\nNow the real action:\n" + B; // post-latch retry
    const std::string prose_text    = "I'll inspect it.\n" + B;         // pre-latch prose
    const std::vector<std::pair<Intent, std::string>> cases = {
        {Intent::TemplateCompatible, retry_text},
        {Intent::RequireToolAtContentStart, retry_text},
        {Intent::TemplateCompatible, prose_text},
        {Intent::RequireToolAtContentStart, prose_text},
        {Intent::RequireToolAtContentStart, B},
    };
    for (const auto& [intent, text] : cases) {
        fi::ToolCallOutputDecoder decoder(
            contract, 64, false, ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, intent);
        std::string visible;
        for (std::size_t i = 0; i < text.size(); ++i) {
            visible += decoder.feed(text.substr(i, 1));
        }
        const auto terminal = decoder.finish(ninfer::FinishReason::StopToken);
        const fi::ParsedToolCallOutput streamed{.is_tool_call_response = !terminal.tool_calls.empty(),
                                                .content              = visible + terminal.content,
                                                .tool_calls           = terminal.tool_calls,
                                                .diagnostics          = terminal.diagnostics};
        const fi::ParsedToolCallOutput one_shot = fi::parse_qwen_tool_call_output(
            text, 64, *contract, false, ninfer::FinishReason::StopToken,
            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, intent);
        if (streamed.content != one_shot.content || streamed.tool_calls.size() != one_shot.tool_calls.size() ||
            streamed.diagnostics != one_shot.diagnostics) {
            std::printf("fail parity: intent=%d streamed.calls=%zu one_shot.calls=%zu\n",
                        static_cast<int>(intent), streamed.tool_calls.size(), one_shot.tool_calls.size());
        }
        failures += check(streamed.content == one_shot.content &&
                              streamed.tool_calls.size() == one_shot.tool_calls.size() &&
                              streamed.diagnostics == one_shot.diagnostics,
                          "R7-01b: the decoder terminal re-parse carries the intent policy");
    }
    return failures;
}

// R7-01 extended corpus (Round 7 §12-§15): the intent cutoff must leave the initial-entry
// stages (Stage 2, tolerant Stage 3), the fence rules, consecutive wrappers, finish reasons
// and the syntax x intent combinations intact.
int test_r7_intent_retry_cross_matrix() {
    using Intent = ninfer::ToolCallIntentPolicy;
    using Finish = ninfer::FinishReason;
    const auto contract = contract_from_definitions(
        {tool_definition("read", Json{{"path", Json{{"type", "string"}}}}),
         tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    const auto contract_write = contract_from_definitions(
        {tool_definition("write",
                         Json{{"path", Json{{"type", "string"}}},
                              {"content", Json{{"type", "string"}}}})});
    const std::string R = tool_call("read", {{"path", "example.txt"}});
    const std::string B = tool_call("bash", {{"command", "echo REAL"}});
    const std::string B_trunc =
        "<tool_call>\n<function=bash>\n<parameter=command>\necho REAL\n</parameter>\n</function>";
    int failures = 0;

    const auto parse = [&](const std::string& text, Intent intent, bool tolerant = false,
                           Finish finish = Finish::StopToken,
                           ninfer::ToolCallSyntaxMode syntax =
                               ninfer::ToolCallSyntaxMode::QwenWrappedNative) {
        return fi::parse_qwen_tool_call_output(text, 64, *contract, tolerant, finish, syntax,
                                               ninfer::ToolCallAmbiguityPolicy::FailClosed,
                                               intent);
    };
    const auto streaming = [&](const std::string& text, Intent intent, std::size_t chunk,
                               bool tolerant = false) {
        fi::ToolCallOutputDecoder decoder(
            contract, 64, tolerant, ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, intent);
        std::string visible;
        for (std::size_t i = 0; i < text.size(); i += chunk) {
            visible += decoder.feed(text.substr(i, std::min(chunk, text.size() - i)));
        }
        const auto terminal = decoder.finish(Finish::StopToken);
        return fi::ParsedToolCallOutput{.is_tool_call_response = !terminal.tool_calls.empty(),
                                        .content              = visible + terminal.content,
                                        .tool_calls           = terminal.tool_calls,
                                        .diagnostics          = terminal.diagnostics};
    };
    const auto equal = [](const fi::ParsedToolCallOutput& a, const fi::ParsedToolCallOutput& b) {
        return a.content == b.content && a.tool_calls.size() == b.tool_calls.size() &&
               a.diagnostics == b.diagnostics && a.is_tool_call_response == b.is_tool_call_response;
    };

    // C4: the initial entry needs Stage 2 (consistent value-boundary resolution, the P-H
    // nested-example fixture): both modes must agree and keep the Stage-2 completion.
    {
        const std::string text =
            tool_call("write",
                      {{"path", "d.md"},
                       {"content",
                        tool_call("write", {{"path", "x"}, {"content", "hello"}})}});
        const auto parsed = fi::parse_qwen_tool_call_output(
            text, 64, *contract_write, false, Finish::StopToken,
            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, Intent::RequireToolAtContentStart);
        const auto parsed_tc = fi::parse_qwen_tool_call_output(
            text, 64, *contract_write, false, Finish::StopToken,
            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, Intent::TemplateCompatible);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().name == "write",
                          "R7 C4: Stage 2 at the initial entry stays intact in hardened mode");
        failures += check(equal(parsed, parsed_tc), "R7 C4: hardened equals the baseline");
    }
    // C5: the initial entry is tolerant-recoverable (function close consumed, wrapper close
    // cut off): both modes must agree and keep the recovery verdict.
    {
        const std::string text =
            "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>";
        const auto parsed = fi::parse_qwen_tool_call_output(
            text, 64, *contract, true, Finish::StopToken,
            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, Intent::RequireToolAtContentStart);
        const auto parsed_tc = fi::parse_qwen_tool_call_output(
            text, 64, *contract, true, Finish::StopToken,
            ninfer::ToolCallSyntaxMode::QwenWrappedNative,
            ninfer::ToolCallAmbiguityPolicy::FailClosed, Intent::TemplateCompatible);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.diagnostics.fallback_reason ==
                                  ninfer::ToolCallParseFallbackReason::TruncatedTail,
                          "R7 C5: base-zero tolerant recovery stays intact in hardened mode");
        failures += check(equal(parsed, parsed_tc), "R7 C5: hardened equals the baseline");
    }
    // C6: a failed initial region followed by a later tolerant-recoverable call: the later
    // recovery stays the historical TemplateCompatible behavior; hardened mode keeps 0 calls.
    {
        const std::string text = "<tool_call>\nordinary prose\n</tool_call>\n" + B_trunc;
        const auto tc = fi::parse_qwen_tool_call_output(text, 64, *contract, true, Finish::StopToken,
                                                        ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                                        ninfer::ToolCallAmbiguityPolicy::FailClosed,
                                                        Intent::TemplateCompatible);
        failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 1 &&
                              tc.tool_calls.front().name == "bash",
                          "R7 C6: TemplateCompatible preserves the later tolerant recovery");
        const auto soc = fi::parse_qwen_tool_call_output(text, 64, *contract, true, Finish::StopToken,
                                                         ninfer::ToolCallSyntaxMode::QwenWrappedNative,
                                                         ninfer::ToolCallAmbiguityPolicy::FailClosed,
                                                         Intent::RequireToolAtContentStart);
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R7 C6: hardened mode never recovers at a later base");
        failures += check(soc.content == text, "R7 C6: the region returns verbatim");
    }
    // C9: a fenced later call must never become an executable retry base. The fenced
    // marker is a read call and the real later marker a bash call, so the executed name
    // proves the fenced marker was skipped, not used as a base. (Fence markers inside a
    // latched region are not counted by the fence diagnostic: that field covers marker
    // candidates, per the pinned Round-4 scope.)
    {
        const std::string text = "<tool_call>\nordinary prose\n</tool_call>\n```xml\n" + R +
                                 "\n```\n" + B;
        const auto tc = parse(text, Intent::TemplateCompatible);
        failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 1 &&
                              tc.tool_calls.front().name == "bash",
                          "R7 C9: TemplateCompatible skips the fenced marker and retries");
        failures += check(!tc.diagnostics.ended_in_unclosed_fence,
                          "R7 C9: the fence is closed in the region");
        const auto soc = parse(text, Intent::RequireToolAtContentStart);
        failures += check(!soc.is_tool_call_response && soc.tool_calls.empty(),
                          "R7 C9: hardened mode never executes after the initial region");
        failures += check(soc.content == text, "R7 C9: the region returns verbatim");
    }
    // C10: consecutive valid wrappers at the initial structured suffix stay supported in
    // both modes (the initial entry alone carries the whole structured suffix).
    {
        const std::string text = R + "\n" + B;
        const auto tc = parse(text, Intent::TemplateCompatible);
        const auto soc = parse(text, Intent::RequireToolAtContentStart);
        failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 2,
                          "R7 C10: the baseline keeps the consecutive structured calls");
        failures += check(equal(soc, tc), "R7 C10: hardened mode keeps the consecutive calls");
    }
    // Finish-reason matrix (Round 7 §13): the intent restriction is orthogonal to the
    // terminal reason; base-zero recovery stays finish-reason-sensitive as before.
    {
        const std::string text = R + "\nNow the real action:\n" + B;
        const std::vector<Finish> reasons = {Finish::StopToken, Finish::StopString,
                                             Finish::OutputLimit, Finish::ContextCapacity,
                                             Finish::Cancelled, Finish::None};
        for (const Finish reason : reasons) {
            const auto soc = parse(text, Intent::RequireToolAtContentStart, false, reason);
            failures += check(!soc.is_tool_call_response && soc.tool_calls.empty() &&
                                  soc.content == text,
                              "R7 finish matrix: hardened mode is reason-independent");
            const auto tc = parse(text, Intent::TemplateCompatible, false, reason);
            failures += check(tc.is_tool_call_response && tc.tool_calls.size() == 1 &&
                                  tc.tool_calls.front().name == "bash",
                              "R7 finish matrix: the baseline later call is reason-independent");
        }
    }
    // Syntax x intent matrix (Round 7 §14): a bare function entry is not a native top-level
    // entry; in compatibility syntax it is eligible under TemplateCompatible and locked by
    // visible prose under the hardened intent.
    {
        const std::string bare = "<function=bash>\n<parameter=command>\necho REAL\n</parameter>\n</function>";
        const std::string text = "Example:\n" + bare;
        const auto tc_compat = parse(text, Intent::TemplateCompatible, false, Finish::StopToken,
                                     ninfer::ToolCallSyntaxMode::Compatibility);
        failures += check(tc_compat.is_tool_call_response && tc_compat.tool_calls.size() == 1,
                          "R7 syntax matrix: the compat baseline accepts the bare entry");
        const auto soc_compat = parse(text, Intent::RequireToolAtContentStart, false, Finish::StopToken,
                                      ninfer::ToolCallSyntaxMode::Compatibility);
        failures += check(!soc_compat.is_tool_call_response && soc_compat.tool_calls.empty(),
                          "R7 syntax matrix: hardened mode locks the bare entry after prose");
        const auto soc_native = parse(text, Intent::RequireToolAtContentStart, false, Finish::StopToken,
                                      ninfer::ToolCallSyntaxMode::QwenWrappedNative);
        failures += check(!soc_native.is_tool_call_response && soc_native.tool_calls.empty() &&
                              soc_native.content == text,
                          "R7 syntax matrix: the bare entry stays text in native syntax");
    }
    // Streaming invariance (Round 7 §3.13) for the hardened cutoff across the bypass corpus.
    {
        const std::string texts[] = {R + "\nNow the real action:\n" + B,
                                    "<tool_call>\nordinary prose\n</tool_call>\n" + B_trunc,
                                    "<tool_call>\nordinary prose\n</tool_call>\n```xml\n" + B +
                                        "\n```\n" + B};
        for (const auto& text : texts) {
            const auto one_shot = parse(text, Intent::RequireToolAtContentStart);
            for (const std::size_t chunk : {1, 2, 5, 7, 4096}) {
                const auto streamed = streaming(text, Intent::RequireToolAtContentStart, chunk);
                failures += check(equal(streamed, one_shot),
                                  "R7 streaming matrix: hardened streaming equals one-shot");
            }
        }
    }
    return failures;
}

// R8-01 / R9-01 (Round 8 §3.20, Round 9 §2/§3.18): the post-call eligibility corpus pins
// the production parser and the grammar constraint to the same outcome. R9-01 sharpens
// the cross-check: the constraint now REJECTS (verdict) the visible-suffix texts that
// the strict parser demotes to zero calls — a Rejected verdict and committed calls may
// not co-occur, and a cut stream inside a pending second marker is NeedMore
// (EOS-illegal), not a committed open region.
int test_r8_parser_constraint_post_call_cross_check() {
    using Intent  = ninfer::ToolCallIntentPolicy;
    using Finish  = ninfer::FinishReason;
    using Syntax  = ninfer::ToolCallSyntaxMode;
    using Verdict = fi::ToolCallConstraintVerdict;
    const auto contract = contract_from_definitions(
        {tool_definition("read", Json{{"path", Json{{"type", "string"}}}}),
         tool_definition("bash", Json{{"command", Json{{"type", "string"}}}})});
    const std::string R = tool_call("read", {{"path", "example.txt"}});
    const std::string B = tool_call("bash", {{"command", "echo REAL"}});
    int failures = 0;

    const auto parse = [&](const std::string& text, Syntax syntax) {
        return fi::parse_qwen_tool_call_output(text, 64, *contract, false, Finish::StopToken,
                                               syntax, ninfer::ToolCallAmbiguityPolicy::FailClosed,
                                               Intent::RequireToolAtContentStart);
    };
    // The constraint's verdict on the full corpus text (a const probe: no state change —
    // a candidate the constrained sampler would reject is never committed).
    const auto constraint_verdict = [&](const std::string& text, Syntax syntax) {
        fi::ToolCallGrammarConstraint constraint(64, syntax, Intent::RequireToolAtContentStart);
        return constraint.check(text);
    };

    struct Entry {
        const char* name;
        std::string text;
        int parser_calls;  // the production parser's structured outcome
        Verdict verdict;   // the constraint's R9-01 full-text verdict
    };
    const std::vector<Entry> corpus = {
        {"P1", "\n \t\n" + R, 1, Verdict::Allowed},
        {"P2", "Example:\n" + R, 0, Verdict::Allowed},
        {"P3", R + "\n\t \r\n" + B, 2, Verdict::Allowed},
        {"P4", R + "\nNow the real action:\n" + B, 0, Verdict::Rejected},
        {"P5", R + "<tool_x>" + B, 0, Verdict::Rejected},
        {"P6", R + "\n<tool_", 0, Verdict::NeedMore},
        {"P7", R + "\r\n" + B, 2, Verdict::Allowed},
        {"P8", R + "\t" + B, 2, Verdict::Allowed},
        {"P9", R + "\f" + B, 0, Verdict::Rejected},
    };
    for (const Syntax syntax : {Syntax::QwenWrappedNative, Syntax::Compatibility}) {
        for (const Entry& entry : corpus) {
            const auto parsed  = parse(entry.text, syntax);
            const int calls    = static_cast<int>(parsed.tool_calls.size());
            const Verdict verdict = constraint_verdict(entry.text, syntax);
            failures += check(calls == entry.parser_calls,
                              std::string("R8 cross ") + entry.name +
                                  ": the parser outcome is unchanged");
            failures += check(verdict == entry.verdict,
                              std::string("R8 cross ") + entry.name +
                                  ": the constraint R9 verdict agrees");
            failures += check(!(verdict == Verdict::Rejected && calls > 0),
                              std::string("R8 cross ") + entry.name +
                                  ": no contradiction between parser and constraint");
        }
    }

    // P6: CALL1 + partial immediate CALL2 (the input ends inside the second wrapper):
    // the strict parser commits nothing (the region is cut), the constraint keeps the
    // prefix legal (NeedMore, EOS-illegal), and the good continuation completes the
    // marker (Round 9 §3.10/§3.14).
    {
        const std::string text = R + "\n<tool_";
        const auto parsed = parse(text, Syntax::QwenWrappedNative);
        failures += check(parsed.tool_calls.empty(), "R8 cross P6: the strict parser commits nothing");
        fi::ToolCallGrammarConstraint constraint(64, Syntax::QwenWrappedNative,
                                                 Intent::RequireToolAtContentStart);
        failures += check(constraint.check(text) == Verdict::NeedMore,
                          "R9 cross P6: the partial second entry needs more bytes");
        constraint.commit(text);
        failures += check(!constraint.can_terminate(),
                          "R9 cross P6: EOS is illegal inside the pending marker");
        constraint.commit("call>");
        failures += check(constraint.active(),
                          "R9 cross P6: the good continuation completes the marker");
    }
    return failures;
}
int main() {
    int failures = 0;
    failures += test_r5_syntax_mode_native_vs_compatibility();
    failures += test_r5_ambiguity_policy_write_payload();
    failures += test_duplicate_parameter_keeps_last_value();
    failures += test_basic_legacy_parsing();
    failures += test_multiple_calls();
    failures += test_declared_strings_preserve_text();
    failures += test_string_values_preserve_embedded_tool_markup();
    failures += test_parameter_delimiters_in_values();
    failures += test_declared_json_types();
    failures += test_boolean_boundary();
    failures += test_exact_integer_boundary();
    failures += test_empty_declared_non_string_is_omitted();
    failures += test_composed_schema_types(); // N-09 a: re-registered
    failures += test_schema_mismatches_remain_structured();
    failures += test_unsupported_schema_uses_legacy_policy();
    failures += test_strict_structure_and_active_tool_set();
    failures += test_name_limits_and_non_strict_omissions();
    failures += test_conflicting_duplicate_tool_contracts_use_legacy_normalization();
    failures += test_all_or_nothing_structural_commit();
    failures += test_quoted_marker_before_real_call();
    failures += test_later_candidate_must_consume_the_end();
    failures += test_incremental_quoted_marker_preserves_bytes();
    failures += test_incremental_valid_and_boolean();
    failures += test_incremental_fallback_preserves_bytes();
    failures += test_incremental_embedded_parameter_markup();
    failures += test_claude_code_xml_markup_variants();
    failures += test_duplicate_parameters_keep_last_value();
    failures += test_attribute_token_boundary();
    failures += test_mismatched_closing_tags_rejected();
    failures += test_claude_code_plan_and_task_create_exact_repro();
    failures += test_tolerant_recovery();
    failures += test_tolerant_truncated_final_call();
    failures += test_tolerant_missing_function_close_bracket();
    failures += test_tolerant_undeclared_and_value_cut();
    failures += test_grammar_header_forms_one_shot();
    failures += test_streaming_recognizes_grammar_markers();
    failures += test_recovery_policy_phase3();
    failures += test_literal_parameter_opener_in_payload_does_not_nest();
    failures += test_literal_param_opener_in_payload_does_not_nest();
    failures += test_coding_payloads_round_trip();
    failures += test_payload_adversarial_corpus_round_trip();
    failures += test_parameter_value_with_inner_opener_only_closes_at_outer();
    failures += test_mixed_parameter_families_inside_payload_are_opaque();
    failures += test_tolerant_never_commits_open_function();
    failures += test_tolerant_commits_function_closed_missing_wrapper();
    failures += test_eof_parameter_closer_never_executes_open_function();
    failures += test_partial_next_parameter_does_not_reopen_previous_value();
    failures += test_all_delimiter_byte_cuts_preserve_previous_boundary();
    failures += test_all_delimiter_byte_cuts_other_families();
    failures += test_nested_and_cross_nested_wrappers_rejected();
    failures += test_recovery_retry_entry_policy();
    failures += test_marker_breaking_angle_restarts_candidate();
    failures += test_function_calls_holds_call_sequence();
    failures += test_failed_open_wrapper_never_recovers_nested_wrapper();
    failures += test_recovery_only_restarts_after_proven_scope();
    failures += test_function_calls_missing_wrapper_close();
    failures += test_function_calls_close_then_remainder();
    failures += test_entry_forms_compose_consistently();
    failures += test_complete_and_partial_next_entry_have_same_boundary_result();
    failures += test_tolerant_rejects_undeclared_tools();
    failures += test_strict_and_tolerant_share_declared_name_policy();
    failures += test_output_normalization_cannot_emit_out_of_set_name();
    failures += test_placeholder_function_name_is_not_executable();
    failures += test_fenced_content_never_latches();
    failures += test_fence_split_at_every_byte();
    failures += test_round3_spec_corpus();
    failures += test_round3_streaming_equivalence_fuzz();
    failures += test_round3_work_bounds();
    failures += test_round4_stage2_many_values();
    failures += test_round4_work_bounds();
    failures += test_round4_param_family_stage2();
    failures += test_round4_same_family_balance();
    failures += test_round4_fence_diagnostics_scope();
    failures += test_round4_r1_residual_pinned();
    failures += test_r6_complete_unfenced_in_set_example_residual();
    failures += test_r6_intent_policy_matrix();
    failures += test_r7_decoder_intent_parity();
    failures += test_r7_intent_retry_cross_matrix();
    failures += test_r7_hardened_intent_blocks_later_retry_base();
    failures += test_r8_parser_constraint_post_call_cross_check();
    failures += test_r10_indented_wrapper_is_literal();
    failures += test_r10_mixed_indentation_suppressed();
    failures += test_r10_three_space_control_still_latches();
    failures += test_r10_blank_indented_line_does_not_lock();
    failures += test_r10_indented_example_then_real_call();
    failures += test_r10_fenced_control_unaffected();
    failures += test_r10_compat_indented_bare_function_suppressed();
    failures += test_r10_native_bare_function_unchanged();
    failures += test_r10_indented_payload_preserved();
    failures += test_r10_every_byte_split();
    failures += test_r10_multi_boundary_splits();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
