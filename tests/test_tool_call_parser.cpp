#include "models/qwen3_5/frontend/tool_call_stream.h"


#include "models/qwen3_5/frontend/tool_call_parser.h"
#include <nlohmann/json.hpp>

#include <initializer_list>
#include <iostream>
#include <memory>
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
        auto terminal = decoder.finish();
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
    auto bytewise_terminal = bytewise.finish();
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
    const std::string text = tool_call("configure", {{"missing_type", "7"},
                                                     {"alias", "8"},
                                                     {"invalid_type_array", "9"},
                                                     {"partial_anyof", "7.5"},
                                                     {"mixed_composition", "True"},
                                                     {"undeclared", "{\"x\":1}"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.diagnostics.schema_mismatch_arguments == 1,
                      "unsupported schema did not retain legacy policy");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("alias") == 8 &&
                          args.at("invalid_type_array") == 9,
                      "legacy numeric inference changed");
    failures += check(args.at("partial_anyof") == 7.5 && args.at("mixed_composition") == "True",
                      "unsupported composition was partially inferred");
    failures +=
        check(args.at("undeclared").at("x") == 1, "undeclared parameter legacy inference changed");
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
    // R2-I1/CR1: the quoted broken wrapper is a failed open wrapper. Its scope cannot be
    // proven exited from the wire bytes, so the failed wrapper owns the remaining ambiguous
    // bytes and the whole region falls back to content — the real call after it stays
    // non-executable (non-execution over recovery, R2-I7).
    failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty() &&
                          parsed.content == text && parsed.diagnostics.marker_seen &&
                          parsed.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::MalformedStructure,
                      "a quoted broken wrapper before a real call was recovered (CR1)");
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
    auto terminal = decoder.finish();

    int failures = 0;
    failures += check(terminal.tool_calls.empty(),
                      "a failed open wrapper's scope was re-entered during streaming (CR1)");
    failures += check(visible + terminal.content == text,
                      "incremental quoted marker lost or duplicated bytes");
    failures += check(terminal.diagnostics.marker_seen &&
                          terminal.diagnostics.structured_call_count == 0 &&
                          terminal.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::MalformedStructure,
                      "incremental quoted marker changed terminal diagnostics");
    return failures;
}

int test_incremental_valid_and_boolean() {
    fi::ToolCallOutputDecoder legacy(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string visible;
    visible += legacy.feed("Calling weather.  \n<tool_");
    visible += legacy.feed("call>\n<function=get_weather>");
    visible += legacy.feed("\n</function>\n</tool_call>");
    auto legacy_terminal = legacy.finish();
    visible += legacy_terminal.content;

    auto bool_contract =
        output_contract_for("configure", Json{{"enabled", Json{{"type", "boolean"}}}});
    fi::ToolCallOutputDecoder boolean(std::move(bool_contract), 64);
    std::string boolean_visible;
    boolean_visible += boolean.feed("<tool_call>\n<function=configure>\n<parameter=enabled>\nT");
    boolean_visible += boolean.feed("r");
    boolean_visible += boolean.feed("ue\n</parameter>\n</function>\n</tool_call>");
    auto boolean_terminal = boolean.finish();

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
    auto malformed_terminal = malformed.finish();
    restored += malformed_terminal.content;

    fi::ToolCallOutputDecoder ordinary(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string ordinary_text;
    ordinary_text += ordinary.feed("ordinary text  ");
    ordinary_text += ordinary.finish().content;

    const std::string partial_original = "  <tool_x then <tool_";
    fi::ToolCallOutputDecoder partial(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string partial_restored;
    partial_restored += partial.feed("  <too");
    partial_restored += partial.feed("l_x then <tool_");
    partial_restored += partial.finish().content;

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
    auto terminal = decoder.finish();

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

    // A repeated identical parameter is the common agent-harness case: the second write leaves the
    // value alone, and the repair is still counted.
    const std::string identical_dup =
        "<tool_call>\n<function=configure>\n<parameter=value>\nfirst\n</parameter>\n"
        "<parameter=value>\nfirst\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_identical = fi::parse_qwen_tool_call_output(identical_dup, 64, contract);
    failures += check(parsed_identical.is_tool_call_response &&
                          parsed_identical.tool_calls.size() == 1,
                      "duplicate identical parameter was not accepted");
    if (parsed_identical.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed_identical.tool_calls.front().arguments_json);
        failures += check(args.at("value") == "first",
                          "repeated identical parameter value changed");
    }
    failures += check(parsed_identical.diagnostics.duplicate_parameters_repaired == 1,
                      "identical duplicate parameter repair was not recorded");

    // A conflicting repeat keeps the last value, matching the JSON-object rule the `=<name>`
    // markup already follows.
    const std::string conflicting_dup =
        "<tool_call>\n<function=configure>\n<parameter=value>\nfirst\n</parameter>\n"
        "<parameter=value>\nsecond\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_conflicting = fi::parse_qwen_tool_call_output(conflicting_dup, 64, contract);
    failures += check(parsed_conflicting.is_tool_call_response &&
                          parsed_conflicting.tool_calls.size() == 1,
                      "conflicting duplicate parameter fell back to text");
    if (parsed_conflicting.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed_conflicting.tool_calls.front().arguments_json);
        failures += check(args.at("value") == "second",
                          "conflicting duplicate parameter did not keep the last value");
    }
    failures += check(parsed_conflicting.diagnostics.duplicate_parameters_repaired == 1,
                      "conflicting duplicate parameter repair was not recorded");

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
        auto terminal = decoder.finish();
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
// exact value round-trips in both modes and in streaming. Entries that are (or end with) the
// outer closer literal are fundamentally ambiguous at the value end and stay out of this
// matrix; that boundary is pinned by the closer-continuation tests.
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
        for (const FinishReason reason : {FinishReason::StopToken, FinishReason::StopString,
                                          FinishReason::OutputLimit,
                                          FinishReason::ContextCapacity,
                                          FinishReason::Cancelled}) {
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
        for (const FinishReason reason : {FinishReason::StopToken, FinishReason::StopString,
                                          FinishReason::OutputLimit,
                                          FinishReason::ContextCapacity,
                                          FinishReason::Cancelled}) {
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
        for (const FinishReason reason : {FinishReason::StopToken, FinishReason::StopString,
                                          FinishReason::OutputLimit,
                                          FinishReason::ContextCapacity,
                                          FinishReason::Cancelled}) {
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
            auto terminal = decoder.finish(FinishReason::StopToken);
            const auto one_shot = fi::parse_qwen_tool_call_output(
                cut_text, 64, *contract, true, FinishReason::StopToken);
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
            cut_text, 64, *contract, true, FinishReason::OutputLimit);
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
    auto terminal = decoder.finish();
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
    for (const FinishReason reason : {FinishReason::StopToken, FinishReason::StopString,
                                      FinishReason::OutputLimit, FinishReason::ContextCapacity,
                                      FinishReason::Cancelled}) {
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
        auto terminal = decoder.finish();
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
    failures += check_round2_region(trailing, *contract, true, {"read"},
                                    ninfer::ToolCallParseFallbackReason::TruncatedTail,
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
    failures += check_round2_region(mixed, *contract, true, {"read"},
                                    ninfer::ToolCallParseFallbackReason::TruncatedTail,
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
            for (const auto& call : decoder.finish().tool_calls) {
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
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 backtick fence (strict)");
    failures += check_round2_region(backtick, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 backtick fence (tolerant)");
    // Tilde fence without an info string.
    const std::string tilde = "~~~~\n" + example_block + "\n~~~~\n";
    failures += check_round2_region(tilde, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 tilde fence (strict)");
    failures += check_round2_region(tilde, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 tilde fence (tolerant)");
    // An unclosed fence stays open through EOF: the marker inside is content.
    const std::string unclosed = "```xml\n" + example_block + "\n";
    failures += check_round2_region(unclosed, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 unclosed fence (strict)");
    failures += check_round2_region(unclosed, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 unclosed fence (tolerant)");
    // A real call immediately after the closing fence still executes; the fenced example
    // stays content.
    const std::string after_fence = "```xml\n" + example_block + "\n```\n" + real_call;
    failures += check_round2_region(after_fence, *contract, false, {"bash"},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 real call after closed fence (strict)");
    failures += check_round2_region(after_fence, *contract, true, {"bash"},
                                    ninfer::ToolCallParseFallbackReason::None,
                                    "CR6 real call after closed fence (tolerant)");
    // A longer fence contains shorter backtick lines: the inner fences are content.
    const std::string nested_fence = "````\n```\n" + example_block + "\n```\n````\n";
    failures += check_round2_region(nested_fence, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 longer fence contains shorter (strict)");
    failures += check_round2_region(nested_fence, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 longer fence contains shorter (tolerant)");
    // The opening fence line's info string may carry a marker: it is content.
    const std::string info_string = "```xml <tool_call>\n" + example_block + "\n```\n";
    failures += check_round2_region(info_string, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 fence info string (strict)");
    failures += check_round2_region(info_string, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 fence info string (tolerant)");
    // CRLF line endings.
    const std::string crlf = "Example:\r\n\r\n```xml\r\n" + example_block + "\r\n```\r\n";
    failures += check_round2_region(crlf, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 CRLF fence (strict)");
    failures += check_round2_region(crlf, *contract, true, {},
                                    ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                    "CR6 CRLF fence (tolerant)");
    // A complete call before a fence: the terminal policy decides (strict rejects the
    // trailing fence block, tolerant retains the call with a truncation diagnostic).
    const std::string before_fence = real_call + "\nExample:\n\n```\ncontent\n```\n";
    failures += check_round2_region(before_fence, *contract, false, {},
                                    ninfer::ToolCallParseFallbackReason::TrailingContent,
                                    "CR6 call before fence (strict)");
    failures += check_round2_region(before_fence, *contract, true, {"bash"},
                                    ninfer::ToolCallParseFallbackReason::TruncatedTail,
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
        auto terminal = decoder.finish();
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

} // namespace

int test_duplicate_parameter_keeps_last_value() {
    int failures = 0;
    const fi::ToolCallOutputContract contract =
        contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    const std::string duplicate = tool_call("configure", {{"value", "first"}, {"value", "second"}});
    const auto parsed = fi::parse_qwen_tool_call_output(duplicate, 64, contract);

    failures += check(parsed.is_tool_call_response, "duplicate parameter still fell back to text");
    failures += check(parsed.content.empty(), "duplicate parameter left prose behind");
    failures += check(parsed.tool_calls.size() == 1, "duplicate parameter did not yield one call");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().arguments_json == R"({"value":"second"})",
                          "duplicate parameter did not keep the last value");
    }
    failures += check(parsed.diagnostics.fallback_reason ==
                          ninfer::ToolCallParseFallbackReason::None,
                      "duplicate parameter still reported a fallback reason");
    failures += check(parsed.diagnostics.duplicate_parameters_repaired == 1,
                      "duplicate parameter repair was not recorded in diagnostics");
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
    auto terminal = decoder.finish();
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
        auto terminal = decoder.finish();
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
        for (const FinishReason reason : {FinishReason::StopToken, FinishReason::StopString,
                                          FinishReason::OutputLimit,
                                          FinishReason::ContextCapacity,
                                          FinishReason::Cancelled}) {
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
    for (const FinishReason reason : {FinishReason::StopToken, FinishReason::StopString,
                                      FinishReason::OutputLimit,
                                      FinishReason::ContextCapacity,
                                      FinishReason::Cancelled}) {
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
        failures += check(tail.is_tool_call_response && tail.tool_calls.size() == 1 &&
                              tail.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "complete calls before trailing prose were not recovered");
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
        stream(fixture, true, FinishReason::StopToken, terminal, write_contract);
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
        const auto terminal = decoder.finish();
        failures += check(terminal.tool_calls.size() == 2 && visible.empty() &&
                              terminal.content.empty() &&
                              terminal.diagnostics.fallback_reason == Reason::None,
                          "the streaming <function_calls> sequence diverged from the one-shot "
                          "parse");
    }
    return failures;
}

int main() {
    int failures = 0;
    failures += test_duplicate_parameter_keeps_last_value();
    failures += test_basic_legacy_parsing();
    failures += test_multiple_calls();
    failures += test_declared_strings_preserve_text();
    failures += test_string_values_preserve_embedded_tool_markup();
    failures += test_parameter_delimiters_in_values();
    failures += test_declared_json_types();
    failures += test_boolean_boundary();
    failures += test_exact_integer_boundary();
    failures += test_composed_schema_types();
    failures += test_empty_declared_non_string_is_omitted();
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
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
