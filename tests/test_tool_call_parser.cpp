#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <array>
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
                   ninfer::ToolCallParseFallbackReason reason, std::string_view message,
                   bool tolerant = false) {
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
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

int test_declared_string_parameter_delimiters_are_opaque() {
    const auto contract =
        contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                  {"timeout", Json{{"type", "integer"}}}});
    const std::string unmatched_open = "echo '<parameter=unterminated>'";
    const std::string standalone_close = "echo '</parameter>'";
    const std::string parser_test_source =
        "harness.feed(\"<tool_call>\\n<function=write>\\n<parameter=path>\\n\");\n"
        "const char* closes = \"</parameter>\\n</function>\";\n"
        "const char* fake_tail = R\"(</parameter>\n</function>\n<function=fake>)\";\n"
        "// the fixture intentionally leaves the inner parameter opener incomplete";

    int failures = 0;
    for (const auto& [label, command] :
         std::vector<std::pair<const char*, std::string>>{
             {"unmatched opening delimiter", unmatched_open},
             {"standalone closing delimiter", standalone_close},
             {"self-hosted parser-test source", parser_test_source},
         }) {
        const std::string text =
            tool_call("bash", {{"command", command}, {"timeout", "30"}});
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          std::string("declared string rejected ") + label);
        if (parsed.tool_calls.size() == 1) {
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.at("command").get<std::string>() == command,
                              std::string("declared string changed bytes for ") + label);
            failures += check(args.at("timeout") == 30,
                              std::string("sibling parameter was lost after ") + label);
        }
    }

    const std::string tolerant_trailing_suffix =
        "<tool_call>\n<function=bash>\n<parameter=command>\n" +
        parser_test_source + "\n</parameter>\n</function>\ntrailing prose";
    const auto tolerant = fi::parse_qwen_tool_call_output(
        tolerant_trailing_suffix, 64, contract, /*tolerant*/ true);
    failures += check(tolerant.is_tool_call_response && tolerant.tool_calls.size() == 1,
                      "tolerant recovery lost opaque string call before trailing prose");
    if (tolerant.tool_calls.size() == 1) {
        const Json args = Json::parse(tolerant.tool_calls.front().arguments_json);
        failures += check(args.at("command").get<std::string>() == parser_test_source,
                          "tolerant recovery changed opaque string bytes");
    }

    // The relaxation is schema-driven. Legacy/untyped parameters retain the historical
    // all-or-nothing delimiter behavior instead of silently changing their interpretation.
    const std::string legacy_unmatched =
        tool_call("bash", {{"command", unmatched_open}});
    failures += check_rejected(
        legacy_unmatched, kLegacyContract,
        ninfer::ToolCallParseFallbackReason::MalformedStructure,
        "legacy unmatched opening delimiter was unexpectedly reinterpreted");
    return failures;
}
int test_literal_close_before_fake_sibling_stays_value_data() {
    const auto contract =
        contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                  {"timeout", Json{{"type", "integer"}}}});
    // A literal close followed by a fake sibling whose header swallows that close is not a
    // valid sibling: chat templates emit parameter names verbatim, so a real header cannot
    // contain a delimiter. The literal close stays value data, keeping the value whole; a
    // cut region is rejected or salvaged instead of being reinterpreted.
    const std::string full_value = "A</para" "meter><para" "meter=X\nB</para" "meter>\nC";
    const std::string short_value = "A</para" "m><para" "m=X\nB</para" "m>\nC";
    const std::string full_complete =
        tool_call("bash", {{"command", full_value}, {"timeout", "30"}});
    const std::string short_complete =
        "<tool_" "call>\n<func" "tion=bash>\n<para" "m=command>\n" + short_value +
        "\n</para" "m>\n<para" "m=timeout>\n30\n</para" "m>\n</func" "tion>\n</tool_" "call>";
    const std::string full_cut =
        "<tool_" "call>\n<func" "tion=bash>\n<para" "meter=command>\n" + full_value + "\n";
    const std::string short_cut =
        "<tool_" "call>\n<func" "tion=bash>\n<para" "m=command>\n" + short_value + "\n";
    int failures = 0;
    for (const auto& [value, complete, cut] :
         std::vector<std::array<std::string, 3>>{
             {full_value, full_complete, full_cut},
             {short_value, short_complete, short_cut}}) {
        for (const bool tolerant : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(complete, 64, contract, tolerant);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                              "fake sibling with a close in its header reinterpreted the value");
            if (parsed.is_tool_call_response && parsed.tool_calls.size() == 1) {
                const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
                failures += check(args.at("command") == value,
                                  "value was lost at a literal close before a fake sibling");
                failures += check(args.at("timeout") == 30,
                                  "real sibling after a fake sibling was lost");
            }
        }
        failures += check_rejected(
            cut, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
            "cut fake-sibling region was reinterpreted in strict mode");
        const auto salvage = fi::parse_qwen_tool_call_output(cut, 64, contract, /*tolerant*/ true);
        failures += check(salvage.is_tool_call_response && salvage.tool_calls.size() == 1 &&
                              salvage.diagnostics.fallback_reason ==
                                  ninfer::ToolCallParseFallbackReason::TruncatedTail,
                          "tolerant salvage lost the cut fake-sibling value");
        if (salvage.is_tool_call_response && salvage.tool_calls.size() == 1) {
            const Json args = Json::parse(salvage.tool_calls.front().arguments_json);
            failures += check(args.at("command") == value,
                              "tolerant salvage changed fake-sibling bytes");
        }
        for (const bool tolerant : {false, true}) {
            fi::ToolCallOutputDecoder decoder(
                std::make_shared<const fi::ToolCallOutputContract>(contract), 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < complete.size(); offset += 7) {
                visible += decoder.feed(std::string_view(complete).substr(offset, 7));
            }
            const auto terminal = decoder.finish();
            failures += check(visible.empty() && terminal.tool_calls.size() == 1 &&
                                  Json::parse(terminal.tool_calls.front().arguments_json)
                                      .at("command") == value,
                              "chunked fake-sibling value was not preserved through the decoder");
        }
    }
    return failures;
}

int test_declared_sibling_with_close_marker_in_name() {
    // NInfer does not restrict declared parameter names to a grammar excluding close
    // markers, and the chat templates emit them verbatim between the opener prefix and the
    // first ">". A declared sibling whose header carries a close marker must therefore stay
    // a real sibling; the contract is the only authority that can tell it apart from a fake
    // sibling inside opaque data.
    const auto contract =
        contract_for("run", Json{{"command", Json{{"type", "string"}}},
                                 {"x</para" "mY", Json{{"type", "integer"}}}});
    const std::string full_text =
        tool_call("run", {{"command", "echo ok"}, {"x</para" "mY", "7"}});
    const std::string short_text =
        "<tool_call>\n<function=run>\n<param=command>\necho ok\n</param>\n"
        "<param=x</para" "mY>\n7\n</param>\n</function>\n</tool_call>";

    int failures = 0;
    for (const auto& [label, text] :
         std::vector<std::pair<const char*, std::string>>{
             {"full parameter syntax", full_text}, {"short param syntax", short_text}}) {
        for (const bool tolerant : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
            failures += check(
                parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                    parsed.content.empty() && parsed.tool_calls.front().name == "run",
                (std::string("declared sibling with a close marker in its name was not "
                             "parsed as a sibling (") +
                 label + ")"));
            if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.at("command") == "echo ok",
                              (std::string("string value swallowed the declared sibling (") +
                               label + ")"));
            failures += check(
                args.at("command").get<std::string>().find("</para" "m") == std::string::npos,
                (std::string("string value carries sibling markup (") + label + ")"));
            failures += check(args.at("x</para" "mY") == 7,
                              (std::string("declared sibling with a close marker in its name "
                                           "was lost (") +
                               label + ")"));
            failures += check(
                parsed.diagnostics.schema_mismatch_arguments == 0 &&
                    parsed.diagnostics.fallback_reason ==
                        ninfer::ToolCallParseFallbackReason::None,
                (std::string("declared sibling parse reported a spurious diagnostic (") + label +
                 ")"));

            // The streaming decoder must reach the same terminal result.
            fi::ToolCallOutputDecoder decoder(
                std::make_shared<const fi::ToolCallOutputContract>(contract), 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < text.size(); offset += 7) {
                visible += decoder.feed(std::string_view(text).substr(offset, 7));
            }
            const auto terminal = decoder.finish();
            failures += check(
                visible.empty() && terminal.content.empty() && terminal.tool_calls.size() == 1 &&
                    Json::parse(terminal.tool_calls.front().arguments_json) == args,
                (std::string("chunked declared sibling changed the terminal result (") + label +
                 ")"));
        }
    }
    return failures;
}

int test_parameter_opener_grammar_is_shared() {
    // parse_parameter() and the opaque-string boundary detection must use one grammar:
    // a byte sequence is a parameter opener for both stages or for neither. Prefix
    // collisions ("<parameterX>", "<paramXYZ>") are malformed in the parameter position
    // and are not structural siblings after a literal close.
    const auto contract = contract_for("run", Json{{"command", Json{{"type", "string"}}}});
    const std::string malformed_parameter =
        "<tool_" "call>\n<func" "tion=run>\n<para" "meterX>\n7\n</para" "meter>\n</func" "tion>\n"
        "</tool_" "call>";
    const std::string malformed_short =
        "<tool_" "call>\n<func" "tion=run>\n<paramXYZ>\n7\n</para" "m>\n</func" "tion>\n</tool_"
        "call>";
    const std::string empty_name =
        "<tool_" "call>\n<func" "tion=run>\n<para" "meter=>\n7\n</para" "meter>\n</func" "tion>\n"
        "</tool_" "call>";

    int failures = 0;
    for (const bool tolerant : {false, true}) {
        failures += check_rejected(
            malformed_parameter, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
            tolerant ? "tolerant accepted a prefix-collision parameter opener"
                     : "prefix-collision parameter opener was accepted",
            tolerant);
        failures += check_rejected(
            malformed_short, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
            tolerant ? "tolerant accepted a short prefix-collision parameter opener"
                     : "short prefix-collision parameter opener was accepted",
            tolerant);
        failures += check_rejected(
            empty_name, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
            tolerant ? "tolerant accepted a nameless parameter opener"
                     : "nameless parameter opener was accepted",
            tolerant);
    }

    // Properly delimited forms are parameter openers for both stages: a literal close
    // before such a sibling is the value boundary, so the value is "A" and the sibling
    // is consumed as a real parameter.
    struct SiblingForm {
        std::string_view open;
        std::string_view close;
    };
    const SiblingForm sibling_forms[] = {
        {"<para" "meter=timeout>", "</para" "meter>"},
        {"<para" "meter name=\"timeout\">", "</para" "meter>"},
        {"<param=timeout>", "</para" "m>"},
        {"<param name=\"timeout\">", "</para" "m>"},
    };
    for (const auto& form : sibling_forms) {
        const std::string text =
            "<tool_" "call>\n<func" "tion=run>\n<para" "meter=command>\nA</para" "meter>\n" +
            std::string(form.open) + "\n30\n" + std::string(form.close) +
            "\n</func" "tion>\n</tool_" "call>";
        for (const bool tolerant : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                              "a delimited parameter opener was not recognized as a sibling");
            if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.at("command") == "A" && args.at("timeout") == 30,
                              "delimited sibling form changed the opaque value boundary");
            failures += check(
                parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
                "delimited sibling form reported a spurious fallback reason");
        }
    }

    // Consistency: the same prefix-collision bytes inside a declared string are data for
    // both stages: not a sibling, so the literal close stays value bytes and the bytes
    // are never reinterpreted as a parameter.
    const std::string collision_in_value =
        "<tool_" "call>\n<func" "tion=run>\n<para" "meter=command>\nA</para" "meter>\n"
        "<para" "meterX>\n7\n</para" "meter>\n</func" "tion>\n</tool_" "call>";
    for (const bool tolerant : {false, true}) {
        const auto parsed =
            fi::parse_qwen_tool_call_output(collision_in_value, 64, contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "a prefix-collision opener inside a declared string broke the call");
        if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(
            args.size() == 1 && args.at("command") == "A</para" "meter>\n<para" "meterX>\n7",
            "a prefix-collision opener inside a declared string was reinterpreted");
    }
    return failures;
}

int test_fake_sibling_swallowing_structural_close() {
    // A fake sibling whose opener lacks its own terminator swallows the next structural
    // close into its header. A genuine Qwen parameter header carries no markup, so such
    // a header is a fake sibling: the preceding literal close stays value data and no
    // synthetic parameter is created. This must hold for every structural close marker,
    // not only for a swallowed "</param".
    const auto contract = contract_for(
        "bash", Json{{"command", Json{{"type", "string"}}},
                     {"timeout", Json{{"type", "integer"}}}});
    const std::string_view swallowed_markers[] = {"func" "tion", "invoke", "function_calls",
                                                  "tool_" "call"};
    int failures = 0;
    for (const auto& marker : swallowed_markers) {
        const std::string value =
            std::string("A</para") + "meter><para" "meter=X\nB</" + std::string(marker) +
            ">\nC";
        const std::string text = tool_call("bash", {{"command", value}, {"timeout", "30"}});
        for (const bool tolerant : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                              "a fake sibling swallowing a structural close split the value");
            if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.size() == 2 && args.at("command") == value &&
                                  args.at("timeout") == 30,
                              "a contaminated header derived a synthetic parameter");
            failures += check(
                parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
                "a contaminated fake sibling reported a spurious fallback reason");

            // The streaming decoder must reach the identical terminal result; literal
            // delimiter strings are most likely to be split across transport chunks.
            fi::ToolCallOutputDecoder decoder(
                std::make_shared<const fi::ToolCallOutputContract>(contract), 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < text.size(); offset += 7) {
                visible += decoder.feed(std::string_view(text).substr(offset, 7));
            }
            const auto terminal = decoder.finish();
            failures += check(visible.empty() && terminal.tool_calls.size() == 1 &&
                                  Json::parse(terminal.tool_calls.front().arguments_json) == args,
                              "chunked contaminated-header parsing diverged from one-shot");
        }
    }

    // The same scenario in the short <param> spelling.
    const std::string short_value = "A</para" "m><para" "m=X\nB</func" "tion>\nC";
    const std::string short_text =
        "<tool_" "call>\n<func" "tion=bash>\n<para" "m=command>\n" + short_value +
        "\n</para" "m>\n<para" "m=timeout>\n30\n</para" "m>\n</func" "tion>\n</tool_" "call>";
    for (const bool tolerant : {false, true}) {
        const auto parsed = fi::parse_qwen_tool_call_output(short_text, 64, contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "a short fake sibling swallowing a structural close split the value");
        if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 2 && args.at("command") == short_value &&
                              args.at("timeout") == 30,
                          "a short contaminated header derived a synthetic parameter");
        failures += check(
            parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
            "a short contaminated fake sibling reported a spurious fallback reason");
    }
    return failures;
}

int test_declared_suspicious_name_variants() {
    // The contract exception for markup-contaminated headers covers every declared name
    // that carries markup, not only names containing "</param": full close-looking
    // names and plain nested-markup names stay genuine siblings when declared for the
    // current function.
    const auto contract = contract_for(
        "run", Json{{"command", Json{{"type", "string"}}},
                    {"x</para" "meter", Json{{"type", "integer"}}},
                    {"a<b", Json{{"type", "integer"}}}});
    struct SiblingCase {
        const char* label;
        std::string_view open;
        std::string_view close;
        std::string_view name;
    };
    const SiblingCase cases[] = {
        {"full close-looking name", "<para" "meter=x</para" "meter>", "</para" "meter>",
         "x</para" "meter"},
        {"short close-looking name", "<param=x</para" "meter>", "</para" "m>",
         "x</para" "meter"},
        {"nested markup name", "<para" "meter=a<b>", "</para" "meter>", "a<b"},
        {"short nested markup name", "<param=a<b>", "</para" "m>", "a<b"},
    };
    int failures = 0;
    for (const auto& case_ : cases) {
        const std::string text =
            "<tool_" "call>\n<func" "tion=run>\n<para" "meter=command>\necho ok\n</para" "meter>\n" +
            std::string(case_.open) + "\n7\n" + std::string(case_.close) +
            "\n</func" "tion>\n</tool_" "call>";
        for (const bool tolerant : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                              (std::string("declared suspicious sibling not parsed (") +
                               case_.label + ")"));
            if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(
                args.at("command") == "echo ok" && args.at(std::string(case_.name)) == 7 &&
                    args.size() == 2,
                (std::string("declared suspicious sibling changed the arguments (") +
                 case_.label + ")"));
            failures += check(parsed.diagnostics.schema_mismatch_arguments == 0 &&
                                  parsed.diagnostics.fallback_reason ==
                                      ninfer::ToolCallParseFallbackReason::None,
                              (std::string("declared suspicious sibling reported a spurious "
                                           "diagnostic (") +
                               case_.label + ")"));

            // The streaming decoder must reach the same terminal result.
            fi::ToolCallOutputDecoder decoder(
                std::make_shared<const fi::ToolCallOutputContract>(contract), 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < text.size(); offset += 7) {
                visible += decoder.feed(std::string_view(text).substr(offset, 7));
            }
            const auto terminal = decoder.finish();
            failures += check(visible.empty() && terminal.content.empty() &&
                                  terminal.tool_calls.size() == 1 &&
                                  Json::parse(terminal.tool_calls.front().arguments_json) == args,
                              (std::string("chunked declared suspicious sibling diverged (") +
                               case_.label + ")"));
        }
    }
    return failures;
}

int test_suspicious_header_requires_current_tool_contract() {
    // A contaminated header gains structural authority only from the current function's
    // unambiguous contract. A name declared on another tool must not make the header
    // structural, and conflicting duplicate declarations leave the parser in
    // contract-free legacy semantics where no suspicious header bytes are structural.
    const std::string bash_def = tool_definition(
        "bash", Json{{"command", Json{{"type", "string"}}},
                     {"timeout", Json{{"type", "integer"}}}});
    const std::string other_def = tool_definition(
        "other_tool", Json{{"x</para" "mY", Json{{"type", "integer"}}}});
    const auto cross_contract = contract_from_definitions({bash_def, other_def});

    const std::string value = "A</para" "meter><para" "meter=x</para" "mY>\n7\nC";
    const std::string text  = tool_call("bash", {{"command", value}, {"timeout", "30"}});

    int failures = 0;
    for (const bool tolerant : {false, true}) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, *cross_contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "a foreign tool declaration made a contaminated header structural");
        if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 2 && args.at("command") == value && args.at("timeout") == 30,
                          "a foreign tool declaration changed the opaque value boundary");
        failures += check(parsed.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::None,
                          "the cross-tool case reported a spurious fallback reason");
    }

    // Conflicting duplicates make the tool contract ambiguous, so the declared-string
    // path never runs and no version of the contract can authorize a suspicious header.
    const std::string bash_string_def = tool_definition(
        "bash", Json{{"command", Json{{"type", "string"}}},
                     {"x</para" "mY", Json{{"type", "integer"}}}});
    const std::string bash_integer_def = tool_definition(
        "bash", Json{{"command", Json{{"type", "integer"}}},
                     {"timeout", Json{{"type", "integer"}}}});
    const auto ambiguous = contract_from_definitions({bash_string_def, bash_integer_def});
    const std::string ambiguous_text =
        "<tool_" "call>\n<func" "tion=bash>\n<para" "meter=command>\n"
        "A</para" "meter><para" "meter=x</para" "mY\nB</func" "tion>\nC\n"
        "</para" "meter>\n</func" "tion>\n</tool_" "call>";
    const auto parsed_ambiguous =
        fi::parse_qwen_tool_call_output(ambiguous_text, 64, *ambiguous);
    failures += check(parsed_ambiguous.is_tool_call_response &&
                          parsed_ambiguous.tool_calls.size() == 1,
                      "an ambiguous contract demoted a legacy call to text");
    if (parsed_ambiguous.is_tool_call_response && parsed_ambiguous.tool_calls.size() == 1) {
        Json expected = Json::object();
        expected["command"]                     = "A";
        expected["x</para" "mY\nB</func" "tion"] = "C";
        failures += check(
            Json::parse(parsed_ambiguous.tool_calls.front().arguments_json) == expected,
            "an ambiguous contract granted structural authority to a contaminated header");
    }
    return failures;
}

int test_literal_markup_in_declared_string_arguments() {
    // The real-world failure that exposed this bug class: a coding agent emits literal
    // tool-call markup (a fake sibling whose unterminated opener swallowed a function
    // close) inside a declared string argument. The embedded markers must remain value
    // bytes: no early termination, no synthetic parameter, no synthetic boundary, no
    // fallback reason, no demotion to text. Verified for the write.content and
    // bash.command shapes, in strict and tolerant mode, and in the streaming decoder.
    const std::string literal = "A</para" "meter><para" "meter=X\nB</func" "tion>\nC";
    const auto write_contract = contract_from_definitions({
        tool_definition("write", Json{{"content", Json{{"type", "string"}}},
                                      {"path", Json{{"type", "string"}}}})});
    const auto bash_contract = contract_from_definitions({
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}},
                                   {"timeout", Json{{"type", "integer"}}}})});
    const std::string write_call =
        tool_call("write", {{"content", literal}, {"path", "test.cpp"}});
    const std::string bash_call = tool_call("bash", {{"command", literal}, {"timeout", "30"}});

    int failures = 0;
    for (const bool tolerant : {false, true}) {
        const auto write_parsed =
            fi::parse_qwen_tool_call_output(write_call, 64, *write_contract, tolerant);
        failures += check(write_parsed.is_tool_call_response && write_parsed.tool_calls.size() == 1 &&
                              write_parsed.tool_calls.front().name == "write",
                          "literal markup in write.content demoted the call to text");
        if (write_parsed.is_tool_call_response && write_parsed.tool_calls.size() == 1) {
            const Json args = Json::parse(write_parsed.tool_calls.front().arguments_json);
            failures += check(args.size() == 2 && args.at("content") == literal &&
                                  args.at("path") == "test.cpp",
                              "literal markup in write.content changed the arguments");
            failures += check(write_parsed.diagnostics.fallback_reason ==
                                  ninfer::ToolCallParseFallbackReason::None,
                              "literal markup in write.content reported a fallback reason");

            fi::ToolCallOutputDecoder write_decoder(
                std::make_shared<const fi::ToolCallOutputContract>(*write_contract), 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < write_call.size(); offset += 7) {
                visible += write_decoder.feed(std::string_view(write_call).substr(offset, 7));
            }
            const auto terminal = write_decoder.finish();
            failures += check(visible.empty() && terminal.content.empty() &&
                                  terminal.tool_calls.size() == 1 &&
                                  Json::parse(terminal.tool_calls.front().arguments_json) == args,
                              "chunked write.content literal markup diverged from one-shot");
        }

        const auto bash_parsed =
            fi::parse_qwen_tool_call_output(bash_call, 64, *bash_contract, tolerant);
        failures += check(bash_parsed.is_tool_call_response && bash_parsed.tool_calls.size() == 1 &&
                              bash_parsed.tool_calls.front().name == "bash",
                          "literal markup in bash.command demoted the call to text");
        if (bash_parsed.is_tool_call_response && bash_parsed.tool_calls.size() == 1) {
            const Json args = Json::parse(bash_parsed.tool_calls.front().arguments_json);
            failures += check(args.size() == 2 && args.at("command") == literal &&
                                  args.at("timeout") == 30,
                              "literal markup in bash.command created a synthetic parameter");
            failures += check(bash_parsed.diagnostics.fallback_reason ==
                                  ninfer::ToolCallParseFallbackReason::None,
                              "literal markup in bash.command reported a fallback reason");

            fi::ToolCallOutputDecoder bash_decoder(
                std::make_shared<const fi::ToolCallOutputContract>(*bash_contract), 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < bash_call.size(); offset += 7) {
                visible += bash_decoder.feed(std::string_view(bash_call).substr(offset, 7));
            }
            const auto terminal = bash_decoder.finish();
            failures += check(visible.empty() && terminal.content.empty() &&
                                  terminal.tool_calls.size() == 1 &&
                                  Json::parse(terminal.tool_calls.front().arguments_json) == args,
                              "chunked bash.command literal markup diverged from one-shot");
        }
    }
    return failures;
}

int test_fake_sibling_borrowing_ordinary_greater_sign() {
    // A fake parameter opener inside a declared string argument can find its first ">" in
    // ordinary source data - a comparison operator, a shell redirection, more code after a
    // newline - instead of in structural markup. Its extracted header therefore carries no
    // "<" and the markup-contamination heuristic does not trigger; the extracted name is
    // still not a plausible ordinary parameter identifier, so the candidate must not
    // terminate the opaque string. Verified in strict and tolerant mode and in the
    // streaming decoder, which must equal one-shot parsing.
    const auto bash_contract = contract_from_definitions({
        tool_definition("bash", Json{{"command", Json{{"type", "string"}}},
                                   {"timeout", Json{{"type", "integer"}}}})});
    int failures = 0;
    const auto check_fake = [&](const char* label, const char* literal) {
        const std::string value(literal);
        for (const bool tolerant : {false, true}) {
            const std::string call = tool_call("bash", {{"command", value}, {"timeout", "30"}});
            const auto parsed =
                fi::parse_qwen_tool_call_output(call, 64, *bash_contract, tolerant);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                                  parsed.tool_calls.front().name == "bash",
                              (std::string("fake sibling borrowing a '>(") + label +
                               ") demoted the call to text"));
            if (parsed.is_tool_call_response && parsed.tool_calls.size() == 1) {
                const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
                failures += check(args.size() == 2 && args.at("command") == value &&
                                      args.at("timeout") == 30,
                                  (std::string("fake sibling borrowing a '>(") + label +
                                   ") created a synthetic parameter"));
                failures += check(parsed.diagnostics.schema_mismatch_arguments == 0 &&
                                      parsed.diagnostics.fallback_reason ==
                                          ninfer::ToolCallParseFallbackReason::None,
                                  (std::string("fake sibling borrowing a '>(") + label +
                                   ") reported a spurious diagnostic"));

                fi::ToolCallOutputDecoder decoder(std::make_shared<const fi::ToolCallOutputContract>(
                                                      *bash_contract),
                                                  64, tolerant);
                std::string visible;
                for (std::size_t offset = 0; offset < call.size(); offset += 7) {
                    visible += decoder.feed(std::string_view(call).substr(offset, 7));
                }
                const auto terminal = decoder.finish();
                failures += check(visible.empty() && terminal.content.empty() &&
                                      terminal.tool_calls.size() == 1 &&
                                      Json::parse(terminal.tool_calls.front().arguments_json) == args,
                                  (std::string("chunked fake sibling borrowing a '>(") + label +
                                   ") diverged from one-shot"));
            }
        }
    };
    check_fake("comparison operator", "A</para" "meter><para" "meter=X\nif (a > b) C");
    check_fake("shell redirection", "A</para" "meter><para" "meter=X\necho hello > output.txt");
    check_fake("newline and space", "A</para" "meter><para" "meter=X\nsomething > rest");
    check_fake("tab", "A</para" "meter><para" "meter=X\tsomething > rest");

    // An ordinary undeclared parameter keeps its current behavior: a simple undeclared
    // name after a literal close still terminates the opaque string and remains
    // structured with its schema mismatch.
    {
        const std::string call =
            tool_call("bash", {{"command", "ls -la"}, {"extra", "value"}, {"timeout", "30"}});
        const auto parsed = fi::parse_qwen_tool_call_output(call, 64, *bash_contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.diagnostics.schema_mismatch_arguments == 1 &&
                              parsed.diagnostics.fallback_reason ==
                                  ninfer::ToolCallParseFallbackReason::None,
                          "ordinary undeclared parameter stopped terminating the opaque string");
        if (parsed.tool_calls.size() == 1) {
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.size() == 3 && args.at("command") == "ls -la" &&
                                  args.at("extra") == "value" && args.at("timeout") == 30,
                              "ordinary undeclared parameter arguments changed");
        }
    }

    // A genuinely declared unusual name keeps structural authority through the current
    // unambiguous tool contract, even though it carries markup.
    {
        const auto odd_contract = contract_from_definitions({
            tool_definition("bash", Json{{"command", Json{{"type", "string"}}},
                                       {"a<b", Json{{"type", "string"}}}})});
        const std::string call =
            tool_call("bash", {{"command", "A</para" "meter><para" "meter=a<b>\nreal"}});
        const auto parsed = fi::parse_qwen_tool_call_output(call, 64, *odd_contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.diagnostics.schema_mismatch_arguments == 0 &&
                              parsed.diagnostics.fallback_reason ==
                                  ninfer::ToolCallParseFallbackReason::None,
                          "declared markup-bearing name lost its structural authority");
        if (parsed.tool_calls.size() == 1) {
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.size() == 2 && args.at("command") == "A" && args.at("a<b") == "real",
                              "declared markup-bearing name changed the arguments");
        }
    }
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
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.tool_calls.front().name == "bash",
                      "a quoted marker before the real call demoted the structured turn");
    failures += check(parsed.content == "explaining " + quoted + " then the real turn",
                      "quoted marker or intervening prose was not retained as content");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("command") == "echo ok", "recovered call arguments changed");
    }
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
    failures += check(terminal.tool_calls.size() == 1 && terminal.tool_calls.front().name == "bash",
                      "incremental quoted marker hid the real tool call");
    failures += check(visible + terminal.content == "explaining " + quoted + " then the real turn",
                      "incremental quoted marker lost or duplicated bytes");
    failures +=
        check(terminal.diagnostics.marker_seen && terminal.diagnostics.structured_call_count == 1 &&
                  terminal.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
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
    const std::string command =
        "pattern='<parameter=inner>value</parameter>'\n"
        "partial='<parameter=unterminated>'\n"
        "printf '%s %s' \"$pattern\" \"$partial\"";
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
        failures += check(parsed.is_tool_call_response,
                          std::string("tolerant did not recover ") + label);
        failures += check(parsed.tool_calls.size() == 1,
                          std::string("tolerant recovered wrong call count for ") + label);
        if (parsed.tool_calls.size() == 1) {
            failures += check(parsed.tool_calls.front().name == "delete_file",
                              std::string("tolerant lost call name for ") + label);
            const Json arguments = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(arguments == Json{{"filePath", "/tmp/out.js"}},
                              std::string("tolerant lost arguments for ") + label);
        }
        failures += check(parsed.diagnostics.fallback_reason == Reason::TruncatedTail,
                          std::string("tolerant did not flag ") + label + " as truncated tail");
        const auto strict = fi::parse_qwen_tool_call_output(text, 64, *contract);
        failures += check(!strict.is_tool_call_response,
                          std::string("strict recovered a truncated final call: ") + label);
        failures += check(strict.tool_calls.empty(),
                          std::string("strict retained a truncated final call: ") + label);
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
    const auto tolerant = fi::parse_qwen_tool_call_output(undeclared, 64, *contract, true);
    failures += check(tolerant.is_tool_call_response && tolerant.tool_calls.size() == 1 &&
                          tolerant.tool_calls.front().name == "not_a_declared_tool",
                      "tolerant mode did not keep the undeclared-name call structured");
    failures += check(tolerant.diagnostics.fallback_reason == Reason::None,
                      "tolerant undeclared call reported a fallback reason");
    const auto strict = fi::parse_qwen_tool_call_output(undeclared, 64, *contract);
    failures += check(!strict.is_tool_call_response &&
                          strict.diagnostics.fallback_reason == Reason::UndeclaredTool,
                      "strict mode did not reject the undeclared-name call");

    const std::string value_cut = "Now\n"
                                  "<tool_call>\n"
                                  "<function=delete_file>\n"
                                  + open_tag + "/tmp/out";
    const auto cut_tolerant = fi::parse_qwen_tool_call_output(value_cut, 64, *contract, true);
    failures += check(cut_tolerant.is_tool_call_response && cut_tolerant.tool_calls.size() == 1 &&
                          cut_tolerant.tool_calls.front().name == "delete_file",
                      "tolerant mode did not keep the value-cut call");
    if (cut_tolerant.tool_calls.size() == 1) {
        const Json args = Json::parse(cut_tolerant.tool_calls.front().arguments_json);
        failures += check(args.at("filePath").get<std::string>() == "/tmp/out",
                          "tolerant mode lost the partial value-cut argument");
    }
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

int test_parameter_header_grammar_hardening() {
    // The header payload is validated as a whole, quote-aware: a fake opener whose '>'
    // lands in value data yields a payload that is not a valid header (trailing source
    // code, a name that is not a plausible identifier), so it can never become a
    // structural sibling. The real attribute forms - both quote kinds, unquoted values,
    // unknown attributes carrying a quoted '>' - are structural siblings.
    const auto contract = contract_for("run", Json{{"command", Json{{"type", "string"}}},
                                                      {"timeout", Json{{"type", "integer"}}}});
    int failures = 0;

    struct SiblingForm {
        std::string_view open;
        std::string_view close;
    };
    const SiblingForm forms[] = {
        {"<para" "meter name='timeout'>", "</para" "meter>"},
        {"<para" "meter name=timeout>", "</para" "meter>"},
        {"<para" "meter filename=\"a>b\" name=\"timeout\">", "</para" "meter>"},
        {"<para" "m name=\"timeout\">", "</para" "m>"},
        {"<para" "m name=timeout>", "</para" "m>"},
    };
    for (const auto& form : forms) {
        const std::string text =
            "<tool_" "call>\n<func" "tion=run>\n<para" "meter=command>\nA</para" "meter>\n" +
            std::string(form.open) + "\n30\n" + std::string(form.close) +
            "\n</func" "tion>\n</tool_" "call>";
        for (const bool tolerant : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                              (tolerant ? "tolerant " : "strict ") +
                                  std::string("rejected an attribute-form sibling: ") +
                                  std::string(form.open));
            if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.at("command") == "A" && args.at("timeout") == 30,
                              (tolerant ? "tolerant " : "strict ") +
                                  std::string("attribute-form sibling changed the value split: ") +
                                  std::string(form.open));
            failures += check(
                parsed.diagnostics.fallback_reason ==
                        ninfer::ToolCallParseFallbackReason::None &&
                    parsed.diagnostics.schema_mismatch_arguments == 0,
                (tolerant ? "tolerant " : "strict ") +
                    std::string("attribute-form sibling reported a spurious diagnostic: ") +
                    std::string(form.open));
            fi::ToolCallOutputDecoder decoder(
                std::make_shared<const fi::ToolCallOutputContract>(contract), 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < text.size(); offset += 7) {
                visible += decoder.feed(std::string_view(text).substr(offset, 7));
            }
            const auto terminal = decoder.finish();
            failures += check(
                visible.empty() && terminal.content.empty() &&
                    terminal.tool_calls.size() == 1 &&
                    Json::parse(terminal.tool_calls.front().arguments_json).at("command") == "A" &&
                    Json::parse(terminal.tool_calls.front().arguments_json).at("timeout") == 30,
                (tolerant ? "tolerant " : "strict ") +
                    std::string("chunked attribute-form sibling diverged: ") +
                    std::string(form.open));
        }
    }

    // Negative controls: a fake sibling after a literal close. The header payload is not
    // a valid header, so the literal close is not a boundary and the value stays whole.
    const std::string fake_trailing_junk = "A</para" "meter><para" "meter name=\"X\"\nif (a > b) C";
    const std::string fake_quoted_greater =
        "A</para" "meter><para" "meter name=\"if (a > b)\">junk";
    for (const auto& [label, value] :
         std::vector<std::pair<std::string, std::string>>{
             {"trailing source code after a quoted attribute", fake_trailing_junk},
             {"quoted attribute value containing a greater sign", fake_quoted_greater}}) {
        const std::string text = "<tool_" "call>\n<func" "tion=run>\n<para" "meter=command>\n" +
                                 value + "</para" "meter>\n</func" "tion>\n</tool_" "call>";
        for (const bool tolerant : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                              (tolerant ? "tolerant " : "strict ") +
                                  std::string("reinterpreted a fake sibling (") + label + ")");
            if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.size() == 1 && args.at("command") == value,
                              (tolerant ? "tolerant " : "strict ") + std::string("value lost at a fake sibling (") +
                                  label + ")");
            failures += check(
                parsed.diagnostics.fallback_reason ==
                        ninfer::ToolCallParseFallbackReason::None &&
                    parsed.diagnostics.schema_mismatch_arguments == 0,
                (tolerant ? "tolerant " : "strict ") +
                    std::string("fake sibling reported a spurious diagnostic (") + label + ")");
            fi::ToolCallOutputDecoder decoder(
                std::make_shared<const fi::ToolCallOutputContract>(contract), 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < text.size(); offset += 7) {
                visible += decoder.feed(std::string_view(text).substr(offset, 7));
            }
            const auto terminal = decoder.finish();
            failures += check(
                visible.empty() && terminal.content.empty() &&
                    terminal.tool_calls.size() == 1 &&
                    Json::parse(terminal.tool_calls.front().arguments_json).at("command") == value,
                (tolerant ? "tolerant " : "strict ") + std::string("chunked fake sibling diverged (") +
                    label + ")");
        }
    }

    // At a real parameter position the same payloads are malformed headers in both modes.
    failures += check_rejected(
        "<tool_" "call>\n<func" "tion=run>\n<para"
        "meter name=\"timeout\" junk>\n7\n</para" "meter>\n</func" "tion>\n</tool_" "call>",
        contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
        "attribute header with trailing junk was accepted in strict mode");
    failures += check_rejected(
        "<tool_" "call>\n<func" "tion=run>\n<para"
        "meter name=\"X\"\nif (a > b) C>\n7\n</para" "meter>\n</func" "tion>\n</tool_" "call>",
        contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
        "attribute header with trailing junk was accepted in strict mode");
    const std::string junk_real =
        "<tool_" "call>\n<func" "tion=run>\n<para"
        "meter name=\"timeout\" junk>\n7\n</para" "meter>\n</func" "tion>\n</tool_" "call>";
    const auto junk_tol = fi::parse_qwen_tool_call_output(junk_real, 64, contract, /*tolerant*/ true);
    failures += check(!junk_tol.is_tool_call_response &&
                          junk_tol.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::MalformedStructure,
                      "tolerant mode accepted an attribute header with trailing junk");
    return failures;
}

int test_function_opener_grammar_hardening() {
    // The byte after "<function" or "<invoke>" must delimit the prefix: a longer identifier
    // such as "<functionbash>" is not a function opener in either mode, and a synthetic
    // second call built from a prefix-collision opener cannot appear inside a value.
    const auto contract = contract_for("run", Json{{"command", Json{{"type", "string"}}}});
    int failures = 0;

    const std::string collision = "<tool_" "call>\n<function"
                                   "bash>\n<para" "meter=command>\necho hi\n</para" "meter>\n</func"
                                   "tion>\n</tool_" "call>";
    const std::string collision_invoke = "<function" "_calls>\n<invoke"
                                         "bash>\n<para" "meter=command>\necho hi\n</para" "meter>\n</inv" "oke>\n</function_calls>";
    for (const bool tolerant : {false, true}) {
        failures += check_rejected(
            collision, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
            tolerant ? "tolerant accepted a function prefix collision"
                     : "function prefix collision was accepted",
            tolerant);
        failures += check_rejected(
            collision_invoke, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
            tolerant ? "tolerant accepted an invoke prefix collision"
                     : "invoke prefix collision was accepted",
            tolerant);
    }

    failures += check_rejected("<function" ">\n</function>", contract,
                               ninfer::ToolCallParseFallbackReason::InvalidToolName,
                               "empty function name was accepted");
    failures += check_rejected(
        "<function name=\"bad.name\">\n</function>", contract,
        ninfer::ToolCallParseFallbackReason::InvalidToolName,
        "attribute form with an invalid function name was accepted");

    // A prefix-collision opener inside a declared string value: no synthetic second tool
    // call may appear, and the value keeps its bytes (the literal close is not a boundary
    // because the region continues with a real wrapper close).
    const std::string value = "A</para" "meter></func"
                              "tion><function" "bash><para" "meter=command>echo hi";
    const std::string text = "<tool_" "call>\n<func" "tion=run>\n<para" "meter=command>\n" +
                             value + "\n</para" "meter>\n</func" "tion>\n</tool_" "call>";
    for (const bool tolerant : {false, true}) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().name == "run",
                          (tolerant ? "tolerant " : "strict ") +
                              std::string("prefix-collision value produced the wrong calls"));
        if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 1 && args.at("command") == value,
                          (tolerant ? "tolerant " : "strict ") +
                              std::string("prefix-collision value bytes were lost"));
        failures += check(
            parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
            (tolerant ? "tolerant " : "strict ") +
                std::string("prefix-collision value reported a spurious reason"));
        fi::ToolCallOutputDecoder decoder(
            std::make_shared<const fi::ToolCallOutputContract>(contract), 64, tolerant);
        std::string visible;
        for (std::size_t offset = 0; offset < text.size(); offset += 7) {
            visible += decoder.feed(std::string_view(text).substr(offset, 7));
        }
        const auto terminal = decoder.finish();
        failures += check(visible.empty() && terminal.content.empty() &&
                              terminal.tool_calls.size() == 1 &&
                              Json::parse(terminal.tool_calls.front().arguments_json)
                                  .at("command") == value,
                          (tolerant ? "tolerant " : "strict ") +
                              std::string("chunked prefix-collision value diverged"));
    }

    // Real structure still works: a valid function header continues a function_calls
    // container, and a complete wrapper close continues a tool_call region.
    const std::string two_calls = "<function"
                                  "_calls>\n<func" "tion=run>\n<para" "meter=command>\nA</para"
                                  "meter>\n</func" "tion>\n<func" "tion=run>\n<para"
                                  "meter=command>\nB</para" "meter>\n</func"
                                  "tion>\n</function_calls>";
    for (const bool tolerant : {false, true}) {
        const auto parsed = fi::parse_qwen_tool_call_output(two_calls, 64, contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2,
                          (tolerant ? "tolerant " : "strict ") +
                              std::string("valid function successor stopped the container"));
        if (parsed.is_tool_call_response && parsed.tool_calls.size() == 2) {
            const Json a = Json::parse(parsed.tool_calls[0].arguments_json);
            const Json b = Json::parse(parsed.tool_calls[1].arguments_json);
            failures += check(a.at("command") == "A" && b.at("command") == "B",
                              (tolerant ? "tolerant " : "strict ") +
                                  std::string("valid function successor changed the values"));
        }
    }
    return failures;
}

int test_global_continuation_after_wrapper_close() {
    // A complete fake closing boundary inside value data - "</parameter>", "</function>"
    // and a "</tool_call>" - is a boundary only when the wrapper close is a true
    // continuation: EOF after whitespace, a valid top-level construct, or prose carrying
    // no further "</tool_call>". A later wrapper close that the following bytes do not
    // belong to keeps the fake boundary inside the value.
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    int failures = 0;

    const std::string value = "A</para" "meter></func" "tion>\n</tool_" "call>\nB";
    const std::string text = "<tool_" "call>\n<func" "tion=bash>\n<para" "meter=command>\n" +
                             value + "\n</para" "meter>\n</func" "tion>\n</tool_" "call>";
    for (const bool tolerant : {false, true}) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().name == "bash",
                          (tolerant ? "tolerant " : "strict ") +
                              std::string("fake wrapper close split the value"));
        if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 1 && args.at("command") == value,
                          (tolerant ? "tolerant " : "strict ") +
                              std::string("value lost at a fake wrapper close"));
        failures += check(
            parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
            (tolerant ? "tolerant " : "strict ") +
                std::string("fake wrapper close reported a spurious reason"));
        fi::ToolCallOutputDecoder decoder(
            std::make_shared<const fi::ToolCallOutputContract>(contract), 64, tolerant);
        std::string visible;
        for (std::size_t offset = 0; offset < text.size(); offset += 7) {
            visible += decoder.feed(std::string_view(text).substr(offset, 7));
        }
        const auto terminal = decoder.finish();
        failures += check(visible.empty() && terminal.content.empty() &&
                              terminal.tool_calls.size() == 1 &&
                              Json::parse(terminal.tool_calls.front().arguments_json)
                                  .at("command") == value,
                          (tolerant ? "tolerant " : "strict ") +
                              std::string("chunked fake wrapper close diverged"));
    }
    return failures;
}
// The fixture appends a newline before "</parameter>"; declared string parameters then have
// their framing newlines removed exactly as remove_parameter_framing_newlines() does.
std::string expected_value_for(const std::string& value) {
    std::string raw = "\n" + value + "\n";
    std::size_t begin = 0;
    std::size_t end   = raw.size();
    if (raw.size() >= 2 && raw[0] == '\r' && raw[1] == '\n') {
        begin = 2;
    } else if (!raw.empty() && raw[0] == '\n') {
        begin = 1;
    }
    if (end >= begin + 2 && raw[end - 2] == '\r' && raw[end - 1] == '\n') {
        end -= 2;
    } else if (end > begin && raw[end - 1] == '\n') {
        --end;
    }
    return raw.substr(begin, end - begin);
}

// One declared string parameter carrying `value`. Asserts that strict one-shot, tolerant
// one-shot, and streaming (chunk 7, plus 1/2/3 for key fixtures) all preserve the exact
// string: one tool call, the right name, the exact argument bytes, no fallback, no schema
// mismatch, and no synthetic call or parameter.
int check_exact_string_round_trip(std::string_view label, const std::string& tool,
                                  const std::string& param, const std::string& value,
                                  bool key_fixture) {
    const auto contract = contract_for(tool, Json{{param, Json{{"type", "string"}}}});
    const std::string text = "<tool_" "call>\n<func" "tion=" + tool + ">\n<para" "meter=" +
                             param + ">\n" + value + "\n</para" "meter>\n</func" "tion>\n</tool_"
                             "call>";
    const std::string expected = expected_value_for(value);
    int failures = 0;
    for (const bool tolerant : {false, true}) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, tolerant);
        failures += check(
            parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                parsed.tool_calls.front().name == tool &&
                parsed.content.empty() &&
                Json::parse(parsed.tool_calls.front().arguments_json).at(param) == expected &&
                parsed.diagnostics.fallback_reason ==
                    ninfer::ToolCallParseFallbackReason::None &&
                parsed.diagnostics.schema_mismatch_arguments == 0,
            std::string(label) + (tolerant ? ": tolerant one-shot did not round-trip"
                                           : ": strict one-shot did not round-trip"));
        if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { continue; }

        const std::vector<std::size_t> chunks =
            key_fixture ? std::vector<std::size_t>{1, 2, 3, 7} : std::vector<std::size_t>{7};
        for (const std::size_t chunk : chunks) {
            fi::ToolCallOutputDecoder decoder(
                std::make_shared<const fi::ToolCallOutputContract>(contract), 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < text.size(); offset += chunk) {
                visible += decoder.feed(std::string_view(text).substr(offset, chunk));
            }
            const auto terminal = decoder.finish();
            failures += check(
                visible.empty() && terminal.content.empty() &&
                    terminal.tool_calls.size() == 1 &&
                    terminal.tool_calls.front().name == tool &&
                    Json::parse(terminal.tool_calls.front().arguments_json).at(param) ==
                        expected &&
                    terminal.diagnostics.fallback_reason ==
                        ninfer::ToolCallParseFallbackReason::None,
                std::string(label) + (tolerant ? ": tolerant streaming(" + std::to_string(chunk)
                                               : ": strict streaming(" + std::to_string(chunk)) +
                    ") did not round-trip");
        }
    }
    return failures;
}

int test_exact_string_round_trip_matrix() {
    // The mandated fixtures (section 32 negative control, section 57 synthetic second tool,
    // section 78 global continuation) plus literal greater-sign and quote cases.
    int failures = 0;
    failures += check_exact_string_round_trip(
        "sec32", "run", "command",
        "A</para" "meter><para" "meter name=\"X\"\nif (a > b) C", /*key_fixture*/ true);
    failures += check_exact_string_round_trip(
        "sec57", "run", "command",
        "A</para" "meter></func" "tion><function" "bash><para" "meter=command>echo hi",
        /*key_fixture*/ true);
    failures += check_exact_string_round_trip(
        "sec78", "run", "command",
        "A</para" "meter></func" "tion>\n</tool_" "call>\nB", /*key_fixture*/ true);
    failures += check_exact_string_round_trip(
        "quoted_greater", "run", "command", "echo \"a > b\" && x < y", /*key_fixture*/ true);
    failures += check_exact_string_round_trip(
        "single_quoted", "run", "command", "printf 'a > b; c < d'", /*key_fixture*/ true);
    failures += check_exact_string_round_trip(
        "unterminated_fake_header", "run", "command",
        "A</para" "meter><para" "meter name=\"X\"", /*key_fixture*/ true);

    // Deterministic property matrix: ordered fragment pairs that must stay value data. Every
    // fragment is value-inert, so no pair forms a genuine structural boundary; the value must
    // round-trip exactly in both modes and through streaming.
    const std::string fragments[] = {
        "A",
        " > ",
        "a > b",
        "if (a > b) C",
        "<para" "meter name=\"X\"",
        "</para" "meter>",
        "</func" "tion>",
        "<function" "bash>",
        "<invoke" "bash>",
        "\"a > b\"",
        "echo hi",
        "\n",
    };
    for (const auto& a : fragments) {
        for (const auto& b : fragments) {
            const std::string value = a + b;
            if (expected_value_for(value).empty()) { continue; }
            failures += check_exact_string_round_trip(
                ("matrix:" + a + "+" + b).c_str(), "run", "command", value,
                /*key_fixture*/ false);
        }
    }
    return failures;
}



int test_canonical_grammar_findings() {
    // The six findings, pinned against the shared-grammar implementation:
    //  F1   name validity is mode-independent in the structural successor lookahead, so a
    //       "<function=bad.name>" after a literal close never truncates a declared string
    //       value in tolerant mode; the value ends at the next structural boundary instead.
    //  F1b  tolerant-recovered openers (missing '>', doubled '<') are structural in the
    //       lookahead exactly where the consumer recovers them; a recovered opener that
    //       still carries a quoted '>' and a bare name is no header at all.
    //  F2   format whitespace may separate an attribute name from its '=' in function,
    //       invoke, parameter and param headers.
    //  F2b  an attribute value must be followed by whitespace or end-of-header; adjacent
    //       attributes are malformed.
    //  F2c  an unquoted attribute value may not be empty; a quoted one may.
    //  F3   tab, CR and LF delimit the function/invoke prefix in one-shot and streaming.
    using Reason = ninfer::ToolCallParseFallbackReason;
    int failures = 0;
    const auto tc = contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    const auto run =
        contract_for("run", Json{{"command", Json{{"type", "string"}}},
                                 {"timeout", Json{{"type", "integer"}}}});
    const auto write = contract_for("write", Json{{"content", Json{{"type", "string"}}},
                                                   {"path", Json{{"type", "string"}}}});

    // --- F1: invalid-name function after a literal close inside a declared string value.
    const std::string f1 = "<func" "tion_calls>\n<func" "tion=run>\n<para"
                           "meter=command>\nA</para" "meter></func" "tion><func"
                           "tion=bad.name><para" "meter=command>echo hi\n</para"
                           "meter>\n</func" "tion>\n</func" "tion_calls>";
    const std::string f1_value = "A</para" "meter></func" "tion><func"
                                 "tion=bad.name><para" "meter=command>echo hi";
    for (const bool tolerant : {false, true}) {
        const auto parsed = fi::parse_qwen_tool_call_output(f1, 64, run, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              Json::parse(parsed.tool_calls.front().arguments_json)
                                      .at("command") == f1_value &&
                              parsed.diagnostics.fallback_reason == Reason::None,
                          std::string(tolerant ? "tolerant " : "strict ") +
                              "F1 did not keep the value up to the next structural boundary");
        const std::shared_ptr<const fi::ToolCallOutputContract> shared =
            std::make_shared<const fi::ToolCallOutputContract>(run);
        for (const std::size_t chunk : {2u, 3u, 7u}) {
            fi::ToolCallOutputDecoder decoder(shared, 64, tolerant);
            std::string visible;
            for (std::size_t offset = 0; offset < f1.size(); offset += chunk) {
                visible += decoder.feed(std::string_view(f1).substr(offset, chunk));
            }
            const auto terminal = decoder.finish();
            failures += check(visible.empty() && terminal.content.empty() &&
                                  terminal.tool_calls.size() == 1 &&
                                  Json::parse(terminal.tool_calls.front().arguments_json)
                                          .at("command") == f1_value &&
                                  terminal.diagnostics.fallback_reason == Reason::None,
                              std::string(tolerant ? "tolerant " : "strict ") +
                                  "F1 streaming diverged from the one-shot parse");
        }
    }

    // --- F1b: tolerant-recovered opener as structural successor; the quoted-'> form is not
    // a header.
    const std::string recb = "<func" "tion_calls>\n<func" "tion=run>\n<para"
                             "meter=command>\nX</para" "meter></func" "tion>\n<<func"
                             "tion=second\n<para" "meter=command>\nB\n</para" "meter>\n</func"
                             "tion>\n</func" "tion_calls>";
    const auto recb_tol = fi::parse_qwen_tool_call_output(recb, 64, run, /*tolerant*/ true);
    failures += check(
        recb_tol.is_tool_call_response && recb_tol.tool_calls.size() == 2 &&
            recb_tol.tool_calls.front().name == "run" &&
            Json::parse(recb_tol.tool_calls.front().arguments_json).at("command") == "X" &&
            recb_tol.tool_calls.back().name == "second" &&
            Json::parse(recb_tol.tool_calls.back().arguments_json).at("command") == "B" &&
            recb_tol.diagnostics.fallback_reason == Reason::None,
        "F1b tolerant did not recover the missing-'> successor as a second call");
    const std::string recb_value = "X</para" "meter></func" "tion>\n<<func"
                                   "tion=second\n<para" "meter=command>\nB";
    const auto recb_strict = fi::parse_qwen_tool_call_output(recb, 64, run);
    failures += check(recb_strict.is_tool_call_response && recb_strict.tool_calls.size() == 1 &&
                          Json::parse(recb_strict.tool_calls.front().arguments_json)
                                  .at("command") == recb_value &&
                          recb_strict.diagnostics.fallback_reason == Reason::None,
                      "F1b strict did not keep the recovered opener inside the value");

    const std::string recc = "<tool_" "call>\n<<func" "tion=write>\n<para"
                             "meter=content>\nx\n</para" "meter>\n</func" "tion>\n</tool_"
                             "call>";
    failures += check_rejected(recc, write, Reason::InvalidToolName,
                               "F1b tolerant accepted a quoted-'> recovered opener",
                               /*tolerant*/ true);
    failures += check_rejected(recc, write, Reason::MalformedStructure,
                               "F1b strict accepted a quoted-'> recovered opener",
                               /*tolerant*/ false);

    const std::string recc2 = "<tool_" "call>\n<<func" "tion=write\n<para"
                              "meter=content>\nx\n</para" "meter>\n</func" "tion>\n</tool_"
                              "call>";
    const auto recc2_tol = fi::parse_qwen_tool_call_output(recc2, 64, write, /*tolerant*/ true);
    failures += check(recc2_tol.is_tool_call_response && recc2_tol.tool_calls.size() == 1 &&
                          recc2_tol.tool_calls.front().name == "write" &&
                          Json::parse(recc2_tol.tool_calls.front().arguments_json)
                                  .at("content") == "x" &&
                          recc2_tol.diagnostics.fallback_reason == Reason::None,
                      "F1b tolerant did not recover the missing-'> opener in a wrapper");
    failures += check_rejected(recc2, write, Reason::MalformedStructure,
                               "F1b strict recovered an opener it cannot consume",
                               /*tolerant*/ false);

    // --- F2: whitespace around '=' in attribute headers.
    const std::string f2_function = "<tool_" "call>\n<func" "tion name = \"TaskCreate\">\n<para"
                                    "meter name=\"description\">\nhi\n</para" "meter>\n</func"
                                    "tion>\n</tool_" "call>";
    const std::string f2_parameter = "<tool_" "call>\n<func" "tion=TaskCreate>\n<para"
                                     "meter name = \"description\">\nhi\n</para" "meter>\n</func"
                                     "tion>\n</tool_" "call>";
    const std::string f2_invoke = "<tool_" "call>\n<in" "voke name = \"TaskCreate\">\n<para"
                                  "m name = \"description\">\nhi\n</para" "m>\n</in"
                                  "voke>\n</tool_" "call>";
    for (const auto& [label, text] :
         std::vector<std::pair<std::string, std::string>>{
             {"function", f2_function}, {"parameter", f2_parameter}, {"invoke", f2_invoke}}) {
        for (const bool tolerant : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, tc, tolerant);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                                  Json::parse(parsed.tool_calls.front().arguments_json)
                                          .at("description") == "hi" &&
                                  parsed.diagnostics.fallback_reason == Reason::None,
                              std::string(tolerant ? "tolerant " : "strict ") +
                                  "F2 rejected " + label + " whitespace around '='");
        }
    }

    // --- F2b: an attribute value must be separated from the next attribute.
    const std::string f2b_parameter = "<tool_" "call>\n<func" "tion=TaskCreate>\n<para"
                                      "meter name=\"description\"foo=\"bar\">\nhi\n</para"
                                      "meter>\n</func" "tion>\n</tool_" "call>";
    const std::string f2b_function = "<tool_" "call>\n<func" "tion name=\"TaskCreate\"foo="
                                     "\"bar\">\n<para" "meter name=\"description\">\nhi\n</para"
                                     "meter>\n</func" "tion>\n</tool_" "call>";
    for (const bool tolerant : {false, true}) {
        failures += check_rejected(f2b_parameter, tc, Reason::MalformedStructure,
                                   std::string(tolerant ? "tolerant " : "strict ") +
                                       "F2b accepted adjacent parameter attributes",
                                   tolerant);
        failures += check_rejected(f2b_function, tc, Reason::InvalidToolName,
                                   std::string(tolerant ? "tolerant " : "strict ") +
                                       "F2b accepted adjacent function attributes",
                                   tolerant);
    }

    // --- F2c: unquoted attribute values may not be empty; quoted ones may.
    const std::string f2c_empty = "<tool_" "call>\n<func" "tion=TaskCreate>\n<para"
                                  "meter name=\"description\" junk=>\nhi\n</para" "meter>\n</func"
                                  "tion>\n</tool_" "call>";
    const std::string f2c_quoted = "<tool_" "call>\n<func" "tion=TaskCreate>\n<para"
                                   "meter name=\"description\" junk=\"\">\nhi\n</para"
                                   "meter>\n</func" "tion>\n</tool_" "call>";
    for (const bool tolerant : {false, true}) {
        failures += check_rejected(f2c_empty, tc, Reason::MalformedStructure,
                                   std::string(tolerant ? "tolerant " : "strict ") +
                                       "F2c accepted an empty unquoted attribute value",
                                   tolerant);
        const auto parsed = fi::parse_qwen_tool_call_output(f2c_quoted, 64, tc, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              Json::parse(parsed.tool_calls.front().arguments_json)
                                      .at("description") == "hi" &&
                              parsed.diagnostics.fallback_reason == Reason::None,
                          std::string(tolerant ? "tolerant " : "strict ") +
                              "F2c rejected a quoted empty attribute value");
    }

    // --- F3: tab/CR/LF delimit the function and invoke prefix, one-shot and streaming.
    const std::string f3_tab = "<tool_" "call>\n<func" "tion\tname=\"TaskCreate\">\n<para"
                               "meter name=\"description\">\nhi\n</para" "meter>\n</func"
                               "tion>\n</tool_" "call>";
    const std::string f3_newline = "<tool_" "call>\n<func" "tion\nname=\"TaskCreate\">\n<para"
                                   "meter name=\"description\">\nhi\n</para" "meter>\n</func"
                                   "tion>\n</tool_" "call>";
    const std::string f3_cr = "<tool_" "call>\n<in" "voke\r name=\"TaskCreate\">\n<para"
                              "meter name=\"description\">\nhi\n</para" "meter>\n</in"
                              "voke>\n</tool_" "call>";
    const std::string f3_standalone = "<func" "tion\tname=\"TaskCreate\">\n<para"
                                      "meter name=\"description\">\nhi\n</para" "meter>\n</func"
                                      "tion>";
    for (const auto& [label, text] :
         std::vector<std::pair<std::string, std::string>>{
             {"tab", f3_tab}, {"newline", f3_newline}, {"carriage return", f3_cr},
             {"standalone", f3_standalone}}) {
        for (const bool tolerant : {false, true}) {
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, tc, tolerant);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                                  parsed.tool_calls.front().name == "TaskCreate" &&
                                  Json::parse(parsed.tool_calls.front().arguments_json)
                                          .at("description") == "hi" &&
                                  parsed.diagnostics.fallback_reason == Reason::None,
                              std::string(tolerant ? "tolerant " : "strict ") +
                                  "F3 one-shot did not accept " + label +
                                  " prefix delimiter");
            const std::shared_ptr<const fi::ToolCallOutputContract> shared =
                std::make_shared<const fi::ToolCallOutputContract>(tc);
            for (const std::size_t chunk : {2u, 3u, 7u}) {
                fi::ToolCallOutputDecoder decoder(shared, 64, tolerant);
                std::string visible;
                for (std::size_t offset = 0; offset < text.size(); offset += chunk) {
                    visible += decoder.feed(std::string_view(text).substr(offset, chunk));
                }
                const auto terminal = decoder.finish();
                failures += check(visible.empty() && terminal.content.empty() &&
                                      terminal.tool_calls.size() == 1 &&
                                      terminal.tool_calls.front().name == "TaskCreate" &&
                                      terminal.diagnostics.fallback_reason == Reason::None,
                                  std::string(tolerant ? "tolerant " : "strict ") +
                                      "F3 streaming did not confirm the " + label +
                                      " prefix delimiter");
            }
        }
    }
    return failures;
}

int test_invalid_utf8_arguments_are_lossless() {
    using ParsedToolCallOutput = fi::ParsedToolCallOutput;
    using Reason               = ninfer::ToolCallParseFallbackReason;
    int failures = 0;
    const auto contract =
        contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                 {"timeout", Json{{"type", "integer"}}}});
    auto value_text = [](const std::string& value) {
        return std::string("<tool_call>\n<function=bash>\n<parameter=command>\n") + value +
               "\n</parameter>\n</function>\n</tool_call>";
    };
    const auto expect_arguments = [&](const std::string& label, const std::string& name,
                                      const ParsedToolCallOutput& parsed,
                                      std::string_view expected) {
        return check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                         parsed.tool_calls.front().name == name &&
                         parsed.tool_calls.front().arguments_json == expected,
                     std::string(label));
    };

    // Invalid bytes in a declared string value stay structured: each byte maps to a \u00XX
    // escape (0x7F is a valid single-byte code point and passes through raw).
    const std::string high = std::string("\xFF\xFE\x80\x7F");
    const std::string expected_high =
        std::string("{\"command\":\"") + "\\u00ff" + "\\u00fe" + "\\u0080" +
        std::string(1, static_cast<char>(0x7F)) + "\"}";
    {
        const auto parsed = fi::parse_qwen_tool_call_output(value_text(high), 64, contract);
        failures += expect_arguments("invalid UTF-8 in a declared string value was not encoded",
                                     "bash", parsed, expected_high);
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("command").is_string() &&
                              args.at("command") == Json(std::string("\u00FF\u00FE\u0080\u007F")),
                          "invalid UTF-8 byte escapes were not reversible");
    }

    // The legacy path (undeclared function in tolerant mode) encodes identically.
    {
        const std::string text = "<tool_call>\n<function=unknown>\n<parameter=extra>\n" + high +
                                 "\n</parameter>\n</function>\n</tool_call>";
        const std::string expected_extra =
            std::string("{\"extra\":\"") + "\\u00ff" + "\\u00fe" + "\\u0080" +
            std::string(1, static_cast<char>(0x7F)) + "\"}";
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract, true);
        failures += expect_arguments("invalid UTF-8 in a legacy value was not encoded", "unknown",
                                     parsed, expected_extra);
    }

    // A parameter name with an invalid byte is preserved as a \u00XX escape as well.
    {
        const std::string name  = std::string("c") + "\xFF" + "d";
        const std::string text  = std::string("<tool_call>\n<function=bash>\n<parameter=") + name +
                                  ">\nx\n</parameter>\n</function>\n</tool_call>";
        const std::string expected_name =
            std::string("{\"c\\u00ffd\":") + "\"x\"" + "}";
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
        failures += expect_arguments("invalid UTF-8 in a parameter name was not encoded", "bash",
                                     parsed, expected_name);
        failures += check(parsed.diagnostics.schema_mismatch_arguments == 1,
                          "unknown parameter name was not reported as a schema mismatch");
    }

    // Valid UTF-8 is untouched: the multibyte byte sequence stays verbatim in the JSON.
    {
        const std::string utf8  = "\xE4\xB8\x96";
        const auto parsed       = fi::parse_qwen_tool_call_output(value_text("echo " + utf8), 64,
                                                                  contract);
        failures += expect_arguments(
            "valid UTF-8 in a declared string value changed", "bash", parsed,
            std::string("{\"command\":\"echo ") + utf8 + "\"}");
    }

    // A multibyte character cut by the output budget stays structured in tolerant mode with
    // the partial bytes escaped and the tail flagged.
    {
        const std::string cut = std::string("<tool_call>\n<function=bash>\n<parameter=command>\n") +
                                "ab" + "\xE4\xB8";
        const auto parsed = fi::parse_qwen_tool_call_output(cut, 64, contract, true);
        failures +=
            expect_arguments("a budget-cut multibyte value was not kept structured", "bash",
                             parsed, std::string("{\"command\":\"ab\\u00e4\\u00b8\"}"));
        failures += check(parsed.diagnostics.fallback_reason == Reason::TruncatedTail,
                          "a budget-cut multibyte value was not flagged as a truncated tail");
    }

    // A NUL byte is valid UTF-8: it is preserved in the value and escaped by the dumper.
    {
        const std::string nul_value = std::string("a") + std::string(1, '\0') + "b";
        const auto parsed           = fi::parse_qwen_tool_call_output(value_text(nul_value), 64,
                                                                  contract);
        failures += expect_arguments(
            "a NUL byte in a declared string value was not preserved", "bash", parsed,
            std::string("{\"command\":\"a\\u0000b\"}"));
    }

    // The streaming decoder must not throw on the same bytes and must produce the same call.
    {
        const std::string text = value_text(high);
        const std::shared_ptr<const fi::ToolCallOutputContract> shared =
            std::make_shared<const fi::ToolCallOutputContract>(contract);
        fi::ToolCallOutputDecoder decoder(shared, 64);
        std::string visible;
        for (std::size_t offset = 0; offset < text.size(); offset += 3) {
            visible += decoder.feed(std::string_view(text).substr(offset, 3));
        }
        const auto terminal = decoder.finish();
        failures += check(visible.empty() && terminal.content.empty() &&
                              terminal.tool_calls.size() == 1 &&
                              terminal.tool_calls.front().arguments_json == expected_high,
                          "streaming did not encode invalid UTF-8 without throwing");
    }
    return failures;
}

int test_cut_function_opener_reports_invalid_name() {
    using Reason = ninfer::ToolCallParseFallbackReason;
    int failures = 0;
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});

    // A region that ends inside a function opener (the name found no closing '>') is an
    // invalid-name attempt in both modes, like every other failed function entry.
    const std::string cut = "<tool_call>\n<function=ba";
    failures += check_rejected(cut, contract, Reason::InvalidToolName,
                               "a cut function opener in strict mode was not an invalid name");
    failures +=
        check_rejected(cut, contract, Reason::InvalidToolName,
                       "a cut function opener in tolerant mode was not an invalid name", true);
    failures += check_rejected("<function=ba", contract, Reason::InvalidToolName,
                               "a standalone cut function opener was not an invalid name");

    // A wrapper with no function attempt at all stays a structural failure.
    failures +=
        check_rejected("<tool_call>\n", contract, Reason::MalformedStructure,
                       "an empty wrapper was not a structural failure");
    failures +=
        check_rejected("<tool_call>", contract, Reason::MalformedStructure,
                       "a bare wrapper was not a structural failure");

    // A NUL byte in the function name is an invalid name in both modes.
    {
        const std::string nul_name = std::string("<tool_call>\n<function=b") +
                                     std::string(1, '\0') +
                                     "ash>\n</function>\n</tool_call>";
        failures += check_rejected(nul_name, contract, Reason::InvalidToolName,
                                   "a NUL byte in a function name was not an invalid name");
        failures +=
            check_rejected(nul_name, contract, Reason::InvalidToolName,
                           "a NUL byte in a function name was not an invalid name (tolerant)",
                           true);
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
    failures += test_declared_string_parameter_delimiters_are_opaque();
    failures += test_literal_close_before_fake_sibling_stays_value_data();
    failures += test_declared_sibling_with_close_marker_in_name();
    failures += test_parameter_opener_grammar_is_shared();
    failures += test_parameter_header_grammar_hardening();
    failures += test_function_opener_grammar_hardening();
    failures += test_global_continuation_after_wrapper_close();
    failures += test_exact_string_round_trip_matrix();
    failures += test_fake_sibling_swallowing_structural_close();
    failures += test_declared_suspicious_name_variants();
    failures += test_suspicious_header_requires_current_tool_contract();
    failures += test_literal_markup_in_declared_string_arguments();
    failures += test_fake_sibling_borrowing_ordinary_greater_sign();
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
    failures += test_canonical_grammar_findings();
    failures += test_invalid_utf8_arguments_are_lossless();
    failures += test_cut_function_opener_reports_invalid_name();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
