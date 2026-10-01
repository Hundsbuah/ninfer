#include "models/qwen3_5/frontend/tool_call_grammar.h"

#include <iostream>
#include <string>
#include <string_view>

using namespace ninfer::models::qwen3_5::frontend;
namespace { // NOLINT(cert-err58-cpp)

int failures = 0;

void check(bool ok, const char* what, const char* context = "") {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << " " << context << '\n';
    }
}

void expect_status(ToolHeaderStatus actual, ToolHeaderStatus expected, const char* what,
                   const char* context) {
    check(actual == expected, what, context);
}

void expect_marker(ToolMarkerStatus actual, ToolMarkerStatus expected, const char* what,
                   const char* context) {
    check(actual == expected, what, context);
}

int test_function_header_forms() {
    ToolOpenTag tag = {};
    expect_status(parse_tool_open_header("<function=write>", ToolTagKind::Function, tag),
                  ToolHeaderStatus::Complete, "function short form is complete",
                  "<function=write>");
    check(std::string(tag.name) == "write", "function short name", "<function=write>");
    check(tag.consumed == 16, "function short consumed", "<function=write>");

    tag = {};
    expect_status(parse_tool_function_open("<function name=\"write\">", tag),
                  ToolHeaderStatus::Complete, "function attribute form is complete",
                  "<function name=\"write\">");
    check(std::string(tag.name) == "write", "function attribute name",
          "<function name=\"write\">");
    check(tag.consumed == 23, "function attribute consumed", "<function name=\"write\">");

    for (const char* text : {"<function name = \"write\">", "<function\tname=\"write\">",
                             "<function\nname=\"write\">", "<function\r\nname=\"write\">",
                             "<function name='write'>", "<function name=write>",
                             "<function name = write>", "<function\tname = write>"}) {
        tag = {};
        expect_status(parse_tool_function_open(text, tag), ToolHeaderStatus::Complete,
                      "function accepted form is complete", text);
        check(std::string(tag.name) == "write", "function accepted form name", text);
    }

    tag = {};
    expect_status(parse_tool_function_open("<function=\"write\">", tag),
                  ToolHeaderStatus::Complete, "function quoted short name is complete",
                  "<function=\"write\">");
    check(std::string(tag.name) == "write", "function quoted short name value");

    for (const char* text : {"<function>", "<function >", "<invoke>", "<invoke >"}) {
        tag = {};
        expect_status(parse_tool_function_open(text, tag), ToolHeaderStatus::Complete,
                      "bare opener is complete", text);
        check(tag.name.empty(), "bare opener has empty name", text);
    }

    tag = {};
    expect_status(parse_tool_function_open("<function filename=\"x\" name=\"write\">", tag),
                  ToolHeaderStatus::Complete, "function extra attribute is complete",
                  "<function filename=\"x\" name=\"write\">");
    check(std::string(tag.name) == "write", "function extra attribute name");

    tag = {};
    expect_status(parse_tool_function_open("<function name=\"write\" junk=\"x\">", tag),
                  ToolHeaderStatus::Complete, "function trailing attribute is complete",
                  "<function name=\"write\" junk=\"x\">");

    for (const char* text : {"<function name=\"write\"junk=\"x\">",
                             "<function name=\"write\"junk>",
                             "<function=write junk=\"x\">",
                             "<function name=>",
                             "<function name= >",
                             "<function=write name=\"x\">",
                             "<function name=\"write\" junk=>",
                             "<function name>",
                             "<function name junk=\"x\">",
                             "<function #write>",
                             "<function=write junk>"}) {
        tag = {};
        expect_status(parse_tool_function_open(text, tag), ToolHeaderStatus::Invalid,
                      "function broken header is invalid", text);
    }

    for (const char* text : {"<parameter=content>", "<param=content>", "<tool_call>",
                             "<function_calls>", "<invoke=write>"}) {
        tag = {};
        expect_status(parse_tool_open_header(text, ToolTagKind::Function, tag),
                      ToolHeaderStatus::NoMatch, "other family is NoMatch for function", text);
    }
    return 0;
}

int test_function_name_semantics() {
    ToolOpenTag tag = {};
    check(parse_tool_function_open("<function=bad.name>", tag) == ToolHeaderStatus::Complete,
          "dotted name is a syntactically complete header");
    check(std::string(tag.name) == "bad.name", "dotted name value");
    check(!is_valid_tool_name(tag.name, 64), "dotted name is not a valid function name");

    check(is_valid_tool_name("write", 64), "simple name is valid");
    check(is_valid_tool_name("write-2_read", 64), "mixed name is valid");
    check(!is_valid_tool_name("", 64), "empty name is not valid");
    check(!is_valid_tool_name(std::string(65, 'a'), 64), "over-long name is not valid");
    check(!is_valid_tool_name("a b", 64), "space is not a name character");
    check(!is_valid_tool_name("a.b", 64), "dot is not a name character");
    return 0;
}

int test_parameter_header_forms() {
    ToolOpenTag tag = {};
    const struct {
        const char* text;
        ToolTagKind kind;
    } accepted[] = { {"<parameter=content>", ToolTagKind::Parameter},
                     {"<param=content>", ToolTagKind::Param},
                     {"<parameter name=\"content\">", ToolTagKind::Parameter},
                     {"<parameter name = \"content\">", ToolTagKind::Parameter},
                     {"<parameter name='content'>", ToolTagKind::Parameter},
                     {"<parameter\tname=\"content\">", ToolTagKind::Parameter},
                     {"<param name=content>", ToolTagKind::Param} };
    for (const auto& form : accepted) {
        tag = {};
        expect_status(parse_tool_parameter_open(form.text, tag), ToolHeaderStatus::Complete,
                      "parameter accepted form is complete", form.text);
        check(std::string(tag.name) == "content", "parameter accepted form name", form.text);
        check(tag.kind == form.kind, "parameter family selected", form.text);
    }

    for (const char* text : {"<parameter=content>", "<parameter name=\"content\">"}) {
        tag = {};
        expect_status(parse_tool_open_header(text, ToolTagKind::Param, tag),
                      ToolHeaderStatus::NoMatch, "param family does not claim parameter", text);
    }
    for (const char* text : {"<param=content>", "<param name=\"content\">"}) {
        tag = {};
        expect_status(parse_tool_open_header(text, ToolTagKind::Parameter, tag),
                      ToolHeaderStatus::NoMatch, "parameter family does not claim param", text);
    }

    tag = {};
    expect_status(parse_tool_parameter_open("<parameter name=\"a b\">", tag),
                  ToolHeaderStatus::Complete, "parameter quoted name with space is complete",
                  "<parameter name=\"a b\">");
    check(std::string(tag.name) == "a b", "parameter quoted name kept verbatim");

    tag = {};
    expect_status(parse_tool_parameter_open("<parameter>", tag), ToolHeaderStatus::Complete,
                  "bare parameter is complete", "<parameter>");
    check(tag.name.empty(), "bare parameter has empty name");

    for (const char* text : {"<parameter foo=>", "<param=>"}) {
        tag = {};
        expect_status(parse_tool_parameter_open(text, tag), ToolHeaderStatus::Invalid,
                      "parameter broken header is invalid", text);
    }
    // A junk attribute with a non-empty unquoted value is still a syntactically complete header;
    // the empty parameter name is rejected by the region parser, not the grammar.
    tag = {};
    expect_status(parse_tool_parameter_open("<parameter foo=bar=>", tag), ToolHeaderStatus::Complete,
                  "parameter junk value header is complete", "<parameter foo=bar=>");
    check(tag.name.empty(), "parameter junk value header has no name", "");
    return 0;
}

int test_quote_aware_headers() {
    ToolOpenTag tag = {};
    expect_status(parse_tool_parameter_open("<parameter filename=\"a>b\" name=\"content\">", tag),
                  ToolHeaderStatus::Complete,
                  "quoted > terminates only at the matching quote",
                  "<parameter filename=\"a>b\" name=\"content\">");
    check(std::string(tag.name) == "content", "name after quoted > is found");
    check(tag.consumed == std::string_view("<parameter filename=\"a>b\" name=\"content\">").size(),
          "consumed spans the whole quoted header");

    tag = {};
    expect_status(parse_tool_function_open("<function name=\"a>b\" junk=\"x\">", tag),
                  ToolHeaderStatus::Complete, "quoted > in a function header",
                  "<function name=\"a>b\" junk=\"x\">");
    check(std::string(tag.name) == "a>b", "quoted name keeps the > verbatim");

    tag = {};
    expect_status(parse_tool_function_open("<function name='a > b'>", tag),
                  ToolHeaderStatus::Complete, "single-quoted value with >",
                  "<function name='a > b'>");
    check(std::string(tag.name) == "a > b", "single-quoted value kept");

    for (const char* text : {"<parameter name=\"description>", "<parameter name=\"description",
                             "<parameter name=\"desc"}) {
        tag = {};
        expect_status(parse_tool_parameter_open(text, tag), ToolHeaderStatus::NeedMore,
                      "unterminated parameter quote needs more", text);
    }
    for (const char* text : {"<function name='x'", "<function name=\"x\" junk=\"y"}) {
        tag = {};
        expect_status(parse_tool_function_open(text, tag), ToolHeaderStatus::NeedMore,
                      "unterminated function quote needs more", text);
    }
    // R3-05: a quoted value is single-line. A CR or LF before the closing quote is
    // Invalid (the header grammar is line-oriented).
    for (const std::string text :
         {std::string("<parameter name=\"a\nb\">"), std::string("<parameter name=\"a\rb\">"),
          std::string("<parameter name=\"a\r\nb\">"), std::string("<function name=\"a\nb\">")}) {
        tag = {};
        const ToolHeaderStatus st = text.rfind("<parameter", 0) == 0
                                         ? parse_tool_parameter_open(text, tag)
                                         : parse_tool_function_open(text, tag);
        expect_status(st, ToolHeaderStatus::Invalid, "quoted CR/LF before the close is invalid",
                      text.c_str());
    }
    // R3-05 (F8): the maximal header bound (kMaxToolHeaderBytes after the keyword). A
    // header whose terminating '>' is not reached inside the bound is Invalid, not
    // NeedMore; one byte inside the bound stays NeedMore.
    tag = {};
    const std::string over = std::string("<function=") + std::string(1025, 'a');
    expect_status(parse_tool_function_open(over, tag), ToolHeaderStatus::Invalid,
                  "one byte over the 1024-byte bound is invalid", over.c_str());
    tag = {};
    const std::string at_bound = std::string("<function=") + std::string(1022, 'a') + ">";
    expect_status(parse_tool_function_open(at_bound, tag), ToolHeaderStatus::Complete,
                  "header at the 1024-byte bound is complete", at_bound.c_str());
    return 0;
}

int test_marker_prefixes() {
    ToolOpenTag marker = {};
    for (const char* text : {"<tool_call>", "<function_calls>", "<function=write>",
                             "<function name=\"write\">", "<invoke=write>", "<function>",
                             "<function=\"write\">", "<function=very-long-tool-name-0123456789>"}) {
        expect_marker(
            classify_tool_marker_prefix(text, marker, ninfer::ToolCallSyntaxMode::Compatibility),
            ToolMarkerStatus::Complete, "complete marker", text);
    }
    for (const char* text : {"<", "<t", "<to", "<tool_call", "<function", "<function=",
                             "<function=w", "<function=write", "<function name",
                             "<function name=", "<function name=\"write\"",
                             "<function name=\"write", "<invoke", "<invoke=", "<function_c",
                             "<function_ca", "<function_call", "<function_calls", "<function\t",
                             "<function\tname=\"write\"", "<function\n", "<function name=\"w"}) {
        expect_marker(
            classify_tool_marker_prefix(text, marker, ninfer::ToolCallSyntaxMode::Compatibility),
            ToolMarkerStatus::NeedMore, "marker prefix needs more", text);
    }
    for (const char* text : {"<foo=bar>", "<functionx=write>", "<function\"x\">",
                             "<parameter=content>", "<param=content>", "<function_call>",
                             "<function#write>"}) {
        expect_marker(
            classify_tool_marker_prefix(text, marker, ninfer::ToolCallSyntaxMode::Compatibility),
            ToolMarkerStatus::NotMarker, "not a marker", text);
    }
    // R5-07: in native syntax only the wrapped <tool_call> literal latches at top level;
    // the legacy bare/function_calls forms are ordinary text.
    for (const char* text : {"<function=write>", "<function name=\"write\">", "<invoke=write>",
                             "<function>", "<function_calls>", "<function_call>"}) {
        expect_marker(
            classify_tool_marker_prefix(text, marker, ninfer::ToolCallSyntaxMode::QwenWrappedNative),
            ToolMarkerStatus::NotMarker, "native: compatibility entry is not a marker", text);
    }
    expect_marker(classify_tool_marker_prefix("<tool_call>", marker,
                                              ninfer::ToolCallSyntaxMode::QwenWrappedNative),
                  ToolMarkerStatus::Complete, "native: wrapped entry latches", "<tool_call>");
    expect_marker(classify_tool_marker_prefix("<tool_call", marker,
                                              ninfer::ToolCallSyntaxMode::QwenWrappedNative),
                  ToolMarkerStatus::NeedMore, "native: wrapped prefix needs more", "<tool_call");
    expect_marker(classify_tool_marker_prefix("<", marker,
                                              ninfer::ToolCallSyntaxMode::QwenWrappedNative),
                  ToolMarkerStatus::NeedMore, "native: bare '<' may still be the wrapper", "<");
    expect_marker(classify_tool_marker_prefix("<function", marker,
                                              ninfer::ToolCallSyntaxMode::QwenWrappedNative),
                  ToolMarkerStatus::NotMarker, "native: '<function' cannot become the wrapper",
                  "<function");
    return 0;
}

int test_marker_discovery() {
    const std::string_view none = "plain prose without markers";
    check(find_tool_marker(none, 0, ninfer::ToolCallSyntaxMode::Compatibility) == std::string_view::npos,
          "no marker in prose");
    check(find_tool_marker(none, 0, ninfer::ToolCallSyntaxMode::QwenWrappedNative) == std::string_view::npos,
          "native: no marker in prose");

    const std::string_view early = "prefix <function=write> rest";
    check(find_tool_marker(early, 0, ninfer::ToolCallSyntaxMode::Compatibility) == 7,
          "marker position found");
    check(find_tool_marker(early, 0, ninfer::ToolCallSyntaxMode::QwenWrappedNative) == std::string_view::npos,
          "R5-07: native discovery ignores the bare function entry");

    const std::string_view later = "prose then <tool_call> wrapper";
    check(find_tool_marker(later, 0, ninfer::ToolCallSyntaxMode::Compatibility) == 11,
          "wrapper position found");
    check(find_tool_marker(later, 0, ninfer::ToolCallSyntaxMode::QwenWrappedNative) == 11,
          "native: the wrapped entry is discovered");

    const std::string_view after = "<function=write> tail <invoke=other>";
    check(find_tool_marker(after, 0, ninfer::ToolCallSyntaxMode::Compatibility) == 0,
          "earliest marker wins");
    check(find_tool_marker(after, 1, ninfer::ToolCallSyntaxMode::Compatibility) == 22,
          "search from position");
    check(find_tool_marker(after, 0, ninfer::ToolCallSyntaxMode::QwenWrappedNative) == std::string_view::npos,
          "R5-07: native discovery finds nothing in compatibility-only entries");

    // A quoted opener inside prose is a complete marker too: discovery is syntax only, and
    // identity/declaredness is a policy decision made later by the region parser.
    const std::string_view quoted = "see <function=shell> and the real <tool_call>";
    check(find_tool_marker(quoted, 0, ninfer::ToolCallSyntaxMode::Compatibility) == 4,
          "quoted opener is discovered first");
    check(find_tool_marker(quoted, 0, ninfer::ToolCallSyntaxMode::QwenWrappedNative) == 34,
          "native: only the wrapped entry is discovered");

    const std::string_view parameter_only = "text <parameter=content> more";
    check(find_tool_marker(parameter_only, 0, ninfer::ToolCallSyntaxMode::Compatibility) ==
              std::string_view::npos,
          "parameter opener is not a top-level marker");
    return 0;
}

} // namespace

int main() {
    test_function_header_forms();
    test_function_name_semantics();
    test_parameter_header_forms();
    test_quote_aware_headers();
    test_marker_prefixes();
    test_marker_discovery();
    std::cout << (failures == 0 ? "tool_call_grammar_test passed" : "tool_call_grammar_test FAILED")
              << '\n';
    return failures == 0 ? 0 : 1;
}
