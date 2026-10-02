#include "serve/http_server.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include "serve/anthropic_messages.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/request.h"
#include <string>
#include <thread>

namespace {

using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

// R8-02 (Round 8 §4.17): HTTP-level capability error for POST /v1/chat/completions with
// tool_choice=required. The request is rejected while the handler parses the body — before
// any generation starts — and the rendered error body carries the external shape.
int test_r8_chat_capability_error_payload() {
    using namespace ninfer::serve;
    int failures = 0;
    const Json body = Json{{"model", "qwen"},
                           {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                           {"tool_choice", "required"}};
    ApiError error;
    bool rejected = false;
    try {
        (void)parse_chat_completion_request(body, RequestLimits{});
    } catch (const ApiException& exception) {
        rejected = true;
        error    = exception.error();
    }
    failures += check(rejected, "R8 http: required tool_choice is rejected before generation");
    failures += check(error.status == 400 && error.type == "invalid_request_error",
                      "R8 http: the capability error is a 400 invalid_request_error");
    failures += check(error.param == "tool_choice" && error.code == "tool_choice_not_supported",
                      "R8 http: the exception state carries the capability param and code");
    const Json rendered = Json::parse(make_error_body(error));
    failures += check(rendered.at("error").at("type").get<std::string>() == "invalid_request_error" &&
                          rendered.at("error").at("param").get<std::string>() == "tool_choice" &&
                          rendered.at("error").at("code").get<std::string>() ==
                              "tool_choice_not_supported",
                      "R8 http: the rendered body has the external error shape");
    return failures;
}

// R9-02 (Round 9 §4.5): HTTP-level capability error for POST /v1/chat/completions with
// tools[0].function.strict=true. The request is rejected while the handler parses the
// body — before any generation starts — and the rendered error body carries the
// external shape.
int test_r9_chat_strict_error_payload() {
    using namespace ninfer::serve;
    int failures = 0;
    const Json body = Json{{"model", "qwen"},
                           {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                           {"tools", Json::array({Json{{"type", "function"},
                                                       {"function", Json{{"name", "weather"},
                                                                         {"description", "Get weather"},
                                                                         {"parameters", Json{{"type", "object"}}},
                                                                         {"strict", true}}}}})}};
    ApiError error;
    bool rejected = false;
    try {
        (void)parse_chat_completion_request(body, RequestLimits{});
    } catch (const ApiException& exception) {
        rejected = true;
        error    = exception.error();
    }
    failures += check(rejected, "R9 http: strict true is rejected before generation");
    failures += check(error.status == 400 && error.type == "invalid_request_error",
                      "R9 http: the strict error is a 400 invalid_request_error");
    failures += check(error.param == "tools[0].function.strict" &&
                          error.code == "strict_tools_not_supported",
                      "R9 http: the exception state carries the strict param and code");
    const Json rendered = Json::parse(make_error_body(error));
    failures += check(rendered.at("error").at("type").get<std::string>() == "invalid_request_error" &&
                          rendered.at("error").at("param").get<std::string>() ==
                              "tools[0].function.strict" &&
                          rendered.at("error").at("code").get<std::string>() ==
                              "strict_tools_not_supported",
                      "R9 http: the rendered body has the external error shape");
    return failures;
}

// R9-03 (Round 9 §5.14): HTTP-level capability error for POST /v1/chat/completions with a
// custom tool definition. The request is rejected while the handler parses the body and
// the rendered error body carries the external shape.
int test_r9_chat_custom_error_payload() {
    using namespace ninfer::serve;
    int failures = 0;
    const Json body = Json{{"model", "qwen"},
                           {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                           {"tools",
                            Json::array({Json{{"type", "custom"},
                                              {"custom", Json{{"name", "shell"},
                                                               {"description", "Run a shell command"}}}}})}};
    ApiError error;
    bool rejected = false;
    try {
        (void)parse_chat_completion_request(body, RequestLimits{});
    } catch (const ApiException& exception) {
        rejected = true;
        error    = exception.error();
    }
    failures += check(rejected, "R9 http: a custom tool definition is rejected before generation");
    failures += check(error.status == 400 && error.type == "invalid_request_error",
                      "R9 http: the custom error is a 400 invalid_request_error");
    failures += check(error.param == "tools[0].type" && error.code == "tool_type_not_supported",
                      "R9 http: the exception state carries the custom param and code");
    const Json rendered = Json::parse(make_error_body(error));
    failures += check(rendered.at("error").at("type").get<std::string>() == "invalid_request_error" &&
                          rendered.at("error").at("param").get<std::string>() == "tools[0].type" &&
                          rendered.at("error").at("code").get<std::string>() ==
                              "tool_type_not_supported",
                      "R9 http: the rendered custom body has the external error shape");
    return failures;
}

// R9-04 / R9-12 (Round 9 §6.9/§12): HTTP-level error for POST /v1/messages with
// tool_choice=any. The Anthropic external envelope renders type/error/message/request_id
// (param and code are internal-only), so the body test asserts the envelope and the
// message while the internal state keeps param/code.
int test_r9_anthropic_tool_choice_error_payload() {
    using namespace ninfer::serve;
    int failures = 0;
    const Json body = Json{{"model", "claude-local"},
                           {"max_tokens", 4096},
                           {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                           {"tools",
                            Json::array({Json{{"name", "weather"},
                                              {"input_schema", Json{{"type", "object"}}}}})},
                           {"tool_choice", Json{{"type", "any"}}}};
    ApiError error;
    bool rejected = false;
    try {
        (void)parse_anthropic_messages_request(body, RequestLimits{});
    } catch (const ApiException& exception) {
        rejected = true;
        error    = exception.error();
    }
    failures += check(rejected, "R9 http: tool_choice any is rejected before generation");
    failures += check(error.status == 400 && error.type == "invalid_request_error",
                      "R9 http: the Anthropic error carries the 400 status");
    failures += check(error.param == "tool_choice" && error.code == "tool_choice_not_supported",
                      "R9 http: the internal state carries param and code");
    const Json rendered = Json::parse(make_anthropic_error_body(error, "req_r9_test"));
    failures += check(rendered.at("type").get<std::string>() == "error" &&
                          rendered.at("error").at("type").get<std::string>() ==
                              "invalid_request_error" &&
                          rendered.at("error").at("message").is_string() &&
                          rendered.at("request_id").get<std::string>() == "req_r9_test",
                      "R9 http: the Anthropic envelope renders type/message/request_id");
    return failures;
}

int main() {
    int failures = 0;
    using ninfer::serve::api_route_pattern;
    using ninfer::serve::canonical_api_path;

    // An Anthropic SDK given a base URL ending in /v1 adds one more /v1; only that one is removed.
    failures += check(canonical_api_path("/v1/v1/messages") == "/v1/messages" &&
                          canonical_api_path("/v1/v1/models/qwen") == "/v1/models/qwen",
                      "a doubled /v1 prefix must name the /v1 endpoint");
    failures += check(canonical_api_path("/v1/messages") == "/v1/messages" &&
                          canonical_api_path("/v1/v1/v1/messages") == "/v1/v1/messages" &&
                          canonical_api_path("/v1/v1") == "/v1/v1" &&
                          canonical_api_path("/health") == "/health",
                      "only a single doubled /v1 prefix may be removed");

    // Errors on a doubled path keep the envelope and request-id header of the endpoint it names.
    ninfer::serve::ServeOptions options;
    httplib::Request missing_messages;
    missing_messages.path = "/v1/v1/messages/missing";
    httplib::Response missing_response;
    missing_response.status = 404;
    const auto missing_result =
        ninfer::serve::handle_unrendered_http_error(options, missing_messages, missing_response);
    failures += check(missing_result == httplib::Server::HandlerResponse::Handled &&
                          !missing_response.body.empty() &&
                          Json::parse(missing_response.body).at("type") == "error" &&
                          missing_response.has_header("request-id") &&
                          !missing_response.has_header("x-request-id"),
                      "a doubled Anthropic path must keep the Anthropic error envelope");
    httplib::Request oversized_responses;
    oversized_responses.path = "/v1/v1/responses";
    httplib::Response oversized_response;
    oversized_response.status = 413;
    const auto oversized_result = ninfer::serve::handle_unrendered_http_error(
        options, oversized_responses, oversized_response);
    failures += check(oversized_result == httplib::Server::HandlerResponse::Handled &&
                          Json::parse(oversized_response.body).at("error").at("code") ==
                              "request_too_large" &&
                          oversized_response.has_header("x-request-id"),
                      "a doubled OpenAI path must keep the OpenAI error envelope");

    // Route patterns reach the handler under either prefix and keep the endpoint's own captures.
    httplib::Server server;
    server.Post(api_route_pattern("/messages"),
                [](const httplib::Request&, httplib::Response& res) {
                    res.set_content("messages", "text/plain");
                });
    server.Get(api_route_pattern(R"(/models/(.+))"),
               [](const httplib::Request& req, httplib::Response& res) {
                   const std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
                   res.set_content(id, "text/plain");
               });
    const int port = server.bind_to_any_port("127.0.0.1");
    failures += check(port > 0, "test server could not bind a loopback port");
    if (port > 0) {
        std::thread listener([&server] { server.listen_after_bind(); });
        server.wait_until_ready();
        httplib::Client client("127.0.0.1", port);
        const auto post_status = [&client](const char* path) {
            const httplib::Result result = client.Post(path, "{}", "application/json");
            return result ? result->status : -1;
        };
        failures += check(post_status("/v1/messages") == 200 &&
                              post_status("/v1/v1/messages") == 200,
                          "the messages route must answer under /v1 and /v1/v1");
        failures += check(post_status("/messages") == 404 &&
                              post_status("/v1/v1/v1/messages") == 404 &&
                              post_status("/v2/v1/messages") == 404,
                          "only the /v1 and single doubled /v1 prefixes may reach a route");
        const httplib::Result model = client.Get("/v1/v1/models/qwen3.8-27b");
        failures += check(model && model->status == 200 && model->body == "qwen3.8-27b",
                          "a doubled prefix must not shift the route's capture groups");
        server.stop();
        listener.join();
    }

    failures += test_r8_chat_capability_error_payload();
    failures += test_r9_chat_strict_error_payload();
    failures += test_r9_chat_custom_error_payload();
    failures += test_r9_anthropic_tool_choice_error_payload();
    if (failures == 0) { std::cout << "http routes tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
