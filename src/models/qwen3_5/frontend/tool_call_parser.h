#pragma once

#include "ninfer/types.h"
#include "models/qwen3_5/frontend/tool_call_stream.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// Qwen's tool syntax carries each argument as untyped text. This terminal contract records only
// the supported top-level JSON Schema types needed to normalize that text. A type mismatch remains
// a structured call for consumer validation; recursive validation is outside this non-strict
// contract.
struct ToolCallOutputContract {
    enum class SchemaType : std::uint8_t {
        Null    = 1U << 0U,
        Boolean = 1U << 1U,
        Integer = 1U << 2U,
        Number  = 1U << 3U,
        String  = 1U << 4U,
        Object  = 1U << 5U,
        Array   = 1U << 6U,
    };

    struct TypeSet {
        std::uint8_t bits = 0;
    };

    enum class NormalizationPolicy : std::uint8_t {
        Legacy,
        DeclaredTypes,
    };

    struct Parameter {
        std::string name;
        NormalizationPolicy policy = NormalizationPolicy::Legacy;
        TypeSet types;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        bool unambiguous = true;
    };

    std::vector<Tool> tools;
    bool enforce_declared_names = false;
};

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled);

// Parse Qwen's XML-like tool-call format. The parse itself is policy-free: it reports the
// complete calls, the state of the open call, and where the input ended. The recovery policy
// (decide_tool_call_recovery) then decides what may be committed: tolerant mode commits a
// call only when its function close has been consumed (the function close is the
// executability boundary: a missing function close never makes the call executable, whatever
// its parameter values show; a cut parameter value of an open call is never committed), while
// strict mode keeps its all-or-nothing behavior.
// Parse Qwen's XML-like tool-call format. The parse itself is policy-free: it reports the
// complete calls, the state of the open call, and where the input ended. The recovery policy
// (decide_tool_call_recovery) then decides what may be committed: tolerant mode commits a
// call only when its function close has been consumed (the function close is the
// executability boundary: a missing function close never makes the call executable, whatever
// its parameter values show; a cut parameter value of an open call is never committed), while
// strict mode keeps its all-or-nothing behavior. R5-07: the syntax mode selects the top-level
// entry set (native: the wrapped <tool_call> form only); the default keeps the historical
// compatibility entry set for this low-level entry.
[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract, bool tolerant = false,
                            FinishReason finish_reason = FinishReason::None,
                            ToolCallSyntaxMode syntax = ToolCallSyntaxMode::Compatibility,
                            ToolCallAmbiguityPolicy ambiguity =
                                ToolCallAmbiguityPolicy::PayloadFidelity);

// Incrementally publishes bytes that are provably outside a possible terminal Qwen tool-call
// suffix. At terminal time, valid calls are retained structurally; malformed output is restored
// verbatim.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };
    // R5-07: the syntax mode is stored with the decoder; the streaming session constructs the
    // decoder from OutputOptions (the production Qwen3.8 default is the native mode).
    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length, bool tolerant = false,
                          ToolCallSyntaxMode syntax = ToolCallSyntaxMode::Compatibility,
                          ToolCallAmbiguityPolicy ambiguity =
                              ToolCallAmbiguityPolicy::PayloadFidelity);

    [[nodiscard]] std::string feed(std::string_view text);
    [[nodiscard]] Terminal finish(FinishReason finish_reason = FinishReason::None);

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    // Pre-marker scan and region buffering delegate to the incremental machine (one feed
    // rule, no duplicated marker-scan state). The machine's policy is unused here: the
    // finish path re-parses the latched region with the contract-aware policy.
    ToolCallStreamParser machine_{ToolCallParsePolicy{}};
    std::size_t max_tool_name_length_ = 0;
    bool tolerant_                    = false;
    ToolCallSyntaxMode syntax_           = ToolCallSyntaxMode::Compatibility;
    ToolCallAmbiguityPolicy ambiguity_   = ToolCallAmbiguityPolicy::PayloadFidelity;
    bool finished_                    = false;
};

} // namespace ninfer::models::qwen3_5::frontend
