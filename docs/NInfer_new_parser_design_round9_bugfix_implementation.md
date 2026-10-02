# NInfer `new_parser_design` — Round 9 Bugfix / Integration Specification

## 0. Purpose

This document is the **Round-9 implementation specification** for the remaining tool-call findings found after a full-chain review of:

```text
Hundsbuah/ninfer:new_parser_design
```

Pinned reviewed baseline:

```text
882b5a3d88ff5c363c0f94a8094b2ee802ee0201
```

Round 9 deliberately **does not reopen the Stage-1 / Stage-2 / Stage-3 Qwen tool-call parser** unless a new direct parser-core reproducer is found. Round 8 is considered green for its stated scope. Round 9 fixes four separate classes:

```text
R9-01  Future grammar constraint allows suffix text that strict final parsing rejects.
R9-02  OpenAI Chat function strict:true is silently weakened to non-strict behavior.
R9-03  OpenAI Chat custom tools are not preserved as OpenAI custom tools.
R9-04  Anthropic forced / serial / strict tool controls are accepted as advisory.
```

The objectives are:

1. make future constrained tool decoding preserve strict post-trigger tool-sequence semantics;
2. preserve ordinary lazy-trigger prose before the first tool region;
3. preserve directly consecutive legal tool calls;
4. add explicit end-of-generation legality to the CPU grammar state;
5. reject `strict:true` in OpenAI Chat until schema-constrained generation exists;
6. reject Chat custom tools until free-form custom input/output/history semantics exist end-to-end;
7. reject unsupported Anthropic forced, serial, and strict guarantees instead of silently converting them to automatic best effort;
8. preserve neutral cases such as `tool_choice:none`;
9. leave the already stricter OpenAI Responses path unchanged;
10. keep runtime constrained decoding fail-fast.

---

## 0.1 Pinned baseline

Repository:

```text
Hundsbuah/ninfer
```

Branch:

```text
new_parser_design
```

Reviewed HEAD:

```text
882b5a3d88ff5c363c0f94a8094b2ee802ee0201
```

Tree:

```text
59f6543d9f44599ee638b2d0d3c5abbb4188afb5
```

At review time:

```text
GitHub combined status checks: none
GitHub workflow runs: none
```

Repository-reported Round-8 verification:

```text
targeted CPU:
  9 selected
  9 passed
  0 failed

full GPU-hidden:
  147 selected
  46 passed
  0 failed
  101 skipped
  exit 0
```

These results were not independently rerun by the reviewer. Before applying Round 9, pin the current HEAD again and re-review any path that changed after this baseline.

---

## 0.2 Evidence status

### Source-verified at the pinned NInfer baseline

**R9-01**

- strict parser rejects a complete tool call followed by non-whitespace suffix as `TrailingContent`;
- `ToolCallGrammarConstraint` deliberately treats that suffix as unconstrained prose;
- `docs/new_parser_phase4_design.md` explicitly documents this parser/constraint asymmetry;
- runtime constrained decoding remains disabled by startup fail-fast.

**R9-02**

- Chat accepts `function.strict=true`;
- it does not change generation;
- prompt normalization exposes the tool as `strict:false`;
- Responses already rejects strict tool schemas.

**R9-03**

- Chat accepts `type:custom`;
- custom input is lowered to a synthetic JSON function with one string property `input`;
- custom grammar is descriptive text only;
- `GeneratedToolCall` stores only `name` and `arguments_json`;
- Chat response serialization emits every generated tool call as `type:function`;
- Chat history parser accepts only `type:function`.

**R9-04**

- Anthropic `tool_choice:any` and named `tool` become `ToolChoiceMode::Auto`;
- `disable_parallel_tool_use` is parsed but not enforced;
- tool `strict:true` is type-checked but not enforced;
- tests explicitly pin these fields as advisory.

### Externally verified API semantics

**OpenAI**

- function `strict:true` requests strict schema enforcement;
- custom tools use free-form string input rather than JSON function arguments;
- custom tool calls have a distinct custom wire shape;
- a custom grammar constrains generated custom-tool input.

**Anthropic**

- `tool_choice:any` requires a tool on models supporting that mode;
- named `tool` forces that tool on models supporting that mode;
- `disable_parallel_tool_use:true` under `auto` means at most one call;
- under a forced mode it means exactly one call;
- `strict:true` guarantees schema-conforming tool input via grammar-constrained sampling;
- some current Claude families reject `any`/`tool` instead of supporting them.

---

## 0.3 Hard scope boundaries

Do **not** opportunistically change:

```text
Stage-1 greedy parser
Stage-2 consistent completion
Stage-3 recovery
parameter normalization
fence tracking
FailClosed ambiguity policy
tolerant parser policy
OpenAI Responses namespaces
Anthropic server tools/toolsets
MCP execution
response_format behavior
runtime sampler integration
```

unless a new independent reproducer proves such a change necessary.

---

## 0.4 GPU constraint

Do not run:

```text
GPU/model inference
_real model tests
benchmarks
CUDA correctness/race tests
runtime constrained-sampling tests
```

Required verification is CPU/static only:

```text
parser tests
grammar-state tests
OpenAI schema tests
Responses tests
Anthropic schema tests
HTTP error tests
serve-option tests
request-log tests
GPU-hidden full CTest
```

---

# 1. Finding index

| ID | Finding | Severity | Exposure | Round-9 action |
|---|---|---:|---|---|
| R9-01 | Constraint allows post-call suffix prose that strict parser rejects | MEDIUM architecture | dormant while constrained sampling is off | FIX CPU core before activation |
| R9-01a | Pre-trigger and post-call-between-regions states are conflated | root cause | dormant | FIX state model |
| R9-01b | No explicit EOS legality predicate | design gap | dormant | ADD termination semantics |
| R9-02 | Chat `strict:true` silently becomes non-strict | MEDIUM | active | FAIL FAST |
| R9-03 | Chat custom tools lose free-form/wire semantics | MEDIUM | active | FAIL FAST until complete implementation |
| R9-03a | free-form custom input becomes JSON `{input:...}` | root cause | active | REMOVE partial support |
| R9-03b | custom grammar is descriptive only | root cause | active | REMOVE partial support |
| R9-03c | output always serializes generated calls as function | root cause | active | REMOVE false custom support |
| R9-03d | Chat history rejects custom calls | root cause | active | KEEP explicit unsupported state |
| R9-04 | Anthropic `any` / named `tool` becomes Auto | MEDIUM | active | FAIL FAST |
| R9-04b | `disable_parallel_tool_use:true` is advisory | MEDIUM | active | FAIL FAST when callable |
| R9-04c | Anthropic `strict:true` is advisory | MEDIUM | active | FAIL FAST when callable |
| R6-04 | byte-identical call-only quotation | residual | active protocol limitation | DOCUMENT ONLY |
| runtime constrained decoding | sampler integration absent | deferred | fail-fast | KEEP FAIL-FAST |

---

# 2. Round-9 invariants

## R9-I1 — Once a strict tool sequence starts, ordinary suffix prose is illegal

Before the first tool trigger, lazy activation may allow prose. After a tool region has begun and closed, only these are legal:

```text
formatting whitespace
another legal top-level tool entry
end-of-generation
```

Visible suffix prose must be rejected because the final strict parser would demote the tool region as `TrailingContent`.

## R9-I2 — Pre-trigger inactive and post-call-between-regions are different states

`triggered_ == false` is insufficient.

Before first tool:

```text
prose allowed
marker may trigger
EOS allowed
```

After a completed tool:

```text
formatting whitespace allowed
next legal tool marker allowed
EOS allowed
visible prose rejected
```

## R9-I3 — Intent policy and post-call strictness are orthogonal

`ToolCallIntentPolicy` decides whether the **first** tool entry may latch after visible content. Once a tool sequence begins, strict suffix rules apply under both:

```text
TemplateCompatible
RequireToolAtContentStart
```

## R9-I4 — Multiple consecutive tool calls remain valid

Do not turn Round 9 into a one-call limit.

```text
CALL1CALL2
CALL1 + formatting-whitespace + CALL2
```

must remain valid.

## R9-I5 — Candidate-token atomicity

If one candidate token decodes to:

```text
</tool_call>\nvisible
```

the candidate is rejected as a whole. A constrained sampler cannot accept only the legal prefix of a token.

## R9-I6 — EOS legality is explicit

Future sampling needs a direct predicate.

```text
PreTrigger without pending marker  -> EOS legal
TextLocked                         -> EOS legal
InRegion                           -> EOS illegal
BetweenCalls clean                 -> EOS legal
partial next marker                -> EOS illegal
```

## R9-I7 — Unsupported schema guarantees are rejected

Never accept `strict:true` and then behave as non-strict.

## R9-I8 — Standard tool kind must not silently become another standard kind

```text
custom != function
```

If request kind, input semantics, response kind, streaming shape, and history cannot be preserved, Chat custom tools are unsupported.

## R9-I9 — Forced tool choice is not a prompt hint

Fields meaning:

```text
must call
specific tool
at most one tool
strict schema
```

require enforceable behavior or explicit rejection.

## R9-I10 — Inactive guarantees may be neutral

If `tool_choice:none` makes tools non-callable, fields such as `disable_parallel_tool_use:true` need not fail solely because their guarantee is vacuously satisfied.

---

# 3. Track A — R9-01: Constraint suffix semantics

## 3.1 Current strict-parser behavior

A direct parser regression already pins:

```text
CALL + non-whitespace suffix
```

as:

```text
0 structured calls
fallback_reason = TrailingContent
```

This matches the maintained native Qwen template's `NO suffix` instruction.

## 3.2 Current constraint behavior

Current `ToolCallGrammarConstraint::advance()` handles `TrailingContent` by closing the active region and replaying the tail through generic inactive handling:

```cpp
state.triggered_ = false;
state.buffer_.clear();
state.marker_prefix_.clear();
advance_inactive(state, combined.substr(progress.break_offset));
```

Thus:

```text
CALL1 + "\nDone"
```

is allowed by the constraint even though strict parsing later rejects it.

The Phase-4 design explicitly documents this as a deliberate superset approximation. Round 9 changes that design before any sampler activation.

## 3.3 Why this matters

Today it is dormant because:

```text
--constrained-tool-decoding=tool-calls-only
```

fails at startup.

If enabled unchanged:

```text
constraint permits suffix token
→ model emits suffix
→ strict parser demotes tool call
```

A future `tool-calls-only` feature would therefore not guarantee a parser-accepted tool sequence.

## 3.4 Required state model

Preferred explicit state:

```cpp
enum class ToolConstraintPhase : std::uint8_t {
    PreTrigger,
    TextLocked,
    InRegion,
    BetweenCalls,
};
```

Avoid adding another boolean such as `tool_sequence_started_` on top of `triggered_` and `entry_locked_` unless the complete boolean-state product is formally proven.

### PreTrigger

No tool has begun.

TemplateCompatible:

```text
ordinary prose -> PreTrigger
format whitespace -> PreTrigger
marker prefix -> pending
complete marker -> InRegion
EOS -> legal
```

StartOfContent:

```text
format whitespace -> PreTrigger
visible content -> TextLocked
failed marker candidate -> TextLocked
complete marker -> InRegion
EOS -> legal
```

### TextLocked

Hardened mode committed visible text before a tool entry.

```text
later bytes stay ordinary text
later markers never trigger
EOS legal
```

### InRegion

`parse_tool_call_region()` is validating a triggered tool sequence.

```text
EndOfInput legal prefix -> remain InRegion
structural definitive error -> Rejected
Complete -> BetweenCalls
TrailingContent -> replay tail with BetweenCalls rules
```

### BetweenCalls

At least one strict tool region completed.

Legal:

```text
' ' '\t' '\r' '\n'
next legal tool marker/prefix
EOS
```

Illegal:

```text
visible prose
failed marker
form feed
fence text
any other suffix content
```

## 3.5 Do not reuse one generic inactive scanner

After Round 9 there are two inactive grammars:

```text
pre-trigger lazy scan
between-calls strict scan
```

Preferred helpers:

```cpp
advance_pretrigger(...)
advance_between_calls(...)
```

A shared low-level marker-candidate helper is fine, but policy around ordinary bytes must remain explicit.

## 3.6 Shared marker classification

Keep these as syntax source of truth:

```text
classify_tool_marker_prefix(..., syntax_)
failed_marker_candidate_rescan_start(...)
parse_tool_call_region(...)
```

Do not hand-code separate `<tool_call>` recognition in the new phase logic.

## 3.7 Complete transition

On:

```text
ToolCallRegionTermination::Complete
```

transition:

```text
InRegion -> BetweenCalls
```

and clear active region buffers.

Do not return to `PreTrigger`.

## 3.8 TrailingContent transition

Replay:

```cpp
combined.substr(progress.break_offset)
```

through **BetweenCalls**, not PreTrigger.

Expected:

```text
CALL1 + whitespace            -> Allowed, BetweenCalls
CALL1 + next marker           -> Allowed / InRegion
CALL1 + partial next marker   -> NeedMore
CALL1 + visible prose         -> Rejected
CALL1 + failed marker         -> Rejected
```

## 3.9 Same-token atomic cases

### close + prose

State is inside the first call immediately before wrapper close.

Candidate bytes:

```text
</tool_call>\nDone
```

Expected:

```text
check(candidate) == Rejected
```

Never call `commit(candidate)`.

### close + next call marker

Candidate:

```text
</tool_call>\n<tool_call>
```

Expected:

```text
Allowed
second region active
```

## 3.10 Partial second marker

```text
CALL1 + "\n<tool_"
```

Expected:

```text
NeedMore
EOS illegal
```

After:

```text
"call>"
```

second region becomes active.

## 3.11 Failed second marker

```text
CALL1 + "\n<tool_x>"
```

Expected:

```text
Rejected
```

This intentionally differs from Round-8's old post-call text-lock handling. Once a strict tool sequence has begun, failed suffix markup is still illegal suffix content.

## 3.12 Intent cross behavior

TemplateCompatible may still allow:

```text
preamble prose + CALL1
```

but once CALL1 triggers:

```text
CALL1 + suffix prose
```

must be rejected.

StartOfContent still prevents the initial call from latching after visible preamble prose.

## 3.13 EOS / terminal predicate

Add:

```cpp
[[nodiscard]] bool can_terminate() const noexcept;
```

Recommended:

```text
PreTrigger, no partial marker -> true
TextLocked                    -> true
InRegion                      -> false
BetweenCalls clean            -> true
partial marker                -> false
```

Do not overload `finished()` without auditing all callers. Future sampler integration should consult `can_terminate()` for special EOS/EOT candidates rather than treating EOS as empty decoded bytes.

## 3.14 Checkpoint/restore

Required tests:

```text
CALL1
checkpoint
candidate prose -> Rejected
restore
next legal wrapper -> active
```

and:

```text
CALL1 + partial next marker
checkpoint
bad continuation -> Rejected
restore
good continuation -> active
```

All phase state must remain value-semantic.

## 3.15 Syntax modes

Native `BetweenCalls` accepts only native wrapped entry syntax.

Compatibility accepts the configured historical top-level forms through the shared classifier.

Do not hard-code a native-only marker in generic constraint logic.

## 3.16 R9-01 red/green corpus

```text
G1  CALL1                                      Allowed, terminal legal
G2  CALL1 + whitespace                         Allowed, terminal legal
G3  CALL1 + prose                              Rejected
G4  CALL1 + form-feed                          Rejected
G5  CALL1 + CALL2                              Allowed
G6  CALL1 + whitespace + CALL2                 Allowed
G7  CALL1 + partial CALL2 marker               NeedMore, EOS illegal
G8  CALL1 + failed marker                      Rejected
G9  same-token close + prose                   Rejected atomically
G10 same-token close + next marker             Allowed
G11 preamble prose + CALL1 (TemplateCompatible) allowed to trigger
G12 preamble prose + CALL1 (StartOfContent)     no trigger
```

Run native × compatibility and defined candidate partitions.

## 3.17 Files

Expected:

```text
src/models/qwen3_5/frontend/tool_call_grammar_state.h
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp
tests/test_tool_call_grammar_state.cpp
tests/test_tool_call_parser.cpp
docs/new_parser_phase4_design.md
docs/tool_call_parser.md
```

No production Stage-1/2/3 parser source change should be needed.

## 3.18 R9-01 acceptance criteria

- [ ] pre-trigger and between-calls are explicit different states;
- [ ] TemplateCompatible pre-trigger prose remains allowed;
- [ ] hardened pre-trigger lock remains unchanged;
- [ ] first completed tool enters BetweenCalls;
- [ ] whitespace suffix is legal;
- [ ] EOS after clean call/whitespace is legal;
- [ ] direct second legal wrapper is supported;
- [ ] partial next marker is NeedMore and EOS-illegal;
- [ ] visible suffix prose is Rejected;
- [ ] form feed is Rejected;
- [ ] failed next marker is Rejected;
- [ ] close+prose in one candidate is atomically Rejected;
- [ ] close+next-marker in one candidate is legal;
- [ ] checkpoint/restore exact;
- [ ] native/compat matrix green;
- [ ] explicit termination predicate tested;
- [ ] Phase-4 docs no longer describe suffix prose as intentionally unconstrained;
- [ ] runtime constrained decoding remains fail-fast.

---

# 4. Track B — R9-02: OpenAI Chat `strict:true`

## 4.1 Baseline

Current Chat function parsing validates that `strict` is boolean but accepts `true` as advisory. Prompt normalization then presents the function as non-strict.

Existing test explicitly expects this behavior.

## 4.2 Contract

`strict:true` is a schema-adherence guarantee. NInfer currently has no JSON-schema constrained sampler for Chat function arguments.

## 4.3 Required behavior

```text
strict omitted -> accept
strict false   -> accept
strict true    -> HTTP 400
                  param=tools[i].function.strict
                  code=strict_tools_not_supported
non-bool       -> existing field-type error
```

Recommended message:

```text
strict function schema enforcement requires constrained decoding, which the Engine does not provide
```

Mirror the existing Responses philosophy.

## 4.4 Do not substitute post-validation

Generating unconstrained JSON and validating it afterwards is not equivalent to strict generation. It changes streaming, retry, failure, and side-effect semantics. Do not implement `strict:true` by post-hoc validation in this round.

## 4.5 Tests

Replace the advisory test with:

```text
strict true -> strict_tools_not_supported
strict false -> accepted
strict omitted -> accepted
strict non-bool -> type error
```

Add an HTTP-level Chat error test asserting:

```text
HTTP 400
type=invalid_request_error
param=tools[0].function.strict
code=strict_tools_not_supported
```

## 4.6 Files

```text
src/serve/openai_chat_request.cpp
tests/test_openai_schema.cpp
tests/test_http_routes.cpp
docs/serving.md
```

No frontend parser change.

## 4.7 Acceptance criteria

- [ ] no `strict:true` request reaches PromptInput;
- [ ] omitted/false remain accepted;
- [ ] type validation preserved;
- [ ] exact param/code tested;
- [ ] Responses behavior unchanged;
- [ ] docs no longer call Chat strict true advisory-supported.

---

# 5. Track C — R9-03: OpenAI Chat custom tools

## 5.1 Why current behavior is not true custom-tool support

OpenAI custom tools differ from function tools in three essential ways:

```text
1. input is free-form text, not a JSON argument object;
2. output tool-call wire type is custom, not function;
3. an optional custom grammar constrains that free-form input.
```

NInfer currently violates all three.

## 5.2 Current input lowering

`custom_tool_input_schema()` synthesizes:

```json
{
  "type":"object",
  "properties":{"input":{"type":"string"}},
  "required":["input"],
  "additionalProperties":false
}
```

This turns free-form custom input into JSON function arguments.

## 5.3 Current grammar behavior

A declared custom grammar is appended to a description string only. It does not constrain sampling.

Accepting the request therefore overstates support.

## 5.4 Current output kind loss

`GeneratedToolCall` contains only:

```cpp
std::string name;
std::string arguments_json;
```

Chat response serialization unconditionally emits:

```json
{
  "type":"function",
  "function":{"name":"...","arguments":"..."}
}
```

A requested custom tool therefore comes back as a function tool call.

## 5.5 Current history mismatch

`parse_assistant_tool_calls()` accepts only:

```text
type=function
```

A standards-conformant custom tool call cannot be replayed as Chat history.

## 5.6 Recommended Round-9 fix: reject custom tools

At Chat request boundary:

```text
tools[i].type == custom
→ HTTP 400
→ param=tools[i].type
→ code=tool_type_not_supported
```

Suggested message:

```text
OpenAI custom tools require free-form custom-tool input/output semantics that NInfer does not provide
```

Do not attempt a partial full-custom implementation inside this bugfix round.

## 5.7 Remove partial lowering

Once custom definitions are rejected:

```text
custom_tool_input_schema()
```

should have no legitimate caller. Delete it if unused.

## 5.8 `allowed_tools` becomes function-only

After rejecting custom tools, an `allowed_tools` selector with:

```text
type=custom
```

must also reject with `tool_type_not_supported` at the selector's `.type` field.

`allowed_tools.mode=auto` for function tools remains supported.

## 5.9 Named custom tool choice

Round 8 already rejects named custom forcing as `tool_choice_not_supported`. Keep that path. If a request also declares a custom tool, `parse_tools()` runs first and rejects the unsupported custom definition.

## 5.10 History

Existing rejection of `type:custom` history may remain, but docs/error wording must make clear that Chat custom tools are unsupported, not partially supported.

## 5.11 No response serializer refactor required

If custom requests fail before generation, supported Chat tool calls are function calls only. The existing function serializer remains valid.

Add a regression/comment proving this invariant rather than adding unused wire-kind machinery.

## 5.12 Compatibility risk

The source mentions GitHub Copilot CLI/MCP clients as motivation for the synthetic custom-to-function translation.

Before shipping, test those clients. If a real compatibility dependency exists, do **not** silently keep the standard field with wrong semantics. Consider a separate explicit non-standard opt-in such as:

```text
--chat-custom-tools-as-functions
```

Default must remain off if introduced. The flag is not required unless a concrete client regression proves the need.

## 5.13 Why full custom support is a separate project

Complete support needs at least:

```text
ToolDefinition wire kind
GeneratedToolCall wire kind or protocol identity map
free-form input separate from arguments_json
prompt strategy for custom free text
raw-input parser normalization
aggregate custom serializer
streaming custom serializer
custom history parser
tool-result roundtrip
allowed_tools identity by kind + name
custom grammar representation
sampler grammar integration
logging/tests across all paths
```

Do not mix that project into Round 9.

## 5.14 Tests

Replace current "custom tools are served as single-string-input function" success test with explicit rejection.

Required:

```text
custom without format -> rejected
custom with grammar -> rejected
allowed_tools custom selector -> rejected
custom history -> explicitly unsupported
function tools -> unchanged green
```

HTTP test:

```text
HTTP 400
param=tools[0].type
code=tool_type_not_supported
```

## 5.15 Acceptance criteria

- [ ] custom definition rejected;
- [ ] grammar-bearing custom definition rejected;
- [ ] dead custom lowering helper removed;
- [ ] allowed custom selector rejected;
- [ ] function tools unchanged;
- [ ] named custom forcing remains explicit unsupported;
- [ ] custom history explicit unsupported;
- [ ] docs stop claiming custom support;
- [ ] no partial ToolKind refactor in this patch;
- [ ] compatibility impact tested/documented.

---

# 6. Track D — R9-04: Anthropic tool guarantees

## 6.1 Baseline

Current internal selection enum recognizes:

```text
Auto
None
Any
Named
```

but `lower_tools()` converts `Any` and `Named` into automatic selection after validation.

`disable_parallel_tool_use` is parsed then discarded.

Tool `strict:true` is type-checked but not enforced.

Tests explicitly expect these values to remain advisory.

## 6.2 Required `any`

With callable tools:

```text
tool_choice.type=any
```

requires tool use on Anthropic models supporting that mode. NInfer cannot guarantee that.

Reject:

```text
status=400
param=tool_choice
code=tool_choice_not_supported
```

## 6.3 Required named `tool`

Preserve declared-name validation first.

```text
unknown name -> existing unknown-tool error
known name   -> tool_choice_not_supported
```

Do not narrow the tool set and call that forcing.

## 6.4 Persist `disable_parallel_tool_use`

Extend:

```cpp
struct ToolSelection {
    ToolSelectionKind kind = ToolSelectionKind::Auto;
    std::string name;
    bool disable_parallel_tool_use = false;
};
```

Do not discard the parsed boolean.

Behavior:

```text
None + true/false                         -> accepted neutral
Auto + false                              -> accepted
Auto + true + callable tools              -> reject parallel_tool_calls_not_supported
Any/Named + true                          -> forced-choice rejection is primary
```

Recommended param:

```text
tool_choice.disable_parallel_tool_use
```

## 6.5 Persist tool strict state

Extend `ParsedTool`:

```cpp
bool strict = false;
std::size_t source_index = 0;   // recommended for precise diagnostics
```

Parsing:

```text
strict omitted -> false
strict false   -> false
strict true    -> true
non-bool       -> existing type error
```

For callable tools:

```text
strict true -> strict_tools_not_supported
```

For `tool_choice:none`, strict may remain neutral if tools cannot execute. This matches the existing useful neutral treatment of inactive `defer_loading`/`allowed_callers` constraints.

Prefer param:

```text
tools[i].strict
```

if source index is retained.

## 6.6 Do not fake guarantees with prompting

Do not implement:

```text
any/named -> add "MUST call" instruction
```

Prompt compliance is not an API guarantee.

## 6.7 Do not fake serial mode post-generation

For `disable_parallel_tool_use:true`, do not:

```text
keep first call, drop later calls
convert extra calls to text
```

That changes model intent after generation. Fail-fast is the correct Round-9 behavior.

## 6.8 Qwen Code compatibility risk

Source comments state Qwen Code sends `tool_choice:any` on some JSON side queries. Strict fail-fast can therefore break an existing compatibility path.

Required process:

1. test the current Qwen Code integration;
2. record whether `any` is actually required for successful operation;
3. if compatibility is needed, introduce an **explicit non-standard opt-in** rather than silently retaining incorrect default semantics.

Possible later flag:

```text
--anthropic-advisory-tool-choice
```

If added:

```text
default off
request logs record compatibility downgrade
docs mark it non-standard
tests cover strict default and compatibility mode
```

Do not add this flag speculatively before compatibility evidence exists.

## 6.9 Error rendering nuance

Internal `ApiError` carries:

```text
status
type
param
code
message
```

Anthropic external body currently renders:

```text
type
error.type
error.message
request_id
```

Therefore:

```text
unit request tests -> assert internal param/code
HTTP/body tests    -> assert Anthropic envelope + message + request_id
```

Do not expect OpenAI-style `param`/`code` fields in Anthropic JSON unless that serializer is intentionally redesigned.

## 6.10 Count Tokens parity

`/v1/messages/count_tokens` shares prompt normalization. Review whether output-only tool-choice guarantees should be rejected identically or intentionally ignored for counting.

Do not leave an accidental mismatch. Document the decision and test both endpoints.

## 6.11 Tests

Replace the advisory success loop with individual cases:

```text
any + tools                         -> tool_choice_not_supported
named declared tool                 -> tool_choice_not_supported
named unknown tool                  -> existing unknown-tool error
auto + disable_parallel false       -> accepted
auto + disable_parallel true+tools  -> parallel_tool_calls_not_supported
none + disable_parallel true        -> accepted neutral
strict false                        -> accepted
strict true + callable tool         -> strict_tools_not_supported
strict true + none                  -> neutral if chosen policy
non-bool strict                     -> type error
non-bool disable_parallel           -> type error
```

## 6.12 Files

Expected:

```text
src/serve/anthropic_messages_request.cpp
tests/test_anthropic_schema.cpp
HTTP route/error tests
docs/serving.md
```

No Qwen parser change.

## 6.13 Acceptance criteria

- [ ] any rejected;
- [ ] named declared tool rejected as unsupported forcing;
- [ ] named unknown error preserved;
- [ ] auto + disable false accepted;
- [ ] auto + disable true + callable tools rejected;
- [ ] none + disable true neutral;
- [ ] strict false accepted;
- [ ] strict true callable rejected;
- [ ] strict true none policy pinned;
- [ ] validation types preserved;
- [ ] internal codes/params tested;
- [ ] Anthropic external envelope tested;
- [ ] Count Tokens parity reviewed;
- [ ] Qwen Code compatibility tested;
- [ ] no prompt-only or post-hoc fake enforcement.

---

# 7. Implementation phases

## R9-0 — freeze baseline

Record:

```text
current SHA
worktree status
toolchain
build directory
GPU-hidden policy
```

Run targeted baseline before modifications.

## R9-1 — add R9-01 red tests

Pin current unwanted acceptance:

```text
CALL+prose
CALL+form-feed
CALL+failed marker
same-token close+prose
```

Positive controls:

```text
CALL+whitespace
CALL+next call
CALL+whitespace+next call
```

## R9-2 — explicit constraint phases

Introduce `PreTrigger/TextLocked/InRegion/BetweenCalls` or equivalent.

## R9-3 — BetweenCalls scanner

Only formatting whitespace, legal next marker, or termination may survive.

Replay `TrailingContent` from `break_offset` through this scanner.

## R9-4 — termination predicate

Add/test `can_terminate()` or equivalent.

## R9-5 — constraint cross-matrix

Run intent × syntax × token-boundary × checkpoint matrices.

## R9-6 — Chat strict red tests

Change expected result from advisory success to capability rejection.

## R9-7 — Chat strict fail-fast

No Engine changes.

## R9-8 — Chat custom red tests

Pin current synthetic function lowering and grammar advisory behavior.

## R9-9 — reject Chat custom

Reject definitions and allowed selectors; delete dead helper; update docs.

## R9-10 — Anthropic red tests

Split advisory loop into independent failures.

## R9-11 — Anthropic fail-fast

Persist `disable_parallel_tool_use` and `strict`, validate and reject unsupported guarantees.

## R9-12 — HTTP/error tests

At minimum:

```text
OpenAI strict:true
OpenAI custom tool
Anthropic any or strict:true
```

## R9-13 — docs

Update serving capabilities and Phase-4 semantics.

## R9-14 — full regression

Targeted CPU and full GPU-hidden CTest. Do not inherit Round-8 green status.

---

# 8. OpenAI Chat matrix after Round 9

| Request | Expected |
|---|---|
| function, strict omitted | accept |
| function, strict false | accept |
| function, strict true | reject `strict_tools_not_supported` |
| custom tool | reject `tool_type_not_supported` |
| tool_choice auto | accept |
| tool_choice none | accept |
| tool_choice required | reject |
| named function | reject |
| named custom | reject |
| allowed_tools auto + function | accept |
| allowed_tools auto + custom | reject |
| allowed_tools required | reject |
| parallel true | accept |
| parallel false + callable functions | reject |
| parallel false + none/no tools | accept |

---

# 9. Anthropic matrix after Round 9

| Request | Expected |
|---|---|
| auto | accept |
| none | accept |
| any | reject |
| named declared tool | reject forcing |
| named unknown tool | unknown-tool error |
| auto + disable_parallel false | accept |
| auto + disable_parallel true + tools | reject |
| none + disable_parallel true | accept neutral |
| strict false | accept |
| strict true + callable | reject |
| strict true + none | neutral if documented policy |
| existing server/toolset unsupported cases | unchanged |

---

# 10. Recommended error codes

```text
OpenAI Chat strict          strict_tools_not_supported
OpenAI Chat custom          tool_type_not_supported
Anthropic any/named         tool_choice_not_supported
Anthropic disable parallel  parallel_tool_calls_not_supported
Anthropic strict            strict_tools_not_supported
```

Reuse existing Responses codes where possible.

---

# 11. Error precedence

### Chat strict

```text
non-bool -> field validation error
true     -> capability error
```

### Chat custom

```text
malformed tools entry -> structural error
type=custom           -> capability error
```

Do not deeply validate unsupported custom grammar internals first unless project conventions explicitly require it.

### Anthropic named

```text
malformed -> structural error
unknown name -> unknown-tool error
known name -> unsupported forcing
```

### Anthropic strict

```text
non-bool -> structural error
true + callable -> capability error
```

---

# 12. HTTP contract tests

### OpenAI

Assert external body:

```text
HTTP status
type
message
param
code
```

### Anthropic

Assert:

```text
HTTP status
body.type=error
error.type=invalid_request_error
message
request_id
```

Assert internal `ApiError.param/code` separately.

---

# 13. Request-log checks

Capability rejection must occur before generation. If request logging records synchronous preparation failure, add one OpenAI and one Anthropic Round-9 test asserting the exact code/param and that no generation result is recorded.

---

# 14. OpenAI Responses regression gate

Responses must remain unchanged:

```text
strict true rejected
required rejected
named rejected
allowed required rejected
parallel false rejected when callable
normal function tools remain supported
```

No Responses source change should normally be required.

---

# 15. Native parser regression gate

Round 9 must not change:

```text
Stage 1
Stage 2
Stage 3
FailClosed
tolerant recovery
declared-name enforcement
fence behavior
FinishReason handling
```

Re-run the complete previous corpus anyway.

---

# 16. Mandatory previous-round regressions

```text
F1 opaque parameter payload
F2 open function non-executable
F3 EOF boundary
F4 partial opener
wrapper balance
mixed parameter families
breaking '<'
failed-wrapper scope
function_calls balance
trailing-content parser behavior
entry composition
declared identity strict+tolerant
fenced examples
work bounds
FailClosed ambiguity
FinishReason transaction
native/compat syntax
R6 intent policy
R7 later retry cutoff
R7 decoder intent propagation
R7 hardened template split
R8 post-call hardened intent
R8 Chat forced-tool fail-fast
```

---

# 17. CPU test commands

Discover names first:

```powershell
ctest --test-dir build-new-parser -C Release -N
```

Suggested targeted family:

```powershell
ctest --test-dir build-new-parser -C Release `
  -R "ninfer_(tool_call_parser|tool_call_grammar|tool_call_grammar_state|qwen3_5_frontend|openai_schema|openai_responses|anthropic_schema|http_routes|serve_options|request_log)_test" `
  --output-on-failure `
  --parallel 16
```

Use actual registered names.

Full GPU-hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES="99"
ctest --test-dir build-new-parser -C Release `
  -E "_real" `
  --output-on-failure `
  --parallel 16
```

Record selected/passed/failed/skipped/not-run/exit-code. Never call a nonzero exit green.

---

# 18. Review loops

## Loop A — Constraint state validity

Enumerate every phase and transition. If booleans remain, prove impossible combinations cannot occur. Prefer enum states.

## Loop B — Token atomicity

Explicitly test state changes occurring inside one candidate token:

```text
open -> complete -> prose
open -> complete -> next marker
between-calls partial marker -> completion
```

No partial candidate commit.

## Loop C — EOS

Record `can_terminate()` for every phase and pending-prefix state.

## Loop D — OpenAI standard guarantee audit

Search:

```text
advisory
strict
custom
grammar
tool_choice
parallel_tool_calls
```

No accepted standard field may silently lose promised semantics.

## Loop E — Function wire round-trip

After custom fail-fast, prove supported Chat tools have this consistent round-trip:

```text
function request
→ PromptInput function
→ GeneratedToolCall
→ type=function response
→ type=function history
→ next PromptInput
```

## Loop F — Anthropic guarantee audit

Search:

```text
tool_choice
any
disable_parallel_tool_use
strict
advisory
```

No unsupported standard guarantee may silently become Auto.

## Loop G — Qwen Code compatibility

Test real integration before shipping R9-04. If `any` is required, use an explicit compatibility mode rather than wrong standard defaults.

## Loop H — Documentation truthfulness

Remove stale claims:

```text
Chat strict:true advisory-supported
Chat custom tools supported
Anthropic forced choices advisory
Anthropic disable_parallel advisory
Anthropic strict advisory
post-call suffix intentionally unconstrained in future tool-calls-only mode
```

---

# 19. Suggested commit sequence

```text
1.  test(constraint): reproduce strict post-call suffix asymmetry
2.  refactor(constraint): distinguish pre-trigger and between-calls phases
3.  fix(constraint): reject visible suffix after structured tool sequence starts
4.  feat(constraint): expose terminal-acceptance predicate
5.  test(constraint): token/checkpoint/intent/syntax matrix
6.  test(chat): pin strict-tool rejection
7.  fix(chat): reject strict function tools without constrained schema generation
8.  test(chat): pin custom-tool incompatibility
9.  fix(chat): reject unsupported custom tools/selectors
10. test(anthropic): pin forced/serial/strict capability errors
11. fix(anthropic): reject unsupported tool guarantees
12. test(http): verify OpenAI/Anthropic error envelopes
13. docs: update constrained-decoding state semantics
14. docs: update Chat capabilities
15. docs: update Anthropic capabilities
16. docs: record Round-9 progress
```

Do not squash before review.

---

# 20. Definition of Done — R9-01

- [ ] baseline red reproducer proves suffix acceptance;
- [ ] pre-trigger and between-calls are different states;
- [ ] TemplateCompatible pre-trigger prose remains allowed;
- [ ] hardened pre-trigger lock remains;
- [ ] post-call prose rejected under both intents;
- [ ] form-feed suffix rejected;
- [ ] failed next marker rejected;
- [ ] whitespace-only suffix accepted;
- [ ] EOS after complete call/whitespace accepted;
- [ ] direct second call accepted;
- [ ] partial next marker NeedMore and EOS-illegal;
- [ ] same-token close+prose rejected atomically;
- [ ] same-token close+next-marker accepted;
- [ ] checkpoint/restore exact;
- [ ] native/compat matrix green;
- [ ] explicit terminal predicate tested;
- [ ] Phase-4 docs corrected;
- [ ] runtime constrained decoding still fail-fast.

---

# 21. Definition of Done — R9-02

- [ ] strict omitted accepted;
- [ ] strict false accepted;
- [ ] strict true rejected;
- [ ] non-bool strict keeps type error;
- [ ] no strict true reaches PromptInput;
- [ ] HTTP error tested;
- [ ] Responses unchanged;
- [ ] docs updated.

---

# 22. Definition of Done — R9-03

- [ ] custom definition rejected;
- [ ] grammar-bearing custom rejected;
- [ ] custom lowering helper removed if unused;
- [ ] allowed custom selector rejected;
- [ ] named custom forcing explicit unsupported;
- [ ] custom history explicit unsupported;
- [ ] function tools unchanged;
- [ ] docs stop claiming custom support;
- [ ] compatibility impact tested;
- [ ] no incomplete full-custom refactor mixed into patch.

---

# 23. Definition of Done — R9-04

- [ ] any rejected;
- [ ] named declared tool rejected as unsupported forcing;
- [ ] named unknown preserves existing error;
- [ ] auto + disable false accepted;
- [ ] auto + disable true + callable tools rejected;
- [ ] none + disable true neutral;
- [ ] strict false accepted;
- [ ] strict true callable rejected;
- [ ] strict true none policy pinned;
- [ ] internal param/code tested;
- [ ] Anthropic body tested;
- [ ] Count Tokens parity reviewed;
- [ ] Qwen Code compatibility tested;
- [ ] no prompt-only/post-hoc fake enforcement.

---

# 24. Known residual

A complete native sequence such as:

```text
<tool_call>
<function=bash>
<parameter=command>
echo example
</parameter>
</function>
</tool_call>
```

can still mean either execution or quotation when no additional intent signal exists. No byte parser can infer author intent from identical bytes. This remains a protocol-level residual, not a Round-9 implementation bug.

Long-term solution:

```text
explicit TEXT vs TOOL intent channel
```

---

# 25. Future constrained-decoding activation gate

Do not remove startup fail-fast until at least:

- [ ] R9 post-trigger suffix grammar;
- [ ] explicit EOS legality;
- [ ] lazy pre-trigger behavior;
- [ ] hardened intent behavior;
- [ ] native/compat syntax;
- [ ] checkpoint/rollback;
- [ ] speculative accepted-prefix semantics;
- [ ] correction/bonus-token legality;
- [ ] vocabulary token-byte decoding;
- [ ] EOS/special-token handling;
- [ ] hard mask before sampling filters;
- [ ] CUDA graph stable mask shape/address;
- [ ] final parser/constraint acceptance corpus;
- [ ] streaming finalization;
- [ ] GPU runtime correctness;
- [ ] performance benchmarks;
- [ ] no semantic-intent overclaim.

---

# 26. Primary references

OpenAI Function Calling:

```text
https://developers.openai.com/api/docs/guides/function-calling
```

OpenAI Chat Completions API:

```text
https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions
```

OpenAI Structured Outputs:

```text
https://developers.openai.com/api/docs/guides/structured-outputs
```

Anthropic API primer / forced tool use:

```text
https://platform.claude.com/docs/en/claude_api_primer
```

Anthropic parallel tool use:

```text
https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use
```

Anthropic strict tool use:

```text
https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
```

---

# 27. Final sign-off wording if Round 9 is green

Acceptable:

> No remaining HIGH/MEDIUM implementation defect was found in the reviewed active Qwen3.8 native parser path or the reviewed tool-control API paths under the CPU-only verified scope. The future grammar-constraint core now preserves strict post-trigger tool-sequence semantics: visible suffix prose is rejected while legal termination and directly consecutive tool entries remain supported. OpenAI Chat no longer accepts strict schema guarantees or custom-tool wire semantics that NInfer cannot implement, and Anthropic no longer silently weakens forced, serial, or strict tool controls to automatic best-effort behavior. Runtime constrained decoding remains unavailable and fail-fast. The byte-identical semantic quotation residual remains documented.

Do **not** claim:

```text
bug-free
all tool-call errors impossible
full OpenAI parity
full Anthropic parity
semantic quotation solved
```

---

# 28. Engineering conclusion

Round 9 should remain a set of narrow correctness patches, not another parser rewrite.

The core rules are:

```text
R9-01:
"inactive" is not one state.
Before first trigger, prose may be legal.
After tool sequence start, strict suffix grammar applies.

R9-02:
strict:true is a guarantee, not metadata.

R9-03:
custom is a distinct wire/input contract, not a function with {"input":"..."}.

R9-04:
forced/serial/strict tool controls are guarantees, not advisory prompt preferences.
```

The active native Stage-1/2/3 parser should remain untouched unless a new direct parser-core reproducer appears.
