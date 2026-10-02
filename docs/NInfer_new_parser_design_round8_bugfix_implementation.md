# NInfer `new_parser_design` — Round 8 Bugfix / Integration Specification

## 0. Purpose

This document is the **Round-8 implementation specification** for the remaining findings discovered after the full-chain review of `Hundsbuah/ninfer:new_parser_design` at:

```text
ad71244de33212aa92742668502b29c3a72528a3
```

Round 8 is intentionally split into two independent tracks:

```text
Track A — R8-01
ToolCallGrammarConstraint intent-state parity with the production parser

Track B — R8-02
OpenAI Chat Completions tool-control contract correctness
```

Do not merge the two concerns. `R8-01` is a future constrained-decoding architecture defect. `R8-02` is an HTTP/API contract defect. Neither finding justifies another Stage-1/Stage-2/Stage-3 parser rewrite.

The goals are:

1. eliminate the parser-vs-constraint split for `RequireToolAtContentStart`;
2. distinguish a true consecutive tool sequence from `CALL -> visible prose -> later CALL`;
3. remove the `latched_once_` policy that permanently disables text locking after the first tool;
4. stop using `marker_suffix()/rfind('<')` to skip bytes that must participate in the intent gate;
5. preserve speculative checkpoint/restore semantics;
6. make Chat reject tool-control guarantees NInfer cannot actually enforce instead of silently weakening them to `auto`;
7. preserve supported `auto`, `none`, and `allowed_tools.mode=auto` behavior;
8. preserve the existing Responses contract;
9. keep constrained runtime decoding fail-fast;
10. keep the call-only semantic quotation residual explicitly documented.

---

# 0.1 Pinned baseline

Repository:

```text
Hundsbuah/ninfer
```

Branch:

```text
new_parser_design
```

Reviewed Round-8 baseline HEAD:

```text
ad71244de33212aa92742668502b29c3a72528a3
```

Tree:

```text
e4c72f00245d656b1c23dcf9f9521b1d64674c25
```

Relevant Round-7 commits:

```text
7565c40b  fix(frontend): forbid later retry bases in start-of-content mode
3037d93d  fix(frontend): preserve intent policy in decoder terminal reparse
73187827  fix(frontend): R7-02 grammar-constraint intent content-start gate
6fa91ac4  test(frontend): add R7-01 intent retry cross-matrix corpus
5236026e  feat(templates): split hardened tool instruction into explicit template
0408781f  fix(tests): preserve CTest skip code 77 in LinearSwiGLU wrappers
ad71244d  docs: record round-7 parser bugfix progress and spec
```

At review time:

```text
GitHub combined status checks: none
GitHub workflow runs: none
```

The repository's own Round-7 progress reports:

```text
targeted:
  8 selected
  7 passed
  0 failed
  1 skipped
  exit 0

full GPU-hidden:
  147 total
  46 passed
  0 failed
  101 skipped
  exit 0
```

These results are repository-reported rather than independently reproduced by the reviewer.

Before implementation, record the actual current branch SHA again. If it differs, re-verify every code location before applying this document mechanically.

---

# 0.2 Current product policy

At the pinned baseline:

```text
OutputOptions.tool_call_syntax     = QwenWrappedNative
OutputOptions.tool_call_ambiguity  = FailClosed
OutputOptions.tool_call_intent     = TemplateCompatible
OutputOptions.tolerant_tool_calls  = false
```

Optional hardened profile:

```text
--tool-call-syntax qwen-wrapped
--tool-call-ambiguity fail-closed
--tool-call-intent start-of-content
--chat-template tools/chat_templates/qwen3_8_hardened_tools.jinja
```

Runtime constrained tool decoding remains unavailable:

```text
--constrained-tool-decoding=tool-calls-only
-> startup error
```

Keep that fail-fast behavior.

---

# 0.3 Current Qwen3.8 native contract

The current official Qwen3.8-27B template uses:

```text
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value
</parameter>
</function>
</tool_call>
```

The official template states conceptually:

```text
- wrapped <tool_call> form;
- required parameters must be present;
- optional natural-language reasoning may appear BEFORE the function call;
- no suffix after the function call.
```

Primary source:

```text
https://huggingface.co/Qwen/Qwen3.8-27B/blame/main/chat_template.jinja
```

Qwen's function-calling documentation also warns that malformed tool calls can occur in corner cases and recommends robust parsing:

```text
https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md
```

Therefore `QwenWrappedNative` remains the correct production syntax default. `RequireToolAtContentStart` remains an NInfer hardening extension.

---

# 0.4 Current OpenAI Chat contract

Current OpenAI Chat semantics are:

```text
tool_choice = none
    no tool call

tool_choice = auto
    model may emit text or one/more calls

tool_choice = required
    model must call one or more tools

named tool choice
    force the named tool

parallel_tool_calls = false
    exactly zero or one tool call
```

Primary references:

```text
https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions
https://developers.openai.com/api/docs/guides/function-calling
```

At the Round-8 baseline, NInfer Chat accepts several stronger requests but silently weakens them to automatic selection. That is R8-02.

---

# 0.5 Scope discipline

Do **not** opportunistically change unrelated advisory behavior in this round.

Explicit non-goals unless separately reviewed:

```text
strict:true behavior
Anthropic tool_choice:any / named advisory behavior
JSON-schema constrained arguments
hosted tools
MCP execution
custom-tool grammar enforcement
runtime sampler grammar activation
Stage-2 parser rewrite
TEXT-vs-TOOL special-token architecture
```

Round 8 is specifically about:

```text
R8-01:
  post-call hardened intent parity in ToolCallGrammarConstraint

R8-02:
  Chat required / named / allowed_tools.required / parallel_tool_calls=false
```

---

# 0.6 GPU constraint

Do not run:

```text
model inference
GPU correctness tests
_real model tests
CUDA memcheck/racecheck
server model smoke tests
benchmarks
GPU constrained-decoding integration tests
```

Allowed:

```text
CPU parser tests
CPU grammar-constraint tests
request-schema tests
HTTP route/error tests
template tests
serve-option tests
GPU-hidden CPU CTest
static/source audits
deterministic outcome dumps
```

---

# 1. Findings

| ID | Finding | Severity | Type | Action |
|---|---|---:|---|---|
| **R8-01** | Constraint permits `CALL1 -> prose -> CALL2` to trigger CALL2 under `RequireToolAtContentStart`, while production parser forbids the later entry | **MEDIUM** | architecture / future constrained-decoding blocker | **FIX** |
| **R8-01a** | `latched_once_` permanently disables re-locking after the first tool | MEDIUM root cause | state machine | **FIX** |
| **R8-01b** | `marker_suffix(combined)` can skip intent-relevant trailing bytes | MEDIUM root cause | replay/rescan | **FIX** |
| **R8-02** | Chat `tool_choice:"required"` silently maps to Auto | **MEDIUM** | API contract | **FAIL FAST** |
| **R8-02b** | Chat named function/custom choice silently maps to Auto | MEDIUM | API contract | **FAIL FAST** |
| **R8-02c** | Chat `allowed_tools.mode="required"` narrows then maps to Auto | MEDIUM | API contract | **FAIL FAST** |
| **R8-02d** | Chat `parallel_tool_calls:false` is advisory although multiple calls may still occur | MEDIUM | API contract | **FAIL FAST when tools callable** |
| R6-04 | call-only quotation is byte-identical to genuine call | residual | protocol | **DOCUMENT** |
| constrained runtime | sampler constraint not integrated | deferred | non-goal | **KEEP FAIL-FAST** |

---

# 2. Round-8 invariants

## R8-I1 — Parser and constraint must agree after a completed tool

Under `RequireToolAtContentStart`:

```text
CALL1
formatting-whitespace only
CALL2
```

keeps CALL2 eligible.

But:

```text
CALL1
visible prose
CALL2
```

must make CALL2 ordinary text.

This must agree between the production parser and `ToolCallGrammarConstraint`.

---

## R8-I2 — First latch is not permanent permission

The state:

```text
a tool happened once
```

must not mean:

```text
all future markers are eligible
```

Only formatting whitespace plus an immediate valid next tool entry preserves the structured tool sequence.

---

## R8-I3 — Trailing bytes are replayed semantically

After a complete tool region followed by trailing bytes, do not jump to the last `<` with `rfind`.

Use the parser's:

```text
progress.break_offset
```

and run the suffix through the same inactive intent-state transitions as ordinary bytes.

---

## R8-I4 — Consecutive wrappers remain supported

Round 8 must not become a one-tool-only policy.

These remain valid:

```text
<tool_call>CALL1</tool_call><tool_call>CALL2</tool_call>
```

and:

```text
<tool_call>CALL1</tool_call>

<tool_call>CALL2</tool_call>
```

when the gap contains only formatting whitespace.

---

## R8-I5 — Constraint state remains value-semantic

All new state must naturally participate in:

```text
checkpoint()
restore()
```

Speculative rollback must recover the exact tool-entry eligibility state.

---

## R8-I6 — Prose is not a grammar error

Visible prose after a completed call should:

```text
lock future tool entry
```

not:

```text
return Rejected
```

The output is still legal text.

---

## R8-I7 — Unsupported API guarantees are rejected

If Chat asks NInfer to guarantee:

```text
at least one tool call
one exact named tool
at least one tool from allowed subset
maximum one tool call
```

and the Engine cannot guarantee it, return HTTP 400. Do not silently change the request to `auto`.

---

## R8-I8 — Supported Chat modes remain supported

Preserve:

```text
tool_choice omitted
tool_choice auto
tool_choice none
allowed_tools mode auto
parallel_tool_calls true
parallel_tool_calls false when no effective tool is callable
```

---

## R8-I9 — Responses behavior remains unchanged

The Responses adapter already follows fail-fast semantics for these unsupported guarantees. Round 8 must not regress it.

---

# 3. R8-01 — Grammar constraint post-call divergence

## 3.1 Verified baseline state

Current fields:

```cpp
bool triggered_ = false;
bool entry_locked_ = false;
bool latched_once_ = false;
```

Current inactive lock:

```cpp
else if (state.intent_ == ToolCallIntentPolicy::RequireToolAtContentStart &&
         !state.latched_once_ &&
         !is_tool_format_whitespace(byte)) {
    state.entry_locked_ = true;
}
```

Current failed-candidate lock:

```cpp
if (state.intent_ == ToolCallIntentPolicy::RequireToolAtContentStart &&
    !state.latched_once_) {
    state.entry_locked_ = true;
}
```

Current trigger:

```cpp
state.triggered_ = true;
state.latched_once_ = true;
```

Once the first tool triggers, visible prose never locks the gate again.

---

## 3.2 Contradictory tests already exist

The current grammar-state test explicitly expects:

```text
CALL1
prose
CALL2
```

to activate CALL2 under hardened intent.

The Round-7 parser tests expect the corresponding hardened output to produce no later executable call.

Therefore the repository currently contains contradictory semantics between parser and constraint tests.

This is the direct R8-01 proof.

---

## 3.3 Correct minimal state model

The constraint only needs two effective inactive eligibility states:

```text
EntryOpen
TextLocked
```

with `triggered_` representing the active structured region.

Transitions:

```text
START
  EntryOpen

EntryOpen + formatting whitespace
  -> EntryOpen

EntryOpen + valid tool marker
  -> triggered = true

EntryOpen + ordinary visible byte
  -> TextLocked

EntryOpen + failed marker candidate
  -> TextLocked

triggered + complete region
  -> triggered = false
  -> EntryOpen

triggered + complete region + visible trailing bytes
  -> triggered = false
  -> replay tail
  -> TextLocked

TextLocked + anything
  -> TextLocked
  -> never trigger again
```

A dedicated enum is optional; one correctly managed `entry_locked_` boolean is sufficient.

---

## 3.4 Preferred code change

### Minimal option

Remove:

```cpp
bool latched_once_;
```

Keep:

```cpp
bool entry_locked_ = false;
```

Lock on visible inactive content regardless of whether a previous call existed:

```cpp
if (state.intent_ == ToolCallIntentPolicy::RequireToolAtContentStart &&
    !is_tool_format_whitespace(byte)) {
    state.entry_locked_ = true;
}
```

A failed marker candidate similarly locks whenever hardened intent is active.

### Explicit enum option

If a clearer representation is desired:

```cpp
enum class ToolIntentGate : std::uint8_t {
    Open,
    TextLocked,
};
```

Do not add a permanent `HadTool` exemption.

---

# 3.5 Why `marker_suffix()` is wrong for intent

Current helper:

```cpp
std::string marker_suffix(const std::string& text) {
    const std::size_t pos = text.rfind('<');
    return pos == std::string::npos ? std::string{} : std::string(text.substr(pos));
}
```

For:

```text
CALL1
visible prose
<tool_call>
```

this can jump directly to:

```text
<tool_call>
```

and skip:

```text
visible prose
```

which is exactly the evidence required to enter `TextLocked`.

Therefore `marker_suffix(combined)` must not be used as the post-call intent-state replay mechanism.

---

# 3.6 Correct `Complete` transition

When:

```cpp
progress.termination == ToolCallRegionTermination::Complete
```

there is no ordinary trailing text in the successfully parsed slice.

Preferred state update:

```cpp
state.triggered_ = false;
state.buffer_.clear();
state.marker_prefix_.clear();
```

Leave the hardened entry gate open for a directly consecutive wrapper.

Do not retain `</tool_call>` as a future marker candidate.

---

# 3.7 Correct `TrailingContent` transition

The parser already reports:

```cpp
progress.break_offset
```

at the byte that proves trailing content.

For:

```text
</tool_call>\n\nP
```

`skip_ws()` advances to `P`, so `break_offset` is the correct semantic replay frontier.

Required sequence:

```text
1. first structured region completed
2. state.triggered_ = false
3. clear active buffer
4. clear stale marker candidate
5. tail = combined.substr(progress.break_offset)
6. feed tail through the ordinary inactive marker/content state machine
7. first visible byte locks hardened intent
8. later tool markers remain text
```

Do not search for the final `<`.

---

# 3.8 Refactor inactive scanning into one authority

Preferred design: factor the inactive scanner into one helper or local lambda used by both:

```text
normal inactive bytes
TrailingContent replay bytes
```

It must own:

```text
formatting whitespace handling
marker start
marker continuation
NeedMore
Complete marker trigger
NotMarker candidate handling
failed-marker breaking '<' rescan
hardened text locking
```

Do not duplicate these transitions.

Conceptual shape:

```cpp
InactiveAdvance advance_inactive(
    ToolCallGrammarConstraint& state,
    std::string_view bytes);
```

The exact type is implementation-dependent.

---

# 3.9 Avoid recursive `advance()`

Do not recursively call `advance(tail, ...)` from inside `advance()` unless the state-copy semantics are proven.

`check()` runs on a copy while `commit()` writes target state. Recursion makes target/copy semantics less obvious and creates avoidable depth.

Prefer an iterative replay buffer or a shared inactive helper.

---

# 3.10 Same-token reproducer

This must work when a single decoded token or `commit()` contains:

```text
CALL1 + "\nvisible prose\n" + CALL2
```

Expected hardened behavior:

```text
CALL1 syntax validated
visible prose closes future tool eligibility
CALL2 does not activate a new structured region
```

Do not rely on token boundaries.

---

# 3.11 Split reproducer

Also test:

```text
commit(CALL1)
commit("\nvisible ")
commit("prose\n<tool_")
commit("call>")
```

CALL2 must never trigger under hardened intent.

---

# 3.12 Positive consecutive-call control

Input:

```text
CALL1 + "\n\t \r\n" + CALL2
```

Expected:

```text
second wrapper remains eligible
```

Test whole chunk, split chunks, and bytewise commits.

---

# 3.13 Failed-marker after CALL1

Input:

```text
CALL1 + "<tool_x>" + CALL2
```

Expected hardened behavior:

```text
failed candidate is visible content
-> TextLocked
-> CALL2 ordinary text
```

---

# 3.14 Partial immediate CALL2

Chunk 1:

```text
CALL1 + "\n<tool_"
```

Chunk 2:

```text
"call>"
```

Expected:

```text
second wrapper triggers
```

No visible content appeared between calls.

---

# 3.15 Partial failed CALL2

Chunk 1:

```text
CALL1 + "\n<tool_"
```

Chunk 2:

```text
"x>"
```

Chunk 3:

```text
CALL2
```

Expected hardened behavior:

```text
failed candidate locks
later CALL2 blocked
```

---

# 3.16 Checkpoint after CALL1

Required speculative-decoding test:

```text
commit(CALL1)
checkpoint
commit("visible prose")
-> locked
restore(checkpoint)
commit("\n<tool_call>")
-> trigger
```

This proves the post-call gate is value-semantic.

---

# 3.17 Checkpoint with partial marker

```text
CALL1
<tool_
checkpoint
x> -> failed candidate locks
restore
call> -> trigger
```

This catches incomplete marker-state rollback bugs.

---

# 3.18 TemplateCompatible control

Under:

```text
TemplateCompatible
```

preserve historical semantics. Do not accidentally apply hardened post-call locking globally.

---

# 3.19 Syntax cross-product

Run the post-call corpus under:

```text
QwenWrappedNative
Compatibility
```

Intent and syntax remain separate dimensions.

---

# 3.20 Parser/constraint equivalence corpus

Suggested corpus:

```text
P1  whitespace + CALL
P2  prose + CALL
P3  CALL1 + whitespace + CALL2
P4  CALL1 + prose + CALL2
P5  CALL1 + failed marker + CALL2
P6  CALL1 + partial CALL2
P7  CALL1 + CRLF + CALL2
P8  CALL1 + tab + CALL2
P9  CALL1 + form-feed + CALL2
```

For each, record:

```text
production parser structured outcome
constraint second-entry trigger outcome
```

No contradiction may remain.

---

# 3.21 Formatting whitespace

Preserve the parser's explicit formatting-whitespace definition.

At minimum:

```text
space
tab
CR
LF
```

Do not replace with locale-dependent `std::isspace()`.

If form feed is currently visible content, pin that in a test.

---

# 3.22 Stale comments

Update all comments claiming conceptually:

```text
first latch permanently satisfies the content-start requirement
later visible prose does not re-lock
```

That statement is now known to be incorrect relative to the production parser.

Files:

```text
tool_call_grammar_state.h
tool_call_grammar_state.cpp
test_tool_call_grammar_state.cpp
```

---

# 3.23 R8-01 expected files

```text
src/models/qwen3_5/frontend/tool_call_grammar_state.h
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp
tests/test_tool_call_grammar_state.cpp
docs/tool_call_parser.md
docs/NInfer_new_parser_design_round8_bugfix_progress.md
```

Optional cross-check additions:

```text
tests/test_tool_call_parser.cpp
```

The production Stage-1/2/3 parser should require no code change.

---

# 3.24 R8-01 acceptance criteria

- [ ] baseline red test proves current CALL1+prose+CALL2 mismatch;
- [ ] `latched_once_` permanent exemption removed/replaced;
- [ ] visible prose after CALL1 locks hardened intent;
- [ ] CALL1 + whitespace + CALL2 remains supported;
- [ ] CALL1 + prose + CALL2 never triggers CALL2;
- [ ] CALL1 + failed marker + CALL2 never triggers CALL2;
- [ ] partial immediate CALL2 resumes;
- [ ] same-chunk and split-chunk behavior agree;
- [ ] trailing replay begins at `progress.break_offset`;
- [ ] intent-relevant bytes are not skipped by `rfind('<')`;
- [ ] checkpoint/restore post-call works;
- [ ] TemplateCompatible unchanged;
- [ ] syntax x intent matrix green;
- [ ] production parser corpus unchanged;
- [ ] constrained runtime still fail-fast.

---

# 4. R8-02 — Chat Completions silently weakens unsupported guarantees

## 4.1 Current internal representation

Current:

```cpp
enum class ToolChoiceMode {
    Auto,
    None,
};
```

There is no executable `Required` or `Named` mode.

That is acceptable only if unsupported wire requests are rejected before translation.

---

# 4.2 Chat `required`

Current baseline:

```cpp
if (value == "required") {
    output.tool_choice.mode = ToolChoiceMode::Auto;
}
```

OpenAI semantics require one or more tool calls.

NInfer cannot guarantee this.

Required Round-8 behavior:

```text
HTTP 400
param = tool_choice
code = tool_choice_not_supported
```

Suggested message:

```text
tool_choice 'required' cannot be guaranteed by the Engine
```

This matches the existing Responses philosophy.

---

# 4.3 Named function choice

Example:

```json
{
  "tool_choice": {
    "type": "function",
    "function": {"name": "weather"}
  }
}
```

OpenAI semantics:

```text
force weather
```

Current NInfer:

```text
validate name syntax
set Auto
```

Required behavior:

```text
validate structural shape/name
then 400 tool_choice_not_supported
```

Do not narrow to one tool and call that "forced". A one-tool set still permits no tool call.

---

# 4.4 Named custom choice

Same rule for:

```json
{
  "tool_choice": {
    "type": "custom",
    "custom": {"name": "shell"}
  }
}
```

Until exact custom invocation can be guaranteed:

```text
400 tool_choice_not_supported
```

---

# 4.5 allowed_tools required

Current Chat behavior:

```text
validate subset
narrow tools
set Auto
```

This implements subset filtering but not required invocation.

Required:

```text
mode=auto
  supported
  narrow effective tools

mode=required
  400 tool_choice_not_supported
```

Reject before mutating `output.tools`.

---

# 4.6 parallel_tool_calls=false

Current OpenAI contract:

```text
false -> zero or one call
```

Current NInfer Chat:

```text
accepts flag as advisory
may still return multiple calls
```

Required:

```cpp
if (!parallel && output.uses_tools()) {
    bad_request(
        "parallel_tool_calls=false cannot be guaranteed when callable tools are present",
        "parallel_tool_calls",
        "parallel_tool_calls_not_supported");
}
```

Neutral cases remain valid:

```text
no tools
tool_choice none
allowed_tools auto selects an empty effective set
```

---

# 4.7 Preserve parallel=true

`parallel_tool_calls=true` only permits multiple calls. NInfer already supports multiple Qwen wrappers. Keep it.

---

# 4.8 Do not post-process serial semantics in Round 8

Do not attempt:

```text
keep only first call
convert multi-call to text
retry generation
late server error
```

Those require separate streaming/product design.

Fail-fast is the smallest truthful implementation.

---

# 4.9 Do not implement required via prompt wording

"You must call a tool" is a preference, not an executable guarantee.

Prompt-only forcing does not satisfy `tool_choice:required`.

---

# 4.10 Do not implement named forcing by one-tool filtering

Offering only:

```text
{weather}
```

means only:

```text
if a tool is called, it can only be weather
```

It does not mean:

```text
weather must be called
```

Therefore one-tool narrowing is insufficient.

---

# 4.11 Responses adapter is the behavior reference

Current Responses already rejects:

```text
tool_choice required
named/hosted forcing
allowed_tools required
parallel false with callable tools
```

Round 8 should make Chat capability handling consistent without refactoring both adapters unless a clean shared helper already exists.

---

# 4.12 Keep ToolChoiceMode Auto/None

Do not add:

```text
Required
Named
```

to `ToolChoiceMode` until the Engine has actual semantics for them.

Unsupported wire values should die at the API boundary.

---

# 4.13 Existing Chat tests that must change

Current `tests/test_openai_schema.cpp` explicitly asserts:

```text
required is accepted as advisory auto
named choice is accepted as advisory auto
required allowed_tools is accepted as advisory auto
parallel false is accepted as advisory
```

Those assertions must be replaced. Do not keep contradictory legacy tests beside new fail-fast tests.

---

# 4.14 New Chat request tests

## required

Expected:

```text
400
param = tool_choice
code = tool_choice_not_supported
```

## named function

Use a valid declared tool so failure proves unsupported forcing rather than malformed input.

Expected:

```text
400 tool_choice_not_supported
```

## named custom

Same.

## allowed_tools auto

Expected:

```text
accepted
effective set narrowed
```

## allowed_tools required

Expected:

```text
400 tool_choice_not_supported
```

Test both nested and direct compatibility shape if both remain supported syntactically.

## parallel false + callable tools

Expected:

```text
400
param = parallel_tool_calls
code = parallel_tool_calls_not_supported
```

## parallel false + no tools

Expected: accepted.

## parallel false + tool_choice none

Expected: accepted, `uses_tools()==false`.

---

# 4.15 Error precedence

Preserve structural validation:

```text
tool_choice numeric -> invalid type
missing function object -> malformed tool_choice
invalid name syntax -> invalid_tool_name
```

Only structurally valid unsupported guarantees should return `tool_choice_not_supported`.

---

# 4.16 Declared-name lookup for named forcing

Two defensible options:

### Minimal

Validate name syntax, then reject unsupported forcing.

### Strong

Validate declared membership, then reject forcing.

Recommended Round-8 choice:

```text
minimal
```

because the feature is unsupported anyway and `allowed_tools.mode=auto` already covers supported subset validation.

---

# 4.17 HTTP-level test

Add at least one route/error test:

```text
POST /v1/chat/completions
tool_choice = required
```

Expected external shape:

```json
{
  "error": {
    "type": "invalid_request_error",
    "param": "tool_choice",
    "code": "tool_choice_not_supported"
  }
}
```

No generation should start.

---

# 4.18 Documentation update

Replace the current Chat advisory description with a table equivalent to:

| Chat field | Round-8 behavior |
|---|---|
| `tool_choice:auto` | supported |
| `tool_choice:none` | supported |
| `allowed_tools.mode:auto` | supported subset filter |
| `tool_choice:required` | rejected |
| named function/custom | rejected |
| `allowed_tools.mode:required` | rejected |
| `parallel_tool_calls:true` | supported |
| `parallel_tool_calls:false` + callable tools | rejected |
| `parallel_tool_calls:false` + no effective tools | accepted neutral |

Do not call unsupported guarantees advisory-supported after Round 8.

---

# 4.19 Compatibility impact

R8-02 intentionally changes behavior.

Clients that currently send:

```text
required
named choice
allowed_tools required
parallel_tool_calls=false
```

and rely on silent downgrade to Auto will begin receiving HTTP 400.

Document this as an intentional correctness change.

If backward compatibility is later required, use an explicitly NInfer-specific compatibility switch. Do not silently weaken the standard field.

---

# 4.20 OMP / Responses impact

The Responses adapter already rejects unsupported forced tool semantics.

Therefore R8-02 should not change `/v1/responses` behavior.

This is a Chat-only contract correction.

---

# 4.21 R8-02 expected files

```text
src/serve/openai_chat_request.cpp
tests/test_openai_schema.cpp
docs/serving.md
docs/NInfer_new_parser_design_round8_bugfix_progress.md
```

Optional:

```text
tests/test_http_routes.cpp
```

No frontend parser file should be required.

---

# 4.22 R8-02 acceptance criteria

- [ ] required no longer becomes Auto;
- [ ] named function no longer becomes Auto;
- [ ] named custom no longer becomes Auto;
- [ ] allowed_tools required no longer becomes Auto;
- [ ] parallel false + callable tools rejected;
- [ ] auto still supported;
- [ ] none still supported;
- [ ] allowed_tools auto still filters;
- [ ] parallel true still supported;
- [ ] parallel false neutral with no effective tools;
- [ ] exact error param/code tested;
- [ ] at least one HTTP-level capability error tested;
- [ ] Responses unchanged;
- [ ] no parser/core change.

---

# 5. Implementation phases

## Phase R8-0 — freeze baseline

Record:

```text
start SHA
worktree status
compiler/toolchain
build directory
GPU-hidden configuration
```

Run current targeted baseline first.

---

## Phase R8-1 — add red R8-01 tests

Before production code:

```text
replace incorrect post-call prose expectation
add same-token CALL1+prose+CALL2
add split-token variant
add positive whitespace-consecutive control
add checkpoint/restore post-call control
```

At least one hardened assertion must fail on the baseline.

---

## Phase R8-2 — fix post-call gate

Remove/redesign `latched_once_`.

Do not touch Stage 1/2/3.

---

## Phase R8-3 — fix trailing replay

Replace marker-suffix jumping with `break_offset` replay through the inactive gate.

Delete `marker_suffix()` if it has no legitimate remaining caller.

---

## Phase R8-4 — parser/constraint cross-check

Run the post-call eligibility corpus.

No constrained runtime activation.

---

## Phase R8-5 — add red Chat contract tests

Change `tests/test_openai_schema.cpp` advisory expectations into expected capability errors.

---

## Phase R8-6 — implement Chat fail-fast

Use existing `bad_request(...)` and existing error-code vocabulary:

```text
tool_choice_not_supported
parallel_tool_calls_not_supported
```

---

## Phase R8-7 — HTTP-level error test

Verify error serialization, not only parser exception state.

---

## Phase R8-8 — docs

Update `docs/serving.md`, parser/constraint docs, and Round-8 progress.

---

## Phase R8-9 — full regression

Re-run parser, grammar, constraint, frontend, templates, Chat schema, Responses, HTTP routes, serve options, request logging, then the GPU-hidden full CTest.

---

# 6. R8-01 detailed corpus

Use at least:

```text
A  prose + CALL
B  CALL1
C  CALL1 + CALL2
D  CALL1 + whitespace + CALL2
E  CALL1 + prose + CALL2
F  CALL1 + "x" + CALL2
G  CALL1 + failed marker + CALL2
H  CALL1 + partial immediate CALL2
I  CALL1 + partial failed marker + CALL2
J  same commit: CALL1 + prose + CALL2
K  CRLF-separated CALL1/CALL2
L  form-feed-separated CALL1/CALL2
M  checkpoint after CALL1
N  checkpoint inside partial CALL2 marker
```

Expected hardened results:

```text
C/D/H/K -> next tool remains eligible
E/F/G/I/J/L -> next tool must not trigger
M/N -> restore must recover pre-draft eligibility exactly
```

TemplateCompatible controls must preserve baseline semantics.

---

# 7. Whole-chunk vs bytewise constraint parity

Every new constraint fixture must run both:

```text
whole candidate chunk
bytewise check/commit
```

Final state must agree.

This catches fixes accidentally dependent on tokenizer boundaries.

---

# 8. Checkpoint proof

New state words must be copied by normal value semantics.

No custom partial checkpoint logic.

Verify:

```text
Open -> draft prose -> Locked -> restore -> Open
Open with partial marker -> failed draft -> restore -> marker can complete
```

---

# 9. Exact Chat error contract

Recommended:

## required

```text
400
invalid_request_error
param=tool_choice
code=tool_choice_not_supported
```

## named

```text
400
invalid_request_error
param=tool_choice
code=tool_choice_not_supported
```

## allowed required

```text
400
invalid_request_error
param=tool_choice
code=tool_choice_not_supported
```

## parallel false

```text
400
invalid_request_error
param=parallel_tool_calls
code=parallel_tool_calls_not_supported
```

Use existing project wording conventions for messages.

---

# 10. Request-validation ordering

Keep:

```text
parse_tools
parse_tool_choice
parse_parallel_tool_calls
...
```

Inside `parse_tool_choice`:

```text
validate JSON shape
validate nested name syntax where applicable
reject unsupported guarantee
```

Inside `parse_parallel_tool_calls`:

```text
validate bool
true -> accept
false + !uses_tools -> accept neutral
false + uses_tools -> reject
```

---

# 11. Allowed-tools mutation ordering

For `mode=required`, reject before mutating `output.tools`.

For `mode=auto`, keep current subset filtering.

Exception paths should not partially mutate request state.

---

# 12. Responses parity table

After Round 8:

| Semantic | Chat | Responses |
|---|---|---|
| auto | accept | accept |
| none | accept | accept |
| required | reject | reject |
| named force | reject | reject |
| allowed auto | accept | accept |
| allowed required | reject | reject |
| parallel false + callable tools | reject | reject |

Wire schemas differ, but capability honesty matches.

---

# 13. Non-goals

Do not change in Round 8 without a separate reviewed finding:

```text
strict:true advisory behavior
Anthropic forced-tool advisory behavior
custom-tool grammar semantics
Responses namespace behavior
OpenAI output serialization
parser argument normalization
Stage-2 boundary algorithm
```

---

# 14. Outcome dumps

## Constraint dump

Dimensions:

```text
syntax
intent
chunking
checkpoint path
post-call suffix type
```

Expected delta:

```text
RequireToolAtContentStart
+ completed prior call
+ visible content
+ later marker

second trigger: true -> false
```

No production parser delta expected.

## Chat request dump

Cases:

```text
auto
none
required
named function
named custom
allowed auto
allowed required
parallel true
parallel false + tools
parallel false + none
parallel false + no tools
```

Only unsupported-guarantee cases change.

---

# 15. Review loop A — inactive intent state

For every inactive byte ask:

```text
formatting whitespace?
possible marker prefix?
visible content?
gate already locked?
```

No `HadTool` exception.

---

# 16. Review loop B — active-to-inactive transition

Verify separately:

```text
Complete
TrailingContent
EndOfInput
Definitive structural failure
```

### Complete

```text
region closed
gate open for whitespace/immediate wrapper
```

### TrailingContent

```text
region closed
tail replayed
visible content can lock
```

### EndOfInput

```text
remain active/legal prefix
```

### structural break inside open region

```text
Rejected
```

---

# 17. Review loop C — stale marker state

Search:

```text
marker_suffix
rfind('<')
marker_prefix_
```

Prove a completed closing tag is not retained as a future marker candidate and the first byte after a completed region is processed normally.

---

# 18. Review loop D — parser/constraint semantics

Side-by-side:

```text
CALL1 + whitespace + CALL2
CALL1 + prose + CALL2
CALL1 + failed marker + CALL2
```

Any contradiction fails Round 8.

---

# 19. Review loop E — Chat silent weakening search

Search repository for:

```text
tool_choice
required
advisory
parallel_tool_calls
ToolChoiceMode::Auto
```

No standard Chat forcing request may remain silently mapped to Auto.

---

# 20. Review loop F — Responses regression

Verify no behavior change in `openai_responses_request.cpp` and run the Responses tests.

---

# 21. Review loop G — documentation consistency

Check:

```text
docs/serving.md
test assertion messages
source comments
Round-8 progress
```

No documentation may describe rejected Chat guarantees as advisory-supported.

---

# 22. CPU-only verification commands

Discover exact names:

```powershell
ctest --test-dir build-new-parser -C Release -N
```

Suggested targeted run:

```powershell
ctest --test-dir build-new-parser -C Release `
  -R "ninfer_(tool_call_parser|tool_call_grammar|tool_call_grammar_state|qwen3_5_frontend|openai_schema|openai_responses|http_routes|serve_options|request_log)_test" `
  --output-on-failure `
  --parallel 16
```

Use actual registered names if they differ.

Full GPU-hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES="99"

ctest --test-dir build-new-parser -C Release `
  -E "_real" `
  --output-on-failure `
  --parallel 16
```

Record:

```text
total
passed
failed
skipped
not run
CTest exit code
```

Never call nonzero CTest green.

---

# 23. Previous-round regression set

Re-run all pinned classes:

```text
F1 opaque parameter payload
F2 open function non-executable
F3 EOF boundary ambiguity
F4 partial next opener
wrapper balance
mixed param families
breaking '<'
failed-wrapper scope
function_calls balance
trailing content
declared-tool strict/tolerant
fence guard
Stage-2 work bounds
FailClosed ambiguity
FinishReason transaction
native/compat syntax
R6 pre-latch intent
R7 later-base retry cutoff
R7 decoder intent propagation
R7 template split
R8 constraint post-call gate
```

No unexpected parser delta is allowed.

---

# 24. Suggested commits

```text
1. test(constraint): reproduce post-call hardened intent divergence
2. fix(constraint): re-lock after visible content following completed tool region
3. fix(constraint): replay trailing content from parser break offset
4. test(constraint): post-call chunk/checkpoint matrix
5. test(frontend): parser/constraint post-call eligibility cross-check
6. test(chat): pin unsupported forced tool-control requests
7. fix(chat): reject required and named tool choices
8. fix(chat): reject allowed_tools required and parallel false guarantee
9. test(http): verify Chat capability error payloads
10. docs: update Chat tool-control contract and constraint semantics
11. docs: record Round-8 progress
```

Do not squash before review.

---

# 25. Definition of Done — R8-01

- [ ] current mismatch reproduced red;
- [ ] permanent `latched_once_` exemption removed/replaced;
- [ ] visible prose after CALL1 locks hardened intent;
- [ ] whitespace-only gap preserves next CALL;
- [ ] failed marker after CALL1 locks;
- [ ] partial immediate next wrapper resumes;
- [ ] same-chunk/split-chunk parity;
- [ ] tail replay begins at `break_offset`;
- [ ] no `rfind` skipping of intent-relevant prose;
- [ ] checkpoint/restore post-call green;
- [ ] TemplateCompatible unchanged;
- [ ] Native/Compatibility matrix green;
- [ ] production parser corpus unchanged;
- [ ] constrained runtime remains fail-fast.

---

# 26. Definition of Done — R8-02

- [ ] Chat required -> 400 `tool_choice_not_supported`;
- [ ] named function -> 400;
- [ ] named custom -> 400;
- [ ] allowed_tools required -> 400;
- [ ] parallel false + callable tools -> 400 `parallel_tool_calls_not_supported`;
- [ ] auto accepted;
- [ ] none accepted;
- [ ] allowed_tools auto still filters;
- [ ] parallel true accepted;
- [ ] parallel false neutral with no effective tools;
- [ ] exact error param/code tests;
- [ ] HTTP-level error test;
- [ ] Responses unchanged;
- [ ] no Engine/parser changes.

---

# 27. Final sign-off wording if green

Acceptable:

> No remaining HIGH/MEDIUM implementation defect was found in the reviewed active Qwen3.8 native parser path or the reviewed parser/constraint intent-policy boundary under the CPU-only verified scope. The grammar constraint now ends hardened tool-sequence eligibility when visible content follows a completed call while preserving directly consecutive wrappers. Chat Completions no longer silently weakens unsupported forced tool-control requests to automatic selection. Runtime constrained tool decoding remains unavailable and fail-fast. The complete call-only quotation ambiguity remains a documented semantic limitation of the wire protocol.

Do not say:

```text
bug-free
all tool-call errors impossible
phantom calls impossible
semantic intent solved
```

---

# 28. Known residual — unchanged

A byte-identical complete call-only output can still mean either:

```text
execute
```

or:

```text
show as an example
```

when no distinct intent signal exists.

Neither `FailClosed`, `RequireToolAtContentStart`, nor syntax constraints can infer intent from identical bytes.

Long-term solution remains a structural TEXT-vs-TOOL intent channel outside ordinary quotable content.

---

# 29. Future constrained-decoding activation checklist

Before removing the current fail-fast require at least:

- [ ] syntax parity;
- [ ] intent parity including R8 post-call behavior;
- [ ] checkpoint/restore parity;
- [ ] consecutive-wrapper parity;
- [ ] declared-tool integration decision;
- [ ] token-mask integration;
- [ ] speculative rollback tests;
- [ ] GPU sampling tests;
- [ ] streaming/final parser agreement;
- [ ] malformed-tail behavior;
- [ ] benchmark impact;
- [ ] no call-only quotation safety overclaim.

Round 8 fixes only one prerequisite.

---

# 30. Recommended final architecture

```text
model semantic intent
    ↓
TEXT --------------------------> content
    |
    └── TOOL
          ↓
      grammar constraint
          ↓
      canonical Qwen native wire
          ↓
      strict parser
          ↓
      declared tool validation
          ↓
      argument normalization
          ↓
      API tool call
```

Forced API tool choice additionally requires:

```text
request tool policy
    ↓
sampler/intent controller
    ↓
required or named TOOL intent
```

Until that exists, fail-fast is the correct API behavior.

---

# 31. Evidence references

## NInfer baseline

```text
https://github.com/Hundsbuah/ninfer/tree/ad71244de33212aa92742668502b29c3a72528a3
```

Relevant files:

```text
src/models/qwen3_5/frontend/tool_call_grammar_state.h
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp
src/models/qwen3_5/frontend/tool_call_stream.cpp
tests/test_tool_call_grammar_state.cpp
tests/test_tool_call_parser.cpp
src/serve/openai_chat_request.cpp
src/serve/openai_responses_request.cpp
src/serve/request.h
tests/test_openai_schema.cpp
tests/test_openai_responses.cpp
docs/serving.md
```

## Official Qwen3.8 template

```text
https://huggingface.co/Qwen/Qwen3.8-27B/blame/main/chat_template.jinja
```

## Qwen function-calling documentation

```text
https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md
```

## OpenAI Chat Completions reference

```text
https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions
```

## OpenAI function-calling guide

```text
https://developers.openai.com/api/docs/guides/function-calling
```

---

# 32. Engineering conclusion

Round 8 should be a small consistency patch, not another parser rewrite.

For R8-01:

```text
tool was used previously
!=
future tool markers remain permanently eligible
```

Correct rule:

```text
tool-sequence eligibility survives only formatting whitespace and an immediate next valid tool entry
```

For R8-02:

```text
unsupported guarantee
!=
advisory success
```

Correct rule:

```text
if NInfer cannot guarantee a standard API field's promised semantics, reject the request explicitly rather than silently weakening it
```

These changes close the two remaining implementable Round-8 findings without destabilizing the already-hardened native Qwen tool-call parser.
