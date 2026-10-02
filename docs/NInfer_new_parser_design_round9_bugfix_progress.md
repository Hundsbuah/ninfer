# NInfer `new_parser_design` — Round 9 Bugfix Progress

> Companion evidence log for `NInfer_new_parser_design_round9_bugfix_implementation.md`.
>
> Do not inherit Round-8 green status. Every Round-9 finding needs red reproduction, implementation evidence, and regression evidence.

## 1. Baseline

```text
repository: Hundsbuah/ninfer
branch: new_parser_design
reviewed HEAD: 882b5a3d88ff5c363c0f94a8094b2ee802ee0201
implementation start SHA: 882b5a3d88ff5c363c0f94a8094b2ee802ee0201
worktree: user-local machine workarounds (CMakeLists.txt, build_native.bat,
  src/runtime/engine/materialization_budget.h, src/runtime/engine/kv_capacity.cpp);
  excluded from every Round-9 commit
toolchain: MSVC (Visual Studio 18 2026), CUDA 13.4, build-new-parser
GPU policy: CPU/static only; full CTest with CUDA_VISIBLE_DEVICES=99
GitHub status checks at reviewed baseline: none
GitHub workflow runs at reviewed baseline: none
```

## 2. Finding status

| Finding | Severity | Status | Commit(s) | Verification |
|---|---:|---|---|---|
| R9-01 constraint suffix vs strict parser | MEDIUM architecture | DONE | d023f25a (red), 643917cb | grammar_state + parser suites green |
| R9-01a pre-trigger/between-calls conflation | root cause | DONE | 643917cb | explicit phases; matrix green |
| R9-01b explicit EOS legality | design gap | DONE | 643917cb | can_terminate() tested |
| R9-02 Chat strict:true downgrade | MEDIUM | DONE | 5e252077 | schema + http_routes green |
| R9-03 Chat custom false compatibility | MEDIUM | DONE | b3b36b9e | schema + http_routes green |
| R9-03a free-form -> JSON function | root cause | DONE | b3b36b9e | helper deleted; rejection tested |
| R9-03b custom grammar descriptive only | root cause | DONE | b3b36b9e | definitions rejected before lowering |
| R9-03c custom output -> function | root cause | DONE | b3b36b9e | invariant comment; no wire machinery |
| R9-03d custom history unsupported | root cause | DONE | b3b36b9e | explicit type error + test |
| R9-04 Anthropic any/named advisory | MEDIUM | DONE | f5bd58c3 | anthropic + http_routes green |
| R9-04b disable_parallel advisory | MEDIUM | DONE | f5bd58c3 | persisted; fail-fast tested |
| R9-04c strict:true advisory | MEDIUM | DONE | f5bd58c3 | persisted; fail-fast tested |
| R6-04 byte-identical quotation | residual | RESIDUAL | n/a | documented |
| runtime constrained decoding | deferred | DEFERRED | n/a | fail-fast |

## 3. R9-01 red baseline

### CALL + prose

```text
input: CALL1 + "\nDone"
strict parser: 0 calls / TrailingContent
constraint baseline expected bug: Allowed
observed: Allowed (constraint admitted the visible suffix)
red test: d023f25a test_r9_constraint_post_call_suffix (failed pre-fix, green post-fix)
```

### Token-atomic close + prose

```text
state: inside CALL1 immediately before wrapper close
candidate: "</tool_call>\nDone"
Round-9 expected: Rejected
baseline observed: Allowed (the prose suffix stayed admissible)
```

## 4. R9-01 state model

Chosen representation:

```text
enum class ToolConstraintPhase { PreTrigger, TextLocked, InRegion, BetweenCalls }
plus the existing state words (buffer, marker_prefix, parse policy, max length).
active() is exactly (phase == InRegion); a pending marker candidate exists only in
PreTrigger/BetweenCalls; TextLocked always carries an empty marker candidate.
```

Expected logical phases:

```text
PreTrigger
TextLocked
InRegion
BetweenCalls
```

Evidence impossible combinations are eliminated:

```text
phase == TextLocked with a pending marker: impossible (a visible byte clears the candidate)
phase == BetweenCalls with an open region buffer: impossible (the buffer is cleared on the
complete transition and on a rejected candidate)
phase == InRegion with an empty buffer: impossible (the trigger moves the marker into
the region buffer)
```

## 5. R9-01 implementation

Files:

```text
src/models/qwen3_5/frontend/tool_call_grammar_state.h/.cpp (643917cb)
```

Transitions:

```text
Complete: BufferComplete -> phase BetweenCalls, buffer cleared, marker empty
TrailingContent: BufferComplete + visible tail -> BetweenCalls, the tail replays through
  the strict between-calls scanner from the first tail byte (no offset)
EndOfInput: BufferComplete + closed-region tail -> BetweenCalls with the pending marker
  candidate (G7/G8); an open region stays InRegion
BetweenCalls whitespace: space/tab/newline/carriage return stay admissible
BetweenCalls visible byte: Rejected (prose, form feed, any non-< byte)
BetweenCalls partial marker: NeedMore (the pending marker stays completable; EOS-illegal)
BetweenCalls failed marker: NotMarker -> Rejected (a rejected candidate never commits)
```

## 6. R9-01 termination predicate

API:

```text
[[nodiscard]] bool can_terminate() const noexcept  (643917cb)
```

| State | EOS expected | Observed |
|---|---|---|
| PreTrigger, no pending marker | legal | legal (tested) |
| TextLocked | legal | legal (tested) |
| InRegion | illegal | illegal (tested) |
| BetweenCalls clean | legal | legal (tested) |
| partial marker | illegal | illegal (tested) |

## 7. R9-01 corpus

- [x] G1 CALL1
- [x] G2 CALL1 + whitespace
- [x] G3 CALL1 + prose -> Rejected
- [x] G4 CALL1 + form feed -> Rejected
- [x] G5 CALL1 + CALL2
- [x] G6 CALL1 + whitespace + CALL2
- [x] G7 partial second marker -> NeedMore
- [x] G8 failed marker -> Rejected
- [x] G9 same-token close + prose -> Rejected
- [x] G10 same-token close + next marker -> legal
- [x] TemplateCompatible preamble control
- [x] StartOfContent preamble control
- [x] native syntax
- [x] compatibility syntax
- [x] checkpoint BetweenCalls
- [x] checkpoint partial marker

Result:

```text
all green: ninfer_tool_call_grammar_state_test.exe (643917cb); the R8 matrix and the
parser cross-check (R8 cross P1-P9, R9 verdict consistency rule) are green as well
```

## 8. R9-02 red baseline

```text
Chat function strict:true
baseline: accepted; prompt strict:false
observed: accepted; the rendered prompt carried "strict":false (advisory test pinned it)
red: the flipped test failed before the fix (5e252077 red phase), green after
```

## 9. R9-02 implementation

Expected:

```text
strict omitted -> accept
strict false   -> accept
strict true    -> 400 strict_tools_not_supported
non-bool       -> type error
```

Commit:

```text
5e252077 fix: reject Chat strict:true as unguaranteeable guarantee (R9-02)
```

Exact param:

```text
tools[i].function.strict (exact index prefix from parse_tools)
```

HTTP result:

```text
400 invalid_request_error, param=tools[0].function.strict, code=strict_tools_not_supported
(test_r9_chat_strict_error_payload, test_http_routes.cpp)
```

## 10. R9-03 red baseline

```text
custom definition accepted: yes (test pinned "served as a single-string-input function")
synthetic {input:string} lowering: yes (custom_tool_input_schema)
custom grammar descriptive only: yes (appended to the input description string)
response serializer type=function: yes (unconditional in tool_calls_json)
custom history rejected: yes ("only function tool_calls are supported")

## 11. R9-03 implementation

Policy:

```text
Chat custom tools unsupported until complete free-form/wire semantics exist.
```

- [x] `tools[i].type=custom` rejected
- [x] custom grammar therefore cannot be advisory accepted
- [x] `custom_tool_input_schema()` removed if unused
- [x] allowed_tools custom selector rejected
- [x] named custom forcing remains unsupported
- [x] custom history explicitly unsupported
- [x] function tools unchanged
- [x] docs updated

Expected error:

```text
status=400
param=tools[0].type
code=tool_type_not_supported
```

Observed:

```text
status=400 param=tools[0].type code=tool_type_not_supported
(test_r9_chat_custom_error_payload + schema-level tests, b3b36b9e)
```

Compatibility test for clients that used synthetic lowering:

```text
not tested here: no Copilot CLI/MCP client is available in this environment. The source
comment named them as the motivation for the synthetic translation; the spec (§5.12) keeps
the standard field semantics strict and defers a separate opt-in to concrete evidence.
```

Decision on explicit compatibility flag:

```text
no flag introduced (spec §5.12: only if a concrete client regression proves the need;
a separate explicit non-standard opt-in, default off)
```

## 12. R9-04 red baseline

```text
any + tools -> accepted as Auto: yes (advisory loop pinned it)
named declared tool -> accepted as Auto: yes (advisory loop pinned it)
auto + disable_parallel true -> advisory accepted: yes (advisory loop pinned it)
strict true -> advisory accepted: yes (advisory test pinned it)
```

## 13. R9-04 implementation

`ToolSelection` stores:

```text
bool disable_parallel_tool_use = false  (persisted, not discarded)
```

`ParsedTool` stores:

```text
bool strict = false
source index: the lower_tools loop index renders param tools[i].strict
```

Behavior:

- [x] auto accepted
- [x] none accepted
- [x] any rejected
- [x] named known rejected as unsupported forcing
- [x] named unknown preserves unknown-tool error
- [x] auto + disable false accepted
- [x] auto + disable true + tools rejected
- [x] none + disable true neutral
- [x] strict false accepted
- [x] strict true callable rejected
- [x] strict true none policy tested

Commits:

```text
f5bd58c3 fix: reject unguaranteeable Anthropic tool choice and strict (R9-04)
```

Internal codes:

```text
any/named: tool_choice_not_supported
disable_parallel: parallel_tool_calls_not_supported
strict: strict_tools_not_supported
```

Observed params/codes:

```text
tool_choice / tool_choice.disable_parallel_tool_use / tools[i].strict, with the codes
above (test_tools, test_anthropic_schema.cpp)
```

## 14. Anthropic external error body

- [x] HTTP 400
- [x] body type=error
- [x] error.type=invalid_request_error
- [x] useful message
- [x] request_id present

Result:

```text
test_r9_anthropic_tool_choice_error_payload (test_http_routes.cpp): envelope
type/error/message/request_id asserted; internal param/code asserted separately
```

## 15. Count Tokens parity

Messages behavior:

```text
rejects tool_choice_not_supported (any) and strict_tools_not_supported (strict true)
```

Count Tokens behavior:

```text
identical — both endpoints share parse_common_prompt -> lower_tools
```

Intentional difference, if any:

```text
none — parity is deliberate and tested
```

## 16. Qwen Code compatibility

Does current Qwen Code integration send Anthropic `tool_choice:any` to NInfer?

```text
per the pre-existing source comment, Qwen Code sends tool_choice:any for its JSON side
queries. No Qwen Code client is available in this environment, so no live test was run.
```

Does strict fail-fast break an existing supported workflow?

```text
unknown without a live client. The spec (§6.8) forbids a speculative compatibility flag
before evidence exists; the risk is documented in serving.md.
```

If yes, compatibility strategy:

```text
none introduced. An explicit non-standard opt-in (e.g. --anthropic-advisory-tool-choice,
default off, request-log downgrade marker) is the prescribed path if compatibility
evidence appears; never a silent default downgrade.
```

## 17. OpenAI Responses regression

- [x] strict true rejection unchanged
- [x] required rejection unchanged
- [x] named rejection unchanged
- [x] allowed required rejection unchanged
- [x] parallel false rejection unchanged
- [x] normal function calls green

Result:

```text
green: ninfer_openai_responses_test (f5bd58c3 build); Responses code paths untouched
```

## 18. Native parser regression

- [x] Stage 1 unchanged
- [x] Stage 2 unchanged
- [x] Stage 3 unchanged
- [x] FailClosed unchanged
- [x] tolerant behavior unchanged
- [x] declared-name behavior unchanged
- [x] fence behavior unchanged
- [x] FinishReason unchanged
- [x] R8 intent behavior unchanged

Result:

```text
green: ninfer_tool_call_parser_test + ninfer_qwen3_5_frontend_test +
ninfer_tool_call_grammar_test (643917cb build); the production parser is unchanged by
Round 9 — only the constraint cross-check test semantics were updated
```

## 19. Targeted CPU tests

Discover:

```powershell
ctest --test-dir build-new-parser -C Release -N
```

Run actual registered equivalents of:

```powershell
ctest --test-dir build-new-parser -C Release `
  -R "ninfer_(tool_call_parser|tool_call_grammar|tool_call_grammar_state|qwen3_5_frontend|openai_schema|openai_responses|anthropic_schema|http_routes|serve_options|request_log)_test" `
  --output-on-failure `
  --parallel 16
```

Observed:

```text
10 selected
10 passed
0 failed
0 skipped
exit code 0
```

## 20. Full GPU-hidden CTest

```powershell
$env:CUDA_VISIBLE_DEVICES="99"
ctest --test-dir build-new-parser -C Release `
  -E "_real" `
  --output-on-failure `
  --parallel 16
```

Observed:

```text
147 selected
46 passed
0 failed
0 skipped in denominator
101 skipped (GPU-dependent, no device with CUDA_VISIBLE_DEVICES=99)
exit code 0
```

## 21. Review loops

### A — Constraint phases

```text
phases advanced only in advance(); the impossible-combination invariants
hold: a marker candidate exists only in PreTrigger/BetweenCalls, TextLocked
always clears it, InRegion always owns a non-empty buffer. A rejected
candidate never mutates the committed state (check is const; commit asserts
in debug on Rejected).
```

### B — token atomicity

```text
close+prose: Rejected (G9, single token and byte-wise; rejected candidate never commits)
close+next-marker: legal (G10; the wrapper retriggers into InRegion)
```

### C — EOS

```text
can_terminate() tested for all five states (native x compatibility); the
commit fail-fast on Rejected is unchanged; a pending second marker is
EOS-illegal (G7, parser cross P6).
```

### D — OpenAI guarantee audit

Remaining advisory standard fields:

```text
no remaining tool-control guarantees: strict, custom type, required/named
choice, and parallel limiting all fail fast; the remaining accepted fields
are semantically neutral (documented as neutral/advisory metadata, not guarantees)
```

### E — function round-trip

```text
function request -> prompt -> generated call -> function response -> function history
request -> prompt and history -> prompt are pinned in test_openai_schema.cpp;
the response serializer function wire kind is invariant (R9-03 comment);
no regression in the green suites
```

### F — Anthropic guarantee audit

Remaining advisory standard fields:

```text
no remaining forced/serial/strict tool guarantees: any/named, disable_parallel
with callable tools, and strict with callable tools all fail fast; the remaining
accepted fields are documented hints/extensions (cache TTL, preserve_thinking, ...)
```

### G — compatibility

```text
Qwen Code: untested (no client in this environment); risk documented, no flag
Chat clients formerly using custom-as-function: untested (no client available);
spec §5.12 defers an opt-in to concrete evidence
```

### H — docs

- [x] Phase-4 suffix asymmetry removed (D2/D4 rewritten for R9-01)
- [x] Chat strict advisory claim removed (serving.md Round-9 note)
- [x] Chat custom support claim removed (serving.md Round-9 note)
- [x] Anthropic forced advisory claim removed (serving.md Round-9 table)
- [x] disable_parallel advisory claim removed (serving.md Round-9 table)
- [x] Anthropic strict advisory claim removed (serving.md Round-9 table)
- [x] constrained runtime fail-fast retained
- [x] semantic quotation residual retained

## 22. Known residual

```text
Byte-identical complete tool markup cannot encode semantic author intent by itself.
```

Status:

```text
RESIDUAL
```

## 23. Constrained runtime

Expected:

```text
still fail-fast
```

- [x] non-off mode rejected before device work (existing extension rejection
  tests unchanged and green)
- [x] no sampler path accidentally enabled (no sampler changes in Round 9)
- [x] R9-01 recorded as CPU prerequisite only

## 24. Final sign-off

Only if evidence supports it:

```text
No remaining HIGH/MEDIUM implementation defect was found in the reviewed
active Qwen3.8 native parser path or reviewed tool-control API paths under
the CPU-only verified scope.

The future grammar constraint now preserves strict post-trigger tool
sequence semantics: visible suffix prose is rejected while legal
termination and directly consecutive tool entries remain supported.

OpenAI Chat no longer accepts strict schema guarantees or custom-tool wire
semantics that NInfer cannot provide. Anthropic no longer silently weakens
forced, serial, or strict tool-control guarantees to automatic best effort.

Runtime constrained decoding remains unavailable and fail-fast. The
byte-identical semantic quotation ambiguity remains documented.
```

Never sign off with:

```text
bug-free
all tool-call errors impossible
full OpenAI parity
full Anthropic parity
```

Reviewer notes:

```text
R9-01: explicit phases + strict BetweenCalls scanner + can_terminate() (643917cb);
one MSVC STL workaround (std::string::rfind(const char*, size_type) returns npos for a
present literal on MSVC 14.51 — the helper uses std::string overloads only).
R9-02/03/04: serve-layer fail-fast rejections (5e252077, b3b36b9e, f5bd58c3).
Open: Qwen Code and Copilot/MCP client compatibility untested (no clients in this
environment); documented, no speculative compatibility flags.
```
