# NInfer `new_parser_design` — Round 2 Bugfix Implementation Specification

## 0. Purpose, scope, baseline, and relationship to Round 1

This document is the **second independent bugfix/review specification** for the Qwen3.8 tool-call parser on:

- Repository: `Hundsbuah/ninfer`
- Branch: `new_parser_design`
- Round-2 reviewed HEAD: `fcc4eec7fc2696b38681628caa7d251b25750c53`
- Commit message: `fix: tool-call parser F1-F10 bugfixes + constrained flag fail-fast`
- Review date: 2026-09-29
- Primary target: NInfer's Qwen3.8 native tool-call path
- Secondary target: compatibility syntax already accepted by the parser:
  - `<function_calls>...</function_calls>`
  - bare `<function=...>...</function>`
  - `<invoke=...>...</invoke>`
  - `<param=...>...</param>`

Round 1 fixed or materially improved the original F1–F10 findings. **Do not revert those fixes.**
Round 2 exists because a fresh whole-pipeline review of the post-fix commit found new correctness,
recovery-scope, identity-validation, compatibility, and quotation/false-positive problems.

The implementation in this document must therefore be treated as a **delta on top of Round 1**.

The previous implementation/progress documents remain historical evidence. Do not edit their
completed findings to make this round appear green. Record all Round-2 work in a new progress file:

```text
NInfer_new_parser_design_round2_bugfix_progress.md
```

---

## 0.1 Hard test constraint

**Do not run GPU runtime tests, real-model inference tests, CUDA inference benchmarks, or `_real`
tests.**

The local AI/model is running on the server. This task is CPU/parser/integration verification only.

Allowed:

- normal project compilation, including CUDA compilation if required to build the project;
- CPU-only unit/integration tests;
- parser/grammar/frontend/output-session tests;
- CLI/server option parsing and validation tests that do not initialize/use the GPU;
- deterministic parser probes;
- table-driven adversarial tests;
- property-style CPU tests;
- streaming/chunk-boundary tests;
- full CPU CTest runs with `_real` excluded and GPU-runtime tests skipped/hidden.

Prohibited:

- model inference;
- GPU parser validation;
- `_real` tests;
- model benchmarks;
- speculative decoding benchmarks;
- GPU performance testing;
- changing the GPU sampling/logit path merely to satisfy this task.

If a test unexpectedly attempts GPU execution, **do not force it to run**. Record its exact name,
skip behavior, and reason in the progress MD.

---

# 1. Round-2 conclusions that must drive the implementation

The Round-1 design made the parser substantially safer, especially:

- parameter payload is now opaque rather than recursively nested XML;
- an open parameter value is not executable;
- an open function/invoke is not executable;
- function close is the executability boundary;
- streaming and one-shot share the same structural parser;
- marker restart on a breaking `<` is improved;
- the silent `--constrained-tool-decoding tool-calls-only` no-op now fails fast.

Those properties must remain true.

However, the current commit is **not yet fully correct**. The following Round-2 findings are blocking
or hardening items.

| ID | Finding | Severity | Native Qwen `<tool_call>` path |
|---|---|---:|---|
| CR1 | Recovery can escape from a failed still-open wrapper and execute a nested later wrapper | **HIGH** | **YES** |
| CR2 | `<function_calls>` can reach `Complete` at EOF while its wrapper is still open | MEDIUM | compatibility |
| CR3 | Bytes after `</function_calls>` are not validated because the parser returns `Complete` immediately | MEDIUM | compatibility |
| CR4 | Bare function followed by a complete `<tool_call>` is misclassified by close-continuation lookahead | MEDIUM | compatibility / shared boundary logic |
| CR5 | Tolerant mode bypasses the declared-tool-name check and can emit out-of-set tool calls | **HIGH** | **YES when tolerant** |
| CR6 | Syntactically valid tool markup inside final content/code fences can become a phantom tool call | **HIGH false-positive class** | **YES** |
| VG1 | Current HEAD has no GitHub status checks/workflow runs; prior green claims require an independent Round-2 CPU run | verification gap | all |

There is also a **fundamental protocol ambiguity** that cannot be fully repaired by a byte parser:
unfenced ordinary content can contain byte-for-byte valid tool-call markup. If the markup is
identical to a genuine call, syntax alone cannot prove author intent. Round 2 must mitigate the
cases that *are* distinguishable (especially fenced examples and undeclared tools) and explicitly
document the residual ambiguity instead of claiming it is solved.

---

# 2. Evidence classification

Use these labels in comments, tests, progress logs, and final review:

- **Verified from source** — directly visible in the pinned `fcc4eec7` code.
- **Deterministically derived** — follows from a concrete state-machine path and must be pinned by a
  regression test before implementation.
- **Externally corroborated parser class** — the same general failure class is documented by another
  current Qwen parser implementation, but NInfer's own bug must still be proven from NInfer source/tests.
- **Fundamental ambiguity** — cannot be fully resolved without extra framing, escaping, a structured
  output channel, constrained generation, or a policy that intentionally trades availability for safety.
- **Unknown** — insufficient evidence; do not convert to a claim.

For every finding below:

1. add the reproducer first;
2. prove the old behavior;
3. implement the smallest coherent fix;
4. run targeted tests;
5. perform a counterexample pass against the fix;
6. update the progress MD.

---

# 3. Architecture that must be preserved

Do not throw away the useful Round-1 design.

Preserve:

1. `tool_call_grammar.{h,cpp}`
   - one authoritative syntax grammar;
   - `NoMatch / NeedMore / Complete / Invalid`;
   - shared marker classification.

2. `tool_call_stream.{h,cpp}`
   - one-shot/stream structural parser;
   - objective parse progress;
   - recovery policy separated from parse facts.

3. `tool_call_parser.cpp`
   - contract/schema normalization after structural parse.

4. `tool_call_grammar_state.{h,cpp}`
   - CPU grammar-state logic;
   - **do not** wire it into GPU sampling in this task.

5. `OutputSession`
   - reasoning/content channel separation;
   - tool parsing receives only the final content channel.

Do not solve Round-2 defects by reintroducing separate ad-hoc parsers.

---

# 4. New invariants for Round 2

## R2-I1 — An open failed wrapper owns its scope

If a parse fails while:

```cpp
progress.wrapper_at_break != ToolWrapperKind::None
```

a marker found later inside that still-unclosed wrapper must **not** be reinterpreted as a new
top-level executable tool region.

A recovery search may restart only when it has proven that it has left the failed wrapper's scope.

When such proof is impossible from the wire bytes, choose non-execution over recovery.

---

## R2-I2 — Wrapper balance means wrapper balance

For every explicit wrapper family supported by the grammar:

```text
<tool_call>       ... </tool_call>
<function_calls>  ... </function_calls>
```

strict mode must never report `Complete` while an opener remains unmatched.

A compatibility exception must not silently redefine `Complete`.

If backward compatibility intentionally allows an omitted close, model it as an explicit recovery
policy / truncation state, not structurally clean completion.

---

## R2-I3 — Consuming a wrapper close does not consume the rest of the output

After any wrapper close, the state machine must continue at `Top` and verify the remaining bytes.

A parser may report `Complete` only after it has consumed/validated the accepted terminal region,
including the rule for any trailing whitespace/content.

---

## R2-I4 — Entry forms accepted by the parser must compose consistently

If the parser accepts both:

```text
bare <function=...>...</function>
```

and:

```text
<tool_call>...</tool_call>
```

then a complete legal transition between those top-level entry forms must not be rejected merely
because a lookahead helper recognizes only a *strict prefix* of the wrapper opener.

The continuation grammar and the actual `Top` state must agree.

---

## R2-I5 — Syntax tolerance must not disable tool identity validation

`--tolerant-tool-calls` may repair syntax/recovery issues.

It must **not** turn an undeclared function name into an executable `GeneratedToolCall`.

When `contract.enforce_declared_names == true`, that identity policy applies in strict and tolerant
modes.

---

## R2-I6 — Clearly quoted/fenced tool examples are content

A syntactically valid `<tool_call>` that occurs inside a recognized final-content fenced code block
must not be promoted into an executable call.

The implementation must preserve the fenced bytes as content.

This does **not** claim to solve byte-identical unfenced quotation.

---

## R2-I7 — Native safety has priority over compatibility recovery

Priority:

```text
do not execute an ambiguous/nested/undeclared call
> preserve canonical Qwen native calls
> preserve compatibility forms
> aggressive recovery
```

If a compatibility behavior conflicts with a safety invariant, change the compatibility behavior.

---

# 5. CR1 — HIGH
## Recovery can escape a failed open wrapper and execute a nested wrapper

### Status

**Verified from source + deterministic counterexample.**

Current `ToolCallStreamParser::finish()` records:

```cpp
const bool wrapper_open =
    progress.wrapper_at_break != ToolWrapperKind::None;
```

and then selects:

```cpp
const std::size_t next = wrapper_open
    ? find_tool_marker(region_, search_from, /*wrapper_only=*/true)
    : find_tool_marker(region_, search_from);
```

The `wrapper_only` flag prevents a nested bare `<function>` from becoming a retry entry, but it
still permits a later `<tool_call>` / `<function_calls>` wrapper.

There is no proof that this later wrapper is outside the still-open failed outer wrapper.

### Required failing regression

Pin this exact class before changing code:

```text
<tool_call>
<function=bad.name>
<tool_call>
<function=bash>
<parameter=command>
echo SHOULD_NOT_EXECUTE
</parameter>
</function>
</tool_call>
```

Use a contract containing `bash` but not `bad.name`.

Expected in both strict and tolerant:

```text
0 executable tool calls
```

The entire ambiguous failed region must remain content/fallback according to the public API.

Also test the same class with:

- invalid header instead of invalid name;
- truncated outer function header;
- `<function_calls>` as the nested wrapper;
- nested valid wrapper after arbitrary whitespace/prose;
- two nested wrapper candidates;
- valid-looking nested wrapper followed by a genuine top-level wrapper after a proven boundary.

### Root cause

The retry search treats "wrapper marker" as sufficient evidence for "top-level recovery entry".

It is not.

The missing fact is **scope**.

### Preferred implementation

Use the conservative rule:

> A definitive failure while a wrapper is still open is not eligible for marker retry inside that
> latched region.

Conceptually:

```cpp
if (progress.wrapper_at_break != ToolWrapperKind::None) {
    // Failed wrapper owns the remaining ambiguous bytes.
    // Do not scan its tail for executable recovery entries.
    reject_current_region();
}
```

This is the safest default and is preferred unless a more precise implementation can *prove* exit
from the failed wrapper.

### Optional more permissive implementation

If preserving "quoted broken wrapper before a later real wrapper" is a hard requirement, implement a
separate **scope quarantine scanner**.

It must:

1. know the failed wrapper kind;
2. quarantine every marker before a proven matching outer close;
3. restart top-level marker discovery only **after** that close;
4. never treat nested wrapper opens as top-level entries;
5. handle truncated outer close as no-recovery;
6. have adversarial tests where wrapper-looking literals appear in payload.

Do not simply search for the next `<tool_call>`.

Because the payload syntax has no universal escaping, if the scope boundary remains ambiguous,
reject recovery.

### Tests

Mandatory:

```text
test_failed_open_tool_call_never_recovers_nested_tool_call
test_failed_open_tool_call_never_recovers_nested_function_calls
test_failed_open_function_calls_never_recovers_nested_tool_call
test_failed_wrapper_truncation_does_not_reenter
test_recovery_only_restarts_after_proven_wrapper_exit
```

Run all in:

- one-shot;
- streaming 1-byte chunks;
- chunk sizes 2, 3, 5, 7;
- every split around the failed header and nested wrapper opener.

### Acceptance criteria

- No marker located within an unclosed failed wrapper can become executable.
- The implementation contains an explicit proof/policy for wrapper-scope exit.
- Tests demonstrate that the old CR1 reproducer fails before the fix and passes after it.
- Existing real-call recovery outside a proven failed scope remains functional if intentionally supported.

---

# 6. CR2 — MEDIUM
## `<function_calls>` can be structurally `Complete` at EOF while still open

### Status

**Verified from source and explicitly pinned by the current tests.**

After a function inside a `<function_calls>` wrapper closes, the parser moves to `Top`.

Current `Top` EOF handling is effectively:

```cpp
if (i == text.size()) {
    if (top) {
        complete();
        return out;
    }
}
```

It does not first require:

```cpp
s.wrapper == ToolWrapperKind::None
```

The current test suite explicitly calls the missing `</function_calls>` close a "pre-existing
contract" and expects clean completion in strict and tolerant modes.

This conflicts with the Round-1 invariant:

> No sequence with more wrapper opens than closes may be reported `Complete`.

### Required behavior

For:

```text
<function_calls>
<function=read>
<parameter=path>a</parameter>
</function>
```

strict:

```text
no tool-call response
MalformedStructure
```

tolerant:

- may retain the fully function-closed call;
- must record `TruncatedTail`;
- must not report clean `None` diagnostic;
- must not call the region structurally `Complete`.

### Implementation

At `Top` EOF, branch on wrapper state:

```cpp
if (i == text.size()) {
    if (s.wrapper == ToolWrapperKind::None) {
        complete();
    } else {
        truncated(false, i);
    }
    return out;
}
```

The exact code may differ, but `Complete` must imply wrapper balance.

### Tests to change/add

Change the existing "wrapper close is optional at region end" expectation.

Add:

```text
test_function_calls_missing_wrapper_close_strict_rejects
test_function_calls_missing_wrapper_close_tolerant_truncated_tail
```

Test 1 and N calls.

Test all finish reasons accepted by the parser API.

### Acceptance criteria

- `ToolCallRegionTermination::Complete` implies `wrapper_at_end == None`.
- No explicit wrapper family has an implicit-close exception hidden inside structural parsing.
- Any compatibility omission is represented in recovery diagnostics, not clean completion.

---

# 7. CR3 — MEDIUM
## Trailing bytes after `</function_calls>` are not validated

### Status

**Verified from source.**

Current branch:

```cpp
if (s.wrapper == ToolWrapperKind::FunctionCalls &&
    starts_with_at(text, i, "</function_calls>")) {
    ...
    s.wrapper = ToolWrapperKind::None;
    complete();
    return out;
}
```

The parser returns as soon as it consumes `</function_calls>`.

Therefore:

```text
<function_calls>
...
</function_calls>
THIS_IS_TRAILING_CONTENT
```

can be reported complete without validating `THIS_IS_TRAILING_CONTENT`.

This differs from `<tool_call>` handling, which closes the wrapper and returns to `Top`.

### Required implementation

After a valid `</function_calls>` close:

```cpp
s.pos = after_close;
s.wrapper = ToolWrapperKind::None;
s.mode = RegionState::Mode::Top;
continue;
```

Do not call `complete()` there.

Let `Top` decide:

- EOF/whitespace → complete;
- another intentionally supported top-level region → parse according to policy;
- non-whitespace trailing prose → `TrailingContent`;
- partial marker → truncation if objectively a growable prefix.

### Regression tests

```text
test_function_calls_close_then_trailing_content_strict_rejects
test_function_calls_close_then_trailing_content_tolerant_policy
test_function_calls_close_then_whitespace_completes
test_function_calls_close_then_next_valid_region
test_function_calls_close_then_partial_marker
```

Explicit fixture:

```text
<function_calls>
<function=read>
<parameter=path>a</parameter>
</function>
</function_calls>
EXTRA
```

Strict must not report a clean tool response.

### Acceptance criteria

Every explicit wrapper close uses the same high-level terminal rule:

```text
close wrapper
→ wrapper=None
→ Top
→ validate remainder
```

unless a formally documented grammar rule proves otherwise.

---

# 8. CR4 — MEDIUM
## Bare function followed by a complete `<tool_call>` is rejected by lookahead inconsistency

### Status

**Deterministically derived from the current continuation classifier.**

`classify_function_close_continuation()` for `wrapper == None` accepts:

- another complete/partial function opener;
- strict prefix of `</function_calls>`;
- strict prefix of `<tool_call>`.

It does **not** accept a **complete** `<tool_call>` opener in this path.

`is_strict_prefix_of("<tool_call>", text, at)` is false when the complete wrapper and further bytes
are already present.

### Counterexample

```text
<function=read>
<parameter=path>
a.txt
</parameter>
</function>
<tool_call>
<function=bash>
<parameter=command>
echo ok
</parameter>
</function>
</tool_call>
```

Both entry forms are individually supported by the parser. Their legal composition must not make
the first parameter close look like payload merely because the second opener is complete rather
than partial.

### Required implementation

Make function-close continuation use the same top-level entry policy used by `Top`.

Preferred: create a helper such as:

```cpp
enum class TopLevelContinuation {
    Invalid,
    CompleteEntry,
    NeedMoreEntry,
    WrapperClose,
    End,
};
```

or an equivalent reusable classifier.

Do not accumulate another independent list of accepted entry tokens.

At minimum, for `wrapper == None`, a complete:

```text
<tool_call>
<function_calls>
```

must be recognized consistently if those are accepted top-level entry forms.

### Tests

```text
test_bare_function_followed_by_tool_call
test_bare_invoke_followed_by_tool_call
test_bare_function_followed_by_function_calls
test_bare_function_followed_by_bare_function
test_complete_and_partial_next_entry_have_same_boundary_result
```

Run each one-shot and streamed.

### Acceptance criteria

For every top-level entry form:

```text
complete previous call + complete next entry
```

and:

```text
complete previous call + partial prefix of next entry
```

must agree on the previous call's structural boundary.

---

# 9. CR5 — HIGH
## Tolerant mode bypasses declared-tool identity validation

### Status

**Verified from source.**

Current check:

```cpp
if (!policy.tolerant && policy.enforce_declared_names &&
    (policy.declared_check == nullptr ||
     !policy.declared_check(policy.contract, name))) {
    invalid(ToolCallParseFailure::UndeclaredTool, i);
    return out;
}
```

Therefore `tolerant=true` disables the active tool-name contract.

Downstream normalization then performs:

```cpp
const Contract::Tool* tool = find_tool_contract(contract, raw.name);
```

but still builds:

```cpp
GeneratedToolCall{.name = raw.name, ...}
```

when no contract is found.

Result: a syntactically accepted tolerant call can escape with a name not present in `request.tools`.

### Why this is a separate policy bug

Syntax recovery and function identity are independent questions.

Tolerant should mean:

```text
repair/retain allowed syntax damage
```

not:

```text
allow the model to invent function identities
```

Out-of-set tool calls are especially undesirable for coding agents because the client may:

- reject the call;
- abort the turn;
- enter another recovery/handoff;
- expose parser errors;
- mis-handle placeholder tool names echoed from the template.

### Required implementation

Apply declared-name validation in both modes whenever:

```cpp
policy.enforce_declared_names == true
```

Conceptually:

```cpp
if (policy.enforce_declared_names &&
    (policy.declared_check == nullptr ||
     !policy.declared_check(policy.contract, name))) {
    invalid(ToolCallParseFailure::UndeclaredTool, i);
    return out;
}
```

Do not gate it on `!policy.tolerant`.

### Recovery semantics with earlier valid calls

Define and test explicitly:

- single undeclared call → no emitted call;
- valid first call + later undeclared call in a recoverable sequence:
  - strict: reject whole structured region;
  - tolerant: it may retain **only earlier already complete declared calls** with a truncation/broken-tail diagnostic;
  - never emit the undeclared call.

### Defense in depth

After structural parsing and before `GeneratedToolCall` leaves the frontend, add a defensive
invariant/assertion/validation path:

> if the contract enforces declared names, no generated call may have a name absent from the
> contract.

Do not rely exclusively on the state-machine branch.

Prefer a graceful parser diagnostic/fallback over a process-crashing assertion in production.

### Mandatory tests

```text
test_tolerant_rejects_undeclared_single_tool
test_tolerant_never_emits_undeclared_tool_after_valid_call
test_strict_and_tolerant_share_declared_name_policy
test_output_normalization_cannot_emit_out_of_set_name
test_placeholder_function_name_is_not_executable
test_example_function_name_is_not_executable
```

Test streaming and one-shot.

### Acceptance criteria

With `enforce_declared_names=true`:

```text
for every emitted GeneratedToolCall:
    call.name ∈ declared_tools
```

in **all modes**.

---

# 10. CR6 — HIGH false-positive class
## Valid tool markup in final content/code fences can become a phantom call

### Status

**Verified architectural exposure in NInfer + externally corroborated Qwen-parser failure class.**

NInfer's `OutputSession` correctly feeds only final content deltas to `ToolCallOutputDecoder`:

```cpp
if (delta.channel == OutputChannel::Content) {
    delta.text = impl_->tool_call_output.feed(delta.text);
}
```

This protects reasoning-channel text when reasoning separation works.

However, the marker scanner itself has no Markdown quotation/fence state. A valid tool example in
normal final content can therefore latch.

Example:

````text
Here is an example:

```xml
<tool_call>
<function=bash>
<parameter=command>
echo hello
</parameter>
</function>
</tool_call>
```

That block is only documentation.
````

If `bash` is declared, tool-name validation alone cannot distinguish the example from a genuine
call.

Current public Qwen parser reports from vLLM document this same practical false-positive class for
fenced examples and quoted markup. Treat those reports as corroboration, not proof of NInfer's
implementation details.

### Required Round-2 mitigation: fenced-code guard

Implement a lightweight, streaming-safe final-content fence tracker before marker latch.

Required behavior:

- outside a fence: normal marker scanning;
- inside a recognized fenced code block: tool markers are ordinary content;
- after the fence closes: normal marker scanning resumes;
- fence tracking must be chunk-boundary invariant.

Support at least CommonMark-style practical forms used by model output:

- backtick fences ``````
- tilde fences `~~~`
- opener length >= 3;
- closing fence uses same character and length >= opener length;
- optional language/info string on the opening line;
- line-oriented recognition;
- CRLF and LF;
- reasonable indentation handling consistent with the chosen documented rule.

Do not attempt full Markdown parsing beyond what is needed for deterministic fence state.

### Safety rule for unclosed fence

If final content opens a code fence and never closes it, remain in content mode through EOF.

Do not execute a tool marker that appears inside that unclosed fence.

### Required tests

```text
test_tool_call_inside_backtick_fence_is_content
test_tool_call_inside_tilde_fence_is_content
test_tool_call_after_closed_fence_executes
test_tool_call_before_fence_executes_if_terminal_policy_allows
test_unclosed_fence_suppresses_nested_marker
test_longer_fence_contains_shorter_backticks
test_fence_info_string
test_crlf_fence
test_fence_split_at_every_byte
test_real_call_immediately_after_closing_fence
```

### Important limitation: unfenced byte-identical quotation

Do **not** claim that CR6 fully solves quoted tool markup.

This remains fundamentally ambiguous:

```text
The format is:
<tool_call>
<function=bash>
...
</function>
</tool_call>
```

versus a genuine emitted call with the same bytes.

A byte parser cannot infer intent from identical bytes.

Round 2 must therefore:

1. fix fenced examples;
2. reject undeclared names (CR5);
3. keep the native "no suffix" template contract;
4. document residual unfenced ambiguity;
5. prefer non-execution where structure is not proven.

### Optional stronger hardening

Introduce an explicit entry mode:

```cpp
enum class ToolCallEntryMode {
    QwenNativeCanonical,  // only <tool_call> as executable entry
    Compatibility,        // also bare function/invoke and function_calls
};
```

For the NInfer Qwen3.8 native template, prefer `QwenNativeCanonical`.

Benefits:

- smaller false-positive surface;
- fewer compatibility transitions in the native path;
- clearer recovery policy.

Do not make this larger refactor a prerequisite if it risks destabilizing the current parser; treat it
as a separate hardening commit if necessary.

---

# 11. VG1 — Verification gap
## No GitHub CI/status evidence on the pinned Round-2 baseline

At review time, commit:

```text
fcc4eec7fc2696b38681628caa7d251b25750c53
```

has:

```text
combined statuses: none
workflow runs: none
```

The previous progress document contains local test results, but Round 2 must not treat them as proof
that the new findings are fixed.

### Required action

Perform a new independent CPU-only verification after the Round-2 implementation.

Record:

- configure/build command;
- compiler/toolchain;
- targeted parser tests;
- frontend/output-session tests;
- option/engine-validation tests;
- full CPU CTest result;
- every skipped test;
- any non-parser failures and why they are unrelated;
- final implementation SHA.

---

# 12. Recovery redesign for CR1

## 12.1 Current unsafe assumption

Current recovery roughly assumes:

```text
failed wrapper + later wrapper marker
→ later marker can be top-level
```

That is not sufficient.

## 12.2 Required recovery scope state

Add an explicit recovery-scope concept, for example:

```cpp
enum class RecoveryScope {
    TopLevel,
    FailedToolCallWrapper,
    FailedFunctionCallsWrapper,
};
```

or keep the existing wrapper enum but make the policy explicit.

A retry function should receive:

```text
break offset
wrapper-at-break
termination kind
```

and return either:

```text
no retry
```

or a **proven top-level offset**.

Do not let `find_tool_marker()` decide scope by itself.

## 12.3 Preferred conservative algorithm

```text
parse failed
  ↓
wrapper_at_break == None?
  ├─ yes → top-level retry policy may search
  └─ no  → no nested retry; region falls back / prior proven calls only
```

If a prior complete call exists before the failure and tolerant mode allows retention, retaining that
prior call is separate from finding a *new* executable call.

## 12.4 If quarantine scanning is implemented

The scanner must be unit-tested separately.

It may not simply perform:

```cpp
text.find("</tool_call>")
```

and assume the first hit is the real wrapper close if payload can contain that sequence.

When proof is ambiguous, no retry.

---

# 13. Identity-validation redesign for CR5

Separate policy dimensions:

```text
wire syntax policy
recovery/tolerance policy
declared tool identity policy
schema normalization policy
```

Do not encode identity policy as a side effect of strictness.

Suggested parse policy:

```cpp
struct ToolCallParsePolicy {
    bool tolerant_syntax = false;
    bool enforce_declared_names = true;
    ...
};
```

The current `tolerant` field may remain if API compatibility matters, but its semantics must not
disable identity checks.

Add a test that constructs the same output twice:

```text
strict
tolerant
```

and verifies identical name acceptance/rejection.

---

# 14. Wrapper-state corrections for CR2/CR3

## 14.1 Structural completion invariant

Before `complete()`:

```cpp
assert(s.wrapper == ToolWrapperKind::None);
assert(no open function);
assert(no open parameter value);
```

Use runtime logic, not debug assertions alone, to preserve correctness in release builds.

## 14.2 One wrapper-close exit rule

Refactor both wrapper families toward:

```text
consume matching close
→ wrapper=None
→ Top
```

Avoid a special function-calls branch that calls `complete()` directly.

## 14.3 EOF rule

At EOF:

```text
wrapper=None → Complete
wrapper!=None → EndOfInput
```

Then recovery policy decides whether previously function-closed calls are retained.

---

# 15. Shared top-level continuation grammar for CR4

Avoid maintaining these separately:

- `Top` entry logic;
- function-close continuation;
- parameter-close continuation;
- retry marker classification;
- grammar-state trigger.

Create one reusable concept for top-level entry classification where practical.

The helper must distinguish:

```text
Complete
NeedMore
NoMatch/Invalid
```

for:

- `<tool_call>`
- `<function_calls>` if compatibility enabled
- bare `<function...>` if compatibility enabled
- bare `<invoke...>` if compatibility enabled

Parameter boundary logic should call through this rather than reimplementing a partial subset.

---

# 16. Fence-tracker design for CR6

## 16.1 State

Example:

```cpp
struct ToolFenceState {
    bool in_fence = false;
    char fence_char = '\0';     // '`' or '~'
    std::size_t fence_len = 0;
    bool at_line_start = true;
    std::string line_prefix;
};
```

Exact fields may differ.

## 16.2 Ordering

The content scanner should conceptually process:

```text
decoded content
→ fence-state update
→ if outside fence: marker candidate machine
→ if inside fence: publish/hold as ordinary content only
```

Be careful not to publish bytes twice when a fence delimiter and marker candidate share chunks.

## 16.3 Streaming requirement

The result must be independent of chunks:

```text
one chunk
1 byte chunks
2/3/5/7 byte chunks
every two-way split
```

## 16.4 Interaction with reasoning

Fence state applies to final content only.

Do not mix fence state across reasoning/content channel transitions.

## 16.5 Interaction with real calls

After a closing fence:

````text
```xml
<tool_call>example...</tool_call>
```
<tool_call>
<function=real>...</function>
</tool_call>
````

the first block is content and the second call must still be recognized.

---

# 17. Mandatory Round-2 tests

## 17.1 CR1 recovery-scope corpus

Include:

```text
invalid outer function name → nested valid tool_call
invalid outer header → nested valid tool_call
truncated outer function → nested valid tool_call
failed function_calls → nested tool_call
failed tool_call → nested function_calls
multiple nested candidate wrappers
later candidate after no proven close
later candidate after proven safe exit (if supported)
```

## 17.2 CR2/CR3 wrapper corpus

For `<function_calls>`:

```text
0 calls + close
1 call + close
2 calls + close
1 call missing close
2 calls missing close
close + whitespace
close + trailing prose
close + next valid wrapper
partial close at every byte
```

## 17.3 CR4 entry-composition matrix

Rows = previous entry type:

```text
bare function
bare invoke
tool_call
function_calls
```

Columns = next entry type.

For each supported pair:

- complete next entry;
- partial next entry at every byte;
- whitespace variations.

Pin expected behavior explicitly.

## 17.4 CR5 tool-name matrix

Declared set:

```text
read
bash
write
```

Try:

```text
read
bash
write
function_name
example_function_name
unknown
bad.name
```

Run strict/tolerant.

Expected:

- declared valid names: may pass if structure valid;
- undeclared syntactically valid names: never emitted;
- syntactically invalid name: `InvalidToolName`;
- identity policy does not change with tolerance.

## 17.5 CR6 fenced-content corpus

At minimum:

````text
```xml
<tool_call>...</tool_call>
```
````

````text
~~~~
<tool_call>...</tool_call>
~~~~
````

nested backtick text using longer outer fence;

unclosed fence;

CRLF;

marker split across chunks;

closing fence immediately followed by real call.

---

# 18. Cross-product test matrix

The critical bugs are interaction bugs. Do not test each dimension in isolation only.

Cross at least:

- strict vs tolerant;
- one-shot vs streaming;
- finish reason:
  - StopToken
  - StopString where valid
  - OutputLimit
  - ContextCapacity
  - Cancelled if supported;
- wrapper family:
  - ToolCall
  - FunctionCalls
  - None/bare compatibility;
- function family:
  - Function
  - Invoke;
- parameter family:
  - Parameter
  - Param;
- declared vs undeclared tool;
- fenced vs unfenced content.

A table-driven test helper is preferred to duplicated ad-hoc assertions.

---

# 19. Every-byte cut requirements

Re-run the Round-1 cut matrix and add cuts around Round-2 boundaries:

```text
<tool_call>
<function_calls>
</tool_call>
</function_calls>
<function=
<invoke=
```

and fence delimiters:

```text
```
~~~~
```

For every prefix:

- compare one-shot and streaming;
- no cut inside an open function may emit the current function;
- no cut inside an unclosed failed wrapper may expose a nested call;
- an undeclared function may never be emitted;
- a fence-open prefix may not accidentally enable tool parsing inside the code block.

---

# 20. CPU-only build/test commands

Use the existing Round-1 build if clean and compatible, or create a fresh Round-2 build directory.

Recommended:

```powershell
cmake --build build-new-parser --config Release --parallel 8
```

Targeted:

```powershell
ctest --test-dir build-new-parser -C Release `
  -R "ninfer_(tool_call_parser|tool_call_grammar|tool_call_grammar_state|qwen3_5_frontend|engine_options_validation)_test" `
  --output-on-failure
```

Also run relevant OutputSession/frontend tests by their actual registered names if the regex does not
include them.

Full CPU gate:

```powershell
$env:CUDA_VISIBLE_DEVICES="99"
ctest --test-dir build-new-parser -C Release `
  -E "_real" `
  --output-on-failure `
  --parallel 8
```

If hiding the GPU causes a known test to return the wrong skip exit code, record it exactly. Do not
silently call it green.

---

# 21. Implementation phases

## Phase R2-0 — Freeze and reproduce

1. Record HEAD and clean/dirty worktree.
2. Record toolchain/build directory.
3. Build current baseline.
4. Add failing tests for CR1–CR6.
5. Run them against `fcc4eec7` before product changes.
6. Record actual failure/output for every reproducer.

Do not start by changing implementation.

---

## Phase R2-1 — CR1 recovery-scope isolation

1. Implement conservative failed-wrapper scope ownership.
2. Add/keep failing nested-wrapper tests.
3. Verify no nested call escapes.
4. Re-run all Round-1 recovery tests.
5. Review availability regressions separately from safety.

---

## Phase R2-2 — CR2/CR3 wrapper completion cleanup

1. Remove implicit clean EOF for open `function_calls`.
2. Make wrapper close return to `Top`.
3. Validate trailing content.
4. Re-run wrapper/cut matrix.
5. Update docs that currently call the missing close a "pre-existing clean contract".

---

## Phase R2-3 — CR4 continuation unification

1. Centralize top-level continuation classification.
2. Test complete/partial entry pairs.
3. Re-run parameter-boundary adversarial tests from Round 1.
4. Ensure literal wrapper text inside payload remains payload where required.

---

## Phase R2-4 — CR5 declared-tool enforcement

1. Remove tolerance bypass.
2. Add defense-in-depth before emitting generated calls.
3. Test previous-valid + later-undeclared recovery.
4. Test placeholder names.
5. Verify normal declared calls unchanged.

---

## Phase R2-5 — CR6 fenced-content guard

1. Implement streaming fence tracker.
2. Add fence corpus.
3. Verify chunk invariance.
4. Verify real call after closed fence.
5. Document unfenced ambiguity.

---

## Phase R2-6 — Documentation

Update at minimum:

```text
docs/tool_call_parser.md
docs/new_parser_progress.md or a new round2 progress note
CLI/serve docs only if behavior exposed to users changes
```

Correct any Round-1 statements that are no longer accurate, especially:

- `function_calls` EOF clean completion;
- recovery re-entry claims;
- declared-name behavior in tolerant mode;
- quotation/fence limitation.

Do not erase historical evidence from the old progress document; append/correct with explicit Round-2
notes or maintain the new Round-2 file.

---

## Phase R2-7 — Independent CPU verification

Run targeted and full CPU suites from a clean rebuilt state.

No GPU/runtime inference.

---

# 22. Mandatory code-review loops

Do not stop because tests pass.

## Review A — State-machine invariants

Read complete current files:

```text
src/models/qwen3_5/frontend/tool_call_grammar.h
src/models/qwen3_5/frontend/tool_call_grammar.cpp
src/models/qwen3_5/frontend/tool_call_stream.h
src/models/qwen3_5/frontend/tool_call_stream.cpp
src/models/qwen3_5/frontend/tool_call_parser.h
src/models/qwen3_5/frontend/tool_call_parser.cpp
src/models/qwen3_5/frontend/tool_call_grammar_state.h
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp
src/models/qwen3_5/frontend/output_session.h
src/models/qwen3_5/frontend/output_session.cpp
```

Questions:

- Can a nested call escape an open failed wrapper?
- Can `Complete` occur while a wrapper is open?
- Can bytes after a wrapper close go unchecked?
- Do partial and complete next-entry lookaheads agree?
- Can tolerant emit an undeclared name?
- Can final-content fenced markup latch?
- Can the fixes reopen F1–F10?

Any HIGH/MEDIUM finding → fix, regression test, repeat Review A.

---

## Review B — Adversarial recovery review

Actively construct counterexamples around:

- invalid outer header + nested valid call;
- undeclared outer + declared inner;
- declared outer + undeclared next;
- nested wrappers;
- multiple false markers;
- missing wrapper closes;
- wrapper close literals inside parameter payload;
- fenced examples;
- unclosed fences;
- code containing literal `<tool_call>`;
- CRLF and whitespace;
- every finish reason.

Repeat until no reproducible HIGH/MEDIUM finding remains.

---

## Review C — Native-Qwen review

Use the actual NInfer `tools/chat_templates/qwen3_8.jinja`.

Verify the exact native form:

```text
<tool_call>
<function=NAME>
<parameter=NAME>
VALUE
</parameter>
</function>
</tool_call>
```

Test:

- no args;
- one arg;
- many args;
- long code payload;
- literal tag-like payload;
- duplicate args;
- declared name;
- undeclared name;
- prose before call;
- no suffix;
- accidental suffix;
- fenced example before real call.

Focus the final correctness claim specifically on this path.

---

## Review D — Compatibility review

Only after native path is green.

Verify:

- function_calls;
- bare function;
- invoke;
- param;
- compositions among them.

A compatibility failure must not be fixed by weakening native safety.

---

## Review E — Whole-output pipeline

Trace:

```text
decoded tokens
→ reasoning/content separation
→ OutputSession content delta
→ fence guard
→ marker candidate
→ latched region
→ structural parser
→ recovery scope policy
→ declared-name validation
→ argument normalization
→ GeneratedToolCall
→ API response serialization
```

For at least:

1. genuine native call;
2. CR1 nested-recovery attack;
3. tolerant undeclared call;
4. fenced example;
5. malformed open function;
6. valid call after a closed fence.

---

## Review F — Independent CPU full suite

Run full CPU test suite from clean build state.

Document every failure and every skip.

Do not declare "full suite green" if the command returns failures; instead state exact exceptions and
their proven unrelated root cause.

---

# 23. Definition of done

Round 2 is complete only when all are true:

- [ ] CR1 reproducer added and fails on `fcc4eec7`.
- [ ] CR1 fixed: no re-entry inside unclosed failed wrapper.
- [ ] CR2 fixed: open `function_calls` cannot be `Complete`.
- [ ] CR3 fixed: trailing bytes after `</function_calls>` are validated.
- [ ] CR4 fixed: complete/partial top-level entry continuations are consistent.
- [ ] CR5 fixed: tolerant never emits undeclared tools.
- [ ] CR5 defense-in-depth exists or equivalent proof is documented.
- [ ] CR6 fenced code examples remain content.
- [ ] CR6 unclosed fence behavior is safe.
- [ ] Residual unfenced byte-identical ambiguity is explicitly documented.
- [ ] Round-1 F1–F10 targeted regressions remain green.
- [ ] One-shot/streaming equivalence remains green.
- [ ] Every-byte cuts around Round-2 boundaries are green.
- [ ] Native Qwen3.8 canonical path is independently reviewed.
- [ ] Compatibility path is reviewed only after native path.
- [ ] No HIGH/MEDIUM finding remains after Review A.
- [ ] No HIGH/MEDIUM finding remains after adversarial Review B.
- [ ] Whole-pipeline Review E is complete.
- [ ] Independent CPU targeted suite passes.
- [ ] Full CPU suite result is recorded truthfully.
- [ ] No GPU runtime/inference test was run.
- [ ] Final implementation SHA recorded.
- [ ] Current docs no longer claim open `function_calls` is clean completion.
- [ ] Current docs no longer imply tolerant mode may bypass tool identity.
- [ ] Current docs describe fenced and unfenced quotation limits accurately.

---

# 24. Specific regression fixtures to copy into the test plan

## CR1

```text
<tool_call>
<function=bad.name>
<tool_call>
<function=bash>
<parameter=command>
echo SHOULD_NOT_EXECUTE
</parameter>
</function>
</tool_call>
```

Expected:

```text
0 calls
```

---

## CR2

```text
<function_calls>
<function=read>
<parameter=path>a</parameter>
</function>
```

Expected:

```text
strict: reject / MalformedStructure
tolerant: read may be retained / TruncatedTail
```

---

## CR3

```text
<function_calls>
<function=read>
<parameter=path>a</parameter>
</function>
</function_calls>
EXTRA
```

Expected strict:

```text
TrailingContent
0 emitted calls under all-or-nothing strict behavior
```

---

## CR4

```text
<function=read>
<parameter=path>a.txt</parameter>
</function>
<tool_call>
<function=bash>
<parameter=command>echo ok</parameter>
</function>
</tool_call>
```

Expected according to the documented multi-entry policy:

```text
both complete entries parse consistently
```

or, if native-only policy is intentionally selected:

```text
compatibility behavior explicitly disabled/rejected
```

but never "previous parameter close becomes payload because the next wrapper is complete".

---

## CR5

Declared tools:

```text
read
```

Output:

```text
<tool_call>
<function=example_function_name>
</function>
</tool_call>
```

Expected in strict and tolerant:

```text
0 emitted calls
UndeclaredTool
```

---

## CR6

````text
Example:

```xml
<tool_call>
<function=bash>
<parameter=command>echo example</parameter>
</function>
</tool_call>
```
````

Expected:

```text
content preserved
0 calls
```

Positive control immediately after:

```text
<tool_call>
<function=bash>
<parameter=command>echo real</parameter>
</function>
</tool_call>
```

Expected:

```text
1 real call
```

---

# 25. Research/primary-source notes

These are contextual sources, not substitutes for NInfer regression tests.

## NInfer pinned source

```text
https://github.com/Hundsbuah/ninfer/tree/fcc4eec7fc2696b38681628caa7d251b25750c53
```

Primary reviewed files:

```text
src/models/qwen3_5/frontend/tool_call_stream.cpp
src/models/qwen3_5/frontend/tool_call_stream.h
src/models/qwen3_5/frontend/tool_call_grammar.cpp
src/models/qwen3_5/frontend/tool_call_grammar.h
src/models/qwen3_5/frontend/tool_call_parser.cpp
src/models/qwen3_5/frontend/tool_call_parser.h
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp
src/models/qwen3_5/frontend/output_session.cpp
tools/chat_templates/qwen3_8.jinja
tests/test_tool_call_parser.cpp
tests/test_tool_call_grammar_state.cpp
tests/test_engine_options_validation.cpp
```

## Qwen

Qwen's own function-calling documentation warns that corner cases can produce malformed tool calls
that cannot be parsed. That supports the design principle that a post-generation parser must remain
fail-safe rather than assuming model output is always structurally valid.

```text
https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md
```

## vLLM/Qwen parser corroboration

Current public Qwen parser reports demonstrate the same general classes that Round 2 should defend
against:

```text
https://github.com/vllm-project/vllm/issues/57541
  valid <tool_call> markup in fenced code can become a phantom call

https://github.com/vllm-project/vllm/issues/58147
  quoted markup can produce out-of-set and duplicate phantom calls

https://github.com/vllm-project/vllm/issues/58227
  an unclosed quoted <tool_call> can capture a later genuine call

https://github.com/vllm-project/vllm/blob/main/vllm/parser/qwen3.py
  current Qwen3 XML parser format and parser implementation
```

Do not copy vLLM behavior blindly. Use these only to generate adversarial hypotheses, then prove
NInfer behavior with NInfer tests.

---

# 26. Non-goals in Round 2

Do not:

- implement GPU constrained decoding;
- claim malformed generation is impossible;
- write host-language parsers for shell/C++/JSON inside arbitrary string parameters;
- infer quote intent for byte-identical unfenced valid tool markup;
- weaken `</function>`/`</invoke>` executability requirements;
- reintroduce recursive parameter nesting;
- silently restore tolerant execution of open functions;
- silently accept undeclared tool names;
- run GPU/runtime model tests.

---

# 27. Final implementation principle

The Round-2 parser/recovery order of priorities is:

```text
declared tool identity
+ proven wrapper scope
+ proven function closure
+ payload integrity
+ structural balance
> recovery availability
> compatibility convenience
```

A fix is not complete merely because the original reproducer passes.

For every change, ask:

1. Can this make a nested/quoted call executable?
2. Can this make an undeclared name executable?
3. Can this convert an open wrapper into `Complete`?
4. Can this lose or reinterpret payload?
5. Does streaming behave identically?
6. Does the native Qwen path remain safer than before?
7. Have I added a regression test for the counterexample I just considered?

If any answer is uncertain, the item is not green.
