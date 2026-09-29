# NInfer `new_parser_design` — Bugfix Implementation Specification

## 0. Purpose, scope and pinned baseline

This document is an implementation specification for fixing the remaining parser/recovery defects in:

- Repository: `Hundsbuah/ninfer`
- Branch: `new_parser_design`
- Reviewed HEAD: `9927ff3d92f66748d1cbfc4f9d0f9b9a91db2d9f`
- Review date: 2026-09-29
- Primary target: Qwen3.8 native tool-call wire format used by NInfer
- Secondary target: compatibility forms already accepted by the parser (`function_calls`, `invoke`, `param`, bare function/invoke markers)

The implementation must fix **all findings F1–F10 in this specification**, preserve existing correct parser behavior, add regression coverage for each defect, and complete multiple review/test loops before the work is considered done.

### Hard test constraint

**Do not run GPU runtime tests, real-model inference tests, CUDA inference benchmarks, or `_real` tests.**

The local model is running on the server and GPU-runtime validation is outside this task.

Allowed:

- normal compilation, including CUDA compilation if the build requires it;
- CPU-only parser/grammar/frontend/server tests;
- static code review;
- deterministic CPU probes;
- parser fuzz/property-style CPU tests;
- full CPU CTest suite with GPU/`_real` tests excluded or left skipped.

Do **not** force skipped GPU tests to run.

---

## 1. Required outcome

The finished implementation must satisfy these safety/correctness invariants:

### I1 — Opaque parameter payload

A generated parameter value is payload, not recursively parsed XML.

Literal text such as:

```text
<parameter=fake>
<param=x>
</function>
<tool_call>
```

inside a parameter value must not change parser nesting/state merely because it resembles tool markup.

Only an **unambiguously accepted closing boundary for the current outer parameter** may end the value.

### I2 — Never execute an open function

A function/invoke call must **never** become a `GeneratedToolCall` unless the parser has unambiguously consumed its matching:

```text
</function>
```

or:

```text
</invoke>
```

This requirement applies to strict and tolerant modes and to every finish reason:

- `StopToken`
- `StopString` where applicable
- `OutputLimit`
- `ContextCapacity`
- other terminal reasons supported by the output pipeline

A missing outer `<tool_call>` closing wrapper may be recoverable in tolerant mode **after** the function itself has closed.

### I3 — Truncation must not reopen already closed payload

If a complete parameter close is followed by a partial legal next token, for example:

```text
</parameter>
<paramet
```

the parser must retain the fact that the preceding parameter value was closed. The incomplete next token is a truncation of the next structure, not evidence that the previous `</parameter>` belonged to the payload.

### I4 — Strict structure is actually balanced

Strict mode must reject unbalanced/nested wrapper structures that are not part of the supported wire grammar.

For example, two `<tool_call>` opens may not be satisfied by one `</tool_call>` close.

### I5 — One source of truth for marker progression

Streaming marker detection and the grammar-constraint marker detector must use identical byte progression semantics, including the case in which a failed marker candidate ends on a new `<` byte.

No consumer may silently consume a `<` that could start a fresh marker while another consumer retries it.

### I6 — Recovery must not become more permissive than parsing

Recovery may retain already proven complete calls. It may not invent structure, commit an open payload, commit an open function, or reinterpret ambiguous payload as executable structure merely to increase availability.

### I7 — No silent feature no-op

`--constrained-tool-decoding tool-calls-only` must not silently claim functionality while leaving sampling unchanged.

Because GPU sampling integration is out of scope for this CPU-only task, selecting the non-off mode must either:

1. fail fast with an explicit "not implemented/not available in this build" error; or
2. be removed from the user-facing CLI/server surface until the sampling integration exists.

**Do not implement an unverified GPU masking path in this bugfix.**

---

# 2. Evidence classification

The following categories are used throughout this document.

- **Verified from source:** directly visible in the reviewed implementation.
- **Logically derived:** deterministic consequence of the current state machine/code path; must be turned into a regression test before the fix is accepted.
- **Existing test coverage:** already represented in the branch test suite.
- **Inherent protocol ambiguity:** cannot be solved perfectly without changing the Qwen wire protocol itself; the implementation must fail safely.

The official Qwen-style native format serializes string argument payload without an XML escaping mechanism. Therefore arbitrary payload can contain byte sequences identical to the tool delimiters. A parser cannot make every such sequence unambiguous. The required strategy is:

> Preserve payload whenever a candidate delimiter is not structurally justified. When ambiguity remains, prefer non-execution over executing an incompletely proven call.

Do not claim that the finished parser can losslessly disambiguate arbitrary strings that contain an entire syntactically valid remainder of the tool-call protocol. That is impossible without an escape rule, length prefix, or different serialization.

---

# 3. Current architecture that should be preserved

The rewrite already introduced good structural improvements and they should **not** be undone:

1. `tool_call_grammar.{h,cpp}`
   - authoritative wire header grammar;
   - `NoMatch / NeedMore / Complete / Invalid`;
   - marker classification;
   - exact wrapper literals;
   - name validation.

2. `tool_call_stream.{h,cpp}`
   - shared one-shot/streaming parser;
   - objective `ToolCallParseProgress`;
   - separate recovery policy;
   - parser-owned AST.

3. `tool_call_parser.cpp`
   - contract/schema normalization after structural parsing.

4. `tool_call_grammar_state.{h,cpp}`
   - CPU grammar-constraint state using the parser grammar as source of truth.

The fix must remain a **minimal, coherent extension of this design**. Do not reintroduce multiple independent marker/header grammars.

---

# 4. Finding F1 — HIGH
## Literal `<parameter=...>` in payload incorrectly changes nesting depth

### Status

**Verified from source + logically derived.**

Current `ParameterValue` handling initializes:

```cpp
s.depth = 1;
```

and increments the depth whenever a complete parameter opener is seen inside the payload:

```cpp
if (text[scan] == '<') {
    ToolOpenTag nested = {};
    if (parse_tool_parameter_open(text.substr(scan), nested) ==
        ToolHeaderStatus::Complete) {
        ++s.depth;
        scan = scan + nested.consumed;
        continue;
    }
}
```

The outer closer is only accepted once `depth` reaches zero.

### Deterministic counterexample

```text
<tool_call>
<function=write>
<parameter=content>
const x = "<parameter=fake>";
</parameter>
</function>
</tool_call>
```

Current behavior:

1. outer `content` opens → depth 1;
2. literal `<parameter=fake>` in source text → depth 2;
3. real outer `</parameter>` → depth 1;
4. no second `</parameter>` exists;
5. parser reaches EOF with an open value;
6. strict rejects;
7. tolerant also cannot safely commit an open value.

This is exactly the class of failure that matters for `write`, `edit`, shell commands, generated code, XML, Markdown, templates, and source files.

### Root cause

The parser treats markup-like payload as recursively nested tool structure even though the Qwen wire format does not define recursive parameter elements inside a parameter value.

### Required implementation

**Remove parameter-opener-based recursive depth tracking from `ParameterValue`.**

Recommended design:

- A parameter value is an opaque byte range.
- Search only for candidate occurrences of the matching outer close:
  - `</parameter>` for `ToolTagKind::Parameter`;
  - `</param>` for `ToolTagKind::Param`.
- For each candidate closer, classify the following bytes to decide whether this candidate is a plausible structural boundary.
- If the continuation is not compatible with the surrounding function grammar, keep the candidate closer as payload and continue searching.
- Do **not** increment nesting depth when seeing `<parameter=...>` or `<param=...>` inside the value.

This simpler rule already preserves existing embedded-markup cases such as:

```text
before<parameter=inner>value</parameter>after
```

because the inner `</parameter>` is followed by ordinary payload text (`after`) and therefore is not a structural outer boundary. The later real outer closer is followed by a legal function continuation and can be accepted.

### Files

Primary:

- `src/models/qwen3_5/frontend/tool_call_stream.cpp`
- `src/models/qwen3_5/frontend/tool_call_stream.h` if state fields change

Tests:

- `tests/test_tool_call_parser.cpp`
- optionally `tests/test_tool_call_grammar_state.cpp` for prefix/constraint equivalence

### Required regression tests

Add explicit tests for at least:

```text
A <parameter=fake> B
```

```text
A <param=fake> B
```

```text
const s = "<parameter=x>";
```

```text
<parameter=content>
<parameter=inner>
literal without inner closer
</parameter>
```

where the last `</parameter>` is the actual outer close.

Also test real coding payloads:

- C/C++ source string containing `<parameter=...>`;
- TypeScript/JS string containing `<parameter=...>`;
- XML/HTML snippets;
- shell here-doc content;
- JSON string with tool-like markup;
- multi-line patch content.

### Acceptance criteria

- Exact payload returned unchanged.
- One-shot and streaming results identical.
- Every byte-split of the same output yields the same parsed tool call.
- Strict and tolerant both accept a fully well-formed outer call.
- No fake nested depth exists in the implementation after the fix.

---

# 5. Finding F2 — HIGH
## Tolerant recovery executes a function whose closing tag was never observed

### Status

**Verified from source and existing tests.**

Current recovery computes:

```cpp
const bool open_commit =
    !progress.open_value_open &&
    !progress.open_call.parameters.empty();
```

and may return:

```cpp
CommitCallsAndOpenCall
```

This explicitly allows an `open_call` to become executable without a matching function/invoke close.

The existing test suite intentionally accepts a missing function close in tolerant mode.

### Why this is unsafe

A closed parameter is not proof that the model finished expressing the call.

The missing tail may have contained:

- another required parameter;
- a replacement value;
- more command text;
- another payload field;
- a different intended structure.

Executing before `</function>` / `</invoke>` has been observed converts truncation into action.

### Required implementation

**Remove execution of `open_call`.**

Recommended structural change:

1. Delete or retire:
   - `ToolCallRecoveryDecision::CommitCallsAndOpenCall`.
2. Recovery may commit **only** `progress.calls`.
3. A call must be moved into `progress.calls` immediately when its matching function/invoke close has been consumed.
4. For a `<tool_call>` wrapper:
   - consume `</function>` / `</invoke>`;
   - mark/append the call as structurally complete;
   - then enter `ExpectWrapperClose`.
5. If the output ends before `</tool_call>`:
   - strict mode rejects;
   - tolerant mode may retain the already function-closed call with `TruncatedTail`.
6. If the output ends before `</function>` / `</invoke>`:
   - never execute that open call;
   - return text/fallback for that open call;
   - already completed *earlier* calls may still be retained if the policy explicitly permits it.

### Important state-machine consequence

Currently a call inside `<tool_call>` is completed only after the wrapper close. Change that.

Old conceptual sequence:

```text
function close
→ ExpectWrapperClose
→ wrapper close
→ complete_call()
```

Required sequence:

```text
function close
→ complete_call()
→ ExpectWrapperClose
→ wrapper close
→ finish wrapper only
```

Prevent a second `complete_call()` at wrapper close.

### Tests that must change

Any current test asserting:

> missing `</function>` is recovered as executable

must be changed to assert **non-execution**.

### Required test matrix

For each of `StopToken`, `OutputLimit`, `ContextCapacity`:

| Cut point | Strict | Tolerant |
|---|---|---|
| inside parameter value | reject | reject current open call |
| exactly after `</parameter>` | reject | reject current open call |
| inside `</function>` | reject | reject current open call |
| exactly after full `</function>` but before `</tool_call>` | reject | may commit call |
| inside `</tool_call>` after full function close | reject | may commit call |
| complete wrapper | commit | commit |

Also test bare function/invoke compatibility forms separately.

### Acceptance criteria

Search the final code for all paths that produce a `GeneratedToolCall`. For each path, prove that a matching function/invoke close was consumed first.

No `open_call` execution path may remain.

---

# 6. Finding F3 — HIGH
## EOF `</parameter>` ambiguity can produce a truncated payload that becomes executable

### Status

**Logically derived from F2 plus current close-continuation behavior.**

Current close continuation treats end-of-input after a candidate parameter closer as a legal boundary:

```cpp
if (at >= text.size()) {
    return true;
}
```

Example:

```text
<tool_call>
<function=bash>
<parameter=command>
echo 'literal </parameter>
```

The final `</parameter>` may be literal command text rather than a structural delimiter.

### Required fix

F3 is resolved primarily by enforcing F2:

> A parameter boundary alone is never enough to execute the function.

The parser may provisionally classify the last `</parameter>` as a parameter boundary at EOF, but if the function itself never closes, the call remains non-executable.

### Additional requirement

Document this as an inherent wire-format ambiguity.

Do not invent quote parsing for arbitrary shell/code payload. The payload is not defined as XML or shell syntax and the parser cannot safely infer every host language.

### Tests

Add the exact EOF fixture above for:

- `bash.command`
- `write.content`
- `edit.new_string`

Assertions:

- strict: no tool call;
- tolerant: no current tool call;
- content preserved;
- diagnostic is truncation/malformed according to the finalized recovery contract;
- never execute a shortened argument.

---

# 7. Finding F4 — MEDIUM
## Partial next structural token can incorrectly reopen the previous parameter

### Status

**Logically derived from current `is_close_continuation()` behavior.**

Example:

```text
<tool_call>
<function=read>
<parameter=path>
foo
</parameter>
<paramet
```

The first parameter is complete. The next header is merely truncated.

A continuation classifier that only accepts a **complete** next parameter opener can misclassify the preceding real close as payload and run to EOF with `open_value_open=true`.

### Required implementation

Introduce an explicit tri-state continuation result, for example:

```cpp
enum class CloseContinuation {
    Invalid,   // candidate closer is payload
    Complete,  // following bytes prove a structural boundary
    NeedMore,  // following bytes are a legal prefix of the next structural token
};
```

Names may differ; semantics must not.

The boundary classifier must understand legal prefixes of:

- parameter opener;
- current function/invoke close;
- required wrapper close after function close;
- any other supported next-token form in the current state.

When a candidate `</parameter>` is followed by a **NeedMore** continuation:

1. keep the preceding parameter closed;
2. advance to the next parser state;
3. report the following token as truncated/`EndOfInput`;
4. do not reinterpret the already consumed parameter close as payload.

### Additional parser changes

Review `FunctionBody`, `Top`, and `ExpectWrapperClose` for conditions currently guarded by:

```cpp
if (policy.prefix && ...)
```

A syntactically legal prefix at the actual end of generation is an objective truncation fact, not necessarily something that should become a definitive malformed break only because `policy.prefix` is false.

Do not blindly change all branches. For each branch distinguish:

- `NeedMore`: can become legal by appending bytes → `EndOfInput`;
- `Invalid` / impossible transition → `Definitive`.

Strict recovery can still map `EndOfInput` to `MalformedStructure` externally.

### Tests

For every supported delimiter/opening token, cut at every byte:

- `<parameter=...>`
- `<param=...>`
- `</parameter>`
- `</param>`
- `</function>`
- `</invoke>`
- `</tool_call>`
- `</function_calls>`

Verify:

- no earlier closed value is reopened;
- tolerant never executes an open function;
- complete earlier calls remain recoverable according to policy.

---

# 8. Finding F5 — MEDIUM
## Wrapper state is represented by booleans and permits unbalanced nested wrappers

### Status

**Verified from source + logically derived.**

Current state uses:

```cpp
bool wrapper_close_expected = false;
bool inside_function_calls = false;
```

A second `<tool_call>` can set the same boolean again; one close can then clear it.

Counterexample to pin with a test:

```text
<tool_call>
<tool_call>
<function=read>
<parameter=path>x</parameter>
</function>
</tool_call>
```

Strict mode must not treat this as a balanced valid region.

### Required implementation

Replace the ambiguous pair of booleans with an explicit wrapper state, for example:

```cpp
enum class WrapperKind : std::uint8_t {
    None,
    ToolCall,
    FunctionCalls,
};
```

Rules:

- `WrapperKind::None`
  - may enter a supported top-level wrapper;
  - may parse an explicitly supported bare compatibility function if that mode is enabled.
- `WrapperKind::ToolCall`
  - must contain exactly the supported function structure;
  - may not open another wrapper before closing;
  - after the function closes, only the matching wrapper close/prefix is legal.
- `WrapperKind::FunctionCalls`
  - contains its supported sequence of calls;
  - nested wrapper opens are rejected unless the formal grammar explicitly supports them;
  - empty wrapper behavior remains whatever the documented contract requires (currently unrecoverable).

Do not use multiple booleans that can represent impossible combinations.

### Tests

Add:

- nested `<tool_call><tool_call>...`;
- nested `<function_calls><function_calls>...`;
- cross-nesting `<tool_call><function_calls>...`;
- cross-nesting `<function_calls><tool_call>...`;
- stray wrapper closer;
- too many wrapper closers;
- missing outer close;
- multiple sequential valid wrappers at top level.

Test strict and tolerant separately.

### Acceptance criteria

Every wrapper open must have a single explicit state transition and a matching close transition.

No sequence with more opens than closes may be reported `Complete`.

---

# 9. Finding F6 — MEDIUM
## Mixed `parameter`/`param` payload nesting is inconsistent

### Status

**Verified from the combination of current depth increment and family-specific close search.**

Current code accepts both opener families, but nested depth increments on either family while the closer search uses only the outer family.

Example:

```text
<parameter=content>
A <param=x>B</param> C
</parameter>
```

The inner `<param=x>` can increment depth while `</param>` is not the close currently being searched.

### Required fix

F6 should disappear naturally with F1:

- remove recursive depth tracking;
- treat payload as opaque;
- only search for the current outer parameter family's candidate closing literal;
- other family tags inside payload remain ordinary bytes unless they occur after an accepted boundary as the next actual structure.

### Tests

Add both directions:

```text
<parameter=content>A <param=x>B</param> C</parameter>
```

and:

```text
<param=content>A <parameter=x>B</parameter> C</param>
```

Expected: exact payload preservation.

---

# 10. Finding F7 — LOW/MEDIUM
## Initial marker set and retry marker set are asymmetric

### Status

**Verified from source.**

Initial marker recognition supports more forms than the retry loop. The retry loop currently searches only for a later literal `<tool_call>`.

This is safer than blindly retrying from every `<function>` inside malformed content, but it means:

> a marker form accepted as the first entry is not necessarily eligible as a later clean recovery entry.

### Required implementation decision

Do **not** fix this by blindly replacing the retry search with `find_tool_marker()`.

That can cause a function opener nested inside the first malformed/quoted region to be reinterpreted as a new executable top-level call.

Implement one of these two safe designs:

### Preferred design: explicit entry-marker policy

Separate:

- wire **tag grammar** (function/parameter headers);
- top-level **entry marker policy**.

For Qwen native mode, canonical entry should be wrapper-based:

```text
<tool_call>
```

plus any wrapper form intentionally supported by the active template.

Bare `<function=...>` / `<invoke=...>` compatibility should be an explicit compatibility policy, not an accidental consequence of the same marker function.

The same entry-marker policy must be used for:

- initial streaming latch;
- one-shot marker discovery;
- retry/recovery discovery;
- grammar-constraint lazy trigger.

### Acceptable alternative

A parser-aware recovery scanner may support the broader set if it can prove that a candidate is outside the failed region rather than nested inside it.

If this alternative is selected, tests must demonstrate that malformed/quoted nested `<function=...>` cannot become a new executable call.

### Tests

At minimum:

1. malformed quoted `<tool_call>` followed by real `<tool_call>` → recover real call;
2. malformed wrapper containing nested `<function=fake>` → never recover fake nested call;
3. compatibility bare function before real wrapper;
4. failed region followed by supported compatibility marker;
5. multiple false markers before terminal real call;
6. trailing prose after otherwise valid later candidate → do not partially commit if policy says terminal structured region is required.

### Documentation requirement

Document exactly which byte sequences are valid **top-level entry markers** in each mode.

---

# 11. Finding F8 — LOW
## A second `<` that breaks a marker candidate is consumed instead of being retried as a fresh marker start

### Status

**Verified and currently intentional.**

The current stream machine appends the breaking byte to `marker_prefix_`; when classification becomes `NotMarker`, it publishes the whole candidate and consumes the breaking byte.

Thus a stream such as:

```text
<function<tool_call>
```

does not retry the second `<` as a potential new marker.

The grammar-constraint state was recently modified to mirror this exact behavior.

### Required implementation

Make the scanner robust while preserving one source of truth.

Recommended behavior:

If the byte that causes `NotMarker` is `<`:

1. publish only the failed candidate bytes **before** that final `<`;
2. retain the final `<` as the beginning of a fresh marker candidate;
3. continue scanning from the next byte.

If the breaking byte is not `<`, publish the entire failed candidate normally.

Do **not** duplicate this logic independently in:

- `ToolCallStreamParser`;
- `ToolCallGrammarConstraint`.

Factor a shared marker-candidate transition helper/state component, or otherwise guarantee both consumers call the same transition logic.

### Important regression

A `<` inside a quoted header value must **not** restart the marker candidate if the header grammar still classifies the overall candidate as `NeedMore`.

Example already relevant:

```text
<function name="a<b">
```

The grammar decides whether the `<` is legal inside the quote. The marker scanner must not special-case it before grammar classification.

### Tests

- `<function<tool_call>...` → second `<tool_call>` can be recognized;
- `<function name="a<b">` → no incorrect restart at quoted `<`;
- every-byte split variants;
- grammar constraint and stream parser latch at the same byte.

---

# 12. Finding F9 — LOW
## Parameter nesting counter is `uint8_t`

### Status

**Verified.**

Current state contains:

```cpp
std::uint8_t depth = 1;
```

This can wrap after enough artificial nested parameter openers.

### Required fix

The preferred F1 fix removes recursive parameter nesting and therefore removes `depth` completely.

If any depth-like state remains for another purpose:

- use `std::size_t` or a bounded larger integer;
- add an explicit maximum if recursive structures are truly supported;
- reject overflow before increment.

### Acceptance criteria

No unchecked 8-bit nesting counter remains.

---

# 13. Finding F10 — FUNCTIONAL GAP
## `--constrained-tool-decoding tool-calls-only` is currently a silent no-op in sampling

### Status

**Verified from branch documentation and source plumbing.**

The flag reaches `EngineOptions`, but the sampling/logit path does not apply the grammar constraint.

### Constraint for this task

**Do not implement GPU sampling integration.**
**Do not run GPU tests.**

A partially implemented grammar mask without runtime GPU validation would increase risk.

### Required fix for this branch

Eliminate the silent no-op.

Preferred minimal change:

- `off` remains accepted and unchanged;
- selecting `tool-calls-only` fails at startup/request initialization with a clear error such as:

```text
--constrained-tool-decoding=tool-calls-only is not implemented in this build; use off
```

Alternative:

- remove/hide the non-off user-facing option until integration is implemented.

Whichever path is selected:

- CLI and server behavior must match;
- docs must match;
- tests must pin the behavior;
- no user should be able to enable a mode that silently does nothing.

### CPU tests

- CLI option `off` accepted;
- server option `off` accepted;
- non-off mode produces the explicit expected error;
- default remains `off`;
- normal sampling configuration is unaffected when off.

---

# 14. Parameter-boundary redesign

This is the central implementation piece and should be completed before modifying recovery.

## 14.1 Remove recursive payload parsing

Delete state such as:

```cpp
depth
next_close
```

if it exists only for recursive parameter nesting.

Maintain:

```cpp
param_family
param_name
value_begin
```

## 14.2 Candidate-close scan

Pseudo-design:

```cpp
while (true) {
    candidate = find(required_close, scan_from);
    if (candidate == npos) {
        return EndOfInput(open_value = true);
    }

    continuation = classify_close_continuation(
        text,
        candidate + required_close.size(),
        current_function_family,
        current_wrapper_state
    );

    if (continuation == Invalid) {
        scan_from = candidate + required_close.size();
        continue; // candidate close is payload
    }

    // Complete OR NeedMore continuation:
    // the current parameter boundary is retained.
    commit_parameter(text.substr(value_begin, candidate - value_begin));
    pos = candidate + required_close.size();

    if (continuation == NeedMore) {
        // Continue into the next parser state so that the incomplete
        // next structural token becomes EndOfInput, not part of payload.
    }

    mode = FunctionBody;
    continue;
}
```

The actual implementation may differ, but must preserve the invariant.

## 14.3 Continuation classifier

The classifier must be context-aware.

A candidate outer parameter close can be a structural boundary when followed, after format whitespace, by one of:

1. another complete parameter opener;
2. a legal prefix (`NeedMore`) of another parameter opener at end of input;
3. the matching current function/invoke close;
4. a legal prefix of the matching function/invoke close at end of input;
5. end of input — parameter may be provisionally closed, but the function remains open and therefore non-executable;
6. any additional continuation formally required by supported compatibility grammar.

When a full function close follows inside a `<tool_call>` wrapper:

- examine what follows the function close;
- legal:
  - complete `</tool_call>`;
  - legal prefix of `</tool_call>` at EOF;
  - EOF (wrapper truncated after a fully closed function; tolerant recovery may later retain it);
- illegal:
  - arbitrary payload/prose that proves the apparent function close was actually part of the parameter payload.

For bare functions / `function_calls`, use the correct surrounding-state rules rather than the `<tool_call>` rule.

### Do not parse host-language quoting

Do not attempt shell quote parsing, C++ string parsing, JSON parsing, etc. The same parameter can carry any language. Structural decisions must use only the tool wire grammar.

---

# 15. Recovery redesign

## 15.1 Objective progress

Keep the separation:

```text
parse structure
→ produce objective progress
→ apply recovery policy
```

Do not mix tolerant decisions into the structural scanner more than necessary.

## 15.2 Complete-call definition

A `ParsedFunctionCall` belongs in `progress.calls` **iff its matching function/invoke closer has been consumed**.

An open function belongs only in `progress.open_call`.

## 15.3 Simplify recovery decisions

Preferred enum:

```cpp
enum class ToolCallRecoveryDecision {
    Reject,
    CommitCalls,
};
```

If a third state remains for another legitimate reason, it must not mean "execute an open function".

## 15.4 Tolerant rules

Tolerant may commit:

- clean complete region;
- complete function-closed call(s) before:
  - missing wrapper close;
  - truncated wrapper close;
  - later malformed call/trailing region, if existing documented behavior permits it.

Tolerant may not commit:

- open value;
- open parameter header;
- open function/invoke;
- name-only call;
- ambiguous current call whose function closer was not proven.

## 15.5 Finish reason

`FinishReason` may affect diagnostics, but must not make unsafe bytes executable.

OutputLimit/ContextCapacity do not grant permission to infer missing structure.

---

# 16. Wrapper-state redesign

Recommended `RegionState` fields:

```cpp
enum class WrapperKind : std::uint8_t {
    None,
    ToolCall,
    FunctionCalls,
};

WrapperKind wrapper = WrapperKind::None;
bool function_calls_had_call = false;
```

Derive behavior from `wrapper`, rather than carrying contradictory booleans.

Review every mode transition and write a state-transition table in code comments or tests.

Suggested high-level transition table:

| Current state | Input | Next |
|---|---|---|
| Top + None | `<tool_call>` | ExpectFunction + ToolCall |
| Top + None | `<function_calls>` | ExpectFunction + FunctionCalls |
| ToolCall/ExpectFunction | function opener | FunctionHeader |
| FunctionCalls/ExpectFunction | function opener | FunctionHeader |
| FunctionBody | matching function close | complete call; Wrapper-dependent next state |
| ToolCall after function close | `</tool_call>` | Top + None |
| FunctionCalls/Top | next function | FunctionHeader |
| FunctionCalls/Top | `</function_calls>` with >=1 call | Top + None |
| any active wrapper | another wrapper open where nesting unsupported | Definitive invalid |

Pin this with tests; do not rely on comments alone.

---

# 17. Marker scanner unification

Create or refactor a small reusable marker-candidate state machine.

Required inputs:

- current candidate bytes;
- next decoded byte(s).

Required outputs/actions:

- ordinary content to publish;
- candidate remains pending;
- marker completes/latches;
- failed candidate, possibly retaining a final `<` as a fresh candidate.

Both:

- `ToolCallStreamParser`
- `ToolCallGrammarConstraint`

must consume the same marker transition rules.

Add direct unit tests for this component if extracted.

---

# 18. Test implementation plan

## 18.1 Existing test targets to preserve

At reviewed HEAD the relevant CPU targets include:

```text
ninfer_qwen3_5_frontend_test
ninfer_tool_call_parser_test
ninfer_tool_call_grammar_test
ninfer_tool_call_grammar_state_test
```

Do not remove or weaken existing assertions merely to make the suite green.

If an existing expectation conflicts with a new safety invariant (notably tolerant missing-function-close execution), change the expectation and document why in the progress file.

## 18.2 Mandatory targeted tests by finding

Create named test functions that make the finding obvious.

Suggested names:

```text
test_literal_parameter_opener_in_payload_does_not_nest
test_literal_param_opener_in_payload_does_not_nest
test_mixed_parameter_families_inside_payload_are_opaque

test_tolerant_never_commits_open_function
test_tolerant_commits_function_closed_missing_wrapper
test_eof_parameter_closer_never_executes_open_function

test_partial_next_parameter_does_not_reopen_previous_value
test_all_delimiter_byte_cuts_preserve_previous_boundary

test_nested_tool_call_wrapper_rejected
test_cross_nested_wrappers_rejected

test_marker_breaking_angle_restarts_candidate
test_marker_stream_and_constraint_latch_equivalence

test_constrained_tool_decoding_nonoff_fails_fast
```

Names may differ, but each defect needs an isolated regression.

## 18.3 Every-byte cut matrix

Build a canonical call:

```text
<tool_call>
<function=write>
<parameter=path>
a.txt
</parameter>
<parameter=content>
hello
</parameter>
</function>
</tool_call>
```

For **every cut index from 0 through N**:

1. parse the cut output one-shot;
2. feed the same cut output through streaming using:
   - one byte per chunk;
   - fixed chunk sizes 2, 3, 5, 7;
   - all two-way split positions where practical;
3. compare:
   - tool call count;
   - function name;
   - arguments;
   - content;
   - fallback reason;
   - marker diagnostics.

Safety property:

> No cut before the complete matching function close may execute the current function in tolerant mode.

Recovery property:

> A cut after the matching function close but before completion of the outer wrapper may be retained in tolerant mode if all arguments are closed.

## 18.4 Payload adversarial corpus

Use a deterministic list, not random-only fuzzing:

```text
<
>
</parameter>
<parameter=x>
<param=x>
</param>
</function>
</invoke>
<tool_call>
</tool_call>
<function_calls>
</function_calls>
<function=fake>
<invoke=fake>
<function name="fake">
<parameter name="x">
```

Combine with:

- prefix/suffix ordinary text;
- newlines;
- CRLF;
- tabs;
- repeated sequences;
- malformed prefixes;
- tags inside quotes as raw payload text;
- tags at payload start/end.

For each corpus entry embedded into a declared string argument:

- if it is not structurally indistinguishable from a complete real suffix, the exact value must round-trip;
- if it *is* fundamentally ambiguous, the parser must prefer non-execution over unsafe inference.

## 18.5 Marker property tests

For a corpus of prose and marker prefixes:

- one-shot marker discovery;
- stream latch;
- grammar-constraint lazy trigger

must agree on the trigger byte.

Explicitly include:

```text
<function<tool_call>
<function name="a<b">
abc <tool_
abc <tool_call>
abc <<tool_call>
```

## 18.6 Wrapper balance tests

Generate short wrapper sequences up to a small bounded depth (CPU only) and assert:

- unsupported nesting never completes;
- sequential balanced wrappers may complete;
- an extra close never completes;
- a missing close never completes strict;
- tolerant retention never invents a function close.

This can be table-driven rather than requiring a fuzzing framework.

---

# 19. CPU-only build/test commands

Use the repository's existing build directory/configuration if already configured.

### Build targeted tests

Example:

```powershell
cmake --build build-new-parser --config Release --target `
  ninfer_tool_call_parser_test `
  ninfer_tool_call_grammar_test `
  ninfer_tool_call_grammar_state_test `
  ninfer_qwen3_5_frontend_test `
  --parallel 32
```

If the generator does not accept several targets in that form, build them separately or build the normal test target.

### Targeted parser gate

```powershell
ctest --test-dir build-new-parser -C Release `
  -R "ninfer_(tool_call_parser|tool_call_grammar|tool_call_grammar_state|qwen3_5_frontend)_test" `
  --output-on-failure
```

### Broader CPU gate

Run the full CPU suite while excluding `_real` tests:

```powershell
ctest --test-dir build-new-parser -C Release `
  -E "_real" `
  --output-on-failure `
  --parallel 8
```

If any additional test attempts GPU runtime execution despite not matching `_real`:

- do not force it;
- mark it skipped for this task;
- record exact test name and reason in the progress MD.

### Prohibited

Do not run:

- model inference;
- NInfer `_real` tests;
- GPU correctness tests;
- CUDA memcheck/racecheck requiring runtime model execution;
- performance benchmarks;
- agentic model benchmarks;
- server inference smoke tests requiring the local model/GPU.

---

# 20. Implementation phases

## Phase 0 — Freeze and reproduce

1. Record current HEAD.
2. Build targeted CPU parser tests.
3. Run targeted baseline.
4. Add **failing tests only** for F1–F9 before changing implementation where practical.
5. For F10 add failing CLI/server behavior tests for the current silent no-op.
6. Record each expected failure in `NInfer_new_parser_design_bugfix_progress.md`.

Do not "fix" tests by weakening expected safety behavior.

## Phase 1 — Parameter value boundary redesign

Fix together:

- F1;
- F3 structural side;
- F4;
- F6;
- F9.

Tasks:

1. remove recursive parameter depth;
2. implement context-aware close-continuation tri-state;
3. preserve closed previous values across partial next tokens;
4. add adversarial payload matrix;
5. run parser + grammar-state tests;
6. update progress MD.

## Phase 2 — Recovery integrity

Fix F2 and complete F3.

Tasks:

1. complete call at function close;
2. never execute `open_call`;
3. remove `CommitCallsAndOpenCall` if no longer needed;
4. preserve missing-wrapper tolerant recovery after function close;
5. change old tests that intentionally executed missing-function-close calls;
6. run every-byte truncation matrix;
7. update progress MD.

## Phase 3 — Wrapper state

Fix F5.

Tasks:

1. replace ambiguous booleans with explicit wrapper kind/state;
2. reject unsupported nesting;
3. test all open/close combinations;
4. ensure sequential valid calls/wrappers still work;
5. update progress MD.

## Phase 4 — Marker progression and recovery markers

Fix F7/F8.

Tasks:

1. centralize marker candidate transition;
2. reprocess a breaking `<` as fresh candidate;
3. keep quoted `<` behavior grammar-driven;
4. define explicit top-level entry-marker policy;
5. make initial/retry/constraint semantics consistent;
6. add false-marker and recovery tests;
7. update progress MD.

## Phase 5 — User-facing constrained-decoding flag

Fix F10 without GPU runtime work.

Tasks:

1. retain `off`;
2. make non-off fail fast or remove it from user-facing surfaces;
3. align CLI, server, docs and tests;
4. do not alter GPU sampling path;
5. update progress MD.

## Phase 6 — Documentation and diagnostics

Update at minimum:

- `docs/tool_call_parser.md`
- `docs/new_parser_progress.md` or the new bugfix progress file
- CLI/server help for constrained decoding

Document:

- exact safe tolerant-recovery boundary;
- inherent delimiter ambiguity;
- top-level entry marker set;
- non-execution requirement for open functions;
- constrained-decoding availability status.

## Phase 7 — Full CPU verification

1. targeted parser gate;
2. frontend integration gate;
3. CLI/server option tests;
4. full CPU suite excluding `_real`;
5. no GPU execution.

---

# 21. Mandatory multi-round code review

A single "tests pass" result is not enough.

Perform the following review rounds after implementation.

## Review Round A — Structural state-machine review

Read the entire current versions of:

- `tool_call_grammar.h/.cpp`
- `tool_call_stream.h/.cpp`
- `tool_call_parser.h/.cpp`
- `tool_call_grammar_state.h/.cpp`
- `output_session.h/.cpp`

Do not review only the diff.

Checklist:

- Can any open value execute?
- Can any open function execute?
- Can wrapper count/state become inconsistent?
- Can marker scanners disagree?
- Can a truncation convert payload into structure?
- Can a malformed compatibility form bypass native safety?
- Are all state transitions total and deterministic?
- Are `NeedMore` and `Invalid` used consistently?

Record every finding in the progress MD.

If any medium/high finding remains, fix it and repeat Round A.

## Review Round B — Counterexample/adversarial review

Actively try to break the implementation.

Required categories:

- tool-like source-code strings;
- unbalanced literal tags inside payload;
- duplicate tags;
- mixed `parameter`/`param`;
- all delimiters at EOF;
- every byte cut around closers/openers;
- false markers in prose;
- multiple false markers before real call;
- wrapper nesting/cross-nesting;
- CRLF and whitespace variations;
- long names / name limits;
- duplicate parameters;
- undeclared tools;
- multiple calls.

For each newly discovered failure:

1. add regression test first;
2. fix;
3. rerun targeted suite;
4. update progress;
5. restart Round B.

## Review Round C — Regression/compatibility review

Verify that fixes did not regress existing successful behavior:

- normal Qwen native tool call;
- multiple calls;
- attribute-form headers;
- quoted `>` in header;
- duplicate parameter last-value-wins;
- schema normalization;
- boolean/integer/object/array normalization;
- quoted malformed marker before real call;
- streaming byte-split equivalence;
- Claude/agent compatibility forms already covered by tests, unless intentionally changed and documented.

## Review Round D — CPU full-suite review

Run the full CPU suite with GPU/`_real` excluded.

Any unrelated failure introduced by the branch must be investigated rather than ignored.

## Review Round E — Final whole-pipeline review

Trace one normal and one adversarial tool call end-to-end:

```text
generated tokens
→ OutputSession content
→ ToolCallOutputDecoder
→ marker latch
→ region parser
→ recovery policy
→ schema normalization
→ GeneratedToolCall
→ OpenAI/Responses serialization
```

Confirm that the parser never relies on downstream serialization to repair unsafe structure.

---

# 22. Progress file requirements

Maintain:

```text
NInfer_new_parser_design_bugfix_progress.md
```

from the provided template.

Update it **after every implementation phase and every review round**.

For every finding include:

- status:
  - Not started
  - Reproduced
  - Fix in progress
  - Fixed / targeted tests green
  - Review verified
  - Final CPU verified
- root cause;
- exact files/functions changed;
- tests added;
- test command/result;
- regressions found;
- follow-up changes;
- residual uncertainty.

Never mark an item "verified" merely because code compiled.

---

# 23. Definition of done

The task is complete only when all are true:

- [ ] F1 fixed and literal parameter openers in payload round-trip.
- [ ] F2 fixed and no open function can execute.
- [ ] F3 fixed by non-execution of ambiguous EOF-open functions.
- [ ] F4 fixed and partial next tokens do not reopen prior values.
- [ ] F5 fixed and wrapper nesting/balance is explicit.
- [ ] F6 fixed and mixed-family payload markup stays opaque.
- [ ] F7 resolved with documented consistent entry/retry policy.
- [ ] F8 fixed with shared marker progression and breaking-`<` retry.
- [ ] F9 removed or made overflow-safe.
- [ ] F10 no longer exposes a silent no-op mode.
- [ ] Every finding has at least one direct regression test.
- [ ] Every-byte cut safety test is green.
- [ ] One-shot vs streaming equivalence is green.
- [ ] Stream parser vs grammar-constraint marker trigger equivalence is green.
- [ ] Existing parser/grammar/frontend tests remain green except deliberately corrected unsafe expectations.
- [ ] Full CPU CTest suite is green with GPU/`_real` excluded/skipped.
- [ ] No GPU tests were run.
- [ ] Review Rounds A–E completed with no unresolved HIGH/MEDIUM correctness finding.
- [ ] Documentation accurately describes remaining inherent wire ambiguity.
- [ ] Progress MD contains evidence for every completed item.

---

# 24. Source map for the reviewed baseline

Pinned branch:

```text
https://github.com/Hundsbuah/ninfer/tree/9927ff3d92f66748d1cbfc4f9d0f9b9a91db2d9f
```

Primary files:

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

tests/test_tool_call_parser.cpp
tests/test_tool_call_grammar.cpp
tests/test_tool_call_grammar_state.cpp
tests/models/qwen3_5/test_frontend.cpp

docs/tool_call_parser.md
docs/new_parser_progress.md
docs/new_parser_phase4_design.md

src/serve/serve_options.cpp
src/serve/serve_options.h
apps/cli/options.cpp
apps/cli/options.h
include/ninfer/types.h
```

Relevant reviewed commits:

```text
ecf0244aee52438e3d6b3ec7f0ea455f208638be
  consolidate Qwen tool-call wire syntax into one grammar

271fd01a0044f8e301baf91a5e39e4b46dd12420
  policy-free incremental tool-call parser with explicit recovery policy

85961f4ff915c289b141fd5eeabd1d4fc7125a28
  grammar-constrained tool-call decoding state core / reserved flag

d0c16e3fb9ea42fe666a47a660476d5733ea6aa9
  code-review fixes to parser/constraint/decoder
```

---

# 25. Final implementation principle

The priority order for this parser must be:

```text
payload integrity
> proof of complete executable function
> structural correctness
> recovery availability
```

Tolerant mode is permitted to recover **proven complete calls**. It is not permitted to turn incomplete syntax into an action.

If a future change makes the parser "recover more calls" while weakening any of the invariants in this specification, treat that as a regression.
