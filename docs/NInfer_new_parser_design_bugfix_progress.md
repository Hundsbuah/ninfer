# NInfer `new_parser_design` — Bugfix Progress

> Working log for the implementation defined in
> `NInfer_new_parser_design_bugfix_implementation.md`.

## Rules for this progress file

- Update after every implementation phase.
- Update after every code-review round.
- Record exact test commands and results.
- Never mark a finding verified from compilation alone.
- Do not run GPU runtime tests, `_real` tests, model inference or GPU benchmarks.
- If a test needs GPU runtime, leave/mark it skipped and record that fact.
- Preserve evidence for regressions found during review, even after they are fixed.

---

## Baseline

- Repository: `Hundsbuah/ninfer` (local clone `E:\KI\ninfer-custom`)
- Working branch: `new_parser_design`
- Baseline reviewed branch: `new_parser_design`
- Baseline reviewed SHA: `9927ff3d92f66748d1cbfc4f9d0f9b9a91db2d9f`
- Implementation start SHA: `9927ff3d92f66748d1cbfc4f9d0f9b9a91db2d9f` (HEAD at start; clean tree except pre-existing local
  CMake/vcpkg adjustments and the two bugfix docs)
- Date started: 2026-09-29
- Build directory: `build-new-parser` (fresh configure; `build-windows` is reserved for the user's
  running local AI build and was not used)
- Build generator/configuration: Visual Studio 18 2026, x64, CUDA toolset
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4` (v13.4.92),
  `CMAKE_CUDA_ARCHITECTURES=120a`, `CMAKE_PREFIX_PATH=F:\GIT\vcpkg\installed\x64-windows`
  (env `VCPKG_ROOT=F:/GIT/vcpkg`, `VCPKG_TARGET_TRIPLET=x64-windows`), Release, `BUILD_TESTING=ON`,
  `NINFER_BUILD_APPS=ON`
- CPU test environment: Windows 10.0.26200, Python 3.14 (not required), CPU-only CTest runs
- GPU runtime tests: **PROHIBITED for this task** — the full-suite run hides the GPU with
  `CUDA_VISIBLE_DEVICES=99` so every GPU test takes its own skip path (exit 77, "no usable CUDA
  device")

### Baseline build

Command:

```text
cmake --build build-new-parser --config Release --parallel 8
```

Result:

```text
green: all product targets (ninfer-serve, ninfer-cli option code via the CLI options test,
ninfer_model_runtime, ninfer_engine) and all registered test targets.
```

### Baseline targeted CPU tests

Command:

```text
build-new-parser\tests\Release\ninfer_tool_call_grammar_test.exe
build-new-parser\tests\Release\ninfer_tool_call_grammar_state_test.exe
build-new-parser\tests\Release\ninfer_tool_call_parser_test.exe
```

Result:

```text
At baseline (before the Phase 1-4 redesign) the added/updated regression tests reproduced the
findings: the parser suite reported the F1-F9 failures (payload nesting change, tolerant open
function commit, EOF closer execution, partial-token reopen, unbalanced wrapper acceptance,
entry/retry asymmetry, consumed breaking '<'). The grammar and grammar_state suites passed at
baseline: they pin the grammar contract the redesign preserves (shared classifier, header
grammar, constraint trigger).
```

### Baseline full CPU tests

Command:

```text
ctest --test-dir build-new-parser -C Release -E "_real" --output-on-failure --parallel 8
(environment: CUDA_VISIBLE_DEVICES=99)
```

Result:

```text
All CPU tests green except two pre-existing local-worktree effects that are not part of this
task (see Round D: the kv_capacity error message pinned by its test, and the four
linear_swiglu GPU tests whose skip path returns exit 1 instead of 77).
```

Skipped GPU/`_real` tests:

```text
All `_real` targets excluded by the -E filter. 98 of 150 registered ctest entries skip
themselves (no usable CUDA device with the GPU hidden).
```

---

# Finding status

| ID | Finding | Severity | Reproduced | Fix implemented | Targeted tests | Review verified | Final CPU suite |
|---|---|---:|---|---|---|---|---|
| F1 | Literal `<parameter=...>` in payload changes nesting | HIGH | ☑ | ☑ | ☑ | ☑ | ☑ |
| F2 | Tolerant executes open function | HIGH | ☑ | ☑ | ☑ | ☑ | ☑ |
| F3 | EOF `</parameter>` ambiguity can execute truncated payload | HIGH | ☑ | ☑ | ☑ | ☑ | ☑ |
| F4 | Partial next token can reopen previous value | MEDIUM | ☑ | ☑ | ☑ | ☑ | ☑ |
| F5 | Boolean wrapper state permits unbalanced nesting | MEDIUM | ☑ | ☑ | ☑ | ☑ | ☑ |
| F6 | Mixed `parameter`/`param` nesting inconsistency | MEDIUM | ☑ | ☑ | ☑ | ☑ | ☑ |
| F7 | Initial/retry marker policy asymmetric | LOW/MEDIUM | ☑ | ☑ | ☑ | ☑ | ☑ |
| F8 | Breaking second `<` is consumed instead of retried | LOW | ☑ | ☑ | ☑ | ☑ | ☑ |
| F9 | `uint8_t` depth overflow risk | LOW | ☑ | ☑ | ☑ | ☑ | ☑ |
| F10 | constrained-tool-decoding non-off mode is silent no-op | FUNCTIONAL | ☑ | ☑ | ☑ | ☑ | ☑ |

---

# F1 — Literal parameter opener in payload

## Reproduction

Fixture:

```text
<function name="read"><parameter=path>echo 'literal </parameter>
</function>
```

Observed before fix:

```text
The old ParameterValue scan tracked nested-opener depth. A literal `<parameter=...>` inside the
value changed the depth, so a literal opener/closer pair shifted which close closed the value;
the committed payload changed between inputs that differ only in echoed markup.
```

Expected:

```text
The value is an opaque byte range: only the matching outer close (the value's own parameter
family) is a candidate boundary, decided by the continuation bytes. No depth.
```

## Root cause confirmed

Files/functions:

```text
src/models/qwen3_5/frontend/tool_call_stream.cpp (old ParameterValue nested-depth scan)
src/models/qwen3_5/frontend/tool_call_grammar.{h,cpp} (opener grammar used for the depth)
```

Notes:

```text
The grammar's continuation classification already existed for the function close;
the parameter close used a different (depth-based) rule, so the two boundaries disagreed on
echoed markup.
```

## Implementation

Changed files:

```text
src/models/qwen3_5/frontend/tool_call_stream.{h,cpp}
src/models/qwen3_5/frontend/tool_call_grammar.{h,cpp}
```

Summary:

```text
ParameterValue is an opaque byte range (I1): the scan looks for the matching outer close literal
only. Each candidate is classified by classify_close_continuation (Invalid = the closer is
quoted payload, keep scanning; Complete = structural boundary; NeedMore = the input ends right
after the closer, provisional boundary). The classification uses the same wire grammar as the
structure (the next structural token after the closer).
```

## Tests added

```text
test_payload_bytes_round_trip (payload round trip at chunk sizes 1, 2, 3, 5, 7)
test_all_delimiter_byte_families (every-byte truncation matrix of the value/closer families)
F1/F3/F4 echo fixtures in the parser suite
```

Results:

```text
green (ninfer_tool_call_parser_test rc=0, zero FAIL lines)
```

## Review notes

```text
Round B re-verified with an adversarial probe sweep (14 cases, PROBE-OK): a value containing a
complete closer immediately followed by a legal continuation is classified as the boundary —
inherent wire ambiguity, documented in docs/tool_call_parser.md. A nested <tool_call> wrapper and
a fake <invoke ...> opener inside a value stay payload (probe 5/6).
```

---

# F2 — Open function execution in tolerant mode

## Reproduction

```text
<tool_call>
<function=read>
<parameter=path>foo.cpp
</parameter>
```

Before the fix the tolerant recovery committed the call after the last consumed parameter close
without requiring the function close: an open function executed (all finish reasons).

## Root cause confirmed

```text
src/models/qwen3_5/frontend/tool_call_stream.cpp: the call was committed at the wrapper close /
end of region; the open-call state was converted into executable output by the recovery path.
```

## Implementation

Confirm all of the following:

- [x] `open_call` is never converted to executable output. (decide_tool_call_recovery commits
  `progress.calls` only; `progress.open_call` is informational.)
- [x] a call becomes complete when matching function/invoke close is consumed. (FunctionBody:
  `complete_call()` at the function close, before the wrapper close.)
- [x] missing outer wrapper close remains recoverable only after function close.
  (ExpectWrapperClose EOF: the complete call is committed at its function close; the missing
  wrapper close is a truncated tail, tolerant only.
- [x] old unsafe missing-function-close test expectation changed.
  (the missing_function assertions now pin: open function → no call, TruncatedTail, all finish
  reasons; the missing-wrapper-close case pins: closed function + missing wrapper close is
  retained in tolerant with TruncatedTail.)

Changed files:

```text
src/models/qwen3_5/frontend/tool_call_stream.{h,cpp}
src/models/qwen3_5/frontend/tool_call_parser.h (decide_tool_call_recovery contract)
```

## Tests

```text
test_tolerant_missing_function_close (open function never committed)
test_tolerant_commits_function_closed_missing_wrapper (closed function + missing wrapper close
retained with TruncatedTail)
test_recovery_integrity (P3.5/P3.6: open call never executable, all finish reasons,
streaming == one-shot)
```

## Review notes

```text
Round A re-read: the wrapper close no longer commits (code + comment at ExpectWrapperClose).
Round B probe 3 (tool_call + function close, wrapper cut) green.
```

---

# F3 — EOF parameter closer ambiguity

## Reproduction

```text
<tool_call>
<function=bash>
<parameter=command>
echo 'literal </parameter>
```

Before:

```text
The candidate closer at input end was classified with the in-input rule; with the open function
the tolerant path could present the truncated payload as an executed call.
```

After:

```text
A closer at the input end is a provisional boundary (NeedMore): the parameter commits, but a
function that never closes stays open and therefore non-executable (the F2 boundary resolves the
echo case: the last </parameter> of a shell command that echoes the markup). Strict rejects;
tolerant never commits the open call; the content fallback preserves the bytes.
```

Assertions:

- [x] strict does not execute.
- [x] tolerant does not execute current call.
- [x] payload/content preserved according to fallback contract.
- [x] no output-limit/context-capacity exception weakens safety.
  (test_recovery_integrity runs all three FinishReason values; decide_tool_call_recovery does
  not consume the finish reason.)

---

# F4 — Partial next token reopens previous value

## Fixtures tested

```text
</parameter>
<paramet
```

```text
</parameter>
</funct
```

```text
</parameter>
</function>
</tool_
```

Additional:

```text
</parameter>
</invoke>
</funct

</parameter>
<parame

</parameter>
</function_calls

</function>
</tool_
```

## Implementation

Continuation result type:

```text
enum class CloseContinuation { Invalid, Complete, NeedMore }
classify_close_continuation(text, after, fn_family, wrapper):
  skip_ws; EOF -> NeedMore (provisional boundary, F3);
  complete parameter opener -> Complete;
  NeedMore parameter opener at the input end (cut-off next token) -> Complete (truncation, not
  payload);
  function close literal -> classify_function_close_continuation (Legal -> Complete,
  Payload -> Invalid);
  strict prefix of the function close at the input end -> Complete;
  otherwise -> Invalid.
```

State transitions:

```text
ParameterValue: candidate close + Complete/NeedMore -> parameter commits (last-occurrence wins),
mode = FunctionBody; candidate close + Invalid -> scan continues (quoted payload).
FunctionBody / Top / ExpectWrapperClose: cut-off structural token at the slice end ->
truncated (objective truncation fact), never a payload reopen.
```

## Tests/results

```text
Every-byte truncation matrix (test_all_delimiter_byte_families) over the delimiter families;
all F4 fixtures; green in ninfer_tool_call_parser_test (rc=0).
```

---

# F5 — Wrapper balance

## Reproductions

- [x] nested `tool_call`
- [x] nested `function_calls`
- [x] cross nested tool_call → function_calls
- [x] cross nested function_calls → tool_call
- [x] extra close
- [x] missing close
- [x] sequential valid wrappers

Old behavior:

```text
A single boolean wrapper flag: a second wrapper open silently replaced the state (unbalanced
nesting accepted); an extra close balanced against nothing.
```

New wrapper state:

```text
ToolWrapperKind { None, ToolCall, FunctionCalls } in RegionState:
- a second wrapper open before the first closed: definitive MalformedStructure break (no call
  of the region executes, both modes);
- the wrapper's close returns the state to None;
- a stray close at the top level (no open wrapper) balances against nothing: TrailingContent
  (strict) / no-op with the truncation recorded (tolerant);
- a missing close: the function-closed call is retained only in tolerant mode (F2 boundary),
  strict rejects; a <function_calls> region may end at the region end after its last invoke
  close (pre-existing contract, clean completion);
- sequential valid wrappers: each wrapper closes before the next opens: clean.
```

Tests:

```text
test_nested_wrappers (all four nesting variants, definitive break), the stray closer test,
the sequential wrappers test; green.
```

---

# F6 — Mixed parameter families in payload

Fixtures:

```text
<parameter=content>A <param=x>B</param> C</parameter>
```

```text
<param=content>A <parameter=x>B</parameter> C</param>
```

Results:

```text
Both directions: the value is the opaque byte range of the outer family; the inner
other-family open/close is plain payload; the exact payload is preserved in the committed
argument JSON.
```

Confirm:

- [x] exact payload preserved.
- [x] no recursive depth remains.

---

# F7 — Entry/retry marker policy

Chosen design:

```text
parser-aware recovery scanner: one entry-marker set (a complete wrapper literal or a
syntactically complete bare function/invoke opener) shared by the stream latch, the one-shot
discovery, the recovery retries and the constraint lazy trigger. The recovery retry re-enters at
the failed region's break byte (definitive break) or after the region start (EndOfInput), and
while a wrapper was open at the break only a later wrapper literal is a recovery entry.
```

Why safe:

```text
The entry decision is a pure function of the bytes (the same grammar classifies the marker in
every consumer), so the initial and the retry policy cannot diverge; the break offset is the
first byte that proved the break, so a retry never re-enters inside the failed structure; a bare
opener inside a failed wrapper's scope is rejected because it would reinterpret nested payload
as a new call.
```

Top-level markers by mode:

```text
<tool_call>, <function_calls>, a complete <function name="..."> / <invoke name="..."> opener
(strict and tolerant).
```

Recovery markers by mode:

```text
Break with no wrapper open: any entry marker at/after the break byte.
Break with a wrapper open: wrapper literals only (wrapper_only), starting after the failed
wrapper's own open (the nesting break).
```

Adversarial tests:

```text
test_recovery_entry_marker_policy (a nested fake call inside a failed wrapper never executes;
a later wrapper is a legitimate entry; a header that fails at its own start does not loop —
the next == base guard), first-failure diagnostic preserved.
```

Proof nested fake function cannot be recovered:

```text
<tool_call>
<function=read>
<parameter=path>
<tool_call><function=...>   (nested fake)
...
The first region breaks definitively with the wrapper open at the nested <tool_call> open;
the retry search is wrapper_only and starts after the failed wrapper's own open, so the nested
fake <function=...> inside the failed wrapper is never an entry; the later real wrapper is the
only entry. Pinned by test_recovery_entry_marker_policy.
```

---

# F8 — Breaking `<` marker restart

Before:

```
A marker candidate broken by NotMarker consumed the whole candidate including a trailing '<';
`prefix <function<tool_call>...` published `prefix <function<` as content and the inner marker
never latched.
```

After:

```
failed_marker_candidate_retained: when the grammar classifies the candidate as NotMarker, the
failed bytes up to the breaking byte are published as content; if the breaking byte itself is
'<' it is retained as the start of a fresh candidate, so `prefix <function<tool_call>` latches
at the machine's latch byte. A '<' the header grammar still accepts (a quoted attribute value)
keeps the candidate open instead of breaking it.
```

Shared marker transition implementation:

```
classify_tool_marker_prefix / failed_marker_candidate_retained in tool_call_grammar.{h,cpp};
consumed by ToolCallStreamParser::feed (stream latch) and ToolCallGrammarState (constraint lazy
trigger) — one transition table, two consumers.
```

Tests:

```
test_second_angle_flushes_candidate_without_restarting (rewritten for the new semantics: the
trailing '<' is retained, the inner marker latches),
test_quoted_angle_bracket_in_header_keeps_candidate (a quoted '<' keeps the candidate, no
restart),
test_marker_prefixes, test_marker_discovery,
the stream/constraint trigger-equivalence test (the constraint's trigger byte == the stream's
latch byte for every corpus line; the corpus excludes the quoted-angle line whose region is
invalid at the latch byte — that behavior is pinned by the dedicated quoted-angle test).
```

Confirm:

- [x] `<function<tool_call>` can recognize second marker.
- [x] quoted `<` in `<function name="a<b">` does not restart incorrectly.
- [x] stream parser and constraint trigger at same byte.

---

# F9 — 8-bit depth

Resolution:

- [x] depth removed entirely; OR
- [ ] replaced with bounded overflow-safe type.

Details:

```
The F1 redesign removes the nesting-depth counter entirely: the value is an opaque byte range
and the boundary is decided by the continuation classification. No uint8_t (or other) depth
state remains in the region machine; there is nothing to overflow.
```

---

# F10 — constrained-tool-decoding no-op

Chosen behavior:

- [x] non-off fails fast; OR
- [ ] non-off user-facing option removed.

Implementation:

```
validate_engine_options (src/runtime/engine/model_instance.{h,cpp}) is public, CPU-pure and
called by initialize_device before device work and by construct_model; the engine's fail-fast
therefore runs on both the CLI and the serve route, before CUDA initialization. The reserved
mode still parses (EngineOptions keeps the value); the engine refuses it with the explicit
error. No sampling integration is added.
```

CLI behavior:

```
--constrained-tool-decoding off|tool-calls-only parses to the EngineOptions value; the engine
refuses tool-calls-only at startup with the explicit error before any CUDA work. Help text
updated: "tool-calls-only is not implemented in this build and fails at startup; use off".
```

Server behavior:

```
Same: the serve option parses, the GenerationService constructs the Engine, and the engine
validation fails startup with the same explicit error.
```

Docs updated:

```
docs/cli.md (--constrained-tool-decoding row), docs/serving.md (same row),
docs/tool_call_parser.md (constrained tool decoding section: status + exact error message +
pointer to new_parser_phase4_design.md), CLI/serve --help text.
```

Tests:

```
tests/test_engine_options_validation.cpp (new CPU target ninfer_engine_options_validation_test):
default is Off; Off validates; ToolCallsOnly throws std::invalid_argument with exactly
"--constrained-tool-decoding=tool-calls-only is not implemented in this build; use off";
the normal option combination is unaffected (Off still validates after the rejection).
ninfer_cli_options_test, ninfer_serve_options_test: green (the flag still parses; off is the
default).
```

Confirm:

- [x] no GPU sampling integration added.
- [x] `off` remains unchanged.
- [x] no silent no-op remains.

---

# Phase log

## Phase 0 — Baseline and failing regressions

Status: `DONE`

Changes:

```
No product changes. Baseline build green; the regression tests for F1-F9 were written/updated
against the new design and reproduce the findings at baseline (parser suite failing on the
F1-F9 fixtures; grammar + grammar_state suites pinning the preserved grammar contract).
```

New failing tests before implementation:

```
ninfer_tool_call_parser_test: the F1-F9 fixture tests (payload nesting, tolerant open
function, EOF closer, partial next token, wrapper balance, entry/retry, breaking '<') failed
at baseline.
```

Targeted result:

```
recorded above; after the fix all three suites rc=0.
```

Review:

```
The failing set was reviewed one finding at a time; each fix was verified by its fixture before
the next phase.
```

---

## Phase 1 — Parameter boundary redesign

Status: `DONE`

Changes:

```
F1/F3/F4 core: opaque value byte range, classify_close_continuation (Invalid/Complete/NeedMore),
classify_function_close_continuation, provisional boundary at input end.
```

Tests:

```
payload round trip, every-byte truncation matrix, F1/F3/F4 fixtures — green.
```

Failures discovered during implementation:

```
The every-byte matrix exposed the function_calls EOF contract: a region ending right after the
last invoke close is a clean completion (pre-existing contract pinned by
test_tolerant_commits_function_closed_missing_wrapper), while a cut inside the wrapper close
literal is a truncated tail. The cut-matrix test was adjusted to the three-tier expectation
(no call / clean completion / TruncatedTail / complete).
```

Resolution:

```
ExpectWrapperClose EOF rule: FunctionCalls kind -> clean completion; ToolCall kind ->
truncated(false, i). Both modes; documented in docs/tool_call_parser.md.
```

---

## Phase 2 — Recovery integrity

Status: `DONE`

Changes:

```
F2: the function close is the executability boundary (complete_call at the function close);
decide_tool_call_recovery commits complete calls only; the open call never executes.
```

Every-byte truncation matrix result:

```
green: for every cut position of the reference regions the one-shot and the streamed (chunk
1..7) results agree; a cut before the function close commits nothing, a cut after it retains
the call in tolerant mode with TruncatedTail.
```

---

## Phase 3 — Wrapper state

Status: `DONE`

Changes:

```
F5: ToolWrapperKind state machine; unbalanced nesting is a definitive break; the wrapper close
returns to None; the stray closer balances against nothing.
```

Tests:

```
all four nesting variants + stray close + sequential wrappers — green.
```

---

## Phase 4 — Marker policy/progression

Status: `DONE`

Changes:

```
F7/F8: one shared entry-marker set; the recovery retry at the break byte with the wrapper_only
policy; the breaking '<' is retained as a fresh candidate (failed_marker_candidate_retained);
the constraint lazy trigger shares the transition table.
```

Tests:

```
entry-marker policy tests, the rewritten F8 tests, the stream/constraint trigger-equivalence
test — green (parser + grammar + grammar_state suites).
```

---

## Phase 5 — constrained-decoding user-facing behavior

Status: `DONE`

Changes:

```
F10: validate_engine_options fail-fast (before CUDA work), CLI/serve help + docs updated.
```

Tests:

```
ninfer_engine_options_validation_test (new CPU target) + CLI/serve option tests — green.
```

---

## Phase 6 — Documentation

Status: `DONE`

Files updated:

```
docs/tool_call_parser.md (marker entry and recovery section; tolerant safety boundary;
fundamental delimiter ambiguity; constrained decoding status), docs/cli.md, docs/serving.md,
docs/new_parser_phase4_design.md (acceptance gate reference), this progress file.
```

Accuracy review:

```
Every behavior claim in the docs was checked against the pinned tests; the F8 test wording and
the function_calls EOF contract were corrected during the review.
```

---

# Code Review Round A — Full state-machine review

Status: `DONE`

Reviewed whole files, not only diff:

- [x] `tool_call_grammar.h`
- [x] `tool_call_grammar.cpp`
- [x] `tool_call_stream.h`
- [x] `tool_call_stream.cpp`
- [x] `tool_call_parser.h`
- [x] `tool_call_parser.cpp`
- [x] `tool_call_grammar_state.h`
- [x] `tool_call_grammar_state.cpp`
- [x] `output_session.h`
- [x] `output_session.cpp`

Findings:

| ID | Severity | File/function | Finding | Fixed | Regression test |
|---|---:|---|---|---|---|
| A1 | LOW | tool_call_parser.h | stale recovery-contract comment (old "cut string value" wording for the tolerant boundary) | yes (comment only) | n/a (contract pinned by the existing recovery tests) |
| A2 | LOW | tool_call_stream.h | Termination::Complete comment and the state-machine comment described the removed nested-depth scan and missed the function_calls region-end completion | yes (comment only) | n/a |

Round A conclusion:

```
No behavior finding. All transitions are byte-driven and total over the region alphabet; the
recovery policy is pure and consumes the objective progress only. The tolerant entry into the
FunctionHeader state (NoMatch/Invalid/NeedMore) re-parses with the canonical header grammar and
terminates in truncated/invalid exactly as the strict path would; no infinite loop (the inner
scan is bounded by the text size, the outer loop strictly advances pos on every non-terminal
transition).
```

If any HIGH/MEDIUM finding existed, record the repeated Round A below.

### Round A repeat(s)

```
None required (no HIGH/MEDIUM finding).
```

---

# Code Review Round B — Adversarial/counterexample review

Status: `DONE`

Categories exercised:

- [x] source-code payload
- [x] shell payload
- [x] JSON/XML/HTML payload
- [x] literal tool tags
- [x] unbalanced literal tags
- [x] mixed param families
- [x] delimiter at EOF
- [x] every byte truncation
- [x] CRLF/whitespace
- [x] false markers in prose
- [x] multiple false markers
- [x] nested/cross wrappers
- [x] multiple calls
- [x] undeclared tool
- [x] duplicate parameter
- [x] long/invalid names

New findings:

| ID | Severity | Fixture | Root cause | Fix | Test |
|---|---:|---|---|---|---|
| B1 | INFO | `<function name="a"><parameter=x> </parameter></function>` (tolerant) | probe expectation error, not a parser bug: the legacy normalization trims the value; a whitespace-only value normalizes to the empty string (the wire value itself is the opaque byte range) | n/a (probe expectation corrected) | n/a (probe deleted after the sweep) |

Round B repeat count:

```
1 (the 14-case probe sweep after the pinned category tests).
```

Final Round B conclusion:

```
The 14-case adversarial probe sweep passed (PROBE-OK, 0 failures): open value at EOF not
committed; open function at EOF not executed; function close + missing wrapper close retained
(tolerant); a trailing byte after the function close classifies the close as quoted payload
(call stays open — the F1 boundary); a fake invoke opener and a nested tool_call wrapper inside
a value stay payload; an empty name is invalid in both modes; a double closer keeps the first
closer as quoted payload inside the committed value; a cut function close leaves the call
open; a quoted closer followed by a real opener commits both parameters; a closer-only value
commits empty; a cut second call keeps the first (function_calls); a whitespace-only value
trims to empty (legacy normalization); a second top-level region after a complete call is
rejected in strict. The probe was deleted after the sweep (temporary). No parser bug found; B1
is a probe-expectation correction.
```

---

# Code Review Round C — Compatibility/regression review

Status: `DONE`

Verified:

- [x] normal native Qwen tool call
- [x] multiple calls
- [x] attribute headers
- [x] quoted `>` header
- [x] duplicate parameter last-value-wins
- [x] schema normalization
- [x] boolean/integer/number/object/array handling
- [x] quoted malformed marker before real call
- [x] streaming chunk equivalence
- [x] existing compatibility forms
- [x] OutputSession integration

Regressions found:

```
None in the tool-call path. (The full-suite run reported five failures outside this path, see
Round D: one pre-existing local worktree change (kv_capacity) and four GPU kernel tests whose
skip path returns exit 1 instead of 77.)
```

Fixes:

```
n/a
```

---

# Code Review Round D — Full CPU suite

Status: `DONE` (see the final run below for the canonical numbers)

Command:

```text
ctest --test-dir build-new-parser -C Release -E "_real" --output-on-failure --parallel 8
(environment: CUDA_VISIBLE_DEVICES=99)
```

Result:

```text
97% tests passed, 5 tests failed out of 150 (47 passed, 98 skipped, 5 failed).
```

Total tests:

```text
150 registered ctest entries (the `_real` targets are excluded by the -E filter; GPU tests skip
themselves with exit 77 when no device is visible).
```

Passed:

```text
47
```

Failed:

```text
5 — ninfer_kv_capacity_test: pre-existing local worktree change (the explicit reservation error
throw in src/runtime/engine/kv_capacity.cpp is commented out in the user's working tree; the
test pins the message). Not caused by this task.
4× ninfer_linear_swiglu_{q4_a16,q8_a16,nvfp4,fp8}_test: GPU kernel correctness tests; the
shared run_profile() returns 77 on the no-device path, but the test mains map any non-zero
result to 1 (`return failures == 0 ? 0 : 1;`), so ctest records them as Failed although they
print "SKIP: no usable CUDA device". Pre-existing harness quirk, only visible with the GPU
hidden; not caused by this task.
```

Skipped GPU/`_real`:

```
98 of 150 (no usable CUDA device with the GPU hidden; `_real` excluded by the filter).
```

Confirm no GPU test was run:

- [x] YES

---

# Code Review Round E — End-to-end pipeline review

Status: `DONE`

Normal trace:

```text
generated bytes
→ OutputSession::feed (the decoder's feed, chunk by chunk)
→ ToolCallStreamParser::feed: the pre-marker scan holds the marker candidate, the whitespace is
  held in pending_ws_, the content is published
→ a complete marker latches (wrapper literal or a complete function/invoke opener); the region
  accumulates
→ finish(): parse_region on the latched region (marker latch, function header, function body,
  parameter value as an opaque range with the continuation classification, wrapper state)
→ decide_tool_call_recovery (strict all-or-nothing / tolerant complete calls only)
→ normalize_parsed_tool_call (contract-aware parameter normalization)
→ GeneratedToolCall (name + arguments_json)
→ API serialization (the protocol adapter owns the wire identifier)

Concretely: "Calling weather. \n<tool_call>\n<function=get_weather>\n<parameter=city>Oslo
</parameter>\n</function>\n</tool_call>" → marker latched at <tool_call>; the region parse
commits get_weather at its function close; the wrapper close closes the wrapper; the recovery
is clean (Complete, no diagnostic); the argument JSON {"city":"Oslo"}.
```

Adversarial trace:

```text
"prefix <function<tool_call>\n<function=read>\n<parameter=path>foo.cpp</parameter>\n</function>\n</tool_call>"
→ the pre-marker scan: the candidate <function breaks as NotMarker at the second '<' (F8):
  `prefix <function` is published, the '<' is retained; the machine latches at the
  <tool_call> literal, at the same byte the constraint's lazy trigger would fire (the shared
  transition table); the region parse is clean; the content prefix `prefix <function` is
  returned verbatim.
```

Questions answered:

- [x] Can an open value execute? **NO**
- [x] Can an open function execute? **NO**
- [x] Can a wrapper imbalance complete strict? **NO**
- [x] Can literal `<parameter=...>` alter payload nesting? **NO**
- [x] Can a partial next token reopen a prior parameter? **NO**
- [x] Can stream and constraint marker logic diverge? **NO**
- [x] Can non-off constrained decoding silently do nothing? **NO** (F10 fail-fast)

Final Round E findings:

```
None.
```

---

# Final targeted CPU test run

Command:

```text
build-new-parser\tests\Release\ninfer_tool_call_grammar_test.exe
build-new-parser\tests\Release\ninfer_tool_call_grammar_state_test.exe
build-new-parser\tests\Release\ninfer_tool_call_parser_test.exe
build-new-parser\tests\Release\ninfer_engine_options_validation_test.exe
build-new-parser\tests\Release\ninfer_cli_options_test.exe
build-new-parser\tests\Release\ninfer_serve_options_test.exe
```

Result:

```text
all rc=0 (zero FAIL lines).
```

---

# Final full CPU test run

Command:

```text
ctest --test-dir build-new-parser -C Release -E "_real" --output-on-failure --parallel 8
(environment: CUDA_VISIBLE_DEVICES=99)
```

Actual result:

```text
97% tests passed, 5 tests failed out of 149: 46 passed, 98 skipped (no usable CUDA device
with the GPU hidden), 5 failed. The five failures are exactly the two pre-existing local-
worktree effects recorded in Round D (kv_capacity; the four linear_swiglu tests whose skip
path returns exit 1 instead of 77). No failure in the tool-call path. All targeted suites
(grammar, grammar_state, parser, F10 validation, CLI/serve options, frontend integration) are
among the 46 passed.
```

GPU/runtime tests:

```text
NOT RUN (GPU hidden via CUDA_VISIBLE_DEVICES=99; every GPU test took its own skip path)
```

---

# Final unresolved items

There must be **no unresolved HIGH or MEDIUM parser correctness finding**.

| Severity | Item | Reason unresolved | Impact |
|---:|---|---|---|
| (none) | (none) | (none) | (none) |

Inherent wire-format ambiguity that remains by design:

```
A parameter value that contains a complete closer immediately followed by a legal structural
continuation (e.g. echoed `</parameter>` directly followed by another parameter opener, or by
the function close followed by the wrapper close/region end) is classified as the boundary —
the parser cannot distinguish echoed markup from a real boundary on those bytes. The parser
never guesses inside that class: it commits the candidate value up to the classified boundary
and preserves the bytes verbatim. The same applies at the input end: a closer at EOF is a
provisional boundary (the parameter commits, the function stays open and non-executable), and
a cut-off next structural token is a truncation, never a payload reopen. Documented in
docs/tool_call_parser.md (fundamental delimiter ambiguity).
```

---

# Final sign-off checklist

- [x] F1 complete.
- [x] F2 complete.
- [x] F3 complete.
- [x] F4 complete.
- [x] F5 complete.
- [x] F6 complete.
- [x] F7 complete.
- [x] F8 complete.
- [x] F9 complete.
- [x] F10 complete.
- [x] Direct regression test for every finding.
- [x] Every-byte cut matrix green.
- [x] One-shot/stream equivalence green.
- [x] Stream/constraint marker equivalence green.
- [x] Targeted parser tests green.
- [x] Frontend integration tests green.
- [x] CLI/server CPU tests green.
- [x] Full CPU suite green (modulo the two pre-existing local-worktree effects recorded in
      Round D; none in the tool-call path).
- [x] GPU/`_real` tests not run.
- [x] Review A complete with no unresolved HIGH/MEDIUM.
- [x] Review B complete with no unresolved HIGH/MEDIUM.
- [x] Review C complete.
- [x] Review D complete.
- [x] Review E complete.
- [x] Docs accurately describe tolerant safety boundary.
- [x] Docs accurately describe protocol ambiguity.
- [x] No silent constrained-decoding no-op.
- [x] Final implementation SHA recorded below.

Final implementation SHA:

```text
9927ff3d92f66748d1cbfc4f9d0f9b9a91db2d9f (HEAD; the implementation is the uncommitted
worktree on top of it — no commit was requested). Changed files: apps/cli/options.cpp,
docs/cli.md, docs/serving.md, docs/tool_call_parser.md,
src/models/qwen3_5/frontend/tool_call_grammar.{h,cpp},
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp,
src/models/qwen3_5/frontend/tool_call_stream.{h,cpp},
src/runtime/engine/engine.cpp, src/runtime/engine/model_instance.{h,cpp},
src/serve/serve_options.cpp, tests/cmake/RuntimeTests.cmake,
tests/test_tool_call_grammar_state.cpp, tests/test_tool_call_parser.cpp,
tests/test_engine_options_validation.cpp (new),
docs/NInfer_new_parser_design_bugfix_implementation.md + this progress file (new).
```

Final date:

```text
2026-09-29
```
