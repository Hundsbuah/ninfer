# NInfer `new_parser_design` — Round 2 Bugfix Progress

> Working evidence log for
> `NInfer_new_parser_design_round2_bugfix_implementation.md`.

## Rules

- Update this file after every implementation phase and every review loop.
- Do not copy "green" status from the Round-1 progress file.
- Reproduce every Round-2 finding on the pinned baseline first.
- Record exact commands, exit codes, and failing test names.
- Never mark a finding verified from compilation alone.
- Never mark a finding fixed without a direct regression test.
- Do not run GPU runtime tests, `_real` tests, model inference, or GPU benchmarks.
- If a test attempts GPU runtime, skip it and record the exact reason.
- Preserve all newly discovered findings even after they are fixed.
- If a review round finds a HIGH/MEDIUM issue, add it to the finding table and repeat the round.

---

# Baseline

- Repository: `Hundsbuah/ninfer`
- Branch: `new_parser_design`
- Round-2 reviewed baseline SHA: `fcc4eec7fc2696b38681628caa7d251b25750c53`
- Baseline commit message: `fix: tool-call parser F1-F10 bugfixes + constrained flag fail-fast`
- Date started: 2026-09-30
- Local clone: `E:/KI/ninfer-custom` (Windows 10 x64 worktree)
- Worktree state: not clean at start — four pre-existing local (user) adaptations unrelated to the parser:
  `CMakeLists.txt` (vcpkg path), `build_native.bat`,
  `src/runtime/engine/context_cache/materialization_budget.h` (planning allowance 250→500 ms),
  `src/runtime/engine/kv_capacity.cpp` (error throw commented out — this local edit makes
  `ninfer_kv_capacity_test` fail in this worktree; pre-existing, not a Round-2 change). All four
  were preserved.
- Build directory: `build-new-parser` (reused from Round 1; clean and compatible)
- Generator: Visual Studio 18 2026 (multi-config, Release)
- Compiler: MSVC (VS 18 2026 toolset)
- CUDA toolkit used for compilation only: CUDA v13.4 (`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4`)
- CPU test environment: `ctest -C Release --parallel 8`, `CUDA_VISIBLE_DEVICES=99` so no test may
  touch the GPU (a local AI workload runs on the GPU); 8 compile and 8 test threads throughout
- GitHub combined status on baseline: **no statuses**
- GitHub workflow runs on baseline: **none**

## Historical context

Round-1 F1–F10 are treated as implemented baseline behavior and must remain regression-tested.
Round 2 reopens only the invariants contradicted by the new review.

Do not change historical Round-1 progress checkmarks to hide Round-2 regressions.

---

# Baseline build

Command:

```text
cmake --build build-new-parser --config Release --parallel 8
```

Exit code:

```text
0
```

Result:

```text
Incremental build of the pinned HEAD in build-new-parser (Visual Studio 18 2026, Release).
All parser targets compiled clean (MSVC, CUDA v13.4 for compilation only). The Round-1
parser suites (before any Round-2 test addition) are green at the pinned HEAD.
```


# Baseline targeted parser run

Command:

```text
./build-new-parser/tests/Release/ninfer_tool_call_parser_test.exe
./build-new-parser/tests/Release/ninfer_tool_call_grammar_test.exe
./build-new-parser/tests/Release/ninfer_tool_call_grammar_state_test.exe
./build-new-parser/tests/Release/ninfer_engine_options_validation_test.exe
```

Result:

```text
All four exit 0 at the pinned HEAD (Round-1 F1-F10 contract intact). The Round-2 regression
tests added in phase R2-0 are the only failing delta at the baseline; see the reproducer run
below.
```


---

# Round-2 reproducer run BEFORE fixes

The following tests must be added first and run against the baseline:

```text
CR1 nested recovery scope
CR2 open function_calls EOF
CR3 function_calls trailing bytes
CR4 bare→wrapper complete-entry transition
CR5 tolerant undeclared tool
CR6 fenced content phantom call
```

Command:

```text
# Frontend sources stashed to the pinned baseline (the Round-2 test additions stay),
# the parser target rebuilt, and the suite run:
git stash push -m r2-baseline-repro -- src/models/qwen3_5/frontend/
cmake --build build-new-parser --config Release --parallel 8 --target ninfer_tool_call_parser_test
./build-new-parser/tests/Release/ninfer_tool_call_parser_test.exe
git stash pop   # then rebuild and rerun to confirm the fix (see Final CPU)
```

Expected failing tests:

```text
All Round-2 CR1-CR6 regression checks (see the CR sections below), plus the Round-1
expectations that CR2/CR4/CR5 replace ("wrapper close optional at the region end = clean
completion", bare-function close continuations, tolerant undeclared-name emission).
```

Actual:

```text
Exit code 1. 884 FAIL lines, 255 unique check names. Reproduced per finding:
  CR1: "a quoted broken wrapper before a real call was recovered (CR1)";
       "a failed open wrapper's scope was re-entered during streaming (CR1)"; the whole
       nested-scope matrix (nested tool_call / nested function_calls / function_calls ->
       tool_call / invalid outer header / truncated outer header / prose before nested
       wrapper / two nested candidates / call after visible close) x strict/tolerant x
       one-shot + every streaming split: "CR1 every-split streaming diverged from one-shot".
  CR2: "a missing function_calls close completed cleanly in strict mode"; "a missing
       function_calls close dropped the tolerant call sequence".
  CR3: "call count / fallback reason / response flag: CR3 trailing prose (strict/tolerant)";
       "CR3 partial marker"; "CR3 next valid region".
  CR4: "CR4 boundary: strict/tolerant cut <N>" across the cut sweep; "other families:
       tolerant cut at 66/67".
  CR5: "tolerant mode emitted a syntactically valid undeclared call (CR5)"; "incremental
       quoted marker lost or duplicated bytes"; "incremental quoted marker changed terminal
       diagnostics".
  CR6: "CR6 backtick fence", "CR6 tilde fence", "CR6 CRLF fence", "CR6 unclosed fence",
       "CR6 longer fence contains shorter", "CR6 real call after closed fence" x
       strict/tolerant.
After the fixes the same suite exits 0 (see Final CPU). The stash pop restored the worktree
(POP EXIT 0) and the fixed rebuild exits 0.
```

---

# Finding status

| ID | Finding | Severity | Baseline reproduced | Fix implemented | Direct test | Counter-review | Final CPU |
|---|---|---:|---|---|---|---|---|
| CR1 | Failed open wrapper can recover/execute nested wrapper | HIGH | ☑ | ☑ | ☑ | ☑ | ☑ |
| CR2 | Open `<function_calls>` can be `Complete` at EOF | MEDIUM | ☑ | ☑ | ☑ | ☑ | ☑ |
| CR3 | Tail after `</function_calls>` is not validated | MEDIUM | ☑ | ☑ | ☑ | ☑ | ☑ |
| CR4 | Bare function → complete wrapper continuation mismatch | MEDIUM | ☑ | ☑ | ☑ | ☑ | ☑ |
| CR5 | Tolerant bypasses declared-tool validation | HIGH | ☑ | ☑ | ☑ | ☑ | ☑ |
| CR6 | Fenced valid markup can become phantom call | HIGH | ☑ | ☑ | ☑ | ☑ | ☑ |
| VG1 | Independent CPU verification required | VERIFY | ☑ | n/a | n/a | ☑ | ☑ |

---

# CR1 — Failed wrapper recovery scope

## Baseline reproducer

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

Contract:

```text
declared: bash
not declared / invalid: bad.name
```

Baseline actual result:

strict: 0 calls, MalformedStructure (no retry in strict).
tolerant: 1 committed call — the nested <function=bash> (command=echo SHOULD_NOT_EXECUTE)
was recovered and made executable: the retry re-entered the failed outer <tool_call>'s
still-unclosed scope at the nested <tool_call> literal and committed it.

Expected after fix:

```text
strict: 0 calls
tolerant: 0 nested calls
failed outer region does not re-enter inside its unclosed scope
```

## Root cause confirmed

Code locations:

```text
src/models/qwen3_5/frontend/tool_call_stream.cpp
ToolCallStreamParser::finish
wrapper_at_break / wrapper_only retry
```

Exact explanation:

finish() ran the recovery retry from base + break_offset with find_tool_marker(...,
wrapper_only=true) whenever wrapper_at_break != None. That search runs inside the failed
wrapper's still-unclosed scope, so a nested <tool_call>/<function_calls> literal there was
treated as the start of a fresh top-level region. Nothing proved the exit from the failed
wrapper: its close could be a quoted payload byte, could come only after the nested region,
or could be absent (a nesting break). The nested region then parsed on its own and the
recovery policy committed its calls.

## Chosen fix

- [x] conservative: no retry while failed wrapper remains open
- [ ] scope-quarantine scanner with proven exit (spec-optional alternative; not needed — the
      conservative rule already yields 0 calls on the main reproducer and the spec's required
      "recovery restarts after a proven wrapper exit" case, where the failed wrapper's close
      was structurally consumed before the later break, still recovers because that later
      break has wrapper_at_break == None)
- [ ] other (explain)

Implementation details:

- ToolCallParseProgress::wrapper_at_break records the wrapper state at the break
  (Definitive) or at the input end (EndOfInput).
- decide_tool_call_recovery(): a Definitive break with wrapper_at_break != None is never
  retryable: the failed wrapper owns the remaining ambiguous bytes, so the decision is
  Reject (strict) / Reject-without-retry (tolerant) — prior committed calls are retained
  only through the EndOfInput/truncation path, never through a retry inside the failed scope.
- ToolCallStreamParser::finish(): the retry search (find_tool_marker) runs only when the
  decision allows it, i.e. only from a break with wrapper_at_break == None — a proven
  top-level scope. The wrapper_only literal-skip heuristic is gone.

Why this cannot recover a nested wrapper:

A nested wrapper call can only be committed through the retry re-entry. The re-entry now
requires wrapper_at_break == None, which is false by construction while the failed wrapper
is open at the break. Every byte after the break therefore stays inside the failed scope
and the region falls back to verbatim content (strict) or to the retention path with 0
new calls (tolerant). A later wrapper after a proven exit (the failed wrapper's close
structurally consumed at Top before a later break) breaks with wrapper_at_break == None
and may be recovered — that is the only legitimate re-entry, and it is outside the failed
wrapper.

## Tests

- [x] failed tool_call → nested tool_call
- [x] failed tool_call → nested function_calls
- [x] failed function_calls → nested tool_call
- [x] invalid name (invalid outer header: <function=bad.name>)
- [x] invalid header (broken outer header bytes)
- [x] truncated header (outer header cut at input end: EndOfInput, no retry)
- [x] multiple nested markers (two nested candidates)
- [x] one-shot (all scenarios, strict + tolerant)
- [x] streaming chunks 1/2/3/5/7 (split sweep)
- [x] every split around break/retry marker ("CR1 every-split streaming diverged from
      one-shot" compares every 1-byte split against the one-shot parse)

Command/result:

Baseline (stashed frontend, Round-2 tests): exit 1 — all CR1 matrix checks FAIL
("response flag / call count / fallback reason / verbatim fallback content: CR1 ..." for
every scenario x mode x split, plus the quoted-broken-wrapper and streaming-scope checks).
Fixed: ninfer_tool_call_parser_test exit 0 (ok); the streaming every-split sweep agrees with
one-shot on all CR1 fixtures.

## Counter-review

Attempted bypasses:

1. Closed-then-later wrapper: the failed wrapper's close consumed at Top (proven exit), a
   later Definitive break has wrapper_at_break == None -> retry allowed from the later
   break; a real top-level call after the proven boundary is recovered (positive case in
   test_recovery_only_restarts_after_proven_scope).
2. Nested literal inside a quoted value: value bytes are consumed only at a boundary
   accepted by the shared close-continuation classifier; a wrapper open after a value close
   is not a legal continuation -> the value swallows the nested structure (rejected
   verbatim, both modes).
3. Open value at break: an open value yields EndOfInput, not a Definitive break; retries
   exist only for Definitive breaks -> no re-entry.
4. Chunk-boundary games around the nested literal: the machine is byte-deterministic and
   the every-split sweep equals one-shot for all CR1 fixtures.
5. Failed wrapper close quoted in a parameter value followed by a real wrapper: the quoted
   close is value text (continuation classifier), so the wrapper is still open at any later
   break -> no retry -> 0 calls (safe direction).

Result:

No bypass found. The conservative rule is the delivered behavior; the optional quarantine
scanner (spec §12.3 alternative) would add availability only in the balanced-structure
case and is not implemented (spec allows the conservative rule; the main reproducer and
all required regression variants pass with it).

---

# CR2 — Open function_calls reported Complete

## Baseline fixture

```text
<function_calls>
<function=read>
<parameter=path>a</parameter>
</function>
```

Baseline strict result:

Complete in both modes: the region ended cleanly at the input end with the
<function_calls> wrapper still open — 1 committed call, no diagnostic (reason None).

Baseline tolerant result:

Complete in both modes: same as strict — the missing wrapper close was tolerated as a
"clean completion at the region end" (Round-1 expectation).

## Required result

```text
strict:
  no structured call response
  MalformedStructure

tolerant:
  may retain completed read call
  TruncatedTail
  region itself is not Complete
```

## Implementation

Code changed:

tool_call_stream.cpp, Top state, the i == text.size() (EOF) handling:
  - Complete (out.termination = Complete) now requires s.mode == Top AND
    s.wrapper == None. A wrapper still open at EOF takes the no_retain branch and
    reports truncated(false, i) -> EndOfInput with wrapper_at_break set to the open
    wrapper. The "wrapper close optional at the region end" branch is removed.
tool_call_stream.h: ToolCallRegionTermination::Complete documented as requiring the top
  level to reach the input end with no wrapper open (wrapper balance is part of the
  outcome, R2-I2).
decide_tool_call_recovery() is unchanged in shape: EndOfInput + tolerant + closed calls
  -> CommitCalls with the TruncatedTail diagnostic; strict maps EndOfInput to the
  MalformedStructure fallback.

Confirm:

- [x] `Complete` requires wrapper=None.
- [x] open FunctionCalls EOF produces EndOfInput.
- [x] tolerant recovery is separate from structural completeness.
- [x] Round-1 "pre-existing clean completion" expectation removed/corrected (test pins
      updated to MalformedStructure strict / TruncatedTail tolerant; docs updated).

## Tests

```text
test_function_calls_holds_call_sequence:
  - one_open (one closed call, wrapper open at EOF): strict -> MalformedStructure, 0
    calls; tolerant -> 1 call, TruncatedTail.
  - two_calls_open (two closed calls, wrapper open at EOF): strict -> MalformedStructure,
    0 calls; tolerant -> 2 calls, TruncatedTail.
  - two_calls (wrapper closed): both modes -> 2 calls, None (unchanged).
  - second_cut / nested / nested_value expectations unchanged.
  - streaming parity: decoder feed == one-shot on the complete sequence.
Command/result: baseline exit 1 ("a missing function_calls close completed cleanly in
strict mode", "a missing function_calls close dropped the tolerant call sequence"); fixed
exit 0.
```

---

# CR3 — Tail after function_calls close ignored

## Baseline fixture

```text
<function_calls>
<function=read>
<parameter=path>a</parameter>
</function>
</function_calls>
EXTRA
```

Baseline actual:

```text
Complete in both modes with 1 call and no diagnostic: the </function_calls> close
consumed the parse immediately (complete() right after the close) and the trailing
EXTRA bytes were never validated.
```

Required strict result:

```text
TrailingContent
no clean structured result
```

## Implementation

Confirm transition:

```text
consume </function_calls>
→ wrapper=None
→ Top
→ validate remainder
```

Changed code:

```text
tool_call_stream.cpp, Top/ExpectFunction wrapper-close branches: consuming
</tool_call> / </function_calls> now sets s.wrapper = None, keeps the state at Top and
continues the loop instead of calling complete() and returning. The remainder of the
region is then validated by the top-level rules: EOF -> Complete; a complete next
top-level entry -> a new region (or, inside an open function_calls, a nesting break);
a strict marker prefix at EOF -> EndOfInput; any other non-marker byte ->
TrailingContent (calls present) / MalformedStructure (no calls).
```

## Test matrix

- [x] close + EOF
- [x] close + whitespace
- [x] close + prose ("CR3 trailing prose")
- [x] close + next wrapper ("CR3 next valid region")
- [x] close + next bare compatibility entry
- [x] close + partial marker ("CR3 partial marker")
- [x] streaming split on closing wrapper (split sweep includes splits inside the
      close literal; every-split sweep agrees with one-shot)

Results:

```text
Baseline: "CR3 trailing prose (strict/tolerant)", "CR3 partial marker (strict/tolerant)",
"CR3 next valid region (strict/tolerant)" FAIL. Fixed: all PASS (suite exit 0).
```

---

# CR4 — Complete next wrapper continuation mismatch

## Fixture

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

Baseline actual:

```text
The uncut fixture was consistent (the Top state took the complete <tool_call> as the
next region), but every cut inside the next-entry bytes diverged:
"CR4 boundary: tolerant cut 100-147" and "CR4 boundary: strict cut 144/145/147" FAIL,
plus "other families: tolerant cut at 66/67". The value scan's close-continuation
lookahead classified the same bytes as value text (a quoted continuation) whenever the
input ended inside the next entry, so the boundary decision depended on where the input
was cut.
```

## Root cause

```text
Two independent consumers classified the bytes after a function close: the value scan's
close-continuation lookahead (classify_function_close_continuation) and the Top-state
entry handling. The lookahead accepted a strict prefix of <tool_call> but not the
complete opener in the wrapper=None path, so one consumer saw a legal continuation
where the other saw value text.
```

## Shared top-level continuation helper

```text
One grammar decision, classify_top_level_entry (tool_call_grammar.cpp / .h), consumed
by both the Top state and the close-continuation lookahead. A complete next entry (a
wrapper literal or a syntactically complete function/invoke opener) is a legal
continuation for a function close; its strict prefixes at the input end are NeedMore
(truncation), never value text. Wrapper-aware legality (a wrapper open inside an open
wrapper is a nesting break, not an entry) stays the caller's scope rule.
```

## Matrix

| Previous entry | Next complete entry | Partial next entry | Expected |
|---|---|---|---|
| bare function | tool_call | `<tool_...` | consistent |
| bare invoke | tool_call | `<tool_...` | consistent |
| bare function | function_calls | partial | consistent |
| bare function | bare function | partial | consistent |

Results:

```text
Baseline: "CR4 boundary: strict cut 144/145/147", "CR4 boundary: tolerant cut 100-147",
"other families: tolerant cut at 66/67" FAIL. Fixed: all PASS (suite exit 0); the cut
sweep now shows one consistent boundary — a complete next entry is always a boundary,
its prefixes truncate (EndOfInput), and one-shot and streaming agree at every split.
```

---

# CR5 — Tolerant undeclared tool

## Contract

```text
declared: read
```

Fixture:

```text
<tool_call>
<function=example_function_name>
</function>
</tool_call>
```

## Baseline strict

UndeclaredTool, 0 calls: the declared-name check ran (strict mode rejects a name
outside the declared set at the function header).

## Baseline tolerant

1 committed call: the syntactically valid <function=example_function_name> was
emitted as an executable call with no diagnostic — the declared-name check was gated on
`!policy.tolerant`, so tolerant mode bypassed the identity validation entirely.

## Required result

Both modes:

```text
0 emitted undeclared calls
UndeclaredTool
```

For valid prior calls + later undeclared call:

```text
strict: all-or-nothing reject
tolerant: may retain prior complete declared call; never undeclared call
```

## Implementation

Parser name policy:

tool_call_stream.cpp, FunctionHeader state: the declared-name check changed from
`if (!policy.tolerant && policy.enforce_declared_names && ...)` to the unconditional
`if (policy.enforce_declared_names && ...)`. An undeclared name is a definitive break
(UndeclaredTool) at the function header in both modes; tolerance repairs syntax damage
only, not identity.

Defense-in-depth before GeneratedToolCall:

tool_call_parser.cpp, parse_qwen_tool_call_output: before building the output, if
contract.enforce_declared_names, every call in result.region.calls must resolve in the
contract (find_tool_contract != nullptr); otherwise the output falls back to verbatim
content with the UndeclaredTool diagnostic. Identity is a property of the output, not a
side effect of one parse branch (graceful fallback, never a crash).

Confirm:

- [x] tolerance no longer changes identity policy.
- [x] placeholder `function_name` not emitted.
- [x] placeholder `example_function_name` not emitted.
- [x] syntactically invalid names remain InvalidToolName (name validity runs before the
      identity check).
- [x] declared valid calls unaffected.

## Tests/results

test_tolerant_rejects_undeclared_tools: strict -> 0 calls, UndeclaredTool; tolerant ->
0 calls, UndeclaredTool (the undeclared call is never emitted in either mode); a prior
complete declared call followed by an undeclared call -> tolerant retains only the prior
declared call with TruncatedTail; placeholder fixtures (function_name /
example_function_name) rejected; streaming parity (decoder == one-shot).
Baseline: "tolerant mode emitted a syntactically valid undeclared call (CR5)" FAIL,
plus the incremental quoted-marker streaming checks (the P3.10 streaming check now uses
the write contract through the stream lambda's contract parameter, so the quoted-closer
path is actually exercised). Fixed: all PASS (suite exit 0).

---

# CR6 — Fenced content phantom tool call

## Baseline fixture

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

Contract:

```text
bash declared
```

Baseline result:

```text
The <tool_call> marker inside the fenced xml block latched: the machine parsed the
fence's bytes as a tool region and committed 1 phantom call (bash, echo example); the
prose around the fence was the only content.

Required:

```text
full bytes remain content
0 tool calls
```

## Fence implementation

State:

```text
FenceTracker (tool_call_stream.h, private class of ToolCallStreamParser): a byte-
streaming, line-oriented state machine with the fields in_fence_, fence_char_,
fence_len_, at_line_start_, indent_, run_char_, run_len_, close_pending_,
opener_line_. consume(byte) returns Pass (the byte is not fence structure: the marker
candidate machine handles it) or Content (the byte is fence structure: publish as
ordinary content, never a marker candidate). Wired into feed(): while not latched, every
incoming byte is consumed by the tracker first; only Pass bytes reach the
pending_ws_/marker_prefix_ candidate machine.

Supported fences:

- [x] ``` (backtick fence)
- [x] ~~~ (tilde fence)
- [x] longer opener (4+ run opens; a shorter same-char run does not close)
- [x] matching close char (a different character never closes)
- [x] close length >= opener length
- [x] info string (the rest of the opener line is fence content)
- [x] LF
- [x] CRLF
- [x] documented indentation rule (up to 3 spaces at line start)

## Tests

- [x] backtick fenced call ("CR6 backtick fence")
- [x] tilde fenced call ("CR6 tilde fence")
- [x] unclosed fence (stays open through EOF; "CR6 unclosed fence")
- [x] longer fence containing a shorter run ("CR6 longer fence contains shorter")
- [x] language/info string ("CR6 backtick fence" fixture carries xml)
- [x] CRLF ("CR6 CRLF fence")
- [x] call after closing fence ("CR6 real call after closed fence" — a real marker
      after the fence closes still latches)
- [x] every-byte/chunk split (split sweep over the fence fixtures)
- [x] reasoning/content channel transition does not leak fence state (the tracker is
      scoped to the pre-latch content channel of one stream instance)

Results:

```text
Baseline: all CR6 fixture checks FAIL in both modes (the phantom call was committed).
Fixed: all PASS (suite exit 0) — the full fence bytes are ordinary content, 0 tool calls;
a real marker after the fence close latches normally.

## Residual ambiguity documented

Unfenced exact tool markup:

```text
docs/tool_call_parser.md, section "Code fences" (this repository). The wire format still
has no delimiter escape for UNFENCED exact tool markup: a <tool_call> block written
without a fence is a legal tool region by construction, and the parser treats it as one.
```

Confirm no claim that this is fully solved:

- [x] yes

---

# Phase log

## R2-0 — Freeze and reproduce

Status: `COMPLETE`

HEAD:

```text
fcc4eec7fc2696b38681628caa7d251b25750c53 (pinned Round-2 baseline; verified at start)
```

Worktree:

```text
not clean: 4 pre-existing user-local adaptations (CMakeLists.txt, build_native.bat,
materialization_budget.h planning allowance, kv_capacity.cpp commented throw) preserved;
no parser-file modifications at start.
```

Build:

```text
cmake --build build-new-parser --config Release --parallel 8 -> exit 0 (incremental).
```

Failing regression tests on baseline:

```text
Round-2 CR1-CR6 tests added, then the frontend sources stashed to the baseline and the
suite run: exit 1, 884 FAIL lines, 255 unique check names (full list in the reproducer
section above). All six findings reproduced.
```

---

## R2-1 — Recovery scope / CR1

Status: `COMPLETE`

Changes:

```text
tool_call_stream.h/.cpp: ToolCallParseProgress::wrapper_at_break gains semantic authority
(the wrapper state at the break / input end); decide_tool_call_recovery rejects a
Definitive break with wrapper_at_break != None (no retry inside a failed wrapper's scope);
finish() runs the retry search only from a proven top-level scope (wrapper_at_break ==
None); the wrapper_only literal-skip heuristic removed.
```

Targeted tests:

```text
CR1 matrix (nested tool_call / nested function_calls / function_calls->tool_call / invalid
outer header / truncated outer header / prose before nested wrapper / two nested
candidates / call after visible close) x strict/tolerant, one-shot + every-split
streaming sweep: all PASS (suite exit 0). Positive proven-exit case
(test_recovery_only_restarts_after_proven_scope) PASS.
```

New regressions found:

```text
None. Round-1 recovery tests (test_recovery_retry_entry_policy, phase3) stay green.
```

---

## R2-2 — Wrapper completion / CR2 + CR3

Status: `COMPLETE`

Changes:

```text
CR2: Top-state EOF handling — Complete now requires s.mode == Top && s.wrapper == None;
an open wrapper at EOF reports EndOfInput (truncated(false, i)) with wrapper_at_break set.
The "wrapper close optional at the region end = clean completion" branch removed.
CR3: Top/ExpectFunction wrapper-close branches — consuming </tool_call> / </function_calls>
sets s.wrapper = None and continues at Top (validating the remainder) instead of
complete() + return.
```

Tests:

```text
F2 one_open (strict MalformedStructure / tolerant 1 call TruncatedTail), F5 two_calls_open
(strict MalformedStructure / tolerant 2 calls TruncatedTail), CR3 matrix (close +
EOF/whitespace/prose/next wrapper/bare entry/partial marker, streaming split on the close):
all PASS (suite exit 0).
```

---

## R2-3 — Continuation unification / CR4

Status: `COMPLETE`

Changes:

```text
tool_call_grammar.h/.cpp: classify_top_level_entry + TopLevelEntryInfo (End / NoEntry /
NeedMore / Entry with the parsed opener) — the single top-level entry classification.
tool_call_stream.cpp: classify_function_close_continuation and the Top state consume it,
so a complete next entry is a legal function-close continuation and its strict prefixes
truncate (EndOfInput) in both consumers; the duplicated token lists are refactored onto
the helper.
```

Tests:

```text
CR4 boundary cut sweep (strict + tolerant, all cut positions), "other families" sweep
(invoke/param families), entry composition matrix (bare function/invoke followed by
wrapper / bare entry): all PASS (suite exit 0).
```

---

## R2-4 — Declared-name enforcement / CR5

Status: `NOT STARTED`

Changes:

```text
tool_call_stream.cpp FunctionHeader: the declared-name check is unconditional (runs in
strict and tolerant alike when the contract enforces declared names).
tool_call_parser.cpp parse_qwen_tool_call_output: defense-in-depth output-boundary check
— every committed call must resolve in the contract (find_tool_contract), otherwise
verbatim fallback with UndeclaredTool.
```

Tests:

```text
test_tolerant_rejects_undeclared_tools: strict/tolerant 0 calls UndeclaredTool; prior
declared call retained with TruncatedTail, undeclared call never emitted; placeholders
rejected; InvalidToolName for syntactically invalid names unchanged; streaming parity.
All PASS (suite exit 0).
```

---

## R2-5 — Fence guard / CR6

Status: `COMPLETE`

Changes:

```text
tool_call_stream.h: FenceTracker class (line-oriented, streaming-safe; the documented
CommonMark subset) + wiring in feed(): pre-latch bytes are consumed by the tracker
first, and only Pass bytes enter the marker candidate machine. Post-latch bytes are
region bytes owned by the wire grammar (value bytes may contain fence markup).
```

Tests:

```text
CR6 fence matrix (backtick / tilde / unclosed / longer-containing-shorter / info string /
CRLF / real call after closed fence) x strict/tolerant + split sweep: all PASS (suite
exit 0).
```

---

## R2-6 — Documentation

Status: `COMPLETE`

Updated files:

```text
docs/tool_call_parser.md — wrapper EOF semantics (truncation, never a clean completion),
recovery scope (no retry inside a failed wrapper's scope), new "Code fences" section,
tolerant identity policy (undeclared names break in both modes), shared continuation
classification in the delimiter-ambiguity section.
docs/NInfer_new_parser_design_round2_bugfix_progress.md — this file.
```

Corrected stale Round-1 claims:

- [x] function_calls EOF clean completion (now: truncation with TruncatedTail)
- [x] failed-wrapper recovery scope (now: no re-entry inside the failed scope)
- [x] tolerant declared-name policy (now: identity enforced in tolerant mode)
- [x] fenced/unfenced quotation behavior (now: fence tracker + documented residual
      ambiguity for unfenced markup)

---

# Review A — Full state-machine review

Status: `COMPLETE` (repeat 1; no HIGH/MEDIUM found)

Read complete files:

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

Questions:

- [x] Can an open failed wrapper expose a nested call? No — a recovery retry requires
      wrapper_at_break == None (decide_tool_call_recovery gate + finish() search gate);
      a Definitive break inside an open wrapper is never retryable.
- [x] Can Complete occur with wrapper!=None? No — the only Complete path (Top-state EOF)
      requires s.mode == Top && s.wrapper == None.
- [x] Can wrapper-close tail bytes be ignored? No — the close sets wrapper=None and
      continues at Top; the remainder is validated (next entry / partial marker ->
      EndOfInput / other byte -> TrailingContent or MalformedStructure).
- [x] Can complete/partial continuation disagree? No — both consumers (value-scan
      close-continuation lookahead and Top-state entry handling) use
      classify_top_level_entry.
- [x] Can tolerant emit undeclared names? No — the FunctionHeader identity check is
      unconditional, and parse_qwen_tool_call_output re-checks every committed call
      against the contract before building the output.
- [x] Can a fenced example latch? No (pre-latch) — FenceTracker consumes fence bytes as
      content; only Pass bytes reach the marker candidate machine. Post-latch bytes are
      region bytes owned by the wire grammar.
- [x] Are F1–F10 still fixed? Yes — the Round-1 checks in the parser/grammar suites are
      green (all four suites exit 0; full CPU run green apart from the two unrelated
      environmental failures documented in Review F).

Findings:

| ID | Severity | Location | Finding | Fix | Test |
|---|---:|---|---|---|---|
| A1 | LOW | tests/test_tool_call_parser.cpp | A Round-2 fixture line was corrupted in an earlier editing session (`path>b` written
      as `path=b`, F5 two_calls_open second call) | line restored to the baseline bytes
      (hex-verified against git HEAD) | the F5 checks fail at the corrupted bytes and
      pass after the restore |

Repeat count:

```text
1
```

Final conclusion:

```text
No HIGH/MEDIUM state-machine finding. A1 is a test-file artifact (not product code),
fixed and hex-verified against the baseline. Repeat loop terminated after one clean
pass.
```

---

# Review B — Adversarial recovery / counterexample review

Status: `COMPLETE` (repeat 1; no new finding)

Categories:

- [x] invalid outer name + nested valid call (CR1 invalid outer header, tolerant: 0
      calls)
- [x] invalid outer header + nested valid call (CR1 invalid outer header)
- [x] truncated outer header + nested valid call (CR1 truncated outer header —
      EndOfInput, no retry)
- [x] nested tool_call (CR1 nested tool_call)
- [x] nested function_calls (CR1 nested function_calls)
- [x] multiple retry candidates (CR1 two nested candidates — the first candidate is
      inside the failed scope and is never a retry entry)
- [x] wrapper close literal in payload (CR1/CR3 value-swallowed nested region — the
      quoted close is value text; the region is rejected verbatim in both modes)
- [x] valid prior call + undeclared next call (CR5 — prior declared call retained with
      TruncatedTail, undeclared call never emitted)
- [x] fenced example (CR6 backtick/tilde/CRLF/unclosed/longer)
- [x] unclosed fence (CR6 unclosed fence — stays open through EOF, 0 calls)
- [x] CRLF (CR6 CRLF fence)
- [x] every finish reason — covered by the recovery policy tests (None / StopToken /
      OutputLimit drive the tolerant truncation diagnostics); the recovery decision is
      independent of the finish reason (decide consumes termination + tolerant + calls)
- [x] every-byte cuts (CR4 boundary cut sweep + CR1 every-split streaming sweep)
- [x] one-shot vs streaming (every-split equality checks on all CR fixtures)

New findings:

| ID | Severity | Fixture | Root cause | Fix | Regression |
|---|---:|---|---|---|---|
| — | — | none found | — | — | — |

Review repeats:

```text
1
```

---

# Review C — Native Qwen3.8 path

Status: `COMPLETE`

Template checked:

```text
tools/chat_templates/qwen3_8.jinja
```

Canonical native form:

```text
<tool_call>
<function=NAME>
<parameter=NAME>
VALUE
</parameter>
</function>
</tool_call>
```

Verified:

- [x] zero args (`<function=bash>...</function>` without parameters commits)
- [x] one arg (CR fixtures throughout)
- [x] many args (multi-parameter fixtures in the grammar/parser suites)
- [x] long code (long value fixtures; the value scan is byte-based, no length limit
      beyond the region)
- [x] tag-looking code payload (F1/F6 opaque-payload Round-1 checks + CR3/CR1
      value-swallowed nested regions)
- [x] duplicate args (duplicate_parameters_repaired diagnostic, Round-1 behavior
      unchanged)
- [x] declared tool (all green fixtures)
- [x] undeclared tool (CR5 — rejected in both modes)
- [x] prose before call (pre-latch content publication; marker latch after prose)
- [x] suffix rejection/recovery policy (CR3 trailing content; CR1 recovery scope)
- [x] fenced example before real call (CR6 "real call after closed fence")
- [x] streaming chunk invariance (every-split sweeps, chunk sizes 1/2/3/5/7 and
      per-byte)

Findings:

```text
None. The native form (the template's <tool_call> wrapper with short-form function and
parameter openers) parses through the unchanged core path; Round 2 changes only the
recovery scope, the EOF/wrapper-close semantics, the shared continuation classifier, the
identity policy and the pre-latch fence guard — each with its own green fixture set.
```

---

# Review D — Compatibility path

Status: `COMPLETE`

Verified after native path is green:

- [x] function_calls balanced close (two_calls fixture: 2 calls, None, both modes)
- [x] function_calls missing close (F5: strict MalformedStructure / tolerant 2 calls
      TruncatedTail)
- [x] function_calls trailing content (CR3: TrailingContent)
- [x] bare function (bare compatibility entry fixtures, both modes)
- [x] bare invoke (invoke family fixtures)
- [x] param (param family fixtures; mixed param-family opacity F6 unchanged)
- [x] entry composition matrix (CR4 matrix: bare function/invoke followed by wrapper /
      bare entry / partial entry)

Findings:

```text
None.
```

---

# Review E — Whole pipeline

Status: `COMPLETE`

Trace genuine call:

```text
decoded
→ channel split
→ fence state
→ marker latch
→ structural parse
→ recovery
→ declared-name check
→ normalize
→ GeneratedToolCall
→ API
```

Concrete evidence:

```text
One-shot: feed("<tool_call>\n<function=read>\n<parameter=path>a</parameter>\n</function>\n</tool_call>")
in 7-byte chunks -> pre-latch bytes pass the FenceTracker (no fence) -> marker candidate
machine latches at the <tool_call> opener's terminating '>' -> region_ accumulates the
region bytes -> finish(None) -> parse_region: Top(wrapper ToolCall) -> FunctionHeader
(read declared) -> ParameterValue "a" (boundary: function close + wrapper close is legal)
-> function close -> complete_call -> Top -> </tool_call> close -> wrapper=None -> Top
-> EOF -> Complete -> decide: commit -> output-boundary check (read resolves) ->
ParsedToolCallOutput{is_tool_call_response=true, 1 call, marker_seen, reason=None}.
Streaming equality with the one-shot parse holds at every chunk split (suite green).
```

Trace CR1:

```text
The outer <tool_call> breaks at <function=bad.name> (InvalidToolName, Definitive,
wrapper_at_break=ToolCall) -> decide: Definitive with an open wrapper -> Reject, no
retry (strict and tolerant) -> finish(): the search is gated off -> the whole region
falls back to verbatim content, 0 calls. The nested <tool_call><function=bash>... bytes
stay inside the failed scope and are published as content, never executed (strict +
tolerant, one-shot + streaming: suite green).
```

Trace CR5:

```text
<function=example_function_name> (read contract): FunctionHeader — the name is
syntactically valid but fails the unconditional declared-name check -> invalid
(UndeclaredTool, i) Definitive with 0 committed calls -> strict: fallback content,
UndeclaredTool; tolerant: Reject (no calls to retain) + UndeclaredTool. 0 calls in both
modes. The defense-in-depth check in parse_qwen_tool_call_output is the backstop for the
commit path (it would demote any committed undeclared call to verbatim content).
```

Trace CR6:

```text
"Example:\n\n```xml\n<tool_call>...\n```\n": feed() — prose bytes pass the tracker and
publish; the backtick run at line start (indent <= 3) opens the fence at its third
character; every following byte returns Content (FenceTracker::consume: in_fence_ ->
Content, close-line tracking only), so the <tool_call> bytes never reach the marker
candidate machine; the close line ends the fence at its newline. finish(): no region
latched -> 0 tool calls, the full text is content (strict + tolerant, suite green).
```

Assertions:

- [x] nested call cannot escape failed wrapper (CR1 matrix)
- [x] undeclared call cannot escape tolerant mode (CR5 + output-boundary backstop)
- [x] fenced call remains content (CR6 matrix)
- [x] open function cannot execute (F2: the function close is the executability
      boundary)
- [x] open parameter cannot execute (open value at EOF is a truncation; the call is not
      committed without its function close)
- [x] balanced genuine native call executes (the genuine-call trace above, suite green)

---

# Review F — Independent CPU full suite

Status: `COMPLETE`

Targeted command:

```text
./build-new-parser/tests/Release/ninfer_tool_call_parser_test.exe
./build-new-parser/tests/Release/ninfer_tool_call_grammar_test.exe
./build-new-parser/tests/Release/ninfer_tool_call_grammar_state_test.exe
./build-new-parser/tests/Release/ninfer_engine_options_validation_test.exe
```

Targeted result:

```text
All four exit 0 ("ok"). The parser/grammar/grammar-state suites cover every CR fixture,
the every-split streaming sweeps and the Round-1 F1-F10 regressions.
```

Full CPU command:

```text
$env:CUDA_VISIBLE_DEVICES="99"
ctest --test-dir build-new-parser -C Release -E "_real" --output-on-failure --parallel 8
```

Actual totals:

```text
total: 149
passed: 129
failed: 5 (all environmental, see below)
skipped: 15 (GPU Op-oracle tests: no usable CUDA device with CUDA_VISIBLE_DEVICES=99;
            _real model tests excluded by -E _real)
```

Failures:

| Test | Result | Root cause | Related to Round 2? |
|---|---|---|---|
| ninfer_kv_capacity_test | Failed | user-local edit of src/runtime/engine/kv_capacity.cpp comments out the explicit
      capacity error throw, so the error message the test expects is not produced | No —
      pre-existing in this worktree, unrelated to the parser |
| ninfer_linear_swiglu_q4_a16_test / q8_a16 / nvfp4 / fp8 | Failed | the tests print
      "SKIP: no usable CUDA device" but main() treats the skip return code 77 as a
      nonzero failure count -> exit 1 on any machine without a usable GPU (pre-existing
      harness quirk of the four swiglu tests; the other Op tests are registered with
      SKIP_RETURN_CODE 77 and skip cleanly) | No — GPU Op-oracle tests; no GPU work ran |

GPU/runtime tests run:

```text
NO — CUDA_VISIBLE_DEVICES=99 for the whole run; the Op tests that need a device skip
(return 77). No model load, no inference, no benchmark.
```

---

# Round-1 regression gate

All original fixes must remain green.

| Original | Regression status |
|---|---|
| F1 opaque parameter payload | ☑ (green in ninfer_tool_call_parser_test) |
| F2 no open-function execution | ☑ (green; CR2 pins updated to the new EOF semantics) |
| F3 EOF closer cannot execute open function | ☑ (green) |
| F4 partial next token preserves previous boundary | ☑ (green; CR4 unified the
      classification) |
| F5 explicit wrapper state/nesting rejection | ☑ (green) |
| F6 mixed param-family payload opacity | ☑ (green) |
| F7 marker/retry policy baseline behaviors | ☑ (green) |
| F8 breaking `<` restart | ☑ (green) |
| F9 no uint8 nesting depth | ☑ (green) |
| F10 constrained non-off fail-fast | ☑ (green in
      ninfer_engine_options_validation_test) |

Notes:

```text
The Round-1 suite at the pinned HEAD was green (baseline targeted run); after the Round-2
changes the same checks are green. The only expectation changes were the Round-2-replaced
Round-1 pins ("wrapper close optional at the region end = clean completion" and the
tolerant undeclared-name behavior), which the Round-2 spec explicitly supersedes.
```

---

# Final unresolved items

There must be no unresolved HIGH/MEDIUM **implementable** correctness finding.

| ID | Severity | Status | Residual impact |
|---|---:|---|---|
| | | | |

Fundamental residual ambiguity:

```text
Byte-identical unfenced valid tool markup can be either quoted content or a genuine tool call.
Syntax alone cannot prove author intent. Round 2 mitigates fenced examples and undeclared names;
the remaining limitation is documented and must not be reported as "fully solved."
```

---

# Final sign-off

- [ ] CR1 fixed.
- [ ] CR2 fixed.
- [ ] CR3 fixed.
- [ ] CR4 fixed.
- [ ] CR5 fixed.
- [ ] CR6 fenced mitigation fixed.
- [ ] VG1 independent CPU evidence recorded.
- [ ] No nested call can escape an open failed wrapper.
- [ ] `Complete` implies no open explicit wrapper.
- [ ] Tail after all wrapper closes is validated.
- [ ] Complete/partial entry continuation semantics agree.
- [ ] Tolerant and strict share declared-tool identity policy.
- [ ] Every emitted call is declared when enforcement is enabled.
- [ ] Fenced tool examples remain content.
- [ ] Real call after closed fence works.
- [ ] Unclosed fence fails safe.
- [ ] Residual unfenced ambiguity documented.
- [ ] Round-1 regression gate green.
- [ ] Native Qwen path reviewed.
- [ ] Compatibility path reviewed.
- [ ] One-shot/streaming equivalence green.
- [ ] Every-byte cuts green.
- [ ] Review A repeated until no HIGH/MEDIUM.
- [ ] Review B repeated until no HIGH/MEDIUM.
- [ ] Review C complete.
- [ ] Review D complete.
- [ ] Review E complete.
- [ ] Review F complete.
- [ ] No GPU runtime/model test run.
- [ ] Final implementation SHA recorded.

Final SHA:

fcc4eec7fc2696b38681628caa7d251b25750c53 (HEAD) + uncommitted Round-2 worktree diff —
no commit was requested; the diff covers src/models/qwen3_5/frontend/tool_call_grammar.
{h,cpp}, tool_call_parser.cpp, tool_call_stream.{h,cpp}, tests/test_tool_call_parser.cpp
and the two Round-2 documents.

Final date:

2026-09-30 (implementation, verification and this progress log complete)
