# NInfer `new_parser_design` — Round 8 Bugfix Progress

> Working checklist for:
>
> `NInfer_new_parser_design_round8_bugfix_implementation.md`
>
> Do not mark a finding VERIFIED until the dedicated reproducer and regression tests pass.

---

## 1. Baseline

Repository:

```text
Hundsbuah/ninfer
```

Branch:

```text
new_parser_design
```

Reviewed Round-8 baseline HEAD (pinned):

```text
ad71244de33212aa92742668502b29c3a72528a3
```

Actual implementation start SHA:

```text
ad71244de33212aa92742668502b29c3a72528a3
```

Worktree:

```text
clean (no local machine workarounds at implementation start)
```

Toolchain:

```text
MSVC (Visual Studio 18 Community, x64) + CUDA 13.1, vcpkg x64-windows (F:\GIT\vcpkg),
CMake generator "Visual Studio 18 2026", Release
```

Build directory:

```text
build-new-parser
```

GPU policy:

```text
NO model/GPU runtime tests for this task (local AI runs on the same GPU).
CPU-only parser/constraint/serve/schema/template tests; full CTest with
CUDA_VISIBLE_DEVICES=99.
```

---

## 2. Finding status

| Finding | Severity | Status | Commit(s) | Verification |
|---|---:|---|---|---|
| R8-01 constraint `CALL1 -> prose -> CALL2` divergence | MEDIUM | FIXED | `006bc52` (red) + `391bccf` + `73d4e2c` + `c095fc5` + `4c5b5d1` | 6 red assertions red->green; corpus A-N green; P1-P9 cross-check green (both syntaxes) |
| R8-01a `latched_once_` permanent exemption | MEDIUM root cause | FIXED | `391bccf` | field removed; lock independent of latch history; checkpoint green |
| R8-01b `marker_suffix()` / `rfind('<')` skip | MEDIUM root cause | FIXED | `73d4e2c` | helper deleted; replay from `progress.break_offset`; single-chunk partial case green |
| R8-02 Chat `required` -> Auto | MEDIUM | FIXED | `88c196e` (red) + `863072a` | 400 `tool_choice_not_supported` |
| R8-02b named function/custom -> Auto | MEDIUM | FIXED | `88c196e` (red) + `863072a` | 400 `tool_choice_not_supported` after name-syntax validation |
| R8-02c `allowed_tools.mode=required` -> Auto | MEDIUM | FIXED | `88c196e` (red) + `2cded6e` | 400 `tool_choice_not_supported` before mutating `output.tools` |
| R8-02d `parallel_tool_calls=false` advisory | MEDIUM | FIXED | `88c196e` (red) + `2cded6e` | 400 `parallel_tool_calls_not_supported` when `uses_tools()`; neutral otherwise |
| R6-04 byte-identical call-only residual | residual | DOCUMENTED | n/a | unchanged (`docs/tool_call_parser.md`) |
| constrained runtime decoding | non-goal | KEPT FAIL-FAST | n/a | unchanged startup error |

---

## 3. R8-01 red baseline

Red commit: `006bc52` (test only, production code at `ad71244d`).

Baseline run of `ninfer_tool_call_grammar_state_test`:

```text
FAIL: r8 soc 6: prose after a completed call locks the gate        (flipped R7 case 6)
FAIL: r8 E: prose after CALL1 keeps CALL2 from triggering          (split)
FAIL: r8 E: the locked gate keeps a later marker content           (split)
FAIL: r8 J: CALL2 and any later marker stay content                (same commit)
FAIL: r8 I: the failed candidate keeps later markers content       (partial failed marker)
FAIL: r8 M: the draft prose locks the draft gate                   (checkpoint)
```

Controls green on baseline: A (pre-latch prose), B (gate open after closed region),
H (partial immediate CALL2 resumes), TC (TemplateCompatible unchanged).

The R8-01b single-chunk case (`commit(CALL1 + prose + "<tool_")` then `commit("call>")`)
failed additionally on the post-`391bccf` code, proving the `marker_suffix()` skip is a
separate defect from the `latched_once_` exemption:

```text
FAIL: r8 01b: the replayed prose keeps CALL2 from triggering
```

The contradictory R7-02 test expectation (case 6: "prose after a latched region does not
re-lock") was replaced by the R8-01 expectation, not kept beside it.

---

## 4. R8-01 implementation

Commits:

```text
391bccf  fix(constraint): re-lock after visible content following completed tool region
73d4e2c  fix(constraint): replay trailing content from parser break offset
```

Changes (`src/models/qwen3_5/frontend/tool_call_grammar_state.{h,cpp}` only; the
production Stage-1/2/3 parser is unchanged):

```text
- removed `latched_once_` (R8-01a): visible non-formatting-whitespace content now locks
  the gate at any point under RequireToolAtContentStart (before the first entry or after
  a completed call); a failed marker candidate likewise locks whenever the hardened
  intent is active (R8-I2)
- the `Complete` transition clears the stale `marker_suffix(combined)` candidate:
  retaining the closing tag as a future candidate would flush it as a failed candidate
  and lock the gate for a directly consecutive wrapper (Round 8 §3.6)
- the `TrailingContent` transition replays `combined.substr(progress.break_offset)`
  through the shared inactive scanner instead of jumping to the last `<`
  (R8-01b, Round 8 §3.7); the stale `marker_suffix()` helper was deleted (no remaining
  callers)
- the inactive scanner was factored into one private static helper
  `advance_inactive(state, bytes)` used by both ordinary inactive bytes and the
  post-call replay (Round 8 §3.8): it owns formatting-whitespace handling, marker
  start/continuation, candidate classification, the failed-candidate breaking-`<`
  rescan, the hardened text lock, and the complete-marker trigger (no recursive
  `advance()`, Round 8 §3.9)
- stale comments claiming the first latch permanently satisfies the content-start
  requirement were corrected (Round 8 §3.22)
```

Invariants preserved:

```text
- R8-I4: directly consecutive wrappers (no gap or formatting-whitespace gap) keep the
  second entry eligible
- R8-I6: visible prose after a completed call locks the gate and never yields Rejected
- R8-I5: all gate state words are value-semantic (checkpoint/restore)
- R8-I8: TemplateCompatible semantics unchanged (the gate is SOC-only)
```

---

## 5. R8-01 tests

Commits:

```text
006bc52  test(constraint): reproduce post-call hardened intent divergence (red)
c095fc5  test(constraint): post-call chunk/checkpoint matrix
4c5b5d1  test(frontend): parser/constraint post-call eligibility cross-check
```

`tests/test_tool_call_grammar_state.cpp`:

```text
- R7 case 6 flipped: CALL1 + prose + marker must NOT trigger (R8-01)
- test_r8_constraint_post_call_gate: corpus A, B, E (split), J (same commit), I (partial
  failed marker), H (partial immediate CALL2 control), M (checkpoint after CALL1),
  R8-01b (single-chunk CALL1 + prose + partial CALL2), TC control
- test_r8_post_call_chunk_checkpoint_matrix: C/D/K (no gap, mixed whitespace, CRLF) in
  whole-chunk + bytewise + split form with agreeing final states; E/L (prose and form
  feed — pinned as visible content, not format whitespace) whole-chunk + bytewise; N
  (checkpoint inside a partial second marker: failed draft locks, restore keeps the
  marker completable); QwenWrappedNative x SOC cross
```

`tests/test_tool_call_parser.cpp`:

```text
- test_r8_parser_constraint_post_call_cross_check: corpus P1-P9 under QwenWrappedNative
  and Compatibility — for each entry the production parser outcome (call count) and the
  constraint second-entry trigger outcome are recorded and asserted to agree with no
  contradiction (eligible == (calls >= 1)); P6 (partial second entry) pins parser 0
  calls + constraint legal prefix. Production parser expectations are unchanged from
  the R7 corpus (R + "\n" + B = 2 calls; R + prose + B = 0 calls verbatim).
```

Every new constraint fixture runs both whole-chunk and bytewise commits (Round 8 §7).

---

## 6. R8-02 red baseline

Red commit: `88c196e` (test only; `src/serve/openai_chat_request.cpp` at `ad71244d`).

Baseline run of `ninfer_openai_schema_test`:

```text
FAIL: required tool choice is rejected as an unguaranteeable guarantee
FAIL: named function tool choice is rejected as an unguaranteeable forcing
FAIL: named custom tool choice is rejected as an unguaranteeable forcing
FAIL: required allowed_tools is rejected as an unguaranteeable guarantee
FAIL: parallel_tool_calls=false with callable tools is rejected
```

Neutral cases stayed green on baseline and after the fix: `tool_choice:none` +
`parallel_tool_calls:false` (executable tools removed), `parallel_tool_calls:false`
without tools, `tool_choice:auto` without tools.

---

## 7. R8-02 implementation

Commits:

```text
863072a  fix(chat): reject required and named tool choices
2cded6e  fix(chat): reject allowed_tools required and parallel false guarantee
```

Changes (`src/serve/openai_chat_request.cpp` only; the Responses adapter is unchanged):

```text
- tool_choice "required": 400 param=tool_choice code=tool_choice_not_supported
  (was: silent Auto)
- named function choice: structural shape + name-syntax validation preserved, then
  400 tool_choice_not_supported (was: silent Auto; a one-tool set still permits no
  tool call, so forcing is unguaranteeable — Round 8 §4.10, minimal option §4.16)
- named custom choice: same rule
- allowed_tools mode=required: 400 tool_choice_not_supported before mutating
  output.tools (Round 8 §11); mode=auto keeps the subset filter
- parallel_tool_calls=false + output.uses_tools(): 400
  param=parallel_tool_calls code=parallel_tool_calls_not_supported (wording mirrors
  the existing Responses rejection); neutral without effective tools
- ToolChoiceMode stays Auto/None (Round 8 §4.12): unsupported wire values die at the
  API boundary
```

Error precedence (Round 8 §4.15) preserved: numeric `tool_choice` -> invalid type;
missing function/custom object -> malformed `tool_choice`; invalid name syntax ->
`invalid_tool_name`; only structurally valid unsupported guarantees return
`tool_choice_not_supported`.

---

## 8. R8-02 tests

Commits:

```text
88c196e  test(chat): pin unsupported forced tool-control requests (red)
df8f044  test(http): verify Chat capability error payloads
```
`tests/test_openai_schema.cpp`: the four advisory assertions were replaced by exact
`param`/`code` capability-error assertions (required, named function, named custom,
allowed_tools required, parallel false + callable tools); the neutral cases remain
accepted. `tests/test_http_routes.cpp`: `test_r8_chat_capability_error_payload` parses
an actual `POST /v1/chat/completions` body with `tool_choice:required` through
`parse_chat_completion_request` (the handler's parse path — no generation starts),
verifies the `ApiError` state (400 / invalid_request_error / param / code), and the
rendered `make_error_body` JSON shape:

```json
{"error": {"type": "invalid_request_error", "param": "tool_choice",
           "code": "tool_choice_not_supported"}}
```

---

## 9. Targeted CPU test run

Command (build-new-parser, Release, 16 threads):

```text
ctest -R "ninfer_(tool_call_parser|tool_call_grammar|tool_call_grammar_state|qwen3_5_frontend|openai_schema|openai_responses|http_routes|serve_options|request_log)_test" --parallel 16
```

```text
selected:  9
passed:    9
failed:    0
skipped:   0
exit code: 0
```

Failures:

```text
none
```

---

## 10. Full GPU-hidden CTest

Command:

```text
CUDA_VISIBLE_DEVICES=99 ctest -E "_real" --output-on-failure --parallel 16
```

Result (recorded after the run completed):

```text
selected:  147
passed:    46
failed:    0
skipped:   101   (GPU op/model tests hidden by CUDA_VISIBLE_DEVICES=99)
exit code: 0
```

---

## 11. Regression review (review loops A-F)

- [x] A — inactive intent state: one shared `advance_inactive` authority; no `HadTool`
  exception; whitespace / marker prefix / visible content / locked gate handled
  uniformly for ordinary bytes and replay bytes
- [x] B — active-to-inactive transitions: `Complete` (gate open, stale candidate
  cleared), `TrailingContent` (break-offset replay), `EndOfInput` (legal prefix
  retained), structural break inside an open region (`Rejected`)
- [x] C — stale marker state: `marker_suffix` / `rfind('<')` deleted; a completed
  closing tag is not retained as a future candidate (Complete clears
  `marker_prefix_`); the first byte after a completed region is processed normally
- [x] D — parser/constraint semantics: P1-P9 cross-check under both syntaxes; no
  contradiction (the contradictory R7-02 expectation was replaced, not kept)
- [x] E — Chat silent-weakening search: no standard Chat forcing request remains
  mapped to Auto; the Anthropic `tool_choice:any` / `disable_parallel_tool_use`
  advisory behavior is a separate endpoint (explicit non-goal, unchanged)
- [x] F — Responses regression: `openai_responses_request.cpp` untouched;
  `ninfer_openai_responses_test` green

Previous-round regression set re-run (via the full GPU-hidden gate): F1-F8, CR1-CR4,
FailClosed, FinishReason transaction, native/compat syntax, R6 pre-latch intent,
R7 later-base retry cutoff, R7 decoder intent propagation, R7 template split —
covered by the green targeted + full suites.

---

## 12. Outcome dumps

Constraint dump (`RequireToolAtContentStart`, the only changed dimension):

```text
+ completed prior call + visible content + later marker:
  second trigger: true -> false   (E, J, I, 01b, L)
+ completed prior call + formatting whitespace + later marker:
  second trigger: true -> true    (C, D, H, K — unchanged)
+ checkpoint/restore post-call:
  exact eligibility recovered     (M, N — unchanged semantics, now value-complete)
No production parser delta (parser corpus unchanged, cross-check green).
```

Chat request dump (only unsupported-guarantee cases change):

```text
auto                       accepted (unchanged)
none                       accepted (unchanged)
required                   400 tool_choice_not_supported        (was: advisory auto)
named function             400 tool_choice_not_supported        (was: advisory auto)
named custom               400 tool_choice_not_supported        (was: advisory auto)
allowed auto               accepted, subset filter (unchanged)
allowed required           400 tool_choice_not_supported        (was: advisory auto)
parallel true              accepted (unchanged)
parallel false + tools     400 parallel_tool_calls_not_supported (was: advisory)
parallel false + none      accepted neutral (unchanged)
parallel false + no tools  accepted neutral (unchanged)
```

---

## 13. Known residuals (unchanged)

- R6-04: a byte-identical complete call-only output can still mean "execute" or
  "show as an example" when no distinct intent signal exists; neither `FailClosed`,
  `RequireToolAtContentStart`, nor the syntax constraint can infer intent from
  identical bytes. The long-term structural TEXT-vs-TOOL intent channel remains
  outside ordinary quotable content.
- Runtime constrained tool decoding remains unavailable and fail-fast at startup.
- Round 8 fixes one prerequisite of the constrained-decoding activation checklist
  (intent parity including post-call behavior); the remaining checklist items are
  unchanged.

---

## 14. Sign-off

Wording if the gates are green:

> No remaining HIGH/MEDIUM implementation defect was found in the reviewed active
> Qwen3.8 native parser path or the reviewed parser/constraint intent-policy boundary
> under the CPU-only verified scope. The grammar constraint now ends hardened
> tool-sequence eligibility when visible content follows a completed call while
> preserving directly consecutive wrappers. Chat Completions no longer silently
> weakens unsupported forced tool-control requests to automatic selection. Runtime
> constrained tool decoding remains unavailable and fail-fast. The complete call-only
> quotation ambiguity remains a documented semantic limitation of the wire protocol.

Reviewer notes: the constraint change is contained to
`tool_call_grammar_state.{h,cpp}` (no production parser delta); the Chat change is
contained to `openai_chat_request.cpp` (Responses adapter untouched). The
contradictory R7-02 expectation was replaced by the R8-01 one; no other
production-path regression was observed in the CPU-only scope.
