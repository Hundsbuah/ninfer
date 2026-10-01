# NInfer `new_parser_design` — Round 7 Bugfix Progress

> Working checklist for:
>
> `NInfer_new_parser_design_round7_bugfix_implementation.md`
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

Reviewed Round-7 start SHA:

```text
4a69b7a31b6570bfd7fce220a822492b24a161e4
```

Actual implementation start SHA:

4a69b7a31b6570bfd7fce220a822492b24a161e4

Worktree:

```text
dirty — local machine workarounds only (CMakeLists.txt, build_native.bat,
materialization_budget.h, kv_capacity.cpp); no parser-related local edits
```

Toolchain:

```text
MSVC (Visual Studio 2022, x64) + CUDA 13.1, vcpkg x64-windows (F:\GIT\vcpkg),
CMake generator "Visual Studio 17 2022", Release
```

Build directory:

```text
build-new-parser
```

GPU policy:

```text
NO model/GPU runtime tests for this task.
CPU-only parser/frontend/template/serve/constraint tests.
```

GitHub CI at reviewed start SHA:

```text
combined status: none
workflow runs: none
```

---

## 2. Finding status

| Finding | Severity | Status | Commit(s) | Verification |
|---|---:|---|---|---|
| R7-01 hardened intent later-base retry bypass | HIGH | FIXED | `24355cfc` (red) + `7565c40b` (fix) + `6fa91ac` (matrix) | C1-C10 red→green; cross-matrix green; streaming==one-shot |
| R7-01b decoder terminal re-parse drops `intent_` | HIGH coupling | FIXED | `3037d93d` (fix + parity test) | decoder/one-shot parity green |
| R7-02 grammar constraint not intent-aware | MEDIUM architecture | FIXED (strategy A) | `73187827` (fix + matrix test) + `3a8e742e` (frontend test) | constraint matrix green; suites green |
| R7-03 hardened template instruction unconditional | MEDIUM | FIXED | `5236026` | Python template suite 7/7 + C++ render parity |
| R7-04 Round-6 CTest sign-off / skip propagation | LOW / VERIFY | FIXED | `0408781` | all four wrappers exit 77 GPU-hidden; audit clean |
| R6-04 byte-identical call-only residual | residual | RESIDUAL | n/a | documented |
| R6-07 runtime constrained decoding | non-goal | DEFERRED | n/a | fail-fast |

---

## 3. Baseline red repro — R7-01

### Test A

```text
valid call
+ prose
+ later valid call
```

Policies:

```text
TemplateCompatible
RequireToolAtContentStart
```

Baseline observed:

```text
`test_r7_hardened_intent_blocks_later_retry_base` FAILED on baseline
(`24355cfc`): RequireToolAtContentStart executed the later retry base
instead of rejecting the turn as content.
```

Expected after fix:

```text
TemplateCompatible:
  preserve historical behavior

RequireToolAtContentStart:
  0 calls
  entire bytes restored as Content
```

### Test B

```text
<tool_call>
ordinary prose
<tool_call>valid later call...</tool_call>
```

Baseline observed:

```text
same as Test A — the second (later) marker was re-latched and executed
```

Expected hardened after fix:

```text
0 calls
verbatim fallback
```

---

## 4. R7-01 implementation

Commit:

```text
`7565c40b` (fix), `24355cfc` (red repro), `6fa91ac` (extended matrix)
```

Implementation:

```text
src/models/qwen3_5/frontend/tool_call_stream.cpp (finish retry chain, ~line 1236):
under RequireToolAtContentStart the retry attempt chain is cut to the initially
latched entry only; later wrapper bases never become executable attempts.
```

Proof:

```text
RequireToolAtContentStart admits only the originally latched entry.
Later retry bases never enter the executable attempt chain.
```

Tests:

- [x] Stage-1 later retry blocked
- [x] prose_after_wrapper retry blocked
- [x] base-zero Stage 2 preserved
- [x] base-zero tolerant recovery preserved
- [x] later-base tolerant recovery blocked
- [x] TemplateCompatible unchanged
- [x] bytewise streaming == one-shot
- [x] API serialization has no tool item on hardened rejection

Result:

```text
ninfer_tool_call_parser_test: ok (C1-C10 + cross-matrix green, GPU-hidden)
```

---

## 5. R7-01b implementation

Commit:

```text
`3037d93d`
```

Change:

```text
ToolCallOutputDecoder::finish()
passes intent_ to parse_qwen_tool_call_output(...)
```

Parity test:

```text
test_r7_decoder_intent_parity (tests/test_tool_call_parser.cpp): the decoder
terminal re-parse must equal the one-shot parse for both intent policies
```

Result:

```text
ninfer_tool_call_parser_test: ok (decoder/one-shot parity green, GPU-hidden)
```

---

## 6. R7-02 constraint intent parity

Decision:

```text
IMPLEMENT NOW (strategy A): the constraint mirrors the parser's one-shot
pre-latch intent gate (latched_once_ + entry_locked_), value-semantic state
```

If implemented:

- [x] constructor accepts intent
- [x] state stores intent (+ latched_once_)
- [x] hardened locks after visible content
- [x] whitespace keeps gate open
- [x] failed marker candidate locks
- [x] checkpoint/restore preserves intent state
- [x] consecutive wrappers match parser (first latch satisfies the gate)
- [x] syntax × intent cross-matrix green

Evidence:

```text
test_r7_intent_content_start_gate (8 cases, tests/test_tool_call_grammar_state.cpp)
+ test_r7_reasoning_plus_hardened_retry_rejection (frontend OutputSession, commit
`3a8e742e`): all green GPU-hidden; parser/grammar/frontend suites pass
```

---

## 7. R7-03 template/runtime policy

Decision:

```text
two explicit templates (option A), no hidden rendering switch
```

Recommended:

```text
qwen3_8.jinja                  upstream-compatible
qwen3_8_hardened_tools.jinja   start-of-content prompting
```

Checks:

- [x] compatible template allows optional pre-call natural-language reasoning
- [x] hardened template forbids visible preamble/suffix
- [x] both compile in Python/Jinja
- [x] both compile/render in C++ frontend (ninfer_jinja_test parity)
- [x] tools JSON unchanged
- [x] reasoning effort unchanged
- [x] history rendering unchanged
- [x] serving docs explain parser flag vs prompt template

Result:

```text
tests/text/test_chat_templates.py: 7/7 OK (incl. test_tool_instruction_policies
and C++ render parity over both templates)
```

---

## 8. R7-04 test-gate truthfulness

Round-6 progress corrected:

- [x] no longer says full gate green with 4 CTest failures
- [x] explains console "SKIP" vs actual process exit code
- [x] names LinearSwiGLU wrapper conversion `77 -> 1`

Optional code fix:

- [x] q4_a16 preserves 77
- [x] q8_a16 preserves 77
- [x] nvfp4 preserves 77
- [x] fp8 preserves 77
- [x] repository audited for same anti-pattern
- [x] CTest sees skipped tests as skipped

Result:

```text
all four wrappers print "SKIP: no usable CUDA device" and exit 77 GPU-hidden;
full gate now reports them Skipped; repo audit: other op tests already return 77
directly from main()
```
## 9. Targeted CPU test run

Command:

```powershell
ctest --test-dir build-new-parser -C Release `
  -R "ninfer_(tool_call_parser|tool_call_grammar|tool_call_grammar_state|qwen3_5_frontend|serve_options|request_log|chat_templates|engine_options)_test" `
  --output-on-failure `
  --parallel 16
```

Observed:

```text
8 selected
7 passed
0 failed
1 skipped (ninfer_engine_options_test, GPU-dependent)
exit code 0
```

Failures:

```text
none
```

---

## 10. Full GPU-hidden gate

Command:

```powershell
$env:CUDA_VISIBLE_DEVICES="99"

ctest --test-dir build-new-parser -C Release `
  -E "_real" `
  --output-on-failure `
  --parallel 16
```

Observed:

```text
147 total
46 passed (CPU tests executed)
0 failed
101 skipped (GPU-dependent, via SKIP_RETURN_CODE 77)
0 not run (CTest exit code 0 — gate is green)
```

Do not call this green unless CTest exits successfully.

---

## 11. Regression review

- [x] F1 literal parameter opener payload
- [x] F2 open function non-executable
- [x] F3 EOF parameter ambiguity
- [x] F4 partial next opener
- [x] F5 wrapper balance
- [x] F6 mixed parameter forms
- [x] F8 breaking `<`
- [x] CR1 failed-wrapper scope
- [x] CR2 function_calls missing close
- [x] CR3 trailing content
- [x] CR4 continuation consistency
- [x] declared tool enforcement
- [x] fence suppression
- [x] FinishReason transaction
- [x] FailClosed structural ambiguity
- [x] R6 syntax-mode constraint
- [x] R6 pre-latch intent gate
- [x] R7 retry intent invariant

Result:

```text
all covered: the full GPU-hidden gate (147/147, exit 0) runs the parser,
grammar, grammar-state, frontend, serve-options, request-log, chat-template,
and engine-option targets; no regression in the F1-F8/CR1-CR4/FailClosed/
syntax-mode/pre-latch-gate behaviors
```

---

## 12. Known residuals

### R6-04

```text
A complete declared canonical call-only byte sequence is still
semantically indistinguishable from a genuine action without an external
TEXT-vs-TOOL intent signal.
```

Status:

```text
RESIDUAL — not a parser implementation defect.
```

### Constrained decoding

```text
Runtime constrained sampling remains unavailable.
```

Status:

```text
DEFERRED / fail-fast.
```

---

## 13. Final sign-off

Only use if supported by actual results:

```text
No remaining HIGH/MEDIUM implementation defect was found in the reviewed
canonical Qwen3.8 parser/integration path under the CPU-only verified scope.

RequireToolAtContentStart now constrains retry/recovery entry selection as
well as the initial marker latch; streaming and one-shot terminal parsing
carry the same intent policy.

The call-only semantic quotation residual remains documented. Runtime
constrained decoding remains unavailable and fail-fast.
```

Do not use:

```text
bug-free
all tool-call errors solved
phantom calls impossible
```

Final reviewer notes:

```text
CPU-only verified scope; no GPU runtime execution per user constraint.
The four GPU-dependent LinearSwiGLU wrappers now skip correctly (exit 77).
The call-only semantic quotation residual (R6-04) and the lack of runtime
constrained decoding remain as documented.
```
