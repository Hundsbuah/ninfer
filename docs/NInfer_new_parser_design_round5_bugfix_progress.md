# NInfer `new_parser_design` — Round 5 Bugfix Progress

## Baseline
- start SHA: `705c45e39261b44fa6dd9bea54d6f8d8ea901fad` (`fix: complete merge interface (envelope wide_verification + dedupe log_colour test)`)
- Round-5 spec SHA/path: `docs/NInfer_new_parser_design_round5_bugfix_implementation.md` (untracked, working tree)
- upstream architecture parent: `ae779f1d464de81aff36f9bd6303c7e3446630c9` (`Wallawalla47/ninfer-custom:master`)
- build directory: `build-new-parser` (not `build-windows`)
- toolchain: MSVC via CMake `Visual Studio 18 2026` generator, Release, 16 build / 16 test threads; vcpkg `F:/GIT/vcpkg` (x64-windows)
- GPU tests: NOT RUN (local AI occupies the GPU; CPU-only round)

## Finding status
| ID | Status | Commit | Tests | Notes |
|---|---|---|---|---|
| R5-01 | FIXED | `321965bb` | `ninfer_qwen3_5_frontend_test` (terminal-reason matrix, cancellation/cut regressions) | terminal reason owned by the OutputSession preview transaction; reasonless `commit_preview()` restored; `ToolCallOutputDecoder::finish(finish_reason)` fed the stored reason |
| R5-02 | FIXED | `321965bb` | same as R5-01 + `41c881a9` | the decoder `finish` API now carries the reason explicitly; no call site can terminalize with a silently defaulted reason |
| R5-03 | FIXED | `d1b9f776` | `ninfer_request_log_test` | request-log serialization restored the Round-4 tool-call diagnostics |
| R5-04 | FIXED | `d1b9f776` | `ninfer_request_log_test` (restored assertions, `85e81ccb`) | operational-log tool-call diagnostics restored |
| R5-05 | FIXED | `fc7225dd` | `ninfer_serve_options_test` (help text) | serve help and docs corrected for tolerant/fallback semantics |
| R5-06 | FIXED | `0a9cce21` | `test_round4_r1_residual_pinned` (FailClosed rows), `test_r5_ambiguity_policy_write_payload`, `ninfer_serve_options_test` flag rows | `ToolCallAmbiguityPolicy` (PayloadFidelity / FailClosed); FailClosed default at the product boundary (`OutputOptions`, `--tool-call-ambiguity`); a Stage-2 value boundary that stands while an earlier closer chain had already formed a complete call is refused — region as text, `ambiguous_structure`, no call executes; balanced nested payloads are not ambiguous (nesting decides, not the policy) |
| R5-07 | FIXED | `ee4ec149` | `test_r5_syntax_mode_native_vs_compatibility`, `ninfer_qwen3_5_frontend_test`, `ninfer_serve_options_test` | `ToolCallSyntaxMode` (QwenWrappedNative / Compatibility); native default at the product boundary (`OutputOptions`, `--tool-call-syntax`); compatibility remains the low-level entry default |
| R5-08 | FIXED | `fc7225dd` | n/a (documentation) | constrained-decoding docs no longer overstate runtime protection

## Step results
### Step 0 baseline
- targeted suites (16 threads): 8/8 PASSED — `ninfer_tool_call_parser_test`, `ninfer_tool_call_grammar_test`, `ninfer_tool_call_grammar_state_test`, `ninfer_qwen3_5_frontend_test`, `ninfer_request_log_test`, `ninfer_pretty_logging_test`, `ninfer_serve_options_test`, `ninfer_engine_options_test` (the spec's `ninfer_engine_options_validation_test` does not exist as a target; the engine-options validation assertions live in `ninfer_engine_options_test`)
- full CPU gate: not run at baseline (run at final)
- outcome dump: `build-new-parser/round5/outcomes_step0.txt`, 36306 lines, sha256 `7a4130c4327433c2` (first 16 hex chars); guard tool built from the four frontend sources + `build-new-parser/round5/outcome_dump.cpp` (Round-4 Appendix-B corpus, unchanged)

### Step 1 terminal transaction (R5-01/R5-02)
- commit: `321965bb`
- code changes: `output_session.{h,cpp}` — the preview transaction now stores and validates the terminal reason; `commit_preview()` is reasonless again; the decoder `finish` receives the stored reason; metadata cleared after commit/rollback.
- focused tests: `ninfer_qwen3_5_frontend_test` PASSED (16 threads)
- outcome-dump delta: none expected for the direct parser dump (parser core untouched)

### Step 2 finish-reason matrix tests (R5-01/R5-02)
- commit: `41c881a9`
- focused tests: `ninfer_qwen3_5_frontend_test` PASSED
- outcome-dump delta: none

### Step 3 log diagnostics (R5-03/R5-04)
- commit: `d1b9f776`
- code changes: `operational_log.cpp`, `request_log.cpp` — tool-call diagnostics restored in both logs.
- focused tests: `ninfer_request_log_test` PASSED
- outcome-dump delta: none (parser untouched)

### Step 4 log regression tests (R5-03/R5-04)
- commit: `85e81ccb`
- focused tests: `ninfer_request_log_test`, `ninfer_serve_options_test` PASSED
- outcome-dump delta: none

### Step 5 docs/help (R5-05/R5-08)
- commit: `fc7225dd`
- code changes: `docs/serving.md`, `docs/new_parser_phase4_design.md`, `docs/tool_call_parser.md` — tolerant, fallback and constrained-decoding wording corrected.
- focused tests: n/a (docs); help assertions in `ninfer_serve_options_test` PASSED
- outcome-dump delta: none (parser untouched; byte-identical to the step-0 dump, verified at final)

### Step 6 native vs compatibility syntax (R5-07)
- commit: `ee4ec149`
- code changes: `types.h` (`ToolCallSyntaxMode`, `OutputOptions.tool_call_syntax`), grammar entry-set selection (streaming pre-latch, marker scan, Stage-1 entry, Stage-3 recovery), parser entry/decoder plumbing, `OutputSession` wiring, `--tool-call-syntax qwen-wrapped|compat` serve flag.
- focused tests: `ninfer_tool_call_parser_test`, `ninfer_qwen3_5_frontend_test`, `ninfer_serve_options_test` PASSED
- outcome-dump delta: compatibility-mode dump unchanged; native mode rejects compatibility-only top-level entries (documented in `docs/tool_call_parser.md`)

### Step 7 explicit ambiguity policy (R5-06)
- commit: `0a9cce21`
- code changes: `types.h` (`ToolCallAmbiguityPolicy`, `OutputOptions.tool_call_ambiguity = FailClosed`); `tool_call_stream.{h,cpp}` (policy `ambiguity`; `Stage2Verdict::Ambiguous`; `run_stage2_base` tracks an earlier complete-call candidate per value and returns `Ambiguous` when a later candidate stands under FailClosed; the finish path converts `Ambiguous` into a whole-region `AmbiguousStructure` failure before any later base or Stage-3 recovery); `tool_call_parser.{h,cpp}` (entry + decoder parameter, internal default PayloadFidelity); `output_session.cpp` (product-boundary default); `serve_options.{h,cpp}` (`--tool-call-ambiguity fail-closed|payload-fidelity` + help); `translate.cpp` (propagation).
- focused tests: `ninfer_tool_call_parser_test`, `ninfer_qwen3_5_frontend_test`, `ninfer_serve_options_test` PASSED (16 threads)
- regression found and fixed before commit: the first `run_stage2_base` edit accidentally dropped the `path.lazy_used` flag in the candidate-stand branch, which broke the R3-R2 and N-05 tolerant-commit tests; restored the flag (after the ambiguity check) and the full parser suite is green.
- outcome-dump delta: compat+payload byte-identical to the step-0 baseline (0 of 36306 lines); compat+fail-closed differs on exactly 2 lines — the two realistic document-write payloads contain the R1 ambiguity class and are refused with `ambiguous_structure` (the documented trade-off); native quadrants carry the R5-07 syntax deltas. Final dump: `build-new-parser/round5/outcomes_final.txt` (145224 lines, 2×2 policy quadrants; compat+payload quadrant sha256 `7cdf6630985ed897` = step-0 quadrant).

### Step 8 docs + progress (R5-06/R5-07 documentation)
- commit: this commit (step-8 docs) on top of `0a9cce21`
- docs: `docs/tool_call_parser.md` (ambiguous-byte-protocol section, including what nesting vs policy decides); `docs/serving.md` (`--tool-call-ambiguity` row); this progress file completed.

## Intentional outcome changes
- syntax-policy deltas: the `qwen-native` quadrants of the final outcome dump differ on 18715 (payload) / 18717 (fail-closed) of 36306 corpus lines: compatibility-only top-level entries (bare `<function=...>`, `<invoke=...>`, `<function_calls>`) are no longer accepted. The compatibility quadrants are unchanged.
- ambiguity-policy deltas: compat+fail-closed differs on exactly 2 corpus lines (the two document-write payloads of the R1 ambiguity class → `ambiguous_structure`, no call). compat+payload is byte-identical to the step-0 baseline; the low-level entry keeps PayloadFidelity as its default, so the pinned R1 fixture remains executable under the historical policy.

## Unresolved residuals
- protocol ambiguity: the byte protocol is ambiguous by construction (no escaping/framing for string values); the policy now makes the choice explicit and FailClosed is the production default. No residual finding.
- constrained decoding runtime: out of scope this round (R5-08 was documentation only).
- any unrelated CPU failures: none. Four failures in the full gate are GPU-dependent kernel tests (`ninfer_linear_swiglu_{q4_a16,q8_a16,nvfp4,fp8}`), each printing `SKIP: no usable CUDA device` — outside this round per spec §8.2 and consistent with the no-GPU constraint (GPU occupied by the local AI).
- guard-tool note: the outcome-dump signature does not carry `parse_budget_exhausted` (inherited from the Round-4 tool); the budget path is covered by the 86-state frontend matrix instead.

## Final sign-off
- final SHA: this step-8 docs commit, on top of `0a9cce21` (step-7 code)
- targeted suites: 8/8 PASSED — `ninfer_tool_call_parser_test`, `ninfer_tool_call_grammar_test`, `ninfer_tool_call_grammar_state_test`, `ninfer_qwen3_5_frontend_test`, `ninfer_request_log_test`, `ninfer_pretty_logging_test`, `ninfer_serve_options_test`, `ninfer_engine_options_test` (16 threads, build directory `build-new-parser`)
- full CPU gate: `ctest -C Release -E "_real" --parallel 16`, `CUDA_VISIBLE_DEVICES=99`: 147 tests — 46 PASSED (CPU), 97 SKIPPED (`no usable CUDA device`), 4 FAILED (GPU-dependent `ninfer_linear_swiglu_*`, out of scope per spec §8.2)
- GPU/model tests: NOT RUN
- remaining HIGH/MEDIUM findings: none — R5-01 through R5-08 all FIXED

