# NInfer `new_parser_design` — Round 6 Bugfix Progress

## Baseline

```text
start SHA:        8b4c4e9904d1347cc73769954273184cb97d287c
                  (= Round-6 spec pinned baseline; code parent fe3490da3bdb1bcac5609ab07dbf5eaf65b712f5)
branch:           new_parser_design (Hundsbuah/ninfer)
worktree:         clean (no modified tracked files; untracked: round-6 spec, graphify-out/,
                  pre-existing local artifacts ninfer.requests.jsonl / vc140.pdb / review_probe.cpp)
toolchain:        MSVC, Visual Studio 2026 (x64), CMake generator "Visual Studio 20 2026" x64,
                  CUDA toolset 13.4 (compile only; no CUDA execution)
build dir:        build-new-parser (separate from build-windows)
parallelism:      16 compile/test threads
GPU tests:        NOT RUN (local AI occupies the GPU); full gate = ctest -E "_real"
```

Baseline targeted run (before any Round-6 change), `build-new-parser`, Release, 16 threads:

```text
6 PASSED:  tool_call_parser, tool_call_grammar, tool_call_grammar_state,
           qwen3_5_frontend, request_log, serve_options
1 SKIPPED: engine_options (skips under this build configuration; recorded as observed)
```

## Finding status

| Finding | Disposition | Evidence |
|---|---|---|
| R6-01 constraint state hard-wired to `Compatibility` | **FIXED** — the constraint state now stores and applies the parsed syntax mode; low-level entry keeps the documented `Compatibility` default, production defaults to `QwenWrappedNative` | commits `b6d7c4fe` + `9c79adf2`; native/compat entry matrix in `test_tool_call_parser.cpp` |
| R6-02 Round-5 sign-off overstates residual closure | **FIXED** — `tool_call_parser.md` and the Round-5 progress now name the general unfenced complete-example ambiguity as an open residual with the semantic-quotation class | commit `f3b5f0c9` |
| R6-03 decoder `finish()` defaults to `FinishReason::None` | **FIXED** — `ToolCallOutputDecoder::finish(FinishReason)` has no default; production path passes the stored session reason; low-level callers pass the reason explicitly | commit `00b68b8f`; callsite audit (production `OutputSession` stored reason; 17 explicit low-level test callsites) |
| R6-04 complete declared unfenced example at EOF remains executable | **PIN + MITIGATE (not fully fixed)** — the wire ambiguity is inherent (complete declared call with no content is byte-identical to a genuine call). Behavior is pinned by a deterministic test; the R6-05 opt-in policy mitigates it for agent-hardened launches; the residual is named in the parser docs and design doc | commit `c18ca270`; mitigation in `417c805b`; residual named in `docs/tool_call_parser.md` and `docs/tool_call_intent_channel_design.md` |
| R6-05 tool calls after ordinary visible content stay eligible | **IMPLEMENTED as opt-in** — new `ToolCallIntentPolicy::RequireToolAtContentStart` (serve flag `--tool-call-intent start-of-content`), off by default; `template-compatible` preserves the upstream Qwen template semantics | commits `417c805b` + `f9871068`; streaming/adversarial matrix incl. Cases A–F and the reasoning-channel case |
| R6-06 no explicit TEXT-vs-TOOL intent channel | **DESIGN ONLY** — architecture design with the 12 spec topics; no production behavior change | `docs/tool_call_intent_channel_design.md` (to be committed with the docs step) |
| R6-07 constrained decoding unavailable | **KEEP FAIL-FAST** — no code change; the limitation stays documented as a non-goal for this round | `docs/tool_call_parser.md` (R6-02/R6-05 notes) |

| Step | Commit | Result |
|---|---|---|
| R6-04 pin test | `c18ca270` | New test passes; pins the executable-unfenced-example behavior verbatim (native + compatibility entry). |
| R6-01 fix | `b6d7c4fe` | Constraint state stores `syntax_`; marker classification and entry transition use the parsed mode. Parser + grammar suites green. |
| R6-01 matrix | `9c79adf2` | Entry matrix covers native/compat × marker families; green. |
| R6-03 API | `00b68b8f` | `finish(FinishReason)` mandatory; all callsites migrated; suites green. |
| R6-02 docs | `f3b5f0c9` | Round-5 sign-off and parser docs corrected; no code change. |
| R6-05 feature | `417c805b` | `ToolCallIntentPolicy` (types.h), stream gate (tool_call_stream.{h,cpp}), parser plumbing, `OutputSession` reasoning gate, serve flag, hardened local template wording; parser/frontend/serve suites green. |
| R6-05 tests | `f9871068` | Streaming/adversarial matrix (Cases A–F), consecutive-call control, reason-insensitivity, reasoning-channel intent-gate case (frontend); green after verification fixes (see below). |
| Verification fix | `f4302358` | Two defects found by the full CPU gate, both in Round-6 work: (1) the hardened template instruction line introduced an unescaped single quote inside the Jinja single-quoted string (`model's`) → Jinja parse error broke `ninfer_chat_templates_test` and crashed the frontend test at template load; fixed by rewording to "the model reasoning channel" (upstream wording preserved as the parenthetical). (2) the new frontend intent-gate test used the wrong session semantics (no reasoning close marker, wrong terminal reason, wrong per-commit channel delta); fixed to the real semantics: reasoning + `</think>` close round, terminal reason = stored budget cut (`OutputLimit`), per-commit channel deltas, held-separator content for the rejected region. Both suites green after the fix. |
| Stale callsite fix | `e44da86` | The R6-05 parser-entry signature change (`text` parameter now `const std::string&`) left one callsite in `test_tool_call_grammar_state.cpp` un-adapted; the full CPU gate surfaced the compile error. **Verification gap disclosure:** incremental targeted builds had masked this, so the earlier `tool_call_grammar_state` "green" runs used a stale binary; the fresh full-build gate re-runs the whole suite with current binaries. |

## Intentional outcome changes

- `ToolCallGrammarConstraint` now tracks the syntax mode instead of hard-wiring `Compatibility` (R6-01). Observable only through the low-level constraint API; production behavior unchanged (production default was and is `QwenWrappedNative`).
- `ToolCallOutputDecoder::finish` requires an explicit `FinishReason` (R6-03). No production behavior change (the production path already passed the stored reason).
- New opt-in serve flag `--tool-call-intent` (`template-compatible` default / `start-of-content` hardened). Default behavior is unchanged; the hardened mode rejects tool regions that do not begin the first non-whitespace content byte (after the reasoning close) and returns the markup as verbatim content (R6-05).
- Hardened local profile instruction line in `tools/chat_templates/qwen3_8.jinja` (local profile/instructions only; upstream-compatible variant documented). The production default profile is unaffected because the serve default intent stays `template-compatible` (R6-05 §8.11).

## Residuals

- R6-04 semantic-quotation residual: a complete declared unfenced call with no preceding content is indistinguishable from a genuine action on the wire bytes. Pinned, mitigated (opt-in), named in docs — not claimable as fixed.
- R6-06: no structural intent channel exists; the design doc names the experimental prerequisites (tokenizer special-token evaluation, model behavior without fine-tuning, sampler/protocol integration) before any implementation.
- R6-07: constrained decoding stays fail-fast; no runtime constrained sampling in this round.
- Low-level `ToolCallStreamParser::finish` keeps its pre-existing `FinishReason::None` default (out of the R6-03 scope; all production callsites pass the reason explicitly).

## Final sign-off

```text
build:        cmake --build build-new-parser --config Release --parallel 16 — no errors
gate:         cmake -E chdir build-new-parser ctest -C Release -E "_real" --parallel 16
              (CUDA_VISIBLE_DEVICES=99; no GPU execution per user constraint)
outcome:      147 total: 143 passed, 4 failed — the gate was NOT globally green.
failures:     ninfer_linear_swiglu_{q4_a16,q8_a16,nvfp4,fp8}_test — all print
              "SKIP: no usable CUDA device", but their main() functions translated
              the helper's exit code 77 to exit code 1, so CTest recorded them as
              failures despite the intended SKIP_RETURN_CODE 77 configuration
              (environmental: local AI occupies the GPU; identical to the Round-5
              record). No CPU-test failures. R7-04 corrected the wrappers and this
              wording.
skips:        GPU-dependent tests that exit 77 directly from main() are counted as
              skipped via SKIP_RETURN_CODE; the four LinearSwiGLU tests above did
              not reach that path in Round 6.
targets:      ninfer_tool_call_parser_test, ninfer_tool_call_grammar_test,
              ninfer_tool_call_grammar_state_test, ninfer_qwen3_5_frontend_test,
              ninfer_serve_options_test, ninfer_request_log_test,
              ninfer_chat_templates_test — all PASSED on the final gate.
```

Round 6 is complete: R6-01–R6-05 implemented, R6-06 designed, R6-07 documented as
fail-fast non-goal. The full CTest -E "_real" gate was not globally green (143 passed,
4 failed — the four GPU-dependent LinearSwiGLU wrappers, environmental); all
parser/frontend/serve/template CPU targets relevant to this change passed.
