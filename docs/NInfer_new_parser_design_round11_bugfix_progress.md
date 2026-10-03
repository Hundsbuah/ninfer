# NInfer `new_parser_design` — Round 11 Bugfix Progress

Status: **complete** — all five Round-11 defects (R11-01 … R11-05) are implemented and
verified. This document reports actual test results, per the Round-11 specification
(`docs/NInfer_new_parser_design_round11_bugfix_implementation.md`).

```text
Branch:   new_parser_design
Baseline: 59fc37c6b6c315bafa561464791e0bf8e2a21927
Toolchain: Visual Studio 18 (2026), CUDA v13.4, sm_120a, vcpkg F:\GIT\vcpkg
Build dir: build-r11 (dedicated; build-windows is in use by the running local AI)
Config:    Release, NINFER_BUILD_APPS=OFF, BUILD_TESTING=ON
```

## 1. Defects and fixes

### R11-01 — recovery retry bypassed the indented-literal rule

**Before:** the Stage-1 recovery/retry search (`find_tool_marker` loop in
`tool_call_stream.cpp`) accepted a later marker as an executable entry base without
applying the Round-10 indented-literal classification. An indented (4+ visual columns)
marker became executable merely because recovery found it.

**Fix:**
- `tool_call_entry_scan.{h,cpp}`: new `ToolCallLineIndentationScan` — one forward pass over
  the full logical text (`pre-latch content + region`) builds a per-physical-line table
  (line start, first non-space column, tab stops); a query binary-searches the line and
  returns true only when the complete marker starts at the line's first non-formatting
  byte and the line's visual column is >= 4.
- `tool_call_stream.cpp`: the retry loop now runs every candidate through
  `candidate_suppressed` with precedence:
  1. fenced candidate → fence-skipped, never counted as indented (R11-I4);
  2. candidate inside an open value's payload after an EndOfInput break → not an entry
     candidate at all (R11-I5: the wire parser owns the payload; it may be the real
     truncated call);
  3. otherwise the indented-literal rule applies over the full physical line; a skipped
     complete candidate increments `indented_markers_suppressed` exactly once (R11-I14).

**Verification:** reproducer probe before the fix: one-shot calls=1, decoder calls=1 on a
fixture whose only later candidate is indented. After: 0/0. Parser tests
`test_r11_retry_indent_bypass`, `test_r11_tab_indent_is_literal` pass (whole-chunk and
byte-wise splits).

### R11-02 — `OutputSession` stripped generic whitespace after the reasoning close

**Before:** the post-reasoning separator strip in `output_session.cpp` used
`std::isspace`, destroying the leading SPACE/TAB (or other formatting) of the first
Content byte after `</think>` — i.e. it destroyed the 4-space/TAB indentation that marks a
displayed tool call as literal.

**Fix:** `feed_content` now strips **only** leading CR/LF bytes after the reasoning close
(`strip_post_reasoning_linebreaks`); the strip stops at the first non-CR/LF byte. A leading
space or tab is genuine Content and is preserved. The `</think>` close itself is only
confirmed by a following line break (existing boundary rule, unchanged).

**Verification:** `test_r11_post_reasoning_separator_preserves_indentation`
(`tests/models/qwen3_5/test_frontend.cpp`) — real `Frontend` + `OutputSession` with
thinking enabled:
- I: reasoning + close + `\n\n` then an indented call → 0 calls, content byte-exact;
- J: reasoning + close + `\n\n` then a real call → 1 call;
- K: indented call split across the reasoning/Content boundary → 0 calls, byte-exact;
- G: close confirmed by its line break, then `" answer"` → the leading SPACE is preserved
  (the old strip destroyed it); the confirming separator is stripped.
All sub-cases pass.

### R11-03 — the streaming decoder's terminal path re-parsed the latched suffix
without context

**Before:** `ToolCallOutputDecoder::finish()` re-ran `parse_qwen_tool_call_output` on the
isolated latched suffix. That contextless re-parse lost the pre-latch physical-line
context (the indented-literal classification) and ran a second, different
materialization, so one-shot and streaming could disagree.

**Fix:**
- `tool_call_parser.{h,cpp}`: new `MaterializedToolCallResult` +
  `materialize_tool_call_result(const ToolCallStreamResult&, const ToolCallOutputContract&)`
  — the single contract-aware materialization (no-marker/Incomplete/declared-name
  defense-in-depth/R3-08 duplicate-parameter rule/schema-aware normalization) shared by
  the one-shot entry and the decoder.
- The decoder now builds its live `ToolCallStreamParser` with the full contract-aware
  policy (declared names, duplicate-parameter check, parameter plausibility, syntax,
  ambiguity, intent) and `finish()` finalizes **that** live machine — no second
  region-only re-parse. The machine already classified every entry over the full
  physical-line context (pre-latch bytes included).
- Fallback content: `latched_region()` when the machine latched but the materialization
  rejected (e.g. the ambiguity rule after a base-0 Stage-1 accept, where `result.tail`
  is empty); `held_tail()` when never latched. Accept path:
  `rtrim_format_whitespace(result.tail)`.

**Verification:** reproducer probe before the fix: one-shot calls=1, decoder calls=0 on
the same fixture. After: 1/1. Equivalence tests
`test_r11_undeclared_and_ambiguous_equivalence` and
`test_r11_one_shot_streaming_equivalence_corpus` pass: one-shot == whole-chunk ==
byte-wise streaming on calls, content and the complete diagnostics record, including the
materialization-boundary fallbacks (undeclared tool, duplicate parameter).

### R11-04 — the "CRLF" split fixture was not real CRLF

**Before:** `test_r10_multi_boundary_splits` built its CR/LF fixture by inserting a bare
`\n` and calling it CRLF, so the between-CR-and-LF split point was never exercised.

**Fix:** the builder now emits true CRLF (`\r\n` per line break); the fixture asserts it
contains `\r\n` and no `\n\r`; splits now include split point 2 (inside the indentation),
`cr_pos + 1` (between CR and LF), and 5 (inside the marker). All splits stay equivalent
to the whole-text parse. Passes.

### R11-05 — fence/indent diagnostic scope was documented inconsistently

**Fix (documentation only):**
- `docs/tool_call_parser.md`: `fenced_markers_suppressed` scope (pre-latch content + the
  region's own markers when the region is returned as text; recovery-retry fence skips are
  **not** counted), the indented-literal bullets (full physical-line context, tab stops,
  whitespace-only lines, open-value payload, retry/rebase applicability, fence precedence,
  single counting), the diagnostics table, and a parser-architecture note that the
  streaming decoder finalizes its live machine at the terminal (no second re-parse; one
  shared materialization with the one-shot entry).
- `include/ninfer/types.h`: the `fenced_markers_suppressed` /
  `indented_markers_suppressed` comments now state the accurate scopes.

## 2. Changed files

```text
M src/models/qwen3_5/frontend/tool_call_entry_scan.h     (+41)  ToolCallLineIndentationScan
M src/models/qwen3_5/frontend/tool_call_entry_scan.cpp    (+60)  line table + query
M src/models/qwen3_5/frontend/tool_call_stream.cpp        (+26/-…) retry candidate gating
M src/models/qwen3_5/frontend/tool_call_parser.h          (+24/-…) MaterializedToolCallResult
M src/models/qwen3_5/frontend/tool_call_parser.cpp        (+185/-…) shared materialization,
                                                                    contract-aware live machine
M src/models/qwen3_5/frontend/output_session.cpp          (+22/-…) CR/LF-only separator strip
M include/ninfer/types.h                                  (+16/-…) diagnostic scope comments
M tests/test_tool_call_parser.cpp                         (+376)  R11 parser/decoder tests +
                                                                    true-CRLF fixture
M tests/models/qwen3_5/test_frontend.cpp                  (+139)  R11 OutputSession tests
M docs/tool_call_parser.md                                (+40/-…) retry/diagnostic/terminal docs
```

No Stage-1/2/3 core redesign; production defaults unchanged
(`QwenWrappedNative + FailClosed + TemplateCompatible + tolerant=false`); no syntax
broadening; no config workaround; no claim that constrained tool generation is
implemented. Untracked local drivers: `r11_build.bat`, `r11_build_tests.bat`
(`build-r11`), `r10_build*.bat` (`build-windows`).

## 3. Tests added

**`tests/test_tool_call_parser.cpp`** (one-shot + streaming decoder, whole-chunk and
byte-wise splits unless noted):
- `test_r11_retry_indent_bypass` — F: retry finds only an indented candidate → not
  latched, `indented_markers_suppressed == 1`; F2: indented then real candidate → the
  real one executes; F3: no indented candidate → no spurious count.
- `test_r11_tab_indent_is_literal` — B: TAB-indented marker (column 4) is literal on both
  syntax modes, returned verbatim, counted, no parse-failure diagnostic.
- `test_r11_same_line_prose_keeps_marker_eligible` — D: prose on the marker's own line
  keeps it eligible (TemplateCompatible), it executes.
- `test_r11_prose_then_indented_marker_is_literal` — E: indented marker after prose is
  literal, `indented_markers_suppressed >= 1`.
- `test_r11_fenced_indented_then_real_call` — H: indented marker inside a fence is
  fence-suppressed (not double-counted as indented); the later column-0 call executes;
  the fence block is returned as content.
- `test_r11_undeclared_and_ambiguous_equivalence` — L/M: undeclared tool (enforced
  names) and duplicate parameter (ambiguity rule) — one-shot and streaming agree on
  calls, content and the complete diagnostics record.
- `test_r11_one_shot_streaming_equivalence_corpus` — N: corpus of accepted/rejected
  fixtures; one-shot == whole-chunk == byte-wise streaming, including the fallback
  content bytes.

**`tests/models/qwen3_5/test_frontend.cpp`**:
- `test_r11_post_reasoning_separator_preserves_indentation` — I/J/K/G: reasoning-close →
  Content separator semantics on the real OutputSession (indented call preserved, real
  call executes, split indented call byte-exact, leading space preserved).

**Fixture fix** (R11-04): `test_r10_multi_boundary_splits` CRLF builder now emits true
`\r\n` and splits between CR and LF.

## 4. Verification (actual results)

Environment: Windows x64, VS 18 (2026), CUDA v13.4, `build-r11` Release.

Targeted (spec-mandated set), `ctest --test-dir build-r11 -C Release`:

```text
1/4 ninfer_request_log_test             Passed
2/4 ninfer_qwen3_5_frontend_test        Passed
3/4 ninfer_tool_call_parser_test        Passed
4/4 ninfer_tool_call_grammar_state_test Passed
100% tests passed out of 4
```

Full CPU-capable CTest suite (`ctest --test-dir build-r11 -C Release`, 161 tests,
wall time ≈ 20.6 min):

```text
0 failed
15 skipped  (same skip class as the baseline: real-model/GPU-dependent tests, each
             individually confirmed self-skipping — #53 ngram_graph_planning_real,
             #54/55/56/57/58 ngram_lifecycle/archive/thinking/stop_chat/concurrent_real,
             #59 qwen3_5_loading_real, #66-73 qwen3_5_prefix/hybrid_prefix/score/
             vision_workspace/dflash2/dflash_prefill/moe/dflash_real; they self-skip
             without a GPU/model artifact and are listed by CTest under "did not run")
146 passed  — identical to the baseline passing set; no reduction
```

Baseline (pre-Round-11, same machine): `146 passed / 15 skipped / 0 failed`. Round 11
must not reduce the passing set.

Reproducer probe (throwaway, deleted after the round): before the fixes — R11-01
one-shot/decoder calls = 1/1 (defect), R11-03 = 1/0 (defect); after — 0/0 and 1/1
respectively. For R11-02 the probe simulated the old `std::isspace` strip and showed it
destroys the leading indentation of the first Content byte; the CR/LF-only rule is
verified by the new frontend sub-cases I/J/K/G (all passing).

## 5. Review checks (spec Appendix B.4)

- No path calls `find_tool_marker()` for an executable retry without the indentation
  eligibility: the only retry loop is the gated `candidate_suppressed` search
  (`tool_call_stream.cpp`).
- No `parse_qwen_tool_call_output()` call on an isolated latched suffix remains: the
  one-shot entry is the only caller (public API + tests); the decoder finalizes its live
  machine.
- No generic `std::isspace()` remains in the reasoning-to-Content separator path:
  CR/LF-only strip (`output_session.cpp`); the only `isspace` occurrences in the affected
  chain are explanatory comments.
- No second contextless terminal entry scan remains.

## 6. Notes

- The dedicated `build-r11` tree was required because `build-windows` hosts the running
  local AI (`ninfer-serve.exe`); building there is disallowed. `build-r11` uses the same
  generator/toolchain as the `build-windows` cache (`Visual Studio 18 2026`,
  `-T cuda=v13.4`, vcpkg `F:\GIT\vcpkg`, `CMAKE_CUDA_ARCHITECTURES=120a`) with
  `NINFER_BUILD_APPS=OFF`. The runtime-DLL staging target
  (`ninfer_stage_test_runtime_dlls`) must be built in a fresh tree (it is an `ALL`
  target, not a dependency of the test targets).
- Throwaway artifacts from the round (`build-windows/r11_repro.*`,
  `build-windows/r11_debug_g.cpp`) were deleted.

## 7. Supersession note (added in Round 12)

Round 11 passed its stated implementation gate at 2c7aa184, but Round-12 adversarial
cross-layer review found additional issues R12-01 … R12-06. Round 12 supersedes the prior
"complete" release sign-off.
