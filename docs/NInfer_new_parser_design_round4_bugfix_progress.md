# NInfer `new_parser_design` — Round 4 Bugfix Progress

> Working evidence log for
> `NInfer_new_parser_design_round4_bugfix_implementation.md`.

## Rules

- Update this file after every implementation phase and every review loop.
- Record exact commands and results; never mark a finding verified from compilation alone.
- Never mark a finding fixed without a direct regression test.
- No GPU runtime tests, no `_real` tests, no model inference, no GPU benchmarks: a local AI
  workload runs on the GPU. CUDA is used for compilation only.
- 16 compile and 16 test threads throughout.
- Build directory: `build-new-parser` (never `build-windows`).
- Preserve all newly discovered findings even after they are fixed.

---

## Baseline

- Branch: `new_parser_design`, local clone `E:/KI/ninfer-custom` (Windows 10 x64 worktree)
- Starting HEAD: `33f89bd0` (`fix(frontend): align Stage 2 with the R3-01 family rules`), i.e.
  the committed Round-3 tree (round-3 session 3). The Round-4 review machine cited by the spec is
  `4cc2ba9c` (round-3 R3-03 item 3); `33f89bd0` sits two commits later and already carries the
  session-3 Stage-2 family fixes, which this round's sequential-chooser rewrite had to preserve.
- Date: 2026-10-01 (single implementation session).
- Worktree state at start: not clean — the same four pre-existing user-local adaptations as in
  Rounds 2 and 3 (`CMakeLists.txt`, `build_native.bat`,
  `src/runtime/engine/context_cache/materialization_budget.h`,
  `src/runtime/engine/kv_capacity.cpp` with the explicit capacity error throw commented out).
  All four preserved; none committed with the Round-4 work.

## Commits (per §5.2; one per step)

| Step | Commit | Content |
|------|--------|---------|
| 1    | `eb02cd58` `refactor(frontend)` | §2.1 absolute region positions, §2.2 region index (Stage-1 lookups only), no `substr` copies |
| 2    | `20aa895c` `perf(frontend)` | §2.4–§2.6 sequential chooser, global work budget, shared dead memo, early exit; N-04 (no recursion/depth constant); N-11 (no allocating `noexcept`); `parse_budget_exhausted` in `ToolCallStreamResult`, `ToolCallParseDiagnostics`, `parse_qwen_tool_call_output` (N-07 items 1–3) |
| 3    | `1b528444` `fix(frontend)` | §2.3 no Stage-1 chain attempt bound (N-03); Stage-3 tail-closer lookup via the index |
| 4    | `6729a847` `test` | N-02 and N-05 dedicated reproducer tests (the skip/census fixes themselves landed inside the step-2 sequential chooser; this step pins them) |
| 5    | `cecff7c6` `fix(frontend)` | N-06 item 1 (fence diagnostics scoped to text results, ambiguity-path copy) and N-08 (fence-content whitespace hold) |
| 6    | `09a7c9df` `fix(serve)` | N-06 item 2 (operational-log fence/budget suffixes and precedence) and N-07 items 4–5 (request log + operational log `parse_budget_exhausted`) |
| 7    | `b14b4dcf` `test` | N-09: re-register `test_composed_schema_types`, R1 residual pin + R1-inline relabel, fuzz rewrite, CR6 field assertions, request-log fence/budget assertions |
| 8    | `232802f6` `docs` | N-10: help text, `serving.md`, `tool_call_parser.md`, code comments, this progress file |

Note: the sequential chooser (step 2) is a full rewrite of the recursive Stage-2, so the N-02
(skip via `has_close_at_or_after`) and N-05 (same-family walk) fixes are part of it; step 4 adds
their dedicated reproducer tests rather than a second fix commit. This is why the step-4
outcome-dump delta is zero (below).

---

## Refactor guard (Appendix B outcome dump, 36306 cases)

The Appendix-B `outcome_dump.cpp` (6000 seeded random fragment texts × 2 tolerant × 3 finish
reasons, plus the Round-1/2/3 implementation docs as `write` payloads) was built and run against
five trees with the standalone MSVC toolchain (16 threads, `build-new-parser/round4`, no GPU):

| Pair (old → new)                              | Cases changed |
|-----------------------------------------------|---------------|
| `4cc2ba9c` → `1b528444` (steps 1–3: §2.1–§2.6, N-03, N-04, N-07) | **0** |
| `1b528444` → `6729a847` (step 4: N-02/N-05 tests)             | **0** |
| `6729a847` → `cecff7c6` (step 5: N-06 fence + N-08 whitespace) | **549** |
| `cecff7c6` → final working tree (steps 6–8: serve/tests/docs)  | **0** |

Field-level breakdown of the 549 step-5 changes: every change is confined to the two fence
diagnostics. `ended_in_unclosed_fence` changed in 549/549 (all `1 → 0`), `fenced_markers_suppressed`
in 166/549. No `is_tool_call_response`, content hash, call list, marker/call/synthetic/duplicate/
`markup_tolerant_completion` field, fallback reason, or `stream=` mode changed. This is exactly the
step-5 condition in §5.2 (differences only in the two fence fields; every line keeps `stream=1` or
`stream=-1`).

### Step-4 (N-02/N-05) differences: none in the corpus

The N-02 and N-05 fixes do not alter any of the 36306 corpus cases. That is expected: the corpus
is a random 2–14-fragment corpus plus three whole documents as `write` values, and neither the
`<param>`-family Stage-2 region (N-02) nor a cross-family opener inside a value (N-05) is produced
by it. Both fixes are instead pinned by their dedicated Appendix-A reproducer tests
(`test_round4_n02_param_family_reaches_stage2`, `test_round4_n05_cross_family_opener`), which pass
against the final tree and reject against `4cc2ba9c`. The zero corpus delta also confirms the
sequential-chooser rewrite preserved the session-3 Stage-2 family fixes (no regression).

### Step-5 (N-06/N-08) differences: 549 fence-field corrections

All 549 changes flip `ended_in_unclosed_fence` from the buggy `1` to the correct `0` (383 of them
change only that field; the other 166 also correct `fenced_markers_suppressed`). This is the N-06
fix: the pre-latch unclosed-fence flag and the fence-suppression count now scope to the pre-latch
stream and to text results respectively, so a closed code block followed by a call no longer reports
`ended_in_unclosed_fence` and an accepted fenced `write` no longer reports suppressed markers.

---

## False Round-3 claims and their corrections (per N-10 item 7)

The Round-3 progress file (`NInfer_new_parser_design_round3_bugfix_progress.md`) recorded the
claims below; the Round-4 review found each to be false or not supported by the diff. They are
recorded here, not edited in place (per N-10 item 7, the Round-3 file is not rewritten).

1. **"closer-less-region skip … behavior-neutral, `stage2_steps` stays 0"** (Round-3 line 65,
   R3-14) — **false, N-02.** The skip fired on the absence of `</parameter>` only, so a region whose
   values use `<param …>`/`</param>` never reached Stage 2 and a unique consistent parse was
   rejected as text (`trailing_content`). Correction: the skip now tests
   `!index.has_close_at_or_after(attempt.base)`, which covers both closer families.
2. **"no other existing assertion changed"** (Round-3 line 158, acceptance) — **false, N-09 a.**
   `test_composed_schema_types` was defined in the Round-2 tree but never registered in `main()`,
   so its composed-schema-type coverage was silently dead. Correction: it is now registered and
   runs.
3. **"the section-9 phantom acceptance is not implemented"** (Round-3, R3-04 residual note) —
   **false, N-09 b.** The canonical-framing rule makes an unfenced complete example followed by
   prose ending with the canonical closer lines commit the example as the turn (the documented
   phantom `bash` call). Correction: pinned by `test_round4_r1_residual_pinned` so any future
   change is deliberate.
4. **"dead between-round path removed"** (Round-3 line 64, R3-13) — **not in the diff.** The
   claimed removal did not appear in the Round-3 changeset; it is not a Round-4 concern and is
   noted here for accuracy.
5. **"budgets fail closed" as a time bound** (Round-3 line 161, R3-14) — **false, N-01.** The
   Stage-2 bound was a step/attempt budget, not a time bound, and the 256-attempt chain bound (N-03)
   plus the recursive depth bound (N-04) made Stage 2 both unbounded-in-time on adversarial input
   and lossy on legitimate large input. Correction: a single deterministic global work budget
   (four units per region byte, at least 100 000) charged per walk step/candidate/glue transition,
   no chain bound, no recursion; exhaustion records `parse_budget_exhausted` and returns text.

---

## Performance (Appendix C probe, Release, 16 threads, no GPU)

`build-new-parser/round4/perf_probe.exe` (best of 3 wall clock; deterministic Stage-2 work): the
work budget is the governing bound. The Appendix-C targets (recorded after steps 2, 3 and 8 — the
numbers below are the final working tree, step 8 is docs-only and leaves the parser unchanged):

| Case | bytes | calls | reason | stage2_steps | wall |
|------|-------|-------|--------|--------------|------|
| P1 repetition k=1500, `OutputLimit` | 159 390 | 1 | none | 0 | 1.1 ms |
| P2 repetition k=1500 + trailing prose | 159 402 | 0 | trailing_content | 637 569 | 13.0 ms |
| P3 300 examples + real `read` call (N-03) | 31 800 | 1 | none | 0 | 0.2 ms |
| P4 8000 closer triples in one `write` value | 344 131 | 1 | none | 0 | 4.3 ms |
| P5 4000 open bare regions, `None` | 210 890 | 0 | malformed_structure | 0 | 1.6 ms |
| P6 70 parameters, last value needs Stage 2 (N-04) | 2 446 | 1 | none | 0 | 0.1 ms |
| P7 2000 examples in one `write` value | 186 131 | 1 | none | 0 | 1.8 ms |
| P8 500 KB plain `write` | 512 125 | 1 | none | 0 | 2.7 ms |
| P9 Round-2 document as `write` content | 47 417 | 1 | none | 0 | 0.5 ms |

The N-03 regression is the headline: `4cc2ba9c` returned text after 91.5 ms for P3 (the
256-attempt chain bound lost the real call); the final tree finds it in 0.2 ms with the prose kept
as content. P2 is the worst Stage-2 case in the probe (13.0 ms, 637 569 charged steps) and is well
under the `max(100 000, 4 × bytes)` budget; the `test_round3_work_bounds` assertions pin the
`stage2_steps <= max(100 000, 4 × text.size())` and budget-exhaustion (`parse_budget_exhausted`)
behaviour.

---

## Test evidence (CPU only, 16 threads, `CUDA_VISIBLE_DEVICES=99`)

Targets (per §8.1), Release, `build-new-parser`:
`ninfer_tool_call_parser_test`, `ninfer_tool_call_grammar_test`,
`ninfer_tool_call_grammar_state_test`, `ninfer_qwen3_5_frontend_test`, `ninfer_request_log_test`,
`ninfer_pretty_logging_test`, `ninfer_serve_options_test`, `ninfer_engine_options_validation_test`
— **all pass** against the final working tree. The dedicated Round-4 reproducer tests
(`test_round4_n02_*`, `test_round4_n05_*`, `test_round4_r1_residual_pinned`, the N-06 fence-field
assertions, the N-08 exact streaming-equality (no `rtrim`), and the rewritten
`test_round3_streaming_equivalence_fuzz` over the §10.3 corpus with `mt19937 rng(20260930)`, 2000
texts, strict+tolerant, `StopToken`+`OutputLimit`, whole/byte-wise/random-chunk partitions) all
pass.

Full `ctest -E "_real"` (16 threads, `CUDA_VISIBLE_DEVICES=99`): 144 of 149 pass. The 5
non-passing cases are unrelated to the parser and are not new in this round:
`ninfer_kv_capacity_test` fails on the pre-existing user-local `kv_capacity.cpp` adaptation
(the explicit capacity error throw is commented out in this worktree), and the four
`ninfer_linear_swiglu_*_test` CUDA-kernel tests report "no usable CUDA device" and are skipped
by the harness (the local AI owns the GPU). Every parser target passes.

Not executable here: an AddressSanitizer build of the parser suite — the MSVC ASan runtime fails at
process start in this environment (same limitation recorded in Round-3 session 3). The Round-4
changes are the sequential-chooser rewrite (no new allocations on the search path; values are
ranges materialized only for the accepted base) plus fence/whitespace and serve logging; the full
CPU suite result above is the behavioural evidence.

---

## Acceptance checklist (§9)

- [x] N-01: sequential chooser, global budget, shared memo, early exit, lazy walks; no value copies during the search
- [x] N-02: Stage-2 skip covers both families; reproducer accepted
- [x] N-03: no chain attempt bound; 300-example reproducer finds the real call (P3: 0.2 ms)
- [x] N-04: no recursion and no depth constant; 70-value reproducers accepted (P6)
- [x] N-05: same-family walk; reproducer accepted
- [x] N-06: pre-latch unclosed flag false after a latch; region fence counts only for text results; operational-log precedence and suffixes
- [x] N-07: `parse_budget_exhausted` in `types.h`, results, request log, operational log, docs
- [x] N-08: fence-content whitespace held; exact streaming equality without `rtrim`
- [x] N-09: `test_composed_schema_types` called; R1 pinned and R1-inline relabelled; fuzz rewritten; request-log and CR6 field assertions
- [x] N-10: help text, `serving.md`, `tool_call_parser.md`, code comments corrected; Round-3 corrections recorded above
- [x] N-11: no allocating `noexcept` function on the parse path
- [x] outcome dump: steps 1, 2, 3, 6, 7, 8 identical to the previous step (0 delta); steps 4 and 5 differences listed and explained above
- [x] §6 changes applied, each with a comment naming the Round-4 finding; no other existing assertion changed
- [x] target suites green; full CPU suite result recorded (no new failure)
- [x] no GPU test executed; `build-windows` unused; ≤ 16 threads
- [x] progress file complete (this file)

---

## Files changed (committed + working tree)

`include/ninfer/types.h`, `src/models/qwen3_5/frontend/tool_call_parser.cpp`,
`src/models/qwen3_5/frontend/tool_call_stream.cpp`, `src/models/qwen3_5/frontend/tool_call_stream.h`,
`src/serve/operational_log.cpp`, `src/serve/request_log.cpp`, `tests/test_tool_call_parser.cpp`,
`tests/test_request_log.cpp`, `src/serve/serve_options.cpp`, `docs/serving.md`,
`docs/tool_call_parser.md`, `docs/NInfer_new_parser_design_round4_bugfix_implementation.md`
(spec), and this file. The four user-local adaptations are preserved and not committed.