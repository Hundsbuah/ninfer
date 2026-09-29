# New Parser Progress

## Baseline
- Upstream repository: https://github.com/Wallawalla47/ninfer-custom (remote `origin`, fetch-only, push disabled)
- Upstream branch: master (`origin/HEAD` -> `origin/master`)
- Upstream SHA: `081bf60285e00c24b507d18e77409035d38648e4`
- Working branch: `test3` (created fresh from `origin/master`)
- Build directory: `build-new-parser`
- Date started: 2026-09-28

## Environment deviations from the plan (verified 2026-09-28)
1. Plan's recorded master SHA `c208d66f6623f898b131781a7cbc000c56bce4b0` is stale. After
   `git fetch origin --prune`, `origin/master` = `081bf60285e00c24b507d18e77409035d38648e4`
   and `origin/HEAD` points to `origin/master` (default branch unchanged: master).
   Branch `test3` was created from `origin/master` = `081bf602...`.
2. Working branch is `test3` per user instruction (plan proposed `new_parser`).
3. The vcpkg prebuilt tree lives at `F:\GIT\vcpkg`, not `C:\vcpkg` (plan text and
   `build_native.bat` say `C:\vcpkg`). Verified: `F:\GIT\vcpkg\installed\x64-windows`
   contains ffmpeg import libs/DLLs (avcodec-63, avformat-63, avutil-61, swscale, ...)
   and libcurl (>= 7.85), and the existing `build-windows/CMakeCache.txt` records
   `CMAKE_PREFIX_PATH=F:\GIT\vcpkg\installed\x64-windows`.
   New configure therefore uses
   `-DCMAKE_PREFIX_PATH="F:/GIT/vcpkg/installed/x64-windows"` plus env `VCPKG_ROOT=F:/GIT/vcpkg`.
4. Toolchain matches the plan: Visual Studio 18 2026 Community
   (`C:\Program Files\Microsoft Visual Studio\18\Community`), CUDA 13.4 (v13.4.92,
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`), x64,
   `CMAKE_CUDA_ARCHITECTURES=120a`, `BUILD_TESTING=ON`.

## Constraints
- No GPU runtime tests (CUDA code may compile; no `_real`/CUDA inference runs, no benchmarks)
- Max build threads: 16
- Max test threads: 16
- `build-windows` must not be used (reserved for the running local AI); `build-new-parser` used instead

## Phase 1
### P1.1 Baseline freeze + progress file
Status: complete
Files changed: `docs/new_parser_progress.md` (this file)
Implementation decision: branch `test3` created from `origin/master` 081bf602 (user-overridden branch name).
Alternatives rejected: `origin/main` (does not exist; remotes verified: origin, ninfer, hundsbuah); reusing `build-windows` (forbidden).
Tests added: none
Tests executed: `ninfer_tool_call_parser_test`
  - configure: `cmake -S . -B build-new-parser -G "Visual Studio 18 2026" -A x64 -T "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4" -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_PREFIX_PATH="F:/GIT/vcpkg/installed/x64-windows" -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF` (env `VCPKG_ROOT=F:/GIT/vcpkg`)
  - build: `cmake --build build-new-parser --config Release --parallel 16 --target ninfer_tool_call_parser_test` (444 s)
  - test: `ctest --test-dir build-new-parser -C Release -R "^ninfer_tool_call_parser_test$" -j 16 --output-on-failure` -> 1/1 PASSED (0.10 s)
Result: baseline green on 081bf602 (33 test functions, strict + tolerant + incremental decoder semantics)
Known limitations: none
Commit: (P1.1 commit, see git log)

Findings from the baseline read (verified against source, pre-Phase-1):
1. `kToolMarkers` (12 literal strings) is a second, separately maintained marker grammar: streaming
   holds bytes only while they stay a prefix of one of those literals, so any tool region opener
   longer than 16 bytes (attribute form `<function name="...">`, long short-form names) is not
   recognized by the streaming decoder at all, while one-shot accepts it. One-shot != streaming.
2. Header parsing uses unquoted `find('>')`: a `>` inside a quoted attribute breaks the header
   (`<parameter filename="a>b" name="x">` parses name `a`), and an unterminated quote is silently
   truncated instead of rejected.
3. `extract_name_from_tag_header` falls back to the raw header text when no `name=` attribute is
   found, so malformed attribute junk can become a "name" (e.g. `<parameter foo=>` yields the
   non-empty name ` foo=` for a parameter, which only passes the non-empty check).
4. `closes_parameter`/`find_parameter_close` (boundary lookahead for quoted closers in values)
   rebuilds opener/closer rules with its own `is_param_open_at` + literal checks instead of the
   canonical header rules.
5. `test_later_candidate_must_consume_the_end` uses literal two-character `\n` escapes (not real
   newlines) for the quoted region; the region then fails with `MalformedStructure` at the first
   non-tag byte, which is the reason the test records. Verified green by build + ctest.
6. Second CPU regression target identified: `ninfer_qwen3_5_frontend_test` (OutputSession-level
   tool-call integration, `tests/models/qwen3_5/test_frontend.cpp`), CPU-only, run per phase gate.
### P1.2 Authoritative wire grammar component
Status: complete
Files changed: `src/models/qwen3_5/frontend/tool_call_grammar.h` (new), `tool_call_grammar.cpp` (new), `frontend_sources.cmake` (+grammar source)
Implementation decision: one namespace-scoped authority — tag families (`ToolTagKind`), exact wrapper
  literals (`tool_open_literal`/`tool_close_literal`), one header grammar
  (`parse_tool_function_open`/`parse_tool_parameter_open`/`parse_tool_header_after_keyword`) with
  statuses NoMatch/NeedMore/Complete/Invalid, plus streaming prefix classification
  (`classify_tool_marker_prefix`) and marker discovery (`find_tool_marker`). Header grammar: short
  form `=value`, attribute list `attr = value`, or bare opener; quoted values (both quote chars, `>`
  may appear inside), unquoted values end at whitespace/`>`/`/`; after a value only whitespace or `>`
  is allowed; the first `name` attribute is the tag name. `ToolOpenTag.consumed` counts from the tag's
  `<` through the terminating `>`.
Alternatives rejected: a PEG/GBNF engine (llama.cpp style) — the wire syntax is fixed and small; the
  hand-written recursive descent returns the same clarity with fewer moving parts and keeps the
  NeedMore/Complete distinction the streaming decoder requires. Keeping the family in the result —
  the consumer needs it for family-matched closers, a `name`-only dispatch could not.
Tests added: none (see P1.3)
Tests executed: standalone probe harness (temporary, deleted) — grammar verified in isolation before
  integration: short/attribute/bare headers, quoted names with `>`, marker discovery and prefix
  classification all correct
Result: grammar standalone-correct
Known limitations: none
Commit: (see P1.5 commit)

### P1.3 Grammar test matrix
Status: complete
Files changed: `tests/test_tool_call_grammar.cpp` (new), `tests/models/qwen3_5/tests.cmake` (+grammar test target)
Implementation decision: direct tests of the public grammar API (no parser indirection), so the
  grammar's own contract is protected before consumers migrate
Tests added: `ninfer_tool_call_grammar_test` — function header forms (short/attribute/bare, quote
  variants, whitespace variations, cut-off prefixes), function name semantics (empty, invalid chars,
  length limit, `name` attribute vs short value), parameter header forms (parameter/param families,
  quoted names containing `>`), quote-aware headers (unterminated quote = cut, not silent
  truncation), marker prefixes (per-byte streaming prefixes; 1/2/3/7 and all-split chunk
  equivalence), marker discovery (first-marker scan, prose rejection)
Tests executed: `ctest -R ninfer_tool_call_grammar_test` -> passed
Result: green
Known limitations: none
Commit: (see P1.5 commit)

### P1.4 Migrate marker discovery and consumers to the grammar
Status: complete
Files changed: `src/models/qwen3_5/frontend/tool_call_parser.cpp`
Implementation decision: removed the second marker grammar (`kToolMarkers`, `matches_any_marker`,
  `is_prefix_of_any_marker`), the header re-implementation (`extract_name_from_tag_header`,
  `valid_function_name`, `kToolOpen`/`kToolClose`), and the unquoted `unquote` helper; all of
  `parse_qwen_tool_call_output`, `parse_tool_call`, `parse_function`, `parse_parameter`,
  `closes_parameter`, `find_parameter_close`, and the streaming decoder's pending-tag logic now take
  tag rules from the grammar. Behavior changes vs. the old parser (expected, no test pinned the old
  bytes):
1. streaming now recognizes attribute-form and long short-form openers (old decoder held only
   literal prefixes up to 16 bytes and leaked those regions to content, while one-shot parsed them);
2. a `>` inside a quoted attribute no longer breaks the header (`<parameter filename="a>b" name="x">`
   now parses the name `x` instead of `a`);
3. unterminated quotes at the region end are rejected (MalformedStructure) instead of silently
   truncated;
4. an attribute value must be followed by whitespace or `>` (missing separator = MalformedStructure);
5. the boundary lookahead (`closes_parameter`/`find_parameter_close`) now uses the grammar's opener
   rule instead of its own `is_param_open_at` + literal checks (plan P3 invariant: one source of
   truth for the "next token after a quoted closer" decision); a malformed opener inside a value no
   longer counts as a nested opener (grammar-consistent, stricter in the data-swallowing direction);
6. strict mode maps a broken function header to MalformedStructure where the old broken-name
   fallback produced InvalidToolName (reason change only; no test pinned the old reason).
Tests added: none (existing matrix carries the semantics; new cases in P1.5)
Tests executed: none yet (build in progress)
Result: (see P1.5)
Known limitations: none
Commit: (see P1.5 commit)

### P1.5 One-shot semantics via the same grammar + parser-level matrix
Status: complete
Files changed: `tests/test_tool_call_parser.cpp` (+2 test functions, registered in main)
Implementation decision: the public API is unchanged (`parse_qwen_tool_call_output`,
  `ToolCallOutputDecoder`, `build_tool_call_output_contract`); one-shot and streaming go through the
  same grammar by construction (same functions, same literals)
Tests added: `test_grammar_header_forms_one_shot` (attribute form, spaces around `=`, tab/newline
  after the keyword, both quote chars, quoted `>` in a parameter attribute, broken header ->
  verbatim fallback with MalformedStructure: missing separator, empty unquoted value, unterminated
  quote) and `test_streaming_recognizes_grammar_markers` (byte-wise feeds of attribute-form,
  long short-form, and invoke attribute-form openers with prose prefix: marker recognized, call
  committed, no byte lost or duplicated)
Tests executed: `ctest -R "parser|grammar|frontend"` -> 3/3 passed (0.51 s total)
Result: Phase 1 gate green — `ninfer_tool_call_parser_test`, `ninfer_tool_call_grammar_test`,
  `ninfer_qwen3_5_frontend_test`
Verification finding (bug found and fixed during this gate): `parse_function`'s Complete branch
  advanced `pos = header_base + opener.consumed`, but `consumed` already counts from the tag's `<`
  including the keyword, so `pos` overshot the tag end by 9-11 bytes. Symptom: every success-path
  test failed while every fallback-path test passed, and the pre-fix frontend test hung in the
  OutputSession feed loop. Isolated with a standalone grammar probe (grammar 100% green) and fixed
  to `pos += opener.consumed`; all gates re-run green afterwards.
Known limitations: none
Commit: this commit

## Environment note (2026-09-28)
- `ninfer_qwen3_5_frontend_test` links the FFMPEG DLLs dynamically (avcodec-63, avformat-63,
  avutil-61, swscale-10, z + transitive avdevice/swresample). `ninfer_stage_test_runtime_dlls` is
  only applied to `ninfer_device_test`, and this machine's PATH does not include
  `F:\GIT\vcpkg\installed\x64-windows\bin`, so the frontend test exe fails to load (shell exit 53)
  unless the DLLs are reachable. Workaround: the vcpkg DLLs were copied into
  `build-new-parser/tests/Release/` (build-dir only; no repo change). The frontend test was never
  executed at the P0 baseline (only the parser test was); the Phase 1 gate executes it and it is
  green.


## Phase 2
### P2.1–P2.5 Incremental parser (state machine, AST, streaming feed, one-shot + decoder migration)
Status: complete
Files changed: `src/models/qwen3_5/frontend/tool_call_stream.h/.cpp` (new), `tool_call_parser.cpp`
  (cutover: the 296-line batch `QwenToolRegionParser` removed; `parse_qwen_tool_call_output` and
  `ToolCallOutputDecoder` now drive one `ToolCallStreamParser` instance), `frontend_sources.cmake`
  (+stream source)
Implementation decision: one policy-aware region state machine (Top/ExpectFunction/FunctionHeader/
  FunctionBody/ParameterHeader/ParameterValue/ExpectWrapperClose) over the Phase-1 grammar; the
  streaming decoder feeds chunks into the same machine the one-shot entry feeds in full (P2: one
  parser for both routes). The tolerant missing-`>` header recovery and the tolerant dispatch to
  broken openers remain parse-point rules (single grammar, P3.11: no tolerant-specific grammar).
Verification findings (bug found and fixed during the gate): two missing `mode = Top` transitions
  (after a wrapper close and after a non-wrapper function close) made every region fail at the
  region end; isolated with a standalone machine probe, fixed, re-run green.
Known limitations: none (all pre-P2 behavior pinned by the test matrix)
Commit: see Phase 3 note (P2+P3 committed together, deviation documented)

## Phase 3
### P3.1–P3.11 Objective parse progress + explicit recovery policy
Status: complete
Files changed: `tool_call_stream.h` (+`ToolCallParseProgress`, `ToolCallRecoveryDecision`/
  `ToolCallRecoveryResult`, `ToolCallRecoveryPolicy`, pure `decide_tool_call_recovery`; `finish`
  gains `FinishReason`), `tool_call_stream.cpp` (parse_region now produces objective progress:
  termination class Complete/EndOfInput/Definitive, open-call state, unrecoverable empty
  function_calls; retention moved out of the parse into the pure decision; `is_close_continuation`
  gains the tool_call-wrapper structural rule), `tool_call_parser.h/.cpp` (entry + decoder carry
  `FinishReason`), `output_session.h/.cpp` (`commit_preview(FinishReason)`), `engine_core.h`
  (per-round `finish_reasons[row]` and the cancel path pass the reason)
Behavior changes vs. the old parser (intended, plan P3.4/P3.10; old pins updated to the new
  contract):
1. a parameter value cut before its closing tag is never committed in tolerant mode (old: the
   partial value was committed and executed); the region now falls back to text with
   `TruncatedTail`, whatever the finish reason was; a name-only truncation keeps its old text
   fallback;
2. inside a `<tool_call>` wrapper a `</parameter>` followed by `</function>`/`</invoke>` is only a
   real closer when the wrapper close (or the region end) follows the function close — a new
   function/invoke opener there is structurally impossible, so the value keeps scanning (P3.10
   regression fixture, function and invoke variants, tested one-shot and streaming). Outside the
   wrapper context the old ambiguity rule is unchanged (documented limitation: truly ambiguous
   wire there still commits at the closer);
3. a previously ambiguous quoted closer pair inside a wrapper (`echo '</parameter></function>'`
   with a real closer later) is now parsed with the markup preserved in the value instead of
   falling back.
Semantics preserved (pinned): strict all-or-nothing (EndOfInput -> MalformedStructure, trailing
  -> TrailingContent, undeclared -> UndeclaredTool); tolerant keeps complete calls before a
  broken later call, trailing prose (TruncatedTail), missing wrapper/function close after a
  complete value (TruncatedTail), the empty function_calls wrapper stays unrecoverable.
Tests added: `test_recovery_policy_phase3` — pure decision matrix (complete/EndOfInput/Definitive
  x strict/tolerant x open states, budget finish reasons, name-only, trailing, later-broken,
  unrecoverable), one-shot + 7-byte-chunked streaming counterparts for each rule under
  StopToken/OutputLimit/ContextCapacity, and the P3.10 regression fixtures (function + invoke,
  tolerant + strict, one-shot + streaming)
Tests updated: `test_parameter_delimiters_in_values` (quoted pair now parses; cut value now
  falls back), `test_tolerant_undeclared_and_value_cut` (value-cut case now expects text
  fallback)
Tests executed: `ctest -R "parser|grammar|frontend"` -> 3/3 passed
Result: Phase 3 gate green
Commit deviation: P2 and P3 share files deeply (the state machine is both the P2 machine and the
  P3 progress producer); reconstructing a P2-only intermediate state would fabricate history, so
  both phases are one commit: `feat: policy-free incremental tool-call parser with explicit
  recovery policy` (plan §8 recommended per-step commits; deviation recorded here and in §12 DoD)

## Phase 4
### P4.1–P4.2 Design doc + llama.cpp grammar research
Status: complete
Files changed: `docs/new_parser_phase4_design.md` (new)
Content: sampling data flow (logits -> mask -> sample), where the constraint state lives (per-request
  OutputSession, not sampling internals), speculative-decode interaction (grammar state must be the
  commit point's state, never the draft's), lazy activation (prose stays unconstrained until the
  full `<tool_call>` marker is seen; the constraint then owns the whole region until its close),
  token/byte semantics (the mask consumes decoded UTF-8 bytes of the accepted prefix, never raw
  token ids), rollback/checkpoint contract for speculative paths, and the acceptance gate for a
  later sampling-integration change. Research log above (2026-09-28 llama.cpp entry) is the primary
  source for the architecture.
Commit: this commit

### P4.3–P4.6 Pure CPU grammar-state core (ToolCallGrammarConstraint)
Status: complete
Files changed: `src/models/qwen3_5/frontend/tool_call_grammar_state.h` (new),
  `tool_call_grammar_state.cpp` (new), `frontend_sources.cmake` (+source), `tool_call_stream.h/.cpp`
  (+public `parse_tool_call_region` with the `prefix` policy flag and the `policy.prefix` branches:
  a machine state that is still open at the input end is a legal partial prefix, not a break — the
  constraint's streaming contract)
Implementation decision: a per-request `ToolCallGrammarConstraint` that advances on decoded UTF-8
  bytes and exposes `check` (verdict without mutation), `commit`, `active`, `finished`,
  `observed_bytes`. While inactive it tracks the `<tool_call>` marker prefix (single source: the
  grammar's `classify_tool_marker_prefix`); at the marker it buffers the region and re-parses the
  full region with the Phase-3 `ToolCallParsePolicy` (strict policy, prefix mode) on every step —
  the machine stays the single authority, the constraint only owns state. `check` copies the state,
  so the verdict is computed without mutation (speculative sampling can probe without committing).
  `ToolCallParsePolicy` gains `prefix` (default false; one-shot parsing is byte-identical to before —
  pinned by the existing P1/P2/P3 matrix).
Speculative semantics (implemented at the state level, per the design doc): commit is the only
  mutation point; `check` is pure; a draft/sampling path that must roll back re-advances a copy.
Sampling integration: intentionally NOT wired into the logits path in this phase (plan gate: build-
  verified CPU core + tests only). The sampling interface design is documented in
  `docs/new_parser_phase4_design.md` with the acceptance gate.
Known limitations: O(n^2) re-parse per committed byte (n = region length); acceptable for
  tool-call regions at this phase's gate, the design doc records the incremental alternative.

### P4.7 Feature flag (default off)
Status: complete
Files changed: `include/ninfer/types.h` (`ConstrainedToolDecoding` enum +
  `EngineOptions::constrained_tool_decoding`, default `Off`), `apps/cli/options.h/.cpp` (member +
  `--constrained-tool-decoding=off|tool-calls-only` flag + help text), `apps/cli/main.cpp` (mapping
  into `EngineOptions`)
Implementation decision: the flag exists end-to-end (CLI -> EngineOptions) but the engine sampling
  path ignores it in this phase (default Off keeps the sampling path bit-identical; the CPU core is
  the deliverable, per the plan gate).

### P4.8 CPU tests + gate
Status: complete
Tests added: `tests/test_tool_call_grammar_state.cpp` (new target `ninfer_tool_call_grammar_state_test`)
  — 30 tests: marker tracking (byte-wise accumulation, false-prefix clearing, trigger boundary),
  region buffering (observed_bytes accounting), parse-policy pinning (full region Complete, cut
  prefixes EndOfInput, broken prefix Definitive), check/commit asymmetry (check never mutates),
  finished/active transitions, and an exhaustive all-byte-splits streaming equivalence test (every
  split point of a realistic region with embedded newlines and `<markup>`: one-byte + long-tail
  candidates must all be accepted and finish exactly once).
Tests executed: `ctest -R "tool_call|output_session|serving|cli"` -> 4/4 passed
  (ninfer_cli_options_test, ninfer_tool_call_parser_test, ninfer_tool_call_grammar_test,
  ninfer_tool_call_grammar_state_test); full CPU suite re-run at the phase gate.
Result: Phase 4 gate green (build-verified, CPU-only, flag default off)
Known limitations: no GPU runtime verification (constraint); no sampling integration (by gate scope)
Commit: this commit

### P4 finding (root cause of the flaky byte-split test, recorded per the reporting rules)
The all-byte-splits streaming test rejected one layout-dependent byte position in early builds.
Root cause: the test itself, not the machine. The test declared `const std::string region` and
derived `const std::string_view first = region.substr(pos, 1)` — `std::string::substr` returns a
temporary `std::string`, so `first`/`second` were views into freed heap memory (use-after-free).
The bytes "worked" until the heap reused that block, then corrupted at a layout-dependent position.
Fix: `region` is a `std::string_view` over the string literal (static storage; `substr` returns a
stable view). Production code was audited for the same pattern (grep `string_view x = y.substr(`
across src/include/apps): every hit reads from a `std::string_view` receiver (view, no temporary) —
`tool_call_grammar.cpp:96`, `tool_call_stream.cpp:241`, `log_colour.h:275`, and the constraint's
`advance` (string_view parameter) — all safe. The constraint core and the Phase-3 machine are
unaffected; the machine's parse of the full valid region is deterministic `Complete + None`
(verified 5x in a throwaway probe, probe deleted).

### Final CPU-only verification (all phases, 2026-09-28)
Command: `ctest --test-dir build-new-parser -C Release --output-on-failure --parallel 8` (full suite)
Result: 160/161 passed on the first run; the single failure (`ninfer_chat_templates_test`)
  was environmental, not a code regression: it runs `tests/text/test_chat_templates.py` with the
  system Python 3.14 (the AGENTS.md miniconda py311 path does not exist on this Windows machine;
  ctest resolves Python from PATH), which lacked `jinja2`. Installed `jinja2 3.1.6` into that
  interpreter (test dependency, task-required) and re-ran: **161/161 CPU tests passed**
  (the 14 `_real` GPU tests are Skipped by design: no GPU runtime tests per the plan constraints).
Environment note: `jinja2` is now a required module of the PATH Python for the CPU suite.

### Final CPU re-verification (delivered tree, 2026-09-29)
Command: `ctest --test-dir build-new-parser -C Release --output-on-failure --parallel 8` (full suite)
State: branch `test3` at `ff784035`, worktree clean (re-verification of the delivered tree; no code
  changes since the 161/161 run above)
Result: **161/161 CPU tests passed** (100% tests passed out of 161; total 981.68 s, parallel 8).
  The 14 `_real` GPU tests remain Skipped by design (no GPU runtime tests per the plan constraints).
Build note: a bare `cmake --build` (without `--config`) in this session first rebuilt the Debug
  configuration (a reconfigure had invalidated its incremental state; its DLL-staging target also
  needed `build-new-parser/tests/Debug` to exist, created in the build dir only). The Release
  configuration was up to date (4.5 s) and is the one verified here.

## Phase 5
### P5.1 Documentation update (serving.md, cli.md, tool_call_parser.md)
Status: complete
Files changed: `docs/serving.md`, `docs/cli.md`, `docs/tool_call_parser.md` (new), `docs/new_parser_progress.md`
Implementation decision: `docs/serving.md` updated in three places — (1) the "NInfer does not apply defaults, ..."
sentence now states that grammar-constrained decoding of the tool wire syntax exists behind
`--constrained-tool-decoding` (default `off`, sampling path unchanged); (2) the tool-call section now
attributes marker recognition, accepted header forms (short, attribute, bare openers with quote-aware
names) and the value/closer rules to the single wire grammar shared by one-shot and streaming parsing,
linking the new product doc; (3) flag table row for `--constrained-tool-decoding` (server flag).
`docs/tool_call_parser.md` is the new product doc (wire format with the grammar as sole authority,
parser architecture, strict vs tolerant recovery, fundamental delimiter ambiguity, constrained
decoding feature and limits, diagnostic fields incl. `truncated_tail`). `docs/cli.md` gets the CLI
flag row. No behavior change; docs describe the implemented contract only (verified against source
and tests before writing).
Alternatives rejected: documenting the constrained feature as active (it is reserved: CPU state core +
flag only, sampling integration is the documented next step with acceptance gate in
`docs/new_parser_phase4_design.md`); keeping the old "or use constrained decoding" sentence (wrong
after the feature flag exists).
Tests added: none (docs only)
Tests executed: `git diff --check` clean; link targets verified (tool_call_parser.md exists, anchor
sections exist in both docs)
Result: product docs describe the delivered contract; the flag is documented on both generation
surfaces (CLI + server) with its reserved state stated accurately
Known limitations: the flag is reserved (no sampling change in this scope); sampling integration is
out of delivery scope per the Phase-4 gate and is recorded as the next step with its acceptance gate
Commit: (this commit)

### P5.2 Feature flag on the server binary (product coherence)
Status: complete
Files changed: `src/serve/serve_options.h`, `src/serve/serve_options.cpp`, `src/serve/generation_service.cpp`,
`tests/test_serve_options.cpp`, `tests/test_cli_options.cpp`
Implementation decision: the reserved flag now exists on both public generation surfaces, mirroring the
existing tool-call flag convention (`--tolerant-tool-calls` is a server flag): `ServeOptions` member
`constrained_tool_decoding` (default `Off`), `--constrained-tool-decoding off|tool-calls-only` parsing
with invalid/missing-value rejection (same error contract as the CLI flag), help text, and the
`ServeOptions -> EngineOptions` mapping in `GenerationService`. The CLI flag (delivered with P4) and
the `EngineOptions` member are unchanged. Default `Off` leaves sampling unchanged on both surfaces.
Regression tests: default off, both valid modes mapped, unknown mode rejected, missing value rejected,
help text contains the flag — in `test_serve_options.cpp` (new cases) and `test_cli_options.cpp` (new
cases for the CLI flag, which previously had none).
Tests executed: `ninfer_cli_options_test`, `ninfer_serve_options_test` (both green), full build green
(16 threads). Note: this work required recovering two files that a partial edit pass had damaged
(`src/serve/serve_options.h` and `src/serve/generation_service.cpp` lost adjacent original lines;
`tests/test_serve_options.cpp` lost test blocks). Recovery: `git checkout origin/master --` on the three
files, then the flag changes re-applied as single clean insertions; final diffs are purely additive
(41 insertions, 0 deletions across the four serve files; verified by `git diff origin/master`). The
duplicate/lost-line damage was detected via `git diff origin/master` before rebuilding; the build then
failed on the missing `default_thinking_budget` member, which confirmed the header damage independently.
Known limitations: the flag is a reservation (no sampling change until the Phase-4 sampling integration
lands, out of delivery scope); no server/CLI binary smoke run because this build configures
`NINFER_BUILD_APPS=OFF` (flag parsing is covered by the options tests instead)
Commit: (this commit)

## Open Questions
- (none yet)

## Research Log
### 2026-09-28 — environment verification
Question: which toolchain/vcpkg/VS versions exist on this machine, and what is the real origin/master?
Sources: `git fetch origin --prune` + `git rev-parse origin/master`; `build-windows/CMakeCache.txt`; `ls F:/GIT/vcpkg/installed/x64-windows`; `nvcc --version`; `build_native.bat`.
Conclusion: origin/master = 081bf602 (plan SHA stale); vcpkg tree = F:\GIT\vcpkg; VS18-2026 + CUDA 13.4 as planned. Branch `test3` created from origin/master.
Confidence: verified (primary sources)

### 2026-09-28 — llama.cpp reference architecture (primary sources)
Question: how does current llama.cpp keep one tool-call syntax across parsing, streaming and constrained decoding?
Sources: local checkout `E:\KI\llama.cpp` at 136887b66 (verified current): `docs/autoparser.md`; `common/chat-auto-parser.h`; `common/chat-auto-parser-generator.cpp` (`build_tool_parser_tag_tagged`); `common/parsers/qwen3-coder.cpp`; `common/chat-peg-parser.h`; `src/llama-grammar.h`.
Conclusion: (a) template diff analysis -> one structured format description -> PEG parser -> optional GBNF grammar; (b) TAG_WITH_TAGGED string values use `until(value_suffix)` — the same fundamental delimiter ambiguity the plan documents; (c) Qwen3-Coder parser: `until("\n</parameter>\n")` framing, required args in any order (permute), lazy grammar triggers on the full `<tool_call>` or complete `<function=NAME>` opener (avoids `<function` prose false positives), preserved tokens `<tool_call>`/`</tool_call>`; (d) `llama_grammar`: stacks over GBNF elements advanced per accepted token (decoded UTF-8), lazy trigger buffer with replay, clone for checkpoint/rollback. NInfer design adopts the architecture (single declarative wire grammar as source of truth, one incremental parser, pure CPU grammar-state core) without copying code.
Confidence: verified (primary sources read in full)
## Code Review & Fixes (2026-09-29)

Full review of the delivered parser stack (target state vs origin/master), with static path
tracing and throwaway probes (compiled against the repo sources with MSVC, deleted
afterwards). Findings and fixes:

- M1 (fixed): the constraint's inactive scan reset the marker candidate on every '<',
  while the machine appends and classifies (a NotMarker candidate is flushed as content
  and the breaking byte is consumed). Bidirectional divergence, verified by probes: on
  `<function<tool_call>...` the constraint overtriggered (the machine publishes
  `<function<` as content and latches at the later bare function opener); on
  `<function name="a<b">...` the machine latches at the header close while the constraint
  lost the candidate and never triggered. Fix: the inactive scan now mirrors
  ToolCallStreamParser::feed byte for byte for the marker candidate (first trigger ==
  parser latch). Regression tests: `test_second_angle_flushes_candidate_without_restarting`,
  `test_quoted_angle_bracket_in_header_keeps_candidate`. Residual (documented in design doc
  D2, not a correctness issue): the rescan after a region close retriggers on any complete
  marker, a superset of the machine's wrapper-only retry on the narrow degenerate case of
  prose between complete calls followed by a bare function/invoke opener; the machine
  remains the final authority (guidance-only difference, Phase-4 integration gate).
- L1 (fixed): docs/tool_call_parser.md said duplicate parameter names keep "the first
  value wins"; the code, serving.md, and the test all use last-value-wins. Corrected.
- L2 (fixed): ToolCallStreamParser::published_ was a write-only copy of content_.
  Removed.
- L3 (fixed): constraint header comments for finished()/observed_bytes() described
  semantics the implementation does not have (finished() is also true initially;
  observed_bytes() is the candidate/region length, not a byte count since
  construction). Corrected.
- L4 (retracted as an invalid finding): the review claimed the strict retry loop could only
  re-prove the same rejection. Wrong: decide_tool_call_recovery commits Complete slices
  under every policy — the strict check applies to failed slices only — so a broken leading
  region is recovered at a later clean wrapper even in strict (pinned by
  test_quoted_marker_before_real_call and test_incremental_quoted_marker_preserves_bytes).
  The proposed short-circuit was implemented, the parser suite caught the regression, and it
  was reverted; finish() keeps the original retry loop.
- L5 (fixed): ToolCallRecoveryPolicy::finish_reason comment claimed the decision can use it
  as a signal; the decision deliberately ignores it (pinned by test). Corrected.
- L6 (fixed): ToolCallOutputDecoder kept a duplicated pre-marker scan (whitespace hold,
  tag hold, latch, flush) instead of delegating to ToolCallStreamParser. The decoder now
  owns one machine (new accessors latched()/latched_region()/held_tail()); feed() delegates
  and finish() reads the latched region. Behavior preserved byte for byte (the decoder's
  is_format_whitespace is the machine's is_tool_format_whitespace; the non-latched finish
  path still parses the empty region as before, keeping its diagnostics).
- L8 (fixed): ToolCallStreamParser data members and publish()/latch() made private (no
  external users); the redundant parse_tool_call_region redeclaration in
  tool_call_grammar_state.h removed (declared in tool_call_stream.h, which it includes).

Verification: full suite re-run green in the Release build (161/161, 19 min; GPU tests are
not runnable here — no free GPU — consistent with the no-GPU-runtime-test constraint; the
13 _real tests are skipped for missing artifacts); probe checks: machine vs constraint on
`<function<tool_call>` (both now treat the inner marker as content; the constraint triggers
at the machine's latch byte, not at the inner marker) and on
`<function name="a<b">` (both trigger at the header close, byte 20: machine
Invalid/InvalidToolName, constraint rejects there as a definitive break).
Commit: (this commit)
