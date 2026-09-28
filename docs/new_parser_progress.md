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
(pending)

## Phase 3
(pending)

## Phase 4
(pending)

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
