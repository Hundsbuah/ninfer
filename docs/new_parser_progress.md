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
