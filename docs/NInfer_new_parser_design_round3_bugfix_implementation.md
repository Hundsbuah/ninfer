# NInfer `new_parser_design` — Round 3 Review and Bugfix Implementation Specification

## 0. Purpose, scope, baseline

This is the **third independent review** of the Qwen tool-call parser on branch `new_parser_design`
and the binding implementation specification for its fixes.

| Item                                              | Value                                                                                                                             |
|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| Reviewed HEAD                                     | `4165d227` (`fix: tool-call parser round-2 bugfixes CR1-CR6`)                                                                     |
| Pre-branch baseline used for differential testing | `081bf602` (parser before `d40a6855`/`ecf0244a`)                                                                                  |
| Review date                                       | 2026-09-30                                                                                                                        |
| Production configuration (user)                   | `ninfer serve` in **strict** mode (no `--tolerant-tool-calls`), Qwen3.6/3.8 wrapped `<tool_call>` format, agent client "oh-my-pi" |
| Original symptom                                  | "When `</tool_call>` appears in text, oh-my-pi fails with a tool_call error." No logs are available.                              |
| Deliverable of this document                      | Findings R3-01 … R3-14, required fixes, exact test changes, validated reference algorithm, acceptance gate                        |

Round 1 (F1–F10) and Round 2 (CR1–CR6) remain historical evidence. **Do not revert** their valid
parts (single grammar, one machine, opaque values, open-function non-execution, CR2/CR3 wrapper
balance, CR4 entry composition, CR5 declared identity). This round is a delta on top of them.

Record all Round-3 work in a new progress file:

```text
docs/NInfer_new_parser_design_round3_bugfix_progress.md
```

### 0.1 Hard constraints (unchanged)

- No GPU runtime tests, no `_real` tests, no inference/benchmark runs. CPU tests only.
- Build directory `build-new-parser`; never `build-windows`; at most 16 build/test threads.
- Before running an unknown test, check its source/CMake for GPU use; if unsure, do not run it.

### 0.2 Safety note for the implementing agent (read first)

The current parser **mis-parses your own tool calls** when a string argument you write contains
multi-line tool-call markup with real line breaks (finding R3-01). While implementing this
specification you will write tests and documentation that contain tool-call examples.

- Write every tool-call fixture as a **single-line C++ string literal with** `\n` **escapes** (for example `"<tool_call>\n<function=read>\n..."`) or through the existing `tool_call(...)` test helper. Never paste multi-line tool-call blocks with real newlines into a file through a tool call until R3-01 is fixed and deployed.
- This document follows that rule: every reproducer below is a single-line escaped literal.

### 0.3 How the evidence was produced

All claims marked **Verified** were executed on CPU (MSVC 19.44, `/std:c++20 /O2`), without GPU:

1. The three existing CPU suites (`test_tool_call_parser.cpp`, `test_tool_call_grammar.cpp`, `test_tool_call_grammar_state.cpp`) were compiled standalone against the HEAD sources: **all pass**.
2. Differential probes ran the same inputs through HEAD, the `081bf602` baseline parser, and a **prototype of the fixes in this document** (scratch copy, not committed).
3. A deterministic fuzz corpus (seed `20260930`, 6 000 texts × strict/tolerant = 12 000 cases, each streamed byte-wise and with two random partitions) compared streaming with one-shot.
4. The prototype was run against the existing suites; every assertion it flips is listed in §6 and is an intended behavior change.

If the local checkout still contains `build-review/` (gitignored), it holds the standalone build
script (`build.bat`: `all`, `probe <name>`, `probeold <name>`, `protoall`, `protoprobe <name>`),
the probes `probe1.cpp` … `probe12.cpp`, the baseline sources (`build-review/old/`) and the
prototype (`build-review/proto/`). They are optional helpers; this document is self-contained.

---

## 1. Executive summary

**Verdict: the branch does not fix the original failure class, and it regresses several common
cases that the pre-branch parser handled.** In strict mode (the production mode) a real tool call is
returned as plain text in all of the following situations, which are frequent in a coding-agent
workflow that edits tool-call parsers, tests and documentation:

| # | Situation (strict mode)                                                                                                         | Baseline 081bf602           | HEAD 4165d227               | Required (validated prototype) |
|---|---------------------------------------------------------------------------------------------------------------------------------|-----------------------------|-----------------------------|--------------------------------|
| 1 | `edit` whose `old_string`/`new_string` contains the **tail** of a tool-call example (closers without openers)                   | text (fail)                 | text (fail)                 | call                           |
| 2 | `write`/`bash` value with closer lines `</parameter>`,`</function>`,`</tool_call>` on own lines (heredoc, docs)                 | text                        | text                        | call                           |
| 3 | `write` value with a **complete** multi-line tool-call example followed by more text                                            | call                        | **text**                    | call                           |
| 4 | value that **ends** with a complete example, value with two consecutive examples, value with a `write` example inside a `write` | call                        | **text**                    | call                           |
| 5 | prose mentions `<tool_call>` (inline) before the real call                                                                      | call                        | **text**                    | call                           |
| 6 | prose contains an unterminated quoted marker (`<function name="x` / `<invoke name='it`) before the real call                    | call                        | **text**                    | call                           |
| 7 | fenced code closed with CRLF, closed at list indentation, or a line starting with ```` ```x``` ```` before the real call        | call                        | **text**                    | call                           |
| 8 | parameter name containing `/`                                                                                                   | call                        | **text**                    | call                           |
| 9 | bare (unwrapped) call cut by the output budget right after an embedded example                                                  | **phantom** `bash` **call** | **phantom** `bash` **call** | rejected                       |

Additional verified defects: tolerant mode commits truncated values, synthetic arguments and prose
examples (R3-03); streaming output differs from one-shot output in whitespace (R3-07, 6/12 000 fuzz
cases); every plain-text answer is recorded as `malformed_structure` in the request log (R3-09); the
Phase-4 constraint core diverges from the parser and is too slow for masking (R3-11); several docs and
help texts are stale (R3-12).

The fixes in this document were prototyped and validated: all situations 1–9 behave as required,
the fuzz corpus shows 0/12 000 streaming mismatches, and the only existing assertions that change are
the intended ones listed in §6.

### 1.1 Finding index

| ID    | Severity                           | Title                                                                                                                   | Mode              | Regression vs baseline        |
|-------|------------------------------------|-------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------------------|
| R3-01 | **HIGH**                           | One-token value-boundary lookahead rejects uniquely parseable regions that contain tool markup                          | strict + tolerant | partly (balanced examples)    |
| R3-02 | **HIGH**                           | R2-I1 scope ownership is over-broad: prose mentioning `<tool_call>` kills the real call                                 | strict + tolerant | yes                           |
| R3-03 | **HIGH** (tolerant)                | Tolerant recovery commits truncated values, synthetic arguments and prose examples; it pre-empts later complete regions | tolerant          | partly                        |
| R3-04 | **HIGH** (compat form)             | Retry re-reads markers inside a still-open or already-consumed value (phantom calls); unbounded retry cost              | strict + tolerant | no (pre-existing, worse cost) |
| R3-05 | MEDIUM                             | Header grammar: quoted values may span lines, headers are unbounded; failed marker candidates are not re-scanned        | strict + tolerant | yes                           |
| R3-06 | MEDIUM                             | FenceTracker defects (CRLF, backtick info string, list indentation), fence-blind retries, silent suppression            | strict + tolerant | yes                           |
| R3-07 | MEDIUM                             | Streaming ≠ one-shot: whitespace inside a failed candidate is published                                                 | strict + tolerant | no (pre-existing)             |
| R3-08 | **HIGH** (integrity)               | Synthetic arguments: a value containing `</parameter>\n<parameter=X>` is silently split (product decision: reject)      | strict + tolerant | no (pre-existing)             |
| R3-09 | LOW                                | Plain text without any marker reports `fallback_reason = malformed_structure`                                           | all               | yes                           |
| R3-10 | LOW                                | Parameter short-form names containing `/` are rejected (and names with whitespace)                                      | all               | yes                           |
| R3-11 | MEDIUM (blocking only for Phase 4) | Grammar-state core diverges from the parser (fences, trailing prose, declared names) and costs O(region) per candidate  | Phase 4           | n/a                           |
| R3-12 | LOW                                | Documentation, help text and progress claims are stale or false                                                         | docs              | n/a                           |
| R3-13 | LOW                                | Finish reason not passed on the active-cancellation path; dead code                                                     | engine            | no                            |
| R3-14 | MEDIUM                             | Parser CPU work runs on the engine thread; several paths are quadratic; no work bounds                                  | all               | partly                        |

---

## 2. Most likely root cause of the original failure

**Deterministically derived and verified:** the pre-branch parser failed exactly when a string
argument contained a `</parameter>` that is followed (after whitespace) by `</function>` and
`</tool_call>` — i.e. the closing tail of a tool-call example — without the matching opener inside
the same value. Its one-token rule "a closer followed by `</function>` ends the value" split the
value early, the real closers then became trailing text, and strict mode returned the entire region
as assistant text (`fallback_reason = trailing_content`). The client then saw raw tool markup instead
of a tool call. The typical trigger is an `edit` of a Markdown/test file around the end of an example
block:

````cpp
// Verified: baseline -> text (trailing_content), HEAD -> text (trailing_content), required -> 1 call
tool_call("edit", {{"path", "docs/tool_call_parser.md"},
                   {"old_string", "foo.cpp\n</parameter>\n</function>\n</tool_call>\n```\n\nNext section"},
                   {"new_string", "bar.cpp\n</parameter>\n</function>\n</tool_call>\n```\n\nNext section (edited)"}})
````

The branch replaced the depth counter by "opaque value + one-token continuation" (Round-1 F1) and
made the lookahead stricter inside `<tool_call>` (P3.10). Neither looks past the wrapper close, so the
class above is still broken, and balanced complete examples (which the old depth counter handled) are
now broken too.

**How to confirm on the production server** (no code change needed): the request log records
`request_done.result.tool_call_parse.fallback_reason`; the operational log prints
`tool markup returned as text | trailing_content | <snippet>`. Every such line with a `write`/`edit`/
`bash` snippet is an instance of this class.

---

## 3. Round-3 invariants

**R3-I1 — Consistent completion.** If a region has a complete parse in which every value boundary is
followed by a legal continuation all the way to the end, the region must be accepted even when the
one-token lookahead picked an earlier boundary that later dead-ends. Backtracking is allowed only on a
**definitive contradiction**; a continuation that merely runs out of input (truncation) stands, so
backtracking can never turn a truncated region into a complete one.

**R3-I2 — Conservative extension.** Every input that the current greedy parse accepts as `Complete`
keeps exactly its current result (only the synthetic-argument rule R3-08 may reject it). The new
logic only runs after the greedy parse and the retry chain accepted nothing.

**R3-I3 — Only a payload scope is owned.** A failed wrapper owns the bytes after its break only if the
break happened after a function opener was attempted inside it. Prose directly after a wrapper
literal owns nothing.

**R3-I4 — A cut value owns its bytes.** If the stream was cut (budget, context, stop string,
cancellation) while a value was open, no marker inside that value may start a new region. After a
natural stop the open value was never closed by the model and its bytes may be re-read.

**R3-I5 — Streaming equals one-shot, byte for byte**, for published content, terminal content,
calls, arguments and diagnostics, for every partition of the input (fuzz-verified).

**R3-I6 — No synthetic arguments.** A parameter that a preceding value could have contained as
payload (a repeated name, or a non-first name outside an unambiguous declared schema) makes the region
ambiguous in a declared tool; the region is returned as text with `ambiguous_structure`
(user product decision, 2026-09-30; plan P2.7).

**R3-I7 — Every demotion is explainable.** Suppressions (fences), markup-tolerant decisions and
ambiguity rejections are visible in diagnostics and logs.

**R3-I8 — Bounded work.** Parser work at `finish()` runs on the engine thread; it must be linear or
explicitly budgeted, and a budget overrun must fail closed (text) with a diagnostic.

---

## 4. Findings

Line numbers refer to HEAD `4165d227`.

### R3-01 — HIGH — One-token value-boundary lookahead rejects uniquely parseable regions

**Status:** Verified (probe1 S1/S1c/S1d/S1e/S1f/S1g/S8, probe2 P-C/P-H/P-B, probe7, probe11).

**Location:**

- `src/models/qwen3_5/frontend/tool_call_stream.cpp:444-500` (`ParameterValue`: the first candidate whose one-token continuation is legal ends the value, no backtracking).
- `tool_call_stream.cpp:72-121` (`classify_function_close_continuation`): inside a `<tool_call>` wrapper, `</function>` followed by `</tool_call>` is `Legal` without looking at the bytes after `</tool_call>`.

**Reproducers (strict;** `tool_call(...)` **is the helper from** `tests/test_tool_call_parser.cpp`**;**
`EX` **=** `tool_call("read", {{"path", "foo.cpp"}})`**):**

| ID   | Input                                                                                                                  | HEAD                                                          | Baseline | Required            |
|------|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|----------|---------------------|
| S1   | `tool_call("write", {{"path","docs/x.md"},{"content","# Example\n" + EX + "\nDone."}})`                                | text, `trailing_content`                                      | call     | call, exact content |
| S1c  | `tool_call("write", {{"path","docs/x.md"},{"content","Example:\n" + EX}})`                                             | text                                                          | call     | call                |
| S1d  | ```` tool_call("write", {{"path","docs/x.md"},{"content","Text\n```xml\n" + EX + "\n```\nMore text"}}) ````            | text                                                          | call     | call                |
| S1e  | `tool_call("edit", {{"path","t.cpp"},{"old_string","R\"(" + EX + ")\""},{"new_string","x"}})`                          | text                                                          | call     | call                |
| S1f  | `tool_call("write", {{"path","n.md"},{"content","Close with:\n</parameter>\n</function>\n</tool_call>\nthen stop."}})` | text                                                          | text     | call                |
| S8   | `tool_call("bash", {{"command","cat <<EOF\n</parameter>\n</function>\n</tool_call>\nEOF"}})`                           | text                                                          | text     | call                |
| S1g  | `tool_call("write", {{"path","docs/x.md"},{"content", tool_call("edit", {{"path","a"},{"old_string","b"}})}})`         | text                                                          | call     | call                |
| P-C  | value = `"Ex:\n" + E + "\n" + E + "\nDone."` with `E = tool_call("bash", {{"command","rm -rf x"}})`                    | text; **tolerant: write truncated + phantom** `bash rm -rf x` | call     | call                |
| P-H  | `tool_call("write", {{"path","d.md"},{"content", tool_call("write", {{"path","x"},{"content","hello"}})}})`            | text; tolerant: `content="hello"`                             | call     | call                |
| TAIL | the `edit` in §2 (LF and CRLF framing)                                                                                 | text                                                          | text     | call                |

**Root cause:** the value boundary is decided by a one-token lookahead. A closer that is followed by
`</function>` `</tool_call>` is accepted even when the bytes after `</tool_call>` (for example
`\nDone.\n</parameter>…`) prove that the region cannot end there. The region then fails as
`TrailingContent`; strict mode returns everything as text, tolerant mode commits the truncated call.
The one-token rule is necessary but not sufficient. The progress file and `tool_call_parser.md`
classify these payloads as "fundamentally ambiguous"; they are not — each has exactly one consistent
complete parse (the tests explicitly excluded them, see `test_payload_adversarial_corpus_round_trip`).

**Required fix:** add **Stage 2 — consistent completion** (reference algorithm in §8.4/§8.5):

1. Stage 1 stays exactly as today (greedy parse + retry chain, with the R3-02/R3-04/R3-06 fixes). If Stage 1 accepts a region, the result is final (R3-I2), subject only to R3-08.
2. Otherwise run Stage 2 on every admissible base of the Stage-1 retry chain (base 0 and every retry base). Stage 2 parses the canonical grammar (no tolerant header repairs) and chooses each value boundary as follows:
   - Candidate order per value: **pass 1 (balanced)** — closers of the value's family that are not consumed by a nested same-family opener inside the value (depth counting, same family only); **pass 2 (lazy)** — all closers of the value's family in order. Pass 2 runs only when pass 1 produced no candidate whose continuation is non-contradicted.
   - A candidate is skipped if its one-token continuation is `Invalid` (existing `classify_close_continuation`).
   - **Plausibility:** if the continuation is a parameter opener of the same call, the next name must not repeat a name of the current call (including the value being closed) and, when the tool has an unambiguous non-empty declared schema, must be declared. Otherwise skip the candidate.
   - **Canonical framing:** once a viable candidate of this value was contradicted, every later candidate of the same value must be preceded by `\n` (the real value close is serialized as `\n</parameter>`). This rejects "real call followed by prose that quotes closers inline" (R6).
   - A candidate is abandoned only if its continuation ends **Definitive**. If it ends **EndOfInput** the candidate stands and the region result is EndOfInput. If all candidates of a value are contradicted, the value is open to the end: the result is EndOfInput (never Complete).
3. Selection over the chain bases: among bases whose Stage-2 result is Complete, take the **earliest base whose chosen path used only pass-1 (balanced) boundaries**; otherwise, if exactly one base completed, take it; if two or more bases completed and none is balanced, reject the region with the new reason `AmbiguousStructure`.
4. A Stage-2 acceptance sets the new diagnostic flag `markup_tolerant_completion = true` (R3-I7) and then passes through the synthetic-argument rule (R3-08).

**Pitfalls (each observed in the prototype):**

- Backtracking on EndOfInput turns a truncated region into a phantom: with the input cut after the second call's `</function>` in `test_complete_and_partial_next_entry_have_same_boundary_result`, an EndOfInput-backtracking completer let the first value absorb the second call. Only Definitive may trigger backtracking.
- Pure lazy order (without the balanced pass) produces phantoms for P-C: the first value ends at the first example and the second example becomes a real `bash` call. The balanced pass must come first.
- Without the canonical-framing rule, a real call followed by `"\nNote: calls end with </parameter>\n</function>\n</tool_call>"` is accepted with the note appended to the argument.
- Without the chain selection rule, an unfenced prose example followed by a real `write` that itself contains an example is ambiguous (S9'); the balanced preference selects the real call (verified R2).
- Do not implement Stage 2 as a second hand-written grammar: reuse the Stage-1 transition helpers (`classify_top_level_entry`, header parsers, `classify_close_continuation`, the function-header checks for name validity and declared identity). The prototype duplicated the machine for speed of validation only.
- No unbounded recursion (R3-I8): explicit stack or a hard depth bound that fails closed.

**Tests (new, all modes, all finish reasons where relevant, streaming at chunk sizes 1/2/3/5/7 and
every split for short fixtures):** S1, S1c, S1d, S1e, S1f, S8, S1g, P-C, P-H, TAIL (LF and CRLF),
R4 (two unfenced examples in one value), R5 (two fenced examples), R2 (prose example + real `write`
with an example → the real `write`), R6 (must stay rejected), R1 (documented residual, see §9),
probe7 scale cases (50/200/500/2000 examples and closer triples: exact content round-trip).

**Acceptance:** all reproducers above are parsed with byte-exact argument values; no previously
accepted input changes (R3-I2) except via R3-08.

---

### R3-02 — HIGH — R2-I1 scope ownership is over-broad

**Status:** Verified (probe1 S2/S2b; HEAD inverted the baseline assertion of
`test_quoted_marker_before_real_call`).

**Location:** `tool_call_stream.cpp:717-725` (`finish()`): any break with
`wrapper_at_break != None` ends the retry chain.

**Reproducers:**

- S2: `"I will emit a <tool_call> block now.\n" + tool_call("read", {{"path","a.txt"}})` — HEAD: text; baseline and required: 1 call, content `"I will emit a <tool_call> block now."`.
- S2b (the baseline test fixture): `"explaining <tool_call>\\n<function=shell>\\n<function=command>\\nprintf broken\\n</parameter>\\n</function>\\n</tool_call> then the real turn\n" + tool_call("bash", {{"command","echo ok"}})` — HEAD: text; required: 1 call `bash`.

**Root cause:** the prose `<tool_call>` latches, the region breaks at the first non-whitespace byte
after the wrapper literal (`ExpectFunction` → `NoEntry`) while the wrapper is formally open, and
R2-I1 treats every such break as an owned payload scope. No function was opened, so there is no
payload to protect; the later real call is discarded. `<function=read>` mentions are unaffected
(wrapper `None`), so the behavior is also inconsistent between marker forms.

**Required fix:** record in `ToolCallParseProgress` a flag `prose_after_wrapper`. Set it when the
break occurs in `ExpectFunction` with a wrapper open, the entry classification is `NoEntry`, and the
break byte does **not** attempt a function opener: `text[i] != '<'` and, after skipping an optional
`|im_start|>`, the bytes do not start with `function` or `invoke`. Evaluate this before the tolerant
header-repair branch (`tool_call_stream.cpp:280`) and break immediately with `MalformedStructure` at
`i`. In `finish()`, a break with `wrapper_at_break != None` ends the chain **unless**`prose_after_wrapper` is set; then the retry searches from the break offset.

**Pitfalls:** do not relax nested wrappers (`<tool_call>\n<tool_call>…`, F5), `'<'`-led broken
openers (`<tool_call>\n<function bad.name>…`, CR1 "invalid outer header"), or breaks at/after a
function header (CR1 "invalid name"): all remain owned. Verified: every CR1/F5 test still passes.

**Tests:** restore the baseline assertions of `test_quoted_marker_before_real_call` and
`test_incremental_quoted_marker_preserves_bytes` (see §6); add S2, `<tool_call>` + `.`/`` ` ``/`\\`
prose variants, and the CR1 negatives unchanged.

**Residual (documented, not fixed):** `"I will now emit <tool_call>\n" + tool_call(...)` (the prose
literal is immediately followed by the real wrapper) remains a nesting break (F5) and is rejected;
`tool_call("bad.name", …) + tool_call("bash", …)` remains rejected (CR1).

---

### R3-03 — HIGH (tolerant) — Tolerant recovery commits wrong or truncated calls

**Status:** Verified (probe1 S1f/S8/S1g tolerant, probe2 P-C/P-D/P-H tolerant, P-A2/P-A3).

**Location:** `tool_call_stream.cpp:683-745` (`finish()`: tolerant recovery is decided per base
**before** later bases are tried); `tool_call_stream.h:121-160` (`decide_tool_call_recovery` ignores
the finish reason and commits calls before any Definitive break).

**Reproducers (tolerant):**

- P-D: `"For example:\n" + tool_call("read", {{"path","ex.txt"}}) + "\nNow the real one.\n" + tool_call("bash", {{"command","echo real"}})` — HEAD and baseline commit the **prose example** `read(ex.txt)`; required: `bash(echo real)`.
- S8 cut after `"\nEOF"` with `OutputLimit` — HEAD commits `bash("cat <<EOF")` (truncated command).
- P-C, P-H, S1f, S1g — HEAD commits truncated/synthetic values (see R3-01).

**Root cause:** (a) recovery of an earlier incomplete region pre-empts a later complete region;
(b) "complete calls + trailing bytes" is committed even when the trailing bytes may be the rest of
the last value (they contain a value closer, or the stream was cut before the real closers);
(c) in tolerant mode, prose after complete calls enters the header-repair path and is reported as
`MalformedStructure` instead of `TrailingContent` (imprecise failure class, plan P3.2).

**Required fix (Stage 3, runs only if Stage 1 and Stage 2 accepted nothing and Stage 2 did not
return **`AmbiguousStructure`**):**

1. Iterate the Stage-1 chain in order; commit the first attempt the recovery decision accepts.
2. `EndOfInput` attempts: unchanged (commit function-closed calls; an open value is never committed).
3. `Definitive` attempts whose committed calls contain at least one parameter value: commit only if the stream ended **naturally** (`FinishReason::StopToken`, or `None` = unknown in the one-shot API) **and** the tail from the break offset to the end contains no `</parameter>` or `</param>` literal. `StopString`, `OutputLimit`, `ContextCapacity`, `Cancelled` are cuts: no commit.
4. `Definitive` attempts whose committed calls have no parameters: unchanged.
5. `unrecoverable_break`: never.
6. Diagnostic when nothing is committed: the strict failure class of the first attempt (`Definitive` → its failure; `EndOfInput` → `TruncatedTail` as today).
7. Move rules 3–4 into `decide_tool_call_recovery` (pure, testable): it receives the finish reason and a precomputed `tail_has_value_closer` flag. Update the comment "Informational only" on `ToolCallRecoveryPolicy::finish_reason`: a cut still never makes an open value safe; it now additionally forbids committing across a definitive tail.
8. In tolerant mode, `Top`/`NoEntry` after at least one complete call must report `TrailingContent` when the header repair fails (same class as strict).

**Pitfalls:** the value-closer check must ignore wrapper/function closers (a stray `</tool_call>`
after a complete call stays recoverable); an attempt whose calls have no parameters cannot be
extended by a later closer and stays recoverable (test "the pre-break call was lost by tolerant
nesting recovery" keeps passing).

**Tests:** P-D tolerant (real call wins), S8-cut tolerant with `OutputLimit` (no commit) and with
`StopToken` (documented residual commit, see §9), the four existing tolerant tests that flip for cut
reasons (§6).

---

### R3-04 — HIGH (compatibility form) — Retry re-reads markers inside a value

**Status:** Verified (probe2 P-A/P-A2/P-A3, probe3 P-G6, fuzz case 4107).

**Location:** `tool_call_stream.cpp:730-741`: for `EndOfInput` the retry searches from `base + 1`,
i.e. inside the structure just consumed, including an open or already-closed value.

**Reproducers:**

- P-A (strict, both `StopToken` and `OutputLimit`): `"<function=write>\n<parameter=path>\nd.md\n</parameter>\n<parameter=content>\nExample:\n" + tool_call("bash", {{"command","rm -rf x"}})` → HEAD and baseline execute `bash("rm -rf x")` from inside the truncated `write` payload.
- P-A3 (tolerant): `"<function=write>\n<parameter=content>\nEx:\n" + E + "\nDone.\n</parameter>\n"` → HEAD tolerant executes the embedded example.
- P-G6: 4 000 bare regions with open values (175 KB) → 57 ms in HEAD (quadratic retry chain).

**Root cause:** the retry treats an EndOfInput region as if nothing had been consumed; payload bytes
of an open or closed value become recovery entries. This is the bare-form analog of CR1 that R2-I1
does not cover (R2-I1 only protects wrapper scopes).

**Required fix:**

1. `EndOfInput` without an open value: retry from `base + break_offset` (after the consumed structure), never `base + 1`.
2. `EndOfInput` with an open value: retry from the value start **only after a natural stop** (`StopToken`/`None`); after a cut (`StopString`, `OutputLimit`, `ContextCapacity`, `Cancelled`) end the chain (R3-I4).
3. Always use `max(base + break_offset, entry_end)` as the search start.
4. Bound the chain (for example 256 attempts); on overflow end the chain and set a diagnostic.

**Why natural stop is allowed:** Qwen3.6 emits wrapped calls. An open bare value followed by a
wrapped call after a natural stop is almost always prose mentioning `<function=…>` fragments before
the real call (the F7 `bare_before` case); after a cut it may be a real truncated call whose payload
contains an example. Verified: `bare_before` keeps working with `StopToken`, P-A is rejected with
`OutputLimit`.

**Tests:** P-A/P-A2/P-A3 × {`StopToken`, `OutputLimit`} × {strict, tolerant}; F7 `bare_before` with a
cut reason (new expectation: rejected); a deterministic work-bound test for P-G6 (§4 R3-14).

---

### R3-05 — MEDIUM — Header grammar and marker-candidate re-scan

**Status:** Verified (probe4 all cases, probe2 P-B, probe3 P-G7).

**Location:** `tool_call_grammar.cpp:28-51` (`parse_value`: a quoted value may contain any byte
including line breaks, `find(quote)` scans to the end); no header length bound;
`tool_call_stream.cpp:655-670` (F8 split retains only a trailing `<` of a failed candidate).

**Reproducers:**

- `"Attr form: <function name=\"x\n" + tool_call("bash", {{"command","ls"}})` → HEAD: text (the quote swallows the real call); baseline and required: 1 call.
- `"The <invoke name='it\n" + tool_call(...)` → same (an apostrophe in prose).
- P-B: `tool_call("bash", {{"command","grep '</parameter><parameter name=\"' file"}})` → HEAD: text (the unterminated quote makes the continuation `NeedMore`, accepted as a boundary); required: 1 call, exact command.
- P-G7: prose `"see <function name=\""` + 40 000 bytes without a quote → 9.4 ms CPU and the whole rest of the answer is held until the end (quadratic re-classification per fed byte).

**Required fix:**

1. `parse_value`: a CR or LF inside a quoted value → `Invalid` (headers are single-line; `>` inside quotes stays allowed, P1.5 unchanged).
2. Header length bound: `kMaxToolHeaderBytes = 1024` bytes after the keyword; a header without its terminating `>` within the bound → `Invalid` (a `NeedMore` beyond the bound becomes `Invalid`).
3. Generalized F8 re-scan in `ToolCallStreamParser::feed`: when a candidate becomes `NotMarker`, publish its bytes up to the next `<` after position 0 and re-feed the remaining bytes through the candidate machine (they already passed the fence tracker). Hold the trailing format whitespace of the published head in `pending_ws_` (R3-07). Implement this transition once and share it with the Phase-4 constraint core (replace `failed_marker_candidate_retained`).

**Tests:** grammar tests (quoted `\n` → `Invalid`; 1 025-byte header → `Invalid`; `>` inside quotes
still `Complete`); the probe4 cases one-shot and streaming; P-B; a deterministic bound test for P-G7.

---

### R3-06 — MEDIUM — FenceTracker defects and fence-blind retries

**Status:** Verified (probe1 S3b/S3c, probe2 P-E1).

**Location:** `tool_call_stream.cpp:534-620` (`FenceTracker::consume`), `finish()` retry search.

**Defects and reproducers (strict; real call =** `tool_call("read", {{"path","a"}})`**):**

- CRLF close: ```` "```xml\r\n" + EX + "\r\n```\r\n" + real ```` → HEAD: text. At `tool_call_stream.cpp:572-578` the `\r` after the closing run clears `close_pending_`, so the fence never closes.
- Backtick info string: ```` "```x``` is inline code\n" + real ```` → HEAD: text. CommonMark: a backtick fence's info string cannot contain a backtick; this line is inline code, not a fence.
- List indentation: ```` "- step:\n ```bash\n make\n ```\nNow calling.\n" + real ```` → HEAD: text (a closer at 4 spaces is not recognized although it is valid inside the list item).
- Retry entries ignore fences: after a failed prose region, a fenced complete example is re-read as a region (tolerant: phantom commit of the example).
- Silent loss: ```` "Here is code:\n```python\nprint(1)\n\n" + real ```` (unclosed fence) → text with `marker_seen = false`: no log line explains why the call disappeared.

**Required fix:**

1. Rewrite `consume` with explicit per-line phases `Indent → Run → Tail → Body` (reference §8.3): CR, LF-only framing and trailing spaces/tabs after a closing run keep the close valid; a backtick in a backtick opener's info string cancels the fence (the line is inline code); a closing run may be indented up to `opener_indent + 3` spaces.
2. Retry candidates in `finish()` skip markers inside a recognized fence. Compute the fence state by running a fresh `FenceTracker` over `region_` from its start (the same bytes in one-shot and streaming).
3. Keep the design decision "an unclosed fence stays open through EOF" (integrity), but make it visible: new diagnostics `fenced_markers_suppressed` (count of complete top-level markers that a fence suppressed, pre-latch and retry) and `ended_in_unclosed_fence` (bool). Count suppressed markers with a shadow candidate scan over fenced bytes (the same `classify_tool_marker_prefix` transition, never latching). The operational log emits a warning when the response has no tool call, `ended_in_unclosed_fence` is true and `fenced_markers_suppressed > 0`.

**Tests:** the three defect reproducers (one-shot + every split); fence-aware retry
(```` "Use <function=read> x\n```xml\n" + EX + "\n```\n" ```` tolerant → no commit); unclosed-fence
diagnostics; existing CR6 tests (reason change only, §6).

---

### R3-07 — MEDIUM — Streaming output differs from one-shot output

**Status:** Verified (fuzz: HEAD 6/12 000 mismatches; prototype 0/12 000).

**Example:** `"\"echo '</parameter>'\n\r\n</tool_call>\n<function\n" + tool_call("bash", {{"command","ls"}})`
→ one-shot content ends with `<function`, streaming content ends with `<function\n`.

**Root cause:** a candidate broken by a whitespace byte is published including that byte; one-shot
right-trims the content before an accepted region, streaming already published it.

**Required fix:** covered by R3-05 item 3 (hold the trailing whitespace of every published failed
candidate head in `pending_ws_`).

**Tests:** add the fuzz as a permanent deterministic test: fixed seed, fragment corpus of §10.3,
≥ 2 000 texts, strict and tolerant, byte-wise and two random partitions; assert identical calls,
arguments, diagnostics and `visible + terminal.content`.

---

### R3-08 — HIGH (integrity) — Synthetic arguments (product decision: reject)

**Status:** Verified (probe12 E1/E2/E3 identical in baseline, HEAD and prototype-before-rule).

**Reproducers (strict, declared tools** `edit{path,old_string,new_string}`**,** `write{path,content}`**):**

- E1: `"<tool_call>\n<function=edit>\n<parameter=path>\nd.md\n</parameter>\n<parameter=old_string>\nx\n</parameter>\n<parameter=content>\nhello\n</parameter>\n<parameter=new_string>\nNEW\n</parameter>\n</function>\n</tool_call>"` → today `edit(old_string="x", content="hello", new_string="NEW")`: the real `old_string` may have been `"x\n</parameter>\n<parameter=content>\nhello"`; a truncated `old_string` can match the wrong place.
- E2: `"<tool_call>\n<function=write>\n<parameter=path>\nd.md\n</parameter>\n<parameter=content>\nIntro\n</parameter>\n<parameter=content>\ntail\n</parameter>\n</function>\n</tool_call>"` → today `content="tail"` (duplicate repair, silent loss of `Intro…`).
- E3 (hallucinated name): same shape as E1 with `mode` instead of `content`.

**Decision (user, 2026-09-30):** in a **declared** tool (the contract contains the tool name):

- any repeated parameter name, or
- any parameter name at position ≥ 2 (not the first parameter) that is not declared, when the tool's schema is unambiguous and has at least one property,

makes the region ambiguous: return the region verbatim as text with the new fallback reason
`ambiguous_structure`. A non-declared **first** parameter keeps today's behavior (emitted with
`schema_mismatch_arguments`), because no preceding value could have contained it. Tools that are not
in the contract (legacy contract, tests only in production terms) keep "last value wins".

**Required fix:**

1. `parse_region` keeps every parameter occurrence in order (remove the merge at `tool_call_stream.cpp:469-486`); `duplicate_parameters_repaired` is computed at the entry for legacy tools only.
2. Apply the rule at the single output boundary in `parse_qwen_tool_call_output`, after any stage accepted a region (Stage 1, Stage 2 or Stage 3), before normalization.
3. `ToolCallParseFallbackReason::AmbiguousStructure` (`"ambiguous_structure"`) in `include/ninfer/types.h`, `tool_call_parse_fallback_reason_name`, `ToolCallParseFailure`, the `to_fallback_reason` mapping, the request log and `docs/serving.md`. It is introduced together with Stage 2, which uses it for the multi-base ambiguity (R3-01 item 3; §10.1 step 5).

**Pitfalls:** Stage 2's plausibility rule and this rule are consistent: Stage 2 never produces a
synthetic argument, so P-H and S1g (which need Stage 2) are accepted, while E1/E2/E3 (accepted by
Stage 1) are rejected. Do not "prefer the absorbing parse" in Stage 1 — it would silently append
markup to the previous value when the model really hallucinated a parameter.

**Tests:** E1, E2, E3 → `ambiguous_structure`; undeclared first parameter → emitted with mismatch;
legacy contract duplicate → last wins; P-H/S1g → accepted; §6 lists the flipped duplicate tests.

---

### R3-09 — LOW — No-marker output reports `malformed_structure`

**Status:** Verified (probe1 S4: baseline `none`, HEAD `malformed_structure`).

**Location:** `tool_call_parser.cpp:474-478`.

**Impact:** every plain-text answer with tools enabled is recorded as
`"fallback_reason": "malformed_structure"` in the request log (misleading statistics; the operational
log hides it only because `marker_seen` is false).

**Required fix:** return default diagnostics (`fallback_reason = None`, `marker_seen = false`) when
no marker was seen. Update the CR6 fence tests (§6).

---

### R3-10 — LOW — Parameter short-form names with `/` (and whitespace)

**Status:** Verified (probe1 S5b `<parameter=a/b>`: baseline call, HEAD text; S5 `<parameter=file path>`: baseline call, HEAD text).

**Location:** `tool_call_grammar.cpp:44-47` (unquoted value stops at `/` and whitespace). The Qwen
template renders `<parameter=` + raw property name + `>`.

**Required fix:** allow `/` in the **parameter** short-form unquoted value (function names keep the
strict rule; `is_valid_tool_name` rejects `/` anyway). Names with whitespace: optional — accept only
when the bytes up to `>` on the same line, trimmed, equal a declared parameter name of the current
tool (requires contract access in the parameter-header step and in the continuation classification);
otherwise document the limitation and pin the rejection with a test.

---

### R3-11 — MEDIUM (blocking only for Phase 4) — Grammar-state core diverges from the parser

**Status:** Verified (probe2 P-F).

**Location:** `src/models/qwen3_5/frontend/tool_call_grammar_state.cpp:34, 48-66, 72-100`;
`docs/new_parser_phase4_design.md:104-110, 136, 241`.

**Defects:**

1. The inactive trigger scan has no fence tracker: ```` "```xml\n<tool_call>\n<function=read>\n" ```` activates the constraint, while the parser never latches there. The design doc's claims "the first trigger is exactly the parser's latch" and "mirrors the machine's feed rule byte for byte" are false since CR6.
2. After a closed region, trailing prose is `Allowed` (constraint inactive), but strict mode rejects the whole region as `TrailingContent`: the constraint does not prevent the failure it exists for.
3. `enforce_declared_names = false`: the constraint admits undeclared function names that the parser rejects (`UndeclaredTool`).
4. Every `check()` copies the buffer and re-parses the whole region: 0.12–0.15 ms per candidate token at a 200 KB open value, i.e. seconds per decoding step for a 150 k vocabulary.
5. It does not know Stage 2/3 semantics or the R3-05 re-scan.

**Required action:** no production impact while `tool-calls-only` fails at startup (keep it failing).
Before enabling Phase 4: share the pre-latch feed transition (fence + re-scan) with the parser,
reject non-whitespace after a closed region unless it starts a marker, pass the declared tool set,
make the state incremental, and add parser/constraint equivalence tests. Correct the design doc now.

---

### R3-12 — LOW — Documentation, help text and progress claims

| Location                                                                     | Problem                                                                                                                                                           | Required change                                                              |
|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| `docs/serving.md:260-262`                                                    | "Grammar-constrained decoding … is available behind `--constrained-tool-decoding`"                                                                                | State that `tool-calls-only` is not implemented and fails at startup         |
| `docs/serving.md:264-275`                                                    | "preserve … balanced nested `<parameter=...>...</parameter>` text"; "Later content is still examined: the first region … that parses becomes the structured turn" | Describe Stage 1/2 boundary semantics and the R3-02/R3-04/R3-06 retry rules  |
| `docs/serving.md:279-284`                                                    | Tolerant keeps "a single final call cut by the output budget" and "an undeclared tool name stays structured" — both false since F2/CR5                            | Rewrite to the R3-03 rules                                                   |
| `docs/serving.md:1078-1089`                                                  | duplicate repair "instead of demoting the call to text"; reason list                                                                                              | Document R3-08, `ambiguous_structure`, new diagnostics, `none` for no marker |
| `src/serve/serve_options.cpp:227-230`                                        | `--tolerant-tool-calls` help: "keep a final call cut by the output budget and an undeclared name"                                                                 | Match R3-03                                                                  |
| `docs/tool_call_parser.md:31`                                                | "balanced nested … text" preserved                                                                                                                                | Match R3-01                                                                  |
| `docs/tool_call_parser.md:111-125`                                           | payloads ending with closers called "ambiguous by construction"                                                                                                   | Explain consistent completion and the residual list (§9)                     |
| `docs/tool_call_parser.md` "Code fences"/"Strict vs. tolerant"/"Diagnostics" | outdated after R3-03/R3-06/R3-08                                                                                                                                  | update                                                                       |
| `docs/new_parser_phase4_design.md`                                           | false equivalence claims (R3-11)                                                                                                                                  | correct                                                                      |
| `docs/new_parser_progress.md`                                                | "one-shot and streaming stay equal by construction" (false, R3-07); closer-ending payloads "fundamentally ambiguous" (false, R3-01)                               | mark as superseded by Round 3                                                |
| `src/serve/operational_log.cpp:355`                                          | snippet search only for `<tool_call>`; bare regions log no snippet                                                                                                | use the region start (first marker)                                          |

**Hygiene (AGENTS.md "one current authority"):** after Round 3 is accepted, fold the stable rules into
`docs/tool_call_parser.md` and remove the temporary plan/progress/round documents
(`ninfer_new_parser_implementation_plan.md`, `new_parser_progress.md`, the three `NInfer_new_parser_design_*` round documents and this one) — only with the user's approval.

---

### R3-13 — LOW — Engine finish reason and dead code

- `src/runtime/engine/engine_core.h:1093` (`cancel_active_requests`): `commit_preview()` is called without `FinishReason::Cancelled` although the preview was `preview_terminal(Cancelled)`. With R3-03/R3-04 the finish reason becomes decision-relevant; pass `FinishReason::Cancelled` (the other terminal site, `complete_cancelled`, already does).
- `find_tool_marker(…, bool wrapper_only)` (`tool_call_grammar.h:150-153`): the parameter is unused — remove it.
- `failed_marker_candidate_retained` (`tool_call_grammar.h`): replaced by the shared re-scan (R3-05).
- `is_ascii_alphanumeric` in `tool_call_parser.cpp`: unused.

---

### R3-14 — MEDIUM — Parser work on the engine thread

`OutputSession::commit_preview` (`output_session.cpp:676`) runs `ToolCallOutputDecoder::feed/finish`
synchronously in the engine loop (`engine_core.h:1331`), so parser CPU time delays every concurrent
request. Measured (HEAD / prototype):

| Workload                                                  | HEAD               | Prototype                                |
|-----------------------------------------------------------|--------------------|------------------------------------------|
| 500 KB plain `write`                                      | 2.2 ms             | 2.7 ms                                   |
| 4 000 bare regions with open values, 175 KB, natural stop | 57 ms (quadratic)  | 54 ms (still quadratic; bound the chain) |
| same, cut reason                                          | 57 ms              | 0.7 ms                                   |
| unterminated quoted header candidate, 40 KB               | 9.4 ms (quadratic) | 0.9 ms                                   |
| Stage 2: 2 000 closer triples in one value (86 KB)        | rejected           | 11.8 ms, exact                           |
| Stage 2: 2 000 complete examples in one value (190 KB)    | rejected           | 3.4 ms, exact                            |

**Required:** bound the retry chain (R3-04), bound headers (R3-05), give Stage 2 a deterministic step
budget (for example `max(200 000, 32 × region bytes)` candidate/glue steps) that fails closed with a
diagnostic, and use `O(1)` memo keys (hash of position, mode, family, wrapper, `fc_had_call`, current
call name and parameter-name set) — not string concatenation. Add deterministic work-bound tests
(assert step counters, not wall-clock time).

---

## 5. Behavior summary after the fixes

| Stage           | Runs when                                | Decides                                                                                                                        |
|-----------------|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Pre-latch feed  | every byte before a latch                | fence tracking (R3-06), marker candidate with re-scan and whitespace hold (R3-05/R3-07)                                        |
| Stage 1         | at `finish()`                            | today's greedy region parse; retry chain with R3-02/R3-04/R3-06 rules; first `Complete` wins                                   |
| Stage 2         | Stage 1 accepted nothing                 | consistent completion per chain base (R3-01); earliest balanced completion, else unique completion, else `ambiguous_structure` |
| Stage 3         | tolerant, nothing accepted, no ambiguity | R3-03 recovery rules over the chain                                                                                            |
| Output boundary | a region was accepted                    | CR5 identity check, R3-08 synthetic-argument rule, normalization                                                               |

Natural stop = `StopToken`, or `None` (unknown; one-shot API default). Cut = `StopString`,
`OutputLimit`, `ContextCapacity`, `Cancelled`. The engine must always pass the real reason (R3-13).
The validated prototype classified `StopString` as natural; this specification deliberately treats it
as a cut because a client stop sequence can end the output inside a value. The difference only
affects tolerant recovery and open-value retries after a `StopString` finish.

---

## 6. Existing assertions that must change (intended)

These are the only assertions of the current suites that the validated prototype flips.

| Test (file tests/test_tool_call_parser.cpp unless stated)                                                            | Current expectation                           | New expectation                                                                                                                     | Reason       |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|--------------|
| `test_quoted_marker_before_real_call` (line 686)                                                                     | text, `MalformedStructure`                    | 1 call `bash`, content `"explaining " + quoted + " then the real turn"`, reason `None` (the baseline assertion)                     | R3-02        |
| `test_incremental_quoted_marker_preserves_bytes` (line 726)                                                          | no call; `visible + terminal.content == text` | 1 call; `visible + terminal.content == "explaining " + quoted + " then the real turn"`; reason `None`, `structured_call_count == 1` | R3-02        |
| `test_fenced_content_never_latches` (line 2181): backtick, tilde, unclosed, longer-fence, info-string, CRLF fixtures | reason `MalformedStructure`                   | reason `None`, `marker_seen == false`, `fenced_markers_suppressed >= 1`; unclosed: `ended_in_unclosed_fence == true`                | R3-09, R3-06 |
| same test, "CR6 call before fence (tolerant)"                                                                        | `{"bash"}`, `TruncatedTail` for all reasons   | `StopToken`: unchanged; `StopString`/`OutputLimit`/`ContextCapacity`/`Cancelled`: no call, reason `TrailingContent`                 | R3-03        |
| `test_function_calls_close_then_remainder` (line 1896), "CR3 trailing prose (tolerant)"                              | `{"read"}`, `TruncatedTail` for all reasons   | split as above                                                                                                                      | R3-03        |
| `test_tolerant_rejects_undeclared_tools` (line 2045), "CR5 valid then undeclared (tolerant)"                         | `{"read"}`, `TruncatedTail` for all reasons   | `StopToken`: unchanged; cut reasons: no call, reason `UndeclaredTool`                                                               | R3-03        |
| `test_recovery_policy_phase3` (line 2590), "complete calls before trailing prose were not recovered"                 | recovered for every reason                    | recovered only for `StopToken`/`None`                                                                                               | R3-03        |
| `test_duplicate_parameter_keeps_last_value` (line 2306)                                                              | last value wins for a declared tool           | `ambiguous_structure`, text; add a legacy-contract variant that keeps last-wins                                                     | R3-08        |
| `test_duplicate_parameters_keep_last_value` (line 896)                                                               | identical/conflicting duplicates repaired     | same change as above                                                                                                                | R3-08        |
| `test_unsupported_schema_uses_legacy_policy` (line 557)                                                              | non-first `undeclared` emitted with mismatch  | move `undeclared` to the first position (legacy inference still covered); add: non-first undeclared → `ambiguous_structure`         | R3-08        |
| `tests/test_request_log.cpp`                                                                                         | reason names/fields                           | add `ambiguous_structure` and the new diagnostic fields                                                                             | R3-06, R3-08 |

The grammar and grammar-state suites did not flip in the prototype; the R3-05 grammar additions are
new assertions only.

---

## 7. New permanent tests

Add to `tests/test_tool_call_parser.cpp` (and the grammar suite where noted). Every structured case
runs strict and tolerant, one-shot and streamed (chunk 1/2/3/5/7 and every split for fixtures below
512 bytes), with every finish reason where the decision depends on it.

 1. **Markup-bearing payloads (R3-01):** S1, S1c, S1d, S1e, S1f, S8, S1g, P-C, P-H, TAIL (LF, CRLF), R4, R5, and the scale cases (50/200/500/2 000 examples and closer triples). Assert byte-exact argument values.
 2. **Adversarial corpus completion:** extend `test_payload_adversarial_corpus_round_trip` with the entries it currently excludes (payloads that end with a closer literal or a closer triple); they must round-trip.
 3. **Chain selection:** R2 (unfenced prose example + real `write` containing an example → the real `write`); R6 (must stay text); R1 (documented residual, assert the current accepted outcome and reference §9 so a future change is deliberate).
 4. **Scope ownership (R3-02):** S2 and variants; CR1/F5 negatives unchanged.
 5. **Retry ownership (R3-04):** P-A, P-A2, P-A3 × {`StopToken`, `OutputLimit`} × {strict, tolerant}; F7 `bare_before` with a cut reason → rejected.
 6. **Headers and re-scan (R3-05):** probe4 cases; P-B; grammar: quoted CR/LF → `Invalid`, 1 025-byte header → `Invalid`, `<parameter=a/b>` → name `a/b`.
 7. **Fences (R3-06):** CRLF close, backtick info string, list indentation, fence-aware retry, unclosed-fence diagnostics.
 8. **Synthetic arguments (R3-08):** E1, E2, E3, first-position undeclared, legacy duplicate.
 9. **Tolerant (R3-03):** P-D; S8 cut with `OutputLimit` (no commit); trailing prose with a value closer in the tail (no commit); stray `</tool_call>` after a complete call (commit, natural stop).
10. **Streaming equivalence fuzz (R3-07):** deterministic seed, fragment corpus §10.3.
11. **Work bounds (R3-14):** 4 000 bare open regions; 40 KB unterminated quoted header; Stage 2 on 2 000 closer triples — assert deterministic step counters below the budget, and fail-closed behavior when a test forces a tiny budget.
12. **Diagnostics:** `ambiguous_structure`, `markup_tolerant_completion`, `fenced_markers_suppressed`, `ended_in_unclosed_fence`, no-marker `None`.

---

## 8. Reference implementation (validated prototype)

The following logic was implemented in a scratch copy and produced every "Required" result in this
document. Port it into the existing files without duplicating the grammar.

### 8.1 Grammar (`tool_call_grammar.cpp`)

```cpp
constexpr std::size_t kMaxToolHeaderBytes = 1024;

// parse_value(text, pos, value, allow_slash)
//   quoted:   scan to the matching quote; CR or LF before it -> Invalid; end of input -> NeedMore
//   unquoted: stop at format whitespace, '>' and (unless allow_slash) '/'
// allow_slash = is_tool_parameter_kind(kind) for the short form; false for attribute values.

ToolHeaderStatus parse_header_bounded(std::string_view text, ToolTagKind kind, ToolOpenTag& out) {
    const bool clipped = text.size() > kMaxToolHeaderBytes;
    const auto status  = parse_header_after_keyword(clipped ? text.substr(0, kMaxToolHeaderBytes) : text,
                                                    kind, out);
    return status == ToolHeaderStatus::NeedMore && clipped ? ToolHeaderStatus::Invalid : status;
}
// parse_open_body and parse_tool_header_after_keyword call parse_header_bounded.
```

### 8.2 Pre-latch feed (`ToolCallStreamParser::feed`)

```cpp
bool ToolCallStreamParser::marker_byte(char byte, std::string& visible) {  // true on latch
    if (!marker_prefix_.empty()) {
        marker_prefix_.push_back(byte);
        ToolOpenTag marker = {};
        const ToolMarkerStatus state = classify_tool_marker_prefix(marker_prefix_, marker);
        if (state == ToolMarkerStatus::Complete) { latch(marker_prefix_); return true; }
        if (state == ToolMarkerStatus::NotMarker) {
            std::string failed = std::move(marker_prefix_);
            marker_prefix_.clear();
            publish(pending_ws_, visible);
            pending_ws_.clear();
            const std::size_t next = failed.find('<', 1);
            std::string_view head  = std::string_view(failed).substr(0, next == npos ? failed.size() : next);
            std::size_t keep = head.size();
            while (keep > 0 && is_tool_format_whitespace(head[keep - 1])) { --keep; }
            publish(head.substr(0, keep), visible);
            pending_ws_.assign(head.substr(keep));           // R3-07
            if (next != npos) {                               // R3-05 re-scan
                const std::string rest = failed.substr(next);
                for (std::size_t j = 0; j < rest.size(); ++j) {
                    if (marker_byte(rest[j], visible)) { region_.append(rest.substr(j + 1)); return true; }
                }
            }
        }
        return false;
    }
    if (byte == '<') { marker_prefix_.push_back(byte); }
    else if (is_tool_format_whitespace(byte)) { pending_ws_.push_back(byte); }
    else { publish(pending_ws_, visible); pending_ws_.clear(); publish(std::string_view(&byte, 1), visible); }
    return false;
}
// feed(): for each byte, a fence Content verdict publishes held bytes and the byte (unchanged);
// otherwise marker_byte(); on latch append the rest of the chunk to region_.
// The re-scan recursion depth is bounded by the '<' count of one candidate (<= header bound).
```

### 8.3 FenceTracker phases

```text
state: in_fence, fence_char, fence_len, fence_indent, opener_line, phase in {Indent, Run, Tail, Body},
       run_char, run_len, indent, close_ok
'\n': if in_fence && !opener_line && close_ok && phase in {Run, Tail} && run_len >= fence_len -> close.
      verdict = in_fence ? Content : Pass (evaluated after a close); reset per-line state.
outside a fence:
  Indent: ' ' -> ++indent (> 3 -> Body), Pass;  '`'/'~' -> Run (run_len = 1), Content;  else Body, Pass
  Run:    same char -> ++run_len; at 3 open (fence_len = 3, fence_indent = indent, opener_line), Content
          else Body, Pass
  Body:   Pass
inside a fence, opener line:
  Run and fence_char -> ++fence_len, Content
  else Tail; if fence_char == '`' and byte == '`' -> cancel the fence (inline code), Pass; else Content
inside a fence, other lines (always Content):
  Indent: ' ' -> ++indent (> fence_indent + 3 -> Body); fence_char -> Run, close_ok = true; else Body
  Run:    fence_char -> ++run_len; format whitespace (incl. '\r', '\t') -> Tail; else Body, close_ok = false
  Tail:   non-whitespace -> Body, close_ok = false
```

### 8.4 `finish()` stages

```text
if !latched: content = everything; diagnostics default (R3-09)
fenced[k] = FenceTracker over region_ (Content verdicts)
chain = []; base = 0
Stage 1 loop (bounded, e.g. 256 attempts):
    p = parse_region(region_[base:], policy)             // unchanged greedy parse (+ prose_after_wrapper)
    if p.termination == Complete: accept(p, base); return
    chain.push(base, p)
    owned      = p.wrapper_at_break != None && !p.prose_after_wrapper
    open_value = p.termination == EndOfInput && p.open_value_open
    if owned || (open_value && !natural_stop): break
    // p.break_offset: Definitive -> the break byte; EndOfInput -> the open value's start when a
    // value is open, otherwise the position where the input ran out (never base + 1)
    from = max(base + p.break_offset, entry_end(base))   // entry_end = base + consumed marker bytes
    next = first find_tool_marker(region_, >= from) with !fenced[next]
    if next == npos || next <= base: break
    base = next
Stage 2: for each attempt in chain: r = ConsistentCompleter(region_[attempt.base:]).run()
    pick earliest Complete with balanced() == true; else the only Complete; if >= 2 Complete and none
    balanced -> reject AmbiguousStructure (no Stage 3)
    on accept: markup_tolerant_completion = true
Stage 3 (tolerant only): for each attempt in chain:
    if Definitive && any committed call has a parameter &&
       (!natural_stop || tail(attempt.base + break_offset) contains "</parameter>" or "</param>"): skip
    decision = decide_tool_call_recovery(progress, ...); first CommitCalls wins (truncated_tail = true)
reject: failure = strict class of chain[0] (EndOfInput in tolerant -> TruncatedTail as today)
```

### 8.5 Stage 2 — `ConsistentCompleter`

```text
Result in {Complete, EndOfInput, Definitive}
solve(state):  glue transitions identical to Stage 1 (Top/ExpectFunction/FunctionHeader/FunctionBody/
               ExpectWrapperClose, name validity, declared identity, wrapper rules); input end inside a
               structure -> EndOfInput; any structural break -> Definitive
at a parameter value (name N, family F, value_begin v):
    r = try(balanced_candidates(v, F), lazy = false)     // skip closers of nested same-family openers
    if r != Definitive: return r
    r = try(all_candidates(v, F), lazy = true)
    if r != Definitive: return r
    return EndOfInput                                     // every closer contradicted: the value is open
try(candidates, lazy):
    skipped_viable = false
    for c in candidates:
        if skipped_viable && text[c - 1] != '\n': continue                       // canonical framing
        if classify_close_continuation(text, c + |close|, fn_family, wrapper) == Invalid: continue
        if next token is a parameter opener with name M in the same call and
           (M == N || M in current call || (tool has unambiguous non-empty schema && M undeclared)): continue
        push parameter (N, text[v:c]); key = hash(state after c, current call name + parameter names)
        r = memo[key] if memo says Definitive else solve(state after c)
        if r != Definitive: if lazy: ++lazy_choices; return r                    // EndOfInput stands
        memo[key] = Definitive; skipped_viable = true; pop parameter and calls to the saved size
    return Definitive
balanced() := lazy_choices == 0 on the returned path
```

Implementation requirements: explicit stack (or a hard depth bound that fails closed), step budget,
`O(1)` memo keys, reuse of Stage-1 transition helpers, no tolerant repairs.

### 8.6 Output boundary (`parse_qwen_tool_call_output`)

```text
after CR5 identity check, for each call:
    tool = find_tool_contract(contract, call.name)
    duplicate = some parameter name repeats
    if tool:
        if duplicate -> AmbiguousStructure (text)
        if tool.unambiguous && !tool.parameters.empty() &&
           some parameter at index >= 1 is not declared -> AmbiguousStructure (text)
    else if duplicate: merge last-wins, count duplicate_parameters_repaired
normalize as today
```

### 8.7 Diagnostics additions (`include/ninfer/types.h`, request log, operational log)

- `ToolCallParseFallbackReason::AmbiguousStructure` → `"ambiguous_structure"` (warning log as other text fallbacks).
- `ToolCallParseDiagnostics::markup_tolerant_completion` (bool) → info log "tool-call boundaries resolved across embedded markup".
- `ToolCallParseDiagnostics::fenced_markers_suppressed` (uint32) and `ended_in_unclosed_fence` (bool) → warning when a response has no call, ended in an unclosed fence and a marker was suppressed.
- Request log JSON: add the three fields; `test_request_log.cpp` pins them.

---

## 9. Residual risks after the fixes (document them; do not claim they are solved)

The Qwen wire format has no escaping. These cases remain byte-level ambiguous; the chosen behavior is
the documented one.

| ID                 | Input shape                                                                                                                          | Behavior after the fixes                                                   | Why                                                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| R1                 | unfenced complete example in prose, then prose that ends with canonical closer lines (`"\n</parameter>\n</function>\n</tool_call>"`) | the example is accepted as a call whose value swallows the prose (phantom) | byte-identical to S1f; accepted because S1f-class payloads are the frequent real case                               |
| P-A4               | a wrapped `write` cut by the budget exactly after an embedded example's `</tool_call>`                                               | committed with truncated content                                           | indistinguishable from a complete call; the protocol reports `length`/`max_tokens`                                  |
| P-A (natural stop) | a bare (unwrapped) call whose open value contains a complete example, ended by `StopToken`                                           | the example is committed                                                   | the model ended its turn inside an unclosed bare value; after a natural stop that value is treated as prose (R3-04) |
| S3                 | unclosed code fence before a real call                                                                                               | call returned as text (now with a warning diagnostic)                      | "suppression is the safe direction" (R2-I6)                                                                         |
| P-N                | `"I will now emit <tool_call>\n"` + real call                                                                                        | text (nesting break)                                                       | F5 wrapper balance                                                                                                  |
| P-M                | call with an invalid name, then a valid call                                                                                         | text                                                                       | CR1 scope ownership after a function attempt                                                                        |
| E3                 | hallucinated non-first parameter name                                                                                                | text (`ambiguous_structure`)                                               | R3-08 product decision                                                                                              |
| TOL                | tolerant, natural stop, unfenced prose example + prose without value closers                                                         | example committed                                                          | tolerant trailing-prose recovery is inherently lenient                                                              |
| NAMES              | parameter names containing whitespace (if the optional R3-10 relaxation is not implemented)                                          | text                                                                       | grammar limitation                                                                                                  |

---

## 10. Implementation order, verification, acceptance

### 10.1 Order (one commit each; Conventional Commits; tests green before the next step)

 1. `test:` add all reproducers of §7 as failing tests (mark expected failures in the progress file; until the step that fixes them, these recorded failures are the only ones allowed).
 2. `fix(frontend):` R3-05 grammar + re-scan + R3-07 whitespace hold.
 3. `fix(frontend):` R3-06 fence tracker rewrite, fence-aware retry, diagnostics fields.
 4. `fix(frontend):` R3-02 `prose_after_wrapper`; R3-04 retry rules and chain bound.
 5. `feat(frontend):` R3-01 Stage 2 with budgets and diagnostics; introduces `AmbiguousStructure` (Stage 2 needs it for the multi-base ambiguity).
 6. `fix(frontend):` R3-03 Stage 3 ordering and rules; tolerant `TrailingContent` class.
 7. `fix(frontend):` R3-08 synthetic-argument rule (uses `AmbiguousStructure`).
 8. `fix(frontend):` R3-09 no-marker diagnostics; R3-10 `/` in parameter names.
 9. `fix(engine):` R3-13 cancellation finish reason; dead-code removal.
10. `docs:` R3-11 design-doc corrections and R3-12 documentation/help updates.

### 10.2 Commands (CPU only)

```powershell
cmake --build build-new-parser --config Release --parallel 16 `
  --target ninfer_tool_call_parser_test ninfer_tool_call_grammar_test ninfer_tool_call_grammar_state_test `
           ninfer_qwen3_5_frontend_test ninfer_request_log_test ninfer_pretty_logging_test `
           ninfer_serve_options_test ninfer_engine_options_validation_test
ctest --test-dir build-new-parser -C Release -j 16 --output-on-failure `
  -R "^(ninfer_tool_call_parser_test|ninfer_tool_call_grammar_test|ninfer_tool_call_grammar_state_test|ninfer_qwen3_5_frontend_test|ninfer_request_log_test|ninfer_pretty_logging_test|ninfer_serve_options_test|ninfer_engine_options_validation_test)$"
```

Then run the full CPU suite excluding `_real` tests as in Round 2, and record results.

### 10.3 Fuzz fragment corpus (for the R3-07 test)

````cpp
const std::vector<std::string> frags = {
    "<tool_call>\n", "</tool_call>\n", "<function=write>\n", "<function=bash>\n", "</function>\n",
    "<parameter=path>\n", "<parameter=content>\n", "<parameter=command>\n", "</parameter>\n",
    "text ", "x\n", "```\n", "```xml\n", "~~~\n", "  ```\n", "<", ">", "\"", "'", "\r\n",
    "<function name=\"", "<invoke=bash>", "</invoke>", "<param=command>", "</param>",
    "<function_calls>\n", "</function_calls>\n", "echo '</parameter>'\n", "<tool_c", "<function",
    "prose. ", "\n"};
// std::mt19937 rng(20260930); 2..15 fragments per text; append
// "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
// with probability 1/3; declared tools write{path,content}, bash{command}.
````

### 10.4 Acceptance checklist

```text
[ ] every reproducer in §4 behaves as "Required", one-shot and streamed, all listed finish reasons
[ ] §6 changes applied, each with a comment naming the Round-3 finding
[ ] §7 tests present and green; fuzz: 0 mismatches
[ ] no other existing assertion changed (compare against the HEAD suite)
[ ] Stage 2 reuses Stage-1 transitions; no second grammar; no unbounded recursion; budgets fail closed
[ ] work-bound tests green (deterministic counters)
[ ] ambiguous_structure and the new diagnostics are in types.h, request log, operational log, docs
[ ] engine passes FinishReason::Cancelled on the active-cancellation path
[ ] docs/help/progress corrections of R3-11/R3-12 applied
[ ] no GPU test executed; build-windows unused; <= 16 threads
[ ] progress file docs/NInfer_new_parser_design_round3_bugfix_progress.md complete
```

---

## Appendix A — Reproducer corpus (single-line literals)

`tool_call(name, {{k, v}, …})` serializes `"<tool_call>\n<function=" + name + ">\n"`, then for each
parameter `"<parameter=" + k + ">\n" + v + "\n</parameter>\n"`, then `"</function>\n</tool_call>"`.
`EX = tool_call("read", {{"path","foo.cpp"}})`, `E = tool_call("bash", {{"command","rm -rf x"}})`,
`real = tool_call("read", {{"path","a"}})`. Declared tools for the probes: `write{path,content}`,
`edit{path,old_string,new_string}`, `bash{command,timeout:int}`, `read{path}`.

| ID             | Text                                                                                                                                                                                       | Required (strict)                                                         |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| S1             | `tool_call("write", {{"path","docs/x.md"},{"content","# Example\n" + EX + "\nDone."}})`                                                                                                    | 1 call, exact content                                                     |
| S1c            | `tool_call("write", {{"path","docs/x.md"},{"content","Example:\n" + EX}})`                                                                                                                 | 1 call                                                                    |
| S1d            | ```` tool_call("write", {{"path","docs/x.md"},{"content","Text\n```xml\n" + EX + "\n```\nMore text"}}) ````                                                                                | 1 call                                                                    |
| S1e            | `tool_call("edit", {{"path","t.cpp"},{"old_string","R\"(" + EX + ")\""},{"new_string","x"}})`                                                                                              | 1 call                                                                    |
| S1f            | `tool_call("write", {{"path","n.md"},{"content","Close with:\n</parameter>\n</function>\n</tool_call>\nthen stop."}})`                                                                     | 1 call                                                                    |
| S1g            | `tool_call("write", {{"path","docs/x.md"},{"content", tool_call("edit", {{"path","a"},{"old_string","b"}})}})`                                                                             | 1 call                                                                    |
| S8             | `tool_call("bash", {{"command","cat <<EOF\n</parameter>\n</function>\n</tool_call>\nEOF"}})`                                                                                               | 1 call                                                                    |
| TAIL           | §2 `edit`, LF and CRLF framing                                                                                                                                                             | 1 call, exact old/new                                                     |
| P-C            | `tool_call("write", {{"path","d.md"},{"content","Ex:\n" + E + "\n" + E + "\nDone."}})`                                                                                                     | 1 call (tolerant too)                                                     |
| P-H            | `tool_call("write", {{"path","d.md"},{"content", tool_call("write", {{"path","x"},{"content","hello"}})}})`                                                                                | 1 call                                                                    |
| R2             | `"Example:\n" + tool_call("bash", {{"command","ls"}}) + "\nNow writing.\n" + tool_call("write", {{"path","d.md"},{"content","Doc\n" + tool_call("bash", {{"command","ls"}}) + "\nmore"}})` | 1 call `write`                                                            |
| R4             | `tool_call("write", {{"path","d.md"},{"content","A:\n" + X + "\nB:\n" + X + "\nEnd."}})`, `X = tool_call("bash", {{"command","ls"}})`                                                      | 1 call                                                                    |
| R6             | `tool_call("bash", {{"command","echo real"}}) + "\nNote: calls end with </parameter>\n</function>\n</tool_call>"`                                                                          | text (`trailing_content`)                                                 |
| S2             | `"I will emit a <tool_call> block now.\n" + real`                                                                                                                                          | 1 call                                                                    |
| S2b            | see R3-02                                                                                                                                                                                  | 1 call                                                                    |
| Q1             | `"Attr form: <function name=\"x\n" + tool_call("bash", {{"command","ls"}})`                                                                                                                | 1 call                                                                    |
| Q2             | `"The <invoke name='it\n" + tool_call("bash", {{"command","ls"}})`                                                                                                                         | 1 call                                                                    |
| P-B            | `tool_call("bash", {{"command","grep '</parameter><parameter name=\"' file"}})`                                                                                                            | 1 call, exact                                                             |
| F1             | ```` "```xml\r\n" + E + "\r\n```\r\n" + real ````                                                                                                                                          | 1 call `read`                                                             |
| F2             | ```` "```x``` is inline code\n" + real ````                                                                                                                                                | 1 call                                                                    |
| F3             | ```` "- step:\n  ```bash\n  make\n    ```\nNow calling.\n" + real ````                                                                                                                     | 1 call                                                                    |
| S3             | ```` "Here is code:\n```python\nprint(1)\n\n" + real ````                                                                                                                                  | text + unclosed-fence diagnostics                                         |
| S5b            | `"<tool_call>\n<function=search>\n<parameter=a/b>\nx\n</parameter>\n</function>\n</tool_call>"` (`search{a/b}`)                                                                            | 1 call                                                                    |
| S4             | `"Just an answer."`                                                                                                                                                                        | text, reason `none`                                                       |
| P-A            | `"<function=write>\n<parameter=path>\nd.md\n</parameter>\n<parameter=content>\nExample:\n" + E`                                                                                            | `OutputLimit`: text; `StopToken`: example call (natural stop, documented) |
| P-A3           | `"<function=write>\n<parameter=content>\nEx:\n" + E + "\nDone.\n</parameter>\n"`                                                                                                           | text (strict and tolerant)                                                |
| P-D (tolerant) | `"For example:\n" + tool_call("read", {{"path","ex.txt"}}) + "\nNow the real one.\n" + tool_call("bash", {{"command","echo real"}})`                                                       | `bash(echo real)`                                                         |
| E1             | see R3-08                                                                                                                                                                                  | text, `ambiguous_structure`                                               |
| E2             | see R3-08                                                                                                                                                                                  | text, `ambiguous_structure`                                               |
| E3             | E1 with `mode` instead of `content`                                                                                                                                                        | text, `ambiguous_structure`                                               |
| FUZZ           | §10.3                                                                                                                                                                                      | 0 streaming/one-shot mismatches                                           |
