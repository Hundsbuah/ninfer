# NInfer `new_parser_design` — Round 4 Bugfix Implementation Specification

## 0. Purpose, scope, baseline

This is the binding implementation specification for the findings of the **fourth review**: the
review of the Round-3 implementation. It is a delta on top of Round 3.

| Item                            | Value                                                                                                                                                                           |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Reviewed HEAD                   | `4cc2ba9c` (`fix(frontend): check both parameter-closer forms in the Stage-3 tail (R3-03 item 3)`) on top of `a34aceaa` (`fix: tool-call parser round-3 bugfixes R3-01..R3-14`) |
| Reference behavior              | `docs/NInfer_new_parser_design_round3_bugfix_implementation.md` and its validated prototype                                                                                     |
| Review date                     | 2026-10-01                                                                                                                                                                      |
| Production configuration (user) | `ninfer serve`, strict mode, Qwen3.6/3.8 wrapped `<tool_call>` + `<parameter=...>` format, agent client oh-my-pi with native tool calls                                         |
| Deliverable of this document    | Findings N-01 … N-11, target design, required fixes, test changes, refactor guard, performance probe, acceptance gate                                                           |

Round 1 (F1–F10), Round 2 (CR1–CR6) and Round 3 (R3-01 … R3-14) stay in force. This round does
not change an accepted Round-3 decision. It repairs the implementation where it deviates from the
Round-3 design or misses its work bounds, and it corrects tests and documentation.

Where this document and the Round-3 specification disagree, this document wins. This applies to
two Round-3 statements: the example chain bound "for example 256 attempts" (R3-04 item 4) and the
Stage-2 step budget, which Round 3 did not define as global (R3-14).

Record all Round-4 work in a new progress file:

```text
docs/NInfer_new_parser_design_round4_bugfix_progress.md
```

### 0.1 Hard constraints (unchanged)

- CPU only: no GPU runtime tests, no `_real` tests, no model inference, no GPU benchmarks.
- Build directory `build-new-parser`; never `build-windows`; at most 16 build and 16 test threads.
- Before running an unknown test, check its source/CMake for GPU use; if unsure, do not run it.
- Every tool-call fixture is a single-line C++ string literal with `\n` escapes or is built with the `tool_call(...)` test helper. This document follows that convention.

### 0.2 How the evidence was produced

All claims marked **Verified** were executed on CPU (MSVC 19.44, `/std:c++20 /O2`; memory checks
with `/fsanitize=address`). They were run against `4cc2ba9c`, the pre-Round-3 HEAD `4165d227`, and
the validated Round-3 prototype:

1. The three parser CPU suites pass on `4cc2ba9c`. The parser suite and the review probes also pass under AddressSanitizer.
2. The Round-3 review probes 1–12 give the prototype's results. The only differences are the two intended Round-3 refinements (`ambiguous_structure`, tolerant `trailing_content`).
3. Deterministic fuzz over the Round-3 §10.3 corpus (seed `20260930`, 6 000 texts × strict/tolerant, byte-wise and two random partitions, exact comparison including diagnostics) gave 0 mismatches. Every per-case decision is identical to the prototype's.
4. Realistic payloads were all byte-exact, streaming-equal and parsed in ≤ 2.6 ms:
   - 8 repository files as `write` content, in LF and in CRLF framing: the parser documents with real multi-line tool-call examples, `tests/test_tool_call_parser.cpp` and `tool_call_stream.cpp`.
   - 400 random `edit`s of the Round-2 document.
5. Targeted probes for this round. Their numbers are in each finding and in Appendix C.

### 0.3 Behavior that must not regress

Everything in 0.2 items 1–4 must still hold. In particular:

- All Round-3 Appendix-A reproducers keep their "Required" outcome, one-shot and streamed.
- Realistic `write`/`edit` payloads with tool-call markup stay byte-exact.
- Streaming equals one-shot for content, calls, arguments and the full diagnostics.
- `StopString` is a cut, and the engine passes `FinishReason::Cancelled` on active cancellation.
- R3-08 (`ambiguous_structure` for synthetic arguments) applies to every accepted region.

A changed outcome that is not explained by a finding of this document is a defect of the
implementation. Section 5 defines a refactor guard that detects such changes.

---

## 1. Executive summary and finding index

The Round-3 implementation fixes the original failure class for the production format. The
remaining defects fall into four groups:

- **Work bounds:** the Stage-2 consistent completion is not bounded in time.
- **Cliffs:** two hard limits (256 chain attempts, 64 recursion levels) introduce functional cliffs that neither HEAD nor the prototype had.
- **Family handling:** two places ignore the parameter family.
- **Diagnostics, streaming, tests and docs:** fence diagnostics are wrong (a closed code block before a call is reported as unclosed, and payload bytes are counted), one corner of streaming output differs, and tests and documentation have drifted.

| ID   | Severity   | Title                                                                                                                                                                                                                                         | Production impact (strict Qwen3.6/3.8)                           |
|------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| N-01 | MEDIUM     | Stage-2 work is quadratic and budgeted per base: degenerate outputs block the engine thread for seconds                                                                                                                                       | rare (degenerate output); then a one-time stall of every request |
| N-02 | MEDIUM     | The Stage-2 skip checks only `</parameter>`: R3-01 is not fixed for the `<param>` family                                                                                                                                                      | none (Qwen does not emit `<param>`)                              |
| N-03 | LOW–MEDIUM | The 256-attempt chain bound loses the real call after ≥ 256 failed marker bases                                                                                                                                                               | degenerate output only                                           |
| N-04 | LOW–MEDIUM | The Stage-2 recursion bound of 64: a region with ≥ 64 parameter values that needs Stage 2 becomes text, with a misleading reason                                                                                                              | rare                                                             |
| N-05 | LOW–MEDIUM | The balance census counts openers of any parameter family (spec: same family), producing a spurious `ambiguous_structure`                                                                                                                     | rare (documents that mention `<param=…>`)                        |
| N-06 | LOW        | Fence diagnostics are wrong in three ways: a closed code block before a call is reported as unclosed (every such response), bytes inside parameter values are counted, and the fence warning hides the fallback reason in the operational log | log quality                                                      |
| N-07 | LOW        | No diagnostic when a work bound fires                                                                                                                                                                                                         | log quality                                                      |
| N-08 | LOW        | Streaming ≠ one-shot for whitespace on a fence close line directly before a call                                                                                                                                                              | cosmetic                                                         |
| N-09 | LOW        | Tests: a suite test is unregistered, the §9 residual R1 is unpinned (wrong comment), the fuzz test is weaker than specified, new log fields are untested                                                                                      | maintainability                                                  |
| N-10 | LOW        | The help text, `serving.md`, `tool_call_parser.md` and the Round-3 progress file contain wrong statements                                                                                                                                     | documentation                                                    |
| N-11 | LOW        | `noexcept` functions allocate: `bad_alloc` becomes `std::terminate`                                                                                                                                                                           | extremely rare                                                   |

---

## 2. Target design for N-01, N-02, N-03, N-04, N-07

These five findings share one redesign of the work model in `ToolCallStreamParser::finish()`.
Implement it as described here; the findings in §3 refer to its subsections. Line numbers refer to
`src/models/qwen3_5/frontend/tool_call_stream.cpp` at `4cc2ba9c`.

### 2.1 One coordinate system: absolute positions

- The region machine runs on the whole `region_` with absolute positions. A Stage-1 attempt or a Stage-2 search for base `b` starts from `RegionState s; s.pos = b;` (mode `ExpectFunction`). It no longer slices `region_.substr(b)`.
- `parse_region_at` takes the state **by reference**: `ToolCallParseProgress parse_region_at(std::string_view text, const ToolCallParsePolicy& policy, RegionState& s, BoundaryMode mode, Stage2Context* cx, const RegionIndex* index)`. `parse_region(text, policy)` creates a local `RegionState` and passes `nullptr, nullptr`.
- `ToolCallParseProgress::break_offset` therefore becomes absolute inside `finish()`:
  - `:1106` `std::max(base + last.progress.break_offset, entry_end)` becomes `std::max(last.progress.break_offset, entry_end)`.
  - `:1186` `attempt.base + attempt.progress.break_offset` becomes `attempt.progress.break_offset`.
  - Update the field comment to "offset in the text passed to the parse".
- `parse_tool_call_region(text, policy)` (used by the grammar-state core and two tests) starts at position 0, so its behavior is unchanged.
- Never copy the region. Replace `classify_tool_marker_prefix(region_.substr(base), …)` (`:1101`, one `std::string` copy per attempt) by `std::string_view(region_).substr(base)`.

### 2.2 Region index

Build once per `finish()` call, after the latch check:

```cpp
// Absolute positions of every value closer in region_, ascending. "</param>" is not a substring
// of "</parameter>", so the two lists are disjoint.
struct RegionIndex {
    std::vector<std::size_t> parameter_closes; // "</parameter>"
    std::vector<std::size_t> param_closes;     // "</param>"
    // First closer of `family` at or after `from`, or npos (std::lower_bound).
    [[nodiscard]] std::size_t next_close(ToolTagKind family, std::size_t from) const noexcept;
    // True when a closer of either family starts at or after `from`.
    [[nodiscard]] bool has_close_at_or_after(std::size_t from) const noexcept;
};
```

- Build both lists with a `find` loop that advances by the literal size.
- In the Greedy `ParameterValue` scan (`:709`), replace `text.find(required_close, scan)` by `index != nullptr ? index->next_close(s.param_family, scan) : text.find(required_close, scan)`. The candidates and their order are the same, so behavior does not change.
- Effect: a Stage-1 attempt costs O(structural tokens · log closers) instead of O(bytes). An open value is recognized without scanning to the end of the region.

### 2.3 Stage-1 chain without an attempt bound (N-03)

- Remove `kMaxChainAttempts` (`:1068`) and its loop counter. The loop still ends when no further admissible marker exists (`next == npos || next <= base`).
- The loop terminates and stays linear because bases strictly increase:
  - A Definitive attempt resumes after its break offset, so its bytes are not parsed again.
  - An EndOfInput attempt with an open value (natural stop) resumes at the next marker after the value start. With the index it costs only its header tokens.
  - An EndOfInput attempt without an open value ends the chain, because its break is the region end.
- Total Stage-1 work is O(region bytes + markers · log closers).
- Do not add another attempt bound. With §2.1 and §2.2 the chain cannot become quadratic.

### 2.4 Stage 2: sequential value chooser (N-01, N-04)

**Key fact.** This holds for the current implementation and for the prototype. Once the parse
reaches the start of the next parameter value, the boundary chosen for the previous value is final:

- `select_value` never returns Definitive. A value whose candidates are all contradicted yields EndOfInput ("value open to the end"), and EndOfInput stands.
- So only the glue between a value close and the next value start (or the region end) can contradict a candidate.

The recursion `select_value → parse_region_at → select_value` is therefore a linear chain of
decisions. Replace it by a loop:

1. **Stop at values.** In Consistent mode, the region machine stops at every parameter value start
   and returns to the driver. At `case RegionState::Mode::ParameterValue` (`:694`), replace
   `out = cc->select_value(s); …` by:

   ```cpp
   if (mode == BoundaryMode::Consistent) {
       out.stage2_at_value = true; // `s` keeps the state at the value start; nothing is moved into `out`
       return out;
   }
   ```
2. **New field.** Add `bool stage2_at_value = false;` to `ToolCallParseProgress`, with a one-line
   comment saying it is Stage-2 internal. Remove `stage2_lazy_boundary_used`; the search object
   tracks it now.
3. **Remove the recursion.** Delete `ConsistentCompleter`, `select_value` and `kMaxStage2Depth`
   (`:200–363`). There is no recursion and no depth constant any more.
4. **Implement the driver** (`run_stage2_base`) as follows. The candidate checks are exactly
   today's checks, in today's order:

```cpp
enum class CandidatePass : std::uint8_t { Balanced, Lazy };
enum class Stage2Verdict : std::uint8_t { Complete, EndOfInput, Definitive, Exhausted };

struct Stage2Path {
    std::vector<std::pair<std::size_t, std::size_t>> values; // chosen [begin, end) per parameter, in order
    bool lazy_used = false;                                   // a pass-2 boundary stands on the path
};

Stage2Verdict run_stage2_base(Stage2Context& cx, std::size_t base, Stage2Path& path,
                              ToolCallParseProgress& out) {
    RegionState s;
    s.pos = base;
    out = parse_region_at(cx.text, cx.policy, s, BoundaryMode::Consistent, &cx, &cx.index);
    for (;;) {
        if (cx.budget.exhausted) { return Stage2Verdict::Exhausted; }
        if (!out.stage2_at_value) {
            switch (out.termination) {
            case ToolCallRegionTermination::Complete:   return Stage2Verdict::Complete;
            case ToolCallRegionTermination::EndOfInput: return Stage2Verdict::EndOfInput;
            case ToolCallRegionTermination::Definitive: return Stage2Verdict::Definitive;
            }
        }
        // `s` is the machine state at the start of a parameter value.
        bool stood = false;
        for (const CandidatePass pass : {CandidatePass::Balanced, CandidatePass::Lazy}) {
            bool skipped_viable = false; // per value and per pass, as today
            CandidateWalk walk(cx.text, s.value_begin, s.param_family, pass);
            std::size_t c = 0;
            while (walk.next(c, cx.budget)) {
                if (cx.budget.charge()) { return Stage2Verdict::Exhausted; } // one unit per candidate
                if (skipped_viable && cx.text[c - 1] != '\n') { continue; }  // canonical framing
                const std::size_t after = c + tool_close_literal(s.param_family).size();
                if (classify_close_continuation(cx.text, after, s.fn_family, s.wrapper) ==
                    CloseContinuation::Invalid) {
                    continue;
                }
                if (!next_parameter_plausible(cx, s, after)) { continue; } // today's rule, unchanged
                RegionState trial = s; // values are not materialized: the copy is small
                trial.current.parameters.push_back(ParsedParameter{.name = s.param_name, .value = {}});
                trial.pos  = after;
                trial.mode = RegionState::Mode::FunctionBody;
                const std::uint64_t key = stage2_state_hash(trial); // today's state_hash
                if (cx.dead.contains(key)) { skipped_viable = true; continue; }
                ToolCallParseProgress r =
                    parse_region_at(cx.text, cx.policy, trial, BoundaryMode::Consistent, &cx, &cx.index);
                if (cx.budget.exhausted) { return Stage2Verdict::Exhausted; }
                if (!r.stage2_at_value && r.termination == ToolCallRegionTermination::Definitive) {
                    cx.dead.insert(key); // the glue up to the next value or the end is contradicted
                    skipped_viable = true;
                    continue;
                }
                path.values.emplace_back(s.value_begin, c); // the candidate stands
                if (pass == CandidatePass::Lazy) { path.lazy_used = true; }
                s     = std::move(trial);
                out   = std::move(r);
                stood = true;
                break;
            }
            if (cx.budget.exhausted) { return Stage2Verdict::Exhausted; }
            if (stood) { break; }
        }
        if (!stood) { return Stage2Verdict::EndOfInput; } // every candidate contradicted: the value is open
    }
}
```

`next_parameter_plausible` is today's plausibility block, moved into a function:

- skip whitespace after the closer;
- if a complete parameter opener follows, its name must not equal `s.param_name` or any name in `s.current.parameters`;
- `policy.parameter_plausible` must not reject it.

**Materialization.** After a base returns `Complete`, `out.calls` holds every parameter of the
accepted path in order, each with an empty value. In Consistent mode only the chooser creates
parameters, so there is exactly one path entry per parameter. Fill the values:

```cpp
std::size_t k = 0;
for (ParsedFunctionCall& call : out.calls) {
    for (ParsedParameter& p : call.parameters) {
        const auto [b, e] = path.values[k++];
        p.value.assign(region_.data() + b, e - b);
    }
}
// k == path.values.size() must hold; if it does not, the implementation is wrong (assert in Debug).
```

### 2.5 Global work budget, shared memo, early exit (N-01, N-07)

```cpp
constexpr std::uint64_t kStage2WorkMinimum = 100'000;
constexpr std::uint64_t kStage2WorkPerByte = 4;

struct WorkBudget {
    std::uint64_t limit = 0;
    std::uint64_t used  = 0;
    bool exhausted      = false;
    // Charge one unit; true once the budget is exhausted.
    bool charge() noexcept {
        if (!exhausted && ++used > limit) { exhausted = true; }
        return exhausted;
    }
};

struct Stage2Context {
    std::string_view text;                  // the whole region_ (absolute positions)
    const ToolCallParsePolicy& policy;
    const RegionIndex& index;
    WorkBudget budget;                      // ONE budget for all bases of this finish() call
    std::unordered_set<std::uint64_t> dead; // post-candidate states proven Definitive, shared by all bases
};
```

- **Budget.** `budget.limit = policy_.stage2_step_budget != 0 ? policy_.stage2_step_budget : std::max(kStage2WorkMinimum, kStage2WorkPerByte * region_.size())`. It is charged:
  - 1 unit per glue transition (today's loop-top charge at `:415`, now on `cx->budget`);
  - 1 unit per candidate;
  - 1 unit per `<` visited by a candidate walk (§2.6).
- **Shared memo.** One `dead` set serves all bases. This is sound because positions are absolute and the key contains every field that decides the glue outcome (today's `state_hash` fields). Only Definitive glue results are stored. Complete, EndOfInput and at-value results are never stored.
- **Base loop with early exit.** The selection outcome is identical to today's rule (earliest balanced; else exactly one; else ambiguous):

```cpp
Stage2Context cx{region_, policy_, index, WorkBudget{.limit = limit}, {}};
std::size_t unbalanced_count = 0;
std::optional<std::pair<std::size_t, std::vector<ParsedFunctionCall>>> first_unbalanced;
bool stage2_exhausted = false;
for (const Attempt& attempt : chain) {
    if (!index.has_close_at_or_after(attempt.base)) { continue; } // N-02: either family
    Stage2Path path;
    ToolCallParseProgress out;
    const Stage2Verdict verdict = run_stage2_base(cx, attempt.base, path, out);
    if (verdict == Stage2Verdict::Exhausted) { stage2_exhausted = true; break; }
    if (verdict != Stage2Verdict::Complete) { continue; }
    materialize(out.calls, path);
    if (!path.lazy_used) {
        // Earliest balanced completion: accept it now and return (calls, tail = region_.substr(0, attempt.base),
        // markup_tolerant_completion = true). Later bases are not searched.
        return accept_stage2(result, std::move(out.calls), attempt.base);
    }
    if (++unbalanced_count == 1) { first_unbalanced.emplace(attempt.base, std::move(out.calls)); }
}
result.stage2_steps = cx.budget.used;
if (stage2_exhausted) {
    result.parse_budget_exhausted = true; // fail closed: nothing from Stage 2 is accepted
} else if (unbalanced_count >= 2) {
    /* AmbiguousStructure: region as text (Stage 3 does not run) */
} else if (unbalanced_count == 1) {
    /* accept first_unbalanced, markup_tolerant_completion = true */
}
// Stage 3 (tolerant) as today.
```

- **After exhaustion.** Nothing from Stage 2 is accepted, even an unbalanced completion found before the budget ran out: with partial information the selection rule cannot be applied. Stage 3 still runs over the Stage-1 chain. Strict mode returns the region as text with the Stage-1 failure class and `parse_budget_exhausted = true`.
- **Policy field.** `ToolCallParsePolicy::stage2_step_budget` keeps its name. Its meaning is now "global Stage-2 work budget of one `finish()` call". Update its comment and the default formula. `ToolCallStreamResult::stage2_steps` reports `budget.used`, the total of all charged units.

### 2.6 Candidate walk

The walk replaces the eager census (`:255–273`), which scanned to the region end once per value.
It yields candidates lazily, so a value whose first candidate stands costs only the distance to that
candidate. Pass 1 (Balanced) and pass 2 (Lazy) use the same tokenization as today's census:

```cpp
class CandidateWalk {
public:
    CandidateWalk(std::string_view text, std::size_t value_begin, ToolTagKind family, CandidatePass pass);
    // Next candidate closer (absolute), ascending. False at the end or when the budget is exhausted.
    bool next(std::size_t& candidate, WorkBudget& budget) {
        const std::string_view close = tool_close_literal(family_);
        for (;;) {
            pos_ = text_.find('<', pos_);
            if (pos_ == std::string_view::npos) { return false; }
            if (budget.charge()) { return false; }
            if (starts_with_at(text_, pos_, close)) {
                const std::size_t at = pos_;
                pos_ += close.size();
                if (pass_ == CandidatePass::Lazy || depth_ == 0) { candidate = at; return true; }
                --depth_; // consumed by a nested same-family opener
                continue;
            }
            ToolOpenTag opener = {};
            // N-05 changes this call to parse_tool_open_header(text_.substr(pos_), family_, opener).
            if (parse_tool_parameter_open(text_.substr(pos_), opener) == ToolHeaderStatus::Complete) {
                ++depth_;
                pos_ += opener.consumed;
                continue;
            }
            ++pos_;
        }
    }
private:
    std::string_view text_;
    std::size_t pos_;
    std::size_t depth_ = 0;
    ToolTagKind family_;
    CandidatePass pass_;
};
```

- In pass 1, a closer at depth 0 is a candidate and leaves the depth at 0. Pass 2 yields every closer and also skips the bytes of complete openers. This is today's `all` list.
- Pass 2 restarts from `value_begin`.

---

## 3. Findings

### N-01 — MEDIUM — Stage-2 work is not bounded in time

**Status:** Verified (Appendix C P1, P2, P3, P4; scaling probes).

**Evidence (**`4cc2ba9c`**, Release, best of 3):**

| Case                                                                                   | Bytes  | 4cc2ba9c                                           | HEAD 4165d227                            |
|----------------------------------------------------------------------------------------|--------|----------------------------------------------------|------------------------------------------|
| P1 degenerate repetition: 1 500 × (`"Example i:\n"` + unfenced example), `OutputLimit` | 159 KB | 1 764 ms, `ambiguous_structure`                    | ≈ 1 ms at 63 KB (last example committed) |
| P2 same + trailing `"That is all."`                                                    | 159 KB | 2 223 ms (3.1 s single run)                        | —                                        |
| P3 300 examples + a real `read` call                                                   | 32 KB  | 91.5 ms, **real call lost**                        | 0.7 ms, real call found                  |
| P4 8 000 closer triples in one `write` value                                           | 344 KB | 108–194 ms (Stage-2 steps grow 4n, time quadratic) | rejected                                 |

**Location:** `tool_call_stream.cpp`:

- `:200–363`: `ConsistentCompleter`, `select_value`.
- `:313`: the value is copied for every candidate.
- `:314` and `:366`: the full `RegionState`, including materialized values, is copied for every candidate and every recursion level.
- `:255–273`: the census scans to the region end once per value.
- `:1123–1139`: each chain base gets a fresh budget of `max(200000, 32 × bytes)`.
- `:1068`: up to 256 bases.

**Root cause:** together these make Stage-2 work O(bases × candidates × value bytes). The step
counter grows linearly, but one step costs O(n), so the counter does not bound time.

**Required fix:** implement §2.2, §2.4, §2.5 and §2.6:

- values become ranges and are materialized only for the accepted base;
- the recursion becomes the sequential chooser;
- the census becomes lazy walks;
- the budget becomes global and is charged per walk step, candidate and glue transition;
- the memo is shared across bases;
- the base loop exits at the first balanced completion.

**Pitfalls:**

- **No decision may change.** The outcome dump (§5, Appendix B) must be byte-identical before and after this step.
- `skipped_viable` resets per pass. The per-candidate charge happens before the framing check, as today.
- Keep views only into `region_`. Never bind a `std::string_view` to a temporary `std::string`; the Round-3 use-after-free came from exactly that.
- At an at-value stop, `parse_region_at` must return without moving `s.current` or `s.region.calls` into `out`. Only the terminal helpers (`invalid`, `truncated`, `complete`) move.
- Tolerant repairs stay disabled in Consistent mode (`effective.tolerant = false`, unchanged).

**Tests:** §7 items 2, 3 and 7 (many values, chain, work bounds with deterministic counters and
budget exhaustion).

**Acceptance:**

- The outcome dump is identical.
- The Appendix C targets are met.
- The three suites pass, also under AddressSanitizer (§8).

---

### N-02 — MEDIUM — Stage-2 skip ignores the `<param>` family

**Status:** Verified. The prototype accepts the reproducer; `4cc2ba9c` rejects it.

**Location:** `tool_call_stream.cpp:1126`:
`region_.find(tool_close_literal(ToolTagKind::Parameter), attempt.base) == npos` skips the base. A
region whose values use `<param …>`/`</param>` never reaches Stage 2. This is the same defect class
that `4cc2ba9c` fixed in Stage 3.

**Reproducer (strict; tools** `write{path,content}`**,** `read{path}`**):**`"<tool_call>\n<function=write>\n<param=path>\ndocs/x.md\n</param>\n<param=content>\n# Example\n<tool_call>\n<function=read>\n<param=path>\nfoo.cpp\n</param>\n</function>\n</tool_call>\nDone.\n</param>\n</function>\n</tool_call>"`

- `4cc2ba9c`: text, `trailing_content`.
- Required: 1 call `write` with arguments `{"path":"docs/x.md","content":"# Example\n<tool_call>\n<function=read>\n<param=path>\nfoo.cpp\n</param>\n</function>\n</tool_call>\nDone."}` and `markup_tolerant_completion == true`.

**Required fix:** skip a base only when `!index.has_close_at_or_after(attempt.base)`, which covers
both families (§2.2, §2.5).

**Tests:** §7 item 1.

---

### N-03 — LOW–MEDIUM — The 256-attempt chain bound loses the real call

**Status:** Verified (Appendix C P3: `4cc2ba9c` returns text after 91.5 ms; HEAD and the prototype
find the call in about 1 ms).

**Location:** `tool_call_stream.cpp:1068`. After 256 failed bases the chain stops. The real call
behind them is never tried, and Stage 2 then runs over all 256 bases (N-01). No diagnostic is set.

**Reproducer (strict):**

```cpp
std::string t;
for (int i = 0; i < 300; ++i) { t += "Example " + std::to_string(i) + ":\n" + tool_call("bash", {{"command", "ls -la"}}) + "\n"; }
t += "Now the real call:\n" + tool_call("read", {{"path", "real.txt"}});
```

- `4cc2ba9c`: text, `ambiguous_structure`.
- Required: 1 call `read`, `{"path":"real.txt"}`. The prose before it is content.

**Required fix:** §2.3 (remove the bound). It is safe only together with §2.1 (no `substr` copies)
and §2.2 (index); without them, Appendix C P5 becomes quadratic again.

**Documented consequence (unchanged from HEAD):** a degenerate output made only of prose and
unfenced complete examples commits its **last** example in strict mode. Byte for byte, it cannot be
distinguished from prose followed by a real call. Add this to the residual list in
`docs/tool_call_parser.md` (N-10).

**Tests:** §7 item 3.

---

### N-04 — LOW–MEDIUM — Stage-2 recursion bound of 64

**Status:** Verified. `4cc2ba9c` returns text for ≥ 64 values; the prototype accepts.

**Location:** `tool_call_stream.cpp:201` (`kMaxStage2Depth = 64`) and `:322`. Each value on the
path adds a recursion level. When the bound fires, the region becomes text with the misleading
reason `trailing_content` (the Stage-1 class).

**Reproducers (strict):**

1. A tool `multi` declares `p0 … p69`:

   ```cpp
   std::string t = "<tool_call>\n<function=multi>\n";
   for (int i = 0; i < 69; ++i) { t += "<parameter=p" + std::to_string(i) + ">\nv" + std::to_string(i) + "\n</parameter>\n"; }
   t += "<parameter=p69>\n# Example\n" + tool_call("read", {{"path", "foo.cpp"}}) + "\nDone.\n</parameter>\n</function>\n</tool_call>";
   ```

   Required: 1 call, `p69` byte-exact.
2. 69 × `tool_call("read", {{"path", "f<i>"}}) + "\n"`, then
   `tool_call("write", {{"path","d.md"},{"content","# Example\n" + tool_call("read", {{"path","foo.cpp"}}) + "\nDone."}})`.
   Required: 70 calls.

**Required fix:** §2.4. The sequential chooser has no recursion and no depth constant.

**Tests:** §7 item 2.

---

### N-05 — LOW–MEDIUM — The balance census counts openers of any parameter family

**Status:** Verified. `4cc2ba9c` returns `ambiguous_structure`; the prototype accepts.

**Location:** `tool_call_stream.cpp:267`. `parse_tool_parameter_open` matches `<parameter…>` and
`<param…>`, but Round-3 §8.5 says "skip closers of nested **same-family** openers". A
`<param=x>` literal inside a `<parameter>` value consumes the real closer in pass 1. The path then
counts as lazy (unbalanced), and with a second completing base the region becomes ambiguous.

**Reproducer (strict; tools** `write{path,content}`**,** `bash{command}`**):**

```cpp
const std::string ex      = tool_call("bash", {{"command", "ls"}});
const std::string content = "Doc <param=x> marker\n" + ex + "\nmore";
const std::string text    = "Example:\n" + ex + "\nNow writing.\n" + tool_call("write", {{"path", "d.md"}, {"content", content}});
```

- `4cc2ba9c`: text, `ambiguous_structure`.
- Required: 1 call `write`, `content` byte-exact. The prose example stays content.

**Required fix:** in `CandidateWalk::next` (§2.6), count only openers of the value's own family:
`parse_tool_open_header(text_.substr(pos_), family_, opener) == ToolHeaderStatus::Complete`. A
different-family opener is ordinary bytes (`++pos_`).

**Pitfall:** this step changes outcomes on purpose. Review every changed line of the outcome dump
(§5 step 4) and confirm that it involves a cross-family opener inside a value.

**Tests:** §7 item 4.

---

### N-06 — LOW — Fence diagnostics are wrong; the fence warning hides the reason

**Status:** Verified.

- A response with a **closed** code block followed by a call (```` "Run this:\n```bash\nls\n```\n" + call ````) reports `ended_in_unclosed_fence = true`. This is the most common agent pattern, so the request log carries this false flag constantly.
- An accepted `write` whose content is ```` "```python\nprint(1)" ```` reports `ended_in_unclosed_fence = true`.
- An accepted `write` with a fenced example reports `fenced_markers_suppressed = 2`.

**Location:**

- `tool_call_stream.cpp:1042`: the pre-latch scan runs over `content_ + pending_ws_ + marker_prefix_`. After a latch, the whitespace held before the marker (including the `\n` that closes a fence line) has moved into `region_`. The scan therefore never sees the closing line end and reports the fence as open.
- `tool_call_stream.cpp:1043–1046`: `compute_fence_diagnostics(pre_latch, region_)` always scans the latched region, including the values of an accepted call.
- `src/serve/operational_log.cpp:333–352`: the fence branch comes **before** the fallback-reason branch, so a rejected region whose text contains an unclosed fence logs `tool-call fence left unclosed` instead of `tool markup returned as text | <reason>`.

**Required fix:**

1. In `finish()`:
   - Always compute the pre-latch part.
   - Its `ended_in_unclosed_fence` is `false` whenever a marker latched: a latch happens only outside a fence, so the pre-latch stream cannot end inside one. Its suppressed-marker count is unaffected (the missing whitespace contains no markers).
   - Add the region part (`compute_fence_diagnostics({}, region_)`) **only on paths that return the region as text**: the Stage-2 ambiguity return and the final reject. Do not add it on any accepting path (Stage 1, Stage 2, Stage 3).
   - The no-latch path is unchanged.
   - `ToolCallOutputDecoder::finish` keeps adding the decoder's own pre-latch part. Apply the same latch rule there (the decoder knows `machine_.latched()`), so streaming stays equal to one-shot.
2. Reorder `render_tool_call_fallback`:
   - When `marker_seen` and the reason is not `None`, keep today's branches (the TruncatedTail info record, then the warning line).
   - The warning line becomes `req#<id> tool markup returned as text | <reason>[ | fenced_markers_suppressed=N][ | parse budget exhausted][ | <snippet>]`:
     - the fence part is present when `N > 0` and `ended_in_unclosed_fence`;
     - the budget part is present when `parse_budget_exhausted` (N-07);
     - the snippet stays last and is unchanged.
   - Emit the separate `tool-call fence left unclosed` warning only when `!marker_seen`, `tool_calls.empty()`, `ended_in_unclosed_fence` and `N > 0`.
3. In `parse_qwen_tool_call_output`, also copy the fence fields and `parse_budget_exhausted` into the diagnostics of the R3-08 `ambiguous_structure` fallback. That path builds fresh diagnostics today.

**Reproducers:**

- N6d, accepted, the common case: ```` "Run this:\n```bash\nls\n```\n" + tool_call("bash", {{"command","ls"}}) ````. `4cc2ba9c`: `ended_in_unclosed_fence == true`. Required: 1 call, both fence fields 0/false.
- N6a, accepted: ```` tool_call("write", {{"path","a.md"},{"content","```python\nprint(1)"}}) ````. Required: 1 call, `fenced_markers_suppressed == 0`, `ended_in_unclosed_fence == false`.
- N6b, accepted: ```` tool_call("write", {{"path","a.md"},{"content","Ex:\n```xml\n" + tool_call("read", {{"path","foo.cpp"}}) + "\n```\nmore"}}) ````. Required: 1 call, both fence fields 0/false.
- N6c, rejected, strict, `OutputLimit`: ```` "<function=read>\n<parameter=path>\nx\n```xml\n" + tool_call("bash", {{"command","ls"}}) + "\n" ````. Required: text, `malformed_structure`, `fenced_markers_suppressed == 2`, `ended_in_unclosed_fence == true`.
- Round-3 S3 (no latch): unchanged (`fenced_markers_suppressed == 2`, `ended_in_unclosed_fence == true`).

**Tests:** §7 items 5 and 10.

---

### N-07 — LOW — No diagnostic when a work bound fires

**Status:** Verified by code reading. `ConsistentCompleter::exhausted` is never reported, and the
256-attempt chain stop sets nothing.

**Required fix:**

1. Add `bool parse_budget_exhausted = false;` to `ToolCallStreamResult` (`tool_call_stream.h`) and to `ninfer::ToolCallParseDiagnostics` (`include/ninfer/types.h`, after `ended_in_unclosed_fence`, with a one-line comment). The defaulted `operator==` covers it.
2. Set it in `finish()` when the Stage-2 budget is exhausted (§2.5).
3. Propagate it in every diagnostics construction in `parse_qwen_tool_call_output`: the fallback path, the ambiguity path and the success path. The no-marker path leaves it false.
4. Request log (`src/serve/request_log.cpp`): add `{"parse_budget_exhausted", …}` next to the Round-3 fields.
5. Operational log: add the suffix from N-06 item 2.
6. Docs: add the field to the diagnostics lists in `docs/serving.md` and `docs/tool_call_parser.md`.

There is no chain-bound diagnostic, because §2.3 removes the bound.

**Tests:** §7 items 7 and 10.

---

### N-08 — LOW — Streaming ≠ one-shot for whitespace on a fence close line

**Status:** Verified. The Round-3 fuzz test hides it with an `rtrim` relaxation.

**Reproducer (strict,** `StopToken`**):**```` std::string("```\nx\n```  \r\n") + tool_call("bash", {{"command","ls"}}) ````

- One-shot content: ```` "```\nx\n```" ````.
- Streamed `visible + terminal.content`: ```` "```\nx\n``` \r" ````.

**Root cause:** `feed()` (`tool_call_stream.cpp:953`) publishes every fence `Content` byte
immediately, including the spaces and `\r` after the closing run. The one-shot entry rtrims them
away before an accepted region.

**Required fix:** in `feed()`, a fence `Content` byte that is format whitespace is held instead of
published. Keep the byte order:

```cpp
if (fence_.consume(byte) == FenceTracker::Verdict::Content) {
    if (is_tool_format_whitespace(byte)) {
        if (!marker_prefix_.empty()) {
            publish(pending_ws_, visible);
            pending_ws_.clear();
            publish(marker_prefix_, visible);
            marker_prefix_.clear();
        }
        pending_ws_.push_back(byte); // held like any other whitespace: may precede a latch
        continue;
    }
    publish(pending_ws_, visible);
    pending_ws_.clear();
    publish(marker_prefix_, visible);
    marker_prefix_.clear();
    publish(std::string_view(&byte, 1), visible);
    continue;
}
```

The fence tracker's verdicts do not change. Only the publication timing of whitespace changes.

**Tests:** §7 items 6 and 9. The rewritten fuzz test compares exactly, without `rtrim`.

---

### N-09 — LOW — Test defects

| # | Location (tests/test_tool_call_parser.cpp at 4cc2ba9c)  | Defect                                                                                                                                                                                                                                                                           | Required change                                                                                                                                                                                |
|---|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| a | `main()` (≈ line 3669)                                  | `test_composed_schema_types()` (defined at line 409) is no longer called. It passes when called.                                                                                                                                                                                 | Call it again from `main()`.                                                                                                                                                                   |
| b | `test_round3_spec_corpus`, block "R1" (lines 3336–3351) | The fixture uses an inline `"Note: </parameter>"`, which the canonical-framing rule rejects. That is **not** the Round-3 §9 residual R1, and the comment claims "the section-9 phantom acceptance is not implemented", which is false: the real R1 still commits a phantom call. | Relabel the block "R1-inline (inline closer note)" and rewrite the comment to say the canonical-framing rule rejects an inline closer. Keep its expectations. Add the real R1 pin (§7 item 8). |
| c | `test_round3_streaming_equivalence_fuzz` (line 3542)    | 512 texts, strict only, `StopToken` only, `rtrim` relaxation                                                                                                                                                                                                                     | Replace the body (§7 item 9).                                                                                                                                                                  |
| d | `test_round3_work_bounds` (lines 3604–3660)             | Asserts `stage2_steps < 4096` for a step definition that changes in this round; the budget-exhaustion case asserts no diagnostic                                                                                                                                                 | Update (§6).                                                                                                                                                                                   |
| e | `tests/test_request_log.cpp`                            | The Round-3 fields (`markup_tolerant_completion`, `fenced_markers_suppressed`, `ended_in_unclosed_fence`) are not asserted                                                                                                                                                       | Assert them and `parse_budget_exhausted` (§7 item 10).                                                                                                                                         |
| f | `test_fenced_content_never_latches` (CR6 fixtures)      | Only the reason is asserted; the fence fields are not                                                                                                                                                                                                                            | Assert `fenced_markers_suppressed >= 1` for each fixture, plus `ended_in_unclosed_fence` for the unclosed-fence fixture.                                                                       |

---

### N-10 — LOW — Wrong statements in help text and documentation

1. `src/serve/serve_options.cpp:227–230`**.** The help claims Stage 2 is a tolerant feature; it
   runs in both modes. Replace the four lines with the text below, and update
   `tests/test_serve_options.cpp` if it pins the text:

   ```text
   "  --tolerant-tool-calls      keep function-closed Qwen calls before a cut-off or\n"
   "                             malformed tail (calls with arguments before a malformed\n"
   "                             tail only after a natural stop); never an open value or\n"
   "                             an undeclared name (strict all-or-nothing by default)\n"
   ```
2. `docs/serving.md`**, tolerant paragraph** (it starts "By default the parser keeps that
   all-or-nothing behaviour. With `--tolerant-tool-calls` a region that fails …"). Replace it with:

   > By default the parser keeps that all-or-nothing behaviour. The consistent completion (Stage 2)
   > is part of both modes. With `--tolerant-tool-calls`, a region that neither stage accepts may
   > still return the calls whose function close was consumed, with the `truncated_tail` reason:
   > - When the output ends inside the region (a cut next call, parameter or wrapper close), the function-closed calls before the cut are returned, whatever the finish reason.
   > - When the region breaks on a malformed token or on trailing text, calls without parameters are returned. Calls with parameters are returned only after a natural stop (`StopToken`, or no reported reason), and only when the remaining text contains no `</parameter>` or `</param>` closer that could still belong to a value.
   > - After a cut (`StopString`, `OutputLimit`, `ContextCapacity`, `Cancelled`), such a region is returned as text.
   >
   > A call whose function close is missing, an undeclared tool name, and a call of a declared
   > tool with a repeated parameter name or (when its declared schema is unambiguous) with a
   > non-declared name after its first parameter are never returned as calls.
3. `docs/serving.md`**, consistent-completion paragraph.** Append: "Stage 2 is bounded by a
   deterministic work budget (four units per region byte, at least 100 000); a region that
   exhausts it is returned as text and records `parse_budget_exhausted`." Add
   `parse_budget_exhausted` to the diagnostics list (≈ lines 1087–1101).
4. `docs/tool_call_parser.md:111–112`**.** Restore the broken sentence: "… still falls back to
   text. **An** undeclared name is a break in tolerant mode …". Also align the rest of that
   paragraph with item 2.
5. `docs/tool_call_parser.md:134–141`**.** Replace the sentence that says an unbalanced candidate
   "is discarded" with:

   > Pass 1 tries only the closers that are balanced against nested openers of the same parameter
   > family; only when no pass-1 candidate stands does pass 2 try every closer of the family in
   > order, and a standing pass-2 boundary marks the completion as unbalanced. The earliest base
   > whose completion is balanced wins; otherwise a single completion wins; two or more unbalanced
   > completions make the region `ambiguous_structure`.

   Add the residual from N-03 (a degenerate output commits its last complete example) and the
   budget rule from item 3.
6. **Code comments in** `tool_call_stream.cpp`**.**
   - The Stage-2 header comment "the only mechanism that makes a truncated region complete" is wrong. Replace it with "resolves value boundaries that the greedy one-token rule chose wrongly; it never completes a truncated region".
   - Update the `break_offset` comment (§2.1).
7. **Round-3 progress file.** Do not rewrite it. In the Round-4 progress file, record these false
   Round-3 claims and their corrections:
   - "closer-less-region skip … behavior-neutral" (false, N-02)
   - "no other existing assertion changed" (false, N-09 a)
   - "the section-9 phantom acceptance is not implemented" (false, N-09 b)
   - "dead between-round path removed" (not in the diff)
   - "budgets fail closed" as a time bound (false, N-01)
   - "surface difference, not a bug" for the streaming whitespace (N-08)

---

### N-11 — LOW — `noexcept` functions allocate

**Location:** `ToolCallStreamParser::marker_byte` (`tool_call_stream.cpp:902` and its declaration in
`tool_call_stream.h`) is `noexcept` but appends to `std::string`. `select_value` is `noexcept` and
allocates; it is removed by §2.4.

**Required fix:**

- Remove `noexcept` from `marker_byte`.
- Do not mark any new Stage-2 function `noexcept` if it allocates (`run_stage2_base`, materialization, `RegionIndex` construction).
- `WorkBudget::charge`, `RegionIndex::next_close` and `has_close_at_or_after` may stay `noexcept`.

The engine already recovers from `std::bad_alloc`; `noexcept` turned that into `std::terminate`.

---

## 4. Behavior summary after Round 4

| Stage           | Runs when                                | Change in this round                                                                                                        |
|-----------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Pre-latch feed  | every byte before a latch                | fence-content whitespace is held (N-08)                                                                                     |
| Stage 1         | at `finish()`                            | absolute positions, index lookups, no attempt bound (N-03)                                                                  |
| Stage 2         | Stage 1 accepted nothing                 | sequential chooser, global budget, shared memo, early exit, both families, same-family balance (N-01, N-02, N-04, N-05)     |
| Stage 3         | tolerant, nothing accepted, no ambiguity | unchanged (also after Stage-2 exhaustion)                                                                                   |
| Output boundary | a region was accepted                    | unchanged (CR5 identity, R3-08, normalization)                                                                              |
| Diagnostics     | always                                   | no pre-latch unclosed flag after a latch; region fence counts only for text results (N-06); `parse_budget_exhausted` (N-07) |

---

## 5. Implementation order and refactor guard

### 5.1 Refactor guard

The outcome-dump tool in Appendix B prints one line per parse outcome:

- the Round-3 fuzz corpus × strict/tolerant × {`None`, `StopToken`, `OutputLimit`};
- three repository documents that this round does not edit, as `write` content;
- 300 random `edit`s of the Round-2 document.

For `StopToken` cases it also checks streaming against one-shot. It is a development tool. It is
not committed and not part of the test suite.

- Put it at `build-new-parser/round4/outcome_dump.cpp`. The build directory is ignored by git.
- Build it standalone (§8.2) and run it **from the repository root**.
- Save each run as `build-new-parser/round4/outcomes_<step>.txt`.
- Compare with `fc /b <a> <b>` or `git diff --no-index --stat <a> <b>`.

On `4cc2ba9c` the review run printed 36 306 lines in about 3 s, and no line contained `stream=0`.

### 5.2 Steps (one commit each; Conventional Commits)

Each step adds its tests (§7) and applies its part of §6 in the same commit. Finish a step only when
the affected suites are green and its outcome-dump condition holds.

| Step | Commit                | Content                                                                                                                                                                                                             | Outcome-dump condition                                                                                                                               |
|------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0    | (no commit)           | Build the Appendix-B and Appendix-C tools against `4cc2ba9c`; save `outcomes_step0.txt` and the probe output                                                                                                        | —                                                                                                                                                    |
| 1    | `refactor(frontend):` | §2.1 absolute positions, §2.2 region index (Stage-1 lookups only), no `substr` copies                                                                                                                               | identical to step 0                                                                                                                                  |
| 2    | `perf(frontend):`     | §2.4–§2.6 sequential chooser, global budget, shared memo, early exit; N-04; N-11; `parse_budget_exhausted` in `ToolCallStreamResult`, `ToolCallParseDiagnostics` and `parse_qwen_tool_call_output` (N-07 items 1–3) | identical to step 1                                                                                                                                  |
| 3    | `fix(frontend):`      | §2.3 no chain attempt bound (N-03)                                                                                                                                                                                  | identical to step 2                                                                                                                                  |
| 4    | `fix(frontend):`      | N-02 (skip via `has_close_at_or_after`) and N-05 (same-family walk)                                                                                                                                                 | differences allowed only for `<param>`-family regions and cross-family openers; list each changed line in the progress file with its explanation     |
| 5    | `fix(frontend):`      | N-06 item 1 (fence diagnostics scope, ambiguity-path copy) and N-08 (whitespace hold)                                                                                                                               | differences allowed only in the two fence fields (`fenced_markers_suppressed`, `ended_in_unclosed_fence`); every line must keep `stream=1` (or `-1`) |
| 6    | `fix(serve):`         | N-06 item 2 and N-07 items 4–5 (operational log, request log)                                                                                                                                                       | identical to step 5                                                                                                                                  |
| 7    | `test:`               | N-09 (re-registration, R1 pin, fuzz rewrite, CR6 field assertions, request-log assertions)                                                                                                                          | identical to step 6                                                                                                                                  |
| 8    | `docs:`               | N-10 (help text, `serving.md`, `tool_call_parser.md`, code comments, Round-4 progress file)                                                                                                                         | identical to step 7                                                                                                                                  |

After steps 2, 3 and 8, run the Appendix-C probe and record the numbers in the progress file.

---

## 6. Existing assertions that change (intended)

| Test (tests/test_tool_call_parser.cpp unless stated)                                                                              | Current expectation                     | New expectation                                                                                            | Finding    |
|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|------------------------------------------------------------------------------------------------------------|------------|
| `test_round3_work_bounds`, "2000 closer triples bound the Stage-2 steps"                                                          | `term.stage2_steps < 4096`              | `term.stage2_steps <= std::max<std::uint64_t>(100000, 4 * text.size())` and `!term.parse_budget_exhausted` | N-01, N-07 |
| same test, "an exhausted Stage-2 budget fails closed"                                                                             | `Invalid`, no calls                     | additionally `term.parse_budget_exhausted`                                                                 | N-07       |
| same test, "4000 bare open regions"                                                                                               | `stage2_steps == 0`                     | unchanged (no closer, so every base is skipped)                                                            | —          |
| `test_round3_streaming_equivalence_fuzz`                                                                                          | 512 texts, strict, `StopToken`, `rtrim` | §7 item 9                                                                                                  | N-08, N-09 |
| `test_round3_spec_corpus`, block "R1"                                                                                             | labelled as the §9 residual             | relabelled "R1-inline", comment corrected; expectations unchanged                                          | N-09       |
| `main()`                                                                                                                          | `test_composed_schema_types()` missing  | called again                                                                                               | N-09       |
| any assertion that expects a non-zero `fenced_markers_suppressed` or `ended_in_unclosed_fence == true` for an **accepted** region | non-zero / true                         | 0 / false                                                                                                  | N-06       |
| `tests/test_request_log.cpp`                                                                                                      | new fields unchecked                    | asserted (§7 item 10)                                                                                      | N-06, N-07 |
| `tests/test_serve_options.cpp` (only if it pins the tolerant help text)                                                           | old help text                           | N-10 item 1 text                                                                                           | N-10       |

No other existing assertion may change.

---

## 7. New tests

Add these to `tests/test_tool_call_parser.cpp` unless stated otherwise:

- Every structured case runs one-shot and streamed: chunk sizes 1/2/3/5/7, plus every split for fixtures below 512 bytes.
- The streamed result must equal the one-shot result exactly (content, calls, arguments, full diagnostics).
- Contracts are declared with string parameters unless stated.

 1. `test_round4_param_family_stage2` **(N-02).** Run the N-02 reproducer in strict and tolerant
    mode. Expect 1 call `write` with exact arguments and `markup_tolerant_completion == true`.
 2. `test_round4_stage2_many_values` **(N-04).** Run both N-04 reproducers. Expect exact values
    (`p69`; the `write` content) and `markup_tolerant_completion == true`.
 3. `test_round4_chain_without_attempt_bound` **(N-03).**
    - The N-03 reproducer gives 1 call `read` `{"path":"real.txt"}`.
    - 300 × (`"Example i:\n"` + example + `"\n"`) with `OutputLimit`, strict, gives 1 call `bash`: the last example (documented residual; the comment references this document's N-03).
    - 4 000 × `"<function=write>\n<parameter=content>\nopen value i\n"` with `None` gives `Invalid` with `stage2_steps == 0`.
 4. `test_round4_same_family_balance` **(N-05).** The N-05 reproducer gives 1 call `write` with exact
    content.
 5. `test_round4_fence_diagnostics_scope` **(N-06).** N6a, N6b, N6c, N6d and Round-3 S3 with the
    expectations from N-06, one-shot and streamed. Also check that the N-08 reproducer reports
    `ended_in_unclosed_fence == false`.
 6. `test_round4_fence_close_whitespace_streaming` **(N-08).** Run the N-08 reproducer over every
    split. `visible + terminal.content` must equal the one-shot content (```` "```\nx\n```" ````)
    exactly, and the calls and diagnostics must be equal.
 7. `test_round4_work_bounds` **(N-01, N-07).** Use `ToolCallStreamParser` with a default policy.
    With `limit = max(100000, 4 × text.size())`:
    - **Repetition with trailing prose:** 300 × (`"Example i:\n"` + example + `"\n"`) + `"That is all."`, `StopToken`, gives `Invalid` and `stage2_steps <= limit + 1`.
    - **Closer triples:**`tool_call("write", {{"path","d.md"},{"content","Doc:\n" + 2000 × "</parameter>\n</function>\n</tool_call>\ntext\n" + "End."}})` gives `Complete`, an exact value (via `parse_qwen_tool_call_output`), `!parse_budget_exhausted` and `stage2_steps <= limit + 1`.
    - **Tiny budget:** `stage2_step_budget = 8` on Round-3 S1 gives `Invalid`, no calls and `parse_budget_exhausted`.
 8. `test_round4_r1_residual_pinned` **(N-09 b).** The fixture is
    `"Example:\n" + tool_call("bash", {{"command","ls"}}) + "\nThen close with\n</parameter>\n</function>\n</tool_call>"`.
    - Strict, and tolerant with `StopToken`: 1 call `bash` with `{"command":"ls\n</parameter>\n</function>\n</tool_call>\nThen close with"}`, content `"Example:"` and `markup_tolerant_completion == true`.
    - The comment states that this is the documented Round-3 §9 residual R1 (phantom acceptance) and that any change must be deliberate.
 9. **Rewritten** `test_round3_streaming_equivalence_fuzz` **(N-08, N-09 c).** Keep the name. Cover:
    - the Round-3 §10.3 fragment corpus, `std::mt19937 rng(20260930)`, 2 000 texts, the same tail rule as today;
    - modes strict and tolerant, finish reasons `StopToken` and `OutputLimit`;
    - partitions: whole text, byte-wise, and random chunk sizes 1..7 seeded with the text index.

    Compare exactly: `visible + terminal.content == one.content`, call names and arguments, and
    `terminal.diagnostics == one.diagnostics`. No `rtrim`.
10. `tests/test_request_log.cpp` **(N-06, N-07).**
    - The `request_done` JSON contains `markup_tolerant_completion`, `fenced_markers_suppressed`, `ended_in_unclosed_fence` and `parse_budget_exhausted` with the outcome's values.
    - `render_tool_call_fallback` precedence and format (N-06 item 2):
      - (i) `marker_seen`, reason `trailing_content`, `ended_in_unclosed_fence`, N = 2, empty text gives `"req#7 tool markup returned as text | trailing content | fenced_markers_suppressed=2"`. With text set, the snippet follows as the last part.
      - (ii) `!marker_seen`, no calls, unclosed fence, N = 2 gives the `tool-call fence left unclosed` warning.
      - (iii) `parse_budget_exhausted` adds `" | parse budget exhausted"` after the fence part and before the snippet.
    - Match the existing `pretty_code` rendering of the reason (today's tests expect `duplicate parameter` for `duplicate_parameter`).

---

## 8. Commands (CPU only)

### 8.1 Suites

```powershell
cmake --build build-new-parser --config Release --parallel 16 `
  --target ninfer_tool_call_parser_test ninfer_tool_call_grammar_test ninfer_tool_call_grammar_state_test `
           ninfer_qwen3_5_frontend_test ninfer_request_log_test ninfer_pretty_logging_test `
           ninfer_serve_options_test ninfer_engine_options_validation_test
ctest --test-dir build-new-parser -C Release -j 16 --output-on-failure `
  -R "^(ninfer_tool_call_parser_test|ninfer_tool_call_grammar_test|ninfer_tool_call_grammar_state_test|ninfer_qwen3_5_frontend_test|ninfer_request_log_test|ninfer_pretty_logging_test|ninfer_serve_options_test|ninfer_engine_options_validation_test)$"
```

Then run the full CPU suite as in Round 3 (`CUDA_VISIBLE_DEVICES=99`,
`ctest --test-dir build-new-parser -C Release -E "_real" --parallel 16`) and record the result.

### 8.2 Standalone tools and AddressSanitizer (Developer Command Prompt, repository root)

```bat
mkdir build-new-parser\round4\obj
set FRONT=src\models\qwen3_5\frontend
set SRCS=%FRONT%\tool_call_grammar.cpp %FRONT%\tool_call_stream.cpp %FRONT%\tool_call_parser.cpp %FRONT%\tool_call_grammar_state.cpp
set FLAGS=/nologo /std:c++20 /EHsc /O2 /utf-8 /W3 /wd4996 /Isrc /Iinclude /Ithird_party /Fobuild-new-parser\round4\obj\
cl %FLAGS% %SRCS% build-new-parser\round4\outcome_dump.cpp /Febuild-new-parser\round4\outcome_dump.exe
cl %FLAGS% %SRCS% build-new-parser\round4\perf_probe.cpp /Febuild-new-parser\round4\perf_probe.exe
build-new-parser\round4\outcome_dump.exe > build-new-parser\round4\outcomes_step0.txt
build-new-parser\round4\perf_probe.exe

rem AddressSanitizer run of the parser suite (run in the same prompt so the ASan runtime is on PATH)
cl /nologo /std:c++20 /EHsc /O1 /Zi /fsanitize=address /utf-8 /W3 /wd4996 /Isrc /Iinclude /Ithird_party ^
   /Fobuild-new-parser\round4\obj\ %SRCS% tests\test_tool_call_parser.cpp /Febuild-new-parser\round4\parser_asan.exe
build-new-parser\round4\parser_asan.exe
```

---

## 9. Acceptance checklist

```text
[ ] N-01: sequential chooser, global budget, shared memo, early exit, lazy walks; no value copies during the search
[ ] N-02: Stage-2 skip covers both families; reproducer accepted
[ ] N-03: no chain attempt bound; 300-example reproducer finds the real call
[ ] N-04: no recursion and no depth constant; 70-value reproducers accepted
[ ] N-05: same-family walk; reproducer accepted
[ ] N-06: pre-latch unclosed flag false after a latch; region fence counts only for text results; operational-log precedence and suffixes
[ ] N-07: parse_budget_exhausted in types.h, results, request log, operational log, docs
[ ] N-08: fence-content whitespace held; exact streaming equality without rtrim
[ ] N-09: test_composed_schema_types called; R1 pinned and R1-inline relabelled; fuzz rewritten; request-log and CR6 field assertions
[ ] N-10: help text, serving.md, tool_call_parser.md, code comments corrected; Round-3 corrections recorded
[ ] N-11: no allocating noexcept function
[ ] outcome dump: steps 1, 2, 3, 6, 7, 8 identical to the previous step; steps 4 and 5 differences listed and explained
[ ] Appendix-C targets met; numbers recorded after steps 2, 3 and 8
[ ] §6 changes applied, each with a comment naming the Round-4 finding; no other existing assertion changed
[ ] target suites green; full CPU suite result recorded (no new failure); parser suite clean under AddressSanitizer
[ ] no GPU test executed; build-windows unused; <= 16 threads
[ ] progress file docs/NInfer_new_parser_design_round4_bugfix_progress.md complete
```

---

## 10. Residual risks after Round 4 (document them; do not claim they are solved)

All Round-3 §9 residuals remain. This round adds two:

| ID  | Input shape                                                                                   | Behavior                                   | Why                                                                                                                         |
|-----|-----------------------------------------------------------------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| DEG | a degenerate output made only of prose and unfenced complete examples                         | strict commits the last example            | byte-identical to prose followed by a real call (same as HEAD `4165d227`)                                                   |
| BUD | a region whose Stage-2 search exceeds the work budget (four units per byte, at least 100 000) | returned as text, `parse_budget_exhausted` | bounded engine-thread work is preferred over completing pathological regions; legitimate payloads use far less (Appendix C) |

---

## Appendix A — Reproducers (single-line literals)

`tool_call(name, {{k, v}, …})` serializes `"<tool_call>\n<function=" + name + ">\n"`, then
`"<parameter=" + k + ">\n" + v + "\n</parameter>\n"` per parameter, then
`"</function>\n</tool_call>"`.

| ID  | Text                                                                                                                                                                                                                                   | Required (strict unless stated)                                                                       |
|-----|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| N2  | `"<tool_call>\n<function=write>\n<param=path>\ndocs/x.md\n</param>\n<param=content>\n# Example\n<tool_call>\n<function=read>\n<param=path>\nfoo.cpp\n</param>\n</function>\n</tool_call>\nDone.\n</param>\n</function>\n</tool_call>"` | 1 call `write`, exact                                                                                 |
| N3  | 300 × (`"Example " + i + ":\n" + tool_call("bash", {{"command","ls -la"}}) + "\n"`) + `"Now the real call:\n" + tool_call("read", {{"path","real.txt"}})`                                                                              | 1 call `read`                                                                                         |
| N4a | `multi{p0..p69}`: 69 × `"<parameter=p<i>>\nv<i>\n</parameter>\n"` + `"<parameter=p69>\n# Example\n" + tool_call("read", {{"path","foo.cpp"}}) + "\nDone.\n</parameter>"`, wrapped as one call                                          | 1 call, `p69` exact                                                                                   |
| N4b | 69 × `tool_call("read", {{"path","f<i>"}}) + "\n"` + `tool_call("write", {{"path","d.md"},{"content","# Example\n" + tool_call("read", {{"path","foo.cpp"}}) + "\nDone."}})`                                                           | 70 calls                                                                                              |
| N5  | `"Example:\n" + EX + "\nNow writing.\n" + tool_call("write", {{"path","d.md"},{"content","Doc <param=x> marker\n" + EX + "\nmore"}})`, `EX = tool_call("bash", {{"command","ls"}})`                                                    | 1 call `write`, exact                                                                                 |
| N6d | ```` "Run this:\n```bash\nls\n```\n" + tool_call("bash", {{"command","ls"}}) ````                                                                                                                                                      | 1 call; fence fields 0/false                                                                          |
| N6a | ```` tool_call("write", {{"path","a.md"},{"content","```python\nprint(1)"}}) ````                                                                                                                                                      | 1 call; fence fields 0/false                                                                          |
| N6b | ```` tool_call("write", {{"path","a.md"},{"content","Ex:\n```xml\n" + tool_call("read", {{"path","foo.cpp"}}) + "\n```\nmore"}}) ````                                                                                                  | 1 call; fence fields 0/false                                                                          |
| N6c | ```` "<function=read>\n<parameter=path>\nx\n```xml\n" + tool_call("bash", {{"command","ls"}}) + "\n" ````, `OutputLimit`                                                                                                               | text, `malformed_structure`, suppressed 2, unclosed true                                              |
| N8  | ```` "```\nx\n```  \r\n" + tool_call("bash", {{"command","ls"}}) ````                                                                                                                                                                  | 1 call; streaming content equals one-shot ```` "```\nx\n```" ````; `ended_in_unclosed_fence == false` |
| R1  | `"Example:\n" + tool_call("bash", {{"command","ls"}}) + "\nThen close with\n</parameter>\n</function>\n</tool_call>"`                                                                                                                  | 1 call `bash` (documented phantom), `markup_tolerant_completion`                                      |
| P2s | 300 × (`"Example " + i + ":\n" + EX + "\n"`) + `"That is all."`                                                                                                                                                                        | text; `stage2_steps <= max(100000, 4 × bytes) + 1`                                                    |

---

## Appendix B — Outcome-dump tool (refactor guard)

Save as `build-new-parser/round4/outcome_dump.cpp`. Build and run as in §8.2. It uses only APIs
that exist at `4cc2ba9c`, so the same source runs before and after every step. It deliberately
does not print `parse_budget_exhausted`, which step 2 adds; the work-bound tests cover that
field.

````cpp
// Round-4 refactor guard: one line per parse outcome. Not part of the product or the test suite.
// Build it standalone against the frontend sources and run it from the repository root; compare
// the output of two builds byte for byte.
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fi = ninfer::models::qwen3_5::frontend;
using Json   = nlohmann::json;
using ninfer::FinishReason;

namespace {

std::uint64_t fnv(std::string_view s) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (const char c : s) {
        h ^= static_cast<std::uint8_t>(c);
        h *= 0x100000001b3ULL;
    }
    return h;
}

std::string tool_call(const std::string& name, const std::vector<std::pair<std::string, std::string>>& params) {
    std::string t = "<tool_call>\n<function=" + name + ">\n";
    for (const auto& [k, v] : params) { t += "<parameter=" + k + ">\n" + v + "\n</parameter>\n"; }
    return t + "</function>\n</tool_call>";
}

std::string definition(const std::string& name, const std::vector<std::string>& props) {
    Json p = Json::object();
    for (const auto& k : props) { p[k] = Json{{"type", "string"}}; }
    return Json{{"type", "function"},
                {"function", Json{{"name", name}, {"parameters", Json{{"type", "object"}, {"properties", p}}}}}}
        .dump();
}

std::string read_lf(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    std::string o;
    o.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (!(s[i] == '\r' && i + 1 < s.size() && s[i + 1] == '\n')) { o += s[i]; }
    }
    return o;
}

std::string diagnostics_signature(const ninfer::ToolCallParseDiagnostics& d) {
    return std::to_string(d.marker_seen) + "," + std::to_string(d.structured_call_count) + "," +
           std::to_string(d.empty_arguments_omitted) + "," + std::to_string(d.schema_mismatch_arguments) + "," +
           std::to_string(d.duplicate_parameters_repaired) + "," + std::to_string(d.markup_tolerant_completion) +
           "," + std::to_string(d.fenced_markers_suppressed) + "," + std::to_string(d.ended_in_unclosed_fence) +
           "," + ninfer::tool_call_parse_fallback_reason_name(d.fallback_reason);
}

std::string outcome_signature(const fi::ParsedToolCallOutput& p) {
    std::string calls;
    for (const auto& c : p.tool_calls) { calls += c.name + "(" + std::to_string(fnv(c.arguments_json)) + ");"; }
    return std::to_string(p.is_tool_call_response) + "|" + std::to_string(fnv(p.content)) + "|" + calls + "|" +
           diagnostics_signature(p.diagnostics);
}

bool streams_like_one_shot(const std::string& text, const std::shared_ptr<const fi::ToolCallOutputContract>& contract,
                           bool tolerant, FinishReason reason, const fi::ParsedToolCallOutput& one, std::uint32_t seed) {
    fi::ToolCallOutputDecoder decoder(contract, 64, tolerant);
    std::mt19937 rng(seed);
    std::string visible;
    for (std::size_t pos = 0; pos < text.size();) {
        const std::size_t n = 1 + rng() % 7;
        visible += decoder.feed(std::string_view(text).substr(pos, n));
        pos += n;
    }
    const auto terminal = decoder.finish(reason);
    if (visible + terminal.content != one.content || terminal.tool_calls.size() != one.tool_calls.size() ||
        !(terminal.diagnostics == one.diagnostics)) {
        return false;
    }
    for (std::size_t i = 0; i < one.tool_calls.size(); ++i) {
        if (terminal.tool_calls[i].name != one.tool_calls[i].name ||
            terminal.tool_calls[i].arguments_json != one.tool_calls[i].arguments_json) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    std::vector<std::string> defs = {definition("write", {"path", "content"}), definition("bash", {"command"}),
                                     definition("read", {"path"}),
                                     definition("edit", {"path", "old_string", "new_string"})};
    const auto contract =
        fi::build_tool_call_output_contract(std::span<const std::string>(defs.data(), defs.size()), true);
    std::uint32_t id = 0;
    auto emit = [&](const std::string& text, bool tolerant, FinishReason reason, bool check_stream) {
        const auto one = fi::parse_qwen_tool_call_output(text, 64, *contract, tolerant, reason);
        const int stream_ok =
            check_stream ? (streams_like_one_shot(text, contract, tolerant, reason, one, id) ? 1 : 0) : -1;
        std::printf("%u|%d|%d|%s|stream=%d\n", id, tolerant ? 1 : 0, static_cast<int>(reason),
                    outcome_signature(one).c_str(), stream_ok);
        ++id;
    };

    // (a) The Round-3 section 10.3 fragment corpus.
    const std::vector<std::string> frags = {
        "<tool_call>\n", "</tool_call>\n", "<function=write>\n", "<function=bash>\n", "</function>\n",
        "<parameter=path>\n", "<parameter=content>\n", "<parameter=command>\n", "</parameter>\n",
        "text ", "x\n", "```\n", "```xml\n", "~~~\n", "  ```\n", "<", ">", "\"", "'", "\r\n",
        "<function name=\"", "<invoke=bash>", "</invoke>", "<param=command>", "</param>",
        "<function_calls>\n", "</function_calls>\n", "echo '</parameter>'\n", "<tool_c", "<function",
        "prose. ", "\n"};
    const std::string tail = tool_call("bash", {{"command", "ls"}});
    std::mt19937 rng(20260930);
    for (int t = 0; t < 6000; ++t) {
        std::string text;
        const int count = 2 + static_cast<int>(rng() % 14);
        for (int i = 0; i < count; ++i) { text += frags[rng() % frags.size()]; }
        if (rng() % 3 == 0) { text += tail; }
        for (const bool tolerant : {false, true}) {
            for (const FinishReason reason : {FinishReason::None, FinishReason::StopToken, FinishReason::OutputLimit}) {
                emit(text, tolerant, reason, reason == FinishReason::StopToken);
            }
        }
    }
    // (b) Realistic payloads: documents that this round does not edit, written verbatim (LF).
    for (const char* path : {"docs/NInfer_new_parser_design_bugfix_implementation.md",
                             "docs/NInfer_new_parser_design_round2_bugfix_implementation.md",
                             "docs/NInfer_new_parser_design_round3_bugfix_implementation.md"}) {
        const std::string body = read_lf(path);
        if (body.empty()) { continue; }
        for (const bool tolerant : {false, true}) {
            emit("I'll write the file.\n\n" + tool_call("write", {{"path", path}, {"content", body}}), tolerant,
                 FinishReason::StopToken, true);
        }
    }
    // (c) Random edits of a document with real multi-line tool-call examples (UTF-8-safe cuts).
    {
        const std::string body = read_lf("docs/NInfer_new_parser_design_round2_bugfix_implementation.md");
        auto continuation = [](char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; };
        std::mt19937 erng(42);
        for (int i = 0; i < 300 && !body.empty(); ++i) {
            std::size_t a = erng() % body.size();
            while (a < body.size() && continuation(body[a])) { ++a; }
            std::size_t e = std::min(body.size(), a + 20 + erng() % 600);
            while (e < body.size() && continuation(body[e])) { ++e; }
            const std::string old_s = body.substr(a, e - a);
            emit(tool_call("edit", {{"path", "docs/x.md"}, {"old_string", old_s}, {"new_string", old_s + "\n(edited)"}}),
                 false, FinishReason::StopToken, true);
        }
    }
    return 0;
}
````

---

## Appendix C — Performance probe

Save as `build-new-parser/round4/perf_probe.cpp`. Build it as in §8.2 (Release `/O2`) and run it
from the repository root. Wall-clock numbers depend on the machine. The hard requirements are the
deterministic counters in §7. The targets below are for a desktop CPU of the review machine's
class; record your measured numbers in the progress file. Any case above **0.5 µs per region byte**
is a failure to investigate.

| Case                                                  | 4cc2ba9c (review machine)       | Target after step 3                          |
|-------------------------------------------------------|---------------------------------|----------------------------------------------|
| P1 repetition k=1500, `OutputLimit` (159 KB)          | 1 764 ms, `ambiguous_structure` | ≤ 10 ms; 1 call (last example, residual DEG) |
| P2 repetition k=1500 + trailing prose (159 KB)        | 2 223 ms, `trailing_content`    | ≤ 60 ms; text (budget may fire)              |
| P3 300 examples + real `read` call (32 KB)            | 91.5 ms, call lost              | ≤ 10 ms; 1 call `read`                       |
| P4 8 000 closer triples in one `write` value (344 KB) | 108.5 ms                        | ≤ 30 ms; exact                               |
| P5 4 000 open bare regions, `None` (211 KB)           | 22.6 ms                         | ≤ 10 ms                                      |
| P6 70 parameters, last value needs Stage 2            | text (depth bound)              | ≤ 5 ms; 1 call                               |
| P7 2 000 examples in one `write` value (186 KB)       | 2.0 ms                          | ≤ 10 ms; exact                               |
| P8 500 KB plain `write`                               | 3.6 ms                          | ≤ 4 ms                                       |
| P9 Round-2 document as `write` content (47 KB)        | 0.7 ms                          | ≤ 2 ms; exact                                |

```cpp
// Round-4 performance probe: wall-clock (best of 3) and deterministic Stage-2 work per case.
// Not part of the product or the test suite. Release build, run from the repository root.
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "models/qwen3_5/frontend/tool_call_stream.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fi = ninfer::models::qwen3_5::frontend;
using Json   = nlohmann::json;
using ninfer::FinishReason;

namespace {

std::string tool_call(const std::string& name, const std::vector<std::pair<std::string, std::string>>& params) {
    std::string t = "<tool_call>\n<function=" + name + ">\n";
    for (const auto& [k, v] : params) { t += "<parameter=" + k + ">\n" + v + "\n</parameter>\n"; }
    return t + "</function>\n</tool_call>";
}

std::string definition(const std::string& name, const std::vector<std::string>& props) {
    Json p = Json::object();
    for (const auto& k : props) { p[k] = Json{{"type", "string"}}; }
    return Json{{"type", "function"},
                {"function", Json{{"name", name}, {"parameters", Json{{"type", "object"}, {"properties", p}}}}}}
        .dump();
}

std::string read_lf(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    std::string o;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (!(s[i] == '\r' && i + 1 < s.size() && s[i + 1] == '\n')) { o += s[i]; }
    }
    return o;
}

double best_ms(const std::function<void()>& f) {
    double best = 1e300;
    for (int r = 0; r < 3; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

} // namespace

int main() {
    std::vector<std::string> defs = {definition("write", {"path", "content"}), definition("bash", {"command"}),
                                     definition("read", {"path"})};
    {
        std::vector<std::string> multi;
        for (int i = 0; i < 70; ++i) { multi.push_back("p" + std::to_string(i)); }
        defs.push_back(definition("multi", multi));
    }
    const auto contract =
        fi::build_tool_call_output_contract(std::span<const std::string>(defs.data(), defs.size()), true);
    const std::string ex = tool_call("bash", {{"command", "ls -la"}});
    const std::string ex_read = tool_call("read", {{"path", "foo.cpp"}});

    struct Case {
        const char* id;
        std::string text;
        FinishReason reason;
    };
    std::vector<Case> cases;
    {
        std::string t;
        for (int i = 0; i < 1500; ++i) { t += "Example " + std::to_string(i) + ":\n" + ex + "\n"; }
        cases.push_back({"P1 repetition k=1500, OutputLimit", t, FinishReason::OutputLimit});
        cases.push_back({"P2 repetition k=1500 + trailing prose", t + "That is all.", FinishReason::StopToken});
    }
    {
        std::string t;
        for (int i = 0; i < 300; ++i) { t += "Example " + std::to_string(i) + ":\n" + ex + "\n"; }
        cases.push_back({"P3 300 examples + real read call", t + "Now the real call:\n" + tool_call("read", {{"path", "real.txt"}}),
                         FinishReason::StopToken});
    }
    {
        std::string content = "Doc:\n";
        for (int i = 0; i < 8000; ++i) { content += "</parameter>\n</function>\n</tool_call>\ntext\n"; }
        cases.push_back({"P4 8000 closer triples in one write value", tool_call("write", {{"path", "d.md"}, {"content", content + "End."}}),
                         FinishReason::StopToken});
    }
    {
        std::string t;
        for (int i = 0; i < 4000; ++i) { t += "<function=write>\n<parameter=content>\nopen value " + std::to_string(i) + "\n"; }
        cases.push_back({"P5 4000 open bare regions, None", t, FinishReason::None});
    }
    {
        std::vector<std::pair<std::string, std::string>> ps;
        for (int i = 0; i < 69; ++i) { ps.push_back({"p" + std::to_string(i), "v" + std::to_string(i)}); }
        ps.push_back({"p69", "# Example\n" + ex_read + "\nDone."});
        cases.push_back({"P6 70 parameters, last value needs Stage 2", tool_call("multi", ps), FinishReason::StopToken});
    }
    {
        std::string content = "Doc:\n";
        for (int i = 0; i < 2000; ++i) { content += ex + "\n"; }
        cases.push_back({"P7 2000 examples in one write value", tool_call("write", {{"path", "d.md"}, {"content", content + "End."}}),
                         FinishReason::StopToken});
    }
    cases.push_back({"P8 500 KB plain write", tool_call("write", {{"path", "big.txt"}, {"content", std::string(500 * 1024, 'x')}}),
                     FinishReason::StopToken});
    cases.push_back({"P9 round-2 document as write content",
                     tool_call("write", {{"path", "d.md"}, {"content", read_lf("docs/NInfer_new_parser_design_round2_bugfix_implementation.md")}}),
                     FinishReason::StopToken});

    for (const Case& c : cases) {
        fi::ParsedToolCallOutput parsed;
        const double ms = best_ms([&] { parsed = fi::parse_qwen_tool_call_output(c.text, 64, *contract, false, c.reason); });
        fi::ToolCallParsePolicy policy;
        policy.max_name_length = 64;
        fi::ToolCallStreamParser machine(policy);
        (void)machine.feed(c.text);
        const fi::ToolCallStreamResult raw = machine.finish(c.reason);
        std::printf("%-44s bytes=%7zu calls=%zu reason=%-20s stage2_steps=%9llu  %8.1f ms\n", c.id, c.text.size(),
                    parsed.tool_calls.size(), ninfer::tool_call_parse_fallback_reason_name(parsed.diagnostics.fallback_reason),
                    static_cast<unsigned long long>(raw.stage2_steps), ms);
    }
    return 0;
}
```