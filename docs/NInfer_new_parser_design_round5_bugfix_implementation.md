# NInfer `new_parser_design` — Round 5 Bugfix Implementation Specification

## 0. Purpose, scope, baseline

This is the binding implementation specification for the findings of the **fifth review**: the
post-merge review of the Round-4 tool-call parser after the branch was merged onto the new
`Wallawalla47/ninfer-custom` architecture. It is a delta on top of Round 4.

| Item | Value |
|---|---|
| Reviewed branch | `Hundsbuah/ninfer:new_parser_design` |
| Reviewed HEAD | `705c45e39261b44fa6dd9bea54d6f8d8ea901fad` (`fix: complete merge interface (envelope wide_verification + dedupe log_colour test)`) |
| Parser-side pre-merge parent | `e76f7466d7d68e09a9598f8c03114f65e3281d07` |
| Architecture merge commit | `bea04ce763a9bf94aa8c241a2bba5a537bb08354` |
| Upstream architecture parent | `ae779f1d464de81aff36f9bd6303c7e3446630c9` from `Wallawalla47/ninfer-custom:master` at review time |
| Reference behavior | `docs/NInfer_new_parser_design_round4_bugfix_implementation.md` + `docs/NInfer_new_parser_design_round4_bugfix_progress.md` |
| Review date | 2026-10-01 |
| Production configuration (user) | `ninfer serve`, Qwen3.8-27B native wrapped `<tool_call>` format, oh-my-pi agent client, strict parser by default; tolerant mode may be tested explicitly |
| Deliverable | Findings R5-01 … R5-08, merge-safe target architecture, required code changes, regression tests, refactor guard, CPU-only validation and acceptance gate |

Rounds 1–4 remain in force unless this document explicitly supersedes them. In particular, the
Round-4 parser-core decisions (fence tracking, wrapper ownership, declared-tool enforcement,
Stage-2 work bounds, streaming equality and fail-closed work-budget behavior) remain the baseline.
Round 5 primarily repairs **integration regressions introduced by the architecture merge** and
makes the terminal-reason contract merge-safe.

Where this document conflicts with Round 4, this document wins. The most important superseding
rule is this:

> **A terminal `FinishReason` is part of the `OutputSession` preview/commit transaction. The caller
> must never be able to commit a terminal preview while silently substituting
> `FinishReason::None`.**

Record all Round-5 implementation work in a new progress file:

```text
docs/NInfer_new_parser_design_round5_bugfix_progress.md
```

### 0.1 Hard constraints

- **CPU only.** Do not run model inference, GPU runtime tests, CUDA correctness tests, GPU
  benchmarks or any test target whose correctness depends on a usable GPU.
- Do not run `_real` tests. A full CPU gate may use `ctest -E "_real"` so the `_real` targets are
  excluded.
- Use `build-new-parser`; do not use `build-windows`.
- At most 16 build threads and 16 test threads.
- Before running an unfamiliar test, inspect its source/CMake registration. If GPU use is
  uncertain, do not run it.
- Do not implement or enable runtime constrained decoding in this round. The existing fail-fast
  behavior for `--constrained-tool-decoding=tool-calls-only` stays in force until it can be
  validated in an appropriate GPU environment.
- Do not weaken Round-4 parser integrity to make an integration test pass. A failing integration
  test must first be treated as evidence that the new architecture is supplying the parser the
  wrong transaction metadata.
- Do not copy whole pre-merge serve/engine files over the new architecture. Port only the
  tool-call-specific semantics described here.
- Every new tool-call fixture follows the existing test convention: a single-line C++ literal
  using `\n` escapes or the existing `tool_call(...)` helper.

### 0.2 Evidence status and verification boundary

The Round-5 findings below are based on a source-level review of the final post-merge branch and a
three-way comparison of:

1. `705c45e3...` — final reviewed post-merge branch;
2. `e76f7466...` — parser-side pre-merge state;
3. `ae779f1d...` — merged upstream architecture parent.

The review established the following directly from source:

- `tool_call_stream.{h,cpp}`, `tool_call_parser.{h,cpp}`, `tool_call_grammar.{h,cpp}`,
  `tool_call_grammar_state.cpp`, `output_session.cpp`, `include/ninfer/types.h` and the main parser
  tests survived the architecture merge without the parser-core Round-4 logic being replaced.
- The new upstream `EngineCore` calls `OutputSession::commit_preview()` without a finish reason,
  while the parser-side `OutputSession::commit_preview(FinishReason = None)` terminalizes the
  tool-call decoder using that argument.
- `finish_reasons[row]` still contains the real terminal reason in `EngineCore`, but it is assigned
  to the request **after** the reasonless `commit_preview()` call.
- Round-4 request-log/operational-log/help changes present in `e76f7466...` are absent again in
  `705c45e3...`.
- The current parser test suite intentionally pins the Round-3/Round-4 R1 phantom-call residual in
  `test_round4_r1_residual_pinned()`.
- `model_instance.cpp` still rejects non-`off` constrained tool decoding at startup; runtime
  constrained sampling is therefore not active.

The Round-4 progress file records substantial **pre-merge** CPU validation. That evidence remains
useful for the unchanged parser core, but it is **not proof that the final post-merge integration
at `705c45e3...` is correct**. Round 5 must therefore rerun the relevant CPU suites after the
integration fixes.

No independent post-merge runtime result may be claimed until the Round-5 progress file records
it.

### 0.3 Behavior that must not regress

The following are hard invariants carried forward from Rounds 1–4:

1. An open parameter value is never executable.
2. A function/invoke call is executable only after its matching function close was consumed.
3. An open or failed wrapper owns its ambiguous scope; recovery may not re-enter inside it.
4. Undeclared tool names never become structured calls, in strict or tolerant mode.
5. Declared duplicate/synthetic parameters retain the Round-3 ambiguity policy.
6. Literal tool markup inside a parameter value remains ordinary payload unless a proven outer
   structural boundary is selected.
7. Streaming and one-shot results remain exactly equal for content, tool calls, arguments and the
   complete `ToolCallParseDiagnostics` object.
8. Fence suppression remains fail-safe: markers inside a recognized code fence do not latch.
9. Stage 2 remains non-recursive and bounded by one deterministic global work budget per
   `finish()` call.
10. Work-budget exhaustion remains fail-closed and observable via `parse_budget_exhausted`.
11. `StopString`, `OutputLimit`, `ContextCapacity` and `Cancelled` remain **cut reasons**.
12. `StopToken` remains a natural stop. `FinishReason::None` may be used by parser-only callers
    whose terminal reason is genuinely unknown, but must not replace a known engine terminal
    reason.
13. Accepted payloads used by the existing `write`/`edit` realistic-payload regression corpus stay
    byte-exact unless a Round-5 safety policy explicitly documents an intended behavior change.
14. The default unconstrained sampling path remains bit-identical; this round does not silently
    activate the Phase-4 grammar constraint.

Any changed behavior not justified by a Round-5 finding is a regression.

---

## 1. Executive summary and finding index

The post-merge parser core is substantially healthier than the early implementation, but the final
branch is **not green**. The architecture merge reintroduced a terminal-metadata bug outside the
parser and rolled back the observability contract that Round 4 added around the parser.

The findings split into four groups:

- **Terminal transaction integrity:** the real `FinishReason` is lost before the parser sees the
  terminal commit, and the current API makes this failure compile silently.
- **Observability regressions:** request/operational logs and their tests lost the Round-4
  diagnostic fields and warnings.
- **Surface/documentation regressions:** help and docs again describe obsolete tolerant-mode
  behavior.
- **Residual ambiguity / native hardening:** the byte protocol still has a pinned phantom-call
  residual and the strict parser accepts more top-level syntax than the selected Qwen3.8 native
  template requires.

| ID | Severity | Title | Production impact for strict Qwen3.8 native |
|---|---|---|---|
| R5-01 | **HIGH** | The architecture merge loses the real terminal `FinishReason` before `ToolCallOutputDecoder::finish()` | **Direct correctness/integrity risk:** a cut can be parsed as a natural stop and can change retry/recovery decisions |
| R5-02 | **MEDIUM–HIGH** | The `commit_preview(FinishReason = None)` API and current tests make R5-01 merge-safe only by convention | Silent future regression risk; current CPU tests can stay green while the production path is wrong |
| R5-03 | **MEDIUM** | Request-log serialization lost four Round-4 parser diagnostics | Production diagnosis loses fence/completion/work-bound evidence |
| R5-04 | **MEDIUM** | Operational-log fence/budget diagnostics and marker-family snippets were rolled back | Important parser failures are again harder to distinguish operationally |
| R5-05 | **LOW–MEDIUM** | `--tolerant-tool-calls`, `serving.md` and parser documentation again describe obsolete behavior | Operator may enable/configure a mode based on false semantics |
| R5-06 | **HIGH (integrity residual)** | R1 phantom acceptance is still intentionally executable in strict mode | Byte-identical prose/example markup can become a real declared tool call |
| R5-07 | **MEDIUM hardening** | Strict Qwen3.8 native parsing still accepts compatibility-only top-level entries (`<function_calls>`, bare `<function>`, `<invoke>`) | Enlarges false-positive/recovery surface beyond the selected native template |
| R5-08 | **LOW–MEDIUM** | Phase-4/constrained-decoding documentation can imply more runtime protection than the current fail-fast implementation provides | Can create false confidence; runtime malformed generation remains possible |

### 1.1 Release-blocking classification

For the production target in §0, the minimum Round-5 release blockers are:

- **R5-01** must be fixed.
- **R5-02** must be fixed strongly enough that the same integration bug cannot silently compile on
  the next architecture merge.
- **R5-03/R5-04** must be restored because the Round-4 parser deliberately relies on those fields
  to expose fail-closed and fence decisions.
- The project must make an explicit, tested policy decision for **R5-06**. It is not acceptable to
  call the implementation “phantom-call safe” while the current strict test intentionally executes
  the reproducer.

R5-07 is strongly recommended for the native production path but may be staged after the R5-01–05
integration repair if compatibility is currently contractual.

---

## 2. Target design

### 2.1 Terminal reason belongs to the `OutputSession` preview transaction

The Round-4 implementation extended:

```cpp
PublishedOutput OutputSession::commit_preview(FinishReason finish_reason)
```

and defaulted the argument to `FinishReason::None`. The new upstream architecture retained its old
reasonless `commit_preview()` call sites. That combination is exactly why the merge regression
compiled.

Round 5 must remove this duplicated caller responsibility.

#### Required invariant

When a preview becomes terminal, the `FinishReason` returned by that preview is part of the
preview state. Committing that preview must terminalize the tool-call decoder with **the same
reason**, without asking `EngineCore` to restate it.

Formally:

```text
preview_model / preview_control / preview_terminal
            |
            | produces terminal decision R
            v
OutputSession preview state stores R
            |
            v
commit_preview()
            |
            +--> terminal ToolCallOutputDecoder::finish(R)
```

The caller may not substitute `None`, a stale reason or a reason from another row.

#### Required representation

Add preview-transaction metadata to `OutputSession::Impl`, for example:

```cpp
struct PreviewTerminalMetadata {
    bool terminal = false;
    FinishReason finish_reason = FinishReason::None;
};

PreviewTerminalMetadata preview_terminal;
```

A simpler pair of fields is acceptable, but the state must be explicit and covered by invariants.
Do **not** rely only on `preview_state.terminal`; the finish reason must be stored alongside it.

#### Required update rules

Every method that creates a preview must set the metadata deterministically:

- `preview_model(...)`
  - if the returned `OutputDecision` is non-terminal: store `{false, None}`;
  - if terminal: store `{true, decision.finish_reason}`.
- `preview_control(...)`
  - same rule.
- `preview_terminal(reason)`
  - store `{true, reason}`.

If a preview is replaced before commit, the metadata is replaced with it. If there is a rollback or
preview-discard path, the metadata must be rolled back/discarded with the rest of the preview
transaction.

#### Required commit rule

Restore the public API to:

```cpp
[[nodiscard]] PublishedOutput commit_preview();
```

and in the terminal block:

```cpp
if (impl_->state.terminal) {
    if (!impl_->preview_terminal.terminal ||
        impl_->preview_terminal.finish_reason == FinishReason::None) {
        std::terminate(); // or project-standard invariant failure
    }
    fi::ToolCallOutputDecoder::Terminal terminal =
        impl_->tool_call_output.finish(impl_->preview_terminal.finish_reason);
    ...
}
```

Use the project-standard invariant mechanism if one already exists; the essential requirement is
that a known terminal engine transaction cannot silently become `None`.

After commit, clear preview metadata together with `preview_ready`:

```cpp
impl_->preview_terminal = {};
```

#### Why this design is required instead of merely patching `EngineCore`

A minimal call-site patch such as:

```cpp
request->output.commit_preview(finish_reasons[row]);
```

would fix the current revision but would preserve the fragile duplicated contract. A future merge
from an upstream branch where `commit_preview()` is reasonless can reproduce the exact bug.

Binding the reason to the preview transaction has four properties:

1. the component that *decides* the terminal reason also stores it;
2. the commit cannot disagree with the preview;
3. upstream `EngineCore` can keep the reasonless `commit_preview()` API;
4. the compiler no longer needs every engine caller to remember a parser-specific side channel.

This is the selected Round-5 architecture.

### 2.2 Finish-reason decision matrix

The tool-call parser must observe the following terminal reasons end to end:

| Engine terminal reason | Parser classification | May be treated as natural stop? |
|---|---|---:|
| `StopToken` | natural stop | yes |
| `StopString` | deliberate cut | **no** |
| `OutputLimit` | deliberate cut | **no** |
| `ContextCapacity` | deliberate cut | **no** |
| `Cancelled` | deliberate cut | **no** |
| `None` | unknown / parser-only legacy caller | only outside a known engine terminal transaction |

This matrix is not documentation only. Section 7 requires tests whose structured-call outcome
changes if `OutputLimit`/`Cancelled` is accidentally replaced by `None`.

### 2.3 Observability is part of the parser contract

The Round-4 diagnostics are not optional logging decoration. They explain why structurally similar
model outputs were accepted, suppressed or returned as text.

The following fields must survive from parser result to request JSON and operational logging:

```text
marker_seen
structured_call_count
empty_arguments_omitted
schema_mismatch_arguments
duplicate_parameters_repaired
markup_tolerant_completion
fenced_markers_suppressed
ended_in_unclosed_fence
parse_budget_exhausted
fallback_reason
```

The merge must not silently delete a field from one sink while leaving it in
`ToolCallParseDiagnostics`.

### 2.4 Native syntax and compatibility syntax must be separable

The selected Qwen3.8 native template uses the wrapped form:

```text
<tool_call>
<function=NAME>
<parameter=ARG>
VALUE
</parameter>
</function>
</tool_call>
```

The parser currently also accepts compatibility forms at top level. Round 5 should make that
surface explicit instead of treating all entries as one strict grammar.

Introduce a syntax policy, for example:

```cpp
enum class ToolCallSyntaxMode : std::uint8_t {
    QwenWrappedNative,
    Compatibility,
};
```

and carry it in `ToolCallParsePolicy` / the output contract.

#### `QwenWrappedNative`

At top level, the only executable entry marker is:

```text
<tool_call>
```

Inside it, the canonical Qwen function/parameter tags remain valid. Bare `<function=...>`,
`<invoke...>` and `<function_calls>` do not latch as executable top-level tool regions.

#### `Compatibility`

Preserve today's wider forms, including the legacy wrapper/bare-function variants required by
existing callers.

Do not infer this mode from arbitrary template filename substrings. Prefer an explicit frontend /
engine option or metadata carried by the selected chat-template implementation.

If changing the default is considered API-breaking, land the mode and tests first, then make the
production Qwen3.8 configuration select `QwenWrappedNative` explicitly.

### 2.5 Ambiguous byte protocol: explicit policy, not an implicit claim

R5-06 cannot be losslessly solved by a byte parser while simultaneously guaranteeing arbitrary
payload fidelity. The same bytes can mean either:

- a real structural closer; or
- literal tool markup inside a string payload.

Round 4 deliberately chose payload fidelity for a class of these cases. That keeps realistic
`write`/`edit` payloads containing tool examples intact, but it also leaves the R1 phantom-call
reproducer executable.

Round 5 therefore requires an explicit product policy rather than pretending the ambiguity is
solved.

Recommended API:

```cpp
enum class ToolCallAmbiguityPolicy : std::uint8_t {
    PayloadFidelity,  // current Round-4 behavior
    FailClosed,       // do not execute an ambiguous Stage-2 reinterpretation
};
```

The exact storage location may be `ToolCallParsePolicy` or the output contract.

#### `PayloadFidelity`

- keeps the current Round-4 behavior;
- preserves embedded raw tool markup inside string parameters;
- keeps the existing R1 residual executable;
- documentation must say so explicitly.

#### `FailClosed`

The parser must reject an ambiguous Stage-2 completion when accepting it requires reinterpreting a
parameter/function/wrapper close that Stage 1 had already used to form a complete call and the
remaining bytes later provide an alternative canonical closer chain.

This is the exact ambiguity class exercised by `test_round4_r1_residual_pinned()`.

The failure reason should be:

```text
ambiguous_structure
```

not `malformed_structure`, because both interpretations are structurally plausible.

**Important trade-off:** the same rule can reject legitimate payloads such as a `write.content`
value that contains a complete tool-call example followed by prose. That is why this must be an
explicit policy and why the realistic payload corpus must be run in both modes.

For an autonomous agent with executable `bash`/`write`/`edit` tools, the recommended production
policy is `FailClosed`. If maximum byte-fidelity for generated documentation is preferred, retain
`PayloadFidelity` knowingly and do not claim phantom-call elimination.

### 2.6 Constrained decoding remains a separate future layer

The CPU grammar-state core may remain, but Round 5 does not connect it to sampling. The runtime
must continue to reject:

```text
--constrained-tool-decoding=tool-calls-only
```

with the existing explicit startup error.

Documentation must describe three separate facts accurately:

1. grammar-state CPU core exists;
2. parser uses that grammar after generation;
3. runtime sampling constraint is not enabled in the reviewed build.

---

## 3. Findings and required fixes

### R5-01 — HIGH — Terminal `FinishReason` is lost before parser terminalization

**Status:** Verified by post-merge source comparison.

**Affected files:**

```text
src/runtime/engine/engine_core.h
src/models/qwen3_5/frontend/output_session.h
src/models/qwen3_5/frontend/output_session.cpp
```

#### Evidence

At the reviewed HEAD, `OutputSession::commit_preview` terminalizes the tool-call decoder with its
argument:

```cpp
PublishedOutput OutputSession::commit_preview(FinishReason finish_reason) {
    ...
    if (impl_->state.terminal) {
        fi::ToolCallOutputDecoder::Terminal terminal =
            impl_->tool_call_output.finish(finish_reason);
        ...
    }
}
```

The merged `EngineCore`, however, uses the upstream reasonless call pattern.

Cancellation path:

```cpp
(void)request->output.preview_terminal(FinishReason::Cancelled);
append_output(request, request->output.commit_preview());
complete_success(request, FinishReason::Cancelled);
```

Normal row commit:

```cpp
finish_reasons[row] = decision.finish_reason;
...
auto published = request->output.commit_preview();
...
request->terminal_reason = finish_reasons[row];
```

The real reason therefore exists, but it reaches the request only **after** the parser has already
finished using the default `None`.

#### Why this changes behavior

`ToolCallStreamParser::finish()` treats:

```cpp
finish_reason == StopToken || finish_reason == None
```

as a natural stop. `StopString`, `OutputLimit`, `ContextCapacity` and `Cancelled` are cuts.

The reason affects both:

- whether an open-value Stage-1 attempt may continue retrying later markers; and
- whether tolerant Stage-3 recovery may commit parameterized completed calls before a broken tail.

Replacing a known `OutputLimit`/`Cancelled` with `None` is therefore a semantic parser input change,
not a logging discrepancy.

#### Downstream impact

The API layer later receives the correct `GenerationOutcome.finish_reason`, but by then the parser
may already have populated `outcome.tool_calls` under the wrong natural-stop policy.

In the Responses API path, a generated tool call is serialized as a `function_call` item with:

```json
"status": "completed"
```

while the response itself may be `"incomplete"` because of `OutputLimit`/`ContextCapacity`.

The correct defense is to ensure a cut-reason parser never emits that call in the first place.

#### Required fix

Implement §2.1. Do **not** stop at a call-site patch.

- Move terminal-reason ownership into the `OutputSession` preview transaction.
- Restore `commit_preview()` to a reasonless public commit operation.
- Store and validate the terminal reason produced by the preview.
- Feed exactly that stored reason to `ToolCallOutputDecoder::finish()`.
- Clear the stored metadata after commit/rollback.

#### Required tests

Section 7 items 1–4.

#### Acceptance

For every terminal reason in §2.2, the OutputSession/engine integration must produce the same
parser decision as a direct one-shot parser call with that reason.

---

### R5-02 — MEDIUM–HIGH — The public API and tests allow the same bug to compile silently

**Status:** Verified.

The reviewed header accepts:

```cpp
commit_preview(FinishReason finish_reason = FinishReason::None)
```

This makes an omitted argument semantically meaningful instead of a compile failure. The merged
upstream architecture therefore compiled without any signal that the parser-side contract had
changed.

The frontend tests also use the same pattern after terminal previews, for example conceptually:

```cpp
const auto decision = session.preview_model(..., FinishReason::OutputLimit);
const auto output = session.commit_preview();
```

and:

```cpp
session.preview_terminal(FinishReason::Cancelled);
session.commit_preview();
```

Those tests verify output mechanics but not the terminal-reason contract.

#### Required fix

R5-02 is closed by the architecture in §2.1 plus tests that make the reason observable through a
parser outcome.

Do not keep either of these designs:

```cpp
commit_preview(FinishReason = FinishReason::None);   // silent omission
```

or:

```cpp
commit_preview(finish_reasons[row]);                 // duplicated caller responsibility only
```

The latter may be used as a temporary diagnostic patch, but it is not the Round-5 final design.

#### Additional invariant tests

Add assertions for invalid internal states:

- terminal committed preview without a stored terminal reason -> invariant failure;
- non-terminal preview must not carry a terminal reason;
- a new preview replaces the previous preview metadata;
- cancellation via `preview_terminal(Cancelled)` commits with `Cancelled` without any engine-side
  reason parameter.

---

### R5-03 — MEDIUM — Request-log serialization lost Round-4 diagnostics

**Status:** Verified by comparing `e76f7466...` with `705c45e3...`.

`ToolCallParseDiagnostics` still contains the Round-4 fields, but the final
`src/serve/request_log.cpp` no longer serializes:

```text
markup_tolerant_completion
fenced_markers_suppressed
ended_in_unclosed_fence
parse_budget_exhausted
```

The pre-merge implementation serialized all four.

#### Required fix

Restore the tool-call JSON object to include, in a stable order:

```cpp
Json{{"marker_seen", diagnostics.marker_seen},
     {"structured_call_count", diagnostics.structured_call_count},
     {"empty_arguments_omitted", diagnostics.empty_arguments_omitted},
     {"schema_mismatch_arguments", diagnostics.schema_mismatch_arguments},
     {"duplicate_parameters_repaired", diagnostics.duplicate_parameters_repaired},
     {"markup_tolerant_completion", diagnostics.markup_tolerant_completion},
     {"fenced_markers_suppressed", diagnostics.fenced_markers_suppressed},
     {"ended_in_unclosed_fence", diagnostics.ended_in_unclosed_fence},
     {"parse_budget_exhausted", diagnostics.parse_budget_exhausted},
     {"fallback_reason",
      ninfer::tool_call_parse_fallback_reason_name(diagnostics.fallback_reason)}};
```

Port this block into the **current** request-log architecture. Do not replace unrelated upstream
logging changes.

#### Required tests

Restore/extend `tests/test_request_log.cpp` so the JSON values are asserted, not just field
presence.

Test at least:

1. clean structured call: all four fields false/0;
2. Stage-2 accepted markup: `markup_tolerant_completion == true`;
3. unclosed fence text fallback: nonzero `fenced_markers_suppressed` and
   `ended_in_unclosed_fence == true`;
4. forced tiny Stage-2 budget: `parse_budget_exhausted == true`.

---

### R5-04 — MEDIUM — Operational-log diagnostics were rolled back

**Status:** Verified by pre-/post-merge source comparison.

The Round-4 implementation contained:

- a separate warning for a no-latch unclosed fence that suppressed markers;
- fence/budget suffixes on normal tool-markup fallback warnings;
- snippets starting from any supported top-level marker family, not only `<tool_call>`.

The reviewed final branch lost those additions.

#### Required fix

Port the Round-4 behavior onto the current `operational_log.cpp`.

Required precedence:

1. **No marker latched, no calls, unclosed fence, suppressed markers > 0**

   ```text
   req#<id> tool-call fence left unclosed | fenced_markers_suppressed=<N>[ | <snippet>]
   ```

2. **Marker seen + `TruncatedTail` + structured calls retained**

   Keep the existing informational record:

   ```text
   req#<id> tolerated tool-call suffix discarded | truncated tail
   ```

3. **Marker seen + fallback reason != none**

   ```text
   req#<id> tool markup returned as text | <reason>
       [ | fenced_markers_suppressed=<N>]
       [ | parse budget exhausted]
       [ | <snippet>]
   ```

   The fence suffix is included only when `N > 0 && ended_in_unclosed_fence`.

4. Snippet marker search covers at least the compatibility families that the active syntax policy
   permits. In Compatibility mode that includes:

   ```text
   <tool_call>
   <function_calls>
   <function=
   <invoke 
   ```

   In `QwenWrappedNative`, the executable marker surface is only `<tool_call>`, but diagnostics may
   still show the first compatibility-looking bytes when they are returned as text. This choice
   should be explicit in the test.

#### Required tests

Restore the Round-4 operational-log assertions and add one post-merge regression test proving that
`parse_budget_exhausted` appears before the snippet.

---

### R5-05 — LOW–MEDIUM — Help and documentation again describe obsolete behavior

**Status:** Verified.

The post-merge `--tolerant-tool-calls` help text again describes behavior that no longer matches the
parser. In particular, it can imply that output-budget-cut calls or undeclared tool names are kept.
The current parser enforces declared tool identity independently of syntax tolerance.

`docs/serving.md` and related parser documentation also contain stale descriptions of tolerant
recovery, duplicate parameters and diagnostics.

#### Required help text

Use semantics equivalent to:

```text
  --tolerant-tool-calls      keep function-closed Qwen calls before a cut-off or
                             malformed tail when the recovery policy proves them
                             independent of the missing bytes; never an open value
                             or an undeclared tool name (strict by default)
```

If R5-06 introduces an ambiguity policy, add a separate option/help block; do not overload
`tolerant-tool-calls` with payload-ambiguity semantics.

#### Required `serving.md` corrections

Document:

- Stage 2 runs in strict and tolerant mode.
- Tolerant mode repairs syntax/recovery; it does not authorize undeclared tools.
- `StopString`, `OutputLimit`, `ContextCapacity` and `Cancelled` are cuts.
- `ambiguous_structure` is a real fallback reason.
- declared duplicate/synthetic parameter cases follow the Round-3 ambiguity rules.
- the four Round-4 diagnostics from R5-03 are present in request logs.
- the ambiguity policy from R5-06 and its trade-off.
- constrained tool decoding is fail-fast/not active in the reviewed runtime build.

#### Required `tool_call_parser.md` corrections

Ensure the document distinguishes:

```text
Stage 1  greedy canonical parse
Stage 2  consistent value-boundary resolution (both modes)
Stage 3  tolerant-only recovery
```

and explicitly records the native-vs-compatibility syntax policy introduced by R5-07.

#### Required progress-file note

The Round-5 progress file must record that the Round-4 help/log behavior existed before the
architecture merge and was lost by merge resolution. Do not rewrite Round-4 history to make the
post-merge state look intentional.

---

### R5-06 — HIGH integrity residual — strict mode still pins a phantom executable call

**Status:** Verified by current committed test expectations.

Current test:

```text
tests/test_tool_call_parser.cpp
  test_round4_r1_residual_pinned()
```

constructs conceptually:

```text
Example:
<tool_call>
<function=bash>
<parameter=command>
ls
</parameter>
</function>
</tool_call>
Then close with
</parameter>
</function>
</tool_call>
```

and expects strict `StopToken` to produce one structured `bash` call whose `command` contains the
first closer chain plus the prose that follows it.

The test comment explicitly calls this a **phantom acceptance** and says the behavior is pinned.

#### Root cause

The raw Qwen wire protocol has no escaping/length framing for string parameter values. Stage 2
therefore has to choose between two byte-identical interpretations:

```text
A. the early </parameter> is structural and the later bytes are prose
B. the early closer chain is literal payload and a later closer is structural
```

Case B is necessary for byte-exact `write`/`edit` payloads containing tool-call examples. Case A is
necessary to avoid executing an example that appeared in prose.

A byte parser cannot prove intent when the bytes are identical.

#### Required Round-5 action

Implement the explicit ambiguity policy from §2.5.

At minimum:

- preserve the existing fixture under `PayloadFidelity` and rename the assertion message so it does
  not imply the parser is globally safe from phantom calls;
- add the same fixture under `FailClosed` and require:

  ```text
  is_tool_call_response == false
  tool_calls.empty()
  content == original input
  fallback_reason == ambiguous_structure
  ```

- add a realistic `write.content` fixture containing a complete nested tool-call example followed
  by prose:
  - `PayloadFidelity` must preserve it byte-exact and return the outer `write` call;
  - `FailClosed` may return the whole region as text, but must not execute an alternative call.

#### Non-goal

Do not claim `FailClosed` can distinguish user intent. It deliberately chooses non-execution when
both interpretations are structurally plausible.

#### Acceptance wording

After this change, the project may claim:

> “The `FailClosed` ambiguity policy prevents the known R1 phantom-call class.”

It may **not** claim:

> “The byte protocol is unambiguous” or “all phantom calls are impossible.”

---

### R5-07 — MEDIUM hardening — Native Qwen3.8 parser accepts compatibility-only top-level syntax

**Status:** Verified by the current shared top-level entry classifier and by the selected Qwen3.8
chat-template format.

The strict parser accepts top-level entries beyond the wrapped Qwen3.8 form, including legacy /
compatibility forms such as bare `<function=...>`, `<invoke...>` and `<function_calls>`.

This behavior is useful for compatibility but increases the number of byte sequences that can latch
as executable structure.

#### Required fix

Implement §2.4.

The parser entry functions must receive the syntax mode. In `QwenWrappedNative`:

- the pre-latch marker classifier only latches `<tool_call>`;
- retry search only finds `<tool_call>`;
- top-level transitions after a closed wrapper accept another `<tool_call>` or end/prose according
  to existing rules;
- `<function_calls>`, bare `<function>` and `<invoke>` are ordinary text at top level.

Inside the wrapper, the canonical `<function=...>` and `<parameter=...>` syntax remains unchanged.

`Compatibility` preserves current behavior.

#### Tests

For each compatibility-only entry:

```text
<function=read>...</function>
<invoke=read>...</invoke>
<function_calls>...</function_calls>
```

assert:

| Mode | Required result |
|---|---|
| `QwenWrappedNative` | text, no structured calls |
| `Compatibility` | current behavior unchanged |

Also verify that a canonical wrapped Qwen3.8 call is identical in both modes.

---

### R5-08 — LOW–MEDIUM — Constrained-decoding documentation can overstate runtime protection

**Status:** Verified by current runtime validation.

`model_instance.cpp` currently rejects every non-off constrained-tool-decoding mode with an explicit
startup exception. That is the correct behavior for an unimplemented feature.

The design documentation, however, contains language about sampling integration and GPU behavior
that can be read as if the mask path is a usable runtime feature.

#### Required fix

Do not implement GPU sampling in Round 5. Instead align docs and help with the executable source:

```text
CPU grammar-state core: implemented/testable
parser use of grammar: implemented
runtime sampling constraint: unavailable in this build
non-off flag: startup error by design
```

Keep the fail-fast validation test. Add/retain a docs note that malformed native tool syntax can
still be generated and that the parser is a post-generation integrity boundary, not a generator
constraint.

---

## 4. Merge-specific file plan

This section is intentionally file-oriented. The Round-5 bug class is largely an integration merge
problem; the implementer must not re-edit the parser core unnecessarily.

### 4.1 `src/models/qwen3_5/frontend/output_session.h`

Required:

- restore `commit_preview()` to a reasonless public API;
- add no defaulted semantic argument;
- no public setter for the terminal reason;
- keep the preview/commit transaction encapsulated.

Recommended private structure is implementation-only in `.cpp`/`Impl`.

### 4.2 `src/models/qwen3_5/frontend/output_session.cpp`

Required:

1. Add preview terminal metadata to `Impl`.
2. In every preview-producing method, set/reset it in the same transaction that sets
   `preview_state`/`preview_ready`.
3. In `commit_preview()`:
   - swap/commit the existing preview state as today;
   - if terminal, require a stored non-`None` reason;
   - call `tool_call_output.finish(stored_reason)`;
   - move tool calls/diagnostics exactly as today;
   - clear preview metadata.
4. If the class has exception/rollback paths, the metadata must follow the same rollback boundary.

Do not change parser semantics in this file.

### 4.3 `src/runtime/engine/engine_core.h`

With §2.1 implemented, no parser-specific finish-reason argument should be necessary at commit
calls. The important review task is therefore to prove all terminal paths follow:

```text
preview producing a terminal decision
→ commit that same preview
→ complete request with the same decision reason
```

Audit at least:

- pre-start cancellation;
- active cancellation;
- ordinary terminal model row;
- terminal control row, if applicable;
- context/output budget termination;
- stop token/string terminalization;
- any salvage/abort path that commits a terminal preview.

Add comments only where the transaction ownership is not obvious. Do not duplicate the finish
reason in two independent stores just for the parser.

### 4.4 `src/serve/request_log.cpp`

Port only the missing fields from R5-03.

### 4.5 `src/serve/operational_log.cpp`

Port the Round-4 tool-call diagnostic branches from R5-04 while preserving new architecture logging
around them.

### 4.6 `src/serve/serve_options.cpp`

Correct tolerant-mode wording. If R5-06/R5-07 add options, keep each option single-purpose:

```text
--tolerant-tool-calls
--tool-call-syntax <qwen-wrapped|compat>
--tool-call-ambiguity <fail-closed|payload-fidelity>
```

Names may be adapted to project conventions, but do not combine syntax, tolerance and ambiguity in
one flag.

### 4.7 `src/models/qwen3_5/frontend/tool_call_stream.{h,cpp}`

Do **not** refactor Stage 2 again as part of R5-01–05.

Changes here are limited to R5-06/R5-07 policy plumbing and verdict gates. Existing Round-4 work
bounds, fence logic, retry scope and candidate-walk code are frozen unless a new failing test proves
otherwise.

### 4.8 `src/models/qwen3_5/frontend/tool_call_parser.cpp`

Carry syntax/ambiguity policy from the frontend contract to the stream parser. Preserve declared
identity/schema normalization.

### 4.9 Tests

Files expected to change:

```text
tests/models/qwen3_5/test_frontend.cpp
tests/test_tool_call_parser.cpp
tests/test_request_log.cpp
tests/test_serve_options.cpp
```

If there is a CPU-only engine harness suitable for terminal commit semantics, add the smallest
possible engine test there. Do not introduce a GPU requirement merely to test metadata propagation.

---

## 5. Implementation order and regression guard

### 5.1 Mandatory pre-change baseline

Before editing Round 5:

1. Record current HEAD SHA.
2. Build the target CPU suites listed in §8.1.
3. Run them once against the current post-merge tree.
4. Record all failures/skips in the Round-5 progress file. Do not silently attribute failures to
   Round 5 before reproducing them on the baseline.
5. Generate a parser outcome dump using the existing Round-4 Appendix-B tool or an equivalent
   current copy.

The parser core is expected to remain behavior-identical through R5-01–05. The outcome dump is the
refactor guard.

### 5.2 Commit sequence

One logical commit per step; Conventional Commits.

| Step | Suggested commit | Content | Allowed parser outcome-dump change |
|---|---|---|---|
| 0 | no commit | baseline CPU suites + outcome dump | — |
| 1 | `fix(frontend): bind terminal reason to output preview transaction` | R5-01/R5-02 OutputSession architecture + focused frontend tests | **none** for direct parser outcome dump |
| 2 | `test(frontend): cover finish-reason-sensitive tool parsing` | end-to-end OutputSession reason matrix, cancellation/cut regressions | none |
| 3 | `fix(serve): restore tool-call parser diagnostics after architecture merge` | R5-03/R5-04 request + operational logs | none |
| 4 | `test(serve): restore tool-call diagnostics assertions` | request/operational/help regression tests | none |
| 5 | `docs(serve): correct tolerant and constrained tool-call semantics` | R5-05/R5-08 docs/help | none |
| 6 | `feat(frontend): separate native and compatibility tool-call syntax` | R5-07 policy + tests | changes only in compatibility-only entry fixtures when native mode is selected |
| 7 | `feat(frontend): make ambiguous tool payload policy explicit` | R5-06 policy + tests | changes only for fixtures classified by the new ambiguity policy |
| 8 | `docs(frontend): document Round-5 ambiguity and native policies` | parser docs + progress completion | none |

If R5-06/R5-07 are intentionally deferred, the progress file must say so and the release status
must remain **not fully green**. R5-01–05 may still be merged independently.

### 5.3 Parser outcome dump

Reuse the Round-4 outcome-signature concept. It must include at least:

```text
is_tool_call_response
content hash
call count + call names + argument hashes
marker_seen
structured_call_count
empty_arguments_omitted
schema_mismatch_arguments
duplicate_parameters_repaired
markup_tolerant_completion
fenced_markers_suppressed
ended_in_unclosed_fence
parse_budget_exhausted
fallback_reason
streaming equality flag
```

Run the corpus for:

```text
strict + tolerant
StopToken + OutputLimit
PayloadFidelity + FailClosed (after R5-06)
QwenWrappedNative + Compatibility (after R5-07)
```

Before R5-06/R5-07, steps 1–5 must be byte-identical to the baseline dump.

---

## 6. Existing assertions that intentionally change

| Test / behavior | Current expectation | Round-5 expectation | Finding |
|---|---|---|---|
| Terminal `OutputSession` preview committed with `commit_preview()` | parser receives default `None` | parser receives the preview's actual terminal reason | R5-01/R5-02 |
| `test_structured_tool_output`-style terminal test | may still return a call under `OutputLimit` when bytes are fully complete | unchanged for fully complete, unambiguous calls | guard |
| Finish-reason-sensitive ambiguous fixture under `OutputLimit` | can diverge from direct parser because reason becomes `None` | must match direct parser `OutputLimit` result | R5-01 |
| Cancellation terminal preview | reasonless parser finish | parser sees `Cancelled` | R5-01 |
| request log JSON | four Round-4 fields absent | fields restored with exact values | R5-03 |
| operational fallback line | no fence/budget suffixes in merged code | Round-4 precedence/suffixes restored | R5-04 |
| tolerant help text | stale merge version | corrected semantics | R5-05 |
| compatibility-only top-level entry in native mode | executable/parsable | text, no call | R5-07 |
| R1 phantom fixture, `PayloadFidelity` | one `bash` call | unchanged | R5-06 |
| R1 phantom fixture, `FailClosed` | no such mode | text, no call, `ambiguous_structure` | R5-06 |

No other existing assertion may change without being listed in the Round-5 progress file with a
finding ID and explanation.

---

## 7. Required new and restored tests

### 7.1 R5-01 — OutputLimit must reach the parser

Add a frontend/OutputSession regression using a fixture whose parser result differs between
`StopToken`/`None` and `OutputLimit`.

Prefer an existing Round-3/4 adversarial fixture rather than inventing a weaker one. Suitable
families include the existing P-A or S8-cut cases.

Test shape:

```cpp
Frontend frontend = ... tolerant tool calls enabled ...;
auto session = ...;
const std::vector<TokenId> tokens = fixture_tokenizer().encode(text);

const auto decision = session.preview_model(
    tokens,
    static_cast<std::uint32_t>(tokens.size()),
    FinishReason::OutputLimit);

CHECK(decision.finished());
CHECK(decision.finish_reason == FinishReason::OutputLimit);

(void)session.commit_preview();
const auto calls = session.take_tool_calls();
```

Required result must match:

```cpp
parse_qwen_tool_call_output(text, ..., tolerant=true, FinishReason::OutputLimit)
```

exactly for calls/content/diagnostics.

The regression must fail against the reviewed merge state because that state terminalizes with
`None`.

### 7.2 R5-01 — Cancelled must reach the parser

Exercise:

```cpp
preview_terminal(FinishReason::Cancelled)
commit_preview()
```

on a session that already buffered an ambiguous/partial tool region.

Required:

```text
OutputSession result == direct parser finish(Cancelled)
```

and no call may be recovered merely because cancellation was replaced by `None`.

### 7.3 R5-01 — StopString and ContextCapacity

Use either direct OutputSession terminal paths or the lightest existing frontend harness to cover:

```text
StopString
ContextCapacity
```

The purpose is not to retest the parser's recovery matrix; parser unit tests already do that. The
purpose is to prove **transaction metadata propagation**.

### 7.4 R5-02 — Preview metadata invariants

Add focused tests for:

1. non-terminal preview -> `commit_preview()` does not terminalize tool parser;
2. terminal preview -> stored finish reason is used;
3. second preview replaces first preview metadata;
4. `preview_terminal(Cancelled)` records `Cancelled`;
5. committing a synthetic terminal preview with missing internal reason triggers the project
   invariant mechanism (only if a test hook can reach this state without undefined behavior).

### 7.5 R5-03 — Request-log diagnostics restoration

Restore a request-log test whose input diagnostics are:

```cpp
ToolCallParseDiagnostics{
    .marker_seen = true,
    .structured_call_count = 2,
    .empty_arguments_omitted = 1,
    .schema_mismatch_arguments = 3,
    .duplicate_parameters_repaired = 4,
    .markup_tolerant_completion = true,
    .fenced_markers_suppressed = 5,
    .ended_in_unclosed_fence = true,
    .parse_budget_exhausted = true,
    .fallback_reason = ToolCallParseFallbackReason::TrailingContent,
}
```

Assert the exact JSON values for every field.

### 7.6 R5-04 — Operational-log precedence

Restore the Round-4 cases:

1. `!marker_seen`, no calls, unclosed fence, suppressed=2 -> `tool-call fence left unclosed`.
2. marker seen, `TrailingContent`, unclosed fence, suppressed=2 -> normal fallback line with fence
   suffix, **not** the standalone fence warning.
3. same plus `parse_budget_exhausted=true` -> budget suffix after fence suffix and before snippet.
4. bare compatibility marker returned as text -> snippet starts at the first compatibility marker
   in Compatibility mode.

### 7.7 R5-05 — Serve help text

If `tests/test_serve_options.cpp` snapshots help text, update the expected text. Otherwise add a
small assertion that the help contains:

```text
never an open value
undeclared tool/name is never returned as a call
```

and does not claim that `OutputLimit` by itself makes a parameterized malformed-tail call
recoverable.

### 7.8 R5-06 — Ambiguity-policy matrix

Run at least these fixtures in both ambiguity policies:

#### A. R1 phantom fixture

```cpp
const std::string text =
    "Example:\n" + tool_call("bash", {{"command", "ls"}}) +
    "\nThen close with\n</parameter>\n</function>\n</tool_call>";
```

Expected:

| Policy | Strict StopToken |
|---|---|
| `PayloadFidelity` | existing one-call result, explicitly documented residual |
| `FailClosed` | text; 0 calls; `ambiguous_structure` |

#### B. Legitimate outer write containing an embedded complete call

```cpp
const std::string embedded = tool_call("read", {{"path", "foo.cpp"}});
const std::string text = tool_call(
    "write",
    {{"path", "docs/x.md"},
     {"content", "# Example\n" + embedded + "\nDone."}});
```

Expected:

| Policy | Result |
|---|---|
| `PayloadFidelity` | 1 `write`, content byte-exact |
| `FailClosed` | either the same proven-safe `write` if the implementation can prove it, or text/0 calls; **never** an alternative inner call |

The progress file must record which result the implementation chooses and why.

### 7.9 R5-07 — Native/compat syntax matrix

For each fixture, run one-shot and streamed:

```text
canonical wrapped <tool_call> ...
bare <function=read> ...
bare <invoke=read> ...
<function_calls> ...
```

Expected:

| Fixture | QwenWrappedNative | Compatibility |
|---|---|---|
| wrapped native | structured call | same structured call |
| bare function | text | current call behavior |
| bare invoke | text | current behavior |
| function_calls wrapper | text | current behavior |

Every-split streaming equality for fixtures < 512 bytes.

### 7.10 R5-08 — constrained decoding remains fail-fast

Keep/add a CPU-only engine-options validation test:

```text
Off            -> accepted
ToolCallsOnly  -> invalid_argument before GPU work
```

The error string must clearly say the mode is not implemented in this build.

---

## 8. Commands — CPU only

### 8.1 Target suites

Use the current CMake target names; at minimum:

```powershell
cmake --build build-new-parser --config Release --parallel 16 `
  --target ninfer_tool_call_parser_test `
           ninfer_tool_call_grammar_test `
           ninfer_tool_call_grammar_state_test `
           ninfer_qwen3_5_frontend_test `
           ninfer_request_log_test `
           ninfer_pretty_logging_test `
           ninfer_serve_options_test `
           ninfer_engine_options_validation_test

ctest --test-dir build-new-parser -C Release --parallel 16 --output-on-failure `
  -R "^(ninfer_tool_call_parser_test|ninfer_tool_call_grammar_test|ninfer_tool_call_grammar_state_test|ninfer_qwen3_5_frontend_test|ninfer_request_log_test|ninfer_pretty_logging_test|ninfer_serve_options_test|ninfer_engine_options_validation_test)$"
```

If Round 5 adds a new CPU-only OutputSession/engine integration target, append it explicitly after
checking its CMake/source for GPU use.

### 8.2 Full CPU gate

```powershell
$env:CUDA_VISIBLE_DEVICES = "99"
ctest --test-dir build-new-parser -C Release -E "_real" --parallel 16 --output-on-failure
```

The purpose of `CUDA_VISIBLE_DEVICES=99` is defensive. It does not turn a GPU test into a CPU test.
Any target that still requires CUDA runtime execution is outside this round and must be reported as
skipped/not run rather than treated as validation evidence.

### 8.3 Parser outcome dump

Use the Round-4 Appendix-B utility or copy it into:

```text
build-new-parser/round5/outcome_dump.cpp
```

Add the two Round-5 policy dimensions only after their implementation:

```text
syntax=qwen-native|compat
ambiguity=payload|fail-closed
```

Save:

```text
outcomes_step0.txt
outcomes_step1.txt
...
outcomes_step8.txt
```

For steps 1–5:

```powershell
fc /b build-new-parser\round5\outcomes_step0.txt build-new-parser\round5\outcomes_step5.txt
```

must report no parser-outcome difference.

### 8.4 Optional AddressSanitizer

Only if the local MSVC ASan runtime works in this environment. If it fails at process start as in
previous rounds, record that limitation; do not substitute a GPU run.

---

## 9. Refactor and merge guards

### 9.1 Guard against another terminal-reason merge regression

This is a specific Round-5 acceptance item.

After implementing §2.1, search the tree for:

```text
commit_preview(
```

There must be **no** overload/default parameter that accepts a caller-supplied `FinishReason` for
normal OutputSession commits.

Search for:

```text
tool_call_output.finish(
```

and prove the finish reason originates from OutputSession preview transaction metadata.

The Round-5 progress file records the call sites and the source of the stored reason.

### 9.2 Guard against lost diagnostics

Search for every field in `ToolCallParseDiagnostics` and record which sinks consume it.

At minimum these four must have both producer and request-log coverage:

```text
markup_tolerant_completion
fenced_markers_suppressed
ended_in_unclosed_fence
parse_budget_exhausted
```

A future merge that removes a sink should fail tests.

### 9.3 Guard against native/compat drift

The executable marker set must be centralized by syntax policy. Do not scatter ad-hoc conditions
across:

```text
feed pre-latch classifier
retry marker search
Top state
function-close continuation classifier
grammar-state trigger
```

Add one shared helper/entry classifier that receives the syntax mode. Tests must cover all entry
points so the parser and grammar-state core cannot disagree about which marker opens a native tool
region.

### 9.4 Guard against ambiguity-policy drift

The policy applies only at the point where Stage 2 would accept an interpretation that is
ambiguous under the R5-06 rule. It must not:

- weaken declared-tool checks;
- change wrapper ownership;
- change Stage-2 work budget;
- make open values executable;
- affect clean Stage-1 calls.

A clean canonical wrapped call must be identical under both ambiguity policies.

---

## 10. Detailed acceptance gate

Round 5 is complete only when every applicable item is recorded in
`docs/NInfer_new_parser_design_round5_bugfix_progress.md`.

### 10.1 R5-01/R5-02 terminal transaction

- [ ] `commit_preview()` no longer accepts/defaults a caller-supplied terminal reason.
- [ ] `OutputSession` stores terminal metadata with the preview.
- [ ] `preview_model` records `StopToken`, `StopString`, `OutputLimit` or `ContextCapacity` exactly
      when returned terminal.
- [ ] `preview_terminal(Cancelled)` stores `Cancelled`.
- [ ] terminal commit feeds exactly the stored reason to `ToolCallOutputDecoder::finish()`.
- [ ] known engine terminal commits can never use `FinishReason::None` silently.
- [ ] non-terminal commits do not terminalize the tool parser.
- [ ] finish-reason-sensitive OutputSession tests fail on `705c45e3...` and pass after the fix.
- [ ] the direct parser and OutputSession integration agree for all tested reasons.

### 10.2 R5-03/R5-04 observability

- [ ] all four Round-4 fields are restored to request-log JSON.
- [ ] exact values are tested.
- [ ] no-latch unclosed-fence operational warning is restored.
- [ ] normal fallback warning carries fence/budget suffixes in the specified order.
- [ ] compatibility-marker snippets are handled deliberately.
- [ ] no unrelated new-architecture logging is lost.

### 10.3 R5-05/R5-08 truthfulness

- [ ] tolerant help text matches executable recovery policy.
- [ ] docs say undeclared tools never become structured calls.
- [ ] docs distinguish natural stops from cuts.
- [ ] docs include `ambiguous_structure` and all diagnostic fields.
- [ ] docs say constrained runtime sampling is unavailable/fail-fast.
- [ ] no document claims GPU runtime validation was performed in this round.

### 10.4 R5-06 ambiguity policy

- [ ] policy is explicit in code, tests and docs.
- [ ] `PayloadFidelity` preserves the existing R1 behavior and realistic embedded-markup payloads.
- [ ] `FailClosed` rejects the known R1 phantom fixture as `ambiguous_structure`.
- [ ] clean canonical calls are identical in both policies.
- [ ] no test/message claims the protocol itself has become unambiguous.

If R5-06 is intentionally deferred:

- [ ] progress file marks it **OPEN**;
- [ ] final status is not “fully green”;
- [ ] docs retain the residual warning.

### 10.5 R5-07 native syntax

- [ ] QwenWrappedNative latches only the canonical `<tool_call>` wrapper at top level.
- [ ] Compatibility preserves current legacy forms.
- [ ] canonical wrapped call is identical in both modes.
- [ ] streaming equals one-shot in both modes.
- [ ] grammar-state trigger uses the same syntax policy as the parser.

If R5-07 is deferred for compatibility reasons, document exactly which current clients require the
wider grammar.

### 10.6 Regression gates

- [ ] parser outcome dump identical through steps 1–5.
- [ ] every intentional step-6/7 difference reviewed and listed.
- [ ] targeted CPU suites all green.
- [ ] full `ctest -E "_real"` result recorded, including unrelated/skipped failures.
- [ ] no GPU/model inference/benchmark executed.
- [ ] Round-4 realistic `write`/`edit` payload corpus rerun in `PayloadFidelity` mode.
- [ ] Round-4 streaming fuzz rerun with exact diagnostics equality.
- [ ] final HEAD SHA recorded.

---

## 11. Progress-file template

Create:

```text
docs/NInfer_new_parser_design_round5_bugfix_progress.md
```

with at least this structure:

```markdown
# NInfer `new_parser_design` — Round 5 Bugfix Progress

## Baseline
- start SHA:
- Round-5 spec SHA/path:
- upstream architecture parent:
- build directory:
- toolchain:
- GPU tests: NOT RUN

## Finding status
| ID | Status | Commit | Tests | Notes |
|---|---|---|---|---|
| R5-01 | OPEN | | | |
...

## Step results
### Step 0 baseline
- targeted suites:
- full CPU gate:
- outcome dump hash / line count:

### Step 1 terminal transaction
- code changes:
- focused tests:
- outcome-dump delta:

...

## Intentional outcome changes
- syntax-policy deltas:
- ambiguity-policy deltas:

## Unresolved residuals
- protocol ambiguity:
- constrained decoding runtime:
- any unrelated CPU failures:

## Final sign-off
- final SHA:
- targeted suites:
- full CPU gate:
- GPU/model tests: NOT RUN
- remaining HIGH/MEDIUM findings:
```

Every “fixed” row must cite a concrete commit and test. “Documented” is not equivalent to “fixed”.

---

## 12. Review checklist after implementation

Perform at least four independent review passes after the code changes.

### Pass A — transaction semantics

Trace every terminal path from:

```text
model/control/cancellation decision
→ OutputSession preview
→ commit
→ ToolCallOutputDecoder::finish
→ GeneratedToolCall
→ GenerationOutcome
```

For each edge, identify the exact `FinishReason` source. Reject the implementation if any known
terminal path can reach parser finish as `None`.

### Pass B — parser invariants

Re-read complete affected parser files, not just the diff. Re-check:

- wrapper ownership;
- open-value non-execution;
- function-close boundary;
- declared identity;
- Stage-2 budget;
- fence suppression;
- syntax-mode consistency;
- ambiguity-policy gate.

### Pass C — serve surfaces

Trace parser diagnostics into:

```text
request log
operational log
OpenAI Chat
OpenAI Responses
Anthropic Messages
```

The protocol serializers do not need to expose every internal diagnostic, but no structured call
may be produced under a parser decision that would have been rejected with the correct terminal
reason.

### Pass D — adversarial counterexamples

Actively try to falsify the implementation with:

- a complete example in prose;
- nested tool markup in `write.content`;
- here-docs containing canonical closer lines;
- partial parameter/function/wrapper closes;
- `OutputLimit` immediately after a misleading closer chain;
- cancellation between rounds;
- fenced tool examples;
- 300+ failed marker bases before a real call;
- tiny Stage-2 budget;
- undeclared tool names;
- bare compatibility markers in native mode.

Do not accept “tests pass” as sufficient if a new reproducible contradictory case exists.

---

## 13. Final expected state

After a complete Round-5 implementation:

### Verified design properties

- Terminal reason is transaction-owned by `OutputSession`, so architecture merges cannot silently
  downgrade known cuts to `None` by omitting a function argument.
- `StopString`, `OutputLimit`, `ContextCapacity` and `Cancelled` reach the parser unchanged.
- Round-4 diagnostics are again visible in request and operational logs.
- Help/documentation match executable behavior.
- Native Qwen3.8 parsing can be restricted to the wrapped format without deleting compatibility
  support globally.
- The known R1 ambiguity is controlled by an explicit policy instead of being hidden behind the
  word “fixed”.
- Runtime constrained decoding remains honestly fail-fast until it is implemented and GPU-verified.

### What Round 5 still cannot prove

Even with all items green, do **not** claim mathematical absence of every possible tool-call error.
The raw Qwen markup protocol can carry byte sequences that are semantically ambiguous without an
escaping/framing channel, and model generation remains unconstrained in this build.

The strongest justified final wording is:

> **No remaining HIGH/MEDIUM implementation defect was found in the reviewed parser/integration
> scope after the Round-5 regression suite, with the documented byte-protocol ambiguity controlled
> by the selected policy and constrained decoding still unavailable.**

If `PayloadFidelity` remains the production ambiguity policy, weaken that wording further and state
that the R1 phantom-call residual remains accepted by design.

---

## Appendix A — Minimal finish-reason regression matrix

The following table must appear in the progress file with actual observed outcomes after the fix.
Do not pre-fill results from expectation alone.

| Fixture | Mode | FinishReason | Expected structured calls | Purpose |
|---|---|---|---:|---|
| clean canonical wrapped call | strict | StopToken | 1 | control |
| clean canonical wrapped call | strict | OutputLimit after complete bytes | 1 if structurally complete | proves cut does not erase already-complete call |
| P-A/S8-cut adversarial | tolerant | StopToken | current natural-stop result | control |
| same adversarial bytes | tolerant | OutputLimit | 0 | detects `None` substitution |
| same adversarial bytes | tolerant | ContextCapacity | 0 | detects cut propagation |
| partial/ambiguous region | tolerant | Cancelled | 0 | cancellation propagation |
| stop-string terminated ambiguous region | tolerant | StopString | 0 | stop-string propagation |

The exact fixture must be selected from existing Round-3/4 tests so its direct-parser expectations
are already pinned.

---

## Appendix B — Merge-diff audit checklist

For the final branch compare:

```text
pre-Round5 HEAD 705c45e3...
vs.
final Round5 HEAD
```

and separately:

```text
e76f7466... parser-side pre-merge
vs.
final Round5 HEAD
```

For every changed file answer:

1. Is the change required by R5-01 … R5-08?
2. If not, why did it change?
3. Does it overwrite a newer upstream architecture feature?
4. Does it reintroduce a Round-1–4 behavior that was intentionally removed?
5. Is there a test that fails without this change?

Particularly scrutinize:

```text
src/runtime/engine/engine_core.h
src/models/qwen3_5/frontend/output_session.{h,cpp}
src/models/qwen3_5/frontend/tool_call_stream.{h,cpp}
src/models/qwen3_5/frontend/tool_call_parser.cpp
src/serve/request_log.cpp
src/serve/operational_log.cpp
src/serve/serve_options.cpp
tests/models/qwen3_5/test_frontend.cpp
tests/test_tool_call_parser.cpp
tests/test_request_log.cpp
tests/test_serve_options.cpp
```

The preferred Round-5 diff should be small in `engine_core.h`: the stronger fix removes the need for
parser-specific reason plumbing there.

---

## Appendix C — Commit-message suggestions

Suggested sequence:

```text
fix(frontend): bind terminal reason to output preview transaction
test(frontend): cover finish-reason-sensitive tool parsing
fix(serve): restore tool-call diagnostics after architecture merge
test(serve): restore parser diagnostic regression coverage
docs(serve): correct tolerant and constrained tool-call semantics
feat(frontend): separate native and compatibility tool syntax
feat(frontend): add explicit ambiguous tool-payload policy
docs(frontend): document round-5 tool-call guarantees and residuals
```

Commit messages are suggestions; semantic separation is the requirement.

---

## Appendix D — Definition of “green” for this branch

The branch is **not green** merely because:

- the parser unit tests pass;
- the Round-4 progress file says CPU tests passed before the architecture merge;
- the project compiles;
- the API response still reports `finish_reason=length` downstream;
- a residual is documented as intentional.

For Round 5, “green” means:

1. the terminal reason observed by the parser is proven end to end;
2. merge-resilient API design prevents silent reason loss;
3. observability regressions are restored and tested;
4. docs/help match executable behavior;
5. native/compatibility policy is explicit;
6. the R1 ambiguity is either fail-closed under the production policy or explicitly remains open;
7. targeted and full CPU gates are recorded against the **final** SHA;
8. no unexplained parser-outcome delta exists.

If any of those conditions is false, the Round-5 progress file must say **OPEN** rather than
“complete”.
