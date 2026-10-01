# NInfer `new_parser_design` — Round 6 Bugfix / Hardening Implementation Specification

## 0. Purpose

This document is the **Round-6 implementation specification** for the remaining tool-call findings
after the Round-5 fixes on `Hundsbuah/ninfer:new_parser_design`.

It is intentionally narrower than Round 3/4/5 in the parser core itself. The current parser has
already accumulated substantial structural hardening. Round 6 must therefore **not** use the
remaining protocol ambiguity as justification for another broad Stage-2 rewrite.

The goals are:

1. close the remaining **real implementation inconsistency** between the production syntax mode and
   the Phase-4 grammar-constraint core;
2. harden the low-level terminal API so the already-fixed FinishReason regression cannot reappear
   through a different caller;
3. correct the Round-5 sign-off language so it does not overstate what `FailClosed` proves;
4. pin the remaining **byte-identical unfenced quotation** class with a direct regression fixture;
5. optionally add a stronger **agent hardening mode** that dramatically reduces that quotation
   class by requiring a native tool call to be the first non-whitespace content of the assistant
   content channel;
6. define, but do **not** silently implement as part of the mandatory bugfix, the long-term
   TEXT-vs-TOOL intent-channel architecture that is required to eliminate the final semantic
   ambiguity class rather than merely mitigate it.

This specification deliberately distinguishes:

- **implementation defect**;
- **API hardening**;
- **documented protocol ambiguity**;
- **optional product hardening**;
- **future architecture**.

Do not mark a protocol limitation "fixed" merely because it is documented.

---

# 0.1 Pinned baseline

Repository:

```text
Hundsbuah/ninfer
```

Branch:

```text
new_parser_design
```

Round-6 reviewed HEAD:

```text
8b4c4e9904d1347cc73769954273184cb97d287c
```

The code-bearing Round-5 implementation is the parent:

```text
fe3490da3bdb1bcac5609ab07dbf5eaf65b712f5
```

The current HEAD only adds:

```text
docs/NInfer_new_parser_design_round5_bugfix_implementation.md
```

on top of that code state.

Do not implement Round 6 against a moving branch without first recording the exact new start SHA.

---

# 0.2 Current verified product defaults

At the pinned baseline, the product boundary uses:

```text
ToolCallSyntaxMode::QwenWrappedNative
ToolCallAmbiguityPolicy::FailClosed
tolerant_tool_calls = false
constrained_tool_decoding = Off
```

The low-level parser APIs intentionally retain historical compatibility defaults:

```text
ToolCallSyntaxMode::Compatibility
ToolCallAmbiguityPolicy::PayloadFidelity
FinishReason::None for some low-level one-shot APIs
```

Round 6 must not accidentally confuse these two layers.

---

# 0.3 Current verified Qwen3.8 native contract

The current official `Qwen/Qwen3.8-27B` chat template states conceptually:

```text
If you choose to call a function ONLY reply in:

<tool_call>
<function=...>
<parameter=...>
...
</parameter>
</function>
</tool_call>

No suffix.
```

It also explicitly permits optional natural-language reasoning **before** the tool call.

Primary source:

```text
https://huggingface.co/Qwen/Qwen3.8-27B/blame/main/chat_template.jinja
```

This has two consequences:

1. `QwenWrappedNative` is the correct production syntax mode for NInfer's selected template.
2. A policy that forbids visible content before a tool call is **stricter than upstream Qwen's
   template contract** and must be represented as an explicit NInfer hardening mode, not smuggled
   into the parser as if it were canonical Qwen syntax.

Qwen's own function-calling documentation also warns that malformed tool calls can still be emitted
in corner cases and recommends robust parsing in production:

```text
https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md
```

---

# 0.4 Current external corroboration

A recent vLLM Qwen3 parser issue demonstrates that syntactically valid `<tool_call>` markup shown as
content can be promoted to a real tool call, including fenced examples:

```text
https://github.com/vllm-project/vllm/issues/57541
```

NInfer's current fence guard already addresses the fenced sub-class.

This external issue is **corroboration only**. NInfer correctness claims must be proved from NInfer
source and NInfer regression tests.

---

# 0.5 Hard test constraint

As in the previous rounds:

**Do not run GPU model inference, `_real` tests, CUDA correctness/runtime tests, model benchmarks, or
GPU constrained-decoding tests in this task.**

The local AI occupies the GPU.

Allowed:

- CPU compilation/tests;
- grammar/constraint-state CPU tests;
- parser/stream/frontend tests;
- OutputSession tests;
- serve-option parsing tests;
- request/operational-log tests;
- deterministic corpus/outcome dumps;
- CPU property/adversarial tests;
- test-only fake token streams;
- compile-time/API-contract tests.

For `--constrained-tool-decoding`, Round 6 changes the **CPU constraint state's syntax-mode
consistency only**. It does not enable runtime constrained sampling.

---

# 1. Findings

| ID | Finding | Severity | Type | Mandatory Round-6 fix |
|---|---|---:|---|---|
| R6-01 | `ToolCallGrammarConstraint` is hard-wired to `Compatibility` while production parser defaults to `QwenWrappedNative` | **MEDIUM architecture** | implementation inconsistency | **YES** |
| R6-02 | Round-5 progress says “No residual finding” although parser docs explicitly retain general unfenced complete-example ambiguity | LOW/MEDIUM | sign-off / documentation correctness | **YES** |
| R6-03 | `ToolCallOutputDecoder::finish()` still defaults to `FinishReason::None` at the low-level API | LOW | API hardening | **YES** |
| R6-04 | A complete, declared, canonical unfenced tool-call example at EOF is byte-identical to a real call and remains executable | **HIGH integrity residual** | protocol ambiguity | **PIN + MITIGATE, do not falsely claim fully fixed** |
| R6-05 | Tool calls after ordinary visible content remain eligible under canonical Qwen semantics | HIGH-integrity surface | optional product hardening | **RECOMMENDED opt-in mode** |
| R6-06 | No explicit TEXT-vs-TOOL structural intent channel exists | architecture limit | future architecture | **DESIGN ONLY in this round** |
| R6-07 | Runtime constrained decoding remains unavailable | known limitation | non-goal | **KEEP FAIL-FAST** |

The Round-5 implementation findings R5-01 through R5-08 are treated as a regression baseline and
must remain green.

---

# 2. Round-6 non-negotiable invariants

## R6-I1 — One syntax mode means one syntax mode

For one request/session, every component that decides whether bytes start a tool region must use the
same `ToolCallSyntaxMode`:

```text
OutputOptions
  ↓
ToolCallOutputDecoder
  ↓
ToolCallStreamParser pre-latch
  ↓
retry marker scan
  ↓
top-level entry transitions
  ↓
ToolCallGrammarConstraint
```

A native parser plus compatibility constraint trigger is not acceptable even if constrained sampling
is currently disabled.

---

## R6-I2 — Constraint-state changes must not activate GPU/runtime constraints

Round 6 may make the CPU constraint state syntax-aware.

It must **not** remove this current fail-fast behavior:

```text
--constrained-tool-decoding=tool-calls-only
→ startup error
```

until the runtime sampler integration receives its own implementation and GPU verification.

---

## R6-I3 — Unknown terminal reason must be explicit

A terminal parser decision that semantically depends on:

```text
StopToken
StopString
OutputLimit
ContextCapacity
Cancelled
```

must not silently receive `None` because a caller omitted an argument.

`FinishReason::None` may remain a supported *semantic value* for deliberately unknown one-shot input,
but invoking it must be explicit.

---

## R6-I4 — FailClosed is not an intent oracle

`ToolCallAmbiguityPolicy::FailClosed` controls a specific structural ambiguity:

```text
early closer chain already forms a complete call
vs
later closer makes those earlier bytes payload
```

It must not be described as proving that every syntactically complete tool call was intended as an
action.

---

## R6-I5 — A mitigation must be named as a mitigation

If Round 6 adds a rule such as:

```text
tool call must start at first non-whitespace content byte
```

that rule reduces quotation false positives but does **not** make a byte-identical example-only
output distinguishable from an actual call.

Tests/docs must say exactly that.

---

## R6-I6 — Native safety must not delete compatibility support

`QwenWrappedNative` remains the production default.

`Compatibility` remains available for clients/models that intentionally rely on:

```text
<function_calls>
bare <function=...>
bare <invoke=...>
```

Round 6 must not remove compatibility globally.

---

## R6-I7 — No Stage-2 rewrite without a new failing reproducer

Round-4/5 Stage 2 is considered frozen in Round 6.

Do not refactor:

- `RegionIndex`;
- `CandidateWalk`;
- global `WorkBudget`;
- shared dead memo;
- sequential boundary resolution;
- R5 ambiguity policy;

unless a new deterministic test demonstrates a concrete defect.

The unfenced quotation residual is **not** evidence that Stage 2 is wrong: a clean Stage-1 call never
enters Stage 2.

---

# 3. R6-01 — MEDIUM
## Grammar constraint state ignores `ToolCallSyntaxMode`

### 3.1 Verified current code

Current constructor:

```cpp
ToolCallGrammarConstraint::ToolCallGrammarConstraint(
    std::size_t max_tool_name_length)
```

stores only:

```cpp
max_tool_name_length_
```

Current `parse_policy()` explicitly does:

```cpp
policy.syntax = ToolCallSyntaxMode::Compatibility;
```

Current inactive marker trigger also does:

```cpp
classify_tool_marker_prefix(
    state.marker_prefix_,
    marker,
    ToolCallSyntaxMode::Compatibility);
```

And the final pending-prefix decision again uses:

```cpp
ToolCallSyntaxMode::Compatibility
```

The class header has no `syntax_` member.

Therefore:

```text
production parser:
    QwenWrappedNative

constraint CPU core:
    Compatibility
```

This violates the Round-5 stated invariant that the selected syntax policy should be shared.

---

# 3.2 Why this matters even though constrained decoding is disabled

At present:

```text
--constrained-tool-decoding tool-calls-only
```

fails at engine startup, so R6-01 is not a current production execution vulnerability.

However, leaving the divergence in place guarantees a future integration trap:

```text
native parser sees bare <function=...> as content
constraint core sees it as a tool trigger
```

A future sampler integration could then mask/gate tokens according to a region the actual parser
would never execute.

This is exactly the sort of latent split-brain architecture Round 1–5 were intended to remove.

---

# 3.3 Required API change

Preferred minimal design:

```cpp
class ToolCallGrammarConstraint {
public:
    explicit ToolCallGrammarConstraint(
        std::size_t max_tool_name_length = 64,
        ToolCallSyntaxMode syntax =
            ToolCallSyntaxMode::Compatibility);

private:
    std::size_t max_tool_name_length_;
    ToolCallSyntaxMode syntax_;
    ...
};
```

Constructor:

```cpp
ToolCallGrammarConstraint::ToolCallGrammarConstraint(
    std::size_t max_tool_name_length,
    ToolCallSyntaxMode syntax)
    : max_tool_name_length_(
          max_tool_name_length == 0 ? 64 : max_tool_name_length),
      syntax_(syntax) {}
```

`parse_policy()`:

```cpp
policy.syntax = syntax_;
```

Inactive trigger:

```cpp
const ToolMarkerStatus status =
    classify_tool_marker_prefix(
        state.marker_prefix_,
        marker,
        state.syntax_);
```

Pending prefix:

```cpp
classify_tool_marker_prefix(
    state.marker_prefix_,
    marker,
    state.syntax_);
```

Use one member value everywhere.

---

# 3.4 Do not derive syntax mode from filenames/templates

Forbidden:

```cpp
if (template_name.contains("qwen3_8")) ...
```

or:

```cpp
if (model_id.contains("Qwen")) ...
```

The syntax mode is already an explicit request/server configuration concept.

Pass the value explicitly.

---

# 3.5 Checkpoint/restore requirement

`ToolCallGrammarConstraint` is value-semantic:

```cpp
checkpoint()
restore()
```

Adding `syntax_` must preserve that property automatically.

Add a test that checkpoints a native constraint, mutates/advances it, restores it, and verifies the
restored instance still treats bare `<function=...>` as ordinary prose.

This prevents a future accidental restore that copies buffer state but not policy state.

---

# 3.6 Required R6-01 test matrix

Add to:

```text
tests/test_tool_call_grammar_state.cpp
```

or the actual registered grammar-state test file.

## Native mode

### Complete wrapper

```text
<tool_call>
```

Expected:

```text
trigger / active
```

### Partial wrapper

Every prefix:

```text
<
<t
<to
<tool_
...
<tool_call
```

Expected:

```text
NeedMore or Allowed-prefix according to existing API
never compatibility trigger
```

### Bare function

```text
<function=read>
```

Expected:

```text
ordinary/unconstrained content
constraint does not become active
```

### Bare invoke

```text
<invoke=read>
```

Expected:

```text
ordinary/unconstrained content
```

### Function-calls wrapper

```text
<function_calls>
```

Expected:

```text
ordinary/unconstrained content
```

## Compatibility mode

All current historical top-level markers must retain current behavior:

```text
<tool_call>
<function_calls>
<function=read>
<invoke=read>
```

Expected:

```text
same verdicts as baseline
```

---

# 3.7 Cross-check against parser

For each top-level candidate:

```text
<tool_call>
<function_calls>
<function=read>
<invoke=read>
```

construct:

```text
parser mode = Native
constraint mode = Native
```

and:

```text
parser mode = Compatibility
constraint mode = Compatibility
```

The entry decision must agree.

The test does not need GPU sampling.

---

# 3.8 R6-01 acceptance criteria

- [ ] `ToolCallGrammarConstraint` stores `ToolCallSyntaxMode`.
- [ ] No `Compatibility` literal remains in marker/region decisions except intentional test/default declarations.
- [ ] `parse_policy().syntax == syntax_`.
- [ ] marker trigger uses `syntax_`.
- [ ] pending-marker classification uses `syntax_`.
- [ ] checkpoint/restore preserves the mode.
- [ ] Native constraint ignores compatibility-only top-level markers.
- [ ] Compatibility behavior is unchanged.
- [ ] runtime constrained decoding remains fail-fast.

---

# 4. R6-02 — LOW/MEDIUM
## Round-5 sign-off overstates residual ambiguity

### 4.1 Verified contradiction

Current `docs/tool_call_parser.md` explicitly says:

```text
a degenerate output made only of prose and unfenced complete examples
commits its last example in strict mode
```

Current Round-5 progress says:

```text
protocol ambiguity:
...
No residual finding.
```

Those cannot both be true.

The correct conclusion is:

```text
the known R1 early-vs-late-closer class is controlled by FailClosed;
general byte-identical unfenced quotation ambiguity remains.
```

---

# 4.2 Required progress correction

Change the Round-5 residual section to wording equivalent to:

```text
- protocol ambiguity:
  The R5 FailClosed policy prevents the reviewed R1 early-vs-late closer
  phantom class at the production boundary. The raw Qwen wire protocol is
  still semantically ambiguous for a complete declared unfenced tool-call
  example whose bytes are identical to an actual call. In particular,
  prose + a complete final example may still commit the final example.
  This is a documented residual, not an unresolved Stage-2 implementation bug.
```

Do **not** rewrite Round-5 historical test results.

Only correct the conclusion.

---

# 4.3 Required parser-doc wording

Ensure `docs/tool_call_parser.md` distinguishes:

```text
R1 structural boundary ambiguity
→ controlled by ToolCallAmbiguityPolicy

complete unfenced quotation ambiguity
→ not distinguishable by the byte parser
```

Recommended dedicated subsection:

```text
### Semantic quotation residual
```

State:

1. fences are handled;
2. undeclared names are handled;
3. compatibility-only top-level markers are excluded in native mode;
4. a fully valid, declared, canonical `<tool_call>` in ordinary unfenced content is still
   indistinguishable from an actual action when the byte sequence itself contains no contradictory
   structure;
5. `FailClosed` is not invoked for a clean Stage-1 completion;
6. an optional stricter agent policy can reduce this class but not eliminate the example-only case.

---

# 4.4 Required sign-off wording

Do not use:

```text
no residual finding
```

Do not use:

```text
all phantom tool calls fixed
```

Acceptable wording:

```text
No remaining HIGH/MEDIUM implementation defect was found in the reviewed
canonical native parser path. A documented semantic ambiguity remains for
byte-identical complete unfenced tool-call quotations; it cannot be resolved
from the wire bytes alone.
```

---

# 5. R6-03 — LOW
## Low-level decoder `finish()` still silently defaults to `FinishReason::None`

### 5.1 Verified current code

Current header:

```cpp
[[nodiscard]] Terminal finish(
    FinishReason finish_reason = FinishReason::None);
```

The production `OutputSession` now correctly calls:

```cpp
tool_call_output.finish(finish_reason);
```

with the terminal reason stored by the preview transaction.

Therefore the Round-5 production regression is fixed.

However, any new low-level caller can still write:

```cpp
decoder.finish();
```

and silently obtain:

```text
FinishReason::None
```

which is treated as a natural/unknown end by recovery policy.

The existing test suite has multiple direct `finish()` calls.

Some are deliberate one-shot tests. The problem is not that `None` exists; the problem is that it is
implicit.

---

# 5.2 Required API hardening

Preferred solution:

```cpp
[[nodiscard]] Terminal finish(FinishReason finish_reason);
```

Remove the default argument.

For callers that genuinely mean "unknown one-shot terminal":

```cpp
decoder.finish(FinishReason::None);
```

must be written explicitly.

This is intentionally verbose.

It makes review grep-able and prevents future accidental terminal-policy drift.

---

# 5.3 Do not invent a misleading semantic alias

Avoid:

```cpp
finish_natural_stop()
```

unless `None` is formally specified as a natural stop.

Current semantics are:

```text
StopToken
None for one-shot callers whose end reason is unknown
```

Those are not identical concepts.

The safest API is explicit:

```cpp
finish(FinishReason::None)
```

for intentionally unknown reason.

---

# 5.4 Update all callsites

Search the entire repository for:

```text
.finish()
```

on `ToolCallOutputDecoder`.

Every call must become one of:

```cpp
finish(FinishReason::StopToken)
finish(FinishReason::StopString)
finish(FinishReason::OutputLimit)
finish(FinishReason::ContextCapacity)
finish(FinishReason::Cancelled)
finish(FinishReason::None) // explicit unknown one-shot only
```

Do not mechanically replace every test with `StopToken`.

That would change semantics.

For each existing reasonless test, classify why `None` is appropriate.

Record the classification in the Round-6 progress file.

---

# 5.5 Compile-time guard

This change should make:

```cpp
decoder.finish();
```

a compile error.

That is intentional.

The type system then protects the terminal policy.

---

# 5.6 R6-03 tests

No new runtime test is strictly required for the signature itself, but add or preserve:

1. direct `None` one-shot parser/decoder test;
2. `OutputLimit` cut-sensitive decoder test;
3. `StopToken` natural-stop test;
4. `Cancelled` test through OutputSession;
5. existing Round-5 finish-reason matrix.

---

# 5.7 R6-03 acceptance criteria

- [ ] `ToolCallOutputDecoder::finish` has no default argument.
- [ ] all direct callsites pass a reason explicitly.
- [ ] production OutputSession behavior is unchanged.
- [ ] no test was blindly converted from unknown end to StopToken.
- [ ] Round-5 finish-reason matrix remains green.

---

# 6. R6-04 — HIGH integrity residual
## Complete declared unfenced example at EOF remains executable

### 6.1 Minimal reproducer

Declared tool:

```text
bash(command: string)
```

Output:

```text
Here is how the call looks:

<tool_call>
<function=bash>
<parameter=command>
echo example
</parameter>
</function>
</tool_call>
```

Finish:

```text
StopToken
```

Production policies:

```text
syntax = QwenWrappedNative
ambiguity = FailClosed
tolerant = false
```

Expected **current** behavior:

```text
marker_seen = true
one structured bash call
content before marker preserved
```

This is not a Stage-2 ambiguity.

The region beginning at `<tool_call>` is a clean Stage-1 completion.

---

# 6.2 Why FailClosed cannot reject it without another policy

The bytes of the structured region are exactly the same bytes an intended call would use.

No competing parameter boundary exists.

No open wrapper exists.

No undeclared name exists.

No fence exists.

No malformed tail exists.

Therefore:

```text
parser intent evidence = none
```

A byte parser cannot infer whether:

```text
"Here is how the call looks:"
```

is explanatory prose or an allowed natural-language preamble before a real call.

The official Qwen3.8 template explicitly permits natural-language reasoning before a function call.

This residual is therefore a **semantic ambiguity of the current native protocol**.

---

# 6.3 Mandatory Round-6 action: pin the residual

Add a direct regression test even if the current behavior remains accepted.

Suggested name:

```text
test_r6_complete_unfenced_in_set_example_residual
```

Run in:

```text
QwenWrappedNative
FailClosed
strict
StopToken
```

Fixture A:

```text
Example:
<tool_call>...bash...</tool_call>
```

Fixture B:

```text
Here is the syntax:
<tool_call>...bash...</tool_call>
```

Fixture C:

```text
<tool_call>...bash...</tool_call>
```

(no prose; positive control / genuinely indistinguishable)

Fixture D:

fenced version of the same call.

Expected current/default-template-compatible behavior:

```text
A/B: structured under TemplateCompatible semantics
C: structured
D: text / no call
```

The test name/comment must state this is a **documented residual**, not a desired global safety proof.

---

# 6.4 Why the residual test is necessary

Without a direct test, future documentation can accidentally claim:

```text
FailClosed eliminates phantom calls
```

while the implementation only eliminates one structural class.

A pinned residual is also necessary before adding the optional R6-05 hardening mode, because the test
becomes the before/after policy comparison.

---

# 7. R6-05 — RECOMMENDED HARDENING
## Optional “tool must start content” policy

### 7.1 Goal

Reduce the dominant unfenced quotation class without changing the native tag grammar.

Policy:

```text
A top-level tool region may latch only if every previously emitted byte
in the assistant Content channel is formatting whitespace.
```

Reasoning channel bytes do not count.

Once any non-whitespace Content byte is committed before a tool marker, the current assistant turn is
locked into text mode and later tool markers are content.

---

# 7.2 This is NOT canonical Qwen behavior

The official Qwen3.8 template permits natural-language text before a tool call.

Therefore this policy must be explicit.

Suggested enum:

```cpp
enum class ToolCallIntentPolicy : std::uint8_t {
    TemplateCompatible,
    RequireToolAtContentStart,
};
```

Product default choice must be deliberate.

Recommended rollout:

```text
Round-6 implementation:
    add both modes
    preserve TemplateCompatible as compatibility default

NInfer + OMP hardened profile:
    RequireToolAtContentStart
    plus tightened qwen3_8 template instruction
```

Do not silently make the global default stricter before corpus/agent validation.

---

# 7.3 Suggested serve flag

Example:

```text
--tool-call-intent template-compatible
--tool-call-intent start-of-content
```

or project-convention equivalent.

Do not overload:

```text
--tool-call-syntax
--tolerant-tool-calls
--tool-call-ambiguity
```

Intent is a separate dimension.

---

# 7.4 Suggested type

In `include/ninfer/types.h`:

```cpp
enum class ToolCallIntentPolicy : std::uint8_t {
    TemplateCompatible,
    RequireToolAtContentStart,
};
```

In `OutputOptions`:

```cpp
ToolCallIntentPolicy tool_call_intent =
    ToolCallIntentPolicy::TemplateCompatible;
```

If a hardened OMP-specific server profile is desired, set it explicitly in the launch configuration
rather than hiding it in parser internals.

---

# 7.5 Where to enforce it

Preferred layer:

```text
ToolCallStreamParser pre-latch scanner
```

Reason:

- one-shot and streaming then share semantics;
- fence state is already present there;
- marker candidate state is already present there;
- retry behavior stays centralized;
- OutputSession does not need a second text scanner.

Add policy to:

```cpp
ToolCallParsePolicy
```

or a dedicated pre-latch policy field passed by `ToolCallOutputDecoder`.

---

# 7.6 Required state machine

Suggested state:

```cpp
enum class ToolEntryGate : std::uint8_t {
    Open,       // only formatting whitespace observed
    Candidate,  // first non-whitespace bytes may be the native marker prefix
    TextLocked, // ordinary visible content committed; no tool marker may latch
};
```

Or equivalent booleans with equally clear invariants.

Do not implement this with ad-hoc string prefix checks scattered through `feed()`.

---

# 7.7 Formatting whitespace

Use the parser's existing formatting-whitespace definition where possible.

Document exactly which bytes count.

At minimum:

```text
' '
'\t'
'\r'
'\n'
```

Consider whether form feed should count only if already treated as format whitespace elsewhere.

Avoid locale-dependent `std::isspace` on signed chars.

---

# 7.8 Streaming transition rules

## Initial whitespace

Input:

```text
"\n  "
```

State:

```text
Open
```

Bytes may be published as content if current streaming semantics permit.

The gate remains open.

---

## First byte begins marker candidate

Input across chunks:

```text
"<to"
"ol_ca"
"ll>"
```

State:

```text
Open → Candidate → tool region latch
```

No ordinary visible content precedes the marker.

---

## Candidate fails

Input:

```text
"<tool_x"
```

Once the candidate is definitively not a native marker:

```text
Candidate → TextLocked
```

All candidate bytes are content.

A later:

```text
<tool_call>
```

must remain content in `RequireToolAtContentStart`.

This is essential. Otherwise the policy can be bypassed by a failed non-whitespace marker-like
prefix.

---

## Ordinary first content

Input:

```text
"Example:"
```

As soon as `E` is known ordinary content:

```text
Open → TextLocked
```

Later tool markup:

```text
<tool_call>...</tool_call>
```

must never latch.

---

## Fence before marker

Input:

````text
```xml
<tool_call>...</tool_call>
```
````

The first backtick is non-whitespace visible content:

```text
Open → TextLocked
```

The existing fence logic remains the primary fence protection; the intent gate is a second safety
layer.

---

## Reasoning channel

Reasoning does not flow through `ToolCallStreamParser`.

Therefore:

```text
reasoning text
→ ignored by this gate
```

After reasoning closes, a content-channel tool marker may still be the first non-whitespace Content
byte.

This is exactly what a hardened coding-agent profile wants.

---

# 7.9 Multiple tool calls

The policy applies to **entry into the structured tool region**, not each call independently.

Once a region begins with:

```text
<tool_call>
...
</tool_call>
<tool_call>
...
</tool_call>
```

existing parser semantics for consecutive wrappers remain unchanged.

Do not TextLock between valid consecutive wrappers inside one latched structured suffix.

---

# 7.10 Trailing content remains invalid

This hardening does not relax:

```text
valid tool call
+ suffix prose
```

Existing strict parser rules remain.

The official Qwen template also says no suffix after a tool call.

---

# 7.11 Tightened NInfer Qwen template for hardened profile

When `RequireToolAtContentStart` is used, change the NInfer local template instruction from:

```text
You may provide optional reasoning for your function call in natural language BEFORE the function call
```

to wording equivalent to:

```text
If you call a function, emit the <tool_call> block immediately in the assistant content channel.
Do not emit visible natural-language content before or after the tool call.
Reason internally in the model's reasoning channel instead.
```

Do not claim this is the upstream Qwen template.

It is an NInfer agent-hardening extension.

Keep the upstream-compatible template available if compatibility matters.

---

# 7.12 R6-05 regression matrix

Run both:

```text
TemplateCompatible
RequireToolAtContentStart
```

## Case A — plain intended call

```text
<tool_call>...</tool_call>
```

Expected:

```text
both → call
```

## Case B — whitespace before call

```text
"\n  \t<tool_call>...</tool_call>"
```

Expected:

```text
both → call
```

## Case C — prose before call

```text
"I'll inspect it.\n<tool_call>...</tool_call>"
```

Expected:

```text
TemplateCompatible → call
RequireToolAtContentStart → text / 0 calls
```

## Case D — explicit example prose

```text
"Example:\n<tool_call>...</tool_call>"
```

Expected:

```text
TemplateCompatible → call (documented residual)
RequireToolAtContentStart → text / 0 calls
```

## Case E — marker-like failed prefix then real call

```text
"<tool_x>\n<tool_call>...</tool_call>"
```

Expected:

```text
RequireToolAtContentStart → text / 0 calls
```

## Case F — fenced example

Expected:

```text
0 calls in both
```

## Case G — reasoning then tool

Simulate OutputSession:

```text
Reasoning channel: "I need to inspect the file"
Content channel: "<tool_call>...</tool_call>"
```

Expected:

```text
RequireToolAtContentStart → call
```

## Case H — content whitespace split bytewise

Every split must match one-shot.

---

# 7.13 What R6-05 still does NOT solve

This remains indistinguishable:

```text
<tool_call>
<function=bash>
<parameter=command>
echo example
</parameter>
</function>
</tool_call>
```

if the assistant intended to show the example **without any preceding content**.

Under both:

```text
TemplateCompatible
RequireToolAtContentStart
```

the bytes look exactly like an intended call.

Do not mark R6-04 globally solved after implementing R6-05.

Correct statement:

```text
RequireToolAtContentStart eliminates the prose-prefixed unfenced quotation class.
Example-only byte-identical output remains fundamentally ambiguous.
```

---

# 7.14 Rollout recommendation for OMP

For maximum safety in an automated coding agent:

```text
--tool-call-syntax qwen-wrapped
--tool-call-ambiguity fail-closed
strict tool calls (no tolerant flag)
--tool-call-intent start-of-content
```

plus the tightened local Qwen template.

Before making that profile the universal default, benchmark:

- tool-call success rate;
- missed valid calls due to model preamble;
- retries/handoffs;
- coding task success;
- tool-call format error rate.

A safety hardening that causes the model to emit preamble and lose legitimate calls may reduce agent
availability even while improving integrity.

Record both dimensions.

---

# 8. R6-06 — FUTURE ARCHITECTURE
## Explicit TEXT-vs-TOOL intent channel

### 8.1 Why this is the real solution to the final semantic ambiguity

A byte parser receives:

```text
<tool_call>...</tool_call>
```

and cannot know whether those bytes mean:

```text
"execute this"
```

or:

```text
"show this as an example"
```

when no surrounding syntax distinguishes the two.

To eliminate that class, intent must arrive through a channel that ordinary content cannot
accidentally reproduce.

Conceptually:

```text
assistant mode = TEXT
```

or:

```text
assistant mode = TOOL
```

before parsing the tool payload.

---

# 8.2 Desired architecture

```text
decoded model state
    ↓
assistant intent
    ├── TEXT
    │     ↓
    │   Content output
    │   <tool_call> bytes remain content
    │
    └── TOOL
          ↓
      tool grammar
          ↓
      parser validation
          ↓
      declared-tool validation
          ↓
      GeneratedToolCall
```

The tool parser should become a validator of a known TOOL channel instead of an intent detector over
arbitrary text.

---

# 8.3 Do not implement a plain text sentinel and call it solved

This is insufficient:

```text
<tool_mode>
<tool_call>...
```

if `<tool_mode>` itself is just ordinary model text that can be quoted.

A robust intent signal should ideally be:

- a reserved tokenizer special token;
- a structured decoder state not emitted into ordinary content;
- a model/runtime output head or side channel;
- or another representation ordinary user-visible text cannot accidentally reproduce.

Whether Qwen3.8 can use a custom special token reliably without retraining must be evaluated
experimentally.

Do not assume it.

---

# 8.4 Round-6 action for R6-06

**Design/document only.**

Create, if desired:

```text
docs/tool_call_intent_channel_design.md
```

Cover:

1. tokenizer/special-token feasibility;
2. chat-template changes;
3. model behavior without fine-tuning;
4. sampler state machine;
5. streaming API representation;
6. Responses/Chat/Anthropic serialization;
7. history replay;
8. stateful response caching;
9. prefix-cache interaction;
10. speculative decoding/checkpoint restore;
11. fallback behavior if intent token never appears;
12. compatibility with existing clients.

Do not merge an unvalidated custom intent token into the production parser as part of this bugfix.

---

# 9. R6-07 — constrained decoding remains a separate future phase

### 9.1 Keep current fail-fast

Current engine validation:

```cpp
if (options.constrained_tool_decoding !=
    ConstrainedToolDecoding::Off) {
    throw std::invalid_argument(
        "--constrained-tool-decoding=tool-calls-only is not implemented in this build; use off");
}
```

Keep it.

Round 6 fixes the CPU grammar-state syntax policy so future activation is coherent.

It does **not** enable runtime masking.

---

# 9.2 What constrained decoding would solve

Once correctly integrated, it can reduce:

- missing close tags;
- invalid function headers;
- invalid wrapper transitions;
- malformed parameter tag structure;
- certain truncation/format failures while a tool region is active.

---

# 9.3 What constrained decoding cannot solve by itself

A fully valid quoted:

```text
<tool_call>...</tool_call>
```

is valid grammar.

Constrained decoding does not know semantic intent.

Therefore:

```text
constrained decoding != phantom-call intent solution
```

The two architectures are complementary:

```text
intent channel decides TEXT vs TOOL
constraint decides legal TOOL bytes
parser validates final TOOL structure
```

---

# 10. Optional parameter-wire research

The current Qwen3.8 selected template uses raw parameter payloads:

```text
<parameter=name>
raw bytes
</parameter>
```

This creates delimiter ambiguity for arbitrary strings.

A theoretically cleaner wire format would use:

- JSON arguments;
- escaping;
- length framing;
- or a binary/structured side channel.

However, changing the native Qwen tool syntax can reduce model tool-use quality because it deviates
from the trained template.

Therefore **do not change the production Qwen3.8 wire format in Round 6**.

If explored later, benchmark native format vs custom JSON/framed format on:

- real coding tool calls;
- long `write.content`;
- shell here-docs;
- XML/HTML source;
- tool-call examples embedded in files;
- multiline whitespace fidelity;
- exact argument preservation.

---

# 11. Implementation phases

## Phase R6-0 — Freeze baseline and evidence

Record:

```text
start SHA
worktree clean/dirty
compiler
build directory
CMake generator
CPU test command
GPU tests NOT RUN
```

Re-run targeted baseline tests before code changes.

Required baseline tests:

```text
ninfer_tool_call_parser_test
ninfer_tool_call_grammar_test
ninfer_tool_call_grammar_state_test
ninfer_qwen3_5_frontend_test
ninfer_request_log_test
ninfer_serve_options_test
ninfer_engine_options_test
```

Add the R6-04 residual test **before** changing intent policy.

---

## Phase R6-1 — Fix constraint syntax-mode divergence

Implement R6-01 only.

Files expected:

```text
src/models/qwen3_5/frontend/tool_call_grammar_state.h
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp
tests/test_tool_call_grammar_state.cpp
```

Possibly supporting test helpers only.

Do not touch Stage 2.

Run:

```text
grammar
grammar_state
tool_call_parser
```

Outcome dump for parser should be byte-identical because production parser behavior is unchanged.

---

## Phase R6-2 — Harden decoder finish API

Implement R6-03.

Files:

```text
src/models/qwen3_5/frontend/tool_call_parser.h
tests/test_tool_call_parser.cpp
possibly other direct decoder callers
```

The production OutputSession call should remain:

```cpp
finish(real_reason)
```

Run parser + frontend tests.

No output behavior should change.

---

## Phase R6-3 — Correct residual documentation

Implement R6-02.

Files:

```text
docs/NInfer_new_parser_design_round5_bugfix_progress.md
docs/tool_call_parser.md
docs/serving.md if necessary
```

Do not rewrite history.

State exactly:

```text
R1 structural class fixed under FailClosed
general semantic quotation residual remains
```

---

## Phase R6-4 — Pin semantic quotation residual

Add R6-04 regression fixtures.

Do not change their expected behavior yet under `TemplateCompatible`.

This phase exists to make the limitation executable/test-visible.

---

## Phase R6-5 — Optional start-of-content hardening

Implement R6-05 behind explicit policy.

Files likely:

```text
include/ninfer/types.h
src/models/qwen3_5/frontend/tool_call_stream.h
src/models/qwen3_5/frontend/tool_call_stream.cpp
src/models/qwen3_5/frontend/tool_call_parser.h
src/models/qwen3_5/frontend/tool_call_parser.cpp
src/models/qwen3_5/frontend/output_session.cpp
src/serve/serve_options.h
src/serve/serve_options.cpp
src/serve/translate.cpp
tools/chat_templates/qwen3_8.jinja   (only hardened local profile/instruction)
tests/test_tool_call_parser.cpp
tests/models/qwen3_5/test_frontend.cpp
tests/test_serve_options.cpp
docs/tool_call_parser.md
docs/serving.md
```

Do not entangle this with R6-01.

Separate commit.

---

## Phase R6-6 — Design future intent channel

Documentation only.

No production behavior change.

---

# 12. Required Round-6 test corpus

## 12.1 Syntax consistency

Cross product:

```text
syntax:
  Native
  Compatibility

entry:
  tool_call
  function_calls
  bare function
  bare invoke

consumer:
  marker classifier
  stream parser
  grammar constraint
```

Expected entry acceptance must agree.

---

## 12.2 Finish-reason API

Direct decoder:

```text
None explicit
StopToken
StopString
OutputLimit
ContextCapacity
Cancelled
```

OutputSession:

reuse Round-5 S8-cut matrix.

---

## 12.3 Quotation residual

Declared tool:

```text
bash
```

Cases:

```text
prose + complete call + EOF
example label + complete call + EOF
complete call only
fenced complete call
undeclared complete call
compat-only bare example under Native
```

---

## 12.4 Intent hardening mode

If R6-05 implemented:

```text
TemplateCompatible
RequireToolAtContentStart
```

for every quotation fixture.

Also:

```text
reasoning channel text + content-start call
whitespace + call
partial marker split every byte
failed marker candidate + later real marker
fence before marker
multiple consecutive calls
```

---

## 12.5 Streaming invariance

For each new fixture:

```text
one-shot
1-byte chunks
2-byte
3-byte
5-byte
7-byte
every two-way split around:
  <
  <tool_call>
  fence opener
  first non-whitespace content byte
```

Same final result.

---

# 13. Adversarial tests

Actively attempt to falsify the new intent gate.

## A. Failed marker candidate

```text
<tool_x
<tool_call>real...</tool_call>
```

Hardened mode:

```text
0 calls
```

---

## B. Leading comments

```text
<!-- example -->
<tool_call>...</tool_call>
```

Hardened mode:

```text
0 calls
```

Do not special-case HTML.

---

## C. Unicode prose

```text
Beispiel:
<tool_call>...</tool_call>
```

First UTF-8 non-whitespace byte locks text.

---

## D. Whitespace-only preamble

```text
\r\n\t  <tool_call>...</tool_call>
```

Must remain executable.

---

## E. Reasoning before call

Reasoning channel:

```text
I should inspect the repo
```

Content channel:

```text
<tool_call>...</tool_call>
```

Must remain executable in hardened mode.

---

## F. Content preamble generated after reasoning

Content:

```text
I will inspect it.
<tool_call>...</tool_call>
```

Hardened mode:

```text
0 calls
```

This is intentional behavior change.

---

## G. Example only

```text
<tool_call>...</tool_call>
```

No byte parser can distinguish intent.

Test remains:

```text
structured
```

and docs say why.

---

# 14. Parser/API review loops

## Review A — syntax-policy single source of truth

Search for:

```text
ToolCallSyntaxMode::Compatibility
ToolCallSyntaxMode::QwenWrappedNative
classify_tool_marker_prefix
find_tool_marker
parse_tool_call_region
```

Every use must be categorized:

```text
intentional default
test fixture
actual decision
```

No actual constraint decision may hard-code Compatibility after R6-01.

---

## Review B — terminal API

Search:

```text
ToolCallOutputDecoder
.finish(
commit_preview
FinishReason::None
```

Verify:

- production terminal path uses stored real reason;
- low-level unknown reason is explicit;
- no new default exists elsewhere.

---

## Review C — quotation / intent

Trace:

```text
Content delta
→ pre-latch fence state
→ intent gate
→ marker candidate
→ native syntax
→ Stage 1
→ FailClosed Stage 2 if needed
→ structured call
```

For:

1. intended call;
2. prose-prefixed call;
3. fenced example;
4. example-only call;
5. undeclared call.

---

## Review D — compatibility

Run same fixtures in Compatibility mode.

The optional intent policy is orthogonal to syntax:

```text
Compatibility + RequireToolAtContentStart
```

must still reject a bare marker that appears after visible prose.

---

## Review E — no Stage-2 regression

Re-run all Round-3/4/5 boundary tests.

No behavior change expected except where a new intent policy is deliberately selected.

---

# 15. CPU-only verification commands

Use the actual project target names.

Recommended targeted run:

```powershell
cmake --build build-new-parser --config Release --parallel 16
```

```powershell
ctest --test-dir build-new-parser -C Release `
  -R "ninfer_(tool_call_parser|tool_call_grammar|tool_call_grammar_state|qwen3_5_frontend|request_log|serve_options|engine_options)_test" `
  --output-on-failure `
  --parallel 16
```

Then full CPU gate:

```powershell
$env:CUDA_VISIBLE_DEVICES="99"

ctest --test-dir build-new-parser -C Release `
  -E "_real" `
  --output-on-failure `
  --parallel 16
```

Record exact:

```text
total
passed
failed
skipped
```

Do not label GPU-dependent tests green if they exit as failures while printing skip.

List them separately.

---

# 16. Outcome dump

Reuse the Round-5 deterministic corpus if available.

Add dimensions only after implemented:

```text
syntax:
  native
  compat

ambiguity:
  payload
  fail-closed

intent:
  template-compatible
  start-of-content
```

Expected:

### R6-01

No parser outcome-dump delta.

Constraint tests only.

### R6-03

No output delta.

Compile/API hardening only.

### R6-05

Intent-policy deltas must be limited to outputs with non-whitespace content before a candidate tool
entry.

If unrelated canonical calls change, investigate.

---

# 17. Suggested commit decomposition

Recommended:

```text
1. test(frontend): pin round-6 unfenced quotation residual
2. fix(frontend): make tool grammar constraint syntax-mode aware
3. test(frontend): cover native/compat constraint entry matrix
4. refactor(frontend): require explicit finish reason in tool decoder
5. docs(frontend): correct round-5 residual sign-off
6. feat(frontend): add optional start-of-content tool intent policy
7. test(frontend): add intent-policy streaming/adversarial matrix
8. docs(frontend): document round-6 hardening and residual ambiguity
9. docs(frontend): add future structured tool-intent design
```

Do not squash all semantic changes into one commit before review.

---

# 18. Round-6 progress file

Create:

```text
docs/NInfer_new_parser_design_round6_bugfix_progress.md
```

Minimum sections:

```text
# Baseline
start SHA
toolchain
build dir
GPU tests NOT RUN

# Finding status
R6-01
R6-02
R6-03
R6-04
R6-05
R6-06
R6-07

# Baseline repro
constraint syntax mismatch
unfenced complete-example residual

# Step results
per commit

# Intentional outcome changes
syntax
intent policy

# Residuals
example-only byte-identical ambiguity
constrained runtime unavailable

# Final sign-off
targeted tests
full CPU gate
CI status
remaining implementable HIGH/MEDIUM findings
remaining protocol limits
```

---

# 19. Definition of Done — mandatory bugfix scope

Round 6 mandatory scope is green only when:

- [ ] R6-01 constraint state accepts `ToolCallSyntaxMode`.
- [ ] R6-01 native constraint ignores bare function/invoke/function_calls entries.
- [ ] R6-01 compatibility constraint preserves prior entry behavior.
- [ ] Parser and constraint entry decisions match per mode.
- [ ] Constraint checkpoint/restore preserves syntax mode.
- [ ] Runtime constrained decoding still fails fast.
- [ ] R6-02 Round-5 progress no longer says “No residual finding”.
- [ ] Parser docs explicitly separate structural FailClosed ambiguity from semantic quotation ambiguity.
- [ ] R6-03 decoder `finish()` no longer has an implicit `None` default.
- [ ] All direct `finish` callers provide an explicit reason.
- [ ] Round-5 terminal-reason matrix remains green.
- [ ] R6-04 unfenced complete-example residual has a direct test.
- [ ] Fenced equivalent remains non-executable.
- [ ] Undeclared equivalent remains non-executable.
- [ ] Native mode still ignores compatibility-only top-level syntax.
- [ ] Stage-2 outcome corpus is unchanged under existing policies.
- [ ] No GPU/model runtime test was run.
- [ ] Targeted CPU suites pass.
- [ ] Full CPU result is recorded honestly.

---

# 20. Definition of Done — optional R6-05 hardening

If `RequireToolAtContentStart` is implemented:

- [ ] policy is explicit in `OutputOptions`.
- [ ] policy is independently configurable from syntax/tolerance/ambiguity.
- [ ] TemplateCompatible preserves Qwen-compatible current behavior.
- [ ] hardened mode accepts whitespace + tool call.
- [ ] hardened mode rejects prose + tool call.
- [ ] failed marker candidate closes the tool-entry gate.
- [ ] reasoning-channel text does not close the Content gate.
- [ ] fenced examples remain text.
- [ ] multiple consecutive valid calls still work.
- [ ] one-shot == streaming.
- [ ] every-byte splits around first marker are covered.
- [ ] docs explicitly state example-only ambiguity remains.
- [ ] NInfer hardened template says no visible preamble before a tool call.
- [ ] agent/corpus validation records any valid-call loss caused by model preambles.

---

# 21. What “green” may mean after Round 6

If only mandatory scope is implemented:

> No remaining HIGH/MEDIUM implementation defect was found in the reviewed canonical Qwen3.8
> parser/integration path. The grammar-constraint CPU core now follows the selected syntax mode and
> terminal reasons are explicit. A documented semantic ambiguity remains for complete declared
> unfenced tool-call quotations whose bytes are identical to a genuine action.

If R6-05 is also enabled in the production OMP profile:

> The hardened agent profile additionally refuses tool calls that appear after visible content,
> eliminating the prose-prefixed unfenced quotation class. A tool-call-only output is still
> semantically indistinguishable from a genuine action without an external TEXT-vs-TOOL intent
> signal.

Do not say:

```text
all tool-call errors solved
phantom calls impossible
wire protocol unambiguous
```

---

# 22. Long-term target architecture

The strongest architecture is:

```text
model output intent
   ↓
TEXT --------------------------→ content
   |
   └── TOOL
         ↓
      grammar-constrained generation
         ↓
      native Qwen tool grammar
         ↓
      strict structural parser
         ↓
      declared-tool enforcement
         ↓
      argument/schema normalization
         ↓
      GeneratedToolCall
```

The current parser should remain the validation boundary even after constrained decoding is added.

Never rely on sampling constraints alone.

---

# 23. Final implementation order

For lowest risk:

```text
R6-04 residual test first
        ↓
R6-01 constraint syntax consistency
        ↓
R6-03 explicit finish reason API
        ↓
R6-02 docs/sign-off correction
        ↓
targeted + full CPU gate
```

Then, only if desired:

```text
R6-05 optional start-of-content hardening
        ↓
agent/corpus validation
        ↓
documentation
```

Finally:

```text
R6-06 structured intent-channel design
R6-07 future constrained-sampling implementation
```

This ordering keeps the already-stable parser core isolated from speculative architecture work.

---

# 24. Primary source / evidence references

## NInfer reviewed baseline

```text
https://github.com/Hundsbuah/ninfer/tree/8b4c4e9904d1347cc73769954273184cb97d287c
```

Relevant current files:

```text
src/models/qwen3_5/frontend/tool_call_grammar_state.h
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp
src/models/qwen3_5/frontend/tool_call_stream.h
src/models/qwen3_5/frontend/tool_call_stream.cpp
src/models/qwen3_5/frontend/tool_call_parser.h
src/models/qwen3_5/frontend/tool_call_parser.cpp
src/models/qwen3_5/frontend/output_session.cpp
include/ninfer/types.h
tools/chat_templates/qwen3_8.jinja
docs/tool_call_parser.md
docs/NInfer_new_parser_design_round5_bugfix_progress.md
```

## Official Qwen3.8 template

```text
https://huggingface.co/Qwen/Qwen3.8-27B/blame/main/chat_template.jinja
```

Verified behavior:

- wrapped `<tool_call>` native format;
- inner `<function=...>`;
- parameter tags;
- no suffix after a call;
- optional natural-language reasoning before a call.

## Qwen function-calling documentation

```text
https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md
```

Relevant point:

- malformed tool-call corner cases can occur;
- robust parsing remains required in production.

## External Qwen parser corroboration

```text
https://github.com/vllm-project/vllm/issues/57541
```

Relevant point:

- syntactically valid tool markup in content can become a phantom call;
- a fence-aware parser mitigates the fenced sub-class;
- intent cannot be inferred from valid markup alone.

---

# 25. Final engineering principle

At this stage, the parser is no longer primarily missing another clever delimiter heuristic.

The remaining engineering boundary is:

```text
STRUCTURE
vs
INTENT
```

Round 1–5 substantially improved structural correctness.

Round 6 should:

```text
1. finish policy consistency,
2. make terminal semantics explicit,
3. pin the semantic residual,
4. optionally narrow the production agent contract,
5. stop before destabilizing Stage 2,
6. reserve true semantic disambiguation for an explicit intent channel.
```

That is the most defensible path to higher tool-call reliability without converting the parser into
an increasingly fragile collection of prose heuristics.
