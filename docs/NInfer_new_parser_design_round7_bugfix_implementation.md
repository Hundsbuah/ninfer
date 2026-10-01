# NInfer `new_parser_design` — Round 7 Bugfix / Hardening Implementation Specification

## 0. Purpose

This document is the **Round-7 implementation specification** for the remaining findings found in
the full-chain review of `Hundsbuah/ninfer:new_parser_design` after the Round-6 changes.

Round 7 is intentionally **not** another general parser rewrite.

The parser core is now materially stronger than it was in the earlier rounds. The new findings are
concentrated in the **composition** of already-existing mechanisms:

```text
intent gate
×
retry/recovery chain
×
terminal re-parse
×
future grammar constraint
×
chat-template policy
×
test-gate reporting
```

The goals are:

1. fix a real integrity bug in `RequireToolAtContentStart`: a later retry base can currently
   execute a tool call after bytes that become visible Content;
2. ensure the streaming decoder's terminal re-parse carries the exact same intent policy as the
   live incremental scanner;
3. prevent the future grammar-constrained decoder from diverging from the parser when
   `RequireToolAtContentStart` is selected;
4. make the local hardened Qwen3.8 prompt contract and the runtime intent policy explicitly
   coherent instead of changing the model instruction unconditionally;
5. correct the Round-6 test-gate wording and, optionally, fix the LinearSwiGLU skip propagation so
   CTest receives exit code `77` rather than `1`;
6. preserve all Round-1 through Round-6 parser invariants and avoid destabilizing Stage 2.

This specification distinguishes:

```text
P0 correctness/integrity
P1 architecture consistency
P1 prompt/runtime policy consistency
P2 verification infrastructure
known protocol residual
future constrained-decoding work
```

Do not collapse these categories into one "parser fixed" claim.

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

Reviewed Round-7 start HEAD:

```text
4a69b7a31b6570bfd7fce220a822492b24a161e4
```

Tree:

```text
085a1739044186e08a13426f40c3d1315236a50b
```

Previous relevant Round-6 commits:

```text
417c805b11eb  feat(frontend): add optional start-of-content tool intent policy
f9871068b59a  test(frontend): add intent-policy streaming/adversarial matrix
f43023581783  fix(frontend): repair intent-gate test session semantics and template string
b4c6c8afa8e2  docs(frontend): document round-6 intent policy and hardening
e44da8685156  test(frontend): adapt grammar-state cross-check to intent-aware parser entry
4a69b7a31b65  docs: record round-6 parser bugfix progress and spec
```

At review time:

```text
GitHub combined status checks: none
GitHub workflow runs for HEAD: none
```

No independent GitHub CI corroborates the branch.

Before implementation, record the actual current branch SHA again. If it differs from the pinned
baseline above, re-run the code-location verification before applying this document mechanically.

---

# 0.2 Current production/default policy state

At the pinned baseline:

```text
OutputOptions.tool_call_syntax     = QwenWrappedNative
OutputOptions.tool_call_ambiguity  = FailClosed
OutputOptions.tool_call_intent     = TemplateCompatible
OutputOptions.tolerant_tool_calls  = false
```

Serving flags:

```text
--tool-call-syntax qwen-wrapped|compat
--tool-call-ambiguity fail-closed|payload-fidelity
--tool-call-intent template-compatible|start-of-content
--tolerant-tool-calls
```

The optional hardened mode is:

```text
--tool-call-intent start-of-content
```

The runtime constrained-decoding option still fails fast and is not production-enabled.

---

# 0.3 Primary-source Qwen contract

The current official Qwen3.8-27B template uses the wrapped native form:

```text
<tool_call>
<function=...>
<parameter=...>
...
</parameter>
</function>
</tool_call>
```

The official template states:

```text
no suffix after a function call
```

and permits optional natural-language reasoning **before** the function call.

Primary source:

```text
https://huggingface.co/Qwen/Qwen3.8-27B/blame/main/chat_template.jinja
```

Therefore:

- `QwenWrappedNative` remains the correct native syntax mode;
- a "tool must be first non-whitespace Content" rule is a stricter NInfer agent policy, not the
  upstream Qwen wire contract;
- if NInfer changes the prompt wording to forbid preamble, that prompt policy must be explicitly
  coupled to the selected runtime policy or clearly shipped as a separate template/profile.

Qwen's function-calling documentation also explicitly acknowledges malformed tool-call corner cases
and recommends robust parsing in production:

```text
https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md
```

External corroboration that syntactically valid tool markup can become a phantom tool call exists in
vLLM's Qwen parser:

```text
https://github.com/vllm-project/vllm/issues/57541
```

This external issue is supporting evidence only. NInfer fixes must be proved against NInfer code and
NInfer tests.

---

# 0.4 GPU / runtime test constraint

As in previous rounds:

**Do not run model inference, `_real` model tests, GPU correctness tests, CUDA memcheck/racecheck,
server model smoke tests, or benchmarks in this Round-7 task.**

Allowed:

- CPU builds;
- parser tests;
- streaming decoder tests;
- frontend OutputSession tests;
- serve-option tests;
- grammar-state CPU tests;
- template render/parse tests;
- CTest registration / skip-behavior tests;
- deterministic corpus and outcome dumps;
- compile-time API checks.

`ToolCallGrammarConstraint` is a CPU core and may be tested.

Runtime constrained sampling remains out of scope.

---

# 1. Round-7 findings

| ID | Finding | Severity | Type | Mandatory |
|---|---|---:|---|---|
| **R7-01** | `RequireToolAtContentStart` can be bypassed by Stage-1 retry / later parse base after the first marker already latched | **HIGH in hardened mode** | correctness / integrity | **YES** |
| **R7-01b** | `ToolCallOutputDecoder::finish()` omits `intent_` when it re-parses the latched region, silently falling back to `TemplateCompatible` | **HIGH coupling bug** | correctness / policy propagation | **YES** |
| **R7-02** | `ToolCallGrammarConstraint` carries syntax but not intent; future constrained decoding can trigger where the hardened parser is text-locked | **MEDIUM architecture** | dormant today, blocking future feature | **YES before constrained runtime activation** |
| **R7-03** | local `qwen3_8.jinja` always emits hardened no-preamble instruction although serving default remains `TemplateCompatible` | **MEDIUM policy mismatch** | prompt/runtime consistency | **YES** |
| **R7-04** | Round-6 progress claims a green full CPU gate while recording 4 CTest failures; LinearSwiGLU skip 77 is converted to exit 1 | LOW / VERIFY | verification correctness | **YES docs; test-wrapper fix recommended** |
| R6-04 residual | byte-identical complete unfenced call-only output cannot be semantically disambiguated by bytes alone | HIGH residual | protocol limit | **NOT a parser bug; keep documented** |
| R6-07 | runtime constrained decoding unavailable | known limitation | future architecture | **KEEP FAIL-FAST** |

---

# 2. Round-7 invariants

## R7-I1 — The intent decision applies to every executable entry, not only the first scanner latch

Under:

```text
RequireToolAtContentStart
```

an executable call may be accepted only if the accepted structured region starts at the first
non-formatting-whitespace byte of the assistant **Content channel**.

This rule applies equally to:

- initial marker latch;
- Stage-1 retry;
- Stage-2 completion at a later base;
- Stage-3 tolerant recovery at a later base;
- streaming decoder terminal re-parse;
- one-shot parser entry.

A later parser base is not a loophole.

---

## R7-I2 — A later accepted base whose prefix becomes visible Content is forbidden in hardened mode

If a candidate result would produce:

```text
result.tail = region_.substr(0, accepted_base)
```

and:

```text
accepted_base > 0
```

then under `RequireToolAtContentStart` that result must not execute.

Reason:

```text
result.tail
→ becomes assistant Content
→ accepted call would appear after visible Content
→ violates the policy definition
```

This is true even if the prefix is malformed tool syntax rather than ordinary prose.

---

## R7-I3 — The first latched region may still be repaired in place

Round 7 must **not** disable Stage 2 or tolerant recovery for base `0`.

Allowed under hardened mode:

```text
base 0
→ Stage 1 fails
→ Stage 2 finds a consistent parameter boundary at base 0
→ accept
```

Allowed when tolerant mode is explicitly enabled:

```text
base 0
→ complete function already closed
→ permitted Stage-3 recovery under existing finish-reason rules
→ accept
```

Forbidden:

```text
base 0 fails
→ retry at base > 0
→ accept later call
```

This distinction is central to avoiding an availability regression.

---

## R7-I4 — One-shot and streaming must carry identical intent policy

The policy tuple:

```text
syntax
ambiguity
intent
tolerant
finish_reason
contract
```

must be identical in:

```text
incremental machine
terminal one-shot re-parse
```

No default parameter may silently substitute a different policy.

---

## R7-I5 — Constraint policy must eventually match parser policy

Before runtime constrained decoding can be enabled, the constraint state must agree with the parser
on both:

```text
ToolCallSyntaxMode
ToolCallIntentPolicy
```

Otherwise the sampler can enter TOOL grammar after bytes the parser has already classified as TEXT.

---

## R7-I6 — Prompt policy and runtime policy must be explicit

Do not claim:

```text
template-compatible
```

while injecting an unconditional instruction that requires:

```text
tool call at content start, no visible preamble
```

The model prompt contract and parser/runtime policy may intentionally differ, but only when that
difference is explicit, documented, and tested.

---

## R7-I7 — Verification reports must use CTest semantics, not console wording

If CTest reports:

```text
4 failed
```

the gate is not green.

A test printing:

```text
SKIP: ...
```

does not make it skipped unless its actual process exit code reaches CTest as the configured
`SKIP_RETURN_CODE`.

---

# 3. R7-01 — HIGH
## Hardened intent gate is bypassed by retry at a later parser base

### 3.1 Verified current architecture

The Round-6 gate is implemented before the first latch.

`publish()`:

```cpp
if (policy_.intent == ToolCallIntentPolicy::RequireToolAtContentStart && !latched_) {
    for (const char byte : bytes) {
        if (!is_tool_format_whitespace(byte)) {
            entry_locked_ = true;
            break;
        }
    }
}
```

`marker_byte()`:

```cpp
if (state == ToolMarkerStatus::Complete) {
    if (policy_.intent == ToolCallIntentPolicy::RequireToolAtContentStart &&
        entry_locked_) {
        ...
        return false;
    }
    latch(marker_prefix_);
    return true;
}
```

This correctly protects a marker found **after content published before any latch**.

Once the first marker latches:

```cpp
latched_ = true;
```

all later bytes enter:

```text
region_
```

and the pre-latch gate is no longer consulted.

---

# 3.2 Verified retry path

Current `finish()` constructs a Stage-1 attempt chain:

```cpp
std::vector<Attempt> chain;
std::size_t base = 0;

for (;;) {
    ...
    parse_region_at(region_, ..., base);
    ...
    std::size_t next = find_tool_marker(region_, from, policy_.syntax);
    ...
    base = next;
}
```

If a later base completes:

```cpp
result.status       = ToolCallStreamStatus::Complete;
result.region.calls = std::move(progress.calls);
result.tail         = region_.substr(0, base);
return result;
```

No check of:

```text
policy_.intent
```

exists in the retry chain.

Stage 2 then iterates:

```cpp
for (const Attempt& attempt : chain)
```

and can likewise accept a later `attempt.base`.

Stage 3 also iterates the whole chain and can commit a later attempt.

Therefore the Round-6 gate protects only:

```text
scanner latch
```

not:

```text
accepted parser base
```

---

# 3.3 Core reproducer A — valid first call + prose + valid later call

Declared tools:

```text
read(path: string)
bash(command: string)
```

Input:

```text
<tool_call>
<function=read>
<parameter=path>
example.txt
</parameter>
</function>
</tool_call>

Now the real action:

<tool_call>
<function=bash>
<parameter=command>
echo REAL
</parameter>
</function>
</tool_call>
```

Policy:

```text
syntax     = QwenWrappedNative
ambiguity  = FailClosed
intent     = RequireToolAtContentStart
tolerant   = false
finish     = StopToken
```

Current path:

```text
first marker latches at content start
→ first call completes
→ prose after call causes Definitive/TrailingContent
→ retry scans to later <tool_call>
→ later bash call completes
→ result.tail = first call + prose
→ tail becomes visible Content
→ bash remains structured/executable
```

Published semantics become:

```text
assistant Content:
    first call markup + prose

structured tool call:
    bash("echo REAL")
```

This violates the hardening contract.

---

# 3.4 Core reproducer B — prose immediately after open wrapper

Input:

```text
<tool_call>
This wrapper is only an example

<tool_call>
<function=bash>
<parameter=command>
echo REAL
</parameter>
</function>
</tool_call>
```

The parser intentionally marks the first break as:

```cpp
prose_after_wrapper = true;
```

That relaxes wrapper ownership for the historical R3-02 retry logic.

Current retry condition:

```cpp
const bool owned =
    wrapper_at_break != ToolWrapperKind::None &&
    !prose_after_wrapper;
```

With:

```text
prose_after_wrapper = true
```

the failed first wrapper is not treated as owning the remainder, and retry can find the later
wrapper.

That historical behavior is valid under `TemplateCompatible`.

It is **not** valid under `RequireToolAtContentStart`, because accepting the later call necessarily
turns the earlier region bytes into visible Content before the call.

---

# 3.5 R7-01 required semantic rule

Define:

```text
allow_later_retry_base =
    policy.intent == TemplateCompatible
```

or equivalently:

```text
RequireToolAtContentStart
→ only base == 0 is executable
```

after the first marker has already latched.

The simplest safe rule is:

```cpp
if (policy_.intent == ToolCallIntentPolicy::RequireToolAtContentStart) {
    break; // do not search another entry base
}
```

after the first failed Stage-1 attempt has been appended to `chain`, but **before**
`find_tool_marker(...)`.

This preserves:

```text
chain[0]
```

for Stage 2 and Stage 3.

---

# 3.6 Preferred implementation point

Current conceptual shape:

```cpp
chain.push_back(Attempt{base, std::move(progress)});
const Attempt& last = chain.back();

const bool owned = ...;
const bool open_value = ...;
if (owned || ...) { break; }

... retry marker search ...
```

Recommended:

```cpp
chain.push_back(Attempt{base, std::move(progress)});
const Attempt& last = chain.back();

if (policy_.intent == ToolCallIntentPolicy::RequireToolAtContentStart) {
    // The initially latched marker was the only valid tool-entry point.
    // Any later base would make region_[0:base] visible Content before
    // the accepted tool call, violating RequireToolAtContentStart.
    break;
}

const bool owned = ...;
...
```

The exact position may be after computing diagnostics if needed, but it must occur before a later
`base` is added to the chain.

---

# 3.7 Why this is preferable to post-hoc rejection

Alternative:

```cpp
if (accepted_base > 0 && hardened)
    reject;
```

in every acceptance site.

That would require duplicating the same invariant in:

```text
Stage 1
Stage 2 balanced
Stage 2 unbalanced
Stage 3
```

and creates future omission risk.

Preferred architecture:

```text
hardened intent
→ later base never enters candidate chain
→ all downstream stages naturally operate on base 0 only
```

This makes the invariant structural.

---

# 3.8 Important exception: pre-latch whitespace

A latched native region may start with formatting whitespace held in:

```text
pending_ws_
```

The region can therefore contain:

```text
"\n  \t<tool_call>..."
```

while the first marker's structural base remains conceptually the first entry.

Do **not** interpret the number of leading whitespace bytes as a later recovery base.

The retry restriction applies to a **new top-level marker chosen after the first latched entry failed**,
not to the existing held whitespace before the first marker.

If current `base` is relative to `region_` including whitespace, verify what value Stage 1 uses.
Do not assume `base == 0` without checking the exact `region_` layout. If leading whitespace means the
first parsed marker is internally found after whitespace, introduce an explicit:

```text
initial_entry_base
```

or:

```text
attempt.is_initial_entry
```

rather than using a fragile numeric comparison.

This is important.

The semantic invariant is:

```text
only the originally latched entry may execute
```

not literally:

```text
offset must equal integer zero
```

if region storage includes held whitespace.

---

# 3.9 Recommended robust representation

Instead of inferring from offsets later:

```cpp
struct Attempt {
    std::size_t base;
    ToolCallParseProgress progress;
    bool initial_entry = false;
};
```

First attempt:

```cpp
Attempt{.base = initial_base, .progress = ..., .initial_entry = true}
```

Retries:

```cpp
Attempt{.base = next, .progress = ..., .initial_entry = false}
```

Then:

```text
RequireToolAtContentStart
→ only initial_entry attempt may be committed
```

However, if the existing code guarantees first `base == 0` even with held whitespace because
`parse_region_at` itself skips format whitespace, keep the simpler code and pin that behavior with a
test.

Do not add state complexity without need.

---

# 3.10 Stage-2 impact

With the preferred retry-chain cutoff:

```text
chain.size() == 1
```

in hardened mode.

Stage 2 still receives the initial attempt and may:

- repair parameter-value boundary interpretation;
- resolve nested/opaque payload ambiguity according to existing policy;
- hit deterministic work budget;
- return `AmbiguousStructure`;
- accept base-zero balanced completion;
- accept base-zero single unbalanced completion under the existing selection rule.

Do not special-case Stage 2 further unless a test proves a separate bug.

---

# 3.11 Stage-3 impact

Tolerant recovery stays possible on the initial attempt.

Examples that remain governed by current logic:

```text
function close consumed, wrapper truncated
natural stop
tolerant enabled
```

may still commit if existing policy allows.

A later retry attempt must never exist in hardened mode.

This avoids duplicating intent logic inside:

```text
decide_tool_call_recovery()
```

which should remain concerned with recovery safety, not entry intent.

---

# 3.12 R7-01 required tests

Add a dedicated section to:

```text
tests/test_tool_call_parser.cpp
```

Suggested test names:

```text
test_r7_hardened_intent_blocks_stage1_later_base
test_r7_hardened_intent_blocks_prose_after_wrapper_retry
test_r7_hardened_intent_preserves_initial_base_stage2
test_r7_hardened_intent_preserves_initial_base_tolerant_recovery
test_r7_hardened_intent_streaming_invariance
```

---

## R7-01 Test A — valid call + prose + later valid call

Input:

```text
CALL(read)
"\nNow the real action:\n"
CALL(bash)
```

Expected:

### TemplateCompatible

Preserve existing baseline behavior.

If the current baseline chooses the later `bash`, pin that.

### RequireToolAtContentStart

```text
0 tool calls
entire generated bytes returned as Content
```

Diagnostic should reflect the first failed structural region:

```text
TrailingContent
```

or the current exact baseline first-attempt fallback class.

Do not invent a new diagnostic unless needed.

---

## R7-01 Test B — wrapper + prose + later call

Input:

```text
<tool_call>
ordinary prose
<tool_call>valid bash...</tool_call>
```

Expected hardened:

```text
0 calls
verbatim fallback
```

This is the direct `prose_after_wrapper` cross-test missing from Round 6.

---

## R7-01 Test C — broken first marker-like region + later call

Construct a first entry that actually latches, then fails.

Do not use:

```text
<tool_x>
```

because that fails before latch and Round-6 already tests that case.

Use something like:

```text
<tool_call>
bad prose
...
```

or another deterministic post-latch failure.

Expected hardened:

```text
later call never executes
```

---

## R7-01 Test D — initial call requiring Stage 2

Reuse a known Stage-2 fixture that:

```text
starts with the valid initial native marker
requires consistent value-boundary resolution
completes at the same initial entry
```

Expected hardened:

```text
same result as TemplateCompatible
```

This proves the fix did not simply disable Stage 2.

---

## R7-01 Test E — initial tolerant recoverable call

Use a base-zero fixture where:

```text
function close consumed
wrapper tail truncated
tolerant true
```

Expected hardened:

```text
same recovery verdict as before
```

This proves base-zero Stage 3 remains intact.

---

## R7-01 Test F — later-base tolerant recovery

Construct:

```text
failed initial latched region
later function-closed recoverable region
```

Expected:

```text
TemplateCompatible → preserve baseline
RequireToolAtContentStart → 0 calls
```

---

# 3.13 Every-split / streaming matrix

For Tests A–F where applicable:

```text
one-shot
whole-stream decoder feed
1-byte chunks
2-byte chunks
3-byte chunks
5-byte chunks
7-byte chunks
split immediately before second <
split inside second <tool_call>
split at first region break byte
split around wrapper close
```

Compare:

```text
visible_streamed + terminal.content
tool_calls
diagnostics
```

against one-shot.

---

# 3.14 R7-01 acceptance criteria

- [ ] hardened mode cannot add a retry base after the initially latched entry.
- [ ] no Stage-1 later-base execution.
- [ ] no Stage-2 later-base execution.
- [ ] no Stage-3 later-base execution.
- [ ] initial-base Stage 2 still works.
- [ ] initial-base tolerant recovery still works.
- [ ] TemplateCompatible behavior is unchanged.
- [ ] full generated bytes are preserved verbatim on hardened rejection.
- [ ] streaming == one-shot for all new fixtures.
- [ ] Round-1 through Round-6 retry/fence tests remain green.

---

# 4. R7-01b — HIGH coupling bug
## `ToolCallOutputDecoder::finish()` drops `intent_` during terminal re-parse

### 4.1 Verified current code

Constructor correctly stores:

```cpp
intent_(intent)
```

and correctly configures the incremental machine:

```cpp
machine_policy.intent = intent_;
```

But terminal `finish()` calls:

```cpp
parse_qwen_tool_call_output(
    region,
    max_tool_name_length_,
    *contract_,
    tolerant_,
    finish_reason,
    syntax_,
    ambiguity_);
```

The one-shot function's final parameter is:

```cpp
ToolCallIntentPolicy intent =
    ToolCallIntentPolicy::TemplateCompatible
```

Therefore the terminal re-parse silently receives:

```text
TemplateCompatible
```

even when the live decoder was constructed with:

```text
RequireToolAtContentStart
```

---

# 4.2 Why Round-6 tests mostly missed it

For the simple case:

```text
visible prose
→ later marker
```

the incremental machine never latches the marker under hardened mode.

Then terminal re-parse sees no latched region, so the dropped `intent_` does not change the result.

The divergence appears once a region **did latch** and terminal parsing/retry semantics become
intent-sensitive.

That is exactly R7-01.

---

# 4.3 Required fix

Change:

```cpp
ParsedToolCallOutput parsed =
    parse_qwen_tool_call_output(
        region,
        max_tool_name_length_,
        *contract_,
        tolerant_,
        finish_reason,
        syntax_,
        ambiguity_);
```

to:

```cpp
ParsedToolCallOutput parsed =
    parse_qwen_tool_call_output(
        region,
        max_tool_name_length_,
        *contract_,
        tolerant_,
        finish_reason,
        syntax_,
        ambiguity_,
        intent_);
```

No default should be relied on when the decoder already stores the policy.

---

# 4.4 Stronger API recommendation

To prevent another policy-propagation omission, consider grouping parse settings:

```cpp
struct ToolCallExecutionPolicy {
    std::size_t max_name_length;
    bool tolerant;
    ToolCallSyntaxMode syntax;
    ToolCallAmbiguityPolicy ambiguity;
    ToolCallIntentPolicy intent;
};
```

Then both:

```text
ToolCallOutputDecoder
parse_qwen_tool_call_output
```

receive the same policy object.

However, this is optional for Round 7.

Do not expand the refactor if the direct `intent_` propagation plus tests is sufficient.

---

# 4.5 Required regression

Construct a `ToolCallOutputDecoder` with:

```text
RequireToolAtContentStart
```

feed a post-latch retry reproducer, then finish.

Compare exactly with:

```text
parse_qwen_tool_call_output(... RequireToolAtContentStart)
```

Assertions:

```text
same content
same tool_calls
same fallback_reason
same marker_seen
same structured_call_count
same fence diagnostics
same ambiguity diagnostics
```

This specifically catches future dropped-policy parameters.

---

# 5. R7-02 — MEDIUM
## Grammar constraint state is syntax-aware but not intent-aware

### 5.1 Verified current state

`ToolCallGrammarConstraint` stores:

```cpp
ToolCallSyntaxMode syntax_;
```

but no:

```cpp
ToolCallIntentPolicy intent_;
```

and no equivalent text-lock state.

Its inactive behavior explicitly allows ordinary prose and later trigger:

```text
ordinary prose
→ marker candidate
→ complete marker
→ triggered_ = true
```

That matches:

```text
TemplateCompatible
```

only.

---

# 5.2 Divergence example

Future session configuration:

```text
parser intent = RequireToolAtContentStart
constraint    = current ToolCallGrammarConstraint
```

Decoded bytes:

```text
Example:
<tool_call>
...
```

Parser:

```text
"E" published
→ entry_locked = true
→ later marker remains Content
```

Constraint:

```text
"Example:" allowed as ordinary prose
→ later marker triggers grammar mode
```

Therefore:

```text
parser = TEXT
constraint = TOOL
```

This violates the extended policy-consistency invariant.

---

# 5.3 Current severity qualification

This is **not a current production execution bug** because:

```text
--constrained-tool-decoding tool-calls-only
```

is still rejected at startup.

Therefore classify:

```text
MEDIUM architecture / future activation blocker
```

not current HIGH.

Do not delay the R7-01 fix waiting for this.

---

# 5.4 Two acceptable implementation strategies

## Strategy A — implement intent in constraint now

Constructor:

```cpp
explicit ToolCallGrammarConstraint(
    std::size_t max_tool_name_length = 64,
    ToolCallSyntaxMode syntax = ToolCallSyntaxMode::Compatibility,
    ToolCallIntentPolicy intent = ToolCallIntentPolicy::TemplateCompatible);
```

State:

```cpp
ToolCallIntentPolicy intent_;
bool entry_locked_ = false;
```

While inactive:

- formatting whitespace keeps gate open;
- ordinary non-whitespace content sets `entry_locked_`;
- failed marker candidate whose published bytes contain non-whitespace locks;
- complete marker under hardened mode triggers only if `entry_locked_ == false`;
- after a completed tool region, decide explicitly whether another region may retrigger.

This last point matters.

---

## Strategy B — defer until constrained-runtime work, but codify the blocker

Keep the CPU constraint syntax-only for now, but add:

```text
docs/tool_call_parser.md:
runtime constrained decoding MUST NOT be enabled with
RequireToolAtContentStart until the constraint carries the same intent state
```

and an engine-level future assertion when constrained runtime is implemented.

Because runtime constrained decoding is not active, Strategy B is acceptable in Round 7 **if the
documentation and future gate are explicit**.

For maximum architectural completeness, Strategy A is preferred.

---

# 5.5 Important semantic question: multiple consecutive calls

Current parser supports:

```text
<tool_call>...</tool_call>
<tool_call>...</tool_call>
```

as one structured suffix.

`RequireToolAtContentStart` should not lock text between those valid consecutive wrappers.

Therefore the future constraint intent state cannot simply do:

```text
tool region complete
→ LockedText
```

It must distinguish:

```text
ToolSequenceOpen
```

from:

```text
ordinary text after tool sequence
```

or rely on the parser's top-level legal continuation logic.

Recommended model:

```text
GateOpen
    whitespace only
    ↓
ToolActive
    first wrapper triggered
    ↓
ToolSequence
    canonical consecutive wrapper may follow
    ordinary visible content ends tool eligibility
```

Do not implement a simplistic one-call-only gate.

---

# 5.6 Constraint checkpoint/restore

If intent state is added now, it must be value-semantic through:

```text
checkpoint()
restore()
```

Tests must prove rollback restores:

```text
intent_
entry_locked_
triggered_
marker_prefix_
buffer_
```

A speculative draft that temporarily emits prose must not permanently text-lock the restored state
after rejection.

---

# 5.7 R7-02 tests if implemented now

Add to:

```text
tests/test_tool_call_grammar_state.cpp
```

Matrix:

### TemplateCompatible

```text
"Example:" + marker
→ marker may trigger
```

### RequireToolAtContentStart

```text
"Example:" + marker
→ never active
```

### Whitespace

```text
"\n \t" + marker
→ trigger
```

### Failed marker candidate

```text
"<tool_x>" + marker
→ hardened state locked
```

### Checkpoint

```text
checkpoint GateOpen
commit prose → locked
restore
commit marker → active
```

### Consecutive wrappers

Pin intended behavior.

### Native vs Compatibility

Cross product:

```text
intent:
  template-compatible
  start-of-content

syntax:
  native
  compat
```

---

# 5.8 R7-02 acceptance criteria

If implemented:

- [ ] constraint stores intent.
- [ ] hardened constraint locks after ordinary visible content.
- [ ] formatting whitespace does not lock.
- [ ] failed marker candidate locks hardened mode.
- [ ] syntax and intent compose correctly.
- [ ] checkpoint/restore preserves intent state.
- [ ] consecutive valid call behavior matches parser.
- [ ] parser/constraint cross-check includes intent dimension.

If deferred:

- [ ] docs explicitly name this as a blocker.
- [ ] runtime constrained decoding stays fail-fast.
- [ ] future implementation checklist contains intent parity before activation.

---

# 6. R7-03 — MEDIUM
## Hardened template instruction is unconditional while runtime hardening is opt-in

### 6.1 Verified current local template

Current:

```text
tools/chat_templates/qwen3_8.jinja
```

always tells the model, when tools are available:

```text
If you call a function, emit the <tool_call> block immediately in the assistant content channel.
Do not emit visible natural-language content before or after the tool call.
```

It labels this an NInfer hardening extension.

However:

```text
ServeOptions.tool_call_intent
```

defaults to:

```text
TemplateCompatible
```

and the template does not receive this serving option.

Therefore the runtime can be:

```text
template-compatible
```

while the selected local prompt is:

```text
start-of-content style
```

---

# 6.2 Why this matters

The Progress MD currently says:

```text
default behavior is unchanged
```

at the runtime parser policy level.

But for users who select:

```text
--chat-template tools/chat_templates/qwen3_8.jinja
```

the prompt behavior **has** changed relative to the upstream template.

The model is now asked not to emit a visible preamble even when runtime parser policy would allow it.

This can change:

- model output distribution;
- tool-call success rate;
- visible assistant preambles;
- malformed-call frequency;
- coding-agent behavior.

That is a behavioral change even though the parser default remains compatible.

---

# 6.3 Correct design options

## Option A — two explicit template files

Recommended for simplicity:

```text
tools/chat_templates/qwen3_8.jinja
tools/chat_templates/qwen3_8_hardened.jinja
```

`qwen3_8.jinja`:

```text
upstream-compatible preamble wording
```

`qwen3_8_hardened.jinja`:

```text
no visible content before/after tool call
reason in reasoning channel instead
```

Then users opt in by selecting both:

```text
--tool-call-intent start-of-content
--chat-template tools/chat_templates/qwen3_8_hardened.jinja
```

Document recommended pairing.

Advantages:

- obvious;
- no hidden Jinja variable;
- low integration risk;
- easy A/B benchmark;
- explicit prompt provenance.

---

## Option B — pass intent policy into Jinja context

Add an internal render value such as:

```text
ninfer_tool_call_intent
```

with:

```text
template-compatible
start-of-content
```

and render different wording conditionally.

Required propagation:

```text
ServeOptions
→ RequestOptions / PromptInput
→ ChatRenderOptions
→ template_parameters()
→ Jinja context
```

Then:

```jinja
{% if ninfer_tool_call_intent == "start-of-content" %}
  hardened instruction
{% else %}
  upstream-compatible instruction
{% endif %}
```

This keeps one file but touches more layers.

---

# 6.4 Recommendation

For Round 7, prefer **Option A** unless there is already a clean per-request template-policy mechanism.

Reason:

```text
R7 is a correctness round
```

not a prompt-config plumbing refactor.

A second template makes behavior explicit and minimizes new coupling.

---

# 6.5 Critical naming rule

Do not call the hardened template:

```text
qwen3_8.jinja
```

while moving upstream-compatible behavior elsewhere.

Use names that communicate intent:

```text
qwen3_8.jinja
qwen3_8_hardened_tools.jinja
```

or equivalent.

Existing users should not silently inherit a stricter prompt unless that is an intentional migration.

---

# 6.6 Template test requirements

For upstream-compatible template:

assert rendered tool instructions contain the equivalent of:

```text
optional natural-language reasoning BEFORE the function call
```

For hardened template:

assert they contain:

```text
emit tool block immediately
no visible natural-language content before or after
```

Also assert both templates:

- compile in C++ renderer;
- compile in Python reference test;
- preserve reasoning-effort instructions;
- preserve tools JSON;
- preserve system/developer message placement;
- preserve assistant generation suffix;
- preserve tool-history rendering;
- preserve `NO suffix`.

---

# 6.7 Server/docs requirements

Serving docs must explicitly distinguish:

```text
parser intent policy
prompt template policy
```

Recommended wording:

```text
--tool-call-intent controls parser execution semantics.
The hardened Qwen template controls model prompting.
For the strictest OMP profile, enable both.
Using only one is valid but intentionally asymmetric.
```

This is more accurate than implying the serve flag automatically changes the template.

---

# 6.8 R7-03 acceptance criteria

- [ ] upstream-compatible local template exists.
- [ ] hardened template exists or intent is passed to Jinja explicitly.
- [ ] no unconditional hardened wording is presented as default-compatible behavior.
- [ ] tests render both modes.
- [ ] docs state parser policy vs prompt policy separately.
- [ ] Round-6 progress wording is corrected if necessary.
- [ ] no parser behavior changes from this template cleanup.

---

# 7. R7-04 — LOW / VERIFY
## Round-6 full CPU gate is reported inconsistently

### 7.1 Verified Round-6 progress record

The document states:

```text
147 total: 143 passed, 4 failed
```

with failures:

```text
ninfer_linear_swiglu_q4_a16_test
ninfer_linear_swiglu_q8_a16_test
ninfer_linear_swiglu_nvfp4_test
ninfer_linear_swiglu_fp8_test
```

and also states:

```text
full CPU gate green
```

and:

```text
GPU-dependent tests report runtime SKIP
(counted as passed via SKIP_RETURN_CODE)
```

Those statements are mutually inconsistent.

---

# 7.2 Verified skip propagation bug

CTest registration helper:

```cmake
function(ninfer_add_op_test name)
  ...
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()
```

LinearSwiGLU support:

```cpp
if (!cuda_available()) {
    std::cout << "SKIP: no usable CUDA device\n";
    return 77;
}
```

But wrapper `main()` does:

```cpp
const int failures = run_profile(...);
return failures == 0 ? 0 : 1;
```

Therefore:

```text
run_profile returns 77
→ main converts 77 to 1
→ CTest sees 1
→ test is FAILED, not SKIPPED
```

The console word `SKIP` is not semantically relevant to CTest.

---

# 7.3 Mandatory documentation fix

Change Round-6 progress to something equivalent to:

```text
Full -E "_real" CTest command was not globally green:
143 passed, 4 failed.

All parser/frontend/serve/template CPU targets relevant to this change passed.
The four remaining failures are GPU-dependent LinearSwiGLU tests. They print
"SKIP: no usable CUDA device", but their main() functions translate the helper's
exit code 77 to exit code 1, so CTest correctly records them as failures despite
the intended SKIP_RETURN_CODE configuration.
```

Do not write:

```text
full CPU gate green
```

until the CTest process exits successfully.

---

# 7.4 Recommended test-wrapper fix

Each LinearSwiGLU `main()` should preserve `77`.

For example:

```cpp
const int result = run_profile(...);

if (result == 77) {
    return 77;
}

std::cout << (result == 0 ? "OK" : "FAIL") << "...";
return result == 0 ? 0 : 1;
```

Better: avoid overloading the variable name `failures` if it can carry `77`.

Use:

```cpp
const int result = run_profile(...);
```

Apply consistently to:

```text
test_q4_a16.cpp
test_q8_a16.cpp
test_nvfp4.cpp
test_fp8.cpp
```

Check whether any other op-test helper returns `77` through a wrapper that converts nonzero to `1`.

Do a repository audit.

---

# 7.5 Alternative cleaner API

If desired later:

```cpp
enum class TestRunResult {
    Pass,
    Fail,
    Skip,
};
```

or helper:

```cpp
int propagate_test_result(int result) {
    if (result == 77) return 77;
    return result == 0 ? 0 : 1;
}
```

For Round 7, explicit preservation is enough.

---

# 7.6 R7-04 tests

No GPU execution required.

Possible CPU-only test methods:

### Static/source test

Verify each LinearSwiGLU wrapper contains a `77` propagation path.

### CTest smoke helper

Add a tiny CPU-only test executable returning `77` under the same registration helper and verify
CTest classifies it as skipped.

This validates infrastructure semantics without touching CUDA.

---

# 7.7 R7-04 acceptance criteria

- [ ] Round-6 progress no longer calls a 4-failure run green.
- [ ] skip wording matches actual CTest semantics.
- [ ] LinearSwiGLU wrapper preserves 77, if this optional code fix is included.
- [ ] no CUDA test is executed for this Round-7 task.
- [ ] a subsequent GPU-hidden CTest run reports those tests as skipped rather than failed, or the
      report explicitly records them as expected environmental failures without calling the gate green.

---

# 8. Known residual that Round 7 must NOT misclassify

## R6-04 — byte-identical call-only quotation

This remains:

```text
<tool_call>
<function=bash>
<parameter=command>
echo example
</parameter>
</function>
</tool_call>
```

If the model intends those exact bytes as:

```text
show an example
```

rather than:

```text
execute a call
```

the parser has no semantic evidence in the bytes themselves.

`RequireToolAtContentStart` does not solve that case because the tool block already starts at content
start.

`FailClosed` does not solve it because a clean Stage-1 parse never enters the structural ambiguity
policy.

Do not regress the documentation into claiming otherwise.

Long-term solution remains:

```text
explicit TEXT-vs-TOOL intent signal
```

outside ordinary quotable content bytes.

---

# 9. Constrained decoding interaction

Round 7 does not enable runtime constrained sampling.

Keep startup fail-fast.

The correct future ordering is:

```text
1. decide TEXT vs TOOL intent
2. if TOOL, enable grammar-constrained generation
3. validate with parser
4. validate declared tool
5. normalize arguments
6. serialize tool call
```

Do not use constrained decoding as a substitute for intent.

A syntactically perfect quoted call remains syntactically perfect.

---

# 10. Files expected to change

Mandatory R7-01/R7-01b:

```text
src/models/qwen3_5/frontend/tool_call_stream.cpp
src/models/qwen3_5/frontend/tool_call_parser.cpp
tests/test_tool_call_parser.cpp
possibly tests/models/qwen3_5/test_frontend.cpp
docs/tool_call_parser.md
docs/NInfer_new_parser_design_round7_bugfix_progress.md
```

R7-02 if implemented now:

```text
src/models/qwen3_5/frontend/tool_call_grammar_state.h
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp
tests/test_tool_call_grammar_state.cpp
docs/tool_call_parser.md
```

R7-03:

```text
tools/chat_templates/qwen3_8.jinja
tools/chat_templates/qwen3_8_hardened_tools.jinja   # recommended
tests/text/test_chat_templates.py
docs/serving.md
```

R7-04:

```text
docs/NInfer_new_parser_design_round6_bugfix_progress.md
tests/ops/linear_swiglu/test_q4_a16.cpp
tests/ops/linear_swiglu/test_q8_a16.cpp
tests/ops/linear_swiglu/test_nvfp4.cpp
tests/ops/linear_swiglu/test_fp8.cpp
```

Do not edit unrelated model/runtime code.

---

# 11. Implementation phases

## Phase R7-0 — baseline freeze

Record:

```text
start SHA
worktree status
compiler
CMake generator
build dir
GPU hidden environment
```

Run targeted existing tests first.

Required baseline set:

```text
ninfer_tool_call_parser_test
ninfer_tool_call_grammar_test
ninfer_tool_call_grammar_state_test
ninfer_qwen3_5_frontend_test
ninfer_serve_options_test
ninfer_request_log_test
ninfer_chat_templates_test
ninfer_engine_options_test if available in this build
```

Record PASS/SKIP/FAIL exactly.

---

## Phase R7-1 — add failing R7-01 regressions FIRST

Before production code change, add tests for:

```text
latched first region
→ failure
→ later valid native wrapper
```

under both:

```text
TemplateCompatible
RequireToolAtContentStart
```

The hardened assertions must fail on the baseline.

This proves the patch addresses a real behavior, not a hypothetical.

Do not merge the production fix before the red test exists.

---

## Phase R7-2 — restrict later retry bases in hardened mode

Implement the minimal semantic fix in:

```text
ToolCallStreamParser::finish()
```

Goal:

```text
RequireToolAtContentStart
→ initial latched entry only
```

Do not rewrite Stage 2.

Run parser tests.

---

## Phase R7-3 — propagate `intent_` into terminal re-parse

Fix:

```text
ToolCallOutputDecoder::finish()
```

Add direct one-shot vs decoder parity test.

Run parser + frontend tests.

---

## Phase R7-4 — grammar constraint decision

Choose explicitly:

```text
A: implement intent parity now
B: document as blocker until constrained-runtime implementation
```

If A:

- implement state;
- add checkpoint/restore matrix;
- add parser/constraint cross-check with intent dimension.

If B:

- leave code untouched;
- add strong docs;
- keep constrained runtime fail-fast.

Do not leave the decision implicit.

---

## Phase R7-5 — template/runtime policy coherence

Recommended:

```text
restore qwen3_8.jinja to upstream-compatible tool preamble wording
add qwen3_8_hardened_tools.jinja
```

Update template tests and serving docs.

Do not change parser defaults here.

---

## Phase R7-6 — verification-report correction

Correct Round-6 progress.

Optionally fix LinearSwiGLU exit-code propagation.

Run CPU-hidden CTest and record actual outcome.

---

## Phase R7-7 — full review loop

Repeat:

```text
A. intent × retry
B. intent × Stage 2
C. intent × tolerant Stage 3
D. intent × streaming
E. intent × reasoning/content split
F. syntax × intent
G. fence × retry
H. template × runtime policy
I. finish reason × recovery
J. declared tools × normalization
K. API serialization
L. test gate semantics
```

Continue until no reproducible HIGH/MEDIUM implementation defect remains in scope.

---

# 12. Detailed R7-01 regression corpus

Use at least two declared tools:

```text
read(path)
bash(command)
```

Helper:

```text
R = valid read call
B = valid bash call
```

Corpus:

```text
C1 = R + "\nprose\n" + B
C2 = "<tool_call>\nprose\n" + B
C3 = malformed-latched-native-region + "\n" + B
C4 = initial base requiring Stage2 only
C5 = initial base tolerant-recoverable only
C6 = initial failed region + later tolerant-recoverable B
C7 = leading whitespace + C1
C8 = reasoning channel text + Content C1
C9 = fenced B inside first region + later real B
C10 = multiple valid consecutive wrappers at initial structured suffix
```

Expected matrix:

| Corpus | TemplateCompatible | StartOfContent |
|---|---|---|
| C1 | baseline later-call behavior | **0 calls** |
| C2 | baseline retry behavior | **0 calls** |
| C3 | baseline retry behavior | **0 calls** |
| C4 | accept/reject per existing Stage2 policy | **same** |
| C5 | recover per existing tolerant policy | **same** |
| C6 | baseline later recovery | **0 calls** |
| C7 | same as C1 except leading whitespace | **0 calls** |
| C8 | reasoning does not count; Content semantics same as C1 | **0 calls** |
| C9 | fence rules preserved | no later-base bypass |
| C10 | valid consecutive structured calls | **preserve intended support** |

For C10, inspect actual current parser semantics before finalizing expected count. Do not assume.

---

# 13. FinishReason matrix

For every malformed/retry fixture where finish reason can affect recovery:

```text
StopToken
StopString
OutputLimit
ContextCapacity
Cancelled
None (low-level explicit only)
```

R7-01 intent restriction is orthogonal to FinishReason:

```text
later base forbidden in hardened mode
```

regardless of terminal reason.

Base-zero tolerant recovery remains finish-reason-sensitive exactly as before.

---

# 14. Syntax × intent matrix

Run:

```text
syntax:
  QwenWrappedNative
  Compatibility

intent:
  TemplateCompatible
  RequireToolAtContentStart
```

Examples:

### Native

```text
prose + <function=read>...
```

must remain text because bare function is not a native top-level entry.

### Compatibility

Bare entry may be eligible under TemplateCompatible.

Under hardened mode:

```text
visible prose before bare entry
→ no execution
```

If R7-02 constraint intent is implemented, cross-check constraint behavior too.

---

# 15. Fence × intent × retry matrix

Test:

```text
fenced first marker
later real call
```

and:

```text
first latched malformed region
fenced later marker
real later marker
```

Rules:

- fenced marker never becomes an executable retry base;
- hardened mode never accepts any later base anyway;
- TemplateCompatible preserves current fence-aware retry behavior.

This ensures the R7-01 cutoff does not accidentally break fence diagnostics.

---

# 16. OutputSession / reasoning integration

Keep the existing Round-6 test:

```text
Reasoning channel:
    I should inspect...

Content:
    <tool_call>...
```

Expected hardened:

```text
call executes
```

Add:

```text
Reasoning:
    I should inspect...

Content:
    first latched malformed region
    visible recovered prefix
    later valid call
```

Expected hardened:

```text
0 calls
```

Reasoning remains irrelevant to the content-start gate.

---

# 17. API serialization checks

If R7-01 is broken, public APIs can serialize:

```text
assistant text
+
tool call
```

after the hardening contract should have rejected the call.

After the fix, add at least one integration-level assertion that the hardened rejected retry case
produces:

### OpenAI Chat

```text
message.content = verbatim text
message.tool_calls absent/empty
finish_reason not "tool_calls"
```

### OpenAI Responses

```text
output_text item
no function_call item
```

### Anthropic

```text
text content block
no tool_use block
```

Use existing response unit-test helpers if available.

Do not add full server/model runtime tests.

---

# 18. Diagnostics

Avoid inventing new diagnostics if existing ones describe the first failed region correctly.

For hardened later-base suppression, acceptable behavior is:

```text
fallback_reason = first attempt's structural reason
```

Examples:

```text
TrailingContent
MalformedStructure
InvalidToolName
...
```

The policy reason itself could be observable later, but Round 7 does not require a new enum.

If a new diagnostic is added, it must be propagated consistently to:

```text
request log
operational log
OpenAI/Anthropic internal outcome
tests
docs
```

That expansion is not necessary for the bugfix.

---

# 19. Template implementation detail

Recommended files:

```text
tools/chat_templates/qwen3_8.jinja
tools/chat_templates/qwen3_8_hardened_tools.jinja
```

Upstream-compatible line:

```text
You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
```

Hardened line:

```text
If you call a function, emit the <tool_call> block immediately in the assistant content channel.
Do not emit visible natural-language content before or after the tool call.
Reason internally in the model reasoning channel instead.
```

Keep all other behavior byte-equivalent unless intentionally changed.

Diff the rendered prompt for a fixed fixture and verify the only semantic difference is the intended
instruction line.

---

# 20. Progress-document truthfulness

Create:

```text
docs/NInfer_new_parser_design_round7_bugfix_progress.md
```

Never pre-mark a finding fixed.

Use states:

```text
OPEN
IMPLEMENTED
TESTED
VERIFIED
DEFERRED
RESIDUAL
```

A finding becomes `VERIFIED` only after its dedicated reproducer and regression suite pass.

---

# 21. Suggested commit sequence

Recommended small commits:

```text
1. test(frontend): reproduce hardened intent retry bypass
2. fix(frontend): forbid later retry bases in start-of-content mode
3. fix(frontend): preserve intent policy in decoder terminal reparse
4. test(frontend): cover retry/stage2/stage3 intent cross-matrix
5. test(frontend): cover OutputSession reasoning plus hardened retry rejection
6. fix/frontend or docs: make grammar constraint intent policy coherent
7. templates: split qwen3.8 compatible and hardened tool instructions
8. test(templates): cover compatible vs hardened rendering
9. docs(frontend): document round-7 intent/retry invariants
10. docs(test): correct round-6 CTest gate wording
11. test(ops): preserve LinearSwiGLU skip code 77
12. docs: record round-7 verification
```

Do not squash before review.

---

# 22. CPU-only verification commands

Build:

```powershell
cmake --build build-new-parser --config Release --parallel 16
```

Targeted:

```powershell
ctest --test-dir build-new-parser -C Release `
  -R "ninfer_(tool_call_parser|tool_call_grammar|tool_call_grammar_state|qwen3_5_frontend|serve_options|request_log|chat_templates|engine_options)_test" `
  --output-on-failure `
  --parallel 16
```

If `chat_templates` has a different registered test name, use the actual CTest listing.

Full GPU-hidden gate:

```powershell
$env:CUDA_VISIBLE_DEVICES="99"

ctest --test-dir build-new-parser -C Release `
  -E "_real" `
  --output-on-failure `
  --parallel 16
```

Record:

```text
Total Test time
total tests
passed
failed
skipped
not run
CTest process exit code
```

Do not summarize a nonzero CTest exit code as green.

---

# 23. Deterministic outcome dump

Reuse the Round-5/6 corpus.

Add dimensions:

```text
syntax
ambiguity
intent
tolerant
finish_reason
stream_partition
```

New expected delta after R7-01:

```text
ONLY:
RequireToolAtContentStart
AND
accepted base would be later than the originally latched entry
```

Those cases change from structured call to text fallback.

Unexpected deltas:

```text
TemplateCompatible
base-zero canonical call
base-zero Stage2 repair
base-zero tolerant recovery
fenced examples
declared-name enforcement
argument normalization
```

must be investigated.

---

# 24. Review loop A — intent ownership

Trace:

```text
Content bytes
→ publish
→ entry_locked
→ marker candidate
→ latch
→ region
→ Stage1 attempt
→ retry eligibility
→ Stage2 bases
→ Stage3 bases
→ result.tail
→ parsed.content
→ GeneratedToolCall
```

At every acceptance point ask:

```text
Would any bytes before this accepted call be returned as visible Content?
```

If yes under hardened mode:

```text
reject
```

---

# 25. Review loop B — policy propagation

Search repository for:

```text
ToolCallSyntaxMode
ToolCallAmbiguityPolicy
ToolCallIntentPolicy
tolerant_tool_calls
parse_qwen_tool_call_output(
ToolCallOutputDecoder(
ToolCallParsePolicy
```

For every constructor/call:

```text
which policy value enters?
is it explicit?
is a default silently substituted?
```

The Round-7 fix must close the known `intent_` omission.

---

# 26. Review loop C — one-shot vs streaming

For every new reproducer:

```text
one-shot parse
```

must equal:

```text
ToolCallOutputDecoder feed chunks + finish
```

in:

```text
content
call count
call names
arguments
fallback reason
marker_seen
structured_call_count
fence diagnostics
ambiguity diagnostics
```

No partial equality.

---

# 27. Review loop D — template vs runtime

For each template profile:

```text
render prompt with tools
```

and write down:

```text
what does the model get instructed to do?
```

Compare with:

```text
what does runtime parser permit?
```

Document intentional asymmetry.

No hidden mismatch.

---

# 28. Review loop E — constraint future-proofing

If R7-02 implemented:

```text
parser entry verdict
constraint trigger verdict
```

must agree for:

```text
whitespace
ordinary prose
failed marker
native wrapper
compat-only marker
second consecutive wrapper
```

under both intent modes.

If R7-02 deferred:

verify:

```text
constrained runtime remains impossible to enable
```

and docs name the intent-parity prerequisite.

---

# 29. Review loop F — previous findings regression

Re-run direct fixtures for:

```text
F1 literal <parameter> in payload
F2 open function never executes
F3 EOF parameter ambiguity
F4 partial next opener
F5 wrapper balance
F6 mixed parameter families
F8 breaking '<'
F9 removed depth overflow
CR1 nested failed wrapper
CR2 function_calls missing close
CR3 trailing content
CR4 continuation consistency
declared tool enforcement
fence guard
FinishReason transaction
FailClosed ambiguity
R6 intent pre-latch gate
R6 grammar syntax mode
```

Round 7 must not reopen any of them.

---

# 30. R7-01 proof obligation

After implementation, the following statement should be mechanically supportable:

> In `RequireToolAtContentStart`, once the first native tool entry has latched, the parser may repair
> or recover only that same entry. It cannot select a later top-level marker as the executable
> entry. Therefore every committed structured call begins at the only entry point that was
> admitted before visible Content was committed.

The proof should be visible in code structure:

```text
no later retry bases enter chain
```

not only in comments.

---

# 31. R7-02 proof obligation

If implemented:

> For any inactive prefix, parser and constraint share both syntax and intent policies. If ordinary
> Content locks the parser out of tool entry, the constraint cannot later enter tool grammar on the
> same bytes.

Prove via cross-product test.

---

# 32. R7-03 proof obligation

> Selecting the upstream-compatible local template does not silently inject the stricter
> start-of-content instruction. Selecting the hardened template does. Runtime parser policy remains
> independently explicit.

Prove via rendered prompt string assertions.

---

# 33. R7-04 proof obligation

> A test intended to skip with code 77 reaches CTest as 77; a failed test reaches CTest as nonzero
> non-skip. Progress documentation reports the CTest result exactly.

Prove with wrapper code and CTest output.

---

# 34. Definition of Done — mandatory P0

Round 7 is **not green** until all of these hold:

- [ ] failing baseline test reproduces R7-01.
- [ ] hardened mode cannot search/accept later retry entry.
- [ ] TemplateCompatible retains historical later-retry behavior.
- [ ] initial-entry Stage 2 remains functional.
- [ ] initial-entry tolerant Stage 3 remains functional.
- [ ] `ToolCallOutputDecoder::finish()` forwards `intent_`.
- [ ] one-shot == streaming for R7-01 corpus.
- [ ] reasoning-before-tool legitimate hardened flow remains green.
- [ ] visible Content before call remains non-executable in hardened mode.
- [ ] entire rejected region is restored verbatim.
- [ ] API serialization contains no tool item for hardened rejected retry.
- [ ] all previous parser regression suites pass.

---

# 35. Definition of Done — R7-02

Either:

### Implemented now

- [ ] constraint stores intent policy.
- [ ] constraint locks after visible prose under hardened mode.
- [ ] whitespace does not lock.
- [ ] failed candidate locks.
- [ ] checkpoint/restore carries intent state.
- [ ] syntax × intent cross-matrix matches parser.
- [ ] consecutive wrapper semantics match parser.

or:

### Explicitly deferred

- [ ] no runtime constrained decoding can be enabled.
- [ ] docs identify intent parity as activation prerequisite.
- [ ] progress marks R7-02 `DEFERRED`, not `FIXED`.

---

# 36. Definition of Done — R7-03

- [ ] prompt hardening is no longer unconditional under a file/profile presented as compatible.
- [ ] compatible and hardened templates/policies are distinguishable.
- [ ] template tests cover both.
- [ ] docs explain recommended OMP hardened pairing.
- [ ] no accidental parser default change.

---

# 37. Definition of Done — R7-04

- [ ] Round-6 progress wording corrected.
- [ ] full test gate is never called green when CTest reports failures.
- [ ] if LinearSwiGLU wrapper fix included: code 77 propagates.
- [ ] subsequent GPU-hidden run records those tests as skipped or honestly failed.
- [ ] no GPU/model execution performed by this task.

---

# 38. Final sign-off wording if everything above is green

Acceptable:

> No remaining HIGH/MEDIUM implementation defect was found in the reviewed canonical Qwen3.8
> parser/integration path under the tested CPU-only scope. `RequireToolAtContentStart` now applies
> to retry/recovery entry selection as well as initial marker latch, and one-shot/streaming paths
> carry the same intent policy. The complete call-only quotation residual remains a documented
> semantic limitation of the native byte protocol. Runtime grammar-constrained decoding remains
> unavailable and fail-fast.

Do **not** write:

```text
fehlerfrei
all tool-call errors impossible
phantom calls impossible
semantic intent fully solved
```

Absolute absence of bugs cannot be proved by this review.

---

# 39. Recommended hardened OMP/NInfer profile after R7-01/R7-03

For an automated coding agent prioritizing integrity:

```text
--tool-call-syntax qwen-wrapped
--tool-call-ambiguity fail-closed
--tool-call-intent start-of-content
(no --tolerant-tool-calls initially)
--chat-template tools/chat_templates/qwen3_8_hardened_tools.jinja
```

This intentionally trades some availability for stricter execution semantics.

Measure:

```text
valid tool-call success
missed call due to visible preamble
malformed call rate
retry/handoff rate
coding benchmark success
phantom-call rate
```

Do not infer the best production tradeoff from parser tests alone.

---

# 40. Non-goals

Do not implement in Round 7 unless a new concrete reproducer demands it:

- Stage-2 algorithm rewrite;
- new delimiter grammar;
- JSON-only custom tool wire format;
- tokenizer special intent token;
- runtime sampler masking;
- GPU constrained decoding;
- model fine-tuning;
- API protocol redesign;
- broad template architecture refactor beyond compatible/hardened separation.

---

# 41. Suggested Round-7 progress structure

Use companion:

```text
NInfer_new_parser_design_round7_bugfix_progress.md
```

Sections:

```text
Baseline
Finding table
Red repro evidence
Implementation commits
Per-finding tests
Outcome deltas
Full CPU gate
Known residuals
CI status
Final review loops
Sign-off
```

The companion file provided with this specification contains a ready-to-fill checklist.

---

# 42. Evidence references

## Reviewed NInfer baseline

```text
https://github.com/Hundsbuah/ninfer/tree/4a69b7a31b6570bfd7fce220a822492b24a161e4
```

Primary affected files:

```text
src/models/qwen3_5/frontend/tool_call_stream.cpp
src/models/qwen3_5/frontend/tool_call_stream.h
src/models/qwen3_5/frontend/tool_call_parser.cpp
src/models/qwen3_5/frontend/tool_call_parser.h
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp
src/models/qwen3_5/frontend/tool_call_grammar_state.h
src/models/qwen3_5/frontend/output_session.cpp
include/ninfer/types.h
tools/chat_templates/qwen3_8.jinja
docs/tool_call_parser.md
docs/NInfer_new_parser_design_round6_bugfix_progress.md
tests/test_tool_call_parser.cpp
tests/test_tool_call_grammar_state.cpp
tests/models/qwen3_5/test_frontend.cpp
tests/cmake/NinferTests.cmake
tests/ops/linear_swiglu/*
```

## Official Qwen3.8 template

```text
https://huggingface.co/Qwen/Qwen3.8-27B/blame/main/chat_template.jinja
```

## Qwen function-calling documentation

```text
https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md
```

## External parser corroboration

```text
https://github.com/vllm-project/vllm/issues/57541
```

---

# 43. Final engineering principle

The Round-7 bug is not evidence that the parser needs another broad rewrite.

It is evidence that a security/integrity policy must be enforced at **every path that can produce an
executable call**.

The correct mental model is:

```text
entry intent
    ↓
allowed executable base set
    ↓
structural parser / recovery
    ↓
contract validation
    ↓
tool call
```

not:

```text
initial marker gate
    ↓
later recovery may choose anything
```

Once `RequireToolAtContentStart` restricts the **accepted base set**, its semantics become coherent
with Stage 1, Stage 2, Stage 3, streaming and API serialization.

That is the smallest, clearest, and most defensible Round-7 correction.
