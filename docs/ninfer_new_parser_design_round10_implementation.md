# NInfer `new_parser_design` — Round 10 Bugfix / Integration Specification

## 0. Purpose

This document is the **Round-10 implementation specification** for the remaining tool-call findings found after the full-chain review of:

```text
Hundsbuah/ninfer:new_parser_design
```

Pinned reviewed HEAD:

```text
916918973ba054fb7b8ec5fa82fa1064a4effcbe
```

The review covered the complete affected chain, not only the latest commits:

```text
Qwen native template
→ OutputSession
→ ToolCallOutputDecoder
→ pre-latch scanner
→ marker classification
→ Stage-1 greedy parser
→ Stage-2 consistent completion
→ Stage-3 tolerant recovery
→ declared-tool contract
→ parameter normalization
→ GeneratedToolCall
→ OpenAI Chat
→ OpenAI Responses
→ Anthropic
→ ToolCallGrammarConstraint
→ diagnostics / request logging
→ tests
```

The central Stage-1 / Stage-2 / Stage-3 parser redesign is considered **green for its current scope**. Round 10 must therefore **not reopen or redesign the parser core unless a new direct parser-core reproducer proves that necessary**.

Round 10 addresses the remaining boundary between:

```text
ordinary/literal assistant content
```

and:

```text
executable top-level tool-call markup
```

Specifically:

```text
R10-01  Markdown-style indented literal/code lines can still latch as executable tool calls.
R10-02  Production pre-latch parsing and ToolCallGrammarConstraint still use different
         pre-trigger classifiers.
R10-03  Indentation-based suppression would otherwise be operationally silent.
```

The objective is to close these remaining integration gaps **without changing the already-correct Stage-1/2/3 structural parser**.

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

Reviewed HEAD:

```text
916918973ba054fb7b8ec5fa82fa1064a4effcbe
```

At review time:

```text
GitHub combined status checks: none
GitHub workflow runs: none
```

Therefore repository-reported or locally run tests must not be represented as independent GitHub CI verification.

Before implementing Round 10:

1. verify that `new_parser_design` still points to the pinned SHA;
2. if HEAD changed, inspect every changed file intersecting this specification before applying it;
3. do not blindly apply line-number-specific edits against a different tree.

---

# 0.2 Current verified state

The following parts are **not Round-10 targets** unless a new direct reproducer proves otherwise.

### Source-verified green

```text
QwenWrappedNative top-level syntax
Stage-1 greedy parser
Stage-2 consistent completion
FailClosed ambiguity handling
Stage-3 recovery
function-close executability boundary
FinishReason propagation
one-shot / streaming terminal reparse
declared-tool identity enforcement
duplicate-parameter ambiguity handling
OpenAI Chat Round-9 strict/custom rejection
OpenAI Responses path
Anthropic Round-9 forced/serial/strict rejection
post-call ToolCallGrammarConstraint state machine
```

Production defaults currently include:

```cpp
ToolCallSyntaxMode tool_call_syntax =
    ToolCallSyntaxMode::QwenWrappedNative;

ToolCallAmbiguityPolicy tool_call_ambiguity =
    ToolCallAmbiguityPolicy::FailClosed;

ToolCallIntentPolicy tool_call_intent =
    ToolCallIntentPolicy::TemplateCompatible;
```

Do **not** change these defaults in Round 10.

In particular:

```text
TemplateCompatible
```

must remain the compatibility/default intent policy.

`RequireToolAtContentStart` remains an optional hardening policy.

---

# 0.3 Existing documented protocol limitation

Round 10 must **not attempt to solve** the already documented semantic quotation residual:

```text
<tool_call>
...
</tool_call>
```

when the bytes are:

```text
complete
canonical
declared
unfenced
structurally valid
```

and are byte-identical to an intended real tool call.

The parser cannot infer author intent from identical bytes.

Existing fixture:

```text
test_r6_complete_unfenced_in_set_example_residual
```

must remain.

Do not introduce semantic heuristics such as:

```text
"Example:"
"Here is the syntax:"
"for example"
"do not execute"
```

into the parser.

That would create language-dependent behavior and would still not solve the byte-identical case.

---

# 0.4 Hard scope boundaries

Round 10 must not opportunistically change:

```text
Stage-1 value-boundary selection
Stage-2 Balanced/Lazy algorithm
Stage-2 work budget
Stage-2 dead-state memo
Stage-3 recovery policy
parameter normalization
declared-tool enforcement
FailClosed semantics
function executability boundary
FinishReason policy
OpenAI tool schemas
Anthropic tool schemas
Qwen chat template
tolerant-tool-call policy
post-call BetweenCalls semantics
sampling
speculative decoding
MTP/DFlash2
runtime constrained sampling activation
```

Do not change:

```text
missing </function> => non-executable
```

That invariant is correct and safety-relevant.

Round 10 is a **pre-latch classification/integration round**.

---

# 1. Finding index

| ID | Finding | Severity | Exposure | Required action |
|---|---|---:|---|---|
| R10-01 | ≥4-column indented tool markup can latch as executable markup | HIGH integrity | active | FIX |
| R10-01a | leading tabs are not treated as literal indentation | HIGH integrity | active | FIX |
| R10-01b | `RequireToolAtContentStart` does not fix whitespace-only indentation | root cause | active | FIX in literal classifier |
| R10-02 | parser and `ToolCallGrammarConstraint` have different pre-trigger machines | MEDIUM architecture | parser active / constraint future-facing | UNIFY |
| R10-02a | Constraint has no fence-aware pre-trigger classification | MEDIUM architecture | dormant until constrained sampling | FIX architecture |
| R10-02b | Future fixes could drift because pre-trigger logic is duplicated | regression risk | ongoing | REMOVE duplication |
| R10-03 | suppressed indented markers would otherwise be silent | MEDIUM observability | active after R10-01 | ADD diagnostics |

---

# 2. Root cause

## 2.1 Current parser behavior

`ToolCallStreamParser::feed()` first sends each byte through the current `FenceTracker`.

Relevant flow:

```cpp
if (fence_.consume(byte) == FenceTracker::Verdict::Content) {
    ...
    continue;
}

if (marker_byte(byte, visible)) {
    ...
}
```

A normal fenced block is protected:

````text
```xml
<tool_call>
...
</tool_call>
```
````

because fence bytes do not enter the executable marker machine.

However, outside a fence, indentation currently behaves differently.

In `FenceTracker::consume()`:

```cpp
if (phase_ == Phase::LineIndent) {
    if (byte == ' ') {
        ++indent_;
        if (indent_ > 3) {
            phase_ = Phase::LineBody;
        }
        return Verdict::Pass;
    }
    ...
}
```

After four spaces the bytes remain:

```text
Verdict::Pass
```

and therefore later reach `marker_byte()`.

The four spaces themselves are only formatting whitespace.

The intent policy explicitly does not lock on formatting whitespace.

Therefore:

```text
    <tool_call>
```

can still latch.

---

# 2.2 Why `RequireToolAtContentStart` is insufficient

`RequireToolAtContentStart` locks only after visible non-formatting content has been published.

Leading:

```text
SPACE
TAB
CR
LF
```

does not lock.

That behavior must remain for ordinary whitespace-prefixed genuine calls such as:

```text


<tool_call>
...
</tool_call>
```

Therefore Round 10 must **not fix R10-01 by changing whitespace into visible content globally**.

Instead it must distinguish:

```text
ordinary formatting whitespace before a call
```

from:

```text
a tool marker occurring on an indented literal line
```

---

# 2.3 Constraint divergence

`ToolCallGrammarConstraint::advance_pretrigger()` currently implements a separate pre-trigger scan.

It directly tracks:

```text
marker_prefix_
intent_
TextLocked
```

and uses:

```cpp
classify_tool_marker_prefix(...)
failed_marker_candidate_rescan_start(...)
```

but it does not execute the same literal/fence scanner as the production parser.

This means:

```text
production parser entry eligibility
```

and:

```text
future constrained-generation entry eligibility
```

are not governed by one authoritative machine.

Round 10 must remove this architectural divergence rather than fixing indentation twice in two independent implementations.

---

# 3. Round-10 invariants

## R10-I1 — Tool entry eligibility has one authority

There must be exactly one shared authority for deciding whether pre-latch bytes may trigger a tool region.

It must govern both:

```text
ToolCallStreamParser
ToolCallGrammarConstraint pre-trigger path
```

Do not independently duplicate:

```text
fence logic
indentation logic
marker-prefix logic
failed-marker rescan
intent locking
```

in two components.

---

## R10-I2 — ≥4 visual indentation columns make a potential marker literal

Outside a fenced block, a top-level marker whose first `<` begins at visual indentation column:

```text
>= 4
```

must be treated as ordinary literal/content bytes and must not latch.

Examples:

```text
    <tool_call>
\t<tool_call>
 \t<tool_call>
  \t<tool_call>
   \t<tool_call>
```

with normal 4-column tab stops all place `<` at column 4 or later.

They must therefore not trigger.

---

## R10-I3 — 0–3 leading spaces preserve current behavior

These remain eligible:

```text
<tool_call>
 <tool_call>
  <tool_call>
   <tool_call>
```

Round 10 must not silently convert the rule into:

```text
any indentation => text
```

The boundary is deliberate and test-pinned:

```text
column 0..3 -> eligible
column >=4 -> literal
```

---

## R10-I4 — Tabs use visual columns, not byte count

A tab must advance to the next 4-column tab stop.

Recommended helper:

```cpp
constexpr std::size_t next_tab_stop(std::size_t column) noexcept {
    return ((column / 4) + 1) * 4;
}
```

Required examples:

```text
column 0 + TAB -> 4
column 1 + TAB -> 4
column 2 + TAB -> 4
column 3 + TAB -> 4
column 4 + TAB -> 8
```

Do not use:

```cpp
++indent;
```

for tabs.

---

## R10-I5 — This is a narrow tool-safety rule, not a full CommonMark parser

Do **not** attempt to implement complete Markdown block parsing.

Do not add:

```text
list container parsing
blockquote parsing
nested container stacks
paragraph-interruption semantics
full CommonMark AST
```

The contract should be explicitly documented as:

> Before a tool entry has latched, a possible tool marker beginning on a line at visual indentation column 4 or greater is classified as literal content and cannot become executable tool markup.

This is deterministic, streaming-safe and directly addresses the integrity boundary.

Do not claim full CommonMark compliance.

---

## R10-I6 — Existing fenced behavior remains unchanged

Existing handling of:

```text
``` fences
~~~ fences
CRLF fences
longer fences
unclosed fences
list-indented fences
backtick-info-string cancellation
```

must remain green.

Round 10 must not modify the existing fence-closing rules unless a new independent fence reproducer proves an actual bug.

In particular, do not reopen the previous `fence_indent + 3` design merely because absolute CommonMark indentation superficially differs; list/container handling makes that comparison non-trivial.

---

## R10-I7 — Fence classification has precedence over indented-literal classification

A byte already classified as fenced content must not also be counted as indentation-suppressed content.

For diagnostics:

```text
fenced marker inside an indented fenced line
```

counts once as:

```text
fenced_markers_suppressed
```

and not additionally as:

```text
indented_markers_suppressed
```

No double counting.

---

## R10-I8 — Literal bytes become real content for intent purposes

Once an indented literal line emits visible bytes such as:

```text
<tool_call>
```

those bytes are ordinary Content.

Therefore under:

```text
RequireToolAtContentStart
```

the entry gate becomes locked exactly as it would for any other visible content.

Example:

```text
    <tool_call> ... literal example ...

<tool_call> ... real call ...
```

Expected:

```text
TemplateCompatible:
    first indented call -> text
    later real call -> may latch

RequireToolAtContentStart:
    first indented call -> text
    its visible '<...' locks content
    later real call -> must remain text
```

This behavior must be test-pinned.

---

## R10-I9 — Ordinary whitespace-only lines do not lock

For:

```text
"    \n"
"\t\n"
"  \t\r\n"
```

with no visible byte on the line:

```text
RequireToolAtContentStart
```

must remain unlocked.

A genuine call on a later non-indented line may still latch.

---

## R10-I10 — Chunk partition must not affect classification

For every new fixture:

```text
one-shot
single-byte streaming
every possible split position
representative multi-chunk splits
```

must produce the same:

```text
content
tool_calls
diagnostics
marker_seen
suppression counters
fallback reason
```

No chunk-sensitive indentation state.

---

## R10-I11 — Stage-1/2/3 receive exactly the same accepted regions as before

Round 10 changes only whether a region **starts**.

Once a marker is legitimately latched:

```text
Stage 1
Stage 2
Stage 3
```

must operate unchanged.

Do not make the region parser Markdown-aware.

Parameter payloads can legitimately contain:

```text
four spaces
tabs
fences
<tool_call>
</parameter>
</function>
```

and those remain payload bytes under the existing region grammar.

Literal suppression applies only **before entry latch / retry-entry qualification**, not inside an already owned parameter value.

---

# 4. Preferred architecture

## 4.1 Do not patch `FenceTracker` and `ToolCallGrammarConstraint` independently

A minimal patch like:

```cpp
if (indent >= 4) ...
```

inside `ToolCallStreamParser`

plus another:

```cpp
if (indent >= 4) ...
```

inside `ToolCallGrammarConstraint`

is explicitly rejected for Round 10.

It would reproduce the architecture problem that caused earlier rounds:

```text
two similar state machines
→ future behavior drift
→ new review round
```

Use one shared pre-trigger scanner.

---

# 4.2 Introduce a shared pre-trigger scanner

Preferred new files:

```text
src/models/qwen3_5/frontend/tool_call_entry_scan.h
src/models/qwen3_5/frontend/tool_call_entry_scan.cpp
```

Add them to:

```text
src/models/qwen3_5/frontend_sources.cmake
```

The class name may differ, but the responsibility must remain narrow.

Recommended conceptual API:

```cpp
class ToolCallEntryScanner {
public:
    ToolCallEntryScanner(ToolCallSyntaxMode syntax,
                         ToolCallIntentPolicy intent);

    struct FeedResult {
        std::string visible;
        bool triggered = false;

        // Number of bytes consumed from this feed through the accepted marker.
        // Valid when triggered=true.
        std::size_t consumed = 0;

        // Full prefix needed to seed the region parser. This may include held
        // formatting whitespace followed by the marker.
        std::string region_prefix;
    };

    [[nodiscard]] FeedResult feed(std::string_view bytes);

    [[nodiscard]] bool locked() const noexcept;

    [[nodiscard]] std::string held_tail() const;

    [[nodiscard]] std::uint64_t rescan_steps() const noexcept;

    [[nodiscard]] std::uint32_t fenced_markers_suppressed() const noexcept;

    [[nodiscard]] std::uint32_t indented_markers_suppressed() const noexcept;

    [[nodiscard]] bool ended_in_unclosed_fence() const noexcept;
};
```

Exact API details may be adapted to fit the codebase, but the shared state must own or delegate all pre-trigger-sensitive state.

---

# 4.3 State owned by the shared scanner

At minimum:

```text
syntax
intent

fence state
line-start / indentation state

pending formatting whitespace
marker candidate prefix
intent lock state

failed-marker rescan state/counter

literal suppression diagnostics
```

Do not leave one of these independently implemented inside `ToolCallStreamParser` and another independently implemented inside `ToolCallGrammarConstraint` unless there is a documented reason.

---

# 4.4 Extract the existing FenceTracker instead of rewriting it

Current nested class:

```text
ToolCallStreamParser::FenceTracker
```

should preferably move into the shared entry scanner implementation.

Preserve its current behavior byte-for-byte before adding indentation handling.

Recommended migration sequence:

```text
Step 1:
    move existing FenceTracker unchanged

Step 2:
    prove existing fence tests still green

Step 3:
    add indentation state

Step 4:
    integrate shared scanner into ToolCallStreamParser

Step 5:
    integrate same scanner into ToolCallGrammarConstraint

Step 6:
    add new equivalence tests
```

This makes regression attribution much easier.

---

# 5. Indented-literal state machine

## 5.1 Required state

Recommended minimal state:

```cpp
struct LineLiteralState {
    std::size_t visual_column = 0;
    bool at_line_start = true;
    bool indented_literal_line = false;
};
```

You may model it as an enum instead.

Example:

```cpp
enum class LinePrefixPhase : std::uint8_t {
    Start,
    Indent,
    LiteralBody,
    NormalBody,
};
```

What matters is deterministic streaming behavior.

---

# 5.2 Visual indentation calculation

At physical line start:

### Space

```cpp
++visual_column;
```

### Tab

```cpp
visual_column = ((visual_column / 4) + 1) * 4;
```

### LF

Reset:

```cpp
visual_column = 0;
at_line_start = true;
indented_literal_line = false;
```

### CR in CRLF

Do not treat `\r` as an indentation column.

Preserve compatibility with current CRLF handling.

LF remains the physical state reset.

---

# 5.3 Deciding that a line is literal

Do not classify whitespace-only prefixes as visible content.

Keep consuming/holding:

```text
space
tab
CR
```

until either:

```text
LF
```

or:

```text
first non-formatting byte
```

appears.

When the first meaningful byte of the line arrives:

```cpp
if (visual_column >= 4) {
    indented_literal_line = true;
}
```

If:

```text
indented_literal_line == true
```

then every remaining byte through the terminating LF is ordinary/literal content and must bypass tool-marker latching.

---

# 5.4 Important publication rule

Do not lose the indentation bytes.

For:

```text
"    <tool_call>"
```

the parser must return exactly:

```text
"    <tool_call>"
```

as content if no later structured region is accepted.

The existing `pending_ws_` behavior is useful here.

When the scanner discovers that the line is literal:

```text
publish held indentation
publish current byte
publish subsequent literal-line bytes
```

Do not trim the indentation.

---

# 5.5 Example transition

Input byte sequence:

```text
' ' ' ' ' ' ' ' '<'
```

State:

```text
column 0
→ 1
→ 2
→ 3
→ 4
```

At `<`:

```text
column == 4
=> literal line
=> flush held four spaces as content
=> publish '<' as content
=> do NOT start marker_prefix_
```

Remaining:

```text
tool_call>
```

is ordinary content.

---

# 5.6 Boundary control

For:

```text
' ' ' ' ' ' '<'
```

the `<` begins at column 3.

Expected:

```text
not literal
```

The three spaces remain ordinary held formatting whitespace and `<` enters the existing marker candidate state.

This control is essential to catch off-by-one bugs.

---

# 6. Marker suppression diagnostics

## 6.1 Add a separate field

Do not overload:

```text
fenced_markers_suppressed
```

with new semantics.

Add:

```cpp
std::uint32_t indented_markers_suppressed = 0;
```

to:

```text
ToolCallStreamResult
ToolCallParseDiagnostics
```

The existing field remains exactly:

```text
markers suppressed by recognized fenced code
```

The new field means:

```text
complete top-level markers suppressed because their '<'
began on an indented literal line
```

---

# 6.2 Marker counting must use the grammar authority

Do not count occurrences using:

```cpp
text.find("<tool_call>")
```

Use:

```cpp
classify_tool_marker_prefix(...)
```

with the configured:

```text
ToolCallSyntaxMode
```

Therefore:

### Native

Count only markers native mode could otherwise latch.

### Compatibility

Count:

```text
<tool_call>
<function_calls>
valid bare <function...>
valid bare <invoke...>
```

according to the existing grammar classifier.

---

# 6.3 Generalize the shadow-marker mechanism

The current fence diagnostics already contain a shadow marker scan.

Refactor this concept so the same grammar-based shadow scanner can feed:

```text
fenced literal bytes
indented literal bytes
```

but accumulate separate counters.

Recommended rule:

```text
Fence has precedence.

If a byte is fenced:
    feed fenced shadow scanner only.

Else if byte is indented-literal:
    feed indented shadow scanner only.

Else:
    normal marker machine.
```

No duplicate suppression count.

---

# 6.4 Propagate diagnostics end-to-end

Update:

```text
include/ninfer/types.h
src/models/qwen3_5/frontend/tool_call_stream.h
src/models/qwen3_5/frontend/tool_call_stream.cpp
src/models/qwen3_5/frontend/tool_call_parser.cpp
src/serve/request_log.cpp
src/serve/operational_log.cpp
tests/test_request_log.cpp
```

Request log JSON should contain:

```json
"indented_markers_suppressed": 1
```

alongside:

```json
"fenced_markers_suppressed": 0
```

Do not change the existing names.

---

# 6.5 Operational warning

A response with:

```text
0 tool calls
marker_seen == false
indented_markers_suppressed > 0
```

should be diagnosable.

Add an operational warning analogous to the fence warning.

Recommended wording:

```text
req#N tool-call marker suppressed in indented literal content |
indented_markers_suppressed=X
```

Do not call it a parser failure.

This is an intentional entry-classification decision.

Do not set:

```text
MalformedStructure
```

merely because a marker was suppressed before latch.

The output remains ordinary content.

---

# 7. Integrate `ToolCallStreamParser`

## 7.1 Replace duplicated pre-latch state

`ToolCallStreamParser` should delegate pre-latch processing to the shared scanner.

Keep the region parser ownership in `ToolCallStreamParser`.

Conceptually:

```text
not latched
    ↓
shared ToolCallEntryScanner
    ↓
ordinary visible bytes OR trigger
    ↓
trigger:
    region_ = accepted region prefix
    append unconsumed chunk remainder
    latched_ = true
```

After latch:

```text
region_.append(chunk)
```

unchanged.

---

# 7.2 Preserve current externally visible behavior

These must remain unchanged:

```text
content_prefix()
latched()
latched_region()
held_tail()
marker_seen()
rescan_steps
fence diagnostics
```

The implementation may delegate them internally.

Do not subtly alter one-shot/streaming output ordering.

---

# 7.3 Do not scan parameter payload indentation

Once:

```text
latched_ == true
```

all remaining bytes belong to:

```text
region_
```

and are interpreted only by the region parser.

The entry scanner must no longer suppress:

```text
    <tool_call>
```

inside a parameter value.

That would corrupt payload fidelity.

---

# 8. Integrate `ToolCallGrammarConstraint`

## 8.1 `advance_pretrigger()` must use the shared scanner

Remove its independent pre-trigger marker classification wherever possible.

Current duplicated behavior around:

```text
marker_prefix_
RequireToolAtContentStart
failed_marker_candidate_rescan_start()
classify_tool_marker_prefix()
```

must delegate to the shared scanner.

The constraint still owns its higher-level phases:

```cpp
enum class ToolConstraintPhase {
    PreTrigger,
    TextLocked,
    InRegion,
    BetweenCalls,
};
```

Do not collapse those phases again.

Round 9 deliberately separated:

```text
PreTrigger
```

from:

```text
BetweenCalls
```

and that must remain.

---

# 8.2 Shared scanner applies only to PreTrigger

Do **not** use the lazy pre-trigger scanner for:

```text
BetweenCalls
```

After the first tool sequence begins, Round-9 strict suffix semantics remain:

```text
whitespace -> allowed
next legal marker -> allowed
partial marker -> NeedMore
visible prose -> Rejected
failed marker -> Rejected
```

No code-fence escape hatch after a completed executable call.

Example:

````text
TOOL_CALL
```text
prose
```
````

must remain invalid suffix content under strict tool-sequence semantics.

Do not let the new shared scanner accidentally make that legal.

---

# 8.3 Phase mapping

When shared pre-trigger scanner reports ordinary content:

### TemplateCompatible

remain:

```text
PreTrigger
```

unless no special phase transition is required.

### RequireToolAtContentStart

if visible literal content has been committed:

```text
PreTrigger -> TextLocked
```

When scanner reports trigger:

```text
PreTrigger -> InRegion
```

seed:

```text
buffer_
```

with exactly the region prefix required for `parse_tool_call_region()` plus remaining candidate bytes.

---

# 8.4 Checkpoint/restore

`ToolCallGrammarConstraint` already supports value-semantic:

```text
checkpoint()
restore()
```

The new shared scanner state must therefore be safely copyable.

Avoid:

```text
raw owning pointers
external mutable state
non-copyable stream objects
hidden shared mutable counters
```

A checkpoint must include:

```text
fence state
line indentation state
marker candidate
held whitespace
intent lock
suppression diagnostics if constraint stores them
```

---

# 9. Exact regression corpus

## 9.1 Native indented wrapper

Use a valid declared call.

Generate an indented copy where **every non-empty call line** has four spaces.

Input:

```text
    <tool_call>
    <function=bash>
    <parameter=command>
    echo example
    </parameter>
    </function>
    </tool_call>
```

Expected:

```text
0 tool calls
content byte-identical
marker_seen == false
indented_markers_suppressed >= 1
fallback_reason == None
```

Run:

```text
strict
tolerant
TemplateCompatible
RequireToolAtContentStart
```

Both one-shot and streaming.

---

# 9.2 Leading TAB

Same fixture using:

```text
\t<tool_call>
```

and tab-indent every relevant line.

Expected:

```text
text
0 calls
suppression diagnostic
```

---

# 9.3 Mixed indentation

Required variants:

```text
" \t<tool_call>"    // '<' at visual column 4
"  \t<tool_call>"   // column 4
"   \t<tool_call>"  // column 4
"\t <tool_call>"    // column 5
```

All suppressed.

---

# 9.4 Three-space control

Input:

```text
   <tool_call>
   ...
```

Expected in Native mode:

```text
valid call still latches
```

assuming the remaining wire markup itself is valid.

This pins:

```text
3 spaces != indented literal
```

---

# 9.5 Empty whitespace line

Input:

```text
"    \n" + REAL_CALL
```

Expected:

```text
REAL_CALL executes
```

including under:

```text
RequireToolAtContentStart
```

because the blank line contained no visible content.

---

# 9.6 Indented example followed by real call

Input:

```text
INDENTED_EXAMPLE
"\n"
REAL_CALL
```

### TemplateCompatible

Expected:

```text
indented example -> content
real call -> structured
content == indented example (rtrim according to existing accepted-region behavior)
tool_calls == 1
```

### RequireToolAtContentStart

Expected:

```text
indented example -> ordinary visible content
gate locks
real call -> text
0 tool calls
```

This is an important intent-policy cross-test.

---

# 9.7 Fenced control

Existing:

````text
```xml
<tool_call>
...
</tool_call>
```
````

must remain:

```text
fenced_markers_suppressed > 0
indented_markers_suppressed == 0
```

even if the fenced lines are also indented.

---

# 9.8 Compatibility bare-function code example

Compatibility mode must also suppress an indented bare entry:

```text
    <function=bash>
    <parameter=command>
    echo example
    </parameter>
    </function>
```

Expected:

```text
text
0 structured calls
indented_markers_suppressed >= 1
```

This prevents Round 10 from fixing only `<tool_call>` while leaving Compatibility's other legal entry families exposed.

---

# 9.9 Native compatibility control

The same bare:

```text
<function=bash>
...
</function>
```

under:

```text
QwenWrappedNative
```

already must not execute.

Round 10 must not change that behavior.

---

# 9.10 Parameter payload control

A real call whose parameter contains:

```text
    <tool_call>
```

must preserve the bytes exactly.

Example:

```text
<tool_call>
<function=write>
<parameter=content>
    <tool_call>
    example only
    </tool_call>
</parameter>
</function>
</tool_call>
```

Expected:

```text
1 structured write call
content argument preserves the indentation and markup byte-exact
```

This proves the new literal guard is pre-latch only.

---

# 9.11 Every-byte split test

For every new representative fixture:

```cpp
for (std::size_t split = 0; split <= text.size(); ++split) {
    decoder.feed(text.substr(0, split));
    decoder.feed(text.substr(split));
    ...
}
```

Compare against one-shot.

At minimum run this for:

```text
4-space native example
TAB native example
mixed-indent example
3-space genuine call
indented example + later real call
```

Also retain existing bytewise streaming test.

---

# 9.12 Multi-boundary split test

Explicitly split inside:

```text
indentation
TAB transition
"<tool_"
"<tool_call>"
CRLF
fence opener
failed marker candidate
```

Examples:

```text
"  " + "  <tool_call>"
" \t" + "<tool_call>"
"\t<tool_" + "call>"
```

Classification must remain identical.

---

# 10. Parser/Constraint cross-equivalence tests

Add a dedicated test that feeds identical pre-trigger byte streams into:

```text
production ToolCallStreamParser
ToolCallGrammarConstraint
```

and checks tool-entry eligibility.

Corpus:

```text
A  genuine native call
B  genuine native call with 1 space
C  genuine native call with 3 spaces
D  native call with 4 spaces
E  native call with TAB
F  fenced call
G  unclosed fenced call
H  ordinary prose + call
I  failed marker + call
J  whitespace-only line + call
K  partial marker
L  compatibility bare function
M  indented compatibility bare function
```

Run under:

```text
QwenWrappedNative × TemplateCompatible
QwenWrappedNative × RequireToolAtContentStart
Compatibility      × TemplateCompatible
Compatibility      × RequireToolAtContentStart
```

The exact higher-level result need not be identical because:

```text
parser returns content/tool calls
constraint returns Allowed/NeedMore/Rejected/state
```

but **entry-trigger eligibility must agree**.

Define a helper assertion around the invariant:

```text
If the production pre-latch machine can latch the prefix,
the constraint pre-trigger machine must be capable of entering InRegion.

If the production machine classifies the candidate as literal/suppressed,
the constraint must not enter InRegion from that candidate.
```

This is the architectural regression gate Round 10 needs.

---

# 11. Diagnostics tests

Update `tests/test_request_log.cpp`.

Test JSON contains:

```json
{
  "fenced_markers_suppressed": 0,
  "indented_markers_suppressed": 1
}
```

for indented input.

Existing fence fixture remains:

```json
{
  "fenced_markers_suppressed": 1,
  "indented_markers_suppressed": 0
}
```

Test the operational renderer.

For indented suppression with no structured calls:

```text
severity = Warning
message contains:
    tool-call marker suppressed in indented literal content
    indented_markers_suppressed=...
```

Do not leak large model output.

Keep snippets bounded if one is added.

---

# 12. Documentation updates

Update:

```text
docs/tool_call_parser.md
docs/new_parser_phase4_design.md
docs/serving.md
```

## `docs/tool_call_parser.md`

Add a section:

```text
Indented literal lines
```

Document:

```text
Before tool-entry latch, a marker beginning at visual indentation
column >=4 is treated as literal content.

Tabs advance to 4-column tab stops.

This is a deterministic tool-safety rule, not complete CommonMark parsing.
```

Also update diagnostics:

```text
indented_markers_suppressed
```

---

## `docs/new_parser_phase4_design.md`

Remove/update the statement that the pre-trigger constraint remains a deliberate no-fence superset.

After Round 10, state explicitly:

```text
Production pre-latch parsing and the CPU grammar constraint share
the same entry classifier for fences, indentation, marker prefixes,
failed-marker rescans and intent locking.
```

Do not claim equivalence for:

```text
BetweenCalls
```

through the lazy scanner; post-call remains intentionally stricter.

---

## `docs/serving.md`

Document the operational diagnostic.

Clarify that a suppressed marker is returned as normal assistant content and is not executed.

---

# 13. Files expected to change

Preferred set:

```text
src/models/qwen3_5/frontend/tool_call_entry_scan.h          NEW
src/models/qwen3_5/frontend/tool_call_entry_scan.cpp        NEW
src/models/qwen3_5/frontend_sources.cmake

src/models/qwen3_5/frontend/tool_call_stream.h
src/models/qwen3_5/frontend/tool_call_stream.cpp

src/models/qwen3_5/frontend/tool_call_grammar_state.h
src/models/qwen3_5/frontend/tool_call_grammar_state.cpp

src/models/qwen3_5/frontend/tool_call_parser.cpp

include/ninfer/types.h

src/serve/request_log.cpp
src/serve/operational_log.cpp

tests/test_tool_call_parser.cpp
tests/test_tool_call_grammar_state.cpp
tests/test_request_log.cpp

docs/tool_call_parser.md
docs/new_parser_phase4_design.md
docs/serving.md
```

Do not touch Stage-1/2/3 source merely to make the diff appear integrated.

In particular, avoid unnecessary modifications to:

```text
tool_call_grammar.cpp
parameter normalization
OpenAI request adapters
Anthropic request adapters
```

unless compilation requires an include/API adjustment.

---

# 14. Implementation order

Use this order so regressions remain attributable.

## Phase A — Refactor without behavior change

1. Add shared pre-trigger scanner files.
2. Move/extract existing fence state machine unchanged.
3. Move the existing marker-prefix/rescan logic into the shared scanner.
4. Make `ToolCallStreamParser` consume it.
5. Run existing parser tests.
6. No new indentation behavior yet.

Required result:

```text
all existing tests green
behavior byte-identical
```

Do not proceed if this phase changes existing semantics unexpectedly.

---

## Phase B — Add indentation literal classification

1. Add visual-column tracking.
2. Add 4-column tab stops.
3. Add `>=4` literal-line threshold.
4. Ensure whitespace-only lines remain formatting only.
5. Ensure literal visible bytes enter normal content/intent locking.
6. Add parser tests.
7. Run parser tests.

---

## Phase C — Constraint integration

1. Replace duplicated `advance_pretrigger()` entry classification with the shared scanner.
2. Keep Round-9 `BetweenCalls` scanner unchanged.
3. Make scanner state part of checkpoint/restore value semantics.
4. Add parser-vs-constraint cross-equivalence corpus.
5. Run grammar-state tests.

---

## Phase D — Diagnostics

1. Add `indented_markers_suppressed`.
2. Propagate through parse results.
3. Add request-log JSON field.
4. Add operational warning.
5. Add tests.

---

## Phase E — documentation and full CPU/static regression

Only after code/tests are green.

---

# 15. Negative requirements

Round 10 is incorrect if it does any of the following.

### Do not execute truncated functions

Never change:

```text
missing </function> -> no executable call
```

### Do not parse Markdown inside parameter values

Once latched, the wire grammar owns the region.

### Do not make all whitespace literal

This would break valid whitespace-prefixed calls.

### Do not change the threshold to 1

Required:

```text
0..3 eligible
>=4 literal
```

### Do not treat every TAB in the output as code

Only indentation before the first non-formatting byte of a physical line affects entry eligibility.

### Do not solve semantic intent with language heuristics

No:

```text
Example:
Syntax:
Demo:
Do not run:
```

special cases.

### Do not remove `TemplateCompatible`

It remains the production compatibility default.

### Do not activate constrained decoding

Round 10 only repairs the CPU/shared entry semantics required before future activation.

### Do not duplicate the implementation

No second independent indentation/fence state machine in the constraint.

---

# 16. Required targeted test targets

At minimum build/run:

```text
ninfer_tool_call_parser_test
ninfer_tool_call_grammar_test
ninfer_tool_call_grammar_state_test
ninfer_request_log_test
```

Also run the protocol/schema tests affected by shared diagnostics/public structs:

```text
OpenAI schema tests
OpenAI Responses tests
Anthropic schema tests
HTTP route tests
serve option tests
```

Use the actual CTest names in the configured build rather than assuming executable naming if they differ.

---

# 17. Full regression gate

After targeted tests:

```text
ctest --test-dir <build> --output-on-failure
```

with GPU/model-dependent real tests allowed to skip according to their existing:

```text
SKIP_RETURN_CODE 77
```

contract.

Do not claim GPU validation from skipped tests.

Required summary must report separately:

```text
selected
passed
failed
skipped
```

and distinguish:

```text
CPU/static verification
GPU/model runtime verification
```

---

# 18. Adversarial review after implementation

After tests pass, perform a new code review specifically attempting to disprove the implementation.

Audit:

```text
1. off-by-one at 3 vs 4 columns
2. TAB visual-column handling
3. CRLF state reset
4. blank indentation-only lines
5. chunk boundaries inside indentation
6. chunk boundaries inside marker prefix
7. fences + indentation overlap
8. Compatibility bare markers
9. Native marker restriction
10. RequireToolAtContentStart lock timing
11. marker suppression diagnostics
12. duplicate diagnostic counting
13. retry-entry fence masking
14. parameter payload unaffected
15. one-shot/streaming equivalence
16. constraint checkpoint/restore
17. BetweenCalls still strict
18. no Stage-1/2/3 behavior drift
```

For every suspected finding:

```text
construct reproducer
→ verify against actual code
→ verify against tests
→ reject speculative finding if reproducer does not stand
```

Do not report hypothetical concerns as confirmed bugs.

---

# 19. Acceptance matrix

The implementation is not complete until all rows are pinned.

| Input | Native TC | Native SOC | Compat TC | Compat SOC |
|---|---|---|---|---|
| call column 0 | call | call | call | call |
| call column 1 | call | call | call | call |
| call column 2 | call | call | call | call |
| call column 3 | call | call | call | call |
| call column 4 | text | text | text | text |
| call after TAB from col 0 | text | text | text | text |
| call after ` \t` | text | text | text | text |
| fenced call | text | text | text | text |
| prose + call | call | text | call | text |
| blank 4-space line + call | call | call | call | call |
| indented example + real call | real call | text | real call | text |
| bare function column 0 | text | text | call | call |
| bare function column 4 | text | text | text | text |

Where:

```text
TC  = TemplateCompatible
SOC = RequireToolAtContentStart
```

---

# 20. Definition of Done

Round 10 is green only when all of the following are true.

### Architecture

- [ ] one shared pre-trigger entry classifier exists;
- [ ] production parser consumes it;
- [ ] grammar constraint PreTrigger consumes it;
- [ ] BetweenCalls remains separate and strict;
- [ ] scanner is value-semantic/copyable for checkpoint/restore;
- [ ] no duplicate fence/indent entry logic remains.

### Indentation

- [ ] marker at column 0 accepted;
- [ ] marker at column 1 accepted;
- [ ] marker at column 2 accepted;
- [ ] marker at column 3 accepted;
- [ ] marker at column 4 suppressed;
- [ ] marker beyond column 4 suppressed;
- [ ] TAB uses 4-column tab stops;
- [ ] mixed space/TAB indentation correct;
- [ ] blank indentation-only line does not lock intent.

### Parser semantics

- [ ] indented native wrapper stays text;
- [ ] indented Compatibility bare call stays text;
- [ ] parameter payload indentation remains byte-exact;
- [ ] Stage-1 parser unchanged;
- [ ] Stage-2 parser unchanged;
- [ ] Stage-3 recovery unchanged;
- [ ] missing function close remains non-executable;
- [ ] FailClosed unchanged.

### Intent

- [ ] TemplateCompatible can still call after normal prose;
- [ ] RequireToolAtContentStart still locks after visible prose;
- [ ] indented literal visible content locks SOC;
- [ ] ordinary formatting-only whitespace does not lock SOC.

### Streaming

- [ ] one-shot == whole-chunk streaming;
- [ ] one-shot == bytewise streaming;
- [ ] one-shot == every-split streaming;
- [ ] CRLF partition invariant;
- [ ] TAB partition invariant;
- [ ] marker-prefix partition invariant.

### Constraint

- [ ] fenced entry eligibility agrees with parser;
- [ ] indented entry eligibility agrees with parser;
- [ ] failed-marker eligibility agrees with parser;
- [ ] genuine marker eligibility agrees with parser;
- [ ] partial marker remains `NeedMore`;
- [ ] checkpoint/restore exact;
- [ ] `can_terminate()` Round-9 behavior unchanged.

### Diagnostics

- [ ] `indented_markers_suppressed` added;
- [ ] fenced counter semantics unchanged;
- [ ] no fence/indent double count;
- [ ] request log emits new field;
- [ ] operational warning exists;
- [ ] suppressed marker does not create fake fallback failure.

### Verification

- [ ] targeted parser tests pass;
- [ ] grammar-state tests pass;
- [ ] request-log tests pass;
- [ ] OpenAI tests pass;
- [ ] Responses tests pass;
- [ ] Anthropic tests pass;
- [ ] HTTP tests pass;
- [ ] full GPU-hidden CTest has zero failures;
- [ ] skipped real/GPU tests reported separately;
- [ ] final adversarial code review has no unresolved confirmed findings.

---

# 21. Final implementation principle

Do not treat Round 10 as another parser rewrite.

The Stage-1/2/3 parser now has a well-defined structural responsibility:

```text
Once bytes legitimately enter a tool region:
    determine their tool-call structure safely.
```

Round 10 fixes the preceding responsibility:

```text
Before a region exists:
    determine whether apparent tool markup is executable wire markup
    or literal assistant content.
```

The desired architecture after Round 10 is:

```text
                         ┌──────────────────────────┐
assistant content ──────►│ Shared Entry Classifier  │
                         │                          │
                         │ fence suppression        │
                         │ indented-literal guard   │
                         │ marker grammar           │
                         │ failed-marker rescan     │
                         │ intent gate              │
                         └─────────────┬────────────┘
                                       │
                      executable entry │
                                       ▼
                         ┌──────────────────────────┐
                         │ Tool Region Parser       │
                         │                          │
                         │ Stage 1 greedy           │
                         │ Stage 2 consistent       │
                         │ Stage 3 recovery         │
                         │ declared-name checks     │
                         │ normalization            │
                         └──────────────────────────┘
```

and independently:

```text
Shared Entry Classifier
        │
        └──► ToolCallGrammarConstraint::PreTrigger
```

while:

```text
ToolCallGrammarConstraint::BetweenCalls
```

remains deliberately stricter after an executable tool sequence has started.

This is the architectural boundary that should prevent another series of isolated parser fixes for the same class of entry-classification bug.
