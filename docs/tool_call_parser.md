# Tool-call parser

The Qwen frontend parses model-generated tool calls out of the assistant stream with a
single wire grammar and one incremental parser. One-shot parsing and streaming go through
the same parser: one-shot feeds the whole text in one chunk, streaming feeds chunks of any
size, and the result is independent of the chunk partition.

## Supported wire forms

The wire grammar is the single source of truth for tag literals, header syntax and marker
recognition. Accepted forms:

- Wrapper: `<tool_call> ... </tool_call>` (zero or more calls) and
  `<function_calls> ... </function_calls>` (one or more calls). A region may also start
  directly with a function or invoke opener (no wrapper). A wrapper that is still open at
  the end of the input is a truncation, never a clean completion: strict rejects the region
  (`MalformedStructure`); tolerant retains the closed calls with `TruncatedTail`.
- Function openers, per family:
  - `<function=NAME>` / `<invoke=NAME>` (short form; `NAME` is the first `name` attribute or
    the short value),
  - `<function name="NAME">` / `<invoke name='NAME'>` (attribute form; both quote characters,
    a `>` inside a quoted value is value text, and whitespace around `=` is allowed),
  - a bare `<function>` / `<invoke>` opener with a `name` attribute.
  The name must be nonempty, must not contain `<` or `>`, and is length-bounded.
- Parameter openers, per family:
  - `<parameter=NAME>` / `<param=NAME>` (short form),
  - `<parameter name="NAME">` / `<param name='NAME'>` (attribute form),
  - a bare opener with a `name` attribute.
- Closes: `</function>`, `</invoke>`, `</parameter>`, `</param>`, `</tool_call>`. A close
  matches its family; the wrapper closes only at its own literal.
- Parameter values are raw bytes: they preserve function/tool-call markers, nested
  `<parameter=...>...</parameter>` text, and any other markup as value bytes; a quoted example
  that closes its own structure stays byte-exact (Stage-2 consistent completion, below).

Header forms that are not prefixes of a valid header (a broken keyword, a missing separator
after a value, an empty unquoted value, an unterminated quote at the region end) are rejected,
never silently truncated.

## Parser architecture

- One wire grammar owns tag families, the exact wrapper literals, the header grammar
  (short / attribute / bare forms with quote awareness), streaming prefix classification and
  marker discovery. No consumer re-implements tag rules.
- One state machine parses the region (marker latch, function header, function body, parameter
  header, parameter value, wrapper close). It is byte-driven and deterministic on the complete
  bytes, so one-shot and streaming agree by construction and are tested for equality at every
  split point of a region.
- The region parse produces an objective parse progress (what was recognized, where the input
  ended, whether a structural break occurred). A separate, pure recovery policy decides what
may be committed; the parser does not decide policy.

## Marker entry and recovery

The top-level entry-marker set is one grammar decision shared by the streaming latch,
one-shot discovery, the recovery retries and the grammar-constraint lazy trigger: a
complete wrapper literal (`<tool_call>`, `<function_calls>`) or a syntactically complete
bare function/invoke opener. The bare form is a compatibility entry at the stream's top
level.

A region that breaks is re-read at the break (a definite structural break) or after the
accepted prefix (a cut at the input end); the bytes before the accepted region remain
verbatim content. A break that leaves a wrapper open is never a recovery entry: the failed
wrapper owns the remaining bytes of its still-unclosed scope, and the recovery search runs
only from a proven top-level scope (a break at `wrapper = None`). Concretely: a later
wrapper literal inside the failed wrapper's scope never opens a fresh region, a bare
function/invoke opener there is never a recovery entry (it would reinterpret nested payload
as a new call), and a wrapper that opened inside the failed wrapper (a nesting break) is
skipped. Wrapper nesting — a second wrapper open before the first one closed — is a
structural break in both modes.

While scanning for the first marker, a candidate broken by a byte the grammar classifies
as NotMarker is published up to that byte; when the breaking byte itself is `<`, it is
retained as the start of a fresh candidate, so a real marker inside a failed prefix
(`prefix <function<tool_call>...`) still latches at the machine's latch byte. A `<` the
header grammar still accepts (a quoted attribute value) keeps the candidate open instead
of breaking it.

## Code fences

Pre-latch content is scanned by a line-oriented fence tracker (a practical CommonMark
subset): a run of at least three backticks or tildes at the start of a line (up to three
spaces of indentation) opens a fence, and only a run of the same character of at least the
opener's length, alone on its line, closes it. Inside a recognized fence the bytes are
ordinary content and never enter the marker candidate machine, so a complete tool marker
written inside a fenced code block of final content cannot latch as a structured call. An
unclosed fence stays open to the end of the input (suppression is the safe direction). The
tracker only sees the pre-latch content channel: once a region has latched, the region's own
bytes are parsed by the wire grammar, which owns value bytes (a value may contain fence
markup). The fence state is also computed over the latched region's bytes, independent of the
chunk partition: recovery retry entries skip markers inside a recognized fence, and the
diagnostics report `fenced_markers_suppressed` (complete markers a fence suppressed, pre-latch
and retry) and `ended_in_unclosed_fence` (the pre-latch stream ended inside an unclosed fence).

## Indented literal lines

Before a tool entry has latched, a possible tool marker that begins a physical line at
visual indentation of four columns or more is classified as **literal content** and cannot
become executable tool markup. This is a deterministic tool-safety rule, not complete
CommonMark parsing:

- Visual columns count the line's leading indentation only: a space is one column, a
  tab the next multiple of four (a tab at column 0, 1, 2, or 3 lands on column 4), and a
  line break resets the line; the column is frozen once the line's first
  non-formatting-whitespace byte has arrived (CR is never a column). That first byte at
  column four or more makes the whole line literal; columns 0-3 keep their baseline
  eligibility (a genuine call indented by at most three spaces still latches).
- A whitespace-only indented line (for example `    ` followed by a break) carries no
  visible byte and does not lock the `start-of-content` intent gate: a genuine call on
  the next line at column zero still executes.
- Once a line is literal, every byte of that line is ordinary content, held for
  publication, and never enters the marker candidate machine. This is a pre-latch rule:
  a marker line indented inside a latched region's parameter value is parsed by the wire
  grammar and preserved byte-exact.
- Fence classification has precedence: a byte the fence tracker owns is fed to the fence
  shadow, never to the indentation shadow, so a marker on an indented line inside a
  recognized fence is counted as fence-suppressed only, never double-counted.
- A literal marker is an entry-classification decision, not a parse failure: the output
  is returned verbatim as ordinary content with `fallback_reason` `None`, the
  `indented_markers_suppressed` diagnostic set, and (for the serve product) an
  operational warning. Under `start-of-content`, the literal line's visible bytes lock
  the intent gate, so a later genuine call in the same response stays text.

The pre-latch classification (fence suppression, indented-literal classification, the
marker candidate, failed-candidate rescans, and the intent gate) runs in one shared
entry-scanner that both the production parser and the CPU grammar-constraint core
consume, so the two consumers decide entry eligibility on the same byte.

## Strict vs. tolerant

Both modes parse with the same grammar and the same state machine; the policy changes only
what an incomplete or trailing region commits.

Strict (default): all-or-nothing. A region that does not end cleanly falls back to verbatim
content with the precise reason (`MalformedStructure` for a cut or broken region, `TrailingContent`
for content after complete calls, `UndeclaredTool` for a name outside the declared tools).

Tolerant (`--tolerant-tool-calls`): commits only calls whose function close has been
consumed. A call whose function close was not consumed is never executable, whatever its
parameter values show: the function close is the executability boundary, not the value
close. Complete calls before a later broken call, and a function-closed final call cut
before its wrapper close, are retained with a `truncated_tail` diagnostic after a natural
stop (StopToken, or no reported reason); after a cut (OutputLimit, StopString,
ContextCapacity, Cancelled), a definitive break leaves nothing committable behind it, so a
region with a suffix or trailing content is returned as text and only a region whose
completion is clean is committed. A name-only
truncation (no closed parameter and no function close) still falls back to text.
An undeclared name is a break in tolerant mode as in strict mode: identity is not a syntax
issue that tolerance repairs, so an undeclared call is never emitted in either mode
(`UndeclaredTool`). The empty `<function_calls>` wrapper is unrecoverable in both modes;
a `<function_calls>` region holds a sequence of calls (another call may follow any function
close), and an unclosed `<function_calls>` at the input end retains the closed calls with
`TruncatedTail` — never a clean completion.

## Delimiter boundaries and consistent completion

The wire format has no delimiter escape. A `</parameter>` inside a value is a real closer only
when the bytes after it form a legal continuation, decided by one shared wire-grammar
classification consumed by both the value scan and the function-close lookahead: another
parameter opener, the function's closer followed by a legal top-level entry (the wrapper close,
another function/invoke opener, or the end of input), or the end of the input. A closer at the
input end is a provisional boundary: the parameter commits, but a function that never closes
stays non-executable.

Any other continuation (immediate markup that is not a structural token, a quoted closer
immediately followed by the next token) is value text — a shell command that echoes the markup.
Inside a `<tool_call>` wrapper the rule is stricter: a `</parameter>` followed by
`</function>`/`</invoke>` is a real closer only when the wrapper close or the region end follows
the function close.

When the first (greedy) boundary choice leads the region into a definitive structural break, the
parser re-parses it with a consistent completion (Stage 2): pass 1 tries only the closers that are
balanced against nested openers of the same parameter family; only when no pass-1 candidate stands
does pass 2 try every closer of the family in order, and a standing pass-2 boundary marks the
completion as unbalanced. The earliest base whose completion is balanced wins; otherwise a single
completion wins; two or more unbalanced completions make the region ambiguous
(`ambiguous_structure`), as does a call of a declared tool that carries a synthetic argument
(a repeated parameter name, or a non-declared name outside the first parameter of a tool with
an unambiguous declared schema). Stage 2 is bounded by a deterministic work budget (four units
per region byte, at least 100 000); a region that exhausts it is returned as text and records
`parse_budget_exhausted`. The residual ambiguity is documented in the Round-3 spec §9: a bare
(unwrapped) call whose open value contains a complete example, ended by a natural stop, commits
the example as the turn (R3-04), and a degenerate output made only of prose and unfenced complete
examples commits its last example in strict mode.

## Native and compatibility syntax

The top-level entry syntax is the output contract's `ToolCallSyntaxMode` (serving:
`--tool-call-syntax`; the production Qwen3.8 default is the native mode):

- `QwenWrappedNative`: the only executable top-level entry is the wrapped `<tool_call>` form.
  `<function_calls>`, bare `<function=...>` and `<invoke=...>` are ordinary text at top level.
  Inside the wrapper, the canonical `<function=...>` / `<parameter=...>` tags are unchanged.
- `Compatibility`: the wider historical entry set — the `<tool_call>` and `<function_calls>`
  wrappers and bare `<function=...>` / `<invoke=...>` openers all latch as top-level tool
  regions.

The mode is a single grammar decision shared by the streaming pre-latch scan, the marker
retry search, the top-level entry transitions after a closed wrapper, and the Phase-4
constraint core (Round-6 R6-01: the constraint constructor takes the same `ToolCallSyntaxMode`,
so the CPU constraint state can never latch an entry the parser would treat as prose). It is
an explicit configuration choice, never inferred from template filenames or other metadata
substrings, and a canonical wrapped Qwen3.8 call parses identically in both modes.
Under `RequireToolAtContentStart` (Phase-4, R6-05; intent parity R7-02, R8-01) the constraint
trigger is a tool-sequence gate mirroring the parser's intent state: visible
non-formatting-whitespace content locks the gate at any point — before the first entry or
after a completed call (Round 8 R8-01: the first latch is not permanent permission) — a
complete marker under a locked gate is ordinary content and never triggers, directly
consecutive wrappers across formatting whitespace stay eligible, and the trailing content
after a closed region replays through the gate from the parser's `break_offset` instead of
jumping to the last `<` (Round 8 §3.5/§3.7). The gate's state words remain value-semantic
across checkpoint/restore, so speculative rollback recovers the exact tool-entry eligibility.

## Ambiguous byte protocol

The raw wire protocol has no escaping or length framing for string parameter values, so the
same bytes can be a structural closer or literal tool markup inside a payload (the canonical
closer lines quoted inside a `write.content` value). A byte parser cannot prove which. The
output contract's `ToolCallAmbiguityPolicy` (serving: `--tool-call-ambiguity`) makes the choice
explicit:

- `FailClosed` (default; the production Qwen3.8 policy): a Stage-2 value boundary is ambiguous
  when a candidate's closer chain stands while an earlier candidate's closer chain had already
  formed a complete call — both interpretations are structurally plausible (a complete call at
  the early close, or the later close as payload). The parser refuses it: the region is
  returned as text with the `ambiguous_structure` fallback reason, and no alternative call may
  execute. This prevents the known R1 phantom-call class without claiming the byte protocol is
  unambiguous.
- `PayloadFidelity` (the historical round-4 behavior): the later closing chain wins; embedded
  tool markup inside a string value is preserved byte-exact. The R1 phantom-acceptance class
  stays executable, and the pinned `test_round4_r1_residual_pinned` fixture documents that
  verdict under this policy.

What is and is not ambiguous is decided by nesting, not by the policy: a well-formed nested
example (its own `<parameter>` openers present) is balanced — its closers are consumed by
nesting depth, only the outer close is a viable boundary, and both policies commit the payload
byte-exact. `FailClosed` refuses only the unbalanced class (an early complete closer chain plus
prose plus the outer closer chain, with no matching openers), which includes some legitimate
but structurally ambiguous `write.content` payloads. Choosing `PayloadFidelity` knowingly is
valid; claiming phantom-call elimination under it is not.

## Semantic quotation residual

The `ToolCallAmbiguityPolicy` above controls one specific structural ambiguity: a Stage-2 value
boundary whose closer chain stands while an earlier candidate's closer chain had already formed
a complete call. It is not an intent oracle, and it must not be described as one.

A different, older class remains: a **complete, declared, canonical, unfenced** `<tool_call>`
example whose bytes are identical to a genuine action. When the byte sequence itself contains
no contradictory structure, a byte parser cannot distinguish "show this example" from
"execute this":

- code fences are handled: a marker inside a recognized fence never latches;
- undeclared names are handled: declared-tool enforcement returns the region as text;
- compatibility-only top-level markers are excluded in native mode;
- but a fully valid, declared, canonical `<tool_call>` in ordinary unfenced content is still
  indistinguishable from an actual action;
- `FailClosed` is not invoked for such a region: a clean Stage-1 completion never reaches
  Stage 2, so no ambiguity policy can refuse it;
- the official Qwen3.8 template explicitly permits natural-language reasoning before a
  function call, so a prose prefix is not by itself evidence of quotation.

This is a documented residual of the native wire protocol, not an implementation bug. The
pinned fixture `test_r6_complete_unfenced_in_set_example_residual` records the current verdict
under the production policies: prose-prefixed and example-only outputs commit their complete
unfenced call; only the fenced equivalent stays text. The optional Round-6 hardening mode
(`--tool-call-intent start-of-content`) reduces the prose-prefixed quotation class; the
example-only byte-identical output remains fundamentally ambiguous and can only be eliminated
by a TEXT-vs-TOOL intent channel (`tool_call_intent_channel_design.md`).

## Constrained tool decoding

`--constrained-tool-decoding off|tool-calls-only` (default `off`) reserves
grammar-constrained decoding of the tool wire syntax: while a tool region is being
generated, the token support is restricted to the wire grammar so the model cannot leave
it.

Status in this build: `tool-calls-only` is not implemented. The CPU grammar-state core is
implemented and tested (it tracks the marker trigger, advances on decoded bytes, and
re-validates the open region with the same strict parser in prefix mode), but the sampling
integration and its GPU verification are not part of the delivered scope, so malformed native tool
syntax can still be generated: the parser is the post-generation consistency boundary, not a
generator constraint. Selecting `tool-calls-only` therefore fails at engine startup — before any
device work, on the CLI
and the server alike — with
`--constrained-tool-decoding=tool-calls-only is not implemented in this build; use off`;
`off` is accepted and leaves the sampling path bit-identical. The design and the
acceptance gate for the sampling integration are documented in
[new_parser_phase4_design.md](new_parser_phase4_design.md).

## Diagnostics

- Failure classes: `MalformedStructure`, `InvalidToolName`, `UndeclaredTool`, `TrailingContent`,
  `TruncatedTail`, `AmbiguousStructure`. The parse entry maps these 1:1 onto the fallback reason
  recorded on the demoted text (`none`, `malformed_structure`, `invalid_tool_name`,
  `undeclared_tool`, `trailing_content`, `truncated_tail`, `ambiguous_structure`). An output
  without any marker records `none`.
- `ambiguous_structure`: the region is ambiguous — two or more Stage-2 bases complete it with
  unbalanced boundaries, or a call of a declared tool carries a synthetic argument (a repeated
  parameter name, or a non-declared name outside the first parameter of a tool with an
  unambiguous declared schema). The region is returned verbatim as text; the legacy last-value
  merge applies only to tools the contract does not declare.
- `truncated_tail`: set on a tolerant truncation that retained complete-enough calls.
- `markup_tolerant_completion`: the structured region was resolved by the Stage-2 consistent
  completion instead of the greedy Stage-1 parse.
- `fenced_markers_suppressed` / `ended_in_unclosed_fence`: complete markers a recognized code
  fence suppressed (pre-latch and retry), and whether the pre-latch stream ended inside an
  unclosed fence.
- `indented_markers_suppressed`: complete top-level markers suppressed because their `<`
  began on an indented literal line (visual column >= 4 outside a fence, pre-latch only).
  The output remains ordinary content; this is an entry-classification diagnostic, not a
  parse failure.
- `parse_budget_exhausted`: the Stage-2 work budget was exhausted and the region was returned
  as text.
- `duplicate_parameters_repaired`: counts repeated parameter names in one call of a tool the
  contract does not declare (the last value wins, as in JSON object syntax); the count is
  exposed for diagnostics.
- Tolerant truncations are logged at Info severity with the finish reason that cut the stream
  (`OutputLimit`, `ContextCapacity`, ...). A budget cut never makes an open value committable.
