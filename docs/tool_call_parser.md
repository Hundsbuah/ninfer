# Tool-call parser

The Qwen frontend parses model-generated tool calls out of the assistant stream with a
single wire grammar and one incremental parser. One-shot parsing and streaming go through
the same parser: one-shot feeds the whole text in one chunk, streaming feeds chunks of any
size, and the result is independent of the chunk partition.

## Supported wire forms

The wire grammar is the single source of truth for tag literals, header syntax and marker
recognition. Accepted forms:

- Wrapper: `<tool_call> ... </tool_call>` (zero or more calls) and
  `<function_calls> ... </function_calls>` (one or more calls; the wrapper close is
  optional at the region end). A region may also start directly with a function or invoke
  opener (no wrapper).
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
- Parameter values are raw bytes: they preserve function/tool-call markers, balanced nested
  `<parameter=...>...</parameter>` text, and any other markup as value bytes.

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
verbatim content. A failed region that broke with a wrapper still open owns the bytes up
to its break as the failed structure: a later wrapper literal opens a fresh top-level
region there, but a bare function/invoke opener in that scope is never a recovery entry
(it would reinterpret nested payload as a new call), and a wrapper that opened inside the
failed wrapper (a nesting break) is skipped. Wrapper nesting — a second wrapper open
before the first one closed — is a structural break in both modes.

While scanning for the first marker, a candidate broken by a byte the grammar classifies
as NotMarker is published up to that byte; when the breaking byte itself is `<`, it is
retained as the start of a fresh candidate, so a real marker inside a failed prefix
(`prefix <function<tool_call>...`) still latches at the machine's latch byte. A `<` the
header grammar still accepts (a quoted attribute value) keeps the candidate open instead
of breaking it.

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
before its wrapper close, are retained with a `truncated_tail` diagnostic. A name-only
truncation (no closed parameter and no function close) still falls back to text: recovery
- The empty `<function_calls>`
  wrapper is unrecoverable in both modes; a `<function_calls>` region holds a sequence of
  calls (another call may follow any function close) and may end at the region end after
  its last invoke close (the wrapper close is optional there, a clean completion).

## Fundamental delimiter ambiguity

The wire format has no delimiter escape. A `</parameter>` inside a value is a real closer
only when the bytes after it form a legal continuation, classified by the same wire
grammar: another parameter opener, the function's closer followed by the wrapper close or
the region end, or the end of the input. A closer at the input end is a provisional
boundary: the parameter commits, but a function that never closes stays non-executable.
Any other continuation (immediate markup that is not a structural token, a quoted closer
immediately followed by the next token) is value text — a shell command that echoes the
markup. Inside a `<tool_call>` wrapper the rule is stricter: a `</parameter>` followed by
`</function>`/`</invoke>` is a real closer only when the wrapper close or the region end
follows the function close.

Because of this boundary, the parser does not claim that arbitrary strings are safe values:
a value containing a markup sequence that matches the closer rule at its end is ambiguous by
construction, and such a region degrades to content instead of guessing.

## Constrained tool decoding

`--constrained-tool-decoding off|tool-calls-only` (default `off`) reserves
grammar-constrained decoding of the tool wire syntax: while a tool region is being
generated, the token support is restricted to the wire grammar so the model cannot leave
it.

Status in this build: `tool-calls-only` is not implemented. The CPU grammar-state core is
implemented and tested (it tracks the marker trigger, advances on decoded bytes, and
re-validates the open region with the same strict parser in prefix mode), but the sampling
integration and its GPU verification are not part of the delivered scope. Selecting
`tool-calls-only` therefore fails at engine startup — before any device work, on the CLI
and the server alike — with
`--constrained-tool-decoding=tool-calls-only is not implemented in this build; use off`;
`off` is accepted and leaves the sampling path bit-identical. The design and the
acceptance gate for the sampling integration are documented in
[new_parser_phase4_design.md](new_parser_phase4_design.md).

## Diagnostics

- Failure classes: `MalformedStructure`, `InvalidToolName`, `UndeclaredTool`, `TrailingContent`,
  `TruncatedTail`. The parse entry maps these 1:1 onto the fallback reason recorded on the
  demoted text.
- `truncated_tail`: set on a tolerant truncation that retained complete-enough calls.
- `duplicate_parameters_repaired`: counts repeated parameter names in one call (the last
  value wins, as in JSON object syntax); the count is exposed for diagnostics.
- Tolerant truncations are logged at Info severity with the finish reason that cut the stream
  (`OutputLimit`, `ContextCapacity`, ...). A budget cut never makes an open value committable.
