# Tool-call parser

The Qwen frontend parses model-generated tool calls out of the assistant stream with a
single wire grammar and one incremental parser. One-shot parsing and streaming go through
the same parser: one-shot feeds the whole text in one chunk, streaming feeds chunks of any
size, and the result is independent of the chunk partition.

## Supported wire forms

The wire grammar is the single source of truth for tag literals, header syntax and marker
recognition. Accepted forms:

- Wrapper: `<tool_call> ... </tool_call>` around one or more calls. A region may also start
  directly with a function or invoke opener (no wrapper).
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

## Strict vs. tolerant

Both modes parse with the same grammar and the same state machine; the policy changes only
what an incomplete or trailing region commits.

Strict (default): all-or-nothing. A region that does not end cleanly falls back to verbatim
content with the precise reason (`MalformedStructure` for a cut or broken region, `TrailingContent`
for content after complete calls, `UndeclaredTool` for a name outside the declared tools).

Tolerant (`--tolerant-tool-calls`): retains complete calls that appear before a later broken
call, and for a single final call cut by the output budget commits the call when every
committed parameter value is unambiguously closed (a call with at least one closed parameter
and no still-open value). The recovered call is reported structurally with a `truncated_tail`
diagnostic. A name-only truncation (no closed parameter) still falls back to text: recovery
never commits an argument whose bytes are not unambiguously closed, because every shipped
tool takes its payload from a string value. The empty `<function_calls>` wrapper is
unrecoverable in both modes.

## Fundamental delimiter ambiguity

The wire format has no delimiter escape. A `</parameter>` inside a value is a real closer
only when whitespace and then another parameter opener, the function's closer, or the end of
the output follow it; any other occurrence is value text (a shell command that echoes the
markup). Inside a `<tool_call>` wrapper the rule is stricter: a `</parameter>` followed by
`</function>`/`</invoke>` is a real closer only when the wrapper close or the region end
follows the function close. A quoted closer that is immediately followed by the next token is
ambiguous and makes the region ordinary content rather than a tool call.

Because of this boundary, the parser does not claim that arbitrary strings are safe values:
a value containing a markup sequence that matches the closer rule at its end is ambiguous by
construction, and such a region degrades to content instead of guessing.

## Constrained tool decoding

`--constrained-tool-decoding off|tool-calls-only` (default `off`) reserves grammar-constrained
decoding of the tool wire syntax: while a tool region is being generated, the token support is
restricted to the wire grammar so the model cannot leave it. With `off` (the default) the
sampling path is unchanged.

Status: the CPU grammar-state core is implemented and tested (it tracks the marker trigger,
advances on decoded bytes, and re-validates the open region with the same strict parser in
prefix mode, where a structure cut at the input end is a legal partial prefix). The sampling
integration and its GPU runtime verification are not part of the delivered scope; the design
and the acceptance gate for that change are documented in
[new_parser_phase4_design.md](new_parser_phase4_design.md).

## Diagnostics

- Failure classes: `MalformedStructure`, `InvalidToolName`, `UndeclaredTool`, `TrailingContent`,
  `TruncatedTail`. The parse entry maps these 1:1 onto the fallback reason recorded on the
  demoted text.
- `truncated_tail`: set on a tolerant truncation that retained complete-enough calls.
- `duplicate_parameters_repaired`: counts repeated parameter names in one call (the first
  value wins); the count is exposed for diagnostics.
- Tolerant truncations are logged at Info severity with the finish reason that cut the stream
  (`OutputLimit`, `ContextCapacity`, ...). A budget cut never makes an open value committable.
