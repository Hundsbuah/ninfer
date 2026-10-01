# NInfer tool-call TEXT-vs-TOOL intent channel — long-term architecture design

Status: **design only (Round 6, R6-06)**. No production behavior change, no code. Nothing here
is validated against the model; every open item requires the experimental evidence named in the
section before any implementation may proceed. The current parser remains the validation
boundary even after every part of this design is in place.

## 1. Problem

The Qwen3.8 native wire format transmits tool-call intent through ordinary content bytes
(`<tool_call>...</tool_call>`). A byte parser receives those bytes and cannot distinguish
"execute this" from "show this as an example" when no surrounding syntax contradicts one
reading. Round 6 mitigates the prose-prefixed quotation class (`--tool-call-intent
start-of-content`) and pins the example-only residual
(`test_r6_complete_unfenced_in_set_example_residual`), but the example-only case remains
fundamentally ambiguous on the wire. Eliminating the class requires intent to arrive through a
channel that ordinary content cannot accidentally reproduce.

## 2. Candidate intent signals

| Signal | Reproducible by content? | Notes |
|---|---|---|
| Plain text sentinel (e.g. `<tool_mode>`) | **Yes** — quotable, fencable, prefix-able | Insufficient by itself (§8.3 of the Round-6 spec). Rejected as the sole mechanism. |
| Reserved tokenizer special token | No (if the tokenizer never emits it as ordinary content) | Strongest signal available without retraining. Requires experimental validation that Qwen3.8 emits/obeys a custom special token (see §3). |
| Structured decoder state / side channel (sampler-mode, output head) | No — not emitted into content | Most robust; largest integration surface (sampler + streaming API + serialization). |
| Model output head (trained classifier) | No | Requires fine-tuning; out of scope for this project. |

A robust design should treat the special token as the **generation-side signal** and the sampler
state machine as the **enforcement-side state**, with the parser remaining the final structural
validator. The two are complementary, not alternatives:

```text
model output intent (special token / sampler state)
   ↓
TEXT ----------------→ content (tool markup stays content)
   |
   └── TOOL
         ↓ grammar-constrained generation (legal TOOL bytes)
         ↓ native Qwen tool grammar (structure)
         ↓ strict structural parser (final validation)
         ↓ declared-tool enforcement (contract)
         ↓ GeneratedToolCall
```

## 3. Tokenizer / special-token feasibility (open)

- The Qwen3.8 tokenizer vocabulary and reserved-token budget must be inspected: a custom intent
  token needs an unused token id that is (a) never produced by ordinary chat continuation
  training, (b) not mapped to a printable sequence the model can be quoted into, and (c)
  addressable by the sampler for masking/injection.
- Without fine-tuning, a model does not know to emit a custom special token. Realistic options:
  1. **Constrain the model to a known control sequence** (e.g. a special token id already in the
     vocabulary with a printable stand-in) and treat the *first byte of the assistant content
     channel* as the intent decision point: the sampler masks all other first-token options when
     the turn is tool-capable. This is a constrained-decoding extension, not a new token.
  2. **Prompt-engineered control token**: the chat template ends the assistant turn with a
     reserved control token; the model's next token is either that token's "text continuation"
     or the `<tool_call>` opener. The template (not the model) decides the channel.
- **Experimental work required** (must run on GPU, out of scope for Round 6):
  - emission rate of the control token on tool-capable vs text prompts;
  - whether masking the control token degrades legitimate text turns;
  - behavior under long reasoning (does the token survive `</think>` closes?);
  - corpus of coding tasks to measure legitimate-call loss (same metric set as the R6-05
    rollout recommendation).

## 4. Chat-template changes

- The assistant generation suffix gains the intent decision point (control token or a
  template-rendered mode hint that the sampler, not the parser, interprets).
- The template remains a **prompt-side** artifact: it never changes what bytes the parser
  accepts. Template changes ship together with sampler-state changes, never with parser changes.
- Upstream Qwen template compatibility: a `template-compatible` profile must keep rendering the
  unmodified upstream suffix (the local `tools/chat_templates/qwen3_8.jinja` hardening is an
  NInfer extension, documented as such).

## 5. Model behavior without fine-tuning

- Baseline expectation: the model follows the template's explicit instruction; corner cases
  (Qwen's own function-calling docs acknowledge malformed emissions) remain possible.
- The intent signal must therefore be **enforced by the runtime**, not trusted from the model:
  the sampler state machine decides the channel, and the parser validates structure. A model
  that emits tool markup in a TEXT-decided turn gets content; a model that emits prose in a
  TOOL-decided turn gets a structural failure surfaced to the client.
- No design here may assume the model emits a custom token reliably. That assumption requires the
  §3 evidence first.

## 6. Sampler state machine

Per active request (one compact decode batch per round; no preemption), the sampler carries:

```text
TurnIntentState {
    enum phase { TextOpen, ToolGateOpen, ToolActive, LockedText };
    // TextOpen: no intent decision yet (only reasoning/formatting whitespace consumed);
    // ToolGateOpen: tool-capable, control decision still available;
    // ToolActive: TOOL decided, grammar-constrained generation active;
    // LockedText: TEXT decided (or tool region closed), markup is content.
}
```

- Transitions are driven by decoded bytes (the same pre-latch gate the Round-6
  `RequireToolAtContentStart` mode already computes in the CPU path) plus the special-token
  decision when §3 validation passes.
- The state is **per-request** and lives with the request's execution context (checkpoint /
  restore must carry it — value-semantic, like `ToolCallGrammarConstraint` after R6-01).
- Constrained decoding (grammar constraints on legal TOOL bytes) attaches only in
  `ToolActive`; the CPU constraint state already follows the selected syntax mode (R6-01), so
  future sampler integration cannot mask tokens for a region the parser would never execute.
- Until runtime constrained sampling ships (R6-07), the state machine's only observable effect
  is the intent decision; `--constrained-tool-decoding` remains fail-fast.

## 7. Streaming API representation

- The public streaming surface (`include/ninfer/engine.h`, output channels) today publishes
  `Reasoning` and `Content` channels plus terminal tool calls. The intent decision is **not** a
  new client-visible channel: a TOOL-decided turn publishes no content until the terminal tool
  call; a TEXT-decided turn publishes content that may include tool markup verbatim.
- One new observable is acceptable: a terminal diagnostic flag
  (`tool_intent_decided_tool` vs `text`) in the existing diagnostics structure, for client
  observability and for the §11 fallback accounting.
- No partial tool calls mid-stream (current product contract: tool calls are terminal).

## 8. Responses / Chat / Anthropic serialization

- OpenAI Responses / Chat and Anthropic surfaces serialize assistant tool calls as structured
  `tool_call` items. Under the intent channel, the serialization rule is unchanged: only
  TOOL-decided, parser-validated calls serialize as tool items; everything else serializes as
  text. The intent decision is an NInfer runtime concept and does not appear in the wire
  protocol.
- History replay (§7) depends on this: replayed assistant turns are re-parsed through the same
  decision point, so a stored turn's intent is re-derived deterministically from stored bytes +
  the turn's stored intent diagnostic (the diagnostic must therefore be persisted with the turn
  when stateful caching is active).

## 9. History replay

- Assistant turns in the chat history are parsed with the **same** syntax/ambiguity/intent
  policy as live generation (single policy per session, R6-I1 extended to history).
- A turn that was TOOL at generation time replays as TOOL: the stored tool-call items are used
  directly (not re-parsed from raw bytes), so history cannot lose a call to a later policy
  change within one session; cross-session policy changes are a product decision recorded in
  the session config.
- Open: whether replayed tool-call items should be re-validated against the current declared
  tool set (declared tools may differ per request today). Decision: **validate on replay**;
  a replayed call that is undeclared under the current set degrades to its raw text form.

## 10. Stateful response caching

- Stateful response caching stores the published response (content + tool calls + diagnostics).
  The intent diagnostic travels with the cached response (see §8), so a cache hit reproduces the
  same channel decision without re-running the sampler state machine.
- The cache key must include the intent policy (like it already must include syntax/ambiguity
  once those affect parseable output), because the same bytes under different policies produce
  different published results.

## 11. Prefix-cache interaction

- Prefix caching (KV reuse) is keyed on prompt bytes; the assistant-turn generation suffix
  (including any §4 control construct) is part of the prompt. A template change that moves the
  intent decision point changes the prompt → the prefix cache invalidates naturally; no special
  handling required.
- The sampler state machine is per-request and stateless across requests: prefix-cache hits
  never carry intent state, only KV. Checkpoint/restore within a request carries the state
  (see §6); speculative-decode branches must fork the state machine per branch and roll back on
  rejection (value-semantic state makes this a copy).

## 12. Fallback behavior if the intent token never appears

- If the special-token mechanism is adopted and the model never emits the control token in a
  tool-capable turn, the runtime must fall back deterministically. Two candidates:
  1. **Treat as TEXT** (safe default): the turn publishes content; tool markup stays content.
     Cost: silent loss of legitimate tool calls if the model simply refused the control token.
  2. **Treat as TOOL gate open** (current R6-05 `TemplateCompatible` behavior): the first valid
     marker latches.
- Recommendation: fall back to the session's explicit `ToolCallIntentPolicy` (i.e. the R6-05
  mode) — the product decision stays where it already is — and record a diagnostic so the
  fallback rate is measurable. The §3 experimental corpus must measure exactly this rate before
  the token mechanism ships.

## 13. Compatibility with existing clients

- Clients that never enable tool calls are unaffected (no tool-capable turn, no intent
  decision).
- Clients with tool calls enabled see unchanged wire behavior under `TemplateCompatible` (the
  compatibility default); the intent channel is a per-session opt-in, like R6-05.
- The OpenAI/Anthropic protocol surfaces are external contracts: the intent decision changes
  **what** serializes (tool item vs text) but not **how**; any policy that makes a previously
  committed call text is a behavior change that must ship with serving documentation updates
  (the same rule the Round-5 protocol-ambiguity change followed).

## 14. Sequencing

1. Round 6 (this round): R6-05 hardening mode + pinned residual (done).
2. Experimental GPU validation of §3 (separate task, requires GPU budget).
3. Sampler state machine + `--constrained-tool-decoding` runtime integration (R6-07 follow-up;
   the CPU constraint state is already syntax-coherent after R6-01).
4. Template/sampler intent decision point + diagnostics + serialization persistence.
5. Client-visible opt-in flag + protocol documentation.

Until step 2 has evidence, steps 3–5 must not merge an unvalidated custom intent token into the
production parser.
