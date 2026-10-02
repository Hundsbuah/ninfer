# Constrained tool decoding — design (Phase 4)

Status: CPU grammar-state core implemented and fully tested. Runtime sampling constraints (logit
masking) are not implemented in this build: the engine refuses `--constrained-tool-decoding`
values other than `off` at startup, the wire grammar is not enforced at generation time, and
malformed native tool syntax can still be generated. The tool-call parser is the post-generation
consistency boundary, not a generator constraint. This document is the P4.1 deliverable: the
sampling data flow as it exists in the source (integration points for a future implementation),
the reference findings, and the design decisions.

## 1. Current sampling data flow (NInfer source)

### 1.1 Where the logits come from

Per round the model forward produces one logit row per compact batch row:

- **Ordinary decode**: `program/decode.cpp` `ordinary_decode_batch` — the model execution writes
  `ordinary.logits` (`[1, B]` BF16, B = compact batch rows) for the round's next-token position.
- **Prefill**: `program/prefill.cpp` and `text.cpp` produce the prefill logits row
  (`kSamplePurposePrefill`) for the first generated token.
- **Speculative target verification**: `program/speculative/target_verification.cpp`
  `target_verify_batch` writes `target_logits` — the target model's logits at the draft positions
  (`[K+1]` columns per row: the K draft positions plus the continuation position).

### 1.2 Where the token is selected

`include/ninfer/ops/sampling.h` → `ops::sample` (single GPU kernel call, one row per request):

1. `adjusted = logits - penalties` (presence/frequency, per-row `SamplingConfig`);
2. greedy rows (`temperature <= 0`): argmax over the adjusted logits;
3. stochastic rows: sort → `top_k` (capped at 20) → `min_p` → `top_p` → renormalize →
   counter-based RNG draw keyed by `(seed, logical_position, purpose)` (deterministic per
   purpose, no state carried between rounds);
4. the drawn `TokenId` is the round's sampled token.

`SamplingConfig` is a device-resident standard-layout struct (`src/runtime/contract/sampling.h`)
carried inside the round ingress (`OrdinaryDecodeIngress.sampling`); the host fills it per round
and it is memcpy'd into the device round buffer (`program/decode.cpp`).

**CUDA graph**: `ops::sample` is inside the captured `ordinary_batch_body`
(`program/decode.cpp`: the lambda containing `ops::sample` is captured by `capture_graph` and
replayed by `run_prepared`). Any per-token constraint must therefore be shape-stable: fixed
addresses, no dynamic kernel count or shape change per token.

### 1.3 Speculative drafts and verification

- **Draft proposal**: the MTP/DFlash backends propose K draft tokens (neural proposal via the
  dedicated heads; ngram via the prompt cache) — see `program/speculative/draft.cpp` and the
  `SpeculativeOptions` (`include/ninfer/types.h`).
- **Target verification**: `ops::speculative_accept_sparse_drafts` /
  `ops::speculative_accept_greedy_drafts` (`target_verification.cpp`) compare the draft tokens
  against the target logits with the same sampling parameters and return the accepted prefix
  length per row. Accepted draft columns are scattered into the continuation hidden store.
- **Accept/correction/bonus tokens**: sampled from the target logits with the purposes
  `kSamplePurposeSpeculativeAccept/Correction/Bonus` (the correction token is the target's
  resample at the first rejected draft position; the bonus token is the continuation token when
  the full draft is accepted).
- **DFlash/DFlash2**: ngram/neural proposals with sparse acceptance
  (`kSamplePurposeDFlash2Proposal` for the proposal sampling).

### 1.4 Final commit

The engine round decisions (`src/runtime/engine/engine_core.h`) turn the round's sampled
accepted tokens into `decisions[row]` (terminal + finish reason); `OutputSession::commit_preview`
then commits the generated tokens into the published output. The tool-call parser sees only the
committed bytes (one-shot or the streaming decoder over the same committed sequence).

## 2. llama.cpp grammar reference (P4.2)

Primary sources read: `src/llama-grammar.cpp` / `llama-grammar.h`, `common/chat-peg-parser.h`,
`common/autoparser.md`, `common/parsers/qwen3-coder.cpp`, `common/chat-auto-parser-generator.cpp`
(also recorded in the Phase 1 research log).

- **State**: a list of rule-element stacks (NFA over code points); the stack is the grammar
  state, checkpoint = `llama_grammar_clone` (deep copy of rules, stacks, partial UTF-8),
  rollback = restoring the clone.
- **Masking** (`llama_grammar_apply`): over the *candidate* token array (the already-filtered
  top-k set), each candidate token is decoded to code points (UTF-8; a partial multi-byte
  character is carried across tokens via `grammar.partial_utf8`), the stack transition is
  simulated, and a candidate that cannot complete a valid path gets logit −inf. End-of-text is
  allowed only when some stack is empty.
- **Trigger** (lazy activation): `awaiting_trigger` mode imposes no masking; the trigger is an
  exact trigger token or a regex matched against an accumulated trigger buffer (per-token byte
  offsets are tracked so the overlapping tokens can be replayed). When the trigger fires, the
  overlapping tokens are accepted into the grammar before normal masking starts.
- **Accept**: `llama_grammar_accept_token` advances the stacks with the decoded code points of
  the finally accepted token only (never the rejected draft suffix).
- **Qwen3-Coder autoparser**: the generated grammar triggers on the complete `<tool_call>`
  marker or a complete `<function=NAME>` opener — never on a bare `<function` prefix — which
  avoids false positives on `<function` in ordinary prose.

## 3. NInfer design decisions

### D1 — the wire grammar is the state

The Phase 1/2 grammar is a deterministic byte state machine (not a GBNF NFA). The CPU
constraint state (`src/models/qwen3_5/frontend/tool_call_grammar_state.{h,cpp}`,
`ToolCallGrammarConstraint`) is byte-driven by that same grammar: `classify_tool_marker_prefix`
(P1 source of truth) decides the trigger, and `parse_tool_call_region` (the strict region
machine, exported from `tool_call_stream.{h,cpp}`) decides in-region legality. **No second
hand-written wire grammar exists** (P4.14): the constraint consumes the parser's grammar
directly, so parser and constraint cannot drift.

### D2 — lazy trigger

No restriction before a complete marker trigger (ordinary prose and reasoning blocks stay
free, matching the Qwen3-Coder reference's trigger choice). The inactive scan is a
**superset approximation** of the parser's pre-latch machine, not a byte-for-byte mirror
(R3-11, verified by probe2 P-F): since CR6 the parser runs a line-oriented fence tracker
over the pre-latch bytes (a complete marker inside a recognized fence never latches), and
since R3-05 its `NotMarker` handling publishes the failed candidate head and re-feeds the
bytes from the next `<` (a deterministic rescan) — the inactive scan runs neither, so it
can trigger where the parser does not latch (e.g. a `<tool_call>` inside a ```` ``` ````
fence).
The first trigger is the parser's latch **modulo the fence**: `<tool_call>`,
`<function_calls>`, or a complete function/invoke opener
(`classify_tool_marker_prefix == Complete`).

**After a region closes** (Round 9 R9-01) the state enters the explicit
`BetweenCalls` phase: only formatting whitespace, a legal next wrapper
(`NeedMore`/`Allowed` on a partial/complete marker), and termination stay
admissible; any visible content byte (prose, form feed, a failed marker) is
`Rejected`, and a rejected candidate never commits. The Round-8 rescan — which
admitted prose between complete calls — is removed: the strict between-calls
scan now agrees with the strict parser's `TrailingContent` demotion
(constraint `Rejected` only when the parser commits no calls), so the
asymmetry documented under R3-11 no longer exists. A candidate ending inside a
marker trigger reports `NeedMore` (legal so far; the state stays pending) —
this mirrors the grammar's own `NeedMore` for partial markers and is what
keeps a partial `<tool` in prose from being rejected.

### D3 — token/byte semantics

The constraint API takes **decoded bytes** of the candidate token (`check`/`commit` take
`std::string_view`), not token IDs: a marker spanning several tokens or a multi-byte UTF-8
character split across tokens is handled by construction (byte-prefix accumulation; the wire
grammar is byte-oriented and performs no character-set validation, consistent with the
parser). A special token whose decoded bytes form a marker behaves identically to the same
bytes from ordinary tokens (byte-level by design). The integration layer decodes candidate
tokens (tokenizer detokenization) before calling the state.

### D4 — verdict semantics

- `Allowed`: the bytes are a legal continuation; the state advances (the region may stay open
  or close).
- `Rejected`: the bytes cause a **definitive** structural break (`MalformedStructure` /
  `InvalidToolName`) at a position where a legal continuation existed before the candidate.
- `TrailingContent` (a non-marker byte after a complete call inside the latched region)
  is no longer unconstrained prose (Round 9 R9-01): the tail replays through the
  strict between-calls scanner — visible content (prose, form feed, a failed
  marker) is `Rejected`, while formatting whitespace and a legal next wrapper stay
  admissible. The strict-mode asymmetry of R3-11 is eliminated: the constraint now
  rejects exactly the suffix texts the strict parser demotes to zero calls, so
  `Rejected` and committed calls may not co-occur (the cross-check rule pinned in
  `test_tool_call_parser.cpp`). The parser remains the final authority; the verdicts
  only guide masking.
- `NeedMore`: the candidate ends inside a marker trigger; the grammar cannot decide yet (the
  bytes are legal so far; masking treats the candidate as allowed, the state stays pending).

### D5 — masking strategy decision (P4.6)

- **Strategy A (host-generated allowed set) — selected for the integration design.** Per
  round, on the host: the allowed token set is computed from the constraint state's allowed
  byte set over the vocabulary, written into a fixed-address device mask buffer that the sample
  kernel consumes as a hard support restriction. Fixed shape → CUDA-graph-safe (D8).
- **Strategy B (device-side grammar) — rejected for v1**: it moves the byte state machine into
  a kernel (complexity, register pressure, and GPU runtime verification is unavailable in this
  environment).
- **Strategy C (candidate-level filtering, the llama.cpp scheme) — rejected without proof**:
  it filters the already top-k-filtered candidate domain, which is only sound if that domain
  is guaranteed to contain the argmax-legal token; no such mathematical guarantee exists for
  the truncated candidate set after the sampling pipeline's filters. Candidate filtering would
  silently degrade to "no legal token" exactly in the regimes where the constraint matters.

### D6 — sampling order (P4.7)

The mask is a **hard support restriction applied before the penalty adjustment**: mask →
`adjusted = logits - penalties` → greedy argmax / (sort → top_k → min_p → top_p → renormalize →
draw). Rationale: a hard restriction (probability 0, logit −inf) is invariant under the
monotone penalty adjustment and the renormalization, so the constrained stochastic draw is
exactly the unconstrained draw conditioned on the legal support. For greedy rows
(`temperature <= 0`) the constrained result is the argmax over the legal support — a
deterministic parse of the most-likely legal continuation.

### D7 — speculative state semantics (P4.8/P4.9)

The constraint state commits **only with finally accepted tokens**:

- `checkpoint()` captures the state before the draft round (value semantics; a deep copy).
- The draft is simulated by committing the draft's decoded bytes on the advanced copy.
- The target verification (device) decides the accepted prefix per row.
  - Accepted prefix = full draft: keep the advanced state (the bonus token commits on it).
  - Accepted prefix < draft: `restore(checkpoint)`, commit the accepted prefix bytes plus the
    **correction token** (the target's resample at the first rejected position) exactly once.
- Per backend: ordinary decode has no draft (the sampled token commits directly); MTP uses the
  neural proposal + `speculative_accept_greedy/sparse_drafts`; DFlash/DFlash2 use ngram/neural
  proposals with sparse acceptance and the `DFlash2Proposal` purpose. The correction token must
  itself be grammar-legal: in the integration flow the host checks its decoded bytes against
  the checkpointed state before committing (a correction token that breaks the syntax is a
  contract violation of the constrained mode and is reported, not silently committed).
- **Verification status**: the state semantics (checkpoint/restore/accept/reject/correction)
  are verified by the CPU test matrix (`tests/test_tool_call_grammar_state.cpp`, P4.12 list
  items: checkpoint/restore, accepted draft prefix, rejected draft rollback, correction token
  progression). The GPU-side verification flow (device acceptance + correction sampling) is
  build-verified only — the device acceptance path is unchanged in this work; the constraint
  state is host-side. Its runtime behavior on GPU is **not verified** (no GPU runtime tests in
  this environment).

### D8 — CUDA graph safety (P4.10)

`ops::sample` is captured into the CUDA graph (1.2). The design therefore never changes shape
per token: the mask is a **fixed-address device buffer** whose *contents* update per round.
The round ingress (`OrdinaryDecodeIngress`) is already a per-round host→device memcpy with a
fixed layout; a mask field (per-row pointer into a pinned mask buffer, or an inline bitset of
the vocabulary) rides in that ingress at a fixed offset → **no extra copies, no recapture**.
Per-token recapture is explicitly not performed. The mask kernel (logit −inf for disallowed
tokens before the existing pipeline, D6) is an additional small elementwise pass over the
logit row, captured once.

### D9 — interaction with penalties and truncation (P4.10)

Penalties (`presence`/`frequency`) operate on the masked support (disallowed tokens are
already −inf and stay −inf under any finite penalty). `top_k`/`min_p`/`top_p` truncate the
masked support; the renormalization preserves the conditional distribution. Greedy rows take
the masked argmax (D6). No sampling parameter can resurrect a disallowed token: the mask is a
hard restriction, not a bias.

### D10 — performance risk (documented, P4.10/P4.13)

- **Host allowed-set computation**: computing the allowed token set from the allowed byte set
  requires scanning the vocabulary (~150k tokens × decoded-byte check) whenever the state's
  allowed byte set changes. The allowed byte set changes only on state transitions (header
  open/close, value open/close), not per byte — an incremental cache amortizes this; the cost
  is host-side CPU and off the GPU critical path (overlappable with the forward pass).
- **H2D of the mask**: a vocabulary bitset (~150k bits ≈ 19 KiB per row) via the existing
  round ingress memcpy — negligible next to the logit transfer.
- **Draft simulation**: the CPU constraint parse per draft column (K ≈ 4–16) over the region
  text — bounded by the draft window and the region size; CPU-only.
- **GPU impact not measured** (no GPU runtime tests in this environment): the additional
  elementwise mask pass and the host computation are the identified costs; the unconstrained
  path (flag `off`) is bit-identical to the current sampling (no extra kernel, no extra copy).

## 4. Feature flag (P4.11)

`--constrained-tool-decoding off|tool-calls-only` (default `off`), parsed in
`src/serve/serve_options.cpp` into `EngineOptions::constrained_tool_decoding`
(`include/ninfer/types.h`, `ConstrainedToolDecoding { Off, ToolCallsOnly }`). Naming follows
the existing kebab-case flag convention. The engine refuses any value other than `off` at
startup (before any device work), because the sampling integration is not wired in: accepting
the mode would claim a constraint the sampling path does not enforce.

## 5. Verification status (P4.13)

| Part | Status |
|---|---|
| CPU grammar-state core (`ToolCallGrammarConstraint`) | implemented, pure CPU, deterministic |
| CPU grammar logic (P4.12 matrix) | verified by `ninfer_tool_call_grammar_state_test` (16 test groups incl. every-byte-split property, checkpoint/restore, draft accept/rollback/correction) |
| Parser grammar as source of truth | verified: the constraint consumes `classify_tool_marker_prefix` + `parse_tool_call_region` directly; no second wire grammar |
| Lazy trigger | tested (partial marker → `NeedMore`; prose never rejects) — the inactive scan is a superset approximation of the parser's pre-latch machine (no fence tracker, no R3-05 rescan; R3-11), not a byte-for-byte mirror |
| Checkpoint/rollback (P4.8) | tested (value semantics; restore reproduces fresh-state behavior) |
| Speculative semantics | documented (D7); CPU state semantics tested; GPU runtime flow build-verified only, **not verified** |
| Sampling integration | **not implemented in this build**: the grammar-state core compiles into `ninfer_model_runtime`, but the sampling pipeline never consults it; the engine refuses the flag at startup (F10) |
| GPU runtime behavior | **not executed** (project constraint: no GPU runtime tests) |
| Unconstrained default path | unchanged; the existing parser/grammar/frontend test gates are green |

Remaining uncertainty: the sampling integration (host allowed-set mask, device -inf pass,
correction-token flow) is a documented follow-up requiring a GPU runtime environment to
verify. Until it lands, `off` is the only accepted value and the tool-call parser is the sole
consistency boundary for the tool wire syntax.
