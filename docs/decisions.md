# Design decisions and interview notes

Updated 2026-09-16. Consolidates the historical decision log and interview questions.

## Positioning and authorship

This is a bounded retrieval workflow with explicit tools/state and measured evidence
quality. It is not an autonomous LLM planner or persistent conversation agent. Development
used coding-agent assistance. The existing hand-v1 questions and expected answers were
authored/reviewed by the project author; generated supplements remain labelled separately.

## Decisions to explain

1. **Stable manifest IDs.** Titles can repeat across contract versions. Deterministic
   filename normalization and explicit exceptions preserve annotation-to-document identity.
2. **Section-aware, page-preserving chunks.** Keep clauses and carve-outs together while
   retaining citation locators. Paragraph/line-wrap and decimal article bugs were fixed
   before the current corpus. Some context still spans page boundaries.
3. **BM25 plus local MiniLM/FAISS.** Exact legal terms and semantic similarity complement
   one another. RRF fuses ranks because raw scores are incompatible; its constant is 60.
4. **Bounded cross-encoder reranking.** It can reorder candidates, not recover a missing
   document. Cross-document failures may need candidate allocation rather than a larger model.
5. **Deterministic search policy.** Term coverage, new evidence and budgets determine
   additional searches. This transparent policy is not LLM planning or answerability.
6. **Depth-one explicit references.** A cap can be qualified elsewhere. Follow references
   inside the same document under a chunk budget; defer deeper traversal without evidence.
7. **Whole evidence items.** Deduplicate by provenance, preserve conflicting versions and
   drop whole items instead of truncating quotations without disclosure.
8. **Citation structure versus support.** A real location can still be wrong evidence.
   The deterministic gate checks provenance; claim_support@2 sees only each claim's own
   citations. Sentence-level extraction has a stable but coarse denominator.
9. **Independent citation audit.** A disabled verifier cannot grade itself. Separate raw
   output, verification and repair. Withdraw legacy 8.6%: malformed attempts appeared in
   its error count but not its denominator. New counting includes them once.
10. **Failures are not refusals.** Timeouts, invalid JSON and fabricated evidence IDs remain
    classified failures, so a broken system cannot score as correct abstention.

## Closeout decisions recorded before live evaluation

- Add one pre-write answerability node with an independent, versioned runtime prompt.
  Validate evidence IDs and verbatim quotations in Python. Partial answers name missing
  aspects. A matching quotation proves provenance, not semantic truth. Retrieval scores
  are not calibrated answerability probabilities.
- Keep runtime gate and offline judge independent. Follow the adoption rule in the
  closeout plan; retain the gate as opt-in if the measured tradeoff fails that rule.
- Hash sorted canonical chunk content rather than IDs alone. Text/boundaries can change
  without changing IDs. Record code SHA and dirty state; never invent historical provenance.
- B toggles loop and xref independently; C separates verification and repair. Both disable
  answerability. D measures that gate alone.
- Default tests use fake embedders/rerankers, temporary API corpus settings and network
  blocking. Explicit model integration tests exercise the real dependencies separately.
- Keep only this guide, evaluation methodology and the closeout checklist. They replace
  the long specification, duplicate interview questions and obsolete authoring tutorial.
- Stop after validation and CV updates. No additional memory layer, multi-agent system,
  hosted demo, vector database or product UI is needed for this portfolio scope.

## Observed closeout decision

Experiment D fails the predeclared adoption rule: zero false answers in both arms,
over-abstention 4/36 versus 11/36, and four validation failures with the new gate.
Application tokens rise from 153,062 to 264,936. Keep `CRA_ANSWERABILITY_GATE=0` as the
default; retain the implementation and tests for explicit experiments. Higher support
among fewer delivered claims does not offset the coverage loss. B also finds no clear
answer-quality advantage from more orchestration in one trial. Do not add more layers.

All four current runs, limitations and inspected failures are linked from
[the result index](../results/README.md). The CV uses only the reproduced retrieval gain
and implemented engineering features. Closeout is complete; further benchmark expansion
is outside the agreed scope.

## Five-minute walkthrough

Use the local trace inspector or `scripts/demo.py --trace`. Explain a clause reference,
the evidence budget, answerability and the citation gate. Open the retrieval comparison
and a failed answerability case. Explain why valid citations can accompany unsupported
claims, and what each controlled experiment actually permits you to claim.

## V2: bounded tool selection and durable runs

The API/CLI default stays V1 (`CRA_AGENT_MODE=v2` opts in), while the local inspector
only displays V2 and passes an explicit per-request mode. V1 remains a backend comparison
baseline; it has no inspector controls or diagram.
The model proposes a typed action; code authorizes scope, reserves budgets, admits original
source chunks and enforces final verification. Five tools share the existing retrieval and
citation components. See [V2 design](v2_design.md) and [tool contracts](tool_contracts.md).

We chose corpus option B before implementation: keep the current index and frozen hand
suite. Generated new cases are explicitly provisional. Document content hashes identify
indexed snapshots because ingestion has no trustworthy legal version labels; we do not
invent version precedence. Character budgets apply even to the first evidence item, and
V2 citations must match a joint admitted locator. This tightens structure, not entailment.

Traces persist before SSE delivery in protected local files. Disconnects cancel with a
terminal record; reconnects cannot silently restart. Dead-process runs are marked failed
at the next startup. This does not provide crash resumption. Provider SDK retries are
disabled to keep attempts accountable. The implementation and provisional fixtures were
created with coding-agent assistance; semantic review is still separate and pending.

The inspector tails the durable journal during execution, rather than waiting for a
whole graph node to return, so model/tool start events remain visible during slow calls.
Only whitelisted tool metadata, evidence locators and budget counters reach progress
events. Mode/arm switches preserve the effective live/stub setting, including degraded
startup. The existing single-file UI uses V2 progress events without new dependencies.

## Retrieval follow-up: selected query expansion, rejected window-index adoption

The user authorized an additional retrieval optimization phase after V2 closeout.
Original query results are fused with bounded domain-vocabulary expansions, using
the existing BM25, MiniLM and cross-encoder models. For explicit comparisons only,
candidate construction balances requested documents and final selection preserves
their coverage within five original chunks. The total reranker pool stays at 40.
Expansion adds local retrieval work, no hosted model calls or new dependencies.

The selected arm achieves 72.7% document-aware evidence-probe recall@5 on 36
development questions versus 61.1% previously (9 wins, 2 losses, 25 ties). Independently,
119 source-authored CUAD queries on 23 development-disjoint documents score 71.8%
versus 70.2% (5/6/108); uncertainty intervals include zero. It is an explicit API
option, not a silent replacement of existing defaults. Full-span diagnostics and
failed categories are reported alongside the legacy probe.

Actual tokenizer measurements confirmed long input truncation, but the window-vector
ablation fell to 58.3% on development. Keep that implementation and separate index
for reproduction only; do not require it for the selected arm. No larger model,
fine-tuning, top-k inflation or paid evaluation was necessary. Source questions and
answers for independent validation were transcribed verbatim; no generated evaluation
content supplies résumé claims. Configuration was frozen before inspecting validation.

See [the full result and failure report](retrieval-improvement-results.md).

## V2 mainline: one tool executor, a local MCP server

V2 is now the only path receiving new features; V1 stays runnable as the historical
baseline at commit `01a04f8`. See the [extension plan](V2_EXTENSION_AND_EVALUATION_PLAN.md).

Before exposing tools over MCP, tool policy moved out of the graph's `execute` node into
`app/agent/tool_session.py`. `ToolExecutor` validates scope, reserves tool/search
capacity under a short lock before dispatch, applies the deadline, retries once,
rejects out-of-scope or altered retriever output, and admits evidence within item and
character limits. The graph and the MCP adapter both call it, so there is one policy
implementation. We rejected an MCP wrapper directly over `ToolRegistry`: it would have
bypassed budgets and admission, or copied them. Planning, model budgets, repair routing,
duplicate-action detection and the final-answer gate remain in the graph.

`ToolSession` is the single mutable store while a call runs. The graph rebuilds one from
its durable state for each execute node and writes it back before every emitted event,
so per-event trace snapshots are unchanged. A characterization test records normalized
event sequences and counters for 19 scenarios before the refactor. The existing V2 tests
and that golden pass without changed assertions.

The MCP server (`app/mcp_server.py`) is a stdio adapter using the official `mcp` SDK. It is
a main dependency because the server is a supported entry point. Scope comes from startup
flags and fails closed; tool arguments only narrow it. One process is one session:
evidence and counters die with it, and reconnecting resets quotas. That is a state
boundary, not identity or abuse protection. Unlike the graph, MCP reserves no call for
final verification, because nothing forces a client to verify. `verify_citation` says
explicitly that it checks structure, not semantic support. Date calculations must cite
admitted evidence; the server cannot check that it is the governing rule. `session_info`
is an adapter-only addition: it exposes granted snapshot IDs and remaining budgets so a
client does not have to guess version IDs.

Errors are MCP `isError` results carrying structured `{code, retryable, message}` and
current usage. Validation errors name fields only, never the rejected values. Logs go to
stderr; stdout carries only protocol frames. Tested with mcp 2.2.0 over stdio; the
connection negotiated protocol revision `2026-07-28`. That revision is stateless per
request, but over stdio the process still maps to exactly one connection.

## Telemetry as a projection of the durable journal

OpenTelemetry export (`app/observability/telemetry.py`, optional `otel` extra) converts a
finished run's `events.jsonl` into spans, using the recorded timestamps. It runs after the
terminal event is written. We rejected live spans inside graph nodes, which would have
duplicated every `emit()` site and put an exporter in the runtime's control path. The
cost is latency: a trace appears when the run ends, not while it streams. The local
inspector covers live viewing.

Attributes follow an explicit allowlist: IDs, locators, counts, statuses, timings, token
usage, version hashes. Questions, prompts, answers, contract text, tool arguments and
exception strings are never exported, including on error paths. A private
`TracerProvider` keeps library-emitted spans (the MCP SDK has OTel hooks) out of the
export. A `telemetry.json` marker makes export idempotent per run and records the trace
ID for correlation. OTLP/HTTP keeps Langfuse one interchangeable backend. Tests validate
the exporter against an in-process receiver and never need Langfuse. Dataset sync,
score upload and online evaluation in Langfuse are out of scope. Repository artifacts
remain the reporting source of record.

The local `.env` setup uses typed `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` fields. This is necessary because Settings rejects undeclared
fields, and reading a dotenv file does not populate the process environment used by
the OTel SDK. Keys use `SecretStr`; the application constructs the endpoint and
authentication headers and passes them directly to the exporter, without mutating
global environment variables. Generic `OTEL_EXPORTER_OTLP_*` environment settings
remain available when all three Langfuse fields are absent. The provider is cached
per process, so configuration changes require a server restart.

On 2026-10-01, manual stub and live V2 UI smoke traces were confirmed in Langfuse
Cloud. This checks delivery and execution visibility; it is not a completed live
answer-quality evaluation.

## Standalone V2 evaluation infrastructure (no completed live quality run yet)

The judge calibration split (20/21, seeded, stratified by category) and the claim
ceiling `C = 7` were fixed and checked in before the standalone live V2 evaluation. `C` came
from the grader's own claim extractor over 82 historical live answers. Batching claims
per judge call was rejected because it would have redefined `claim_support@2`. Claims
beyond `C` are routed to humans, never truncated.

Paid runs go through `evals/budget.py`. It reserves worst-case tokens and currency before
each provider attempt, keeps the reservation when usage is unknown, and persists after
every change, so a crash or resume cannot reset spend. Prices are never built in: the
preflight must state them. The ledger wraps only live clients; the offline stub makes no
provider calls, and wrapping it would break the graph's stub routing.

Human labels are the reported answer-quality outcome. Judge verdicts are written to
sealed files and compared only after labels are complete and frozen by hash. Every
unseal is logged; post-freeze corrections are adjudication records. This is procedural,
not cryptographic, and is disclosed as such. The new answer-rubric judge
(`answer_rubric@1-provisional`) gives criterion-level verdicts. Its pass criteria were
drafted with coding-agent assistance, so per `.claude/rules/evals.md` a `TODO(author)`
flag blocks the live frozen pass until the author reviews them.

## A2A entrypoint: the agent as a delegate, the tools stay on MCP (2026-10-30)

The author wanted the agent reachable by other agents, alongside the existing MCP tool
server. `app/a2a/` is a protocol adapter over the unchanged V2 graph, built on `a2a-sdk`
1.3 (spec 1.0; the public LangGraph sample still targets the 0.3 API and was used only for
structure). Division of labour: MCP serves passive, typed tools that an external model
orchestrates; A2A serves the whole agent as an opaque collaborator that owns its plan,
budgets and citation gate and reports a task lifecycle.

- **Separate process and port**, not routes on the inspector app: the inspector is
  read-only and unauthenticated, while this surface takes tasks and requires a token.
  It mirrors the MCP server being its own process. V2 is forced, because run journals,
  telemetry and cancel journaling exist only there.
- **`input-required` is an adapter scope policy.** The V2 planner never asks questions;
  it abstains. With one indexed snapshot per document, version ambiguity cannot occur;
  the real ambiguity is which contract. A request with no `document_ids` is asked for them
  before any model call. Rejected: mapping abstention to `input-required` (an abstention
  is a correct answer, rule 3), and a planner "clarify" action (changes the core and the
  frozen suites).
- **`context_id` carries no memory** (rule 6). Clarification reply text is ignored, and
  the question is always the task's first user message.
- **Two additive hooks on `ResearchAgent.research()`:** a synchronous, exception-guarded
  `progress` observer, and `external_ref` IDs copied into the manifest and onto the root
  span. Graph topology, prompts, retrieval, budgets and citation checks are untouched.
- **Static bearer token, fail closed.** This authenticates a caller; it is not per-user
  authorization. OAuth2 or mTLS would replace it in a real deployment, and the card format
  already supports declaring that.

Limits: an in-memory task store, no push notifications, and runless tasks traced only in
the server log. Design and review record: Claude Code authored the design and the
implementation, and Codex reviewed the design (two rounds). The author then narrowed the
scope.
