# Eval-Driven Contract Research Agent

A bounded contract QA system over public CUAD/ContractNLI data and labelled synthetic
variants, with deterministic citation checks and an auditable evaluation workflow.

Start with the [plain-language project guide](docs/project-guide.md) to understand
the agent and its evaluations. The [documentation index](docs/README.md) separates
current instructions from historical plans; the [UI and Langfuse walkthrough](docs/ui-langfuse-walkthrough.md)
provides a hands-on exercise.

**V2 is the actively developed path; select it with `CRA_AGENT_MODE=v2`.** It adds
model-selected typed tools, code-enforced document/snapshot scope, execution budgets, a
mandatory citation gate, and persistent local traces. The same tools are also served to
external agents through a [local read-only MCP server](#use-the-tools-from-an-mcp-client).
V1 remains the runtime default and the historical comparison baseline until live V2
evaluation is complete; it receives no new features. The V1 reference revision is commit
`01a04f8`, reproducible with `git checkout 01a04f8 && uv sync --locked` and the commands
in [Reproduce the comparisons](#reproduce-the-comparisons). Result tables below that
predate V2 describe that V1 path. The existing **41-question author-reviewed
regression suite** is unchanged; **24 new provisional cases** exercise definitions,
snapshot comparison, date calculations, and unsupported inputs.

The **36 retrieval questions are the answerable subset of those same 41 cases**;
the other five expect abstention and have no evidence to retrieve. The separate
119-question CUAD suite validates retrieval on development-disjoint documents.

See [V2 implementation and limitations](docs/v2_design.md),
[measured validation status](reports/v2_results.md), and
[three synthetic trace demonstrations](examples/traces/README.md).

## Quickstart

```bash
uv sync --locked
CRA_LIVE_MODEL=0 uv run pytest

# Acquire the public datasets and rebuild the local corpus/index.
CRA_LIVE_MODEL=0 uv run python scripts/download_cuad.py
CRA_LIVE_MODEL=0 uv run python scripts/download_contractnli.py
CRA_LIVE_MODEL=0 uv run python scripts/sample_contracts.py --size 40
CRA_LIVE_MODEL=0 uv run python scripts/sample_contractnli.py --size 10
CRA_LIVE_MODEL=0 uv run python scripts/build_synthetic.py
CRA_LIVE_MODEL=0 uv run python scripts/ingest.py
CRA_LIVE_MODEL=0 uv run python scripts/build_index.py

CRA_LIVE_MODEL=0 uv run python scripts/demo.py --trace "What limits liability, and does it always apply?"
CRA_LIVE_MODEL=0 uv run uvicorn app.api.main:app --reload
```

The local trace inspector is at `http://localhost:8000`. It streams actual LangGraph
node events over SSE and displays selected evidence and citation checks. `docker compose up`
is the alternative local reproducibility path; corpus files are mounted from `data/`.

**Model access is free by default.** Set `CRA_LIVE_MODEL=1` and `ANTHROPIC_API_KEY` only for
an intentional live run. A local `.env` can override defaults, which is why free commands
above set the flag explicitly. Stub answers exercise wiring and cannot measure semantics.
Normal pytest blocks external sockets and model loading; real model tests are opt-in:

```bash
CRA_LIVE_MODEL=0 uv run pytest -m model_download
CRA_LIVE_MODEL=0 uv run pytest -m corpus       # downloaded ContractNLI required
```

On macOS, if the native FAISS/BLAS stack crashes or oversubscribes threads, run retrieval
with `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false`.
The closeout retrieval experiment used these limits and reproduced the prior scores.

## Run V2

```bash
# Offline fixture planner: tests wiring, not answer quality.
CRA_LIVE_MODEL=0 CRA_AGENT_MODE=v2 uv run python -m scripts.demo --trace --docs doc-001 "What limits liability?"
CRA_LIVE_MODEL=0 CRA_AGENT_MODE=v2 uv run uvicorn app.api.main:app
CRA_LIVE_MODEL=0 uv run python -m scripts.v2_demo

# Separate paired artifacts: never pool the old and provisional suites.
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_comparison --suite hand
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_comparison --suite v2
```

`GET /documents` exposes `version_id` content hashes; `POST /research` accepts an optional
`requested_versions` mapping. These identify indexed snapshots, not legal version order.
The local inspector displays only V2 and explicitly requests
`GET /research/stream?mode=v2`, independently of the server's default mode. Changing
retrieval preserves the server's live/stub setting. V2 SSE publishes `start`,
redacted `progress` events with event IDs, then an authorized final `done` response.
The diagram, execution log, evidence locations and budget counters follow these events;
slow model calls publish their start before they finish. Stop closes the stream and
cancels in-flight work. Reconnects carrying `Last-Event-ID` receive
409 instead of silently starting another run. Private `runs/<id>/` files are gitignored.

Click **Try insurance example**, then **Ask** to search all documents in V2 using
`rrf_hybrid_rerank`: “What coverage must Aimmune keep in place while the product is being
tested or sold?” Inspect `doc-011`, page 38, §11.5 for clinical-trial/product-liability
insurance. Coverage amounts are redacted in the source and must not be invented. The
header distinguishes live answers from simulated stub answers; this example is a manual
demo, not an additional evaluation result.

Evidence cards distinguish cited passages from uncited context. Citation checks validate
locations and admission, not semantic support for every claim. Literal `[***]` source
redactions are preserved in answer rendering. Preparing the example does not erase the
previous run's graph; a new **Ask** resets the displayed run.
The final SSE response includes `execution_path`, derived from persisted events. The UI
rebuilds the completed route from it, retaining highlighted nodes, directional edges,
visit counts and a text route even if intermediate UI updates were missed. Unvisited
branches remain gray.

For actual model-selected routing, explicitly enable `CRA_LIVE_MODEL=1`. This adds one
planner call per decision (up to the shared 12-model-call ceiling, including writing and
repair); calls and tokens are recorded. No paid V2 run was used for the checked-in offline
report. The [evaluation protocol](docs/evaluation_protocol.md) explains human labels and
why stub/provisional results cannot become resume claims.

## Use the tools from an MCP client

`app/mcp_server.py` serves the five V2 tools, plus `session_info`, over local stdio. It
calls no model. Scope is granted when the server starts, never by a tool argument, and it
refuses to start without a grant:

```bash
# Scripted external client: launches the server as a subprocess and exercises every tool,
# then shows refused requests and what the server cannot enforce.
CRA_LIVE_MODEL=0 uv run python scripts/mcp_demo.py --doc doc-001

# Register with Claude Code (any stdio MCP host takes the same command).
claude mcp add contract-research -- \
  uv run --directory "$PWD" python -m app.mcp_server --allow doc-001 --allow doc-011
```

Options: `--allow DOC_ID` (repeatable) or `--allow-all`; `--pin DOC_ID=VERSION` refuses
to serve if the indexed snapshot changed; `--retriever` (default `bm25`); `--call-timeout`
(default 30 s); optional `--session-lifetime`; `--max-tool-calls`/`--max-searches`
(defaults match the graph's budget).

Guarantees and limits:

| Enforced by the server | Outside the server's control |
| --- | --- |
| Granted documents/snapshots; arguments only narrow them | The external model's own calls and token spend |
| Tool/search budgets, per-call timeout, no refunds on failure | The external agent's overall deadline |
| Evidence admission and provenance; fabricated or other-session IDs rejected | Whether the client calls `verify_citation` at all |
| Returned text bounded by the admitted-evidence limits | Whether the client uses the verification result |
| Date arithmetic linked to admitted evidence; business-day calendars rejected | Whether that evidence is the governing rule |
| Per-session quotas within one process | Usage across reconnects: a new process starts fresh |

`verify_citation` reports `structurally_valid` and `semantic_support: "not_assessed"`:
a citation that resolves to admitted text may still not support the claim. The graph's
mandatory final-answer gate does not apply to MCP clients. Policy is not duplicated: both
the graph and the MCP adapter call the same executor (`app/agent/tool_session.py`). The
MCP SDK is pinned in `uv.lock` (mcp 2.2.0); the tested stdio connection negotiated protocol
revision `2026-07-28`. The scripted SDK client is the verified external client; the Claude
Code registration above is documented but not part of automated tests.

## Export V2 traces to Langfuse (optional)

Every V2 run already writes a durable local journal (`runs/<id>/events.jsonl`). With the
optional `otel` extra, a finished run is also projected into one OpenTelemetry trace:
an `invoke_agent` root span, one `chat` span per model call, one `execute_tool` span per
tool attempt (with admitted-evidence events), and a `verify_citations` span. Timings come
from the journal. Export is off by default and needs no network in tests.

```bash
uv sync --locked --all-groups --extra otel
```

Fill the Langfuse project settings in your gitignored `.env` (replace the key
placeholders), then restart the process:

```dotenv
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=YOUR_PUBLIC_KEY
LANGFUSE_SECRET_KEY=YOUR_SECRET_KEY
CRA_TELEMETRY_EXPORT=1
```

The base URL is the host, without `/api/public/otel`. Typed settings construct the
OTLP endpoint and authentication directly; no shell header encoding is needed. Keep
export disabled while keys are blank. Generic OTLP backends remain supported through
the standard `OTEL_EXPORTER_OTLP_*` shell environment when all three Langfuse fields
are absent. Do not put undeclared OTEL variables in the application's `.env`.

```bash
CRA_LIVE_MODEL=0 CRA_AGENT_MODE=v2 uv run python -m scripts.demo --docs doc-001 "What is the termination notice period?"
# Or export runs that already finished (each run is exported at most once):
uv run python -m scripts.export_run_telemetry --latest 3
```

Only allowlisted metadata leaves the machine: run/call/evidence IDs, locators, counts,
statuses, timings, token usage and version hashes. Questions, prompts, answers, contract
text, tool arguments and exception messages are never exported, so Langfuse shows the
shape, cost and latency of a run, not its content. `runs/<id>/telemetry.json` records the
exported trace ID; search Langfuse for the `cra.run_id` attribute to go the other way. A
failed or slow collector cannot change a run: export happens after the terminal event is
written, in a bounded background queue. The local journal and inspector stay the source
of record. Attribute names follow the OpenTelemetry GenAI conventions (`gen_ai.*`), with
project fields under `cra.*`.

## Standalone V2 evaluation workflow

`experiments/v2_standalone.py` runs V2 alone over hand-v1, saving each case as it finishes.
It resumes safely and records unexecuted cases. `experiments/v2_judge.py` runs
`claim_support@2` and a criterion-level answer-rubric judge; their verdicts are sealed
until human labels are frozen with `scripts/v2_labels.py`. Paid runs need a budget
preflight (`evals/preflight/`), enforced per provider attempt by a persisted ledger. The
[standalone protocol](docs/evaluation_protocol.md#standalone-v2-evaluation-hand-v1) has the
frozen split, the claim ceiling `C = 7`, the call ceilings and the full command sequence.
No live V2 evaluation has been run yet; no V2 quality number is reported.

## Current results

**Retrieval update (2026-09-26):** the optional `legal_query_coverage_rerank` arm raises
document-aware evidence recall@5 from **45.8% (BM25) / 61.1% (previous hybrid + rerank)
to 72.7%** on 36 development questions (15/2/19 and 9/2/25 wins/losses/ties).
On **119 original CUAD questions across 23 development-disjoint contracts**, it scores
**71.8%**, versus 65.2% / 70.2% for those baselines (19/15/85 and 5/6/108).
One trial, same corpus, scopes and five original chunks throughout. Independent gains
are modest and uncertain; category regressions remain. This is evidence-location
recall, not answer accuracy. See [full ablations, complete-span metrics and failures](docs/retrieval-improvement-results.md)
and [résumé wording](docs/resume-bullets.md). Existing runtime defaults are preserved.

All results below use `hand-v1`, the current 6,270-chunk corpus, k=5 and one trial.
This is a small development benchmark, not an unseen test set or a production SLA.
See [the artifact index](results/README.md), [generated tables](results/RESULTS.md), and
[evaluation definitions](docs/eval-methodology.md).

### A: retrieval quality

36 scored questions; five abstention tasks excluded because they have no expected evidence.
Document allowlists match the scoped API behavior. [Current artifact](results/experiment-a-current.json).

| Retrieval arm | Document recall | Evidence-span recall@5 | MRR |
|---|---:|---:|---:|
| BM25 | 0.944 | 0.458 | 1.000 |
| MiniLM + FAISS | 0.843 | 0.394 | 0.889 |
| RRF hybrid | 0.944 | 0.495 | 1.000 |
| RRF + cross-encoder rerank | 0.944 | 0.611 | 1.000 |

Reranking raises evidence-span recall by **15.3 percentage points** versus BM25 on these
36 questions: **8 wins, 2 losses, 26 ties**. Document recall alone hides that difference.
Cross-document comparison remains weak: both BM25 and rerank score **0.25** span recall
on eight tasks. Reordering cannot recover a document absent from the candidate pool.

### B–D: current controlled comparisons

All three use Sonnet 5 and 41 frozen development questions. Full outputs, tokens and
paired comparisons are pinned in the [artifact index](results/README.md).

- **B — loop × cross-reference:** required-point coverage is 0.602 / 0.628 / 0.632 /
  0.614 for single/no-xref, single/xref, loop/no-xref and loop/xref. Loop and xref main
  effects are only +0.008 and +0.004; retrieval recall is unchanged. This run does not
  establish a reliable answer-quality benefit from additional orchestration.
- **C — citation verification:** delivered references pass the independent structural
  audit at 56/59 ungated, 53/53 verify-only and 63/63 verify+repair. Repair succeeds in
  1/9 triggered tasks. Each arm has its own stochastic initial drafts, so this is not
  a replay of identical answers. Structural validity is not semantic correctness.
- **D — answerability:** both arms abstain on all five expected-abstention tasks.
  On 36 answerable tasks, explicit over-abstentions increase from 4 to 11, with four
  additional execution failures when enabled. Runtime tokens increase from 153,062
  to 264,936. **The gate fails the adoption rule and is disabled by default.**

Behavior counts use runtime status: a `completed` response can contain a prose refusal.
The independent claim judge and [three inspected failures](results/ANALYSIS.md) expose
that limitation. No answer-quality improvement is inferred from retrieval scores.

### Historical evidence

Old live runs used 7,453 chunks and `claim_support@1`, which tested support anywhere in
the selected evidence pool. Those scores are not current `claim_support@2` scores.
Historical JSON is retained in `results/` for auditability.

The old citation ablation's **8.6% rate is withdrawn**: it counted malformed citation
errors in the numerator but omitted those attempts from the denominator. The new ablation
counts malformed attempts explicitly and independently audits raw and delivered answers.
A valid citation location still does not prove that the associated claim is supported.

## Workflow and constraints

```text
question -> search -> assess -> refine/search (bounded)
                       |
                       v
                 resolve references -> select evidence
                       |
                       v
                 answerability (opt-in) -> abstain / write supported scope
                                             |
                                             v
                                      citation verifier
                                      /       |       \
                                  answer   repair x1   abstain
```

- **Retrieval:** BM25, normalized MiniLM embeddings in FAISS, reciprocal-rank fusion and
  cross-encoder reranking. The API and experiments expose all four arms.
- **Context:** section-aware chunks, provenance headers, whole-item evidence budgets,
  deduplication and depth-one clause-reference expansion. Conflicting versions stay distinct.
- **Answerability (opt-in, `CRA_ANSWERABILITY_GATE=1`):** a separate pre-write JSON verdict identifies supported/missing aspects
  and quotes evidence IDs. Python validates the IDs and exact quoted text. Partial answers
  carry their support scope through citation repair. Invalid verdicts remain execution failures.
- **Citation verification:** document/page/section existence, document allowlists and presence
  in retrieved evidence are checked in code, with at most one repair. This is structural
  enforcement; semantic support is scored independently with `claim_support@2`.
- **Execution:** search/iteration/reference/evidence/repair limits and model timeouts are
  enforced in code. Each result records model calls, tokens, elapsed time and trace events.

## Reproduce the comparisons

Every live script should first be exercised with `CRA_LIVE_MODEL=0` and a small `--limit`.
B turns loop and cross-reference on/off independently. C separates no verification,
verification only and verification plus repair. B/C disable answerability to isolate their
named factors. D measures answerability off/on with an independent offline claim judge.

```bash
CRA_LIVE_MODEL=0 uv run python experiments/retrieval_comparison.py --suite hand
CRA_LIVE_MODEL=0 uv run python experiments/agentic_comparison.py --suite hand --limit 2
CRA_LIVE_MODEL=0 uv run python experiments/citation_gate_ablation.py --suite hand --limit 3 --fault-every 2
CRA_LIVE_MODEL=0 uv run python experiments/answerability_comparison.py --suite hand --limit 2

# Explicit paid runs over public benchmark excerpts; reports stay local.
CRA_LIVE_MODEL=1 uv run python experiments/agentic_comparison.py --suite hand --output results/experiment-b-factorial.json
CRA_LIVE_MODEL=1 uv run python experiments/citation_gate_ablation.py --suite hand --output results/experiment-c-isolated.json
CRA_LIVE_MODEL=1 uv run python experiments/answerability_comparison.py --suite hand --claim-support --output results/experiment-d-answerability.json
uv run python scripts/summarize_results.py
```

Reports identify code/worktree state, source and corpus hashes, dataset/model/grader versions,
budgets, per-task outputs and paired comparisons. D records offline judge tokens separately
from application tokens and latency. One trial supports descriptive comparisons only.

A separate [20-task ContractNLI diagnostic](evals/datasets/answerability/README.md) transcribes
expert hypotheses/labels but uses an AI-authored question wrapper. Its version begins
`provisional-generated-`; it is supplementary validation and never supplies resume metrics.

## Scope and limitations

The corpus is small and annotation-selected. General-purpose embedding/reranking models,
limited candidate coverage, depth-one references, unresolved definitions and sentence-level
claim extraction constrain quality. The runtime gate and offline judge use independent
prompts but the same model family; both can misjudge semantics. No zero-hallucination,
production-readiness or held-out generalization claim is made.

## Maintained documentation

- [Documentation index](docs/README.md): reading order, current references and archives.
- [Project guide](docs/project-guide.md): plain-language architecture and evaluation map.
- [UI and Langfuse walkthrough](docs/ui-langfuse-walkthrough.md): run an example and inspect its trace.
- [V2 extension plan](docs/V2_EXTENSION_AND_EVALUATION_PLAN.md): MCP, evaluation and
  telemetry scope with acceptance criteria.
- [Evaluation methodology](docs/eval-methodology.md): data provenance, metrics and limitations.
- [Design decisions](docs/decisions.md): architecture tradeoffs and interview walkthrough.
- [Data and attribution](data/README.md): public sources, sampling and licences.
- [Historical V1 closeout](docs/archive/IMPLEMENTATION_PLAN.md): September 16 deliverables
  and validation, preserved as a record of that revision.

Code is MIT ([LICENSE](LICENSE)). CUAD and ContractNLI annotations are CC BY 4.0; corpus
files are acquired from their public releases and are not distributed in this repository.
Development used coding-agent assistance; the `hand-v1` suite was authored/reviewed by the
project author and generated supplements are labelled separately.
