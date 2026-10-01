# Contract Review Agent — evidence-backed résumé wording

Draft wording for review; the CV source is unchanged. Start with the
[documentation index](README.md), [project walkthrough](project-guide.md), and
[UI and Langfuse walkthrough](ui-langfuse-walkthrough.md) to understand the implementation.

- Built a **LangGraph** contract-research agent with schema-validated tools for clause retrieval, definition lookup and date arithmetic, exposed through a **local MCP server** and **FastAPI/SSE** trace inspector.
- Engineered document-scoped **BM25/dense retrieval**, contract query expansion and **cross-encoder reranking**, raising development **evidence recall@5 from 45.8% to 72.7%**; achieved **71.8% on 119 CUAD queries across 23 development-disjoint contracts**.
- Enforced **document/snapshot scope**, evidence provenance, execution budgets and mandatory citation checks, with bounded repair and regression coverage for unsupported queries, conflicting evidence and prompt injection.
- Implemented metadata-only **OpenTelemetry export compatible with Langfuse** and resumable evaluation tooling with persisted spending controls, frozen human-label workflows and separate claim-support and answer-rubric judges.

## What each bullet means

1. **Agent and interfaces:** LangGraph manages the decision/authorization/tool/write/verify
   route. Pydantic schemas validate external actions. FastAPI/SSE displays actual persisted
   execution events. The MCP adapter exposes the same tools to external agents over local
   stdio and uses the shared executor; it does not run a model or control the client's prose.
2. **Retrieval result:** BM25 supplies lexical matching; local dense embeddings supply
   semantic matching; reciprocal rank fusion combines rankings; the cross-encoder scores
   query/passage pairs. Query expansion and document coverage form an optional arm selected
   through ablations. The percentage measures gold evidence found in five retrieved chunks,
   rather than whether the generated answer is correct.
3. **Runtime controls:** Python enforces document grants, content-hash snapshot scope,
   admitted-evidence limits, call budgets and deadlines. Final verification checks citation
   locations against admitted evidence. It can reject fabricated locations but cannot
   establish semantic support or answer completeness; bounded repair is followed by checks.
4. **Observability and evaluation:** Finished local journals become nested OTLP traces
   containing timings, usage and statuses, without prompts or contract content. Resumable
   evaluation artifacts and paid-run ledgers retain failures and prior spending. Human
   labels are frozen before judge comparison. Tooling is implemented and Langfuse Cloud
   viewing was manually verified for stub and live V2 smoke runs on 2026-10-01; the live
   V2 quality evaluation remains pending.

Use the [interview guide](RAG_AGENT_INTERVIEW_GUIDE.md) for follow-up questions and
implementation evidence behind all four bullets.

## Interview notes

- Development number: 36 non-abstention questions, BM25 baseline versus selected arm;
  the previous hybrid/rerank pipeline scored 61.1%. It is not held-out.
- Independent validation number: original, expert-annotated CUAD clause-extraction questions on
  documents absent from development suites. The original hybrid/rerank baseline
  scored 70.2%, so the additional independent-set gain is modest and uncertain.
- Evidence recall is a document-matched 60-character evidence-location proxy.
  Complete-span recall is stricter: 40.7% to 68.5% on development; 62.4% to 69.1%
  on independent validation. Neither metric is answer accuracy.
- The original 41-question suite has **35 answer + 1 partial + 5 abstain** tasks. The
  retrieval comparison uses the 36 non-abstention tasks; end-to-end evaluation keeps
  all 41. It is distinct from the 119-query validation set, the 24 provisional V2
  cases and ordinary software tests.
- All comparisons retain five original chunks; no top-k increase or corpus/gold
  filtering was used to obtain the improvement. Failed window-index experiments
  and category regressions remain in the report.
- The shared MCP executor enforces per-process session budgets; reconnecting starts
  fresh quotas. It cannot enforce an external model's spending or mandatory verification.
- OpenTelemetry export is implemented and tested against a local OTLP receiver.
  Langfuse Cloud viewing was manually verified for stub and live V2 UI smoke runs on
  2026-10-01. This demonstrates connectivity and execution visibility, not answer quality.
  The exporter sends metadata only, so it is not a prompt/answer inspection or
  evaluation-score service.
- Live V2 quality metrics remain pending. The answer-rubric judge is provisional until
  author review and calibration. The experiment ledger reserves input usage using a
  character-to-token estimate; do not describe it as an exact financial guarantee.

See [results, limitations and reproducible artifacts](retrieval-improvement-results.md).
