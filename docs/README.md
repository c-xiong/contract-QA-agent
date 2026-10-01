# Documentation map

Reviewed against the working tree on 2026-10-01. Start here rather than reading old plans
in filename order.

## Learn the project

1. [Project guide](project-guide.md): what happens to a question, why there are 36 and
   41 cases, what evaluation measures, and how to explain the CV bullets.
2. [UI and Langfuse walkthrough](ui-langfuse-walkthrough.md): run one example, inspect
   evidence and execution, then connect its local journal to a remote trace.
3. [Interview practice](RAG_AGENT_INTERVIEW_GUIDE.md): deeper questions, tradeoffs,
   supported claims, and limitations.

## Current implementation and measurement status

| Area | Implemented | What remains unproven or pending |
|---|---|---|
| V2 agent | Typed tools, bounded execution, source evidence, mandatory structural citation gate | No completed live V2 answer-quality evaluation |
| Retrieval | BM25/dense fusion, reranking, optional legal query expansion and document coverage | Development results do not establish general answer accuracy |
| MCP | Local stdio adapter and shared tool executor | External clients can skip final citation verification |
| Observability | Durable local journals, SSE inspector, optional metadata-only OTLP export; manual stub and live V2 traces viewed in Langfuse Cloud on 2026-10-01 | A successful smoke run does not guarantee delivery of future traces or establish answer quality |
| V2 evaluation | Resumable runner, budget ledger, label freezing, separate LLM judges | Live outputs, human labels, judge calibration, and final quality report are pending |

V2 is the active development path. V1 remains the API/runtime default and historical
baseline; the current UI explicitly selects V2. Stub mode exercises the wiring and
cannot demonstrate answer quality.

## Find the authoritative detail

| Need | Read |
|---|---|
| Install, run, current results | [Repository README](../README.md) |
| Architecture and runtime limits | [V2 design](v2_design.md), [tool contracts](tool_contracts.md) |
| Reasons for design choices | [Decision log](decisions.md) |
| Dataset roles and provenance | [Dataset index](../evals/datasets/README.md), [source data](../data/README.md) |
| Metric meanings and denominators | [Evaluation methodology](eval-methodology.md) |
| Current V2 evaluation procedure | [Evaluation protocol](evaluation_protocol.md#standalone-v2-evaluation-hand-v1) |
| Remaining work and acceptance criteria | [V2 extension roadmap](V2_EXTENSION_AND_EVALUATION_PLAN.md) |
| September retrieval measurements | [Retrieval results](retrieval-improvement-results.md) |
| Historical V1 comparisons and original artifacts | [Results index](../results/README.md) |
| Historical offline V2 validation | [September V2 report](../reports/v2_results.md) |
| Suggested CV wording | [Resume bullets](resume-bullets.md) |

## Historical plans

Completed or superseded plans live in [archive/](archive/README.md). They explain the
development history; they are not current setup instructions or feature-status reports.
Historical reports and dataset READMEs stay next to their artifacts so provenance links
remain meaningful. No result artifacts or frozen datasets were removed.

## Keep the documentation coherent

- Put metric definitions in the methodology and evaluation commands/rules in the protocol.
- Keep the roadmap focused on remaining work; do not treat an old checklist as current status.
- Label examples as simulated, historical live, or newly measured. Name the dataset,
  denominator, runtime revision, and limitations when reporting a number.
- Update the guide, README, and interview status when implementation changes; preserve
  dated historical measurements rather than silently attaching them to new code.
