# V2 evaluation protocol

Keep `hand-v1` byte-for-byte frozen. Report its 41 cases separately from the 24
`provisional-generated-v2-1` cases. The latter's questions, expected dates, capabilities
and rubrics are generated pipeline fixtures. They cannot become reporting data by
removing the provisional label: replace/review content and increment the dataset version.

Before a live comparison, freeze model revision, corpus/index hashes, retrieval arm,
search/evidence/repair budgets and V2 ceilings. Both modes see identical cases and
allowed snapshots. The runner disables the optional V1 answerability gate. V2's stricter
citation gate and dynamic routing are part of the treatment, and should be disclosed.
V1 cases requiring missing capabilities stay in the denominator. Do not remove failures.

```bash
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_comparison --suite hand
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_comparison --suite v2
# Intentional paid experiment, after dataset/rubric review:
CRA_LIVE_MODEL=1 uv run python -m experiments.v2_comparison --suite hand --arm rrf_hybrid_rerank --output reports/v2-hand-live.json
```

Each report includes provenance, per-case predictions, category and answerability subsets,
paired diagnostic wins/losses/ties, and cost medians including failed runs. No repeated
trials, held-out claim, bootstrap interval, p95 or currency estimate. The per-case
`.predictions.jsonl` is flushed after each run. The adjacent `.review.jsonl` starts blank.

An independent reviewer fills task_success, citation support counts, useful-call counts,
reviewer identity and one primary failure category per failed case. Review both arms,
including every apparent V2 win. Multiple correct capability trajectories are acceptable;
capability_order expresses only prerequisite relationships, not one exact tool sequence.

```bash
uv run python -m scripts.score_v2_review reports/v2-hand-live.json reports/v2-hand-live.review.jsonl --output reports/v2-hand-reviewed.json
```

The scorer rejects missing/duplicate cases, inconsistent denominators, unreviewed outcomes,
and successful labels on infrastructure failures. Generated/stub reports remain ineligible
for capability claims even after labels are supplied. Original semantic labels are never
inferred from citation structure or lexical coverage.

| Metric | Treatment |
|---|---|
| Task success | Manual frozen rubric; failures/cancellation always fail. |
| Citation validity | Structural delivered-citation counts and gate-pass task counts. |
| Semantic citation support | Independent supported / inspected citation counts. |
| Evidence recall | Existing span matching at k=5 and trajectory-wide; answerable cases only. |
| Tool selection | Required-capability coverage/order diagnostic; scope/arguments and usefulness manually inspected. |
| Unnecessary calls | Manual usefulness counts and explicit retry/duplicate diagnostics. |
| Abstention | Runtime answer-versus-abstain decision plus separate false abstention; prose refusals still need manual review. |
| Cost | Median latency, model/tool calls, input/output tokens; failures included; unknown usage remains null. |

V1's tool-call count is null: its internal operations were not instrumented as V2 registry
invocations. Do not treat null as zero or claim a cross-mode tool-cost reduction from it.
Stub usage is estimated and latency describes local fixture execution. Initial retrieval
models/indexes are shared; V1 runs before V2 within each pair. Cache state/order is disclosed,
not hidden as a performance result. Production generalization cannot be inferred here.
