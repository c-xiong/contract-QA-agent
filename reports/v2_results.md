# V2 validation report

Date: 2026-09-20. This is an implementation/offline validation report, **not a live-model
capability evaluation**. V2 remains opt-in. No paid V2 calls were made for these artifacts.

## Implemented and checked

- Five typed tools using the existing source store, retrieval arms and citation verifier.
- Conditional tool selection with a separate code-owned authorization boundary.
- Document/snapshot scope, argument validation, strict evidence limits, search/model/tool
  counters, bounded retries/repair, duplicate detection, deadline and final-verifier reserve.
- Persistent protected JSONL snapshots, manifests, summaries, safe SSE progress,
  cancellation/reconnect handling and dead-process failure accounting.
- Separate regression/new-case paired runner, deterministic diagnostics, review forms,
  manual-label aggregation and explicit failure taxonomy.
- Three runnable scripted synthetic examples and extended CI checks.

Validation on the final implementation:

| Check | Observed result |
|---|---|
| Original baseline before edits | 408 tests passed; 3 opt-in tests deselected |
| Final offline suite | **451 passed; 3 deselected** |
| Ruff lint / formatting | Passed |
| Strict mypy | Passed; 88 source files checked |
| Scripted trace demos | Success, repair recovery and abstention all reached expected terminal states |
| V1/V2 hand suite | 41 paired cases, 82 runs retained, one trial, BM25 + deterministic model stub |
| V1/V2 provisional suite | 24 paired cases, 48 runs retained, one trial, BM25 + deterministic model stub |
| Actual hybrid/rerank integration smoke | 2 paired hand cases, 4 runs; real retrieval models + model stub |

The default test exclusion covers the separate model-download/corpus/live-model markers;
real retrieval was exercised by the full baseline reproduction and rerank integration run.
The hosted CI workflow was extended but was not remotely executed in this session.

## V1 retrieval reproduction

The unchanged hand benchmark reproduced the earlier retrieval results: 36 answerable
questions, five expected-abstention cases excluded, k=5, one deterministic trial, identical
corpus/index, scoped retrieval. [Artifact](v1-retrieval-reproduced.json).

| Retrieval arm | Evidence-span recall@5 |
|---|---:|
| BM25 | 0.458 |
| MiniLM/FAISS | 0.394 |
| RRF hybrid | 0.495 |
| RRF + cross-encoder | 0.611 |

These are **V1 retrieval measurements**, not V2 task-success improvements. Per-case,
per-category and paired records are preserved in the artifact. Sample size is small;
these are development results, with no held-out or significance claim.

## Offline paired runs

[Original suite](v2-hand-offline.json) and [provisional suite](v2-v2-offline.json) have
separate dataset hashes, denominators, predictions and blank review forms. Both modes
returned `completed` in every stub run: 41/41 in the original set and 24/24 in the new
set. This status means the structural gate passed, **not that the task was solved**.

On expected-abstention cases, the stub failed to abstain in both modes: 5/5 original
cases and 4/4 provisional cases. Consequently, runtime answer/abstain decisions match
36/41 and 20/24 respectively, with paired differences 0 wins / 0 losses / 41 ties and
0 / 0 / 24. These numbers describe fixture behavior only; they are not answer-quality
scores. Semantic support and task success remain null pending independent review.

The scripted date demo exercises deterministic calculation, but the general offline
planner does not perform live reasoning. Its failure to call date arithmetic on the
new date cases is visible in capability diagnostics; it is not hidden by the citations.
See [three inspected failures](failure_analysis.md).

Actual reranking is also wired through the V2 registry: the
[2-case smoke artifact](v2-rerank-smoke.json) retains both modes' outputs and costs.
This is an integration check, not a comparative performance study.

## Artifacts and reproduction

- `v2-*.json`: provenance, diagnostics, per-category summaries and per-case results.
- `v2-*.predictions.jsonl`: incrementally saved predictions, including failures.
- `v2-*.review.jsonl`: blank human adjudication forms, intentionally unfilled.
- `../examples/traces/`: invented-source demos safe to inspect and check into Git.
- `../runs/`: private, gitignored full execution logs and source excerpts.

Commands are in [the protocol](../docs/evaluation_protocol.md). Source, worktree,
corpus, dataset and available index fingerprints identify each individual artifact's
execution; a later documentation edit does not retroactively change an old manifest.

## Remaining before capability claims

The plan's full evaluation definition of done is **not yet met**: generated new cases
need author replacement/review with a new version, frozen live paired runs need execution,
and independent task-success/citation-support/tool-usefulness labels need completion.
No new CV percentage or claim of improved semantic correctness is supported yet.
Business-day calendars, legal version precedence and crash resumption remain unsupported.
