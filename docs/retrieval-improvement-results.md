# Retrieval improvement: results and limits

The selected optional arm is `legal_query_coverage_rerank`: fuse the original query
with bounded contract-vocabulary expansion, retrieve per document for explicit
comparisons, then rerank 40 candidates and choose five original chunks with document
coverage. No document IDs, party names, gold answers, or eval IDs appear in the rules.
No hosted-model calls or new dependencies were added.

## Protocol

- Same 56 documents, 6,270 original chunks, scopes, k=5, and one deterministic trial.
- `hand-v1` stays frozen: 36 answerable development questions; five abstention cases
  are excluded from retrieval aggregates. It is not a held-out benchmark.
- The legacy baseline was reproduced before optimization. New independent metrics
  enforce document identity and also measure complete annotated text.
- Configuration was selected on development results, then frozen in
  [the selection record](../reports/recall-selection.json) before validation.
- Validation: 119 original expert-annotated CUAD questions across 23 contracts
  absent from existing development suites. Questions and answers were transcribed,
  not generated; every source offset roundtrips. See the
  [frozen manifest](../evals/datasets/cuad_holdout/manifest.json).
- Query expansion, coverage and token windows were ablated separately. No further
  model tuning was performed after inspecting the validation results.

## Development ablations

n=36 answerable questions, one trial, five original chunks per result.
Evidence recall below means the document-matched 60-character evidence probe.
Complete-span recall requires every annotated fragment in full.

| Arm | Evidence recall@5 | Complete-span recall@5 |
|---|---:|---:|
| BM25 | 45.8% | 40.7% |
| Original hybrid + rerank | 61.1% | 56.9% |
| Document coverage only | 62.0% | 57.9% |
| Token-window dense index | 58.3% | 54.2% |
| Token windows + coverage | 59.3% | 55.1% |
| Contract query expansion | 71.8% | 67.6% |
| Query expansion + coverage (selected) | **72.7%** | **68.5%** |

Selected versus BM25: **15 wins / 2 losses / 19 ties**, +26.9 percentage points.
Selected versus original hybrid + rerank: **9 / 2 / 25**, +11.6 points.
The latter's descriptive bootstrap interval crosses zero; do not claim significance.

On eight development cross-document tasks, evidence recall rises from **25.0% to
54.2%**; requested-document coverage rises from 75.0% to 100.0%. The evidence metric,
not document coverage alone, is the relevant improvement. Category sizes are tiny.

The token audit found 1,358/6,270 chunk bodies exceeding the embedder's 254-wordpiece
body budget before headings. This was a real input-length limitation, but repairing
it did **not** improve this benchmark. The separate window index is retained only
for reproducible experiments and is not required by the selected arm.

## Independent validation

n=119 original CUAD clause-extraction questions, 23 development-disjoint documents,
one trial, k=5. These are templated extraction instructions, not new conversational
or multi-contract questions. The underlying corpus was selected for category coverage.

| Arm | Evidence recall@5 | Complete-span recall@5 |
|---|---:|---:|
| BM25 | 65.2% | 62.4% |
| Original hybrid + rerank | 70.2% | 67.4% |
| Selected | **71.8%** | **69.1%** |

Selected versus BM25: **19 wins / 15 losses / 85 ties**, +6.6 points;
document-group bootstrap 95% interval: **−1.3 to +15.2 points**.
Selected versus original hybrid + rerank: **5 / 6 / 108**, +1.6 points;
interval: **−1.7 to +4.9 points**. These intervals include zero. The development gain
does not establish a similarly sized generalization gain. Keep the arm selectable;
existing runtime defaults are preserved.

| Validation category | n | BM25 | Original hybrid + rerank | Selected |
|---|---:|---:|---:|---:|
| Cap On Liability | 23 | 62.4% | 64.3% | 65.8% |
| Change Of Control | 10 | 71.7% | 38.3% | 38.3% |
| Governing Law | 23 | 76.1% | 93.5% | 95.7% |
| Insurance | 7 | 92.9% | 85.7% | 73.8% |
| Non-Compete | 16 | 20.6% | 13.3% | 18.8% |
| Notice Period To Terminate Renewal | 13 | 76.9% | 100.0% | 100.0% |
| Termination For Convenience | 11 | 50.0% | 81.8% | 81.8% |
| Uncapped Liability | 16 | 83.3% | 83.3% | 89.6% |

## Inspected failures

1. `cuad-native-doc-020-01` (liability): original recall 1.0, selected 0.0.
   Gold evidence is in the candidate pool. Expanded liability vocabulary promotes
   indemnification and survival text over the annotated disclaimer: a ranking
   regression, not a candidate-generation failure.
2. `cuad-native-doc-022-07` (insurance): 1.0 to 0.5. Both annotated obligations are
   candidates, but the final five recover only one. Related indemnification and
   compliance passages compete with the second insurance requirement.
3. `cuad-native-doc-035-01` (liability): 1.0 to 0.5. The monetary-cap evidence is
   retained, while the second damages-exclusion annotation is lost after reranking.
   Candidate coverage is 1.0, so enlarging the first-stage pool would not repair it.

These failures were inspected for reporting after configuration freeze; no rules
were added to fit them. Remaining risks include lexical-expansion drift, incomplete
multi-evidence ranking, limited categories, and unknown public training contamination.
Retrieval scores do not establish answer correctness, semantic citation support,
prompt-injection robustness, or production reliability.

## Artifacts and reproduction

- [Legacy baseline](../reports/recall-baseline-legacy.json)
- [Initial ablations and candidate traces](../reports/recall-development.json)
- [Query-expansion ablations and paired comparisons](../reports/recall-query-development.json)
- [Frozen selection](../reports/recall-selection.json)
- [Independent validation and per-task traces](../reports/recall-validation.json)
- [Actual tokenizer audit](../reports/retrieval-input-audit.json)

From the repository root, using the existing environment:

```bash
export CRA_LIVE_MODEL=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
.venv/bin/python scripts/build_retrieval_validation.py
.venv/bin/python experiments/recall_optimization.py --output reports/recall-development-rerun.json
.venv/bin/python experiments/recall_optimization.py --suite hand --arms bm25 rrf_hybrid_rerank legal_query_rerank legal_query_coverage_rerank --output reports/recall-query-rerun.json
.venv/bin/python experiments/recall_optimization.py --suite cuad_holdout --arms bm25 rrf_hybrid_rerank legal_query_coverage_rerank --output reports/recall-validation-rerun.json
```

The initial window ablations require `.venv/bin/python scripts/build_index.py --windows`.
The selected arm uses the existing ordinary dense index. Exact query/text reranker
scores are memoized across arms; wall times in these artifacts are not latency
benchmarks. Single-thread environment settings avoid a native-library crash observed
in this local environment. All historical reports and `hand-v1` were left intact.

## Verification

- Offline suite: **475 passed**, three opt-in tests deselected.
- Actual cached retrieval-model smoke tests: **2 passed**.
- Ruff lint and formatting: passed.
- Strict mypy: passed for 64 source files.
- `git diff --check`: passed.

Changes remain local; no commit or push was made.
