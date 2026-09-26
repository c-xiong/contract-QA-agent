# Retrieval improvement plan

## Objective and reporting contract

Improve evidence retrieval at a fixed five-original-chunk output budget. Report
actual measured results, including regressions, instead of choosing a metric or
test subset after seeing results. No paid model calls are needed for this phase.

- Preserve `hand-v1` and all historical artifacts. It remains a development set.
- Reproduce the legacy baseline before changing retrieval.
- Add a separately versioned, document-aware evidence metric. Score all arms
  under the same metric; do not subtract old-metric and new-metric numbers.
- Freeze an external validation manifest before evaluating candidates on it.
  Use original expert-authored public questions and annotations if available;
  model-generated questions remain provisional and cannot support resume claims.
- Retain the original chunk/citation identities and final k=5. Record candidate
  coverage, per-category results, paired wins/losses/ties, and retrieval timing.
- Select the configuration on development results only. Run the frozen validation
  comparison after selection and report it even if it fails to confirm the gain.

## Bounded experiments

1. Baseline: BM25, dense, RRF, and the existing cross-encoder reranker.
2. Document coverage: on explicit multi-document comparisons, retrieve per
   document and preserve coverage while selecting five total original chunks.
3. Truncation: measure actual tokenizer lengths. Test token-bounded overlapping
   windows mapped back to original chunks, with deduplication before ranking.
4. Combine changes only if ablations support them. Inspect candidate coverage
   before spending work on reranking. Keep ineffective variants out of defaults.

The aspiration is a material gain (roughly ten percentage points), not a promise.
If the bounded experiments do not achieve it, inspect concrete remaining failures
and pursue only a justified next experiment. Never repeatedly tune on validation.

## Delivery

Ship the selected retriever through the existing factory with safe defaults and
meaningful regression tests. Run offline tests, lint, formatting, and strict type
checks. Check in reproducible experiment artifacts and a limitations report.
Write concise resume bullets whose percentages and evaluation descriptions match
those artifacts. Do not claim answer correctness from retrieval measurements.

## Completed outcome

Baseline reproduction, initial ablations, and a failure-driven bounded query-expansion
experiment are complete. The selected arm scores 72.7% versus 61.1% on development;
the frozen 119-question independent validation scores 71.8% versus 70.2%. The latter
gain is modest and uncertain, so the arm is optional rather than a default replacement.
The failed window-index experiment is retained only for reproducibility. All required
checks passed. See [the result report](retrieval-improvement-results.md) and
[the finished résumé bullets](resume-bullets.md).
