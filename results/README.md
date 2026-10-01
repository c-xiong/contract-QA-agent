# Results

> V1 controlled-comparison artifact index. "Current" in the original A–D records means
> the September 16 closeout revision, not a live V2 evaluation or the latest retrieval
> experiment. Later retrieval work is linked below; its artifacts and the offline V2
> validation live in `reports/`. Preserve these records and their original provenance.
> Start with the [project guide](../docs/project-guide.md) for the evaluation map.

V1 closeout completed 2026-09-16. Those runs use 56 documents / 6,270 chunks and the frozen
`hand-v1` development suite. B–D use Sonnet 5, one trial and no fault injection.

| Current artifact | Measurement | Tasks / arms |
|---|---|---|
| [experiment-a-current.json](experiment-a-current.json) | Retrieval; no generation calls | 36 scored / 4 |
| [experiment-b-factorial.json](experiment-b-factorial.json) | Loop × cross-reference | 41 / 4 |
| [experiment-c-isolated.json](experiment-c-isolated.json) | Ungated / verify-only / verify+repair | 41 / 3 |
| [experiment-d-answerability.json](experiment-d-answerability.json) | Answerability off/on; independent claim_support@2 | 41 / 2 |

[RESULTS.md](RESULTS.md) contains generated tables; [ANALYSIS.md](ANALYSIS.md) records the
adoption decision, costs and three inspected failures. Definitions and limitations are
in [evaluation methodology](../docs/eval-methodology.md).

Current artifacts record run ID/time, code SHA and dirty state, source hash, canonical
chunk-content hash, counts and grader/configuration versions. B–D include per-task
outputs, scores and application costs; D records judge usage separately. A and D were
captured before dataset-file hashing was added; they identify `hand-v1` without a dataset
hash. Source hashes reflect the files at each run, not later documentation/type-only
cleanup. Missing metadata has not been backfilled.

## Historical artifacts

| File | Status |
|---|---|
| `experiment-a-retrieval.json` | Earlier 6,270-chunk retrieval result; reproduced by current A |
| `experiment-a-retrieval-corpus7453.json` | Earlier 7,453-chunk retrieval result |
| `eval-hand-bm25.json`, `eval-hand-rerank.json` | Earlier 7,453-chunk live evaluations; claim_support@1 |
| `experiment-b-agentic.json` | Earlier two-arm comparison; loop and xref were confounded |
| `experiment-c-citation-gate.json` | Historical citation rate withdrawn: incorrect denominator |
| `experiment-c-fault-injection.json` | Historical injected failures; not a natural error rate |

Historical JSON is preserved unchanged. These artifacts lack current provenance and
cannot supply a current-model or current-corpus quality claim. The old 8.6% citation
figure has been removed from the CV and withdrawn from the README.

## Reproduce

Follow the [root README](../README.md#reproduce-the-comparisons) for corpus preparation,
offline smoke runs and the bounded live commands. Live runs incur API charges.

```bash
CRA_LIVE_MODEL=0 uv run python scripts/summarize_results.py
```

`evals/runs/` and `experiments/runs/` remain ignored scratch directories. The generated
20-task ContractNLI diagnostic is a supplementary stub check, not a résumé metric.
Promote only artifacts that back a reviewed claim; preserve their original provenance.

## Retrieval follow-up, 2026-09-26

The new artifacts live in `reports/` and preserve all historical comparisons above.
[The follow-up report](../docs/retrieval-improvement-results.md) links the reproduced
baseline, every development ablation, frozen selection, and 119-question independent
CUAD validation. It reports both document-aware evidence probes and complete-span
coverage, including regressions and uncertainty. [Résumé wording](../docs/resume-bullets.md)
distinguishes development gains from the smaller independent-validation gain.
