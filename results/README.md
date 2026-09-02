# Results

The artifacts behind every number in the top-level README — one file per claim.

| File | What it backs | Corpus | n | Model |
|---|---|---:|---:|---|
| `eval-hand-rerank.json` | The grader table and the layer-1 / layer-2 headline | 7,453 | 41 | claude-sonnet-5 |
| `eval-hand-bm25.json` | The BM25 column of the retrieval-to-groundedness table | 7,453 | 41 | claude-sonnet-5 |
| `experiment-a-retrieval.json` | Retrieval comparison, four arms | 6,270 | 36 scored | none (retrieval only) |
| `experiment-a-retrieval-corpus7453.json` | The same comparison before the chunking fixes, kept for the before/after | 7,453 | 36 scored | none (retrieval only) |
| `experiment-b-agentic.json` | Single-pass vs agentic loop | 7,453 | 41 | claude-sonnet-5 |
| `experiment-c-citation-gate.json` | Citation gate ablation, no fault injection | 7,453 | 41 | claude-sonnet-5 |
| `experiment-c-fault-injection.json` | The same ablation with deliberate miscitation, 1 in 3 answers | 7,453 | 41 | claude-sonnet-5 |

Every artifact here is on the `hand-v1` suite: 41 author-written questions with evidence
transcribed from CUAD and ContractNLI annotation. Runs on the earlier model-generated
question set are never promoted here, and no number in the top-level README comes from
one.

Each artifact carries its own `task_count`, `dataset_version`, `grader_versions`,
per-task scores, and per-category breakdown, so any figure in the top-level README can be
recomputed from the file it came from.

**[`RESULTS.md`](RESULTS.md) is the readable version** — the same numbers rendered as
tables by `scripts/summarize_results.py`. It is generated rather than written, so it
cannot drift from the data. Re-run the script after promoting a new run into this
directory.

## Promoting a run

`evals/runs/` and `experiments/runs/` are gitignored. A run is copied here deliberately
when it backs a claim, rather than the whole run history accumulating in git. Those
directories also hold superseded runs — earlier ones used a model-generated question set —
so a figure found there does not necessarily correspond to anything reported.

## Reproduce

```bash
uv run python experiments/retrieval_comparison.py --suite hand          # free
CRA_LIVE_MODEL=1 uv run python scripts/run_eval.py --suite hand \
    --arm rrf_hybrid_rerank --claim-support
CRA_LIVE_MODEL=1 uv run python experiments/agentic_comparison.py --suite hand
CRA_LIVE_MODEL=1 uv run python experiments/citation_gate_ablation.py --suite hand
```
