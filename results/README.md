# Results

The artifacts behind every number in the top-level README. Pinned deliberately, one per
claim, rather than swept in by a glob — `evals/runs/` and `experiments/runs/` stay
gitignored so ad-hoc runs do not accumulate in the history.

| File | What it backs | n | Model |
|---|---|---:|---|
| `eval-hand-rerank.json` | The grader table and the layer-1/layer-2 headline | 41 | claude-sonnet-5 |
| `experiment-a-retrieval.json` | Retrieval comparison, four arms | 36 scored | none (retrieval only) |
| `experiment-b-agentic.json` | Single-pass vs agentic loop | 41 | claude-sonnet-5 |
| `experiment-c-citation-gate.json` | Citation gate ablation, no fault injection | 41 | claude-sonnet-5 |

Each carries its own `task_count`, `dataset_version`, `grader_versions`, per-task scores
and per-category breakdown, so any figure in the top-level README can be recomputed from
the file rather than taken on trust.

**[`RESULTS.md`](RESULTS.md) is the readable version** — every number above rendered as
tables, generated from these artifacts by `scripts/summarize_results.py`. It is generated
rather than written, so it cannot drift from the data. Re-run the script after promoting
any new run into this directory.

Ad-hoc runs land in `evals/runs/` and `experiments/runs/`, which stay gitignored: a run
is promoted here deliberately when it backs a claim, rather than the whole history being
swept into git. Those directories also hold superseded runs — earlier ones used a
model-generated question set — so a figure found there is not necessarily one this
project stands behind.

Reproduce:

```bash
uv run python experiments/retrieval_comparison.py --suite hand          # free
CRA_LIVE_MODEL=1 uv run python scripts/run_eval.py --suite hand \
    --arm rrf_hybrid_rerank --claim-support
CRA_LIVE_MODEL=1 uv run python experiments/agentic_comparison.py --suite hand
CRA_LIVE_MODEL=1 uv run python experiments/citation_gate_ablation.py --suite hand
```
