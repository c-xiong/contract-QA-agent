# Eval datasets

Two suites live under `evals/datasets/`:

| Suite | Source | Status |
|---|---|---|
| `full` | `scripts/build_eval_tasks.py` | **41 tasks, model-generated questions.** Version `provisional-generated-v1`. |
| `smoke` | you | Empty. Hand-written tasks go here. |

## The distinction that matters

`full` has expert-annotated **evidence** and model-generated **questions**. Every report
that reads it prints a warning, and its `dataset_version` starts `provisional-generated-`
so the label travels with the data. Figures derived from it are a pipeline check, not a
capability measurement: the questions were written by the same model family the system
uses, which makes any score partly a measurement of itself.

`smoke` is the suite for hand-written questions. Nothing generates into it.

## Turning generated figures into real ones

Rewrite the `question` field of each task and change `dataset_version`. **Nothing else
changes** — the evidence, the loader, the graders, and the experiments are all unaffected,
and the README warnings key on the version prefix.

Get the expert-annotated half for any document and category:

```bash
uv run python scripts/derive_task_evidence.py --doc doc-001 --list
uv run python scripts/derive_task_evidence.py --doc doc-001 --category "Governing Law"
```

That prints `expected_document_ids` and `expected_evidence` with pages already resolved.
You supply `question`, `category`, `expected_behavior`, `required_points`,
`forbidden_claims`, and `dataset_version`.

## Format

JSONL — one JSON object per line, no enclosing array, no trailing commas. Lines starting
with `#` are ignored, so notes can live in the file.

## What good questions do

- **Answerable from the cited evidence**, and from that evidence specifically.
- **Do not contain the answer.** A question mentioning "New York" hands the retriever
  the answer and measures nothing.
- **Do not echo the clause heading.** A question that is literally "governing law" is
  matched by BM25 on the heading alone; it tests keyword search, not retrieval. SPEC 6.2
  chose eight categories that differ in retrieval character precisely so the suite does
  not become eight copies of one result.
- **Are specific enough to have one right document.** "Can a party terminate for
  convenience?" is answerable from most of the corpus. Scope it, or set
  `allowed_document_ids`.

## Validation

The loader refuses rather than skips — silently dropping a task changes the denominator
of every reported metric. It rejects:

- a `question` still containing `TODO(author)`, `<REPLACE`, or `FIXME`
- an `abstain` task carrying expected evidence
- an `answer`/`partial` task with no `expected_document_ids`
- `expected_evidence` naming a document absent from `expected_document_ids`
- expected documents excluded by `allowed_document_ids` (unpassable by construction)
- duplicate `task_id`, or mixed `dataset_version` within one suite

## Composition

SPEC 15.3 targets 32-40 tasks; SPEC 6.8 maps every category to a ground-truth source.
The generated `full` suite currently holds:

| Category | n | Ground truth |
|---|---:|---|
| single_document_fact_lookup | 10 | CUAD clause spans |
| cross_document_comparison | 8 | CUAD, same category across 3 contracts |
| unanswerable | 5 | ContractNLI `NotMentioned` |
| cross_reference | 4 | CUAD Cap On Liability + Uncapped Liability |
| query_refinement | 4 | CUAD, question in mismatched terminology |
| claim_unsupported | 3 | ContractNLI `Contradiction` |
| conflicting_versions | 3 | Synthetic version pairs |
| prompt_injection | 2 | Synthetic injection |
| similar_document_distractors | 2 | Synthetic paraphrase |
