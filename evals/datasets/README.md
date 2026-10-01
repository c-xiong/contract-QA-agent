# Eval datasets

Suites live under `evals/datasets/`. These are question datasets for measuring
behavior, separate from the Python tests that check implementation rules.

| Suite | Questions | Evidence labels | Version | Status |
|---|---|---|---|---|
| [`hand`](hand/README.md) | author-written/reviewed | Transcribed expert annotations and known synthetic edits | `hand-v1` | **41 development regression tasks: 36 with expected evidence + 5 expected abstentions.** Historical A–D comparisons and the current standalone V2 workflow use this suite. |
| `full` | model-generated | expert annotation | `provisional-generated-v1` | 41 tasks. Superseded by `hand`; kept so earlier runs stay interpretable. |
| `smoke` | — | — | — | Empty. A scratch suite for local checks. |
| [`answerability`](answerability/README.md) | AI-authored generic wrapper | Transcribed ContractNLI labels and hypotheses | `provisional-generated-answerability-v1` | 20 supplementary diagnostic tasks; never headline or resume evidence. |
| [`v2`](v2/README.md) | AI-generated, not author-reviewed | Excerpts from indexed synthetic variants; generated capability expectations | `provisional-generated-v2-1` | 24 supplementary wiring/capability fixtures. Keep separate from `hand`; not live answer-quality or CV evidence. |
| [`cuad_holdout`](cuad_holdout/README.md) | Original CUAD questions | Original expert-annotated answer spans | `cuad-source-questions-v1` | 119 answerable retrieval-validation questions across 23 development-disjoint documents. Measures templated clause retrieval, not end-to-end answer accuracy. |

**36 and 41 do not identify two different hand datasets.** Retrieval recall scores
the 36 cases with expected evidence; the five expected-abstention cases have no gold
passage to find. Answer/abstention evaluation retains all 41. The separate CUAD set
supplies the 119-question retrieval-validation figures in the top-level README.
See the [project guide](../../docs/project-guide.md) for worked examples.

## Why `full` is not reported

`full` has expert-annotated evidence but model-generated questions. A dataset whose
questions were written by the same model family the system uses is partly measuring
itself, so any score from it is a pipeline check rather than a capability measurement.

The label travels with the data: its `dataset_version` starts `provisional-generated-`,
and every report that reads it prints a warning keyed on that prefix. `hand` replaced it
by rewriting the `question` field of each task and changing the version — the evidence,
the loader, the graders, and the experiments were unaffected.

## Authoring a task

Get the expert-annotated half for any document and category:

```bash
uv run python scripts/derive_task_evidence.py --doc doc-001 --list
uv run python scripts/derive_task_evidence.py --doc doc-001 --category "Governing Law"
```

That prints `expected_document_ids` and `expected_evidence` with pages already resolved.
You supply `question`, `category`, `expected_behavior`, `required_points`,
`forbidden_claims`, and `dataset_version`.

See [evaluation methodology](../../docs/eval-methodology.md) for provenance and reporting rules.

## Format

JSONL — one JSON object per line, no enclosing array, no trailing commas. Lines starting
with `#` are ignored, so notes can live in the file.

## What good questions do

- **Answerable from the cited evidence**, and from that evidence specifically.
- **Do not contain the answer.** A question mentioning "New York" hands the retriever
  the answer and measures nothing.
- **Do not echo the clause heading.** A question that is literally "governing law" is
  matched by BM25 on the heading alone; it tests keyword search, not retrieval. The eight
  categories were chosen to differ in retrieval character so that
  the suite does not become eight copies of one result.
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

The following composition applies to the 41 tasks in `hand` and `full`, rather than
all suites listed above. Ground-truth sources are shown below.

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
