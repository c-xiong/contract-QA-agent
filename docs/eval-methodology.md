# Evaluation methodology

Updated 2026-09-16. Measurements live in `results/`; this defines their meaning.

## Corpus and task provenance

The manifest registers 40 CUAD agreements, 10 ContractNLI NDAs and 6 synthetic variants:
56 documents, currently 6,270 chunks. Do not describe synthetic variants as independently
collected real agreements. Acquisition and attribution are in [data/README.md](../data/README.md).
CUAD labels are attorney-supervised; ContractNLI supplies expert Entailment, Contradiction
and NotMentioned labels. Annotation span indices resolve through the document span table
before slicing text; round-trip tests guard this distinction.

`hand-v1` has 41 author-written/reviewed questions, including five expected abstentions.
Questions, labels, required points and evidence are frozen. Corrections require a new
version. This suite was used during development, so it is not a held-out benchmark.
Generated questions, expected answers or rubrics carry a `provisional-generated-*` version
and an explicit suite README. Expert evidence does not make generated questions human-
authored. Such supplementary runs cannot supply resume or headline capability claims.

## Metrics and boundaries

| Metric | Meaning |
|---|---|
| Document recall / reciprocal rank | Expected documents within supplied allowlists, not unrestricted corpus discovery |
| Evidence-span recall@k | Expected clause spans covered by retrieved chunks; A excludes tasks with no expected documents |
| Required points | Deterministic coverage of task-authored points; a lexical proxy |
| Citation validity | Locations exist, are allowed and were retrieved; no guarantee of factual support |
| Citation coverage | Fraction of extracted factual sentences carrying a citation attempt |
| claim_support@2 | Each factual sentence checked only against its own cited excerpts; supporting quotes verified in code |
| Answerability | False answers, correct refusals, over-abstention and execution failures, with separate denominators |

Answerability counts are operational: `completed`/`abstained`/`failed` and the optional
runtime verdict. They do not semantically classify baseline prose. A completed response
can still say evidence is missing; delivery counts must not be called successful answers.
The raw citation exposure metric also counts uncited extracted sentences, not just bad
locations, and is not an independently measured hallucination rate.

Historical `claim_support@1` checked the whole selected-evidence pool and is not comparable
with version 2. Report answerable-only support and evaluated-claim counts with aggregates.
Runtime gate and offline judge have independent prompts but share a model family. Sentence
extraction is deterministic but merges propositions in compound sentences.

The citation ablation independently audits raw and delivered output. The denominator is
parseable references plus malformed attempts. Legacy Experiment C counted malformed errors
only in the numerator; its 8.6% figure is withdrawn, not relabelled as a new measurement.

## Controlled experiments

| Experiment | Factors | Controls |
|---|---|---|
| A | BM25 / dense / RRF / RRF + rerank | Corpus, tasks, allowlists, k |
| B | Loop off/on × cross-reference off/on | Retriever, writer, evidence and repair budgets; answerability off |
| C | No verification / verify only / verify + one repair | Retrieval and writer; answerability off |
| D | Answerability off/on | Same retrieval, writer and citation policy; independent claim_support@2 |

Reports record n, one-trial disclosure, per-category results, paired win/loss/tie counts,
per-task outputs and costs. Timeouts or invalid gate JSON are failures, never correct
abstentions. New artifacts identify git SHA/dirty state, canonical chunk-content hash,
document/chunk counts, dataset/model/retrieval configuration and grader versions. Missing
historical metadata must not be invented. Do not compare different corpora or task versions
as if only one factor changed.

## Reproduction and cost

Use the root README and each script's `--help`. Smoke with `CRA_LIVE_MODEL=0` before live
runs. Normal pytest blocks network; explicit `model_download` tests load actual models.
Live runs need `CRA_LIVE_MODEL=1` and an API key; use `--limit` and an explicit output path.
Answerability adds at most one model call per task; offline support adds calls per factual
sentence. Record judge costs separately from application tokens and latency.

For the 2026-09-16 closeout, Sonnet 5 standard rates are $2 per million input tokens and
$10 per million output tokens; the scheduled September increase was cancelled. Estimates
exclude taxes and provider adjustments. Source: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).

Inspect at least three failures/disagreements. Small samples and one trial support counts
and descriptive tradeoffs, not significance or broad generalization claims.
