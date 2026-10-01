# Evaluation methodology

Scope: definitions for the historical V1 experiments in `results/` and the September
2026 retrieval follow-up in `reports/`. This document explains how scores are computed;
it does not report live V2 answer quality. The standalone V2 evaluation infrastructure
is implemented, but its live run and human-reviewed quality report remain pending.

Start with [the project guide](project-guide.md) for a plain-language explanation of
36 versus 41 questions and the evaluation layers. See [the documentation index](README.md)
for reading order and [the V2 execution protocol](evaluation_protocol.md) for the current
standalone workflow. Historical experiment definitions below were updated 2026-09-16.

## Corpus and task provenance

The manifest registers 40 CUAD agreements, 10 ContractNLI NDAs and 6 synthetic variants:
56 documents, currently 6,270 chunks. Do not describe synthetic variants as independently
collected real agreements. Acquisition and attribution are in [data/README.md](../data/README.md).
CUAD labels are attorney-supervised; ContractNLI supplies expert Entailment, Contradiction
and NotMentioned labels. Annotation span indices resolve through the document span table
before slicing text; round-trip tests guard this distinction.

`hand-v1` has **41 author-written/reviewed questions: 35 expected full answers,
one expected partial answer, and five expected abstentions**. The 36 non-abstention
questions have reference evidence, including the partial-answer comparison whose
redacted source limits what can be concluded. Retrieval Experiment A scores those
36; end-to-end experiments retain all 41 so refusal behavior is evaluated too.
These questions are separate from ordinary `pytest` software tests.
Questions, labels, required points and evidence are frozen. Corrections require a new
version. This suite was used during development, so it is not a held-out benchmark.
Generated questions, expected answers or rubrics carry a `provisional-generated-*` version
and an explicit suite README. Expert evidence does not make generated questions human-
authored. Such supplementary runs cannot supply resume or headline capability claims.

## Metrics and boundaries

Each question can contain several reference evidence spans. For retrieval, its score
is the fraction of those spans found in the returned chunks; the reported mean gives
each question equal weight. For example, finding one of one span on question A and
one of two spans on question B gives `(1 + 0.5) / 2 = 75%` macro recall. It does not
mean that 75% of questions were completely answered. `@5` means five chunks, not five
contracts. Finding the right contract alone also does not mean finding its right clause.

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

The low-level retrieval graders return a noted `1.0` when a task has no expected
documents or spans. That is a not-applicable placeholder, not evidence of successful
retrieval or refusal. Experiment A excludes the five expected-abstention cases before
averaging. Historical B–D aggregate tables use all 41 cases and can include these
placeholders; their retrieval means must not be compared directly with A's 36-case
means. New V2 diagnostics use `null` for evidence recall on expected-abstention cases.

For refusal behavior, keep the denominators separate: five expected-abstention cases
measure correct refusals or unsupported answering; 36 non-abstention cases measure
false refusals, including the expected partial answer. Infrastructure failures remain
failures. Citation/support metrics use their own citation or claim counts, not a fixed
41-question denominator. Human-reviewed task success uses all executed, reviewed
cases, including failures; unexecuted cases are listed explicitly.

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

Historical cost assumptions recorded for the **2026-09-16 closeout** were Sonnet 5
at $2 per million input tokens and $10 per million output tokens, with the scheduled
September increase described as cancelled at that time. These dated assumptions are
retained to interpret those artifacts; they are not a current price quote. Estimates
exclude taxes and provider adjustments. The recorded source was
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).
Verify current provider prices and fill the live-run preflight before a new paid run.

Inspect at least three failures/disagreements. Small samples and one trial support counts
and descriptive tradeoffs, not significance or broad generalization claims.

## September 2026 retrieval improvement

See [the complete ablation and validation report](retrieval-improvement-results.md).
Historical `evidence_span_recall` remains unchanged. New reports additionally use:

- `document_aware_evidence_recall@1`: the same normalized 60-character probe, but
  only chunks from the gold document can match. This is an evidence-location proxy,
  not proof of complete evidence or a correct answer.
- `evidence_fragment_recall@1`: full normalized annotated fragments must occur in
  correct-document chunks. Split only explicit `<omitted>` markers; no fuzzy matches.
- `evidence_complete_span_recall@1`: all fragments of an annotation must be found.
  A span split across chunk boundaries can fail this conservative diagnostic.

Scores are macro-averaged over answerable questions, not pooled over individual
annotations. All arms use the same original corpus, document scopes and final k=5.
The new retriever retains a 40-candidate cross-encoder budget; query expansion and
per-document retrieval add local search work but no hosted model calls.

`cuad-source-questions-v1` contains 119 original CUAD questions on 23 indexed source
documents absent from development suites. Questions and all answer spans were
transcribed verbatim, with source-offset roundtrips. The manifest was frozen before
validation; configuration selection used `hand-v1` only. This validates templated
single-document clause extraction, not conversational or cross-document QA. It has
`review_status=source-annotated`, not a claim of new author review. Document-group
bootstrap intervals are reported; they do not establish significant improvements.

Reranker score memoization in the experiment reuses only exact model/query/text
inputs. Recorded wall times include caching and are not serving-latency claims.
