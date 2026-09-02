# Implementation Plan v2 — Defensible First Version

**Revised 2026-09-01. Version 1 closed 2026-09-02 after S1 and S2.**

An earlier version of this plan aimed at a polished, publicly deployed portfolio system.
That scope is withdrawn. This version aims at something smaller and more honest: a public
repository whose every claim is checked in and reproducible, and whose every design choice
the author can defend in an interview.

Nothing here is deleted because it was wrong. It is deferred because it does not change
what the author can say about the project, and the remaining work does.

| Step | Status |
|---|---|
| S1 — land the citation-scoped evaluation work | **done** 2026-09-02 |
| S2 — every public number traceable to a checked-in artifact | **done** 2026-09-02, minus the provenance stamp |
| S3 — offline default test suite | next |
| S4 — answerability gate | next |
| S5 — factor-isolated ablations | next |

**S1 and S2 are version 1, and version 1 is finished.** The repository now states only
what its checked-in artifacts support. S3-S5 are the next version's work and are specified
below in full; nothing in the repository claims they are done.

## 1. Objective

Bring the repository to a state where:

1. the headline evaluation metric measures what its name says;
2. every number printed in the README has a checked-in artifact behind it;
3. the default test suite runs offline, deterministically, and free;
4. the observed false-answer failure is addressed by a real architectural change, and the
   change is measured against the baseline it replaces;
5. the two architectural ablations attribute their effect to the component they name.

That is the whole target. Items 1-3 make the project credible. Items 4-5 make it
interesting.

## 2. Scope

### In scope

| Step | Outcome | Paid model run |
|---|---|---:|
| S1 | Land the citation-scoped evaluation work already in the working tree | No |
| S2 | Every public number traceable to a checked-in artifact; no broken links | No |
| S3 | Default test suite is offline and deterministic | No |
| S4 | Answerability gate: the false-answer failure becomes an addressable behavior | Small, opt-in |
| S5 | Experiment B isolates loop vs cross-reference; Experiment C isolates verify vs repair | Small, opt-in |

S1-S3 are the credibility floor and cost nothing. S4 is the one capability change worth
making. S5 is what makes the ablations mean what their titles claim.

### Explicitly out of scope

Withdrawn from v1, not deferred pending a trigger — simply not part of this project:

- **any public deployment**: no hosted demo, no deployment workflow, no readiness probe
  for a platform, no per-IP rate limiting, no global spend budget, no abuse controls, no
  post-deploy smoke tests, no published demo URL;
- **anything whose purpose is presentation to a public audience**: recruiter-facing pages,
  curated example galleries, marketing copy in the README.

The existing `Dockerfile` and `compose.yaml` stay as a local reproducibility path. They
are not a deployment story and the README must not present them as one. The trace
inspector at `app/api/static/index.html` stays as a **local** inspection tool under the
CLAUDE.md carve-out — it is how the author reads a run by eye and how a live walkthrough
gets demonstrated on a laptop. It is never deployed.

The README remains public and must remain accurate, because the repository is public. But
its job is to explain the design to a reader who is already looking, not to attract one.

### Deferred until a measurement asks for it

- persistent JSONL traces and a trace query script (v1 PR6). The in-memory trace and the
  SSE inspector already answer "what did this run do". Persistence answers "what did runs
  do over time", which matters only once there is a corpus of runs worth mining;
- everything in v1's deferred list: multi-agent orchestration, GraphRAG, deeper
  cross-reference traversal, legal-domain embeddings, provider abstraction, a vector
  database, bootstrap confidence intervals, a full dev/test dataset rebuild.

## 3. Where version 1 left the repository

Settled by S1 and S2, 2026-09-02:

- claim support is scoped to each claim's own citations (`claim_support@2`), citation
  coverage is reported separately, and the ungated arm of Experiment C is audited
  independently. Lint, format, strict mypy, and the default test suite pass;
- every table in the README names the artifact it came from, and every one of those
  artifacts is checked into `results/` on the `hand-v1` suite. Experiment A was re-run on
  the current 6,270-chunk corpus; the pre-fix run is kept beside it for the comparison;
- the README's Experiment A and B tables had been reporting runs on the *model-generated*
  question set under a `hand-v1` heading. Both now report the author-written suite, and
  the prose claims that depended on the mismatched comparison are corrected;
- no README link targets an ignored path.

Known and unaddressed, in the order the next version should take them:

- `tests/unit/test_dense_and_rerank.py` loads real Hugging Face models, so the default
  suite is not hermetic and CI does not disable network access. The missing-corpus test
  still accepts either outcome (S3);
- artifacts carry dataset, model, retriever, and grader fields but no code version or
  corpus hash, so a future corpus change can silently invalidate a table again. This is
  the one part of S2 not done, and it is what made the drift above possible (S3);
- `evals/datasets/hand/tasks.jsonl` holds 41 tasks, 5 of which expect abstention, and on
  the checked-in live run all 5 were answered. `app/agent/policies.py` decides only
  whether to keep searching; there is no answerability decision anywhere in the graph (S4);
- Experiment B varies the search budget and cross-reference resolution together, and
  Experiment C has two arms, so verification and repair cannot be told apart (S5);
- the live-model tables are all on the pre-fix 7,453-chunk corpus. Refreshing them costs
  an API run each and is deliberately not done.

## 4. Rules that still apply

1. **No number without an artifact.** A figure in the README, a docstring, or a comment
   comes from a run whose report is checked in. Otherwise it is `[TBD]`.
2. **Free by default.** Tests, lint, types, ingestion, indexing, and every experiment's
   stub path spend nothing. A live run happens only behind an explicit environment gate
   and only after the same code path has run against the stub.
3. **Constraints live in code.** Budgets, gates, and verification are deterministic code
   with tests, never a sentence in a prompt.
4. **Runtime and evaluation stay independent.** A runtime policy that decides whether to
   answer must not share prompts or verdicts with the offline grader that scores the
   answer.
5. **Every decision-bearing choice gets a `# DECISION:` comment and a `docs/decisions.md`
   entry.** This is the point of the project now. A change the author cannot defend is
   worse than no change.

## 5. S1 — Land the citation-scoped evaluation work

### Goal

The work is done; it is not landed. Close it out so the repository's default checks pass.

### Work

- fix the four `mypy` errors in `evals/runner.py` (`grade.data` is `dict[str, object]`, so
  the aggregate counters need a typed accessor rather than a bare `int(...)`);
- fix the two `ruff` findings and run `ruff format`;
- confirm the README's wording matches the metric that actually ran: the historical 0.807
  is `claim_support@1`, evidence-pool support, and there is no live `claim_support@2`
  number yet;
- commit in small pieces with real messages.

### Done when

- `ruff check`, `ruff format --check`, `mypy app evals`, and `pytest` all pass;
- the work is committed;
- `docs/decisions.md` records why claim support is now scoped to a claim's own citations
  and why the old metric is retained under its old name rather than overwritten.

**Effort: under an hour.**

## 6. S2 — Honest artifacts and working links

### Goal

A reader who does not trust a number can find the file it came from. A reader who clicks
a link gets a page.

### Work

**Provenance, minimal version.** One helper, used by the eval runner and every experiment,
that stamps each report with:

```text
artifact_schema_version
run_id
started_at
git_sha
git_worktree_dirty
dataset_name, dataset_version
document_count, chunk_count, chunks_sha256
retrieval_arm and top_k
model_id
grader_versions
```

Nothing more. The v1 list of ten hashes is a provenance system for a team that reruns
experiments on a schedule; this project reruns them by hand a few times a year. The one
non-obvious field is `chunks_sha256`, which must change when chunk boundaries change —
that is exactly the drift that made the current README tables unbackable.

Existing artifacts get `artifact_schema_version: "legacy"` and keep the fields they have.
Do not backfill a git SHA or a corpus hash that was not recorded at run time.

**Reconcile the published numbers.** For each README table, either promote the artifact
that actually produced it, or restore the table to the checked-in artifact and label its
corpus. Experiment A is free to rerun on the current corpus, so rerun it and promote it.
For B and C, prefer restoring the checked-in `hand-v1` figures over paying to refresh
them; a table labelled "7,453-chunk corpus, 2026-08-26" is honest, and a footnote saying
the corpus has since changed is more interesting than a fresh number.

**Fix the documentation.** `docs/SPEC.md`, `docs/decisions.md`, `docs/review-questions.md`
and `docs/eval-methodology.md` are gitignored on purpose. Remove the README links that
point into them, or publish a sanitized equivalent outside `docs/`. Decide which, once,
and record it. `results/README.md` must stop claiming more than the files support.

**One consistency check, optional in CI.** A short script that verifies each artifact
parses under its declared schema, that `results/RESULTS.md` regenerates unchanged from the
files beside it, and that no README link targets an ignored path. Wire it into CI only if
it stays under about a hundred lines.

### Done when

- every number in the README names a checked-in file;
- new artifacts carry the provenance block; old ones are labelled legacy;
- a fresh clone has no dangling README link.

**Effort: half a day.**

### What actually happened, 2026-09-02

Done: Experiment A re-run on the current corpus and promoted, with the superseded pre-fix
run kept beside it as `experiment-a-retrieval-corpus7453.json`; the BM25 eval arm and the
fault-injection run promoted so the two tables that quoted them have files; Experiment B
restored to the checked-in `hand-v1` artifact; `results/README.md` and the generated
`RESULTS.md` brought back in line; README links into the ignored `docs/` tree removed.

The defect found on the way was worse than the one being fixed. The Experiment A and B
tables were not merely from an unpromoted run -- they were from the **model-generated**
question set, presented under a `hand-v1` heading, which is what CLAUDE.md rule 1 exists to
prevent. Two prose conclusions rested on comparing those numbers against `hand-v1` numbers:
"every arm is lower now" after the chunking fix (like-for-like, every arm is *higher*), and
an abstention regression that was a suite difference rather than a corpus difference. Both
are corrected.

Not done: the provenance stamp. It is the part that would have made the drift detectable
rather than something to be found by hand, so it moves to the top of the next version's
work rather than being dropped.

## 7. S3 — Offline default test suite

### Goal

`uv run pytest` passes on a machine with no corpus, no model cache, no network, and no API
key. That is a claim worth making and it is currently false.

### Work

- mark the tests that load real Hugging Face models (`tests/unit/test_dense_and_rerank.py`)
  with a `model_download` marker, deselect them by default, and run them in a separate CI
  job that is allowed to hit the network. Do not delete them — dense retrieval and
  reranking are load-bearing and need a real test somewhere;
- make the missing-corpus test prove the missing-corpus behavior and nothing else: build
  the app against an empty corpus directory explicitly rather than accepting
  `status in ("no_corpus", "ok")`, which passes either way;
- ensure application startup imports no model as a side effect;
- keep the offline job free of network access so a regression fails loudly.

The v1 plan proposed a dependency-injection refactor of the API. Skip it. The app already
builds its agent in one place; passing a settings object into that one place is enough,
and a full DI layer is infrastructure the project does not otherwise need.

### Done when

- the default suite passes with network and Hugging Face access disabled;
- the model-loading tests still run somewhere and are labelled;
- the missing-corpus test can only pass for the right reason.

**Effort: half a day.**

## 8. S4 — Answerability gate

This is the one capability change in the plan, and the part of the project most worth
talking about. Everything else is hygiene.

### The failure

On the checked-in live run, all five unanswerable tasks were answered. The system retrieves
topically related text, the writer finds it plausible, and nothing in the graph ever asks
whether the evidence actually answers the question. Deterministic citation verification
does not catch this: a fabricated answer built out of real citations passes every existing
check.

### Work

**Expand the abstention set first, and freeze it before tuning.** Five cases cannot
distinguish a gate that works from a gate tuned to five cases. Add roughly ten new
author-reviewed unanswerable tasks drawn from different documents, as a new dataset
version — `hand-v1` stays frozen. Prefer hard negatives: nearby-but-different clauses,
negations, missing obligations, partial answers. Do not look at the new cases while tuning.

**Add a pre-write answerability node**, after evidence selection and before the writer:

```json
{
  "verdict": "answerable | partial | unanswerable",
  "supported_aspects": [],
  "missing_aspects": [],
  "evidence_ids": [],
  "quotes": [],
  "reason": ""
}
```

The enforcement is in code, not in the prompt:

- an `answerable` or `partial` verdict must name evidence IDs and quote them verbatim;
- every quote is checked against the named evidence in Python. A quote that is not found
  cannot produce an `answerable` verdict;
- `partial` routes to the writer with instructions to answer only the supported aspects and
  to name what remains unknown;
- `unanswerable` produces an evidence-aware abstention that says what was found and why it
  does not answer the question;
- the node runs inside the existing search, evidence, and token budgets.

Do not threshold raw BM25, RRF, dense, or cross-encoder scores as if they shared a scale.
That is the tempting shortcut and it is wrong; write the `# DECISION:` comment explaining
why, because it is a good interview answer.

**Keep the runtime policy and the offline grader independent.** Different prompts,
independently versioned. The grader scores the final answer and never reads the runtime
verdict. Both use the same model family — record that as a stated limitation.

**Measure it, two arms, not three.** Baseline (gate off) versus gate on, identical corpus,
retriever, writer, task order, and budgets. Report raw counts and denominators:

- false answers on unanswerable tasks, and correct abstentions;
- over-abstention on answerable tasks;
- the answered/partial/abstained confusion matrix;
- required-point coverage and offline `claim_support@2`;
- model calls, tokens, and latency.

Report paired per-task wins, losses, and ties. No confidence intervals on a 50-task suite
with one trial — say so rather than implying more precision than one run supports.

The v1 plan's third arm, a separate runtime claim-support policy over the written draft,
is deferred. Measure the answerability gate alone first. If false answers survive it, that
result is the argument for the second gate; if they do not, the second gate was never
needed. That sequencing is itself the answer to "why didn't you add both".

**Pre-register the decision rule in `docs/decisions.md` before the locked run:** materially
fewer false answers, over-abstention at or below 10%, required-point coverage down by no
more than 0.05, offline claim support not reduced, and cost and latency reported either way.
Write the thresholds down before looking at the outputs.

### Done when

- the graph can answer, partially answer, or abstain with evidence-aware reasoning;
- quote verification is deterministic code with unit tests;
- both arms run against the stub for free; the live run is a single gated step;
- the artifact is promoted and the README updated only afterwards.

**Effort: 2-3 days, plus one small live run.**

## 9. S5 — Ablations that isolate what they name

### Experiment B: 2 x 2

Vary the search/refinement loop and cross-reference resolution independently:

| Arm | Loop | Cross-reference |
|---|---:|---:|
| single, no xref | off | off |
| single + xref | off | on |
| loop, no xref | on | off |
| loop + xref | on | on |

Hold everything else constant: corpus, index, dataset, task order, retrieval arm, top-k,
writer, prompts, model, evidence budget, repair policy. Report the main effect of each
factor, the interaction cases, and the cost of each. The current two-arm design cannot
support the sentence "iterative retrieval helps", and that sentence is currently implied.

### Experiment C: three arms

| Arm | Verify | Repair | Final |
|---|---:|---:|---|
| ungated | off | off | return the writer's output |
| verify only | on | off | reject or abstain on failure |
| verify + repair | on | once | finalize after re-verification |

The independent auditor added in S1 (`evals/citation_metrics.py`) inspects the raw writer
output in every arm, so the ungated arm reports a real invalid-citation rate instead of the
1.0 that the ungated verifier returns by construction. No `repair on / verify off` arm:
repair has no trigger without a verification result.

### Cost control

Add `--limit` and an output-path option before any paid run. Print the projected task and
model-call count first. Smoke every arm against the stub. Run each final suite once.

### Done when

- B varies loop and xref independently; C separates raw errors, enforcement, and repair;
- every arm writes the S2 provenance block;
- reports state their trial count and claim no significance from one small run.

**Effort: 1-2 days, plus one small live run.**

## 10. Order and effort

| Step | Effort | Cost | Status |
|---|---|---|---|
| S1 land evaluation work | under an hour | free | done 2026-09-02 |
| S2 artifacts and links | half a day | free | done 2026-09-02, provenance stamp deferred |
| S3 offline tests + provenance stamp | one day | free | next |
| S4 answerability gate | 2-3 days | one small live run | next |
| S5 isolated ablations | 1-2 days | one small live run | next |
| **Remaining** | **4-6 focused days** | **two bounded live runs** | |

S3 first when the work resumes: it is cheap, and the provenance stamp folded into it is
what prevents another silent table drift. S4 is the reason to keep going at all. S5 last,
because it refines experiments that already exist.

## 11. What this version lets the author claim

The point of stopping here is that everything below is defensible from checked-in evidence,
and each line has a design decision behind it that was written down rather than recalled.

**True now, after S1-S2 — this is version 1:**

- hybrid retrieval with reciprocal-rank fusion and cross-encoder reranking, evaluated on a
  versioned suite of author-written tasks with evidence transcribed from expert annotation;
- citation structure and claim support measured as separate concepts, with claim support
  scoped to each claim's own citations rather than the whole evidence pool;
- citation verification implemented as deterministic code — document, page, and section
  existence against the retrieved set — not as an instruction in a prompt;
- every published number backed by a checked-in artifact on the author-written suite, each
  table naming its file and the corpus it ran on;
- a live-model path that costs nothing unless explicitly enabled: tests, CI, the API, and
  Experiment A all run against a deterministic stub.

**After S3:**

- a default test run that is offline and hermetic, and artifacts stamped with the code and
  corpus they came from, so a table cannot silently drift from its source again.

**After S4:**

- an observed failure — every unanswerable question answered — decomposed into a confusion
  matrix rather than an anecdote;
- a specific architectural change aimed at that failure: a pre-write answerability decision
  whose output is enforced in code by verbatim quote checking;
- the change measured against its own baseline on a held-back expanded task set, with the
  decision rule written down before the run;
- the cost of the change reported alongside its benefit.

**After S5:**

- ablations whose arms isolate the component named in the title, so "iterative retrieval
  helps" and "the citation gate helps" are attributable claims rather than bundled ones.

The questions this project should let its author answer without hesitation: why RRF over a
weighted score blend, why a cross-encoder at this position in the pipeline, why these chunk
boundaries, why citations key on document ID rather than title, why claim support is scored
against a claim's own citations, why the answerability gate verifies quotes in code instead
of trusting the verdict, and why each of these was chosen over the alternative that was
considered and rejected. `docs/decisions.md` is where those answers live, and keeping it
current is part of every step above.
