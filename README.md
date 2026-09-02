# Eval-Driven Contract Research Agent

A retrieval agent that answers questions about commercial contracts and cites its
sources by document, page, and section. It finds clauses across 56 real,
expert-annotated agreements, follows cross-references into the carve-outs that qualify a
clause, verifies every citation in deterministic code before returning it, and abstains
when the retrieved evidence does not support an answer.

The other half of the repository is the measurement. A suite of 41 hand-written
questions, 8 independently versioned graders, and three experiments — all backed by
checked-in JSON artifacts under [`results/`](results/), rendered readably in
[`results/RESULTS.md`](results/RESULTS.md). Every figure in this README is recomputable
from one of those files, and each is reported with its sample size and its limitations.

## What a query looks like

```console
$ uv run python scripts/demo.py --trace "What limits liability, and does it always apply?"
corpus   : 56 documents, 6270 chunks
retriever: bm25
model    : claude-sonnet-5

Two different limitation-of-liability provisions appear in the excerpts, and neither is
absolute:

- In doc-006, the exclusion of indirect, incidental, special, exemplary, punitive or
  consequential damages does **not** apply to (a) fraud, willful misconduct or gross
  negligence, (b) a party's breach of its obligations under Article 9 or Section 5.8,
  (c) matters covered under a redacted provision, and (d) damages required to be paid to
  a third party as part of an indemnification claim under Article 11
  [doc-006, p. 65, §7.3].
- In doc-019, Changepoint's liability for direct, indirect, special, incidental and/or
  consequential damages is capped at "the fee paid by End User to Changepoint under such
  agreement," with no carve-outs stated in the excerpt [doc-019, p. 16, §2].

So the answer is no — at least the doc-006 cap is expressly subject to several
exceptions, while the doc-019 cap appears unqualified based on the excerpt provided.

status    : completed
retrieved : 9 chunks from ['doc-036', 'doc-037', 'doc-034', 'doc-006', 'doc-019']
citations : 2 verified
    [doc-006, p. 65, §7.3]
    [doc-019, p. 16, §2]
tokens    : 3708 in, 334 out

trace:
    search           5 chunks (5 total)
    assess           stop: 5 locations, term coverage 100% (thresholds 2, 50%)
    resolve_refs     pulled 4 chunk(s) from 4 reference(s); 1 unresolved; 33 dropped for budget
    select_evidence  7 evidence items (2 via cross-reference)
    write            334 output tokens
    verify           2 citation(s), all verified
    finalize         citations verified
```

Two of the seven evidence items reached the writer through cross-reference resolution
rather than retrieval — that is how the Article 9 and Section 5.8 carve-outs got into the
answer. Both citations were checked against the evidence actually retrieved before the
answer was returned.

`scripts/demo.py` runs the BM25 arm; the API and the eval runner take `--arm` /
`?arm=` and can run any of `bm25`, `dense`, `rrf_hybrid`, `rrf_hybrid_rerank`.

## Quickstart

```bash
uv sync
uv run pytest                                   # no network, no API key, no cost

uv run python scripts/download_cuad.py          # ~101 MB, checksum-verified
uv run python scripts/download_contractnli.py   # ~65 MB
uv run python scripts/sample_contracts.py --size 40
uv run python scripts/sample_contractnli.py --size 10
uv run python scripts/build_synthetic.py
uv run python scripts/ingest.py
uv run python scripts/build_index.py

uv run python scripts/demo.py --trace "What limits liability, and does it always apply?"
uv run python experiments/retrieval_comparison.py --suite hand
uvicorn app.api.main:app --reload                # http://localhost:8000
```

**Nothing spends money unless `CRA_LIVE_MODEL=1`.** Every model call routes to a
deterministic stub by default, so the tests, CI, the API, and Experiment A all run on a
fresh clone with no API key.

```bash
docker compose up      # corpus mounted from ./data, never baked into the image
```

## Results

**Suite:** 41 hand-written questions · `hand-v1` · evidence transcribed from CUAD
(attorney-supervised) and ContractNLI (expert-labelled) · **model `claude-sonnet-5`** ·
retriever `rrf_hybrid+rerank` · 1 trial · k=5.

> **Corpus versions.** The chunker changed on 2026-08-28: it had been measuring section
> headings against the wrap column instead of the paragraph, so a liability cap was
> routinely severed from the carve-outs qualifying it. Re-ingestion took the corpus from
> 7,453 chunks to 6,270. Retrieval-only figures are free to re-run and are current at
> **6,270**. Live-model figures cost an API run each and are still reported on **7,453**,
> the corpus they were measured on. Every table states which corpus it ran on and links
> the artifact it came from; nothing below is an estimate.

### A valid citation is not a supported claim

| Layer | What it checks | Score |
|---|---|---:|
| **1 — deterministic gate** | citation resolves to a real, allowed, *retrieved* location | **1.000** |
| **2 — historical evidence-pool support** (`claim_support@1`) | each sentence is checked against the complete selected-evidence pool | **0.807** |

*Corpus 7,453 — measured before the 2026-08-28 chunking fix.*
[`results/eval-hand-rerank.json`](results/eval-hand-rerank.json)

Every citation the system emitted was structurally valid. The historical **0.807** score
means about one factual claim in five was not supported by the answer's complete selected-
evidence pool. It does **not** measure whether each claim was supported by its own citation.
No live `claim_support@2` score has been run or estimated yet.

A deterministic citation checker is necessary and provably insufficient. The gap is
visible only because the two layers are graded separately instead of collapsed into one
"citation quality" number.

### Retrieval quality propagates to groundedness

The same 41 questions, same model, same graders — only the retrieval arm changed:

| | BM25 | RRF + rerank | |
|---|---:|---:|---|
| evidence span recall | 0.492 | **0.626** | right clause, not just right document |
| **historical evidence-pool support** | 0.706 | **0.807** | `claim_support@1` |
| required points | 0.510 | 0.630 | |

*Corpus 7,453 — measured before the 2026-08-28 chunking fix.*
[`results/eval-hand-bm25.json`](results/eval-hand-bm25.json) ·
[`results/eval-hand-rerank.json`](results/eval-hand-rerank.json)

Better evidence makes the writer measurably more grounded. That link is the argument for
spending effort on retrieval rather than on prompt wording, and it is visible only because
retrieval and groundedness are graded as separate dimensions on the same runs.

### Experiment A — retrieval comparison

n = 36 scored (5 abstain tasks excluded: with no expected documents, every arm scores 1.0
by definition). Calls no model, so it is free and fully reproducible.
**Corpus 6,270 — current.** [`results/experiment-a-retrieval.json`](results/experiment-a-retrieval.json)

| Arm | document recall | **evidence span recall** | MRR |
|---|---:|---:|---:|
| BM25 | 0.944 | 0.458 | 1.000 |
| Dense (MiniLM-L6) | 0.843 | 0.394 | 0.889 |
| RRF hybrid (k=60) | 0.944 | 0.495 | 1.000 |
| RRF hybrid + cross-encoder rerank | 0.944 | **0.611** | 1.000 |

Document recall saturates once each task's allowlist is honoured — the three arms with a
lexical component tie at 0.944 and only the dense-only arm falls behind — which makes
**span recall the discriminating metric**: right *clause*, not just right document.
Reranking wins it by +0.153 over BM25, on **8 wins / 2 losses / 26 ties** (n = 36).

Reranking does *not* fix `cross_document_comparison`: both arms score 0.25 span recall on
those 8 tasks, and document recall there is 0.75 against 1.00 everywhere else. A
cross-encoder re-scores what fusion handed it, so a contract that was only marginally
represented in the top 40 is not recoverable at that stage. That category is the one place
where the retrieval design is visibly the limit rather than the ranking.

The same suite on the 7,453-chunk corpus scored span recall 0.421 / 0.384 / 0.468 / 0.574
([`results/experiment-a-retrieval-corpus7453.json`](results/experiment-a-retrieval-corpus7453.json),
superseded and kept for the comparison). Every arm is higher after the four chunking
fixes, and the rerank arm gained the most (+0.037), which is the expected direction: a
carve-out folded back into the clause it qualifies is a better cross-encoder input than
the fragment it used to be.

### Experiment B — single-pass RAG vs agentic retrieval

n = 41, live model, `rrf_hybrid+rerank`, identical writer and verifier in both arms.
**Corpus 7,453 — measured before the 2026-08-28 chunking fix.**
[`results/experiment-b-agentic.json`](results/experiment-b-agentic.json)

| | required points | span recall | searches | tokens |
|---|---:|---:|---:|---:|
| single-pass | 0.596 | 0.626 | 41 | 103,826 |
| agentic loop | **0.663** | 0.626 | 64 | 136,994 |

**7 wins / 1 loss / 33 ties** on required points, for 56% more searches and 32% more
tokens. Span recall is identical in both arms: the loop's extra rounds changed what the
writer *did* with the evidence, not which evidence was found — the refinement trigger
fires on term coverage, and a second query that returns the same chunks costs a search
without adding one.

Two caveats against over-reading it. This is one trial at n = 41, so a 7-task margin is
directional and nothing more. And the corpus has since changed: four chunking fixes now
keep a carve-out attached to the clause it qualifies, which is exactly the evidence an
extra search round used to recover. **Whether the loop still pays on the 6,270-chunk
corpus has not been measured** — that rerun costs an API run and is queued behind the work
in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md), not asserted here.

The experiment also cannot yet say *which* component earned the difference: the agentic
arm turns on the search loop and cross-reference resolution together. Separating them is a
2 x 2 design, and it is the first thing this experiment needs.

### Experiment C — citation gate ablation

n = 41, live model, **no fault injection** — this is what `claude-sonnet-5` actually did.
**Corpus 7,453 — measured before the 2026-08-28 chunking fix.**
[`results/experiment-c-citation-gate.json`](results/experiment-c-citation-gate.json) The 2 `section_not_found`
failures below are the ones most likely to have moved: one of the four fixes normalises
`9.0` and `Section 9.0` to Article 9, so a reference the chunker previously could not
resolve now resolves.

| | value |
|---|---:|
| Citations emitted with the gate **off** | 70 |
| Of those, failing verification | **6 (8.6%)** |
| Tasks affected | 8 / 41 |
| `unparseable` / `section_not_found` | 4 / 2 |
| Citation validity, gate **on** | 1.000 |
| Token cost of the gate | +15% (129,458 → 148,636) |

**Sonnet miscites about one citation in twelve**, and the gate catches and repairs all of
them for 15% more tokens.

The failure *modes* are the more useful part. A separate fault-injection run
([`results/experiment-c-fault-injection.json`](results/experiment-c-fault-injection.json)) —
where the harness deliberately cites a real-but-unretrieved location once every three
answers — produced 10 `document_not_allowed` and 3 `not_in_evidence`, all caught. The real
model produced neither. It does not invent documents; it makes **formatting errors** (4 unparseable
citations) and cites **sections the chunker never indexed** (2). The second of those is
partly a defect in this system rather than the model's: a section that exists in the
contract but that section detection missed is rejected as though it were fabricated. That
distinction exists only because the gate reports a specific error code per failure instead
of a boolean.

### Full eval suite

| Grader | Score | Reading |
|---|---:|---|
| `citation_validity` | 1.000 | every citation structurally valid |
| `forbidden_claims` | 1.000 | never asserted a claim the corpus contradicts |
| `reciprocal_rank` | 1.000 | an expected document was always rank 1 |
| `document_recall` | 0.935 | |
| `claim_support@1` | 0.807 | historical evidence-pool support; not own-citation support |
| `abstention_correctness` | 0.854 | all 6 failures one-directional — see limitations |
| `required_points` | 0.630 | |
| `evidence_span_recall` | 0.626 | |

n = 41, 8 independently versioned graders, `grader_versions` recorded with every result.
**Corpus 7,453 — measured before the 2026-08-28 chunking fix.**
[`results/eval-hand-rerank.json`](results/eval-hand-rerank.json); the BM25 arm of the same
suite is [`results/eval-hand-bm25.json`](results/eval-hand-bm25.json).

Retrieval-only re-runs on the current corpus raised span recall for every arm (Experiment
A above), so the retrieval rows here are most likely low. The answer-quality rows have
**not** been re-measured — that costs an API run — and no estimate of where they would
land is offered.

### Corpus and parser

| Measurement | Value | Command |
|---|---:|---|
| CUAD filename → PDF resolution | **510 / 510** | `scripts/sample_contracts.py` |
| Clause-span match rate | **91.5%** (332/363) | `scripts/check_span_match_rate.py` |
| Documents · chunks | 56 · 6,270 | `scripts/ingest.py` |
| Cost per live query | **~$0.01** | `scripts/estimate_cost.py` |
| Container image | 1.5 GB | `docker build .` |

## How it works

```
INGESTION — three source types, one store
  CUAD PDFs   ─▶ parser ─▶ section-aware chunker ─┐
  synthetic   ─▶ derived text ────────────────────┼─▶ store ─┬─▶ BM25
  ContractNLI ─▶ span segmentation ───────────────┘          └─▶ FAISS

QUERY
  question ─▶ search ─▶ assess ─┬─(thin)──▶ refine ─▶ search   (bounded: 3/3/1)
                                └─(enough)─▶ resolve_refs ─▶ normalize
                                                   │
                            follows "subject to Section 8.3"
                                                   ▼
                                    write ─▶ citation gate ─┬─▶ answer
                                              ▲             ├─▶ repair (×1)
                                              └─────────────┘
                                                            └─▶ abstain

EVALUATION   41 tasks ─▶ 8 versioned graders ─▶ JSON artifact
```

Two components carry most of the weight.

**The citation gate** ([`app/evidence/citation_verifier.py`](app/evidence/citation_verifier.py)).
A model can cite a real document, a real page, and a real section it was never shown; four
of five checks pass. The fifth — is this grounded in evidence actually retrieved — is the
one that catches it. It is deterministic code with tests, never a prompt instruction.

**Cross-reference resolution** ([`app/evidence/cross_reference.py`](app/evidence/cross_reference.py)).
A cap in §8.1 is qualified by carve-outs in §8.3. Retrieval returns 8.1 because it matches
"liability cap". An answer citing only 8.1 states an unqualified cap, cites a real page,
and passes *every* citation check — and is still wrong. Only following the reference fixes
it, because incompleteness leaves no signature at the citation layer.

### Trace inspector

`uvicorn app.api.main:app` serves a read-only page at `/` that streams the agent's real
node-by-node execution over SSE, driven by `graph.astream` rather than a front-end timer.
It shows which node is running, how many times the loop went round, which evidence was
pulled by cross-reference rather than retrieval, and each citation with its excerpt and
verification status. One self-contained HTML file — no build step, no dependencies.

It found two real bugs during development: a router that read a status *label* instead of
the sufficiency verdict (so the loop ran to budget on every question, and the
term-coverage policy was dead code), and one citation counted twice when the writer cited
the same clause in two sentences.

### Repository layout

| Path | Contents |
|---|---|
| [`app/ingestion/`](app/ingestion/) | CUAD + ContractNLI adapters, PDF parser, chunker, manifest, synthetic builder, store |
| [`app/retrieval/`](app/retrieval/) | Protocol, BM25, dense/FAISS, RRF fusion, cross-encoder rerank |
| [`app/agent/`](app/agent/) | LangGraph workflow, state, policies, typed tools, model adapter |
| [`app/evidence/`](app/evidence/) | Citation parser, deterministic gate, cross-reference resolver, claim support |
| [`app/api/`](app/api/) | FastAPI service and the read-only trace inspector |
| [`evals/`](evals/) | Schema, loader, runner, 8 graders |
| [`experiments/`](experiments/) | Experiments A, B, C |
| [`results/`](results/) | The JSON artifact behind every number above |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | What is built, what is next, and what was deliberately dropped |

## Known limitations

- **`abstention_correctness` is 0.854 and every failure is one-directional**: all five
  unanswerable tasks were answered rather than abstained on. Retrieval over 6,270 chunks
  almost always returns *something*, and layer 1 cannot tell "on topic" from "answers the
  question". Closing the gap means a pre-write answerability decision whose verdict is
  enforced in code — specified as step S4 of [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md),
  not built.
- **Dense retrieval and reranking use small general-purpose models** with no legal
  training. Their Experiment A numbers are a floor for the approach, not a ceiling.
- **Cross-reference resolution is depth 1** with a 4-chunk budget, and resolving "Section
  11" pulls its subsections too — on one document that meant a COUNTERPARTS clause
  consuming budget.
- **BM25 has no stopword handling**; high-document-frequency terms self-suppress via
  negative Okapi IDF, which mostly helps and occasionally returns nothing.
- **Several live-model tables predate the 2026-08-28 chunking fix** and are labelled with
  the corpus they ran on rather than refreshed, because re-running each costs an API run.
- **n = 41.** A one- or two-task difference is noise. Every comparison above reports
  paired win/loss/tie counts for that reason, and no significance is claimed.
- Definitions are detected but not resolved.

## Evaluation methodology

Questions are **hand-written**. Evidence labels are transcribed from expert annotation —
CUAD clause spans, ContractNLI hypothesis labels, and synthetic edits known by
construction — never inferred. A dataset whose questions were written by the model family
under evaluation partly measures itself, so the schema records question provenance per
task (`question_source`, `question_source_lang`) and the loader refuses a task still
carrying placeholder text.

Reporting follows one rule throughout: no bare aggregate. Every table carries its n, every
comparison carries paired counts, and per-category breakdowns accompany every mean.

The design decisions behind the chunker, the fusion constant, the graders, and the agent
loop are argued in `# DECISION:` comments on the modules that make them — each naming what
was chosen, what was rejected, and what would change if the alternative were taken. The
specification and the decision log are kept out of this repository; what is left of them
that a reader needs is in the code and in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Data and licensing

Code is MIT ([`LICENSE`](LICENSE)). The corpora are **not** distributed here; they are
downloaded by `scripts/` under their own terms — CUAD v1 (The Atticus Project) and
ContractNLI (Koreeda & Manning), both **CC BY 4.0**. Those licences cover the annotations;
the underlying contracts are public SEC filings. Attribution and the sampling criterion
are in [`data/README.md`](data/README.md).
