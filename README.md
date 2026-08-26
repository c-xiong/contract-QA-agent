# Eval-Driven Contract Research Agent

A contract research agent over 56 real, expert-annotated commercial contracts. It finds
clauses, answers with document and page citations, follows cross-references into the
carve-outs that qualify a clause, and abstains when the evidence does not support an
answer.

**The claim is not "I built a RAG system." It is "I measured one."** Every number below
comes from a checked-in JSON artifact under [`results/`](results/) — rendered readably in
[`results/RESULTS.md`](results/RESULTS.md) — on a suite of 41
hand-written questions graded against attorney annotations, reported with its sample size
and its limitations.

---

## Results

**Suite:** 41 hand-written questions · `hand-v1` · evidence transcribed from CUAD
(attorney-supervised) and ContractNLI (expert-labelled) · **model `claude-sonnet-5`** ·
retriever `rrf_hybrid+rerank` · 1 trial · k=5.

### The headline: a valid citation is not a supported claim

| Layer | What it checks | Score |
|---|---|---:|
| **1 — deterministic gate** | citation resolves to a real, allowed, *retrieved* location | **1.000** |
| **2 — claim support** (model-graded) | the cited text actually *supports* the sentence | **0.807** |

Every citation the system emitted was structurally valid. **About one factual claim in
five was still not supported by the evidence cited for it** — the answer drifted past what
the excerpt said while pointing at something adjacent.

A deterministic citation checker is necessary and provably insufficient, and the gap is
only visible because the two layers are graded separately rather than collapsed into one
"citation quality" number.

### Retrieval quality propagates to groundedness

The same 41 questions, same model, same graders — only the retrieval arm changed:

| | BM25 | RRF + rerank | |
|---|---:|---:|---|
| evidence span recall | 0.492 | **0.626** | right clause, not just right document |
| **claim support** | 0.706 | **0.807** | unsupported claims 29% → **19%** |
| required points | 0.510 | 0.630 | |

Better evidence makes the writer measurably more grounded. That link is the argument for
spending effort on retrieval rather than on prompt wording, and it is only visible because
retrieval and groundedness are graded as separate dimensions on the same runs.

### Experiment A — retrieval comparison

n = 36 scored (5 abstain tasks excluded: no expected documents, so every arm scores 1.0
by definition). Calls no model; free and fully reproducible.

| Arm | document recall | **evidence span recall** | MRR |
|---|---:|---:|---:|
| BM25 | 0.944 | 0.421 | 1.000 |
| Dense (MiniLM-L6) | 0.889 | 0.384 | 0.944 |
| RRF hybrid (k=60) | **0.954** | 0.468 | 1.000 |
| RRF hybrid + cross-encoder rerank | 0.926 | **0.574** | 1.000 |

Document recall saturates once each task's allowlist is honoured, so **span recall is the
discriminating metric** — right *clause*, not just right document. Reranking wins it by a
wide margin (+0.15 over BM25) while costing a little document recall: it drops a document
on 2 of 8 cross-document comparisons, because re-scoring the top 40 by query-document
relevance can push out a contract that was only marginally represented.

### Experiment B — single-pass RAG vs agentic retrieval

n = 41, live model, `rrf_hybrid+rerank`, identical writer and verifier in both arms.

| | required points | span recall | searches | tokens |
|---|---:|---:|---:|---:|
| single-pass | 0.596 | 0.626 | 41 | 103,826 |
| **agentic loop** | **0.663** | 0.626 | 64 | 136,994 |

**+0.067 on required points, 7 wins / 1 loss / 33 ties**, for 56% more searches and 32%
more tokens. The loop finds the same *documents* — span recall is identical — but the
extra evidence lets the writer state more of the facts an answer needs.

Worth stating plainly: the same experiment against a deterministic stub writer showed
**no difference at all**. The gain is in what the model does with more evidence, not in
retrieval, and only a live run could see it.

### Experiment C — citation gate ablation

n = 41, live model, **no fault injection** — this is what `claude-sonnet-5` actually did.

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

The failure *modes* are the interesting part. A separate fault-injection run — where the
harness deliberately cites a real-but-unretrieved location once every three answers —
produced 10 `document_not_allowed` and 3 `not_in_evidence`. The real model produced
neither. It does not invent documents; it makes **formatting errors** (4 unparseable
citations) and cites **sections our chunker never indexed** (2). The second of those is
partly a defect in this system rather than the model's: a section that exists in the
contract but that section detection missed is rejected as though it were fabricated.

That distinction only exists because the gate reports a specific error code per failure
instead of a boolean.

### Full eval suite

| Grader | Score | Reading |
|---|---:|---|
| `citation_validity` | 1.000 | every citation structurally valid |
| `forbidden_claims` | 1.000 | never asserted a claim the corpus contradicts |
| `reciprocal_rank` | 1.000 | an expected document was always rank 1 |
| `document_recall` | 0.935 | |
| `claim_support` | 0.807 | **19% of claims unsupported by their own citation** |
| `abstention_correctness` | 0.854 | all 6 failures one-directional — see limitations |
| `required_points` | 0.630 | |
| `evidence_span_recall` | 0.626 | |

n = 41, 8 independently versioned graders, `grader_versions` recorded with every result.

### Corpus and parser

| Measurement | Value | Command |
|---|---:|---|
| CUAD filename → PDF resolution | **510 / 510** | `scripts/sample_contracts.py` |
| Clause-span match rate | **91.5%** (332/363) | `scripts/check_span_match_rate.py` |
| Documents · chunks | 56 · 7,453 | `scripts/ingest.py` |
| Cost per live query | **~$0.01** | `scripts/estimate_cost.py` |
| Container image | 1.5 GB | `docker build .` |

---

## Quickstart

```bash
uv sync
uv run pytest                                   # 312 tests, no network, no API key

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
deterministic stub by default, so tests, CI, the API, and Experiment A all run on a fresh
clone with no API key.

```bash
docker compose up      # corpus mounted from ./data, never baked into the image
```

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

Two components carry the weight.

**The citation gate** (`app/evidence/citation_verifier.py`). A model can cite a real
document, a real page, and a real section it was never shown; four of five checks pass.
The fifth — is this grounded in evidence actually retrieved — is the one that catches it.
Deterministic code with tests, never a prompt instruction.

**Cross-reference resolution** (`app/evidence/cross_reference.py`). A cap in §8.1 is
qualified by carve-outs in §8.3. Retrieval returns 8.1 because it matches "liability cap".
An answer citing only 8.1 states an unqualified cap, cites a real page, and passes *every*
citation check — and is still wrong. Only following the reference fixes it, because
incompleteness has no signature at the citation layer.

## Layout

| Path | Contents |
|---|---|
| `app/ingestion/` | CUAD + ContractNLI adapters, PDF parser, chunker, manifest, synthetic builder, store |
| `app/retrieval/` | Protocol, BM25, dense/FAISS, RRF fusion, cross-encoder rerank |
| `app/agent/` | LangGraph workflow, state, policies, typed tools, model adapter |
| `app/evidence/` | Citation parser, deterministic gate, cross-reference resolver, claim support |
| `app/api/` | FastAPI service and a read-only trace inspector |
| `evals/` | Schema, loader, runner, 8 graders |
| `experiments/` | Experiments A, B, C |

## Trace inspector

`uvicorn app.api.main:app` serves a read-only page at `/` that streams the agent's real
node-by-node execution over SSE, driven by `graph.astream` rather than a front-end timer.
It shows which node is running, how many times the loop went round, which evidence was
pulled by cross-reference rather than retrieval, and each citation with its excerpt and
verification status. One self-contained HTML file, no build step, no dependencies.

It found two real bugs during development: a router that read a status *label* instead of
the sufficiency verdict (so the loop ran to budget on every question, and the
term-coverage policy was dead code), and one citation counted twice when the writer cited
the same clause in two sentences.

## Known limitations

- **`abstention_correctness` is 0.854 and every failure is one-directional**: the system
  answers `NotMentioned` questions instead of abstaining. Retrieval on a 7,453-chunk
  corpus almost always returns *something*, and layer 1 cannot tell "on topic" from
  "answers the question". Closing it means routing layer 2's verdict back into the
  abstention decision, which is not built.
- **Experiment C has no live measurement yet** — the reported 31.7% is a fault-injection
  rate, not a model's.
- **Dense and rerank use small general-purpose models** with no legal training. Their
  Experiment A numbers are a floor for the approach, not a ceiling.
- **Cross-reference resolution is depth 1** with a 4-chunk budget, and resolving "Section
  11" pulls its subsections too — on one document that meant a COUNTERPARTS clause
  consuming budget.
- **BM25 has no stopword handling**; high-document-frequency terms self-suppress via
  negative Okapi IDF, which mostly helps and occasionally returns nothing.
- **n = 41.** A one- or two-task difference is noise. Every comparison above reports
  paired win/loss/tie counts for that reason; no significance is claimed.
- Definitions are detected but not resolved.

## Evaluation methodology

Questions are **hand-written**. Evidence labels are transcribed from expert annotation —
CUAD clause spans, ContractNLI hypothesis labels, and synthetic edits known by
construction — never inferred. A dataset whose questions were written by the model family
under evaluation measures itself, so the schema records question provenance per task
(`question_source`, `question_source_lang`) and the loader refuses a task still carrying
placeholder text.

Reporting follows one rule throughout: no bare aggregate. Every table carries its n, every
comparison carries paired counts, and per-category breakdowns accompany every mean.

## Data and licensing

Code is MIT (`LICENSE`). The corpora are **not** distributed here and are downloaded by
`scripts/` under their own terms — CUAD v1 (The Atticus Project) and ContractNLI (Koreeda
& Manning), both **CC BY 4.0**. Those licences cover the annotations; the underlying
contracts are public SEC filings. Attribution and the sampling criterion are in
`data/README.md`.
