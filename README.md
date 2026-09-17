# Eval-Driven Contract Research Agent

A bounded contract QA workflow that retrieves clauses, follows explicit cross-references,
optionally checks evidence answerability, and verifies citation locations
in deterministic code. The corpus contains 40 CUAD agreements, 10 ContractNLI NDAs and
6 labelled synthetic variants: **56 documents and 6,270 chunks**.

The main engineering artifact is the evaluation: a frozen **41-question, author-reviewed
suite**, versioned retrieval/answer graders, controlled ablations and checked-in per-task
results. The system is a LangGraph workflow with explicit Python tools and working state;
it does not claim autonomous tool selection, an LLM planner or persistent conversation memory.

## Quickstart

```bash
uv sync --locked
CRA_LIVE_MODEL=0 uv run pytest

# Acquire the public datasets and rebuild the local corpus/index.
CRA_LIVE_MODEL=0 uv run python scripts/download_cuad.py
CRA_LIVE_MODEL=0 uv run python scripts/download_contractnli.py
CRA_LIVE_MODEL=0 uv run python scripts/sample_contracts.py --size 40
CRA_LIVE_MODEL=0 uv run python scripts/sample_contractnli.py --size 10
CRA_LIVE_MODEL=0 uv run python scripts/build_synthetic.py
CRA_LIVE_MODEL=0 uv run python scripts/ingest.py
CRA_LIVE_MODEL=0 uv run python scripts/build_index.py

CRA_LIVE_MODEL=0 uv run python scripts/demo.py --trace "What limits liability, and does it always apply?"
CRA_LIVE_MODEL=0 uv run uvicorn app.api.main:app --reload
```

The local trace inspector is at `http://localhost:8000`. It streams actual LangGraph
node events over SSE and displays selected evidence and citation checks. `docker compose up`
is the alternative local reproducibility path; corpus files are mounted from `data/`.

**Model access is free by default.** Set `CRA_LIVE_MODEL=1` and `ANTHROPIC_API_KEY` only for
an intentional live run. A local `.env` can override defaults, which is why free commands
above set the flag explicitly. Stub answers exercise wiring and cannot measure semantics.
Normal pytest blocks external sockets and model loading; real model tests are opt-in:

```bash
CRA_LIVE_MODEL=0 uv run pytest -m model_download
CRA_LIVE_MODEL=0 uv run pytest -m corpus       # downloaded ContractNLI required
```

On macOS, if the native FAISS/BLAS stack crashes or oversubscribes threads, run retrieval
with `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false`.
The closeout retrieval experiment used these limits and reproduced the prior scores.

## Current results

All results below use `hand-v1`, the current 6,270-chunk corpus, k=5 and one trial.
This is a small development benchmark, not an unseen test set or a production SLA.
See [the artifact index](results/README.md), [generated tables](results/RESULTS.md), and
[evaluation definitions](docs/eval-methodology.md).

### A: retrieval quality

36 scored questions; five abstention tasks excluded because they have no expected evidence.
Document allowlists match the scoped API behavior. [Current artifact](results/experiment-a-current.json).

| Retrieval arm | Document recall | Evidence-span recall@5 | MRR |
|---|---:|---:|---:|
| BM25 | 0.944 | 0.458 | 1.000 |
| MiniLM + FAISS | 0.843 | 0.394 | 0.889 |
| RRF hybrid | 0.944 | 0.495 | 1.000 |
| RRF + cross-encoder rerank | 0.944 | 0.611 | 1.000 |

Reranking raises evidence-span recall by **15.3 percentage points** versus BM25 on these
36 questions: **8 wins, 2 losses, 26 ties**. Document recall alone hides that difference.
Cross-document comparison remains weak: both BM25 and rerank score **0.25** span recall
on eight tasks. Reordering cannot recover a document absent from the candidate pool.

### B–D: current controlled comparisons

All three use Sonnet 5 and 41 frozen development questions. Full outputs, tokens and
paired comparisons are pinned in the [artifact index](results/README.md).

- **B — loop × cross-reference:** required-point coverage is 0.602 / 0.628 / 0.632 /
  0.614 for single/no-xref, single/xref, loop/no-xref and loop/xref. Loop and xref main
  effects are only +0.008 and +0.004; retrieval recall is unchanged. This run does not
  establish a reliable answer-quality benefit from additional orchestration.
- **C — citation verification:** delivered references pass the independent structural
  audit at 56/59 ungated, 53/53 verify-only and 63/63 verify+repair. Repair succeeds in
  1/9 triggered tasks. Each arm has its own stochastic initial drafts, so this is not
  a replay of identical answers. Structural validity is not semantic correctness.
- **D — answerability:** both arms abstain on all five expected-abstention tasks.
  On 36 answerable tasks, explicit over-abstentions increase from 4 to 11, with four
  additional execution failures when enabled. Runtime tokens increase from 153,062
  to 264,936. **The gate fails the adoption rule and is disabled by default.**

Behavior counts use runtime status: a `completed` response can contain a prose refusal.
The independent claim judge and [three inspected failures](results/ANALYSIS.md) expose
that limitation. No answer-quality improvement is inferred from retrieval scores.

### Historical evidence

Old live runs used 7,453 chunks and `claim_support@1`, which tested support anywhere in
the selected evidence pool. Those scores are not current `claim_support@2` scores.
Historical JSON is retained in `results/` for auditability.

The old citation ablation's **8.6% rate is withdrawn**: it counted malformed citation
errors in the numerator but omitted those attempts from the denominator. The new ablation
counts malformed attempts explicitly and independently audits raw and delivered answers.
A valid citation location still does not prove that the associated claim is supported.

## Workflow and constraints

```text
question -> search -> assess -> refine/search (bounded)
                       |
                       v
                 resolve references -> select evidence
                       |
                       v
                 answerability (opt-in) -> abstain / write supported scope
                                             |
                                             v
                                      citation verifier
                                      /       |       \
                                  answer   repair x1   abstain
```

- **Retrieval:** BM25, normalized MiniLM embeddings in FAISS, reciprocal-rank fusion and
  cross-encoder reranking. The API and experiments expose all four arms.
- **Context:** section-aware chunks, provenance headers, whole-item evidence budgets,
  deduplication and depth-one clause-reference expansion. Conflicting versions stay distinct.
- **Answerability (opt-in, `CRA_ANSWERABILITY_GATE=1`):** a separate pre-write JSON verdict identifies supported/missing aspects
  and quotes evidence IDs. Python validates the IDs and exact quoted text. Partial answers
  carry their support scope through citation repair. Invalid verdicts remain execution failures.
- **Citation verification:** document/page/section existence, document allowlists and presence
  in retrieved evidence are checked in code, with at most one repair. This is structural
  enforcement; semantic support is scored independently with `claim_support@2`.
- **Execution:** search/iteration/reference/evidence/repair limits and model timeouts are
  enforced in code. Each result records model calls, tokens, elapsed time and trace events.

## Reproduce the comparisons

Every live script should first be exercised with `CRA_LIVE_MODEL=0` and a small `--limit`.
B turns loop and cross-reference on/off independently. C separates no verification,
verification only and verification plus repair. B/C disable answerability to isolate their
named factors. D measures answerability off/on with an independent offline claim judge.

```bash
CRA_LIVE_MODEL=0 uv run python experiments/retrieval_comparison.py --suite hand
CRA_LIVE_MODEL=0 uv run python experiments/agentic_comparison.py --suite hand --limit 2
CRA_LIVE_MODEL=0 uv run python experiments/citation_gate_ablation.py --suite hand --limit 3 --fault-every 2
CRA_LIVE_MODEL=0 uv run python experiments/answerability_comparison.py --suite hand --limit 2

# Explicit paid runs over public benchmark excerpts; reports stay local.
CRA_LIVE_MODEL=1 uv run python experiments/agentic_comparison.py --suite hand --output results/experiment-b-factorial.json
CRA_LIVE_MODEL=1 uv run python experiments/citation_gate_ablation.py --suite hand --output results/experiment-c-isolated.json
CRA_LIVE_MODEL=1 uv run python experiments/answerability_comparison.py --suite hand --claim-support --output results/experiment-d-answerability.json
uv run python scripts/summarize_results.py
```

Reports identify code/worktree state, source and corpus hashes, dataset/model/grader versions,
budgets, per-task outputs and paired comparisons. D records offline judge tokens separately
from application tokens and latency. One trial supports descriptive comparisons only.

A separate [20-task ContractNLI diagnostic](evals/datasets/answerability/README.md) transcribes
expert hypotheses/labels but uses an AI-authored question wrapper. Its version begins
`provisional-generated-`; it is supplementary validation and never supplies resume metrics.

## Scope and limitations

The corpus is small and annotation-selected. General-purpose embedding/reranking models,
limited candidate coverage, depth-one references, unresolved definitions and sentence-level
claim extraction constrain quality. The runtime gate and offline judge use independent
prompts but the same model family; both can misjudge semantics. No zero-hallucination,
production-readiness or held-out generalization claim is made.

## Maintained documentation

- [Closeout checklist](docs/IMPLEMENTATION_PLAN.md): deliverables and the stopping point.
- [Evaluation methodology](docs/eval-methodology.md): data provenance, metrics and limitations.
- [Design decisions](docs/decisions.md): architecture tradeoffs and interview walkthrough.
- [Data and attribution](data/README.md): public sources, sampling and licences.

Code is MIT ([LICENSE](LICENSE)). CUAD and ContractNLI annotations are CC BY 4.0; corpus
files are acquired from their public releases and are not distributed in this repository.
Development used coding-agent assistance; the `hand-v1` suite was authored/reviewed by the
project author and generated supplements are labelled separately.
