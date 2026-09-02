# Data

Sources, licenses, and attribution for every document in this corpus. Nothing in
`data/` except this file and `corpus_manifest.json` is checked in — the corpora are
large and every artifact is reproducible from `scripts/`.

```bash
uv run python scripts/download_cuad.py         # ~101 MB, checksum-verified
uv run python scripts/download_contractnli.py  # ~65 MB
uv run python scripts/sample_contracts.py --size 40
uv run python scripts/sample_contractnli.py --size 10
uv run python scripts/build_synthetic.py
uv run python scripts/ingest.py
uv run python scripts/build_index.py           # dense FAISS index
```

## Layout

| Path | Checked in | Contents |
|---|---|---|
| `data/corpus_manifest.json` | **yes** | `doc-NNN` → source file. The authority on document identity. |
| `data/raw/` | no | Downloaded corpora, exactly as acquired. Never written to by app code. |
| `data/processed/` | no | Parsed documents and chunks (JSONL). Rebuilt by `scripts/ingest.py`. |
| `data/index/` | no | Retrieval indices. Rebuilt from `data/processed/`. |

## Sources

### CUAD v1 — Contract Understanding Atticus Dataset

510 real commercial contracts drawn from SEC EDGAR, with 13,101 clause annotations
across 41 categories, labeled under the supervision of practicing attorneys.

- Landing page: <https://www.atticusprojectai.org/cuad>
- Canonical archive: <https://zenodo.org/records/4595826> (`CUAD_v1.zip`, md5 `c38f490a984420b8a62600db401fafd5`)
- Annotation code: <https://github.com/TheAtticusProject/cuad>

**License: CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>).

> The Contract Understanding Atticus Dataset (CUAD) v1 is created by The Atticus
> Project and licensed under CC BY 4.0. Hendrycks, D., Burns, C., Chen, A., and Ball,
> S. "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review." NeurIPS
> Datasets and Benchmarks, 2021.

The license covers **the annotations**. The underlying contracts are public filings
published by the U.S. Securities and Exchange Commission and are public records.

Used here: `full_contract_pdf/` as the ingestion corpus, and `master_clauses.csv` as
the ground-truth source. `full_contract_txt/` is deliberately not ingested — parsing
the real PDFs exercises the parser on real layout problems (running headers, page
breaks mid-clause, signature blocks) that the extracted text would hide.

### ContractNLI

607 real non-disclosure agreements, each annotated against the same 17 hypotheses as
Entailment, Contradiction, or NotMentioned, with evidence spans for the first two.

- <https://stanfordnlp.github.io/contract-nli/>
- Archive: <https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip>
- Koreeda, Y. and Manning, C. D. "ContractNLI: A Dataset for Document-level Natural
  Language Inference for Contracts." Findings of EMNLP 2021. **CC BY 4.0.**

Ten NDAs are used, each selected for carrying both `Contradiction` and `NotMentioned`
labels. `NotMentioned` is expert-labelled abstention ground truth; `Contradiction` is a
claim-support test set where on-topic evidence exists and contradicts the claim.

**No pagination.** ContractNLI ships extracted text with a span table, not paginated
PDFs. Chunks from this slice carry `page_number = null` and cite as `[doc-051, §s37]`.
Synthesizing page numbers was rejected: fabricated provenance in a provenance system is
the wrong trade (SPEC 6.3).

**One integration trap.** An annotation's `spans` field holds **indices into the
document's `spans` list**, not character offsets. Reading them as offsets yields valid
substrings at the wrong place with no error. `tests/unit/test_contractnli.py` round-trips
a known annotation back to its literal text and demonstrates the wrong reading.

### Synthetic overlay

Six documents, each derived from a real CUAD contract already in the corpus, so ground
truth is known by construction. Per SPEC 6.6, exactly three uses:

| Use | Count | Construction |
|---|---:|---|
| Conflicting versions | 2 | A numeric term and the effective date altered |
| Prompt injection | 2 | An instruction-like passage inserted into the body |
| Near-duplicate distractors | 2 | Clauses paraphrased into a separate document |

Every synthetic document is named `synthetic-*`, carries `is_synthetic: true` on its
`Document` and manifest entry, and records its own edits in its artifact under
`data/raw/synthetic/`.

**One adaptation forced by the data.** SPEC 6.6 says to alter "the liability cap figure".
CUAD contracts are SEC filings under confidential treatment, so the cap is usually
redacted (`[*]`, `[***]`) or expressed as a formula — none of the first five contracts
has a dollar amount on its liability page. The builder falls back to altering a notice
period, and the `edits` record names which field actually changed. See
`docs/decisions.md`.

## Corpus composition (current)

| Slice | Registered | Ingested | Purpose |
|---|---:|---:|---|
| CUAD | 40 | 40 | Retrieval and citation ground truth |
| ContractNLI | 10 | 10 | Abstention and claim-support ground truth |
| Synthetic | 6 | 6 | Conflicts, injection, distractors |
| **Total** | **56** | **56** | 6,270 chunks |

Within the 55-60 target set in `docs/SPEC.md` §6.7. The corpus is deliberately small:
it is selected for annotation coverage, not for document count.

## Sampling criterion

Recorded verbatim in `data/corpus_manifest.json` and reproduced in
`docs/eval-methodology.md` §2, so the corpus can be rebuilt rather than merely
described. The sampler is seeded and append-only.

## Document identity

`doc-NNN` identifiers are assigned by `data/corpus_manifest.json` and are **never
reused, renumbered, or repointed**. Citations, eval ground truth, and retrieval
allowlists all key on them, and a silently repointed ID would invalidate every eval
label without failing a single test. Growing the corpus appends; removing a document
retires its entry and permanently spends its ID. See `docs/decisions.md`, 2026-08-25.
