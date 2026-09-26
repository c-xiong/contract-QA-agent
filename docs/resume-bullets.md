# Contract Review Agent — evidence-backed résumé wording

- Built a **LangGraph** contract-review agent with **schema-validated tools** for clause retrieval, definition lookup, and calendar-date arithmetic, served through **FastAPI with SSE streaming**.
- Built document-scoped **BM25 + dense retrieval** with contract query expansion and **cross-encoder reranking**, raising development **evidence recall@5 from 45.8% to 72.7%**; achieved **71.8% on 119 held-out CUAD queries across 23 contracts**.
- Enforced **document and snapshot scoping**, execution budgets, and mandatory citation checks; added bounded repair and regression coverage for unanswerable queries, conflicting evidence, and prompt injection.

## Interview notes

- Development number: 36 answerable questions, BM25 baseline versus selected arm;
  the previous hybrid/rerank pipeline scored 61.1%. It is not held-out.
- Held-out number: original, expert-annotated CUAD clause-extraction questions on
  documents absent from development suites. The original hybrid/rerank baseline
  scored 70.2%, so the additional independent-set gain is modest and uncertain.
- Evidence recall is a document-matched 60-character evidence-location proxy.
  Complete-span recall is stricter: 40.7% to 68.5% on development; 62.4% to 69.1%
  on independent validation. Neither metric is answer accuracy.
- The original 41-question regression suite includes five abstention cases. It is
  distinct from the 119-question validation set and from ordinary software tests.
- All comparisons retain five original chunks; no top-k increase or corpus/gold
  filtering was used to obtain the improvement. Failed window-index experiments
  and category regressions remain in the report.

See [results, limitations and reproducible artifacts](retrieval-improvement-results.md).
