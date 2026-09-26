# CUAD original-question validation

`tasks.jsonl` transcribes original CUAD questions and all their expert-annotated
answers, with exact source-offset roundtrips. No question, answer, or rubric was
generated for this suite. `source-annotated` does not claim a new author review.

The selection rule was fixed before evaluating improved retrievers: use active
CUAD documents absent from every existing development suite, retain all answerable
questions in the project's existing eight categories, and never filter by whether
an answer can be retrieved or matched to a chunk. The manifest contains hashes,
excluded documents, and original source QA IDs and offsets.

This is document-disjoint validation of **templated clause extraction**, not a
held-out conversational, multi-contract, or end-to-end answer-quality benchmark.
The corpus was previously selected for category coverage. All model choices must
be frozen on `hand-v1` before inspecting these results. Public training-data
contamination is unknown. The existing 56-document corpus is indexed as normal;
the holdout concerns task/document use in optimization, not index exclusion.

Reproduce with `CRA_LIVE_MODEL=0 PYTHONPATH=. python scripts/build_retrieval_validation.py`.
