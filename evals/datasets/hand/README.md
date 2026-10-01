# `hand` — the reporting suite

41 development regression tasks, dataset version `hand-v1`. Every question here is hand-written, and every
evidence span, required point, forbidden claim, expected behavior, and coverage note has
been reviewed by hand against the source document. Historical A–D experiments and
the current standalone V2 evaluation workflow use this suite. It was used during
development and is not a held-out benchmark.

**41 = 36 cases with expected evidence + 5 expected abstentions.** Retrieval recall
uses the same suite but scores only the 36 cases with something to retrieve;
answer/abstention evaluation includes all 41. The 119-question CUAD retrieval
validation in the top-level README is a [separate suite](../cuad_holdout/README.md).
See the [dataset index](../README.md) for every suite's role and limitations.

Evidence labels are transcribed from expert annotation — CUAD clause spans, ContractNLI
hypothesis labels, and synthetic edits known by construction — rather than inferred.

Changing a question, rubric, expected behavior, or evidence label requires a new
`dataset_version`, so that results measured on different ground truth are never averaged
together.

```bash
uv run python scripts/author_tasks.py --review
uv run python scripts/author_tasks.py --validate
uv run python scripts/author_tasks.py --redo lookup-002
```
