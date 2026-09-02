# `hand` — the reporting suite

41 tasks, dataset version `hand-v1`. Every question here is hand-written, and every
evidence span, required point, forbidden claim, expected behavior, and coverage note has
been reviewed by hand against the source document. This is the suite the numbers in the
top-level README come from.

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
