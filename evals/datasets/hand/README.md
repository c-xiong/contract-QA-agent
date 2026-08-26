# Hand-authored eval suite

The repository author has reviewed every question, manually selected evidence span,
required point, forbidden claim, expected behavior, and coverage note in this suite.
The frozen dataset version is `hand-v1`.

Results from `hand-v1` may be used for reporting when they come from a reproducible run
and follow the repository's reporting rules, including task count, per-category results,
paired comparisons, inspected failures, and a sample-size limitation.

Any later change to a question, rubric, expected behavior, or evidence label must create
a new dataset version so old and new results are not mixed.

Useful commands:

```bash
uv run python scripts/author_tasks.py --review
uv run python scripts/author_tasks.py --validate
uv run python scripts/author_tasks.py --redo lookup-002
```
