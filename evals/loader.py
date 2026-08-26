"""Load and validate eval datasets from JSONL.

Validation is strict and loud. A malformed task must stop the run, not be skipped:
silently dropping a task changes the denominator of every reported metric, and a
metric whose n moved for an unrecorded reason is not reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from evals.schema import EvalTask

TASKS_FILENAME = "tasks.jsonl"


class DatasetError(RuntimeError):
    """The dataset is missing, malformed, or internally inconsistent."""


def suite_path(suite: str, root: Path | None = None) -> Path:
    base = root or Path(__file__).parent / "datasets"
    return base / suite / TASKS_FILENAME


def load_suite(suite: str, root: Path | None = None) -> list[EvalTask]:
    """Load one eval suite by name."""
    path = suite_path(suite, root)
    if not path.exists():
        readme = path.parent.parent / "README.md"
        raise DatasetError(
            f"No eval suite at {path}.\n"
            f"The generated suite is 'full' (model-written questions, expert-annotated "
            f"evidence). The 'smoke' suite is reserved for hand-written tasks and is "
            f"empty by default.\n"
            f"See {readme} for the schema and the distinction between them."
        )
    return load_tasks(path)


def load_tasks(path: Path) -> list[EvalTask]:
    tasks = list(_parse(path))
    if not tasks:
        raise DatasetError(f"{path} contains no tasks.")

    seen: dict[str, int] = {}
    for index, task in enumerate(tasks, start=1):
        if task.task_id in seen:
            raise DatasetError(
                f"{path}: duplicate task_id {task.task_id!r} on lines {seen[task.task_id]} "
                f"and {index}."
            )
        seen[task.task_id] = index

    versions = {t.dataset_version for t in tasks}
    if len(versions) > 1:
        raise DatasetError(
            f"{path}: mixed dataset_version values {sorted(versions)}. A suite is versioned "
            "as a whole so results can be compared across runs."
        )
    return tasks


def _parse(path: Path) -> Iterator[EvalTask]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            # '#' comments are not JSONL, but a dataset file benefits from a header.
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{number}: invalid JSON: {exc}") from exc
            try:
                yield EvalTask.model_validate(payload)
            except ValidationError as exc:
                raise DatasetError(f"{path}:{number}: invalid task:\n{exc}") from exc
