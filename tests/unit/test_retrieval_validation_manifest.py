"""The frozen source-validation suite remains intact and development-disjoint."""

import hashlib
import json
from pathlib import Path

from evals.loader import load_suite


def test_validation_manifest_matches_tasks_and_excludes_development_documents():
    root = Path("evals/datasets/cuad_holdout")
    manifest = json.loads((root / "manifest.json").read_text())
    assert (
        hashlib.sha256((root / "tasks.jsonl").read_bytes()).hexdigest() == manifest["tasks_sha256"]
    )
    tasks = load_suite("cuad_holdout")
    assert len(tasks) == manifest["task_count"] == 119
    docs = {doc for task in tasks for doc in task.expected_document_ids}
    assert docs == set(manifest["document_ids"])
    assert len(docs) == 23
    assert not docs.intersection(manifest["excluded_document_ids"])
    assert all(task.review_status == "source-annotated" for task in tasks)
    assert all(
        task.expected_evidence and all(e.source == "cuad" for e in task.expected_evidence)
        for task in tasks
    )
    for task in tasks:
        source = manifest["source_annotations"][task.task_id]
        assert len(source["answer_offsets"]) == len(task.expected_evidence)
        assert task.question.startswith("Highlight the parts (if any) of this contract")
