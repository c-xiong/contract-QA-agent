"""Transcribe untouched CUAD source questions; never generate questions or answers.

The selection rule is fixed independently of retrieval outcomes: all answerable
questions in the existing eight project categories, over active CUAD documents
not referenced by any pre-existing development suite. Source offsets must roundtrip.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from app.config import get_settings
from app.ingestion.cuad import CATEGORY_COLUMNS
from app.ingestion.manifest import CorpusManifest, manifest_path
from evals.loader import load_tasks
from evals.schema import EvalTask, ExpectedEvidence

VERSION = "cuad-source-questions-v1"
DESTINATION = Path("evals/datasets/cuad_holdout")
DEVELOPMENT_SUITES = ("hand", "full", "answerability", "v2")


def build() -> None:
    settings = get_settings()
    source = settings.raw_dir / "CUAD_v1" / "CUAD_v1.json"
    source_data = json.loads(source.read_text(encoding="utf-8"))
    originals = {entry["title"].casefold(): entry for entry in source_data["data"]}
    manifest = CorpusManifest.load(manifest_path(settings.data_dir))
    excluded: set[str] = set()
    exclusions: dict[str, str] = {}
    for suite in DEVELOPMENT_SUITES:
        root = Path("evals/datasets") / suite
        for path in sorted(root.rglob("tasks.jsonl")):
            exclusions[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for task in load_tasks(path):
                excluded.update(task.allowed_document_ids or [])
                excluded.update(task.expected_document_ids)

    tasks: list[EvalTask] = []
    provenance: dict[str, object] = {}
    documents: list[str] = []
    for doc_id, entry in sorted(manifest.active.items()):
        if entry.corpus_source != "cuad" or doc_id in excluded:
            continue
        documents.append(doc_id)
        original = originals[Path(entry.source_filename).stem.casefold()]
        for paragraph in original["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                category = qa["id"].rsplit("__", 1)[-1]
                if category not in CATEGORY_COLUMNS or not qa["answers"]:
                    continue
                task_id = f"cuad-native-{doc_id}-{list(CATEGORY_COLUMNS).index(category):02d}"
                evidence: list[ExpectedEvidence] = []
                offsets = []
                for answer in qa["answers"]:
                    start, text = answer["answer_start"], answer["text"]
                    if context[start : start + len(text)] != text:
                        raise ValueError(f"Source offset mismatch: {qa['id']}")
                    evidence.append(
                        ExpectedEvidence(document_id=doc_id, span_text=text, source="cuad")
                    )
                    offsets.append({"start": start, "end": start + len(text)})
                tasks.append(
                    EvalTask(
                        task_id=task_id,
                        category=category,
                        question=qa["question"],
                        allowed_document_ids=[doc_id],
                        expected_document_ids=[doc_id],
                        expected_evidence=evidence,
                        expected_behavior="answer",
                        dataset_version=VERSION,
                        review_status="source-annotated",
                        rubric_version="retrieval-only-source-spans-v1",
                    )
                )
                provenance[task_id] = {
                    "source_qa_id": qa["id"],
                    "title": original["title"],
                    "answer_offsets": offsets,
                }
    task_bytes = ("\n".join(task.model_dump_json() for task in tasks) + "\n").encode()
    payload = {
        "dataset_version": VERSION,
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tasks_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "task_count": len(tasks),
        "document_ids": documents,
        "category_counts": dict(sorted(Counter(t.category for t in tasks).items())),
        "excluded_document_ids": sorted(excluded),
        "development_suite_hashes": exclusions,
        "selection": "All answerable original CUAD questions in existing eight categories, on active documents absent from development suites; no filtering by retrieval outcomes or chunk matchability.",
        "question_authorship": "Verbatim original CUAD source questions; no model-generated questions, answers, or rubrics.",
        "limitations": [
            "Templated single-document clause extraction, not conversational or cross-document QA.",
            "Corpus previously selected for category coverage; not a random sample of contracts.",
            "Public model training contamination cannot be excluded.",
            "Not newly reviewed by the project author; source annotation provenance is retained.",
        ],
        "source_annotations": provenance,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    tasks_path = DESTINATION / "tasks.jsonl"
    if tasks_path.exists() and tasks_path.read_bytes() != task_bytes:
        raise ValueError(
            "Frozen validation differs; create a new explicit version rather than overwrite"
        )
    tasks_path.write_bytes(task_bytes)
    (DESTINATION / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Frozen {len(tasks)} source questions over {len(documents)} documents; sha256={payload['tasks_sha256']}"
    )


if __name__ == "__main__":
    build()
