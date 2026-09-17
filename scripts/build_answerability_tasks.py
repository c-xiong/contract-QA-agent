"""Create a labelled, provisional answerability diagnostic from ContractNLI labels.

The hypothesis and label are copied from the expert dataset. The generic question
wrapper is AI-authored, so this is NOT an additional human-authored reporting suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.config import get_settings
from app.ingestion.contractnli import load_corpus
from app.ingestion.manifest import CorpusManifest, manifest_path
from evals.schema import EvalTask, ExpectedEvidence

VERSION = "provisional-generated-answerability-v1"


def build(out: Path) -> None:
    settings = get_settings()
    hypotheses, source_documents = load_corpus(settings.raw_dir / "contract-nli")
    source_by_id = {f"contractnli:{doc.doc_id}": doc for doc in source_documents}
    manifest = CorpusManifest.load(manifest_path(settings.data_dir))
    tasks: list[EvalTask] = []
    sources: dict[str, object] = {}
    for entry in sorted(manifest.active.values(), key=lambda item: item.document_id):
        if entry.corpus_source != "contractnli":
            continue
        document = source_by_id[entry.source_filename]
        for choice in ("NotMentioned", "Entailment"):
            candidates = sorted(
                (key, annotation)
                for key, annotation in document.annotations.items()
                if annotation.choice == choice
                and (choice == "NotMentioned" or annotation.has_evidence)
            )
            if not candidates:
                continue
            key, annotation = candidates[0]
            hypothesis = hypotheses[key].hypothesis
            task_id = f"answerability-{entry.document_id}-{choice.lower()}"
            negative = choice == "NotMentioned"
            tasks.append(
                EvalTask(
                    task_id=task_id,
                    category="missing_obligation" if negative else "supported_obligation",
                    question=f"Does this contract establish the following statement? {hypothesis}",
                    allowed_document_ids=[entry.document_id],
                    expected_document_ids=[] if negative else [entry.document_id],
                    expected_evidence=[]
                    if negative
                    else [
                        ExpectedEvidence(
                            document_id=entry.document_id, span_text=text, source="contractnli"
                        )
                        for text in annotation.span_texts
                    ],
                    expected_behavior="abstain" if negative else "answer",
                    dataset_version=VERSION,
                )
            )
            sources[task_id] = {
                "source_document_id": document.doc_id,
                "hypothesis_key": key,
                "hypothesis": hypothesis,
                "label": choice,
                "span_indices": annotation.span_indices,
                "document_sha256": hashlib.sha256(document.text.encode()).hexdigest(),
            }
    out.mkdir(parents=True, exist_ok=True)
    content = "".join(task.model_dump_json() + "\n" for task in tasks)
    (out / "tasks.jsonl").write_text(content)
    (out / "source-labels.json").write_text(
        json.dumps(
            {
                "dataset_version": VERSION,
                "selection": "first sorted labelled hypothesis per document and class",
                "tasks_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "sources": sources,
            },
            indent=2,
        )
        + "\n"
    )
    (out / "README.md").write_text(
        "# Answerability diagnostic (provisional)\n\n"
        f"{len(tasks)} tasks, `{VERSION}`. Expert ContractNLI hypotheses, labels and span indices "
        "are transcribed mechanically. The generic question wrapper is AI-authored. "
        "These are supplementary pipeline checks, not human-authored capability or resume evidence.\n\n"
        "Selection is frozen before running the gate: one sorted NotMentioned and one Entailment "
        "hypothesis per registered NDA. Documents belong to the existing corpus; this is not "
        "a held-out document benchmark. Empty required_points are not an answer-quality measure. "
        "Report behavior counts and independent citation support. Source mappings and hashes "
        "are in source-labels.json. Regenerate with scripts/build_answerability_tasks.py.\n"
    )
    print(f"Wrote {len(tasks)} provisional tasks to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evals/datasets/answerability"))
    build(parser.parse_args().out)
