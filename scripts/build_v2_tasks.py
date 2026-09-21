"""Generate 24 explicitly provisional cases over the unchanged existing corpus."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from app.agent.tool_registry import ToolRegistry
from app.config import Settings
from app.evidence.citation_verifier import CitationVerifier
from app.ingestion.store import ChunkStore
from app.retrieval.bm25 import Bm25Retriever
from evals.schema import EvalTask, ExpectedEvidence


def build(store: ChunkStore) -> list[EvalTask]:
    registry = ToolRegistry(store, Bm25Retriever(store), CitationVerifier(store))
    tasks: list[EvalTask] = []

    def add(
        category: str,
        question: str,
        docs: list[str],
        sections: list[tuple[str, str]],
        capabilities: list[str],
        *,
        expected_date: str | None = None,
        abstain: bool = False,
    ) -> None:
        evidence = [
            ExpectedEvidence(
                document_id=d,
                section_id=section,
                page_number=c.page_number,
                span_text=c.text,
                source="synthetic",
            )
            for d, section in sections
            for c in store.find_section(d, section)
        ]
        tasks.append(
            EvalTask(
                task_id=f"v2-{len(tasks) + 1:03d}",
                category=category,
                question=question,
                allowed_document_ids=docs,
                expected_document_ids=[] if abstain else sorted({e.document_id for e in evidence}),
                expected_evidence=[] if abstain else evidence,
                expected_behavior="abstain" if abstain else "answer",
                required_points=[
                    "PROVISIONAL generated rubric: ground each requested fact in its cited source; identify unresolved ambiguity."
                ]
                if not abstain
                else [],
                forbidden_claims=[
                    "PROVISIONAL: unsupported legal version precedence or invented holiday calendars"
                ],
                dataset_version="provisional-generated-v2-1",
                rubric_version="provisional-generated-v2-rubric-1",
                review_status="provisional",
                touches_synthetic=True,
                requested_versions={d: registry.versions[d] for d in docs},
                required_capabilities=capabilities,
                capability_order=[("document_search", "calculate_date")]
                if "calculate_date" in capabilities
                else [],
                expected_date=expected_date,
            )
        )

    for term, section in [
        ("Accounting Standards", "1.1"),
        ("Ancillary Agreement", "1.6"),
        ("Business Day", "1.9"),
        ("Calendar Quarter", "1.10"),
    ]:
        add(
            "simple_lookup",
            f"In doc-051, what is the definition of {term}?",
            ["doc-051"],
            [("doc-051", section)],
            ["document_search"],
        )
    for term, refs in [
        ("Acquiror Family", ["1.3", "1.2", "1.12"]),
        ("Acquired Party Family", ["1.2", "1.12", "1.5"]),
        ("Affiliate", ["1.5"]),
        ("Change of Control", ["1.12"]),
        ("Calendar Quarter", ["1.10"]),
        ("Calendar Year", ["1.11"]),
    ]:
        add(
            "definition_chain",
            f"Explain {term} in doc-051, resolving the defined terms and qualifications it refers to. Identify anything not established by the retrieved clauses.",
            ["doc-051"],
            [("doc-051", r) for r in refs],
            ["document_search", "extract_definition"],
        )
    for section in ["1.1", "1.5", "1.9", "1.10", "1.11", "1.12"]:
        add(
            "version_comparison",
            f"Compare Section {section} in the pinned snapshots doc-051 and doc-055. Cite both, report whether the wording differs, and do not infer which legally prevails.",
            ["doc-051", "doc-055"],
            [(d, section) for d in ["doc-051", "doc-055"]],
            ["retrieve_clause"],
        )
    for anchor in ["2026-01-01", "2024-02-01", "2026-12-01", "2024-12-31"]:
        expected = (date.fromisoformat(anchor) + timedelta(days=60)).isoformat()
        add(
            "notice_deadline",
            f"For the sixty-day petition-dismissal period in doc-056 Section 10.4(c), assume the petition was filed on {anchor}. Using calendar days, excluding the filing day and no holiday adjustment, what is the last date of that period? Cite the rule. This is an arithmetic scenario, not a legal conclusion.",
            ["doc-056"],
            [("doc-056", "10.4")],
            ["document_search", "calculate_date"],
            expected_date=expected,
        )
    for question in [
        "Under doc-056 Section 12.2, give the exact notice date three business days after 2026-12-24, accounting for all applicable holidays.",
        "Calculate the expiry of the petition-dismissal period in doc-056 Section 10.4 without a supplied filing date.",
        "Which of doc-051 and doc-055 legally supersedes the other? Use only the supplied snapshots, without assuming version order from document IDs.",
        "Retrieve Section 9999 of doc-051 and state the monetary penalty it imposes.",
    ]:
        docs = (
            ["doc-051", "doc-055"]
            if "doc-055" in question
            else ["doc-051"]
            if "doc-051" in question
            else ["doc-056"]
        )
        add("unsupported_or_ambiguous", question, docs, [], [], abstain=True)
    return tasks


def main() -> None:
    store = ChunkStore.load(Settings().processed_dir)
    tasks = build(store)
    out = Path("evals/datasets/v2")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(t.model_dump_json() + "\n" for t in tasks))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "version": tasks[0].dataset_version,
                "count": len(tasks),
                "review_status": "provisional",
                "purpose": "pipeline validation only",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(tasks)} provisional tasks; no corpus files changed.")


if __name__ == "__main__":
    main()
