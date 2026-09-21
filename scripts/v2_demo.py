"""Generate three inspectable synthetic V2 traces, without corpus downloads or API calls."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from app.agent.graph import build_graph
from app.agent.llm import ModelResponse
from app.agent.state import initial_state
from app.agent.tool_registry import ToolRegistry
from app.agent.v2 import PLANNER_SYSTEM
from app.evidence.citation_verifier import CitationVerifier
from app.ingestion.store import ChunkStore
from app.retrieval.bm25 import Bm25Retriever
from app.schemas.chunk import Chunk
from app.schemas.document import Document


class DemoClient:
    """Scripted actions depend on actual returned evidence IDs and result status."""

    model_id = "scripted-offline-demo"
    simulated = True

    def __init__(self, kind: str, version: str):
        self.kind, self.version = kind, version
        self.writes = 0

    async def complete(self, system: str, user: str) -> ModelResponse:
        if system != PLANNER_SYSTEM:
            self.writes += 1
            text = (
                "Invalid fixture citation [doc-999, p. 1]"
                if self.kind == "repaired" and self.writes == 1
                else "The notice period is 30 calendar days. [doc-900, p. 1, §1]"
            )
            if self.kind == "success":
                text = "The deadline is 2026-01-31, applying 30 calendar days after 2026-01-01. [doc-900, p. 1, §1]"
            return ModelResponse(text, 20, 10, self.model_id)
        observed = json.loads(user)
        results = list(observed["results"].values())
        tools = {r["tool"] for r in results}
        action: dict[str, Any] = {"tool": "draft", "arguments": {}, "reason": "ready"}
        if not results:
            action = {
                "tool": "retrieve_clause",
                "arguments": {
                    "doc_id": "doc-900",
                    "version_id": self.version,
                    "clause_ref": "999" if self.kind == "abstained" else "1",
                },
                "reason": "resolve_reference",
            }
        elif results[-1]["status"] == "empty":
            action = {"tool": "abstain", "reason": "insufficient_evidence"}
        elif self.kind == "success" and "calculate_date" not in tools:
            action = {
                "tool": "calculate_date",
                "arguments": {
                    "anchor_date": "2026-01-01",
                    "offset": 30,
                    "unit": "days",
                    "direction": "after",
                    "convention": "calendar_days",
                },
                "reason": "compute_deadline",
                "rule_evidence_ids": [observed["evidence"][0]["ref"]["evidence_id"]],
            }
        return ModelResponse(json.dumps(action), 20, 10, self.model_id)


def fixture_store() -> ChunkStore:
    document = Document(
        document_id="doc-900",
        title="Synthetic demo contract",
        source_path="synthetic-demo",
        page_count=1,
        corpus_source="synthetic",
        is_synthetic=True,
    )
    chunk = Chunk(
        chunk_id="doc-900-c1",
        document_id="doc-900",
        document_title=document.title,
        section_path=["1"],
        page_number=1,
        text="Notice must be delivered 30 calendar days after the supplied start date.",
        token_count=18,
    )
    return ChunkStore([document], [chunk])


def save_fixture(path: Path, destination: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["state"]["trace_path"] = f"runs/{row['run_id']}/events.jsonl"
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    destination.with_suffix(".manifest.json").write_text(
        path.with_name("manifest.json").read_text()
    )


async def run(output: Path, runs: Path) -> None:
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=True)
    store = fixture_store()
    registry = ToolRegistry(store, Bm25Retriever(store), CitationVerifier(store))
    for kind in ("success", "repaired", "abstained"):
        client = DemoClient(kind, registry.versions["doc-900"])
        graph = build_graph(
            registry.retriever, registry.verifier, client, store, agent_mode="v2", runs_dir=runs
        )
        question = (
            "Using calendar days, when is notice due if the start date is 2026-01-01?"
            if kind == "success"
            else "What is the notice rule?"
            if kind == "repaired"
            else "What does nonexistent Section 999 say?"
        )
        final = await graph.ainvoke(
            initial_state(question, allowed_document_ids=["doc-900"]),
            config={"recursion_limit": 100},
        )
        expected = "abstained" if kind == "abstained" else "completed"
        if final["status"] != expected:
            raise RuntimeError(f"{kind}: unexpected status {final['status']}")
        await asyncio.to_thread(save_fixture, Path(final["trace_path"]), output / f"{kind}.jsonl")
        print(
            f"{kind}: {final['status']}; tools={final['tool_calls']}; repairs={final['repair_attempts']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/traces"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    asyncio.run(run(args.output, args.runs))


if __name__ == "__main__":
    main()
