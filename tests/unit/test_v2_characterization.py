"""Characterization of V2 graph traces, recorded before the shared-executor refactor.

Each scenario's durable events are normalized (random call IDs, hashes, timings and the
provenance manifest removed) and compared with a checked-in golden file. The golden
fixes event names, order, nodes, per-event counters, status and admitted evidence, so a
refactor of tool execution cannot silently change graph semantics.

Regenerate only for a deliberate, reviewed behavior change:
    UPDATE_V2_GOLDEN=1 CRA_LIVE_MODEL=0 uv run python -m pytest tests/unit/test_v2_characterization.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from app.agent.state import Budget
from app.agent.tool_registry import ToolRegistry
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk
from tests.unit.test_v2 import action, registry, run

__all__ = ["registry"]  # re-exported pytest fixture

GOLDEN = Path(__file__).parent / "golden" / "v2_trace_characterization.json"
COUNTERS = (
    "status",
    "stop_reason",
    "failure",
    "tool_calls",
    "model_calls",
    "searches_used",
    "retries",
    "repair_attempts",
    "argument_corrections",
    "verification_ok",
)


def normalize(trace_path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(trace_path).read_text().splitlines()]
    call_ids: dict[str, str] = {}
    normalized = []
    for row in rows:
        data = dict(row["data"])
        data.pop("manifest", None)
        data.pop("elapsed_ms", None)
        for key in ("result_hash",):
            if key in data:
                data[key] = "<hash>"
        if "call_id" in data:
            data["call_id"] = call_ids.setdefault(data["call_id"], f"call-{len(call_ids) + 1}")
        state = row["state"]
        normalized.append(
            {
                "event_type": row["event_type"],
                "node": row["node"],
                "data": data,
                "state": {k: state.get(k) for k in COUNTERS}
                | {
                    "evidence_ids": sorted(state.get("evidence_by_id", {})),
                    "tool_results": [
                        {
                            "tool": r.get("tool"),
                            "status": r.get("status"),
                            "attempt": r.get("attempt"),
                            "error_code": (r.get("error") or {}).get("code"),
                        }
                        for r in state.get("tool_results_by_call_id", {}).values()
                    ],
                    "calculations": state.get("calculations"),
                },
            }
        )
    return normalized


def flaky(registry: ToolRegistry) -> None:
    base = registry.retriever

    class Flaky:
        name = "flaky"
        calls = 0

        async def search(
            self, query: str, *, top_k: int, filters: RetrievalFilters | None = None
        ) -> list[RetrievedChunk]:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient")
            return await base.search(query, top_k=top_k, filters=filters)

    registry.retriever = Flaky()


def blocking(registry: ToolRegistry) -> None:
    class Blocking:
        name = "blocking"

        async def search(
            self, query: str, *, top_k: int, filters: RetrievalFilters | None = None
        ) -> list[RetrievedChunk]:
            time.sleep(0.15)  # noqa: ASYNC251 -- deliberately simulate native blocking retrieval
            return []

    registry.retriever = Blocking()


def leaky(registry: ToolRegistry) -> None:
    leaked = registry.store.chunks_for("doc-901")[0]

    class Leaky:
        name = "leaky"

        async def search(
            self, query: str, *, top_k: int, filters: RetrievalFilters | None = None
        ) -> list[RetrievedChunk]:
            return [RetrievedChunk(chunk=leaked, fused_score=0, retrieval_query=query)]

    registry.retriever = Leaky()


def broken(registry: ToolRegistry) -> None:
    class Broken:
        name = "broken"

        async def search(
            self, query: str, *, top_k: int, filters: RetrievalFilters | None = None
        ) -> list[RetrievedChunk]:
            raise RuntimeError("provider payload must not leak")

    registry.retriever = Broken()


CALCULATION = action(
    "calculate_date",
    anchor_date="2026-01-01",
    offset=30,
    unit="days",
    direction="after",
    convention="calendar_days",
)


def scenarios(registry: ToolRegistry) -> dict[str, dict[str, Any]]:
    version = registry.versions["doc-900"]
    first_ref = registry.ref(registry.store.chunks_for("doc-900")[0]).evidence_id
    return {
        "complete": {"actions": [action("document_search", query="notice"), action("draft")]},
        "definition_and_missing_clause": {
            "actions": [
                action("retrieve_clause", doc_id="doc-900", version_id=version, clause_ref="999"),
                action(
                    "extract_definition", doc_id="doc-900", version_id=version, term="Start Date"
                ),
                action("draft"),
            ],
            "answers": ["The Start Date is 2026-01-01. [doc-900, p. 1, §2]"],
        },
        "derived_date": {
            "actions": [
                action("document_search", query="notice"),
                CALCULATION | {"rule_evidence_ids": ["FIRST"]},
                action("draft"),
            ]
        },
        "ungrounded_date": {"actions": [CALCULATION]},
        "duplicate_action": {"actions": [action("document_search", query="notice")] * 3},
        "invalid_arguments": {
            "actions": [action("document_search", query="notice", unexpected=True)] * 2
        },
        "tool_budget_one": {
            "actions": [action("document_search", query="notice"), action("draft")],
            "budget": Budget(max_tool_calls=1),
        },
        "search_budget_zero": {
            "actions": [action("document_search", query="notice"), action("draft")],
            "budget": Budget(max_searches=0),
        },
        "evidence_chars_one": {
            "actions": [action("document_search", query="notice"), action("draft")],
            "budget": Budget(max_evidence_chars=1),
        },
        "evidence_items_one": {
            "actions": [action("document_search", query="notice"), action("draft")],
            "budget": Budget(max_evidence_items=1),
        },
        "repair_then_verified": {
            "actions": [action("document_search", query="notice"), action("draft")],
            "answers": ["bad [doc-999, p. 1]", "notice [doc-900, p. 1, §1]"],
        },
        "verification_reserve": {
            "actions": [action("document_search", query="notice"), action("draft")],
            "answers": ["bad"],
            "budget": Budget(max_tool_calls=2),
        },
        "retry_after_transient": {
            "actions": [action("document_search", query="notice"), action("draft")],
            "setup": flaky,
        },
        "deadline_blocking_search": {
            "actions": [action("document_search", query="notice")],
            "budget": Budget(deadline_seconds=0.05),
            "setup": blocking,
        },
        "retriever_scope_leak": {
            "actions": [action("document_search", query="notice")],
            "setup": leaky,
        },
        "tool_unavailable": {
            "actions": [action("document_search", query="notice")],
            "setup": broken,
        },
        "empty_search_limit": {
            "actions": [action("document_search", query=f"missing{i}") for i in range(3)]
        },
        "public_verifier_then_gate": {
            "actions": [
                action("retrieve_clause", doc_id="doc-900", version_id=version, clause_ref="1"),
                action(
                    "verify_citation",
                    answer="notice [doc-900, p. 1, §1]",
                    evidence_ids=[first_ref],
                ),
                action("draft"),
            ],
            "answers": ["wrong [doc-901, p. 1, §1]", "wrong again [doc-901, p. 1, §1]"],
        },
        "scope_widening": {
            "actions": [
                action("document_search", query="notice"),
                action("document_search", query="private", doc_ids=["doc-901"]),
            ]
        },
    }


async def test_v2_traces_match_characterization(registry: ToolRegistry, tmp_path: Path) -> None:
    observed: dict[str, Any] = {}
    names = list(scenarios(registry))
    for name in names:
        # A fresh registry fixture state per scenario: setup hooks replace the retriever.
        original = registry.retriever
        spec = scenarios(registry)[name]
        if "setup" in spec:
            spec["setup"](registry)
        try:
            state, _ = await run(
                registry,
                tmp_path / name,
                spec["actions"],
                budget=spec.get("budget"),
                answers=spec.get("answers"),
            )
        finally:
            registry.retriever = original
        observed[name] = normalize(state["trace_path"])
    if os.environ.get("UPDATE_V2_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(observed, indent=1, ensure_ascii=False) + "\n")
        pytest.skip("golden regenerated")
    expected = json.loads(GOLDEN.read_text())
    assert sorted(observed) == sorted(expected)
    for name in names:
        assert observed[name] == expected[name], name
