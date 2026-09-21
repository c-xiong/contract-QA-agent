"""Adversarial execution checks using synthetic contracts and scripted models."""

# ruff: noqa: ASYNC240 -- tiny local fixture artifacts
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from app.agent.graph import build_graph
from app.agent.llm import ModelResponse
from app.agent.state import Budget, ResearchState, initial_state
from app.agent.tool_registry import Action, AuthorizationError, DateCalculation, ToolRegistry
from app.agent.v2 import PLANNER_SYSTEM
from app.api.stream import research_events, v2_research_events
from app.config import Settings
from app.evidence.citation_verifier import CitationVerifier
from app.ingestion.store import ChunkStore
from app.retrieval.bm25 import Bm25Retriever
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk
from tests.conftest import make_chunk, make_document, make_retrieved


class ScriptedModel:
    model_id = "scripted-v2-fixture"

    def __init__(self, actions: list[dict[str, Any]], answers: list[str] | None = None):
        self.actions = iter(actions)
        self.answers = iter(answers or ["The term is 30 days. [doc-900, p. 1, §1]"])
        self.observations: list[dict[str, Any]] = []

    async def complete(self, system: str, user: str) -> ModelResponse:
        if system == PLANNER_SYSTEM:
            prompt = json.loads(user)
            self.observations.append(prompt)
            action = next(self.actions)
            if action.get("rule_evidence_ids") == ["FIRST"]:
                action = action | {
                    "rule_evidence_ids": [prompt["evidence"][0]["ref"]["evidence_id"]]
                }
            text = json.dumps(action)
        else:
            text = next(self.answers)
        return ModelResponse(text, input_tokens=10, output_tokens=5, model_id=self.model_id)


def action(tool: str, **arguments: Any) -> dict[str, Any]:
    return {
        "tool": tool,
        "arguments": arguments,
        "reason": "ready" if tool == "draft" else "missing_evidence",
    }


@pytest.fixture
def registry() -> ToolRegistry:
    chunks = [
        make_chunk(
            "doc-900-c1",
            section_path=["1"],
            text="Notice must be sent 30 calendar days after the Start Date. Subject to Section 2.",
        ),
        make_chunk("doc-900-c2", section_path=["2"], text='"Start Date" means 2026-01-01.'),
        make_chunk(
            "doc-901-c1",
            document_id="doc-901",
            section_path=["1"],
            text="Private notice must be sent 60 days after the start.",
        ),
    ]
    chunks += [
        make_chunk("doc-901-c2", document_id="doc-901", text="zebra orchard"),
        make_chunk("doc-901-c3", document_id="doc-901", text="apple orange"),
    ]
    store = ChunkStore([make_document(), make_document("doc-901")], chunks)
    return ToolRegistry(store, Bm25Retriever(store), CitationVerifier(store))


async def run(
    registry: ToolRegistry,
    path: Path,
    actions: list[dict[str, Any]],
    *,
    budget: Budget | None = None,
    answers: list[str] | None = None,
) -> tuple[ResearchState, ScriptedModel]:
    model = ScriptedModel(actions, answers)
    graph = build_graph(
        registry.retriever, registry.verifier, model, registry.store, agent_mode="v2", runs_dir=path
    )
    state = initial_state("When is notice due?", allowed_document_ids=["doc-900"], budget=budget)
    result = await graph.ainvoke(state, config={"recursion_limit": 100})
    return result, model


def events(state: ResearchState) -> list[dict[str, Any]]:
    rows = [json.loads(s) for s in Path(state["trace_path"]).read_text().splitlines()]
    assert [r["sequence"] for r in rows] == list(range(1, len(rows) + 1))
    terminal = [
        r
        for r in rows
        if r["event_type"] in {"run_completed", "run_abstained", "run_failed", "run_cancelled"}
    ]
    assert len(terminal) == 1
    return rows


@pytest.mark.parametrize(
    "tool,args",
    [
        ("document_search", {"query": "notice", "doc_ids": ["doc-901"]}),
        ("retrieve_clause", {"doc_id": "doc-901", "version_id": "wrong", "clause_ref": "1"}),
        ("extract_definition", {"doc_id": "doc-900", "version_id": "wrong", "term": "Start Date"}),
    ],
)
def test_authorization_before_dispatch(
    registry: ToolRegistry, tool: str, args: dict[str, Any]
) -> None:
    with pytest.raises(AuthorizationError):
        registry.validate(
            Action.model_validate(action(tool, **args)), ["doc-900"], registry.versions, 5
        )


async def test_search_clamps_and_blocks_untrusted_results(registry: ToolRegistry) -> None:
    payload = registry.validate(
        Action.model_validate(action("document_search", query="notice", top_k=999)),
        ["doc-900"],
        registry.versions,
        3,
    )
    assert payload.model_dump()["top_k"] == 3

    class Leaky:
        name = "leaky"

        async def search(
            self, query: str, *, top_k: int, filters: RetrievalFilters | None = None
        ) -> list[RetrievedChunk]:
            return [make_retrieved(registry.store.chunks_for("doc-901")[0])]

    registry.retriever = Leaky()
    with pytest.raises(AuthorizationError):
        await registry.execute(
            payload,
            allowed=["doc-900"],
            versions=registry.versions,
            evidence={},
            retrieved=[],
            call_id="c",
        )


@pytest.mark.parametrize(
    "anchor,offset,direction,expected",
    [
        ("2024-02-28", 1, "after", "2024-02-29"),
        ("2024-03-01", 1, "before", "2024-02-29"),
        ("2026-12-31", 1, "after", "2027-01-01"),
        ("2026-01-01", 0, "after", "2026-01-01"),
    ],
)
async def test_calendar_arithmetic(
    registry: ToolRegistry, anchor: str, offset: int, direction: str, expected: str
) -> None:
    payload = DateCalculation.model_validate(
        dict(
            anchor_date=anchor,
            offset=offset,
            direction=direction,
            unit="days",
            convention="calendar_days",
        )
    )
    result = await registry.execute(
        payload, allowed=[], versions={}, evidence={}, retrieved=[], call_id="c"
    )
    assert result.data["date"] == expected
    assert not result.evidence_refs


@pytest.mark.parametrize("convention", ["business_days", "holidays"])
async def test_unsupported_calendar(registry: ToolRegistry, convention: str) -> None:
    payload = DateCalculation.model_validate(
        dict(
            anchor_date="2026-01-01",
            offset=1,
            direction="after",
            unit="days",
            convention=convention,
        )
    )
    result = await registry.execute(
        payload, allowed=[], versions={}, evidence={}, retrieved=[], call_id="c"
    )
    assert result.error and result.error.code == "unsupported_convention"


async def test_complete_and_trace(registry: ToolRegistry, tmp_path: Path) -> None:
    state, model = await run(
        registry, tmp_path, [action("document_search", query="notice"), action("draft")]
    )
    assert state["status"] == "completed"
    assert state["verification_ok"]
    assert state["model_calls"] == 3
    assert (state["input_tokens"], state["output_tokens"]) == (30, 15)
    assert model.observations[1]["evidence"]
    assert events(state)[-1]["event_type"] == "run_completed"
    assert Path(state["trace_path"]).stat().st_mode & 0o077 == 0


async def test_definition_path_and_missing_clause(registry: ToolRegistry, tmp_path: Path) -> None:
    version = registry.versions["doc-900"]
    state, _ = await run(
        registry,
        tmp_path,
        [
            action("retrieve_clause", doc_id="doc-900", version_id=version, clause_ref="999"),
            action("extract_definition", doc_id="doc-900", version_id=version, term="Start Date"),
            action("draft"),
        ],
        answers=["The Start Date is 2026-01-01. [doc-900, p. 1, §2]"],
    )
    assert state["status"] == "completed"
    results = list(state["tool_results_by_call_id"].values())
    assert results[0]["data"]["lookup_status"] == "not_found"
    assert results[1]["data"]["lookup_status"] == "ok"
    events(state)


async def test_derived_provenance(registry: ToolRegistry, tmp_path: Path) -> None:
    calculation = action(
        "calculate_date",
        anchor_date="2026-01-01",
        offset=30,
        unit="days",
        direction="after",
        convention="calendar_days",
    ) | {"rule_evidence_ids": ["FIRST"]}
    state, _ = await run(
        registry,
        tmp_path,
        [action("document_search", query="notice"), calculation, action("draft")],
    )
    assert state["calculations"][0]["date"] == "2026-01-31"
    assert set(state["calculations"][0]["rule_evidence_ids"]) <= state["evidence_by_id"].keys()
    assert all(not r.get("derived") for r in state["evidence_by_id"].values())
    events(state)


async def test_ungrounded_calculation_rejected(registry: ToolRegistry, tmp_path: Path) -> None:
    state, _ = await run(
        registry,
        tmp_path,
        [
            action(
                "calculate_date",
                anchor_date="2026-01-01",
                offset=30,
                unit="days",
                direction="after",
                convention="calendar_days",
            )
        ],
    )
    assert state["status"] == "failed" and state["tool_calls"] == 0
    events(state)


async def test_duplicate_and_correction_bounds(registry: ToolRegistry, tmp_path: Path) -> None:
    a = action("document_search", query="notice")
    state, _ = await run(registry, tmp_path, [a, a, a])
    assert state["stop_reason"] == "repeated_action_without_evidence"
    assert state["tool_calls"] == 2
    events(state)
    invalid = action("document_search", query="notice", unexpected=True)
    state, _ = await run(registry, tmp_path, [invalid, invalid])
    assert state["status"] == "failed" and state["tool_calls"] == 0
    assert state["argument_corrections"] == 2
    events(state)


@pytest.mark.parametrize(
    "budget",
    [
        Budget(max_tool_calls=1),
        Budget(max_model_calls=1),
        Budget(max_searches=0),
        Budget(deadline_seconds=0),
    ],
)
async def test_budget_exhaustion_never_completes(
    registry: ToolRegistry, tmp_path: Path, budget: Budget
) -> None:
    state, _ = await run(
        registry,
        tmp_path,
        [action("document_search", query="notice"), action("draft")],
        budget=budget,
    )
    assert state["status"] == "failed" and not state["verification_ok"]
    assert state["tool_calls"] <= budget.max_tool_calls
    assert state["model_calls"] <= budget.max_model_calls
    events(state)


async def test_strict_evidence_budget(registry: ToolRegistry, tmp_path: Path) -> None:
    state, _ = await run(
        registry,
        tmp_path,
        [action("document_search", query="notice"), action("draft")],
        budget=Budget(max_evidence_chars=1),
    )
    assert not state["evidence"] and state["status"] == "abstained"
    events(state)


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   ",
        "Unsupported answer without citations",
        "Wrong section [doc-900, p. 1, §2]",
        "Forbidden [doc-901, p. 1, §1]",
    ],
)
async def test_invalid_drafts_cannot_pass(
    registry: ToolRegistry, tmp_path: Path, answer: str
) -> None:
    state, _ = await run(
        registry,
        tmp_path,
        [action("document_search", query="notice", top_k=1), action("draft")],
        answers=[answer, answer],
    )
    assert state["status"] == "abstained"
    assert state["repair_attempts"] == 1
    assert not state["verification_ok"]
    events(state)


async def test_repair_and_reserved_verification(registry: ToolRegistry, tmp_path: Path) -> None:
    state, _ = await run(
        registry,
        tmp_path,
        [action("document_search", query="notice"), action("draft")],
        answers=["bad [doc-999, p. 1]", "notice [doc-900, p. 1, §1]"],
    )
    assert state["status"] == "completed" and state["repair_attempts"] == 1
    assert state["tool_calls"] == 3
    events(state)
    state, _ = await run(
        registry,
        tmp_path,
        [action("document_search", query="notice"), action("draft")],
        answers=["bad"],
        budget=Budget(max_tool_calls=2),
    )
    assert state["status"] == "failed" and state["tool_calls"] == 2
    assert state["model_calls"] == 3  # no repair model call when verification cannot fit
    events(state)


async def test_retry_attempts_count_as_searches(registry: ToolRegistry, tmp_path: Path) -> None:
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
    state, _ = await run(
        registry, tmp_path, [action("document_search", query="notice"), action("draft")]
    )
    assert state["status"] == "completed"
    assert state["searches_used"] == 2 and state["retries"] == 1 and state["tool_calls"] == 3
    assert any(r["event_type"] == "tool_failed" for r in events(state))


async def test_requested_version_rejected(registry: ToolRegistry, tmp_path: Path) -> None:
    model = ScriptedModel([])
    graph = build_graph(
        registry.retriever,
        registry.verifier,
        model,
        registry.store,
        agent_mode="v2",
        runs_dir=tmp_path,
    )
    state = initial_state("notice", allowed_document_ids=["doc-900"])
    state["requested_versions"] = {"doc-900": "invented"}
    final = await graph.ainvoke(state)
    assert final["status"] == "failed" and final["model_calls"] == 0
    events(final)


async def test_sse_disconnect_has_one_terminal(registry: ToolRegistry, tmp_path: Path) -> None:
    from app.agent.runner import ResearchAgent

    class Slow(ScriptedModel):
        async def complete(self, system: str, user: str) -> ModelResponse:
            await asyncio.Event().wait()
            return ModelResponse("unused")

    agent = ResearchAgent(registry.store, Settings(agent_mode="v2", runs_dir=tmp_path))
    agent.graph = build_graph(
        registry.retriever,
        registry.verifier,
        Slow([]),
        registry.store,
        agent_mode="v2",
        runs_dir=tmp_path,
    )
    stream = research_events(agent, "notice", allowed_document_ids=["doc-900"])
    frame = await anext(stream)
    assert frame.startswith("event: start")
    # A model that never finishes must still publish model_started in real time.
    async with asyncio.timeout(3):
        while True:
            frame = await anext(stream)
            assert "id:" in frame and "Notice must" not in frame
            if '"event_type": "model_started"' in frame:
                break
    await stream.aclose()
    path = next(tmp_path.glob("*/events.jsonl"))
    rows = [json.loads(r) for r in path.read_text().splitlines()]
    assert rows[-1]["event_type"] == "run_cancelled"
    assert len([r for r in rows if r["event_type"] == "run_cancelled"]) == 1


async def test_cancellation_during_model(registry: ToolRegistry, tmp_path: Path) -> None:
    class Slow(ScriptedModel):
        async def complete(self, system: str, user: str) -> ModelResponse:
            await asyncio.sleep(5)
            return ModelResponse("unused")

    graph = build_graph(
        registry.retriever,
        registry.verifier,
        Slow([]),
        registry.store,
        agent_mode="v2",
        runs_dir=tmp_path,
    )
    task = asyncio.create_task(graph.ainvoke(initial_state("notice")))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    rows = [json.loads(r) for r in next(tmp_path.glob("*/events.jsonl")).read_text().splitlines()]
    assert rows[-1]["event_type"] == "run_cancelled"


async def test_deadline_interrupts_readonly_native_work(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    import time

    class Blocking:
        name = "blocking"

        async def search(
            self, query: str, *, top_k: int, filters: RetrievalFilters | None = None
        ) -> list[RetrievedChunk]:
            time.sleep(0.15)  # noqa: ASYNC251 -- deliberately simulate native blocking retrieval
            return []

    registry.retriever = Blocking()
    state, _ = await run(
        registry,
        tmp_path,
        [action("document_search", query="notice")],
        budget=Budget(deadline_seconds=0.05),
    )
    assert state["status"] == "failed"
    assert state["tool_calls"] == 1 and state["searches_used"] == 1
    assert not state["verification_ok"]
    events(state)


async def test_empty_search_reformulation_limit(registry: ToolRegistry, tmp_path: Path) -> None:
    state, _ = await run(
        registry, tmp_path, [action("document_search", query=f"missing{i}") for i in range(3)]
    )
    assert state["status"] == "abstained" and state["searches_used"] == 2
    events(state)


async def test_tool_failure_never_counts_as_abstention(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    class Broken:
        name = "broken"

        async def search(
            self, query: str, *, top_k: int, filters: RetrievalFilters | None = None
        ) -> list[RetrievedChunk]:
            raise RuntimeError("provider payload must not leak")

    registry.retriever = Broken()
    state, _ = await run(registry, tmp_path, [action("document_search", query="notice")])
    assert state["status"] == "failed"
    assert "provider payload" not in Path(state["trace_path"]).read_text()
    events(state)


async def test_public_verifier_does_not_bypass_final_gate(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    version = registry.versions["doc-900"]
    ref = registry.ref(registry.store.chunks_for("doc-900")[0])
    state, _ = await run(
        registry,
        tmp_path,
        [
            action("retrieve_clause", doc_id="doc-900", version_id=version, clause_ref="1"),
            action(
                "verify_citation",
                answer="notice [doc-900, p. 1, §1]",
                evidence_ids=[ref.evidence_id],
            ),
            action("draft"),
        ],
        answers=["wrong [doc-901, p. 1, §1]", "wrong again [doc-901, p. 1, §1]"],
    )
    assert state["status"] == "abstained"
    assert not state["verification_ok"]
    assert len([e for e in events(state) if e["event_type"] == "verification_completed"]) == 2


async def test_injection_is_data_and_scope_cannot_expand(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    state, _ = await run(
        registry,
        tmp_path,
        [
            action("document_search", query="notice"),
            action("document_search", query="private", doc_ids=["doc-901"]),
        ],
    )
    assert state["status"] == "failed"
    assert all(e.document_id == "doc-900" for e in state["evidence"])
    assert state["searches_used"] == 1
    events(state)


def test_recovery_marks_dead_process_failed(registry: ToolRegistry, tmp_path: Path) -> None:
    from app.observability.events import emit, recover_interrupted

    directory = tmp_path / "dead"
    directory.mkdir()
    path = directory / "events.jsonl"
    (directory / "manifest.json").write_text(json.dumps({"pid": 99999999}))
    emit(
        {"run_id": "dead", "trace_path": str(path), "status": "researching"},
        "run_started",
        "initialize",
    )
    assert recover_interrupted(tmp_path) == 1
    assert recover_interrupted(tmp_path) == 0
    assert json.loads(path.read_text().splitlines()[-1])["event_type"] == "run_failed"


async def test_ambiguous_definitions_and_wrong_versions(registry: ToolRegistry) -> None:
    d = make_document()
    chunks = [
        make_chunk(f"doc-900-c{i}", section_path=[str(i)], text=f'"Term" means value {i}.')
        for i in range(1, 3)
    ]
    store = ChunkStore([d], chunks)
    reg = ToolRegistry(store, Bm25Retriever(store), CitationVerifier(store))
    payload = reg.validate(
        Action.model_validate(
            action(
                "extract_definition",
                doc_id="doc-900",
                version_id=reg.versions["doc-900"],
                term="Term",
            )
        ),
        ["doc-900"],
        reg.versions,
        5,
    )
    result = await reg.execute(
        payload, allowed=["doc-900"], versions=reg.versions, evidence={}, retrieved=[], call_id="c"
    )
    assert result.data["lookup_status"] == "ambiguous" and len(result.evidence_refs) == 2


async def test_sse_complete_redacts_progress(registry: ToolRegistry, tmp_path: Path) -> None:
    from app.agent.runner import ResearchAgent

    agent = ResearchAgent(registry.store, Settings(agent_mode="v2", runs_dir=tmp_path))
    frames = [
        f async for f in v2_research_events(agent, "notice", allowed_document_ids=["doc-900"])
    ]
    assert frames[-1].startswith("event: done")
    assert all("Notice must" not in frame and '"state"' not in frame for frame in frames[:-1])
    path = next(tmp_path.glob("*/events.jsonl"))
    rows = [json.loads(r) for r in path.read_text().splitlines()]
    assert (
        len(
            [
                r
                for r in rows
                if r["event_type"].startswith("run_") and r["event_type"] != "run_started"
            ]
        )
        == 1
    )


def test_public_events_expose_metadata_only() -> None:
    from app.observability.events import public

    event = {
        "run_id": "run",
        "sequence": 3,
        "event_type": "tool_completed",
        "node": "execute",
        "data": {"call_id": "call", "arguments": {"query": "PRIVATE"}},
        "state": {
            "budget": {"max_tool_calls": 12, "extra": "PRIVATE"},
            "tool_results_by_call_id": {
                "call": {
                    "tool": "document_search",
                    "status": "ok",
                    "elapsed_ms": 8,
                    "data": {"source_text": "PRIVATE"},
                }
            },
            "evidence_by_id": {
                "ev": {
                    "evidence_id": "ev",
                    "doc_id": "doc-900",
                    "version_id": "v1",
                    "chunk_id": "c1",
                    "page": 1,
                    "section": "2",
                    "text": "PRIVATE",
                }
            },
            "usage_known": False,
            "input_tokens": 100,
            "output_tokens": 50,
        },
    }
    projected = public(event)
    assert "PRIVATE" not in json.dumps(projected)
    assert projected["tool"] == "document_search" and projected["tool_status"] == "ok"
    assert projected["elapsed_ms"] == 8 and projected["limits"]["max_tool_calls"] == 12
    assert projected["input_tokens"] is None and projected["output_tokens"] is None
    assert projected["evidence_refs"][0]["chunk_id"] == "c1"


async def test_runner_uses_persistent_execution_boundary(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    from app.agent.runner import ResearchAgent

    agent = ResearchAgent(registry.store, Settings(agent_mode="v2", runs_dir=tmp_path))
    result = await agent.research("notice", allowed_document_ids=["doc-900"])
    assert result.status == "completed" and result.run_id and result.trace_path
    rows = [json.loads(line) for line in Path(result.trace_path).read_text().splitlines()]
    assert rows[-1]["event_type"] == "run_completed"
    assert result.tool_calls >= 2
