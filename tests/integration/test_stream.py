"""The SSE endpoint behind the trace inspector.

The inspector's node highlighting is only trustworthy if these events correspond to the
backend actually entering each node, so the ordering and the terminal payload are pinned
here rather than eyeballed in the browser.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.runner import ResearchAgent
from app.api.main import create_app
from app.api.stream import GRAPH_NODES, extract_metrics, sse
from app.config import Settings
from app.ingestion.store import ChunkStore


@pytest.fixture
def app(store: ChunkStore, settings: Settings) -> FastAPI:
    # bm25 so the fixture store is not measured against the on-disk FAISS index,
    # which is built for the real corpus.
    application = create_app(retrieval_arm="bm25", settings=settings)
    application.state.agent = ResearchAgent(store, settings)
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def collect(client: TestClient, **params: str) -> list[tuple[str, dict]]:
    """Read a full SSE stream into (event, payload) pairs."""
    events: list[tuple[str, dict]] = []
    with client.stream("GET", "/research/stream", params=params) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        name = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: ") and name:
                events.append((name, json.loads(line[6:])))
    return events


class TestFraming:
    def test_frame_is_well_formed(self) -> None:
        frame = sse("node", {"node": "search"})
        assert frame.startswith("event: node\ndata: ")
        assert frame.endswith("\n\n")

    def test_payload_never_contains_a_raw_newline(self) -> None:
        """A newline inside `data:` would split one frame into two and desync the page."""
        frame = sse("done", {"answer": "line one\nline two"})
        body = frame.split("data: ", 1)[1].rstrip("\n")
        assert "\n" not in body
        assert json.loads(body)["answer"] == "line one\nline two"


class TestStream:
    def test_v2_selection_streams_tools_and_limits(
        self, app: FastAPI, client: TestClient, tmp_path: Path
    ) -> None:
        app.state.agent.settings = app.state.agent.settings.model_copy(
            update={"runs_dir": tmp_path}
        )
        events = collect(client, question="What is the governing law?", mode="v2")
        start, done = events[0][1], events[-1][1]
        assert events[0][0] == "start" and events[-1][0] == "done"
        assert start["agent_mode"] == "v2" and start["live_model"] is False
        assert app.state.agent.settings.agent_mode == "v1"
        progress = [p for name, p in events if name == "progress"]
        assert progress and not any(name in ("node", "error") for name, _ in events)
        assert any(p["tool"] == "document_search" for p in progress)
        assert any(p["tool_status"] == "ok" for p in progress)
        assert all(p["limits"]["max_tool_calls"] == 12 for p in progress)
        assert all("state" not in p and "arguments" not in p for p in progress)
        assert {p["node"] for p in progress} <= set(start["nodes"])
        assert done["total_ms"] >= 0 and done["steps"] == len(progress)
        assert done["run_id"] and done["usage"]["tool_calls"] > 0
        assert done["usage"]["max_model_calls"] == 12
        expected_path = []
        for event in progress:
            if not expected_path or expected_path[-1] != event["node"]:
                expected_path.append(event["node"])
        assert done["execution_path"] == expected_path
        assert expected_path[0] == "initialize" and expected_path[-1] == "terminal"
        assert client.get("/health").json()["agent_mode"] == "v1"

    def test_rejects_unknown_mode(self, client: TestClient) -> None:
        response = client.get("/research/stream", params={"question": "law", "mode": "v3"})
        assert response.status_code == 422

    def test_emits_start_then_nodes_then_done(self, client: TestClient) -> None:
        events = collect(client, question="What is the governing law?")
        names = [n for n, _ in events]
        assert names[0] == "start"
        assert names[-1] == "done"
        assert "error" not in names
        assert names.count("done") == 1

    def test_start_describes_the_run(self, client: TestClient) -> None:
        _, payload = collect(client, question="governing law")[0]
        assert payload["documents"] == 2
        assert payload["chunks"] == 4
        assert payload["live_model"] is False
        assert set(payload["nodes"]) == set(GRAPH_NODES)

    def test_node_events_are_ordered_and_timed(self, client: TestClient) -> None:
        nodes = [p for n, p in collect(client, question="governing law") if n == "node"]
        assert [p["index"] for p in nodes] == list(range(1, len(nodes) + 1))
        assert all(p["elapsed_ms"] >= 0 for p in nodes)
        # total_ms is cumulative, so it must never go backwards.
        totals = [p["total_ms"] for p in nodes]
        assert totals == sorted(totals)

    def test_every_streamed_node_has_a_box_on_the_strip(self, client: TestClient) -> None:
        """A node with no box would execute invisibly in the UI."""
        nodes = {p["node"] for n, p in collect(client, question="governing law") if n == "node"}
        assert nodes <= set(GRAPH_NODES), f"not drawn: {nodes - set(GRAPH_NODES)}"

    def test_done_carries_the_answer_and_verified_citations(self, client: TestClient) -> None:
        done = [p for n, p in collect(client, question="What is the governing law?") if n == "done"]
        assert len(done) == 1
        payload = done[0]
        assert payload["status"] == "completed"
        assert payload["answer"]
        assert payload["citations"]
        assert payload["citation_errors"] == []

    def test_citations_carry_the_excerpt_the_page_shows(self, client: TestClient) -> None:
        """Without this the page would need a second request per citation to show the
        contract language, which is the whole point of inspecting one."""
        done = next(p for n, p in collect(client, question="governing law") if n == "done")
        citation = done["citations"][0]
        assert citation["excerpt"]
        assert citation["source_chunk_id"]
        assert citation["source_chunk_id"] in {item["source_chunk_id"] for item in done["evidence"]}
        assert citation["pulled_by"] in ("search", "cross_reference")
        assert citation["internal"].startswith("[doc-")
        assert "doc-" not in citation["display"], "display form uses the title"

    def test_usage_reports_budget_denominators(self, client: TestClient) -> None:
        done = next(p for n, p in collect(client, question="governing law") if n == "done")
        usage = done["usage"]
        assert usage["searches_used"] <= usage["max_searches"]
        assert usage["iterations"] <= usage["max_iterations"]
        assert usage["repair_attempts"] <= usage["max_repair_attempts"]

    def test_abstention_still_produces_a_done_event(self, client: TestClient) -> None:
        """An abstention is an outcome, not a stream failure."""
        done = next(p for n, p in collect(client, question="zzzz qqqq nonexistent") if n == "done")
        assert done["status"] == "abstained"
        assert done["abstained"] is True
        assert done["citations"] == []
        assert done["answer"]

    def test_allowlist_is_honoured(self, client: TestClient) -> None:
        done = next(
            p
            for n, p in collect(client, question="confidential information", document_ids="doc-901")
            if n == "done"
        )
        assert {e["document_id"] for e in done["evidence"]} <= {"doc-901"}

    def test_unknown_document_is_a_client_error(self, client: TestClient) -> None:
        response = client.get(
            "/research/stream", params={"question": "x", "document_ids": "doc-999"}
        )
        assert response.status_code == 400
        assert "doc-999" in response.json()["detail"]

    def test_empty_question_is_rejected(self, client: TestClient) -> None:
        assert client.get("/research/stream", params={"question": ""}).status_code == 422


class TestMetrics:
    def test_search_reports_new_versus_total(self) -> None:
        """The strip shows "5 chunks (8 total)"; both halves come from here."""
        from tests.conftest import make_chunk, make_retrieved

        chunks = [make_retrieved(make_chunk(f"c{i}", "doc-900")) for i in range(8)]
        metrics = extract_metrics("search", {"retrieved_chunks": chunks}, seen_chunks=5)
        assert metrics["total_chunks"] == 8
        assert metrics["new_chunks"] == 3

    def test_select_evidence_reports_the_cross_reference_share(self) -> None:
        from app.schemas.evidence import Evidence

        evidence = [
            Evidence(
                evidence_id="e01",
                topic="t",
                document_id="doc-900",
                document_title="T",
                page_number=1,
                excerpt="x",
                source_chunk_id="c1",
                pulled_by=p,
            )
            for p in ("search", "search", "cross_reference")
        ]
        metrics = extract_metrics("select_evidence", {"evidence": evidence}, 0)
        assert metrics["evidence"] == 3
        assert metrics["via_cross_reference"] == 1

    def test_absent_values_are_omitted_not_null(self) -> None:
        """The page branches on presence; a null would render as the string "null"."""
        assert extract_metrics("write", {}, 0) == {}


class TestArms:
    def test_arms_endpoint_lists_every_experiment_arm(self, client: TestClient) -> None:
        assert client.get("/arms").json() == list(
            ["bm25", "dense", "rrf_hybrid", "rrf_hybrid_rerank"]
        )

    def test_unknown_arm_falls_back_rather_than_failing(self, client: TestClient) -> None:
        """An arm the factory does not know is a UI bug, not a reason to refuse a run."""
        events = collect(client, question="governing law", arm="nonsense")
        assert events[-1][0] == "done"

    def test_unbuildable_arm_degrades_instead_of_erroring(self, client: TestClient) -> None:
        """A stale FAISS index must not make the inspector unusable."""
        events = collect(client, question="governing law", arm="rrf_hybrid_rerank")
        assert events[-1][0] == "done"
        assert events[0][1]["retriever"] == "bm25"


class TestPage:
    def test_index_is_served(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "trace inspector" in response.text
        assert response.headers["cache-control"] == "no-store"

    def test_page_is_self_contained(self, client: TestClient) -> None:
        """CLAUDE.md rule 4: no npm, no bundler, no framework. The only external
        reference permitted is the font stylesheet."""
        body = client.get("/").text
        external = [
            line
            for line in body.splitlines()
            if ('src="http' in line or 'href="http' in line)
            and "fonts.googleapis.com" not in line
            and "fonts.gstatic.com" not in line
        ]
        assert external == [], f"external dependency in the page: {external}"
