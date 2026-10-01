"""MCP adapter boundary checks: scope grants, evidence provenance, budgets and sessions.

Policy tests connect in-process; `test_stdio_*` launch the real server over stdio.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters

from app.agent.state import Budget
from app.agent.tool_registry import ToolRegistry
from app.agent.tool_session import ExecutionLimits
from app.mcp_server import (
    ConfigurationError,
    ContractToolServer,
    ServerConfig,
    build_server,
    default_limits,
    grant_scope,
)
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk
from tests.unit.test_v2 import registry

__all__ = ["registry"]  # re-exported pytest fixture

REPO = Path(__file__).resolve().parents[2]


def tools_for(
    registry: ToolRegistry,
    allow: list[str] | None = None,
    *,
    limits: ExecutionLimits | None = None,
    call_timeout: float = 30.0,
    lifetime: float | None = None,
) -> ContractToolServer:
    allowed, versions = grant_scope(registry, allow or ["doc-900"])
    return ContractToolServer(
        registry,
        ServerConfig(
            allowed,
            versions,
            limits or default_limits(),
            call_timeout_seconds=call_timeout,
            session_lifetime_seconds=lifetime,
        ),
    )


async def call(tools: ContractToolServer, name: str, **arguments: Any) -> tuple[bool, Any]:
    async with Client(build_server(tools)) as client:
        result = await client.call_tool(name, arguments)
    return result.is_error, result.structured_content


def error_code(content: dict[str, Any]) -> str:
    return content["error"]["code"]


# --- startup grants ------------------------------------------------------------


def test_scope_fails_closed(registry: ToolRegistry) -> None:
    with pytest.raises(ConfigurationError, match="no document scope"):
        grant_scope(registry, [])
    with pytest.raises(ConfigurationError, match="unknown document"):
        grant_scope(registry, ["doc-404"])
    with pytest.raises(ConfigurationError, match="either"):
        grant_scope(registry, ["doc-900"], allow_all=True)
    with pytest.raises(ConfigurationError, match="does not match"):
        grant_scope(registry, ["doc-900"], pins=["doc-900=sha256:stale"])
    with pytest.raises(ConfigurationError, match="granted document"):
        grant_scope(registry, ["doc-900"], pins=[f"doc-901={registry.versions['doc-901']}"])
    allowed, versions = grant_scope(
        registry, ["doc-900"], pins=[f"doc-900={registry.versions['doc-900']}"]
    )
    assert allowed == ("doc-900",) and versions == {"doc-900": registry.versions["doc-900"]}
    assert grant_scope(registry, [], allow_all=True)[0] == tuple(registry.versions)


async def test_tools_are_discoverable_and_read_only(registry: ToolRegistry) -> None:
    async with Client(build_server(tools_for(registry))) as client:
        listed = (await client.list_tools()).tools
    assert {t.name for t in listed} == {
        "session_info",
        "document_search",
        "retrieve_clause",
        "extract_definition",
        "calculate_date",
        "verify_citation",
    }
    assert all(t.annotations and t.annotations.read_only_hint for t in listed)
    assert all(t.output_schema for t in listed)


async def test_session_info_reports_only_the_grant(registry: ToolRegistry) -> None:
    failed, info = await call(tools_for(registry), "session_info")
    assert not failed
    assert [d["doc_id"] for d in info["documents"]] == ["doc-900"]
    assert info["documents"][0]["version_id"] == registry.versions["doc-900"]
    assert info["reserved_final_verification"] is False
    assert info["usage"]["tool_calls"] == 0


# --- scope and argument checks ---------------------------------------------------


@pytest.mark.parametrize(
    "name,arguments,code",
    [
        ("document_search", {"query": "notice", "doc_ids": ["doc-901"]}, "document_not_allowed"),
        (
            "document_search",
            {"query": "notice", "doc_ids": ["doc-900", "doc-901"]},
            "document_not_allowed",
        ),
        (
            "retrieve_clause",
            {"doc_id": "doc-901", "version_id": "any", "clause_ref": "1"},
            "document_not_allowed",
        ),
        (
            "extract_definition",
            {"doc_id": "doc-900", "version_id": "sha256:invented", "term": "Start Date"},
            "version_not_allowed",
        ),
        ("document_search", {"query": ""}, "invalid_arguments"),
        ("document_search", {"query": "notice", "top_k": 0}, "invalid_arguments"),
    ],
)
async def test_scope_widening_and_invalid_arguments_rejected_before_dispatch(
    registry: ToolRegistry, name: str, arguments: dict[str, Any], code: str
) -> None:
    tools = tools_for(registry)
    failed, content = await call(tools, name, **arguments)
    assert failed and error_code(content) == code
    assert tools.session.tool_calls == 0 and not tools.session.evidence_by_id


async def test_retriever_leak_is_rejected_and_not_admitted(registry: ToolRegistry) -> None:
    leaked = registry.store.chunks_for("doc-901")[0]

    class Leaky:
        name = "leaky"

        async def search(
            self, query: str, *, top_k: int, filters: RetrievalFilters | None = None
        ) -> list[RetrievedChunk]:
            return [RetrievedChunk(chunk=leaked, fused_score=0, retrieval_query=query)]

    registry.retriever = Leaky()
    tools = tools_for(registry)
    failed, content = await call(tools, "document_search", query="notice")
    assert failed and error_code(content) == "retriever_scope_violation"
    assert not tools.session.evidence_by_id
    assert "Private notice" not in str(content)


# --- evidence provenance ---------------------------------------------------------


async def test_search_admits_and_marks_new_evidence(registry: ToolRegistry) -> None:
    tools = tools_for(registry)
    async with Client(build_server(tools)) as client:
        first = (await client.call_tool("document_search", {"query": "notice"})).structured_content
        again = (await client.call_tool("document_search", {"query": "notice"})).structured_content
    assert first and again
    assert first["status"] == "ok"
    assert all(e["doc_id"] == "doc-900" for e in first["evidence"])
    assert all(e["newly_admitted"] for e in first["evidence"])
    assert not any(e["newly_admitted"] for e in again["evidence"])
    assert again["usage"]["searches_used"] == 2
    ids = {e["evidence_id"] for e in first["evidence"]}
    assert ids == set(tools.session.evidence_by_id)


async def test_fabricated_evidence_is_rejected(registry: ToolRegistry) -> None:
    tools = tools_for(registry)
    failed, content = await call(
        tools, "verify_citation", answer="notice [doc-900, p. 1, §1]", evidence_ids=["e-fake"]
    )
    assert failed and error_code(content) == "unknown_evidence"
    # Graph accounting semantics: a dispatched-then-rejected call consumed its reservation.
    assert tools.session.tool_calls == 1


async def test_modified_citation_fails_structural_check(registry: ToolRegistry) -> None:
    tools = tools_for(registry)
    version = registry.versions["doc-900"]
    async with Client(build_server(tools)) as client:
        clause = (
            await client.call_tool(
                "retrieve_clause", {"doc_id": "doc-900", "version_id": version, "clause_ref": "1"}
            )
        ).structured_content
        assert clause
        ids = [e["evidence_id"] for e in clause["evidence"]]
        valid = (
            await client.call_tool(
                "verify_citation", {"answer": "notice [doc-900, p. 1, §1]", "evidence_ids": ids}
            )
        ).structured_content
        # Same admitted evidence, but the citation points at a section never admitted.
        altered = (
            await client.call_tool(
                "verify_citation", {"answer": "notice [doc-900, p. 1, §2]", "evidence_ids": ids}
            )
        ).structured_content
    assert valid and altered
    assert valid["structurally_valid"] and valid["semantic_support"] == "not_assessed"
    assert not altered["structurally_valid"]
    assert {e["code"] for e in altered["errors"]} & {"not_in_evidence", "section_not_found"}


async def test_evidence_is_isolated_between_sessions(registry: ToolRegistry) -> None:
    first, second = tools_for(registry), tools_for(registry)
    failed, found = await call(first, "document_search", query="notice")
    assert not failed
    ids = [e["evidence_id"] for e in found["evidence"]]
    assert ids
    failed, content = await call(
        second, "verify_citation", answer="notice [doc-900, p. 1, §1]", evidence_ids=ids
    )
    assert failed and error_code(content) == "unknown_evidence"
    failed, content = await call(
        second,
        "calculate_date",
        anchor_date="2026-01-01",
        offset=30,
        direction="after",
        convention="calendar_days",
        rule_evidence_ids=ids,
    )
    assert failed and error_code(content) == "missing_governing_evidence"


# --- date rules ------------------------------------------------------------------


async def test_date_calculation_requires_admitted_rule(registry: ToolRegistry) -> None:
    tools = tools_for(registry)
    arguments = {
        "anchor_date": "2026-01-01",
        "offset": 30,
        "direction": "after",
        "convention": "calendar_days",
    }
    async with Client(build_server(tools)) as client:
        ungrounded = await client.call_tool("calculate_date", arguments | {"rule_evidence_ids": []})
        assert ungrounded.is_error and ungrounded.structured_content
        assert error_code(ungrounded.structured_content) == "missing_governing_evidence"
        found = (await client.call_tool("document_search", {"query": "notice"})).structured_content
        assert found
        rule = [found["evidence"][0]["evidence_id"]]
        grounded = (
            await client.call_tool("calculate_date", arguments | {"rule_evidence_ids": rule})
        ).structured_content
        unsupported = await client.call_tool(
            "calculate_date",
            arguments | {"rule_evidence_ids": rule, "convention": "business_days"},
        )
    assert grounded and grounded["date"] == "2026-01-31" and grounded["rule_evidence_ids"] == rule
    assert "governing" in grounded["limitation"]
    assert unsupported.is_error and unsupported.structured_content
    assert error_code(unsupported.structured_content) == "unsupported_convention"
    assert len(tools.session.calculations) == 1


# --- budgets, deadlines and sessions ---------------------------------------------


def limits(**overrides: Any) -> ExecutionLimits:
    base = default_limits(Budget())
    return ExecutionLimits(**{n: getattr(base, n) for n in ExecutionLimits.__slots__} | overrides)


async def test_tool_and_search_budgets_are_enforced(registry: ToolRegistry) -> None:
    tools = tools_for(registry, limits=limits(max_tool_calls=3, max_searches=1))
    version = registry.versions["doc-900"]
    clause = {"doc_id": "doc-900", "version_id": version, "clause_ref": "1"}
    async with Client(build_server(tools)) as client:
        assert not (await client.call_tool("document_search", {"query": "notice"})).is_error
        second = await client.call_tool("document_search", {"query": "start"})
        assert not (await client.call_tool("retrieve_clause", clause)).is_error
        assert not (await client.call_tool("retrieve_clause", clause)).is_error
        over = await client.call_tool("retrieve_clause", clause)
    assert second.is_error and second.structured_content
    assert error_code(second.structured_content) == "search_budget_exhausted"
    assert over.is_error and over.structured_content
    assert error_code(over.structured_content) == "tool_budget_exhausted"
    assert tools.session.tool_calls == 3 and tools.session.searches_used == 1


async def test_concurrent_calls_cannot_overspend(registry: ToolRegistry) -> None:
    tools = tools_for(registry, limits=limits(max_tool_calls=3, max_searches=10))
    async with Client(build_server(tools)) as client:
        results = await asyncio.gather(
            *(client.call_tool("document_search", {"query": f"notice {i}"}) for i in range(10))
        )
    succeeded = [r for r in results if not r.is_error]
    refused = [r for r in results if r.is_error]
    assert len(succeeded) == 3
    assert all(
        r.structured_content and error_code(r.structured_content) == "tool_budget_exhausted"
        for r in refused
    )
    assert tools.session.tool_calls == 3 and tools.session.searches_used == 3


async def test_concurrent_admission_respects_evidence_limit(registry: ToolRegistry) -> None:
    tools = tools_for(registry, allow=["doc-900", "doc-901"], limits=limits(max_evidence_items=2))
    async with Client(build_server(tools)) as client:
        results = await asyncio.gather(
            client.call_tool("document_search", {"query": "notice"}),
            client.call_tool("document_search", {"query": "zebra orchard apple orange"}),
            client.call_tool("document_search", {"query": "start date"}),
        )
    assert all(not r.is_error for r in results)
    assert len(tools.session.evidence_by_id) == 2
    returned = sum(len(r.structured_content["evidence"]) for r in results if r.structured_content)
    withheld = sum(r.structured_content["withheld_hits"] for r in results if r.structured_content)
    assert withheld > 0 and returned >= 2


def blocking(registry: ToolRegistry) -> None:
    class Blocking:
        name = "blocking"

        async def search(
            self, query: str, *, top_k: int, filters: RetrievalFilters | None = None
        ) -> list[RetrievedChunk]:
            time.sleep(0.15)  # noqa: ASYNC251 -- deliberately simulate native blocking retrieval
            return []

    registry.retriever = Blocking()


async def test_call_timeout_consumes_reservation(registry: ToolRegistry) -> None:
    blocking(registry)
    tools = tools_for(registry, call_timeout=0.05)
    failed, content = await call(tools, "document_search", query="notice")
    assert failed and error_code(content) == "read_timeout"
    # The timed-out attempt is not refunded; its retry found the deadline expired.
    assert tools.session.tool_calls == 1 and tools.session.searches_used == 1
    # The session survives a per-call timeout.
    failed, info = await call(tools, "session_info")
    assert not failed and info["usage"]["tool_calls"] == 1


async def test_cancelled_call_keeps_its_reservation(registry: ToolRegistry) -> None:
    blocking(registry)
    tools = tools_for(registry)
    task = asyncio.create_task(tools.call("document_search", {"query": "notice"}))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tools.session.tool_calls == 1 and not tools.session.evidence_by_id


async def test_session_lifetime_expiry(registry: ToolRegistry) -> None:
    tools = tools_for(registry, lifetime=0.01)
    await asyncio.sleep(0.03)
    failed, content = await call(tools, "document_search", query="notice")
    assert failed and error_code(content) == "session_expired"
    assert tools.session.tool_calls == 0


async def test_client_can_skip_verification(registry: ToolRegistry) -> None:
    """Negative demonstration: nothing forces an MCP client through verify_citation."""
    tools = tools_for(registry)
    async with Client(build_server(tools)) as client:
        found = await client.call_tool("document_search", {"query": "notice"})
    assert not found.is_error
    # The client now holds evidence text and may write any prose; the server never sees it.
    called = {r["tool"] for r in tools.session.tool_results.values()}
    assert called == {"document_search"}


# --- real stdio process ----------------------------------------------------------


def corpus(registry: ToolRegistry, tmp_path: Path) -> Path:
    registry.store.save(tmp_path / "processed")
    return tmp_path / "processed"


def params(processed: Path, *args: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server", "--processed-dir", str(processed), *args],
        cwd=str(REPO),
        env={"CRA_LIVE_MODEL": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )


async def test_stdio_server_lists_and_calls_tools(registry: ToolRegistry, tmp_path: Path) -> None:
    processed = corpus(registry, tmp_path)
    async with Client(params(processed, "--allow", "doc-900")) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        found = await client.call_tool("document_search", {"query": "notice"})
        refused = await client.call_tool(
            "document_search", {"query": "notice", "doc_ids": ["doc-901"]}
        )
    assert "verify_citation" in names and len(names) == 6
    assert not found.is_error and found.structured_content
    assert found.structured_content["evidence"]
    assert refused.is_error and refused.structured_content
    assert error_code(refused.structured_content) == "document_not_allowed"
    # A new process is a new session: evidence and quotas do not carry over.
    async with Client(params(processed, "--allow", "doc-900")) as client:
        info = (await client.call_tool("session_info", {})).structured_content
    assert info and info["usage"]["tool_calls"] == 0 and info["usage"]["evidence_items"] == 0


def test_stdio_server_refuses_to_start_without_scope(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    processed = corpus(registry, tmp_path)
    completed = subprocess.run(
        [sys.executable, "-m", "app.mcp_server", "--processed-dir", str(processed)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""  # stdout is reserved for protocol frames
    assert "no document scope" in completed.stderr


def test_stdio_unsupported_protocol_gets_a_supported_counteroffer(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    """MCP negotiation: the server never echoes an unknown revision; the client decides."""
    from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS

    processed = corpus(registry, tmp_path)
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "1999-01-01",
            "capabilities": {},
            "clientInfo": {"name": "probe", "version": "0"},
        },
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.mcp_server",
            "--processed-dir",
            str(processed),
            "--allow",
            "doc-900",
        ],
        cwd=REPO,
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    reply = json.loads(completed.stdout.splitlines()[0])
    assert reply["id"] == 1
    offered = reply["result"]["protocolVersion"]
    assert offered != "1999-01-01" and offered in HANDSHAKE_PROTOCOL_VERSIONS
