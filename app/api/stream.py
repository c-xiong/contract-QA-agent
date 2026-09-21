"""Server-sent events for the trace inspector. See CLAUDE.md rule 4 (carve-out).

V1 reports completed graph updates. V2 tails persisted events during execution, including
model/tool starts before slow calls return. Highlighting follows these events rather than
a front-end animation timer. Contract data remains read-only; V2 writes local run traces.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, suppress
from dataclasses import dataclass
from typing import Any

from app.agent.runner import ResearchAgent
from app.agent.state import Budget, initial_state
from app.api.schemas import render_evidence
from app.evidence.normalizer import normalize_evidence
from app.schemas.evidence import Evidence

logger = logging.getLogger(__name__)

# The canonical node order the page draws. Nodes not in this list still stream; they
# simply have no box on the strip, which is a design bug rather than a runtime one.
GRAPH_NODES = (
    "search",
    "assess",
    "refine",
    "resolve_refs",
    "select_evidence",
    "answerability",
    "write",
    "verify",
    "repair",
    "finalize",
    "abstain",
    "fail",
)


def sse(event: str, payload: dict[str, Any]) -> str:
    """One SSE frame. `data` must not contain a raw newline, so json is compact."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@dataclass
class NodeMetrics:
    """Per-node numbers the page shows next to the node, beyond the trace detail."""

    payload: dict[str, Any]


def extract_metrics(node: str, delta: dict[str, Any], seen_chunks: int) -> dict[str, Any]:
    """Pull the few numbers each node is worth annotating with.

    Deliberately per-node rather than a generic state dump: the page shows "5 chunks
    (8 total)" under `search` and "4 via cross-reference" under `select_evidence`, and
    a generic dump would show neither usefully.
    """
    metrics: dict[str, Any] = {}

    if node == "search":
        chunks = delta.get("retrieved_chunks") or []
        metrics["total_chunks"] = len(chunks)
        metrics["new_chunks"] = max(0, len(chunks) - seen_chunks)
        queries = delta.get("search_queries") or []
        if queries:
            metrics["query"] = queries[-1]
        metrics["searches_used"] = delta.get("searches_used")

    elif node == "assess":
        metrics["stop_reason"] = delta.get("stop_reason")
        metrics["iterations"] = delta.get("research_iterations")

    elif node == "refine":
        queries = delta.get("search_queries") or []
        if queries:
            metrics["query"] = queries[-1]
        metrics["stopped"] = delta.get("stop_reason") == "no_new_evidence"

    elif node == "resolve_refs":
        chunks = delta.get("retrieved_chunks") or []
        metrics["pulled"] = sum(1 for c in chunks if c.pulled_by == "cross_reference")

    elif node == "select_evidence":
        evidence: list[Evidence] = delta.get("evidence") or []
        metrics["evidence"] = len(evidence)
        metrics["via_cross_reference"] = sum(
            1 for e in evidence if e.pulled_by == "cross_reference"
        )

    elif node == "answerability":
        decision = delta.get("answerability")
        metrics["verdict"] = decision.verdict if decision else "failed"
        metrics["model_calls"] = delta.get("model_calls")

    elif node == "write":
        metrics["input_tokens"] = delta.get("input_tokens")
        metrics["output_tokens"] = delta.get("output_tokens")

    elif node == "verify":
        metrics["citations"] = len(delta.get("citations") or [])
        metrics["errors"] = len(delta.get("citation_errors") or [])
        metrics["error_codes"] = sorted({e.code for e in (delta.get("citation_errors") or [])})

    elif node == "repair":
        metrics["attempt"] = delta.get("repair_attempts")

    elif node in ("finalize", "abstain", "fail"):
        metrics["status"] = delta.get("status")
        metrics["failure"] = delta.get("failure")

    return {k: v for k, v in metrics.items() if v is not None}


def build_done_payload(final: dict[str, Any], agent: ResearchAgent) -> dict[str, Any]:
    """The completed-run payload: answer, citations, evidence, budget usage."""
    titles = {d.document_id: d.title for d in agent.store.documents}

    evidence: list[Evidence] = final.get("evidence") or []
    if not evidence and final.get("retrieved_chunks"):
        # An abstention short-circuits before select_evidence; show what was retrieved
        # anyway so the page can explain why the run stopped.
        evidence = normalize_evidence(final["retrieved_chunks"], max_chars=24_000)

    citations = []
    for index, citation in enumerate(final.get("citations") or [], start=1):
        title = titles.get(citation.document_id, citation.document_id)
        parts = [title]
        if citation.page_number is not None:
            parts.append(f"p. {citation.page_number}")
        if citation.section_id is not None:
            parts.append(f"§{citation.section_id}")

        # Attach the evidence excerpt this citation points at, so the page can show the
        # contract language without a second request. Matched on citable location --
        # the same key the verifier grounds on.
        excerpt = None
        source_chunk = None
        pulled_by = None
        for item in evidence:
            if item.document_id != citation.document_id:
                continue
            if citation.page_number is not None and item.page_number != citation.page_number:
                continue
            if citation.section_id is not None and item.section_id != citation.section_id:
                continue
            excerpt, source_chunk, pulled_by = item.excerpt, item.source_chunk_id, item.pulled_by
            break

        citations.append(
            {
                "index": index,
                "internal": citation.render(),
                "display": ", ".join(parts),
                "document_id": citation.document_id,
                "document_title": title,
                "page_number": citation.page_number,
                "section_id": citation.section_id,
                "excerpt": excerpt,
                "source_chunk_id": source_chunk,
                "pulled_by": pulled_by,
            }
        )

    budget: Budget = final.get("budget") or Budget()
    return {
        "status": final.get("status", "failed"),
        "abstained": final.get("status") == "abstained",
        "answer": final.get("final_answer") or "",
        "citations": citations,
        "citation_errors": [
            {"code": e.code, "raw": e.raw, "detail": e.detail}
            for e in (final.get("citation_errors") or [])
        ],
        "evidence": [e.model_dump(mode="json") for e in render_evidence(evidence)],
        "failure": final.get("failure"),
        "failure_detail": final.get("failure_detail"),
        "usage": {
            "searches_used": final.get("searches_used", 0),
            "max_searches": budget.max_searches,
            "iterations": final.get("research_iterations", 0),
            "max_iterations": budget.max_research_iterations,
            "repair_attempts": final.get("repair_attempts", 0),
            "max_repair_attempts": budget.max_repair_attempts,
            "cross_reference_pulled": sum(
                1 for c in (final.get("retrieved_chunks") or []) if c.pulled_by == "cross_reference"
            ),
            "max_cross_reference_chunks": budget.max_cross_reference_chunks,
            "tool_calls": final.get("tool_calls", 0),
            "max_tool_calls": budget.max_tool_calls,
            "model_calls": final.get("model_calls", 0),
            "max_model_calls": budget.max_model_calls,
            "retries": final.get("retries", 0),
            "deadline_seconds": budget.deadline_seconds,
            "evidence_items": len(final.get("evidence_by_id", {})),
            "max_evidence_items": budget.max_evidence_items,
            "input_tokens": final.get("input_tokens", 0),
            "output_tokens": final.get("output_tokens", 0),
        },
        "trace": [{"step": e.step, "detail": e.detail} for e in (final.get("trace") or [])],
    }


async def research_events(
    agent: ResearchAgent,
    question: str,
    *,
    allowed_document_ids: list[str] | None,
) -> AsyncIterator[str]:
    """Stream one research run as SSE frames."""
    if agent.settings.agent_mode == "v2":
        async with aclosing(
            v2_research_events(agent, question, allowed_document_ids=allowed_document_ids)
        ) as events:
            async for frame in events:
                yield frame
        return
    loop = asyncio.get_running_loop()
    started = loop.time()

    yield sse(
        "start",
        {
            "question": question,
            "allowed_document_ids": allowed_document_ids,
            "retriever": agent.retriever.name,
            "model_id": agent.client.model_id,
            "live_model": agent.settings.live_model,
            "agent_mode": agent.settings.agent_mode,
            "documents": len(agent.store.documents),
            "chunks": len(agent.store),
            "nodes": list(GRAPH_NODES),
        },
    )

    state = initial_state(
        question,
        allowed_document_ids=allowed_document_ids,
        budget=Budget(max_chunks_per_query=agent.settings.retrieval_top_k),
    )

    final: dict[str, Any] = {}
    index = 0
    seen_chunks = 0
    last_at = started

    try:
        async for mode, chunk in agent.graph.astream(state, stream_mode=["updates", "values"]):
            # LangGraph types the multi-mode chunk as a union; both branches are dicts
            # in practice, and each is narrowed before use.
            if mode == "values":
                if isinstance(chunk, dict):
                    final = chunk
                continue
            if not isinstance(chunk, dict):
                continue

            for node, delta in chunk.items():
                if not isinstance(delta, dict):
                    continue
                index += 1
                now = loop.time()
                trace = delta.get("trace") or []
                detail = trace[-1].detail if trace else ""
                metrics = extract_metrics(node, delta, seen_chunks)
                if node in ("search", "resolve_refs"):
                    seen_chunks = len(delta.get("retrieved_chunks") or [])

                yield sse(
                    "node",
                    {
                        "index": index,
                        "node": node,
                        "detail": detail,
                        "elapsed_ms": round((now - last_at) * 1000),
                        "total_ms": round((now - started) * 1000),
                        "metrics": metrics,
                    },
                )
                last_at = now

        payload = build_done_payload(final, agent)
        payload["total_ms"] = round((loop.time() - started) * 1000)
        payload["steps"] = index
        yield sse("done", payload)

    except asyncio.CancelledError:
        # The viewer navigated away or closed the tab. Not an error; do not report one.
        raise
    except Exception as exc:
        # A stream that dies silently looks to the page like a hung run. Report the
        # failure as an event so the UI can show it, and log the traceback server-side.
        logger.exception("research stream failed")
        yield sse("error", {"detail": f"{type(exc).__name__}: {exc}"})


async def v2_research_events(
    agent: ResearchAgent,
    question: str,
    *,
    allowed_document_ids: list[str] | None,
) -> AsyncGenerator[str, None]:
    """Persist first, then publish only safe events. Disconnects cancel, never restart."""
    from pathlib import Path

    from app.agent.execution import v2_states
    from app.agent.state import ResearchState
    from app.observability.events import public

    state = initial_state(
        question,
        allowed_document_ids=allowed_document_ids,
        budget=Budget(max_chunks_per_query=agent.settings.retrieval_top_k),
    )
    final: ResearchState = state
    delivered = 0
    execution_path: list[str] = []
    loop = asyncio.get_running_loop()
    started = loop.time()
    yield sse(
        "start",
        {
            "question": question,
            "allowed_document_ids": allowed_document_ids,
            "retriever": agent.retriever.name,
            "model_id": agent.client.model_id,
            "live_model": agent.settings.live_model,
            "agent_mode": "v2",
            "documents": len(agent.store.documents),
            "chunks": len(agent.store),
            "nodes": [
                "initialize",
                "decide",
                "authorize",
                "execute",
                "write",
                "verify",
                "repair",
                "terminal",
            ],
        },
    )

    async def execute() -> None:
        nonlocal final
        async with aclosing(v2_states(agent.graph, state, agent.settings.runs_dir)) as states:
            async for merged in states:
                final = merged

    # Tail the durable journal while a node is running. Awaiting only graph values
    # batches model_started/tool_started behind slow calls and makes progress misleading.
    task = asyncio.create_task(execute())
    try:
        while True:
            finished = task.done()
            trace_path = final.get("trace_path") or state.get("trace_path")
            if trace_path:
                try:
                    lines = (await asyncio.to_thread(Path(trace_path).read_text)).splitlines()
                except FileNotFoundError:
                    lines = []
                for line in lines[delivered:]:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        break  # A concurrently appended final line may be incomplete.
                    delivered += 1
                    payload = public(event)
                    node = event["node"]
                    if not execution_path or execution_path[-1] != node:
                        execution_path.append(node)
                    payload["total_ms"] = round((loop.time() - started) * 1000)
                    yield f"id: {event['event_id']}\n" + sse("progress", payload)
            if finished:
                await task  # propagate a sanitized transport error if execution failed
                break
            await asyncio.wait({task}, timeout=0.05)
        payload = build_done_payload(dict(final), agent)
        payload.update(
            run_id=final.get("run_id"),
            stop_reason=final.get("stop_reason"),
            agent_mode="v2",
            total_ms=round((loop.time() - started) * 1000),
            steps=delivered,
            verification_ok=final.get("verification_ok", False),
            execution_path=execution_path,
        )
        if not final.get("usage_known", True):
            payload["usage"]["input_tokens"] = None
            payload["usage"]["output_tokens"] = None
        yield sse("done", payload)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("V2 research stream failed")
        yield sse(
            "error", {"detail": "The run failed. Inspect its local trace for the terminal reason."}
        )
    finally:
        if not task.done():
            task.cancel()
        # The execution boundary has already persisted the terminal outcome.
        with suppress(asyncio.CancelledError, Exception):
            await task
