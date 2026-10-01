"""Shared tool execution boundary for the V2 graph and the MCP server.

DECISION: one executor owns tool policy for every caller. The graph and the MCP adapter
  both call `ToolExecutor`; neither reimplements scope checks, budget reservation,
  retries, provenance checks or evidence admission. Alternative rejected: an MCP-only
  wrapper around `ToolRegistry`, which would leave the graph's copies of these rules in
  `v2.py` and let the two implementations drift.

DECISION: `ToolSession` is the single mutable store for granted scope, counters and
  admitted evidence while a call executes. The graph keeps its durable state between
  nodes, so its execute node builds a session from state, and the executor's event
  callback writes the session back before every emitted event and when the node ends.
  At any moment exactly one copy is authoritative. The MCP server keeps one session for
  its process lifetime instead.

Planning, model-call budgets, repair routing, duplicate-action detection and the
graph's mandatory final-answer gate stay in `v2.py`; this is not an orchestrator.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import monotonic, perf_counter
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from app.agent.tool_registry import (
    Action,
    AuthorizationError,
    Contract,
    DateCalculation,
    EvidenceRef,
    Search,
    ToolError,
    ToolRegistry,
    ToolResult,
    digest,
)
from app.schemas.evidence import Evidence
from app.schemas.retrieval import RetrievedChunk

EventSink = Callable[..., None]


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    """Per-session tool limits. `reserved_tool_calls` is held back for a caller's gate.

    The graph reserves one tool call for its mandatory final verification. MCP has no
    mandatory finalization, so it reserves none and must not imply the same guarantee.
    """

    max_tool_calls: int
    max_searches: int
    max_retries: int
    max_chunks_per_query: int
    max_evidence_items: int
    max_evidence_chars: int
    reserved_tool_calls: int = 0


@dataclass
class ToolSession:
    """Granted scope plus the counters and evidence one caller has accumulated."""

    allowed: tuple[str, ...]
    versions: Mapping[str, str]
    limits: ExecutionLimits
    evidence_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    calculations: list[dict[str, Any]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    tool_calls: int = 0
    searches_used: int = 0
    retries: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Granted scope is immutable for the session's lifetime; arguments only narrow it.
        self.allowed = tuple(self.allowed)
        self.versions = MappingProxyType(dict(self.versions))

    @property
    def evidence_chars(self) -> int:
        return sum(len(e.excerpt) for e in self.evidence)


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """A dispatched tool's final result, or the reason nothing (further) was dispatched.

    `stop_reason` uses the graph's existing stop vocabulary; callers map it to their own
    terminal behavior (the graph stops the run, MCP returns a structured error).
    """

    tool: str
    result: ToolResult | None
    stop_reason: str | None = None
    admitted: tuple[str, ...] = ()
    # The AuthorizationError code behind a `tool_scope_violation`, e.g. unknown_evidence.
    error_code: str | None = None


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def authorize(self, session: ToolSession, action: Action) -> Contract:
        """Validate arguments against granted scope before any capacity is reserved.

        Raises AuthorizationError for scope/provenance violations and pydantic
        ValidationError for malformed arguments; callers keep their own mappings.
        """
        payload = self.registry.validate(
            action,
            list(session.allowed),
            dict(session.versions),
            session.limits.max_chunks_per_query,
        )
        if isinstance(payload, DateCalculation) and (
            not action.rule_evidence_ids
            or not set(action.rule_evidence_ids) <= session.evidence_by_id.keys()
        ):
            raise AuthorizationError("missing_governing_evidence")
        return payload

    def _reserve(self, session: ToolSession, payload: Contract, attempt: int) -> str | None:
        limits = session.limits
        with session.lock:
            if session.tool_calls >= limits.max_tool_calls - limits.reserved_tool_calls:
                return "tool_budget_exhausted"
            if isinstance(payload, Search) and session.searches_used >= limits.max_searches:
                return "search_budget_exhausted"
            # Each dispatched attempt consumes capacity, including failures and retries;
            # nothing is refunded afterwards.
            session.tool_calls += 1
            if attempt:
                session.retries += 1
            if isinstance(payload, Search):
                session.searches_used += 1
                session.search_queries = [*session.search_queries, payload.query]
        return None

    def _admit(self, session: ToolSession, hit: RetrievedChunk) -> str | None:
        ref = self.registry.ref(hit.chunk)
        limits = session.limits
        with session.lock:
            if ref.evidence_id in session.evidence_by_id:
                return None
            if (
                len(session.evidence_by_id) >= limits.max_evidence_items
                or session.evidence_chars + len(hit.chunk.text) > limits.max_evidence_chars
            ):
                return None
            c = hit.chunk
            session.evidence_by_id[ref.evidence_id] = ref.model_dump(mode="json")
            session.retrieved = [*session.retrieved, hit]
            session.evidence = [
                *session.evidence,
                Evidence(
                    evidence_id=ref.evidence_id,
                    topic=hit.retrieval_query,
                    document_id=c.document_id,
                    document_title=c.document_title,
                    section_path=c.section_path,
                    section_title=c.section_title,
                    page_number=c.page_number,
                    excerpt=c.text,
                    source_chunk_id=c.chunk_id,
                    pulled_by=hit.pulled_by,
                ),
            ]
        return ref.evidence_id

    async def execute(
        self,
        session: ToolSession,
        tool: str,
        payload: Contract,
        *,
        deadline_at: float,
        rule_evidence_ids: list[str] | None = None,
        on_event: EventSink | None = None,
    ) -> ExecutionOutcome:
        """Reserve, dispatch with bounded retry, verify provenance, then admit evidence."""

        def event(event_type: str, **data: Any) -> None:
            if on_event is not None:
                on_event(event_type, **data)

        limits = session.limits
        attempts = min(1, max(0, limits.max_retries)) + 1
        result: ToolResult | None = None
        for attempt in range(attempts):
            if monotonic() >= deadline_at:
                return ExecutionOutcome(tool, result, "deadline_exhausted")
            refused = self._reserve(session, payload, attempt)
            if refused:
                return ExecutionOutcome(tool, result, refused)
            call_id = uuid4().hex
            event(
                "tool_started",
                call_id=call_id,
                tool=tool,
                attempt=attempt,
                arguments_hash=digest(payload.model_dump(mode="json")),
            )
            started = perf_counter()
            # The lock is never held across this await: slow reads must not serialize
            # unrelated reservations, and a cancelled call keeps its consumed reservation.
            try:
                result = await asyncio.wait_for(
                    self.registry.execute(
                        payload,
                        allowed=list(session.allowed),
                        versions=dict(session.versions),
                        evidence={
                            k: EvidenceRef.model_validate(v)
                            for k, v in session.evidence_by_id.items()
                        },
                        retrieved=session.retrieved,
                        call_id=call_id,
                    ),
                    timeout=max(0.001, deadline_at - monotonic()),
                )
            except AuthorizationError as exc:
                event("tool_failed", call_id=call_id, code="scope_violation")
                return ExecutionOutcome(tool, None, "tool_scope_violation", error_code=str(exc))
            except (TimeoutError, ConnectionError) as exc:
                result = ToolResult(
                    call_id=call_id,
                    status="error",
                    error=ToolError(
                        code="read_timeout"
                        if isinstance(exc, TimeoutError)
                        else "read_unavailable",
                        retryable=True,
                        safe_message="Read failed.",
                    ),
                )
            except RuntimeError:
                result = ToolResult(
                    call_id=call_id,
                    status="error",
                    error=ToolError(code="tool_unavailable", safe_message="Tool failed."),
                )
            result = result.model_copy(
                update={
                    "elapsed_ms": (perf_counter() - started) * 1000,
                    "result_hash": result.result_hash or digest(result.model_dump(mode="json")),
                }
            )
            with session.lock:
                session.tool_results[call_id] = result.model_dump(mode="json") | {
                    "tool": tool,
                    "attempt": attempt,
                }
            event(
                "tool_failed" if result.status == "error" else "tool_completed",
                call_id=call_id,
                result_hash=result.result_hash,
            )
            if not (result.error and result.error.retryable and attempt < attempts - 1):
                break
        assert result is not None  # the loop dispatched at least once or returned
        if (
            result.status == "error"
            and result.error
            and (result.error.retryable or result.error.code == "tool_unavailable")
        ):
            return ExecutionOutcome(tool, result, "tool_unavailable")
        admitted: list[str] = []
        for raw in result.data.get("hits", []):
            evidence_id = self._admit(session, RetrievedChunk.model_validate(raw))
            if evidence_id:
                admitted.append(evidence_id)
                event("evidence_added", evidence_id=evidence_id)
        if tool == "calculate_date" and result.status == "ok":
            with session.lock:
                session.calculations = [
                    *session.calculations,
                    result.data | {"rule_evidence_ids": list(rule_evidence_ids or [])},
                ]
        return ExecutionOutcome(tool, result, None, tuple(admitted))
