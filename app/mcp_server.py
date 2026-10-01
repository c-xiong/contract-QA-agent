"""Local read-only MCP server over the V2 contract tools (stdio transport only).

DECISION: this module is a protocol adapter. Scope checks, budget reservation, retries,
  provenance checks and evidence admission run in the same `ToolExecutor` the V2 graph
  uses; nothing here re-decides tool policy. Alternative rejected: calling
  `ToolRegistry` directly from MCP handlers, which would bypass budgets and admission.

DECISION: scope is granted at process start (`--allow`/`--allow-all`, optional
  `--pin`), never by a tool argument. Missing scope fails closed before serving.
  Tool arguments may only narrow that grant.

DECISION: one stdio process serves one client connection and is one session. There is
  no `start_session` tool and no persistence: exit discards evidence and counters, and a
  new process starts with fresh quotas. A session is a state boundary, not an identity
  or a global anti-abuse limit. The 2026-07-28 protocol revision is stateless per
  request, but over stdio the process still maps to exactly one connection.

What this server cannot enforce: an external client may skip `verify_citation`,
ignore its result, or write prose the server never sees. The graph's mandatory
final-answer gate does not apply to MCP clients, and no final-verifier call is reserved.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import monotonic
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ValidationError

from app.agent.state import Budget
from app.agent.tool_registry import (
    Action,
    AuthorizationError,
    ToolRegistry,
    ToolResult,
)
from app.agent.tool_session import ExecutionLimits, ToolExecutor, ToolSession
from app.config import Settings
from app.evidence.citation_verifier import CitationVerifier
from app.ingestion.store import ChunkStore
from app.retrieval.factory import build_retriever

logger = logging.getLogger("app.mcp_server")

SERVER_VERSION = "mcp-adapter-1"
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


class ConfigurationError(ValueError):
    """Server startup configuration is missing or invalid; the server does not start."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    allowed: tuple[str, ...]
    versions: dict[str, str]
    limits: ExecutionLimits
    call_timeout_seconds: float = 30.0
    # None disables the optional session lifetime; see module docstring.
    session_lifetime_seconds: float | None = None


def grant_scope(
    registry: ToolRegistry,
    allow: Sequence[str],
    *,
    allow_all: bool = False,
    pins: Sequence[str] = (),
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Resolve the startup grant. Fails closed on missing, unknown or mismatched scope."""
    if allow_all and allow:
        raise ConfigurationError("use either --allow or --allow-all, not both")
    if not allow_all and not allow:
        raise ConfigurationError("no document scope granted; pass --allow DOC_ID or --allow-all")
    allowed = tuple(registry.versions) if allow_all else tuple(dict.fromkeys(allow))
    unknown = [d for d in allowed if d not in registry.versions]
    if unknown:
        raise ConfigurationError(f"unknown document ids: {', '.join(sorted(unknown))}")
    versions = {d: registry.versions[d] for d in allowed}
    for pin in pins:
        doc_id, sep, version = pin.partition("=")
        if not sep or doc_id not in versions:
            raise ConfigurationError(f"pin must be DOC_ID=VERSION for a granted document: {pin}")
        # Snapshots are content hashes of the loaded index; a stale pin refuses to serve.
        if versions[doc_id] != version:
            raise ConfigurationError(f"pinned snapshot does not match the loaded index: {doc_id}")
    return allowed, versions


def default_limits(budget: Budget | None = None) -> ExecutionLimits:
    budget = budget or Budget()
    return ExecutionLimits(
        max_tool_calls=budget.max_tool_calls,
        max_searches=budget.max_searches,
        max_retries=budget.max_retries,
        max_chunks_per_query=budget.max_chunks_per_query,
        max_evidence_items=budget.max_evidence_items,
        max_evidence_chars=budget.max_evidence_chars,
        reserved_tool_calls=0,
    )


class EvidenceItem(BaseModel):
    evidence_id: str
    doc_id: str
    version_id: str
    chunk_id: str
    document_title: str
    page: int | None
    section: str | None
    section_title: str | None
    text: str
    newly_admitted: bool


class Usage(BaseModel):
    tool_calls: int
    max_tool_calls: int
    searches_used: int
    max_searches: int
    evidence_items: int
    max_evidence_items: int
    evidence_chars: int
    max_evidence_chars: int


class EvidenceResult(BaseModel):
    status: Literal["ok", "empty"]
    lookup_status: str | None = None
    truncated: bool = False
    evidence: list[EvidenceItem]
    # Hits the admission limits refused. Their text is withheld: it cannot be cited.
    withheld_hits: int
    usage: Usage


class DateResult(BaseModel):
    date: date
    inputs: dict[str, Any]
    rule_evidence_ids: list[str]
    derived: bool = True
    limitation: str = (
        "Arithmetic and evidence linkage only. The server does not check that the linked "
        "clause is the governing contractual rule."
    )
    usage: Usage


class CitationCheckResult(BaseModel):
    structurally_valid: bool
    semantic_support: Literal["not_assessed"] = "not_assessed"
    normalized_answer: str
    citations: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    limitation: str = (
        "Checks that each citation resolves to admitted evidence in this session. It does "
        "not judge whether the cited text supports the claim, and clients may skip it."
    )
    usage: Usage


class ScopeDocument(BaseModel):
    doc_id: str
    version_id: str
    title: str


class SessionInfo(BaseModel):
    server_version: str
    documents: list[ScopeDocument]
    call_timeout_seconds: float
    session_lifetime_seconds: float | None
    session_seconds_remaining: float | None
    reserved_final_verification: bool = False
    usage: Usage


def tool_ok(model: BaseModel) -> CallToolResult:
    """A successful result; the SDK validates `structured_content` against the model."""
    payload = model.model_dump(mode="json")
    return CallToolResult(
        content=[TextContent(type="text", text=model.model_dump_json())],
        structured_content=payload,
    )


def tool_error(code: str, message: str, *, retryable: bool = False, **extra: Any) -> CallToolResult:
    """A structured, machine-readable tool failure (MCP `isError` result)."""
    payload = {"error": {"code": code, "retryable": retryable, "message": message}} | extra
    return CallToolResult(
        content=[TextContent(type="text", text=f"{code}: {message}")],
        structured_content=payload,
        is_error=True,
    )


STOP_MESSAGES = {
    "tool_budget_exhausted": "Session tool-call budget is exhausted.",
    "search_budget_exhausted": "Session search budget is exhausted.",
    "deadline_exhausted": "The call deadline expired before dispatch.",
    "tool_scope_violation": "Rejected by scope or provenance checks.",
    "tool_unavailable": "The tool failed; see error code.",
}


class ContractToolServer:
    """Owns the process session and maps MCP tool calls onto the shared executor."""

    def __init__(self, registry: ToolRegistry, config: ServerConfig):
        self.registry = registry
        self.config = config
        self.executor = ToolExecutor(registry)
        self.session = ToolSession(
            allowed=config.allowed, versions=config.versions, limits=config.limits
        )
        self.started_at = monotonic()

    # --- session bookkeeping -------------------------------------------------
    def remaining_lifetime(self) -> float | None:
        if self.config.session_lifetime_seconds is None:
            return None
        return self.config.session_lifetime_seconds - (monotonic() - self.started_at)

    def usage(self) -> Usage:
        s, limits = self.session, self.session.limits
        return Usage(
            tool_calls=s.tool_calls,
            max_tool_calls=limits.max_tool_calls,
            searches_used=s.searches_used,
            max_searches=limits.max_searches,
            evidence_items=len(s.evidence_by_id),
            max_evidence_items=limits.max_evidence_items,
            evidence_chars=s.evidence_chars,
            max_evidence_chars=limits.max_evidence_chars,
        )

    def info(self) -> SessionInfo:
        titles = {d.document_id: d.title for d in self.registry.store.documents}
        return SessionInfo(
            server_version=SERVER_VERSION,
            documents=[
                ScopeDocument(doc_id=d, version_id=self.session.versions[d], title=titles[d])
                for d in self.session.allowed
            ],
            call_timeout_seconds=self.config.call_timeout_seconds,
            session_lifetime_seconds=self.config.session_lifetime_seconds,
            session_seconds_remaining=self.remaining_lifetime(),
            usage=self.usage(),
        )

    # --- dispatch ------------------------------------------------------------
    async def call(
        self, tool: str, arguments: dict[str, Any], rule_evidence_ids: list[str] | None = None
    ) -> tuple[ToolResult | CallToolResult, tuple[str, ...]]:
        remaining = self.remaining_lifetime()
        if remaining is not None and remaining <= 0:
            return (
                tool_error("session_expired", "Session lifetime has ended; start a new process."),
                (),
            )
        # The MCP client supplies no planner reason; a fixed code satisfies the schema.
        action = Action.model_validate(
            {
                "tool": tool,
                "arguments": {k: v for k, v in arguments.items() if v is not None},
                "reason": "missing_evidence",
                "rule_evidence_ids": rule_evidence_ids or [],
            }
        )
        try:
            payload = self.executor.authorize(self.session, action)
        except AuthorizationError as exc:
            message = "Rejected before dispatch by the granted scope."
            return tool_error(str(exc), message, usage=self.usage().model_dump()), ()
        except ValidationError as exc:
            # Field names only: rejected values are the caller's data.
            fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
            message = f"Invalid fields: {', '.join(fields)}"
            return tool_error("invalid_arguments", message, usage=self.usage().model_dump()), ()
        timeout = self.config.call_timeout_seconds
        if remaining is not None:
            timeout = min(timeout, remaining)
        outcome = await self.executor.execute(
            self.session,
            tool,
            payload,
            deadline_at=monotonic() + timeout,
            rule_evidence_ids=action.rule_evidence_ids,
        )
        result = outcome.result
        if outcome.stop_reason:
            code = outcome.error_code or outcome.stop_reason
            if result is not None and result.error is not None:
                code = result.error.code
            error = tool_error(
                code,
                STOP_MESSAGES.get(outcome.stop_reason, "Tool call refused."),
                retryable=bool(result and result.error and result.error.retryable),
                usage=self.usage().model_dump(),
            )
            return error, outcome.admitted
        assert result is not None
        if result.status == "error" and result.error is not None:
            error = tool_error(
                result.error.code,
                result.error.safe_message,
                retryable=result.error.retryable,
                usage=self.usage().model_dump(),
            )
            return error, outcome.admitted
        return result, outcome.admitted

    async def evidence_call(self, tool: str, arguments: dict[str, Any]) -> CallToolResult:
        result, admitted = await self.call(tool, arguments)
        if isinstance(result, CallToolResult):
            return result
        admitted_now = set(admitted)
        items: list[EvidenceItem] = []
        withheld = 0
        for ref in result.evidence_refs:
            if ref.evidence_id not in self.session.evidence_by_id:
                withheld += 1
                continue
            chunk = self.registry.store.get_chunk(ref.chunk_id)
            assert chunk is not None  # provenance was checked against this store
            items.append(
                EvidenceItem(
                    evidence_id=ref.evidence_id,
                    doc_id=ref.doc_id,
                    version_id=ref.version_id,
                    chunk_id=ref.chunk_id,
                    document_title=chunk.document_title,
                    page=ref.page,
                    section=ref.section,
                    section_title=chunk.section_title,
                    text=chunk.text,
                    newly_admitted=ref.evidence_id in admitted_now,
                )
            )
        return tool_ok(
            EvidenceResult(
                status="ok" if items else "empty",
                lookup_status=result.data.get("lookup_status"),
                truncated=bool(result.data.get("truncated", False)),
                evidence=items,
                withheld_hits=withheld,
                usage=self.usage(),
            )
        )


def build_server(tools: ContractToolServer) -> MCPServer:
    server: MCPServer = MCPServer(
        name="contract-research",
        version=SERVER_VERSION,
        instructions=(
            "Read-only tools over a fixed, server-granted set of contract snapshots. Call "
            "session_info first for document ids, version ids and remaining budgets. Cite "
            "only evidence returned by these tools, as [doc_id, p. N, §section]. "
            "verify_citation checks citation structure against admitted evidence only; it "
            "does not establish that the cited text supports a claim."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    async def session_info() -> SessionInfo:
        """Granted documents with their snapshot version ids, limits and remaining budgets."""
        return tools.info()

    @server.tool(annotations=READ_ONLY)
    async def document_search(
        query: str, doc_ids: list[str] | None = None, top_k: int = 5
    ) -> Annotated[CallToolResult, EvidenceResult]:
        """Search granted contracts. doc_ids may narrow, never widen, the granted scope."""
        return await tools.evidence_call(
            "document_search", {"query": query, "doc_ids": doc_ids, "top_k": top_k}
        )

    @server.tool(annotations=READ_ONLY)
    async def retrieve_clause(
        doc_id: str, version_id: str, clause_ref: str
    ) -> Annotated[CallToolResult, EvidenceResult]:
        """Retrieve all chunks of an exact section number, e.g. clause_ref="7.2"."""
        return await tools.evidence_call(
            "retrieve_clause",
            {"doc_id": doc_id, "version_id": version_id, "clause_ref": clause_ref},
        )

    @server.tool(annotations=READ_ONLY)
    async def extract_definition(
        doc_id: str, version_id: str, term: str
    ) -> Annotated[CallToolResult, EvidenceResult]:
        """Find where a defined term is defined ("X" means ...). Ambiguity is reported."""
        return await tools.evidence_call(
            "extract_definition", {"doc_id": doc_id, "version_id": version_id, "term": term}
        )

    @server.tool(annotations=READ_ONLY)
    async def calculate_date(
        anchor_date: date,
        offset: int,
        direction: Literal["before", "after"],
        convention: Literal["calendar_days", "business_days", "holidays"],
        rule_evidence_ids: list[str],
        unit: Literal["days"] = "days",
    ) -> Annotated[CallToolResult, DateResult]:
        """Calendar-day arithmetic linked to admitted evidence stating the governing rule.

        rule_evidence_ids must name evidence already returned in this session.
        Business-day and holiday conventions are rejected as unsupported.
        """
        result, _ = await tools.call(
            "calculate_date",
            {
                "anchor_date": anchor_date.isoformat(),
                "offset": offset,
                "unit": unit,
                "direction": direction,
                "convention": convention,
            },
            rule_evidence_ids,
        )
        if isinstance(result, CallToolResult):
            return result
        return tool_ok(
            DateResult(
                date=date.fromisoformat(result.data["date"]),
                inputs=result.data["inputs"],
                rule_evidence_ids=rule_evidence_ids,
                usage=tools.usage(),
            )
        )

    @server.tool(annotations=READ_ONLY)
    async def verify_citation(
        answer: str, evidence_ids: list[str]
    ) -> Annotated[CallToolResult, CitationCheckResult]:
        """Check that an answer's citations resolve to the named admitted evidence.

        Structural check only; semantic support is not assessed.
        """
        result, _ = await tools.call(
            "verify_citation", {"answer": answer, "evidence_ids": evidence_ids}
        )
        if isinstance(result, CallToolResult):
            return result
        return tool_ok(
            CitationCheckResult(
                structurally_valid=bool(result.data["ok"]),
                normalized_answer=result.data["answer"],
                citations=result.data["citations"],
                errors=result.data["errors"],
                usage=tools.usage(),
            )
        )

    return server


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow", action="append", default=[], metavar="DOC_ID")
    parser.add_argument("--allow-all", action="store_true")
    parser.add_argument("--pin", action="append", default=[], metavar="DOC_ID=VERSION")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--call-timeout", type=float, default=30.0)
    parser.add_argument("--session-lifetime", type=float, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument("--max-searches", type=int, default=None)
    return parser.parse_args(argv)


def create(argv: Sequence[str] | None = None) -> tuple[MCPServer, ContractToolServer]:
    args = parse_args(argv)
    settings = Settings()
    processed = args.processed_dir or settings.processed_dir
    store = ChunkStore.load(processed)
    retriever = build_retriever(args.retriever, store, args.index_dir or settings.index_dir)
    registry = ToolRegistry(store, retriever, CitationVerifier(store))
    allowed, versions = grant_scope(registry, args.allow, allow_all=args.allow_all, pins=args.pin)
    limits = default_limits()
    overrides = {
        k: v
        for k, v in (
            ("max_tool_calls", args.max_tool_calls),
            ("max_searches", args.max_searches),
        )
        if v is not None
    }
    if overrides:
        limits = ExecutionLimits(**{**_fields(limits), **overrides})
    if args.call_timeout <= 0 or (args.session_lifetime is not None and args.session_lifetime <= 0):
        raise ConfigurationError("timeouts must be positive")
    config = ServerConfig(
        allowed=allowed,
        versions=versions,
        limits=limits,
        call_timeout_seconds=args.call_timeout,
        session_lifetime_seconds=args.session_lifetime,
    )
    tools = ContractToolServer(registry, config)
    return build_server(tools), tools


def _fields(limits: ExecutionLimits) -> dict[str, Any]:
    return {name: getattr(limits, name) for name in ExecutionLimits.__slots__}


def main(argv: Sequence[str] | None = None) -> int:
    # stdout carries only protocol frames; every log line goes to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    try:
        server, _ = create(argv)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
