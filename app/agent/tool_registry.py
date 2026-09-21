"""V2 read-only tools; schema validation and scope checks precede dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import date, timedelta
from itertools import pairwise
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.evidence.citation_verifier import CitationVerifier
from app.ingestion.store import ChunkStore
from app.retrieval.base import Retriever
from app.schemas.chunk import Chunk
from app.schemas.evidence import CitationError
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk

TOOL_VERSION = "v2-tools-1"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Search(Contract):
    query: str = Field(min_length=1, max_length=2000)
    doc_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1)


class Clause(Contract):
    doc_id: str
    version_id: str
    clause_ref: str = Field(min_length=1, max_length=100)


class Definition(Contract):
    doc_id: str
    version_id: str
    term: str = Field(min_length=1, max_length=200)


class DateCalculation(Contract):
    anchor_date: date
    offset: int = Field(ge=0, le=365000)
    unit: Literal["days"]
    direction: Literal["before", "after"]
    convention: Literal["calendar_days", "business_days", "holidays"]


class CitationCheck(Contract):
    answer: str
    evidence_ids: list[str]


class Action(Contract):
    tool: Literal[
        "document_search",
        "retrieve_clause",
        "extract_definition",
        "calculate_date",
        "verify_citation",
        "draft",
        "abstain",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: Literal[
        "initial_search",
        "missing_evidence",
        "resolve_reference",
        "resolve_definition",
        "compare_versions",
        "compute_deadline",
        "check_citations",
        "ready",
        "insufficient_evidence",
    ]
    rule_evidence_ids: list[str] = Field(default_factory=list)


class EvidenceRef(Contract):
    evidence_id: str
    doc_id: str
    version_id: str
    chunk_id: str
    page: int | None
    section: str | None
    # DECISION: offsets are chunk-relative, because ingestion has no source offsets.
    start_offset: int = 0
    end_offset: int
    text_hash: str


class ToolError(Contract):
    code: str
    retryable: bool = False
    safe_message: str


class ToolResult(Contract):
    call_id: str = Field(default_factory=lambda: uuid4().hex)
    status: Literal["ok", "empty", "error"]
    data: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    error: ToolError | None = None
    elapsed_ms: float = 0
    cache_hit: bool = False
    result_hash: str = ""


SCHEMAS: dict[str, type[Contract]] = {
    "document_search": Search,
    "retrieve_clause": Clause,
    "extract_definition": Definition,
    "calculate_date": DateCalculation,
    "verify_citation": CitationCheck,
}


class AuthorizationError(ValueError):
    """Rejected before executing a tool or admitting evidence."""


class ToolRegistry:
    def __init__(self, store: ChunkStore, retriever: Retriever, verifier: CitationVerifier):
        self.store, self.retriever, self.verifier = store, retriever, verifier
        # DECISION: metadata lacks legal version labels. Identify indexed snapshots,
        # never infer v1/v2 or legal precedence from filenames or model arguments.
        self.versions = {
            d.document_id: "sha256:"
            + digest([c.model_dump(mode="json") for c in store.chunks_for(d.document_id)])
            for d in store.documents
        }

    def ref(self, chunk: Chunk) -> EvidenceRef:
        text_hash = digest(chunk.text)
        return EvidenceRef(
            evidence_id="e-" + digest([chunk.chunk_id, text_hash])[:20],
            doc_id=chunk.document_id,
            version_id=self.versions[chunk.document_id],
            chunk_id=chunk.chunk_id,
            page=chunk.page_number,
            section=chunk.section_id,
            end_offset=len(chunk.text),
            text_hash=text_hash,
        )

    def validate(
        self, action: Action, allowed: list[str], versions: dict[str, str], top_k: int
    ) -> Contract:
        schema = SCHEMAS[action.tool]
        payload = schema.model_validate(action.arguments)
        if isinstance(payload, Search):
            ids = payload.doc_ids if payload.doc_ids is not None else allowed
            if not set(ids) <= set(allowed):
                raise AuthorizationError("document_not_allowed")
            return payload.model_copy(
                update={"doc_ids": ids, "top_k": min(payload.top_k, top_k, 10)}
            )
        if isinstance(payload, (Clause, Definition)):
            if payload.doc_id not in allowed:
                raise AuthorizationError("document_not_allowed")
            if payload.version_id != versions.get(payload.doc_id):
                raise AuthorizationError("version_not_allowed")
        return payload

    def verify(
        self, answer: str, retrieved: list[RetrievedChunk], allowed: list[str]
    ) -> dict[str, Any]:
        verdict = self.verifier.verify(answer, retrieved, allowed_document_ids=allowed)
        errors = list(verdict.errors)
        for citation in verdict.citations:
            if not any(
                citation.document_id == r.document_id
                and (citation.page_number is None or citation.page_number == r.chunk.page_number)
                and (citation.section_id is None or citation.section_id in r.chunk.section_path)
                for r in retrieved
            ):
                errors.append(
                    CitationError(
                        code="not_in_evidence",
                        raw=citation.raw,
                        citation=citation,
                        detail="The complete locator is absent from admitted evidence.",
                    )
                )
        return {
            "ok": bool(answer.strip() and verdict.citations) and not errors,
            "answer": verdict.normalized_answer or answer,
            "citations": [c.model_dump(mode="json") for c in verdict.citations],
            "errors": [e.model_dump(mode="json") for e in errors],
        }

    async def execute(
        self,
        payload: Contract,
        *,
        allowed: list[str],
        versions: dict[str, str],
        evidence: dict[str, EvidenceRef],
        retrieved: list[RetrievedChunk],
        call_id: str,
    ) -> ToolResult:
        started = perf_counter()
        hits: list[RetrievedChunk] = []
        data: dict[str, Any] = {}
        status: Literal["ok", "empty", "error"] = "ok"
        if isinstance(payload, Search):
            # Native BM25/FAISS/reranking work must not block the event loop's deadline.
            # Cancelled workers can finish read-only work; they never mutate run state.
            def search() -> list[RetrievedChunk]:
                return asyncio.run(
                    self.retriever.search(
                        payload.query,
                        top_k=payload.top_k,
                        filters=RetrievalFilters(document_ids=payload.doc_ids),
                    )
                )

            hits = await asyncio.to_thread(search)
            hits = hits[: payload.top_k]
        elif isinstance(payload, Clause):
            chunks = self.store.find_section(payload.doc_id, payload.clause_ref)
            # All split chunks of an exact section are returned within a fixed bound.
            # Reused section labels on disjoint pages are explicitly ambiguous.
            pages = sorted({c.page_number for c in chunks if c.page_number is not None})
            ambiguous = any(b - a > 1 for a, b in pairwise(pages))
            data = {
                "lookup_status": "ambiguous" if ambiguous else "ok" if chunks else "not_found",
                "truncated": len(chunks) > 5,
            }
            hits = [
                RetrievedChunk(chunk=c, fused_score=0, retrieval_query=payload.clause_ref)
                for c in chunks[:5]
            ]
        elif isinstance(payload, Definition):
            pattern = re.compile(
                r'["\u201c\u201d\']?'
                + re.escape(payload.term)
                + r'["\u201c\u201d\']?\s+(?:means|shall mean|has the meaning|is defined)',
                re.I,
            )
            chunks = [c for c in self.store.chunks_for(payload.doc_id) if pattern.search(c.text)]
            data = {
                "lookup_status": "ambiguous"
                if len(chunks) > 1
                else "ok"
                if chunks
                else "not_found",
                "truncated": len(chunks) > 5,
            }
            hits = [
                RetrievedChunk(chunk=c, fused_score=0, retrieval_query=payload.term)
                for c in chunks[:5]
            ]
        elif isinstance(payload, DateCalculation):
            if payload.convention != "calendar_days":
                return ToolResult(
                    call_id=call_id,
                    status="error",
                    error=ToolError(
                        code="unsupported_convention",
                        safe_message="Business-day and holiday calendars are unsupported.",
                    ),
                )
            try:
                value = payload.anchor_date + timedelta(
                    days=payload.offset * (1 if payload.direction == "after" else -1)
                )
            except OverflowError:
                return ToolResult(
                    call_id=call_id,
                    status="error",
                    error=ToolError(
                        code="date_out_of_range",
                        safe_message="Date is outside the supported range.",
                    ),
                )
            data = {
                "date": value.isoformat(),
                "inputs": payload.model_dump(mode="json"),
                "derived": True,
            }
        elif isinstance(payload, CitationCheck):
            if not set(payload.evidence_ids) <= set(evidence):
                raise AuthorizationError("unknown_evidence")
            ids = {evidence[e].chunk_id for e in payload.evidence_ids}
            data = self.verify(payload.answer, [r for r in retrieved if r.chunk_id in ids], allowed)
        for hit in hits:
            if hit.document_id not in allowed or self.versions[hit.document_id] != versions.get(
                hit.document_id
            ):
                raise AuthorizationError("retriever_scope_violation")
            if self.store.get_chunk(hit.chunk_id) != hit.chunk:
                raise AuthorizationError("untrusted_evidence")
        refs = [self.ref(h.chunk) for h in hits]
        if isinstance(payload, (Search, Clause, Definition)):
            status = "ok" if hits else "empty"
            data["hits"] = [h.model_dump(mode="json") for h in hits]
            data["candidate_ids"] = [r.evidence_id for r in refs]
        return ToolResult(
            call_id=call_id,
            status=status,
            data=data,
            evidence_refs=refs,
            elapsed_ms=(perf_counter() - started) * 1000,
            result_hash=digest({"data": data, "refs": [r.model_dump() for r in refs]}),
        )
