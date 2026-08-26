"""FastAPI application. See docs/SPEC.md section 14.

Thin by design. The API validates input, resolves an agent, and renders the result;
every decision that matters lives in the layers beneath it.

The corpus and indices load once at startup rather than per request: BM25 index
construction and the FAISS load both take seconds, and doing that per request would
make the API's latency a property of the index rather than of the question.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent.llm import ModelError
from app.agent.runner import ResearchAgent
from app.api.schemas import (
    DocumentSummary,
    HealthResponse,
    ResearchRequest,
    ResearchResponse,
    TraceItem,
    render_citations,
    render_evidence,
)
from app.api.stream import research_events
from app.config import Settings, get_settings
from app.ingestion.store import ChunkStore, StoreError
from app.retrieval.factory import ARM_NAMES, UnknownArmError, build_retriever

STATIC_DIR = Path(__file__).parent / "static"

logger = logging.getLogger(__name__)


def build_agent(settings: Settings, arm: str = "bm25") -> tuple[ResearchAgent, str | None]:
    """Load the corpus and assemble an agent. Returns (agent, degradation message).

    Two degradations are operational states rather than reasons to refuse to start, and
    both are reported instead of being silently absorbed:

    - A missing FAISS index (nobody ran build_index.py) falls back to BM25.
    - CRA_LIVE_MODEL=1 with no API key falls back to the stub. Crashing here would give
      an orchestrator a traceback and the operator no idea which of two env vars is
      wrong; the inspector shows the message instead.
    """
    store = ChunkStore.load(settings.processed_dir)
    note: str | None = None

    try:
        retriever = build_retriever(arm, store, settings.index_dir)
    except (UnknownArmError, RuntimeError) as exc:
        note = f"Retrieval arm {arm!r} unavailable, using bm25: {exc}"
        logger.warning(note)
        retriever = build_retriever("bm25", store, settings.index_dir)

    try:
        return ResearchAgent(store, settings, retriever=retriever), note
    except ModelError as exc:
        note = f"Live model unavailable, using the deterministic stub: {exc}"
        logger.warning(note)
        stubbed = settings.model_copy(update={"live_model": False})
        return ResearchAgent(store, stubbed, retriever=retriever), note


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # An agent injected before startup wins. Tests assemble a small synthetic corpus and
    # attach it directly; without this the lifespan would load the real 56-document
    # corpus over the top and every assertion about the fixture would fail confusingly.
    if getattr(app.state, "agent", None) is not None:
        logger.info("Using pre-injected agent; skipping corpus load")
        app.state.degraded = getattr(app.state, "degraded", None)
        yield
        return

    try:
        app.state.agent, app.state.degraded = build_agent(settings, arm=app.state.retrieval_arm)
        logger.info(
            "Loaded %d documents, %d chunks, retriever=%s",
            len(app.state.agent.store.documents),
            len(app.state.agent.store),
            app.state.agent.retriever.name,
        )
    except StoreError as exc:
        # Start anyway and report unhealthy. A container that exits on a missing corpus
        # gives an orchestrator a crash loop and no diagnostic; one that starts and says
        # what is wrong is debuggable.
        logger.error("No ingested corpus: %s", exc)
        app.state.agent = None
        app.state.degraded = str(exc)
    yield


def agent_for_arm(app: FastAPI, arm: str) -> ResearchAgent:
    """Return an agent using `arm`, building and caching it on first use.

    DECISION: arms are built lazily and cached, not all built at startup.
      Building every arm eagerly costs ~30 s of boot (the cross-encoder and the FAISS
      load dominate) even when the session only ever uses BM25. Lazy keeps startup fast
      for the common case and pays the cost once, on the first request that asks for a
      heavier arm -- which is the right trade for a tool used to poke at one question at
      a time.
    """
    base: ResearchAgent | None = getattr(app.state, "agent", None)
    if base is None:
        raise HTTPException(
            status_code=503, detail="No ingested corpus. Run scripts/ingest.py and restart."
        )
    if arm == base.retriever.name or arm not in ARM_NAMES:
        return base

    cache: dict[str, ResearchAgent] = app.state.arm_cache
    if arm not in cache:
        settings = get_settings()
        try:
            retriever = build_retriever(arm, base.store, settings.index_dir)
        except (UnknownArmError, RuntimeError) as exc:
            # A stale or missing index is an operational state, not a client error.
            # Refusing the request would make the inspector unusable rather than
            # degraded, and the operator would see a 400 instead of the reason.
            logger.warning("Cannot build arm %r, using %s: %s", arm, base.retriever.name, exc)
            return base
        cache[arm] = ResearchAgent(base.store, settings, retriever=retriever)
    return cache[arm]


def create_app(retrieval_arm: str = "rrf_hybrid") -> FastAPI:
    app = FastAPI(
        title="Eval-Driven Contract Research Agent",
        description=(
            "Answers questions about commercial contracts with document and page "
            "citations, and abstains when the evidence does not support an answer."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.retrieval_arm = retrieval_arm
    app.state.arm_cache = {}
    app.state.degraded = None

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            """The trace inspector. Read-only; see CLAUDE.md rule 4."""
            return FileResponse(STATIC_DIR / "index.html")

    def require_agent(request: Request) -> ResearchAgent:
        agent: ResearchAgent | None = getattr(request.app.state, "agent", None)
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="No ingested corpus. Run scripts/ingest.py and restart.",
            )
        return agent

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        agent: ResearchAgent | None = getattr(request.app.state, "agent", None)
        settings = get_settings()
        degraded = getattr(request.app.state, "degraded", None)
        if agent is None:
            return HealthResponse(
                status="no_corpus",
                documents=0,
                chunks=0,
                retriever="none",
                model_id=settings.model_id,
                live_model=False,
                note=degraded,
            )
        return HealthResponse(
            status="degraded" if degraded else "ok",
            documents=len(agent.store.documents),
            chunks=len(agent.store),
            retriever=agent.retriever.name,
            model_id=agent.client.model_id,
            # The agent's own settings, not the process settings: they differ exactly
            # when the live model was requested and could not be built.
            live_model=agent.settings.live_model,
            note=degraded,
        )

    @app.get("/documents", response_model=list[DocumentSummary])
    async def documents(request: Request) -> list[DocumentSummary]:
        agent = require_agent(request)
        return [
            DocumentSummary(
                document_id=d.document_id,
                title=d.title,
                agreement_type=d.agreement_type,
                page_count=d.page_count,
                corpus_source=d.corpus_source,
                is_synthetic=d.is_synthetic,
                chunk_count=len(agent.store.chunks_for(d.document_id)),
            )
            for d in agent.store.documents
        ]

    @app.get("/documents/{document_id}", response_model=DocumentSummary)
    async def document(document_id: str, request: Request) -> DocumentSummary:
        agent = require_agent(request)
        found = agent.store.get_document(document_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"No such document: {document_id}")
        return DocumentSummary(
            document_id=found.document_id,
            title=found.title,
            agreement_type=found.agreement_type,
            page_count=found.page_count,
            corpus_source=found.corpus_source,
            is_synthetic=found.is_synthetic,
            chunk_count=len(agent.store.chunks_for(document_id)),
        )

    @app.post("/research", response_model=ResearchResponse)
    async def research(payload: ResearchRequest, request: Request) -> ResearchResponse:
        agent = require_agent(request)

        # An allowlist naming documents that do not exist is a client error, reported as
        # one. Silently dropping them would run the query against the whole corpus and
        # return an answer the caller believes was scoped.
        if payload.document_ids:
            unknown = [d for d in payload.document_ids if not agent.store.has_document(d)]
            if unknown:
                raise HTTPException(status_code=400, detail=f"Unknown document_ids: {unknown}")

        result = await agent.research(payload.question, allowed_document_ids=payload.document_ids)
        titles = {d.document_id: d.title for d in agent.store.documents}

        return ResearchResponse(
            question=result.question,
            answer=result.answer,
            status=result.status,
            abstained=result.abstained,
            citations=render_citations(result, titles),
            citation_errors=[f"{e.code}: {e.detail}" for e in result.citation_errors],
            evidence=render_evidence(result.evidence),
            documents_searched=result.retrieved_document_ids,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            model_id=agent.client.model_id,
            trace=(
                [TraceItem(step=e.step, detail=e.detail) for e in result.trace]
                if payload.include_trace
                else None
            ),
        )

    @app.get("/arms", response_model=list[str])
    async def arms() -> list[str]:
        return list(ARM_NAMES)

    @app.get("/research/stream", include_in_schema=False)
    async def research_stream(
        request: Request,
        question: str = Query(min_length=1, max_length=2000),
        document_ids: str | None = Query(default=None),
        arm: str | None = Query(default=None),
    ) -> StreamingResponse:
        """Stream one research run as server-sent events.

        Read-only, and the only endpoint the inspector calls.
        """
        agent = agent_for_arm(request.app, arm or request.app.state.retrieval_arm)

        allowed = (
            [d.strip() for d in document_ids.split(",") if d.strip()] if document_ids else None
        )
        if allowed:
            unknown = [d for d in allowed if not agent.store.has_document(d)]
            if unknown:
                raise HTTPException(status_code=400, detail=f"Unknown document_ids: {unknown}")

        return StreamingResponse(
            research_events(agent, question, allowed_document_ids=allowed),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Without this, a reverse proxy will buffer the whole stream and the
                # page shows nothing until the run finishes -- which defeats the point.
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
