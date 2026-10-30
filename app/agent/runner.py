"""Assemble and run the research agent. Shared by scripts/demo.py and the eval runner."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import dataclass
from time import perf_counter
from typing import cast

from app.agent.answerability import AnswerabilityDecision
from app.agent.execution import v2_states
from app.agent.graph import build_graph
from app.agent.llm import ModelClient, build_client
from app.agent.state import Budget, ResearchState, TraceEvent, initial_state
from app.config import Settings
from app.evidence.citation_verifier import CitationVerifier
from app.ingestion.store import ChunkStore
from app.observability.events import index_hash, recover_interrupted
from app.observability.telemetry import configure as configure_telemetry
from app.retrieval.base import Retriever
from app.retrieval.bm25 import Bm25Retriever
from app.schemas.evidence import Citation, CitationError, Evidence
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchResult:
    """Everything a grader or a human needs to judge one run."""

    question: str
    answer: str
    status: str
    citations: list[Citation]
    citation_errors: list[CitationError]
    retrieved: list[RetrievedChunk]
    evidence: list[Evidence]
    trace: list[TraceEvent]
    input_tokens: int
    output_tokens: int
    failure: str | None = None
    failure_detail: str | None = None
    answerability: AnswerabilityDecision | None = None
    model_calls: int = 0
    elapsed_seconds: float = 0.0
    run_id: str | None = None
    trace_path: str | None = None
    tool_calls: int = 0
    stop_reason: str | None = None
    usage_known: bool = True

    @property
    def abstained(self) -> bool:
        return self.status == "abstained"

    @property
    def retrieved_document_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for item in self.retrieved:
            seen.setdefault(item.chunk.document_id, None)
        return list(seen)


class ResearchAgent:
    """One assembled agent over one store. Reused across questions."""

    def __init__(
        self,
        store: ChunkStore,
        settings: Settings,
        retriever: Retriever | None = None,
        *,
        answerability_gate: bool | None = None,
        client: ModelClient | None = None,
    ):
        if settings.agent_mode == "v2":
            configure_telemetry(
                settings.telemetry_export,
                timeout_seconds=settings.telemetry_timeout_s,
                settings=settings,
            )
            recover_interrupted(settings.runs_dir)
        self.store = store
        self.settings = settings
        self.retriever = retriever or Bm25Retriever(store)
        self.verifier = CitationVerifier(store)
        # An injected client lets experiments wrap live calls in a budget ledger.
        self.client = client or build_client(settings)
        self.answerability_gate = (
            settings.answerability_gate if answerability_gate is None else answerability_gate
        )
        self.graph = build_graph(
            self.retriever,
            self.verifier,
            self.client,
            store,
            answerability_gate=self.answerability_gate,
            agent_mode=settings.agent_mode,
            runs_dir=settings.runs_dir,
            index_fingerprint=index_hash(settings.index_dir)
            if settings.agent_mode == "v2"
            else None,
        )

    async def research(
        self,
        question: str,
        *,
        allowed_document_ids: list[str] | None = None,
        budget: Budget | None = None,
        requested_versions: dict[str, str] | None = None,
        progress: Callable[[ResearchState], None] | None = None,
        external_ref: dict[str, str] | None = None,
    ) -> ResearchResult:
        """Run one question to a terminal state.

        `progress` and `external_ref` exist for protocol adapters (app/a2a) and are V2-only.

        DECISION: `progress` is a synchronous observer, called with each merged V2 state
          and never awaited, so a slow consumer cannot spend the run's deadline. If it
          raises, it is logged and disabled; the exception never reaches `v2_states`,
          which would otherwise journal the run as interrupted. Alternative rejected: an
          async callback, which would let a stalled stream consumer stall the graph.
        """
        state = initial_state(
            question,
            allowed_document_ids=allowed_document_ids,
            budget=budget or Budget(max_chunks_per_query=self.settings.retrieval_top_k),
        )
        if requested_versions is not None:
            state["requested_versions"] = requested_versions
        if external_ref:
            state["external_ref"] = dict(external_ref)
        # ainvoke returns the merged state dict; LangGraph types it as dict[str, Any].
        started = perf_counter()
        if self.settings.agent_mode == "v2":
            final = state
            async with aclosing(v2_states(self.graph, state, self.settings.runs_dir)) as states:
                async for merged in states:
                    final = merged
                    if progress is not None:
                        try:
                            progress(merged)
                        except Exception:  # noqa: BLE001 -- an observer never fails a run
                            logger.warning(
                                "Progress observer failed; disabled for run %s",
                                merged.get("run_id"),
                            )
                            progress = None
        else:
            final = cast(
                ResearchState, await self.graph.ainvoke(state, config={"recursion_limit": 100})
            )

        return ResearchResult(
            run_id=final.get("run_id"),
            trace_path=final.get("trace_path"),
            tool_calls=final.get("tool_calls", 0),
            stop_reason=final.get("stop_reason"),
            usage_known=final.get("usage_known", True),
            question=question,
            answer=final.get("final_answer") or "",
            status=final.get("status", "failed"),
            citations=final.get("citations", []),
            citation_errors=final.get("citation_errors", []),
            retrieved=final.get("retrieved_chunks", []),
            evidence=final.get("evidence", []),
            trace=final.get("trace", []),
            input_tokens=final.get("input_tokens", 0),
            output_tokens=final.get("output_tokens", 0),
            failure=final.get("failure"),
            failure_detail=final.get("failure_detail"),
            answerability=final.get("answerability"),
            model_calls=final.get("model_calls", 0),
            elapsed_seconds=perf_counter() - started,
        )


def load_agent(settings: Settings) -> ResearchAgent:
    return ResearchAgent(ChunkStore.load(settings.processed_dir), settings)
