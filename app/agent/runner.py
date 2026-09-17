"""Assemble and run the research agent. Shared by scripts/demo.py and the eval runner."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import cast

from app.agent.answerability import AnswerabilityDecision
from app.agent.graph import build_graph
from app.agent.llm import build_client
from app.agent.state import Budget, ResearchState, TraceEvent, initial_state
from app.config import Settings
from app.evidence.citation_verifier import CitationVerifier
from app.ingestion.store import ChunkStore
from app.retrieval.base import Retriever
from app.retrieval.bm25 import Bm25Retriever
from app.schemas.evidence import Citation, CitationError, Evidence
from app.schemas.retrieval import RetrievedChunk


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
    ):
        self.store = store
        self.settings = settings
        self.retriever = retriever or Bm25Retriever(store)
        self.verifier = CitationVerifier(store)
        self.client = build_client(settings)
        self.answerability_gate = (
            settings.answerability_gate if answerability_gate is None else answerability_gate
        )
        self.graph = build_graph(
            self.retriever,
            self.verifier,
            self.client,
            store,
            answerability_gate=self.answerability_gate,
        )

    async def research(
        self,
        question: str,
        *,
        allowed_document_ids: list[str] | None = None,
        budget: Budget | None = None,
    ) -> ResearchResult:
        state = initial_state(
            question,
            allowed_document_ids=allowed_document_ids,
            budget=budget or Budget(max_chunks_per_query=self.settings.retrieval_top_k),
        )
        # ainvoke returns the merged state dict; LangGraph types it as dict[str, Any].
        started = perf_counter()
        final = cast(ResearchState, await self.graph.ainvoke(state))

        return ResearchResult(
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
