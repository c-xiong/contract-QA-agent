"""Agent state. See docs/SPEC.md section 11.2.

AUTHOR-OWNED (CLAUDE.md rule 2). Every `# DECISION:` is a choice to defend.

SPEC 11.2 specifies the full state for the Sprint 2 agent. This is the Sprint 0
subset: a single search, a single write, a single verification. Fields the loop will
need are listed and commented rather than silently omitted, so the growth path is
visible and nobody has to rediscover what was intended.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, TypedDict

from app.schemas.evidence import Citation, CitationError, Evidence
from app.schemas.retrieval import RetrievedChunk

Status = Literal["planning", "researching", "writing", "completed", "abstained", "failed"]

# SPEC 11.4's failure taxonomy. A closed set, so failures can be counted per category
# across an eval run instead of arriving as unclassifiable free text.
FailureKind = Literal[
    "invalid_tool_input",
    "retrieval_unavailable",
    "no_results",
    "model_timeout",
    "tool_timeout",
    "output_validation",
    "context_limit",
    "citation_verification",
    "budget_exhausted",
    "internal_error",
]


@dataclass(frozen=True, slots=True)
class Budget:
    """Execution limits, enforced in code.

    DECISION: budgets live in the state as data, not as constants read at the call site.
      Chosen so that a run's limits are recorded in its own trace. When an eval result
      is surprising, "what budget was this run given" must be answerable from the
      artifact, not by checking out the commit and reading a config file.
      Sprint 0 uses only `max_searches`, which is 1. The rest are declared now because
      SPEC 11.3 fixes their values, and a budget field added later tends to be added
      after something has already overrun.
    """

    # SPEC 11.3 fixes these values. Sprint 0 ran with max_searches=1 (a single search,
    # no loop); Sprint 2 opens the loop, so the defaults are now the spec's.
    max_searches: int = 3
    max_research_iterations: int = 3
    max_queries_per_iteration: int = 2
    max_chunks_per_query: int = 5
    max_evidence_chars: int = 24_000
    max_repair_attempts: int = 1
    # Cross-reference resolution: depth 1 with its own chunk budget (SPEC 9.5), kept
    # separate from the search budget so a pulled carve-out never costs a search.
    max_cross_reference_depth: int = 1
    max_cross_reference_chunks: int = 4

    def with_searches(self, n: int) -> Budget:
        """Narrow the search budget, leaving every other limit intact."""
        return replace(
            self, max_searches=n, max_research_iterations=min(n, self.max_research_iterations)
        )


@dataclass
class TraceEvent:
    """One thing that happened, in order. The debugging surface for the whole system."""

    step: str
    detail: str
    data: dict[str, object] = field(default_factory=dict)


class ResearchState(TypedDict, total=False):
    """State threaded through the graph.

    DECISION: TypedDict with total=False, not a Pydantic model.
      LangGraph merges partial dictionaries returned by each node; a node returns only
      the keys it changed. A Pydantic model would either have to be reconstructed whole
      at every node or validated against required fields it does not yet have.
      The cost is real and worth naming: no validation on assignment. A node that
      writes a wrong type here fails later and elsewhere. That is why every value in
      this dict is either a primitive or an already-validated Pydantic object -- the
      validation happens at the boundary, in the schemas.
    """

    # --- Input ---
    question: str
    allowed_document_ids: list[str] | None

    # --- Retrieval ---
    search_queries: list[str]
    research_iterations: int
    stop_reason: str | None
    # The sufficiency verdict itself. `stop_reason` is a LABEL and is deliberately not
    # sufficient to route on: "sufficient_evidence" is returned both when the evidence
    # is sufficient (stop) and when the loop is about to search again. Routing on the
    # label made the loop run to budget regardless of the verdict.
    continue_research: bool
    # How many retrieved chunks had already been assessed when `assess` last ran.
    # Chunks past this index are what the most recent search actually added, which is
    # what the "did this iteration make progress?" policy needs. Kept as an index
    # rather than a set of locations so the state stays JSON-serializable for traces.
    assessed_chunk_count: int
    retrieved_chunks: list[RetrievedChunk]
    evidence: list[Evidence]

    # --- Generation ---
    draft_answer: str
    final_answer: str | None
    citations: list[Citation]
    citation_errors: list[CitationError]
    repair_attempts: int

    # --- Control ---
    status: Status
    failure: FailureKind | None
    failure_detail: str | None
    budget: Budget
    searches_used: int
    trace: list[TraceEvent]

    # --- Cost ---
    input_tokens: int
    output_tokens: int

    # Still absent rather than present-and-unused (SPEC 11.2): research_brief,
    # research_topics, messages. An unused field reads as implemented; these arrive with
    # the supervisor extension if it is ever built.


def initial_state(
    question: str,
    *,
    allowed_document_ids: list[str] | None = None,
    budget: Budget | None = None,
) -> ResearchState:
    """Build the starting state. Every list is initialized, so no node handles None."""
    return ResearchState(
        question=question,
        allowed_document_ids=allowed_document_ids,
        search_queries=[],
        research_iterations=0,
        stop_reason=None,
        continue_research=False,
        assessed_chunk_count=0,
        retrieved_chunks=[],
        evidence=[],
        draft_answer="",
        final_answer=None,
        citations=[],
        citation_errors=[],
        repair_attempts=0,
        status="planning",
        failure=None,
        failure_detail=None,
        budget=budget or Budget(),
        searches_used=0,
        trace=[],
        input_tokens=0,
        output_tokens=0,
    )
