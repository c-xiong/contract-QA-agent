"""The LangGraph research workflow. See docs/decisions.md

AUTHOR-OWNED (CLAUDE.md rule 2). Every `# DECISION:` is a choice to defend.

The Sprint 2 workflow:

    search ──▶ assess ──┬─(insufficient)─▶ refine ──▶ search      (bounded loop)
                        │
                        └─(stop)─▶ resolve_refs ──▶ select_evidence ──▶ answerability
                                                                         │
                                                          answer/partial ─┴─▶ write
                                                          unanswerable ────▶ END
                                                                          │
                                        ┌─────────────────────────────────┘
                                        ▼
                                     verify ──┬─(ok)──────▶ finalize
                                              ├─(fixable)─▶ repair ──▶ write
                                              └─(no)──────▶ abstain

What makes this agentic rather than a pipeline is `assess`: the agent decides whether
what it has is enough, and searches again with a widened query when it is not. Every
bound on that loop lives in `policies.py` as deterministic code, never as a prompt
instruction (CLAUDE.md rule 5).

`resolve_refs` is the SPEC 9.5 step and the reason this system can answer a question
about a liability cap correctly: it follows "subject to Section 8.3" from the retrieved
cap into the carve-out that qualifies it, which no amount of citation checking would
otherwise catch.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.answerability import (
    ANSWERABILITY_SYSTEM,
    ANSWERABILITY_VERSION,
    AnswerabilityError,
    build_answerability_prompt,
    render_abstention,
    validate_answerability,
)
from app.agent.llm import ModelClient, ModelError, StubClient
from app.agent.policies import locations, refine_query, should_continue_research
from app.agent.prompts import REPAIR_SYSTEM, WRITER_SYSTEM, build_repair_prompt, build_writer_prompt
from app.agent.state import ResearchState, TraceEvent
from app.agent.tools import SearchContractsInput, search_contracts
from app.evidence.citation_verifier import CitationVerifier
from app.evidence.cross_reference import resolve_cross_references
from app.evidence.normalizer import normalize_evidence
from app.ingestion.store import ChunkStore
from app.retrieval.base import Retriever
from app.schemas.evidence import Evidence
from app.schemas.retrieval import RetrievedChunk


def _trace(state: ResearchState, step: str, detail: str, **data: object) -> list[TraceEvent]:
    return [*state.get("trace", []), TraceEvent(step=step, detail=detail, data=data)]


def to_evidence(retrieved: list[RetrievedChunk], max_chars: int) -> list[Evidence]:
    """Convert retrieved chunks into evidence, deduplicated and budget-bounded.

    DECISION: deduplicate on (document_id, page_number, section_id), not on chunk_id.
      Two chunks from one mechanically-split long section are different chunk_ids but
      the same citable location. Showing the writer both wastes context and invites it
      to cite the same place twice as if it were two sources.
      Rejected: deduplicating on chunk_id only, which is a no-op given the retriever
      already returns distinct chunks.

    DECISION: the character budget truncates the evidence list, never an excerpt.
      A half-sentence excerpt is worse than an absent one: the writer cannot tell that
      it was cut, and a liability cap truncated before its carve-out reads as
      unqualified. Dropping whole items keeps every excerpt honest, at the cost of
      showing fewer. SPEC 13.2's "trim excerpts to the relevant span" is a Sprint 2
      concern, and it must trim to a *complete* span.
    """
    evidence: list[Evidence] = []
    seen: set[tuple[str, int | None, str | None]] = set()
    used = 0

    for index, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        key = (chunk.document_id, chunk.page_number, chunk.section_id)
        if key in seen:
            continue

        if used + len(chunk.text) > max_chars and evidence:
            break
        seen.add(key)
        used += len(chunk.text)

        evidence.append(
            Evidence(
                evidence_id=f"e{index:02d}",
                topic=item.retrieval_query,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                section_path=chunk.section_path,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                excerpt=chunk.text,
                source_chunk_id=chunk.chunk_id,
                pulled_by=item.pulled_by,
            )
        )
    return evidence


ResearchGraph = CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]


def build_graph(
    retriever: Retriever,
    verifier: CitationVerifier,
    client: ModelClient,
    store: ChunkStore | None = None,
    *,
    answerability_gate: bool = False,
) -> ResearchGraph:
    """Compile the research workflow.

    `store` is optional only so Sprint 0 call sites keep working; without it,
    cross-reference resolution is skipped and the trace says so rather than silently
    doing nothing.
    """

    async def search(state: ResearchState) -> ResearchState:
        budget = state["budget"]

        # DECISION: the search budget is checked in code before the call, not enforced
        # by asking the model to search a bounded number of times.
        #   CLAUDE.md rule 5. A model instructed to stop is a model that usually stops.
        #   The point of this project is the difference between "usually" and "always".
        if state.get("searches_used", 0) >= budget.max_searches:
            return ResearchState(
                status="failed",
                failure="budget_exhausted",
                failure_detail=f"search budget of {budget.max_searches} already spent",
                trace=_trace(state, "search", "budget exhausted"),
            )

        queries = state.get("search_queries", []) or [state["question"]]
        query = queries[-1]
        try:
            output = await search_contracts(
                SearchContractsInput(query=query, top_k=budget.max_chunks_per_query),
                retriever,
                allowed_document_ids=state.get("allowed_document_ids"),
            )
        except (ValueError, RuntimeError) as exc:
            return ResearchState(
                status="failed",
                failure="retrieval_unavailable",
                failure_detail=str(exc),
                trace=_trace(state, "search", f"retriever raised: {exc}"),
            )

        # Accumulate across iterations, deduplicating on chunk id. Later iterations add
        # to the evidence pool rather than replacing it -- a refined query that finds
        # something new must not discard what the first query found.
        existing = state.get("retrieved_chunks", [])
        known = {c.chunk.chunk_id for c in existing}
        merged = existing + [c for c in output.results if c.chunk.chunk_id not in known]

        return ResearchState(
            search_queries=queries,
            retrieved_chunks=merged,
            searches_used=state.get("searches_used", 0) + 1,
            status="researching",
            trace=_trace(
                state,
                "search",
                f"{len(output.results)} chunks ({len(merged)} total)",
                query=query,
            ),
        )

    async def assess(state: ResearchState) -> ResearchState:
        """Decide whether the evidence so far is enough. Deterministic; see policies.py."""
        retrieved = state.get("retrieved_chunks", [])
        boundary = state.get("assessed_chunk_count", 0)
        already_seen, newly_added = retrieved[:boundary], retrieved[boundary:]

        verdict = should_continue_research(
            state, new_chunks=newly_added, previous_locations=locations(already_seen)
        )
        return ResearchState(
            research_iterations=state.get("research_iterations", 0) + 1,
            stop_reason=verdict.reason,
            continue_research=verdict.should_continue,
            assessed_chunk_count=len(retrieved),
            trace=_trace(
                state,
                "assess",
                f"{'continue' if verdict.should_continue else 'stop'}: {verdict.detail}",
            ),
        )

    async def refine(state: ResearchState) -> ResearchState:
        """Widen the query deterministically. See policies.refine_query."""
        queries = state.get("search_queries", [])
        widened = refine_query(state["question"], queries)
        if widened is None:
            return ResearchState(
                stop_reason="no_new_evidence",
                trace=_trace(state, "refine", "no new query available; stopping"),
            )
        return ResearchState(
            search_queries=[*queries, widened],
            trace=_trace(state, "refine", f"widened query: {widened!r}"),
        )

    async def resolve_refs(state: ResearchState) -> ResearchState:
        """Pull sections that the retrieved evidence points at. SPEC 9.5."""
        retrieved = state.get("retrieved_chunks", [])
        if store is None or not retrieved:
            return ResearchState(
                trace=_trace(
                    state, "resolve_refs", "skipped (no store)" if store is None else "no evidence"
                )
            )

        budget = state["budget"]
        report = resolve_cross_references(
            retrieved,
            store,
            max_depth=budget.max_cross_reference_depth,
            max_chunks=budget.max_cross_reference_chunks,
        )
        return ResearchState(
            retrieved_chunks=retrieved + report.pulled,
            trace=_trace(
                state,
                "resolve_refs",
                report.summary(),
                followed=[f"{a} -> §{b}" for a, b in report.followed],
                unresolved=[f"{a} §{b}" for a, b in report.skipped_unresolved],
            ),
        )

    async def select_evidence(state: ResearchState) -> ResearchState:
        retrieved = state.get("retrieved_chunks", [])

        # DECISION: no results is an abstention, not a failure, and not a model call.
        #   "No contract in the corpus addresses this" is a correct and useful answer.
        #   Calling the model with zero evidence would produce a fluent answer from
        #   parametric knowledge about contracts in general -- exactly the ungrounded
        #   output the whole system exists to prevent -- and would cost money to do it.
        if not retrieved:
            return ResearchState(
                status="abstained",
                final_answer=(
                    f"I found no relevant passages for: {state['question']}\n\n"
                    f"Searched {len(state.get('search_queries', []))} query "
                    f"against the indexed corpus and no passage matched."
                ),
                trace=_trace(state, "select_evidence", "no results; abstaining"),
            )

        evidence = normalize_evidence(retrieved, max_chars=state["budget"].max_evidence_chars)
        pulled = sum(1 for e in evidence if e.pulled_by == "cross_reference")
        return ResearchState(
            evidence=evidence,
            status="writing",
            trace=_trace(
                state,
                "select_evidence",
                f"{len(evidence)} evidence items ({pulled} via cross-reference)",
            ),
        )

    async def answerability(state: ResearchState) -> ResearchState:
        evidence = state.get("evidence", [])
        active = client.with_evidence(evidence) if isinstance(client, StubClient) else client
        calls = state.get("model_calls", 0) + 1
        try:
            response = await active.complete(
                ANSWERABILITY_SYSTEM, build_answerability_prompt(state["question"], evidence)
            )
        except ModelError as exc:
            return ResearchState(
                status="failed",
                failure="model_timeout" if "exceeded" in str(exc) else "internal_error",
                failure_detail=str(exc),
                model_calls=calls,
                trace=_trace(state, "answerability", f"model error: {exc}"),
            )
        usage = ResearchState(
            input_tokens=state.get("input_tokens", 0) + response.input_tokens,
            output_tokens=state.get("output_tokens", 0) + response.output_tokens,
            model_calls=calls,
        )
        try:
            decision = validate_answerability(response.text, evidence)
        except AnswerabilityError as exc:
            # DECISION: invalid output is a failure, not a correct abstention. Returning
            # abstained here would reward a broken judge on every unanswerable eval task.
            return ResearchState(
                **usage,
                status="failed",
                failure="output_validation",
                failure_detail=str(exc),
                trace=_trace(state, "answerability", f"invalid verdict: {exc}"),
            )
        update = ResearchState(
            **usage,
            answerability=decision,
            trace=_trace(
                state,
                "answerability",
                f"{decision.verdict}: {decision.reason}",
                version=ANSWERABILITY_VERSION,
                verdict=decision.verdict,
                simulated=isinstance(client, StubClient),
            ),
        )
        if decision.verdict == "unanswerable":
            update["status"] = "abstained"
            update["final_answer"] = render_abstention(state["question"], decision, evidence)
        return update

    async def write(state: ResearchState) -> ResearchState:
        evidence = state.get("evidence", [])
        active = client.with_evidence(evidence) if isinstance(client, StubClient) else client

        repairing = state.get("repair_attempts", 0) > 0
        if repairing:
            errors = [e.detail for e in state.get("citation_errors", [])]
            system, user = (
                REPAIR_SYSTEM,
                build_repair_prompt(
                    state["question"], evidence, errors, state.get("answerability")
                ),
            )
        else:
            system, user = (
                WRITER_SYSTEM,
                build_writer_prompt(state["question"], evidence, state.get("answerability")),
            )

        try:
            response = await active.complete(system, user)
        except ModelError as exc:
            return ResearchState(
                status="failed",
                failure="model_timeout" if "exceeded" in str(exc) else "internal_error",
                failure_detail=str(exc),
                model_calls=state.get("model_calls", 0) + 1,
                trace=_trace(state, "write", f"model error: {exc}"),
            )

        return ResearchState(
            draft_answer=response.text,
            input_tokens=state.get("input_tokens", 0) + response.input_tokens,
            output_tokens=state.get("output_tokens", 0) + response.output_tokens,
            model_calls=state.get("model_calls", 0) + 1,
            # "rewrite", not "repair": the repair node already emits a "repair" event,
            # and two different steps sharing one trace label makes a trace unreadable
            # exactly when it is needed -- counting attempts during a failure.
            trace=_trace(
                state,
                "rewrite" if repairing else "write",
                f"{response.output_tokens} output tokens",
                model=response.model_id,
            ),
        )

    async def verify(state: ResearchState) -> ResearchState:
        result = verifier.verify(
            state.get("draft_answer", ""),
            state.get("retrieved_chunks", []),
            allowed_document_ids=state.get("allowed_document_ids"),
        )
        return ResearchState(
            # The verifier deterministically canonicalizes safe mechanical near-misses
            # such as two complete locators joined by a semicolon in one bracket.  Keep
            # that normalized text so the final answer and inspector chips agree with
            # the citations that actually passed the gate.
            draft_answer=(
                result.normalized_answer
                if result.normalized_answer is not None
                else state.get("draft_answer", "")
            ),
            citations=result.citations,
            citation_errors=result.errors,
            trace=_trace(state, "verify", result.summary()),
        )

    async def finalize(state: ResearchState) -> ResearchState:
        return ResearchState(
            final_answer=state.get("draft_answer", ""),
            status="completed",
            trace=_trace(state, "finalize", "citations verified"),
        )

    async def repair(state: ResearchState) -> ResearchState:
        return ResearchState(
            repair_attempts=state.get("repair_attempts", 0) + 1,
            trace=_trace(state, "repair", "retrying with citation errors fed back"),
        )

    async def abstain(state: ResearchState) -> ResearchState:
        # DECISION: an abstention states what was searched and why the evidence was
        # insufficient. A bare "I don't know" is a bug (SPEC 13.5).
        #   The failed citations are named explicitly. If the system abstained because
        #   the model cited a document it was never shown, the person reading the output
        #   should be able to see that, not be told the corpus was unhelpful.
        errors = state.get("citation_errors", [])
        detail = "\n".join(f"  - {e.code}: {e.detail}" for e in errors[:5])
        return ResearchState(
            status="abstained",
            failure="citation_verification",
            final_answer=(
                f"I could not produce a verifiably cited answer to: {state['question']}\n\n"
                f"Evidence was retrieved, but the citations in the draft answer did not "
                f"pass verification after {state.get('repair_attempts', 0)} repair "
                f"attempt(s):\n{detail}"
            ),
            trace=_trace(state, "abstain", f"{len(errors)} unresolved citation errors"),
        )

    async def fail(state: ResearchState) -> ResearchState:
        """Terminal node for hard failures, distinct from abstention.

        DECISION: a hard failure is not an abstention, and must not be relabelled as one.
          An earlier version routed budget exhaustion and model timeouts through the
          `abstain` node. That node sets status="abstained" and
          failure="citation_verification", so every infrastructure failure arrived in
          the eval report as a citation problem -- the system would have looked like it
          was abstaining carefully when it was actually timing out.
          The distinction matters for grading: an abstention on an unanswerable task is
          correct behavior and scores as a pass, while a timeout on the same task is a
          broken run that happens to produce no answer. Collapsing them makes the
          abstention metric meaningless.
          This node preserves the failure kind set upstream (SPEC 11.4) and emits the
          controlled user-facing outcome that section requires.
        """
        kind = state.get("failure") or "internal_error"
        detail = state.get("failure_detail") or "no further detail"
        return ResearchState(
            status="failed",
            final_answer=(
                f"This request could not be completed: {state['question']}\n\n"
                f"Failure: {kind}\n{detail}"
            ),
            trace=_trace(state, "fail", f"{kind}: {detail}"),
        )

    # --- Routing ------------------------------------------------------------

    def after_search(state: ResearchState) -> str:
        return "fail" if state.get("status") == "failed" else "assess"

    def after_assess(state: ResearchState) -> str:
        """Loop, or move on.

        Routes on the verdict boolean, never on `stop_reason`. An earlier version read
        the label, and because "sufficient_evidence" is returned BOTH when the evidence
        is sufficient (stop) and when the loop is about to search again, the router
        continued in both cases -- the loop ran to budget on every question and the
        term-coverage policy was dead code at the routing layer. Found by watching a
        live run in the inspector: assess logged "stop: coverage 62%" and the graph
        searched again anyway.
        """
        if not state.get("continue_research", False):
            return "resolve_refs"
        budget = state["budget"]
        if (
            state.get("searches_used", 0) >= budget.max_searches
            or state.get("research_iterations", 0) >= budget.max_research_iterations
        ):
            return "resolve_refs"
        return "refine"

    def after_refine(state: ResearchState) -> str:
        return "resolve_refs" if state.get("stop_reason") == "no_new_evidence" else "search"

    def after_resolve(state: ResearchState) -> str:
        return "select_evidence"

    def after_select(state: ResearchState) -> str:
        if state.get("status") == "abstained":
            return END
        return "answerability" if answerability_gate else "write"

    def after_answerability(state: ResearchState) -> str:
        if state.get("status") == "failed":
            return "fail"
        return END if state.get("status") == "abstained" else "write"

    def after_write(state: ResearchState) -> str:
        return "fail" if state.get("status") == "failed" else "verify"

    def after_verify(state: ResearchState) -> str:
        """Pass, repair once, or abstain.

        DECISION: exactly one repair attempt, counted in state, not left to the model.
          SPEC 11.3 fixes the limit at one. It is enforced by comparing a counter to
          the budget, so an infinite repair loop is structurally impossible rather than
          unlikely. A second failure means the model cannot cite what it was given, and
          the correct output is an abstention that says so.
        """
        errors = state.get("citation_errors", [])
        uncited = not state.get("citations") and not errors and bool(state.get("draft_answer"))
        if not errors and not uncited:
            return "finalize"
        if state.get("repair_attempts", 0) < state["budget"].max_repair_attempts:
            return "repair"
        return "abstain"

    graph: StateGraph[ResearchState, None, ResearchState, ResearchState] = StateGraph(ResearchState)
    graph.add_node("search", search)
    graph.add_node("assess", assess)
    graph.add_node("refine", refine)
    graph.add_node("resolve_refs", resolve_refs)
    graph.add_node("select_evidence", select_evidence)
    graph.add_node("answerability", answerability)
    graph.add_node("write", write)
    graph.add_node("verify", verify)
    graph.add_node("finalize", finalize)
    graph.add_node("repair", repair)
    graph.add_node("abstain", abstain)
    graph.add_node("fail", fail)

    graph.set_entry_point("search")
    graph.add_conditional_edges("search", after_search, ["assess", "fail"])
    graph.add_conditional_edges("assess", after_assess, ["refine", "resolve_refs"])
    graph.add_conditional_edges("refine", after_refine, ["search", "resolve_refs"])
    graph.add_edge("resolve_refs", "select_evidence")
    graph.add_conditional_edges("select_evidence", after_select, ["answerability", "write", END])
    graph.add_conditional_edges("answerability", after_answerability, ["write", "fail", END])
    graph.add_conditional_edges("write", after_write, ["verify", "fail"])
    graph.add_conditional_edges("verify", after_verify, ["finalize", "repair", "abstain"])
    graph.add_edge("repair", "write")
    graph.add_edge("finalize", END)
    graph.add_edge("abstain", END)
    graph.add_edge("fail", END)

    return graph.compile()
