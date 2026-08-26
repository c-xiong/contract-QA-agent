"""Execution limits and the research-loop control policy.

Every bound here is deterministic code with a test, which is what CLAUDE.md rule 5
means in practice.
"""

from __future__ import annotations

from app.agent.policies import (
    MIN_TERM_COVERAGE,
    content_terms,
    refine_query,
    should_continue_research,
    term_coverage,
)
from app.agent.state import Budget, initial_state
from tests.conftest import make_chunk, make_retrieved


def state_with(chunks: list, **overrides: object):
    state = initial_state("what is the aggregate liability cap for negligence")
    state["retrieved_chunks"] = chunks
    for key, value in overrides.items():
        state[key] = value  # type: ignore[literal-required]
    return state


def chunk_with(text: str, chunk_id: str = "doc-900-c1", page: int = 1):
    # Distinct pages matter: locations deduplicate on (document, page, section), so
    # chunks sharing a page count as ONE location however many of them there are.
    return make_retrieved(make_chunk(chunk_id, "doc-900", page_number=page, text=text))


class TestTermCoverage:
    def test_counts_content_words_only(self) -> None:
        assert content_terms("what is the governing law") == {"governing", "law"}

    def test_full_coverage(self) -> None:
        chunks = [chunk_with("the governing law is Delaware")]
        assert term_coverage("what is the governing law", chunks) == 1.0

    def test_zero_coverage(self) -> None:
        chunks = [chunk_with("insurance certificates must be provided")]
        assert term_coverage("what is the governing law", chunks) == 0.0

    def test_no_chunks_is_zero(self) -> None:
        assert term_coverage("governing law", []) == 0.0

    def test_question_with_no_content_words_is_fully_covered(self) -> None:
        """Otherwise the loop spins on a question there is nothing to cover."""
        assert term_coverage("what is it", []) == 1.0


class TestSufficiency:
    def test_on_topic_evidence_stops_the_loop(self) -> None:
        chunks = [
            chunk_with("the aggregate liability cap for negligence is stated here", "c1", 1),
            chunk_with("aggregate liability and negligence are addressed", "c2", 2),
        ]
        verdict = should_continue_research(
            state_with(chunks), new_chunks=chunks, previous_locations=set()
        )
        assert verdict.should_continue is False
        assert verdict.reason == "sufficient_evidence"

    def test_off_topic_evidence_continues_the_loop(self) -> None:
        """The case the count-based policy could not see.

        Five chunks were retrieved, so a chunk-count threshold is satisfied -- but none
        of them mention what was asked about. Experiment B returned a perfect null with
        the count-based policy because this case never triggered a second search.
        """
        chunks = [
            chunk_with("insurance certificates and coverage levels", f"c{i}", i + 1)
            for i in range(5)
        ]
        assert term_coverage("aggregate liability cap negligence", chunks) < MIN_TERM_COVERAGE
        verdict = should_continue_research(
            state_with(chunks), new_chunks=chunks, previous_locations=set()
        )
        assert verdict.should_continue is True

    def test_search_budget_stops_the_loop_regardless_of_coverage(self) -> None:
        chunks = [chunk_with("unrelated text about insurance", "c1")]
        verdict = should_continue_research(
            state_with(chunks, searches_used=3, budget=Budget(max_searches=3)),
            new_chunks=chunks,
            previous_locations=set(),
        )
        assert verdict.should_continue is False
        assert verdict.reason == "search_budget_exhausted"

    def test_iteration_budget_stops_the_loop(self) -> None:
        chunks = [chunk_with("unrelated insurance text", "c1")]
        verdict = should_continue_research(
            state_with(
                chunks,
                searches_used=1,
                research_iterations=3,
                budget=Budget(max_searches=9, max_research_iterations=3),
            ),
            new_chunks=chunks,
            previous_locations=set(),
        )
        assert verdict.should_continue is False
        assert verdict.reason == "iteration_budget_exhausted"

    def test_a_search_adding_nothing_new_stops_immediately(self) -> None:
        """Without this, a refined query returning the same chunks burns the budget."""
        chunk = chunk_with("insurance text", "c1")
        previous = {(chunk.chunk.document_id, chunk.chunk.page_number, chunk.chunk.section_id)}
        verdict = should_continue_research(
            state_with([chunk]), new_chunks=[chunk], previous_locations=previous
        )
        assert verdict.should_continue is False
        assert verdict.reason == "no_new_evidence"

    def test_no_results_at_all(self) -> None:
        verdict = should_continue_research(state_with([]), new_chunks=[], previous_locations=set())
        assert verdict.should_continue is False
        assert verdict.reason == "no_results"


class TestRefineQuery:
    def test_drops_stopwords_and_keeps_discriminative_terms(self) -> None:
        refined = refine_query("What is the governing law of this agreement?", [])
        assert refined is not None
        assert "the" not in refined.split()
        assert "governing" in refined

    def test_orders_by_length(self) -> None:
        """In contract language the long words are the discriminative ones."""
        refined = refine_query("what are the indemnification and tax terms", [])
        assert refined is not None
        assert refined.split()[0] == "indemnification"

    def test_returns_none_when_the_query_repeats(self) -> None:
        """A refiner that always returns something guarantees the budget is burned."""
        first = refine_query("governing law jurisdiction", [])
        assert first is not None
        assert refine_query("governing law jurisdiction", [first]) is None

    def test_returns_none_for_a_question_with_no_content_words(self) -> None:
        assert refine_query("what is it?", []) is None


class TestBudget:
    def test_defaults_match_the_spec(self) -> None:
        """SPEC 11.3: at most 3 retrieval iterations, 2 queries each, 5 chunks per query,
        1 citation repair, cross-reference depth 1."""
        budget = Budget()
        assert budget.max_research_iterations == 3
        assert budget.max_queries_per_iteration == 2
        assert budget.max_chunks_per_query == 5
        assert budget.max_repair_attempts == 1
        assert budget.max_cross_reference_depth == 1

    def test_with_searches_narrows_iterations_too(self) -> None:
        """A one-search budget with three iterations allowed would loop without searching."""
        narrowed = Budget().with_searches(1)
        assert narrowed.max_searches == 1
        assert narrowed.max_research_iterations == 1
        assert narrowed.max_chunks_per_query == Budget().max_chunks_per_query


class TestRoutingContract:
    """The verdict boolean is what routes; `stop_reason` is a label.

    Both a stop and a continue can carry reason "sufficient_evidence". Routing on the
    label made the loop run to budget on every question and turned the term-coverage
    policy into dead code -- caught by watching a live run in the trace inspector.
    """

    def test_stop_and_continue_share_a_reason_label(self) -> None:
        on_topic = [
            chunk_with("the aggregate liability cap for negligence is stated here", "c1", 1),
            chunk_with("aggregate liability and negligence are addressed", "c2", 2),
        ]
        stop = should_continue_research(
            state_with(on_topic), new_chunks=on_topic, previous_locations=set()
        )
        off_topic = [chunk_with("insurance coverage levels", f"c{i}", i + 1) for i in range(3)]
        go = should_continue_research(
            state_with(off_topic), new_chunks=off_topic, previous_locations=set()
        )

        assert stop.reason == go.reason == "sufficient_evidence"
        assert stop.should_continue is False
        assert go.should_continue is True, (
            "the label cannot distinguish these; only should_continue can"
        )
