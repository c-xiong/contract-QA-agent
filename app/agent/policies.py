"""Execution limits and control policy. See docs/decisions.md

DECISION-BEARING MODULE. See docs/decisions.md.

Every limit here is deterministic code with a test. None of them is a prompt
instruction. That distinction is the point of CLAUDE.md rule 5: a model told to stop
after three searches usually stops after three searches, and "usually" is not a budget.

The functions are pure -- they take state and return a verdict -- so the graph stays
readable and every policy is testable without running an agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.agent.state import Budget, ResearchState
from app.schemas.retrieval import RetrievedChunk

StopReason = Literal[
    "sufficient_evidence",
    "search_budget_exhausted",
    "iteration_budget_exhausted",
    "no_new_evidence",
    "no_results",
]


@dataclass(frozen=True, slots=True)
class ResearchVerdict:
    """Whether to keep researching, and why not if not."""

    should_continue: bool
    reason: StopReason
    detail: str


# DECISION: "enough evidence" tests topical relevance, not chunk count.
#   Rejected first, and worth recording because a measurement killed it: the original
#   test was "at least MIN_EVIDENCE_CHUNKS distinct locations retrieved". With
#   `top_k = 5` and a threshold of 3, the first search satisfies it every single time.
#   Experiment B ran with that policy and returned a perfect null -- 41 of 41 tasks
#   tied, 41 searches in both arms -- because the refinement loop was unreachable. The
#   threshold was not conservative, it was dead code.
#
#   The test now asks whether the retrieved text actually covers the question's content
#   words. If a question's distinctive terms are largely absent from everything
#   retrieved, retrieval probably missed the topic, and a widened query is worth a
#   search. This fires exactly when it should and stays silent when the first search
#   already landed.
#
#   Rejected: thresholding on the retriever's score. Not portable across arms -- BM25
#   returns unbounded positive scores, RRF returns values near 0.03, and a cross-encoder
#   returns logits. Any absolute threshold would mean something different per arm and
#   would silently make Experiment A and Experiment B interact.
#   Rejected: a model-judged sufficiency check. What most agent tutorials do, and it
#   makes the loop unreproducible across runs of the same task -- which would make
#   Experiment B measure the sampler as much as the design.
#
#   Consequence: term coverage is a lexical proxy. It cannot tell "on the right topic
#   but the wrong clause" from "the right clause", and it under-fires on questions
#   phrased entirely in synonyms -- which is the query-refinement category, precisely
#   where the loop should help most. That limitation belongs in the Experiment B writeup.
MIN_EVIDENCE_CHUNKS = 2

# Fraction of the question's content words that must appear somewhere in the retrieved
# text for the evidence to count as on-topic.
MIN_TERM_COVERAGE = 0.5


def locations(chunks: list[RetrievedChunk]) -> set[tuple[str, int | None, str | None]]:
    """Distinct citable locations, which is what evidence coverage actually counts.

    Keyed the same way `to_evidence` deduplicates, so "how much evidence do I have"
    and "how much evidence will the writer see" cannot drift apart.
    """
    return {(c.chunk.document_id, c.chunk.page_number, c.chunk.section_id) for c in chunks}


def should_continue_research(
    state: ResearchState,
    *,
    new_chunks: list[RetrievedChunk],
    previous_locations: set[tuple[str, int | None, str | None]],
) -> ResearchVerdict:
    """Decide whether to run another retrieval iteration."""
    budget: Budget = state["budget"]
    searches_used = state.get("searches_used", 0)
    iterations = state.get("research_iterations", 0)

    if not new_chunks and not previous_locations:
        return ResearchVerdict(False, "no_results", "no passage matched any query")

    combined = previous_locations | locations(new_chunks)

    # DECISION: a search that adds no new location ends the loop immediately.
    #   Without this, a refined query that happens to return the same chunks burns the
    #   entire budget re-retrieving them. The check is on locations rather than on chunk
    #   ids because two chunks from one split section are the same citable place, and
    #   "found the same clause again" is not progress.
    if new_chunks and not (locations(new_chunks) - previous_locations):
        return ResearchVerdict(
            False, "no_new_evidence", "refined query returned only already-seen locations"
        )

    all_chunks = list(state.get("retrieved_chunks", []))
    coverage = term_coverage(state["question"], all_chunks)

    if len(combined) >= MIN_EVIDENCE_CHUNKS and coverage >= MIN_TERM_COVERAGE:
        return ResearchVerdict(
            False,
            "sufficient_evidence",
            f"{len(combined)} locations, term coverage {coverage:.0%} "
            f"(thresholds {MIN_EVIDENCE_CHUNKS}, {MIN_TERM_COVERAGE:.0%})",
        )

    if searches_used >= budget.max_searches:
        return ResearchVerdict(
            False,
            "search_budget_exhausted",
            f"{searches_used}/{budget.max_searches} searches used",
        )

    if iterations >= budget.max_research_iterations:
        return ResearchVerdict(
            False,
            "iteration_budget_exhausted",
            f"{iterations}/{budget.max_research_iterations} iterations used",
        )

    return ResearchVerdict(
        True,
        "sufficient_evidence",
        f"{len(combined)} location(s), term coverage {coverage:.0%}; searching again",
    )


# DECISION: query refinement is deterministic text transformation, not a model call.
#   The refinement trigger fires when the first search under-delivers. What it produces
#   is a widened query built from the question itself: stopwords dropped, and the
#   highest-signal terms kept.
#   Rejected: asking the model to rewrite the query. Almost certainly better queries,
#   and three costs that matter here -- a model call per iteration, non-reproducible
#   eval runs, and a confound in Experiment B where "agentic beat single-pass" could be
#   explained by the rewriter rather than by the loop.
#   This is a deliberately weak refiner. If Experiment B shows the loop helps at all
#   with a refiner this crude, that is a stronger result than showing it helps with a
#   model-written one, because the loop is doing the work.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "such",
        "any",
        "some",
        "all",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
        "there",
        "here",
        "it",
        "its",
        "their",
        "our",
        "your",
        "my",
        "me",
        "we",
        "you",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
    }
)


def content_terms(text: str) -> set[str]:
    """Content words of a query: the terms retrieval should have matched on."""
    terms = {w.strip(".,;:?!()\"'").casefold() for w in text.split()}
    return {t for t in terms if t and t not in _STOPWORDS and len(t) > 2}


def term_coverage(question: str, chunks: list[RetrievedChunk]) -> float:
    """Fraction of the question's content words appearing in the retrieved text.

    A lexical proxy for "did retrieval land on the topic". Returns 1.0 for a question
    with no content words, because there is nothing to cover and the loop should not
    spin on it.
    """
    wanted = content_terms(question)
    if not wanted:
        return 1.0
    if not chunks:
        return 0.0
    haystack = " ".join(c.chunk.text for c in chunks).casefold()
    return sum(1 for term in wanted if term in haystack) / len(wanted)


def refine_query(question: str, previous_queries: list[str]) -> str | None:
    """Produce a widened query, or None if nothing new can be tried.

    Returning None matters: a refiner that always returns something guarantees the loop
    burns its full budget even when it has nothing left to try.
    """
    terms = [w.strip(".,;:?!()\"'").casefold() for w in question.split()]
    keywords = [t for t in terms if t and t not in _STOPWORDS and len(t) > 2]
    if not keywords:
        return None

    # Longest words first: in contract language the discriminative terms are the long
    # ones ("indemnification", "confidentiality"), not the short connectives.
    keywords.sort(key=len, reverse=True)
    candidate = " ".join(keywords[:6])

    if not candidate or candidate in {q.casefold() for q in previous_queries}:
        return None
    return candidate
