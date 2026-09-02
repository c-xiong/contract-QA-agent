"""End-to-end: question -> retrieval -> answer -> citation gate -> eval scoring.

Runs entirely on the deterministic stub and a synthetic store. No network, no API key,
no corpus on disk -- so CI exercises the whole wiring for free.
"""

from __future__ import annotations

import pytest

from app.agent.llm import ModelError, ModelResponse, StubClient
from app.agent.runner import ResearchAgent
from app.agent.state import Budget
from app.config import Settings
from app.evidence.claim_support import CLAIM_SUPPORT_SYSTEM
from app.ingestion.store import ChunkStore
from evals.runner import behavior_matches, run_suite
from evals.schema import EvalTask


@pytest.fixture
def agent(store: ChunkStore, settings: Settings) -> ResearchAgent:
    return ResearchAgent(store, settings)


class TestVerticalSlice:
    async def test_question_produces_a_verified_cited_answer(self, agent: ResearchAgent) -> None:
        result = await agent.research("What is the governing law?")
        assert result.status == "completed"
        assert result.citations, "an answer must carry at least one verified citation"
        assert result.citation_errors == []
        assert "doc-900" in result.retrieved_document_ids

    async def test_trace_records_every_step_in_order(self, agent: ResearchAgent) -> None:
        result = await agent.research("What is the governing law?")
        steps = [e.step for e in result.trace]

        # The loop may run `search`/`assess`/`refine` more than once; what must hold is
        # that the phases occur in order and each happens at least once.
        assert steps[0] == "search"
        assert steps[-1] == "finalize"
        for phase in ("assess", "resolve_refs", "select_evidence", "write", "verify"):
            assert phase in steps, f"{phase} missing from trace: {steps}"
        assert steps.index("resolve_refs") < steps.index("select_evidence")
        assert steps.index("select_evidence") < steps.index("write")
        assert steps.index("write") < steps.index("verify")

    async def test_research_loop_is_bounded_by_the_budget(self, agent: ResearchAgent) -> None:
        """The loop cannot exceed its search budget, whatever the evidence looks like."""
        result = await agent.research("liability", budget=Budget(max_searches=2))
        searches = [e for e in result.trace if e.step == "search"]
        assert 1 <= len(searches) <= 2

    async def test_cross_reference_pulls_the_carve_out(self, agent: ResearchAgent) -> None:
        """SPEC 9.5, the case the whole component exists for.

        The fixture's 8.1 caps liability "Except as set out in Section 8.3"; 8.3 is the
        carve-out. A question about the cap retrieves 8.1 lexically. Only reference
        resolution brings 8.3 along, and without it the answer would be confidently
        wrong while every citation check passed.
        """
        result = await agent.research("aggregate liability shall not exceed fees paid")
        pulled = [c for c in result.retrieved if c.pulled_by == "cross_reference"]
        assert pulled, "expected at least one chunk pulled by cross-reference"
        assert any("8.3" in c.chunk.section_path for c in result.retrieved)

        events = [e for e in result.trace if e.step == "resolve_refs"]
        assert events and "pulled" in events[0].detail

    async def test_allowlist_confines_retrieval(self, agent: ResearchAgent) -> None:
        result = await agent.research(
            "Confidential information definitions", allowed_document_ids=["doc-901"]
        )
        assert set(result.retrieved_document_ids) <= {"doc-901"}

    async def test_no_results_abstains_without_calling_the_model(
        self, agent: ResearchAgent
    ) -> None:
        """A fluent answer from parametric knowledge is the exact failure to avoid."""
        result = await agent.research("zzzz nonexistent terminology qqqq")
        assert result.status == "abstained"
        assert result.output_tokens == 0, "no model call should have been made"
        assert "no relevant passages" in result.answer.lower()

    async def test_search_budget_is_enforced_in_code(
        self, store: ChunkStore, settings: Settings
    ) -> None:
        agent = ResearchAgent(store, settings)
        result = await agent.research("governing law", budget=Budget(max_searches=0))
        assert result.status == "failed"
        assert result.failure == "budget_exhausted"


class BadCitationClient:
    """Emits a citation to a real, correctly paginated location that was never retrieved.

    doc-901 page 1 section 1 exists in the store. A question about governing law
    retrieves only doc-900 chunks, so this citation is fabricated support -- and every
    check except grounding passes.
    """

    model_id = "bad-citation-stub"

    async def complete(self, system: str, user: str) -> ModelResponse:
        return ModelResponse(
            text="The agreement is governed by Delaware law [doc-901, p. 1, §1].",
            output_tokens=12,
            model_id=self.model_id,
        )


class UncitedClient:
    model_id = "uncited-stub"

    async def complete(self, system: str, user: str) -> ModelResponse:
        return ModelResponse(text="The cap is one million dollars.", model_id=self.model_id)


class FailingClient:
    model_id = "failing-stub"

    async def complete(self, system: str, user: str) -> ModelResponse:
        raise ModelError("Model call exceeded 60.0s")


class CompoundCitationClient:
    """Reproduces the live model's two locators inside one pair of brackets."""

    model_id = "compound-citation-stub"

    async def complete(self, system: str, user: str) -> ModelResponse:
        return ModelResponse(
            text=(
                "Liability is capped, subject to an exclusion "
                "[doc-900, p. 1, §8.1; doc-900, p. 2, §8.3]."
            ),
            output_tokens=20,
            model_id=self.model_id,
        )


class CitationScopedEvalClient:
    """Offline writer and semantic judge for the claim-support integration path."""

    model_id = "fixture-judge"

    async def complete(self, system: str, user: str) -> ModelResponse:
        if system == CLAIM_SUPPORT_SYSTEM:
            assert "governed by the laws of Delaware" in user
            assert "aggregate liability" not in user
            return ModelResponse(
                text=(
                    '{"verdict":"supported","quote":"governed by the laws of '
                    'Delaware","reason":"fixture"}'
                ),
                model_id=self.model_id,
            )
        return ModelResponse(
            text="The agreement is governed by Delaware law [doc-900, p. 3, §14.2].",
            model_id=self.model_id,
        )


class TestCitationGate:
    async def test_compound_citation_is_normalized_without_repair(
        self, store: ChunkStore, settings: Settings
    ) -> None:
        from app.agent.graph import build_graph

        agent = ResearchAgent(store, settings)
        agent.client = CompoundCitationClient()  # type: ignore[assignment]
        agent.graph = build_graph(agent.retriever, agent.verifier, agent.client, store)

        result = await agent.research("What limits aggregate liability?")

        assert result.status == "completed"
        assert result.citation_errors == []
        assert [citation.section_id for citation in result.citations] == ["8.1", "8.3"]
        assert ";" not in result.answer
        assert "[doc-900, p. 1, §8.1] [doc-900, p. 2, §8.3]" in result.answer
        assert not any(event.step == "repair" for event in result.trace)

    async def test_ungrounded_citation_is_repaired_once_then_abstains(
        self, store: ChunkStore, settings: Settings
    ) -> None:
        """The gate's whole point: a real, correctly paginated, fabricated citation.

        doc-901 page 1 exists and is correctly paginated. It was never retrieved for
        this question. Every check except grounding passes, so only the gate stops it.
        """
        from app.agent.graph import build_graph

        agent = ResearchAgent(store, settings)
        agent.client = BadCitationClient()  # type: ignore[assignment]
        agent.graph = build_graph(agent.retriever, agent.verifier, agent.client)

        result = await agent.research("What is the governing law?")
        assert result.status == "abstained"
        assert result.failure == "citation_verification"
        assert any(e.code == "not_in_evidence" for e in result.citation_errors)

        steps = [e.step for e in result.trace]
        assert steps.count("repair") == 1, "exactly one repair attempt (SPEC 11.3)"
        assert steps.count("rewrite") == 1, "the repair triggers exactly one rewrite"
        assert steps[-1] == "abstain"

    async def test_abstention_says_what_went_wrong(
        self, store: ChunkStore, settings: Settings
    ) -> None:
        """A bare "I don't know" is a bug (SPEC 13.5)."""
        from app.agent.graph import build_graph

        agent = ResearchAgent(store, settings)
        agent.client = BadCitationClient()  # type: ignore[assignment]
        agent.graph = build_graph(agent.retriever, agent.verifier, agent.client)
        result = await agent.research("What is the governing law?")
        assert "not_in_evidence" in result.answer

    async def test_uncited_answer_does_not_pass_the_gate(
        self, store: ChunkStore, settings: Settings
    ) -> None:
        """Otherwise the cheapest way to pass is to cite nothing."""
        from app.agent.graph import build_graph

        agent = ResearchAgent(store, settings)
        agent.client = UncitedClient()  # type: ignore[assignment]
        agent.graph = build_graph(agent.retriever, agent.verifier, agent.client)
        result = await agent.research("What is the liability cap?")
        assert result.status == "abstained"

    async def test_model_failure_is_classified_not_raised(
        self, store: ChunkStore, settings: Settings
    ) -> None:
        from app.agent.graph import build_graph

        agent = ResearchAgent(store, settings)
        agent.client = FailingClient()  # type: ignore[assignment]
        agent.graph = build_graph(agent.retriever, agent.verifier, agent.client)
        result = await agent.research("What is the liability cap?")
        assert result.status == "failed"
        assert result.failure == "model_timeout"


class TestEvalLoop:
    async def test_suite_runs_and_scores_a_task(
        self, agent: ResearchAgent, settings: Settings
    ) -> None:
        """The Sprint 0 exit criterion, on a synthetic fixture task.

        The real suite's content is author-written (CLAUDE.md rule 1); this fixture
        exercises the machinery, not the dataset.
        """
        task = EvalTask.model_validate(
            {
                "task_id": "fixture-001",
                "category": "single_document_fact_lookup",
                "question": "Which state's law governs this agreement?",
                "expected_behavior": "answer",
                "dataset_version": "fixture-v1",
                "allowed_document_ids": ["doc-900"],
                "expected_document_ids": ["doc-900"],
                "expected_evidence": [
                    {
                        "document_id": "doc-900",
                        "page_number": 3,
                        "span_text": "This Agreement is governed by the laws of Delaware.",
                        "source": "manual",
                    }
                ],
            }
        )

        report = await run_suite("fixture", [task], agent, settings)
        assert report.task_count == 1
        assert report.runs[0].behavior_matched is True
        assert report.mean("document_recall") == 1.0
        assert report.mean("evidence_span_recall") == 1.0

        assert report.grader_versions, "grader versions are recorded with each result"
        assert "n=1" in report.summary(), "sample size appears in the summary (SPEC 15.7)"

    async def test_report_serializes_for_the_artifact(
        self, agent: ResearchAgent, settings: Settings
    ) -> None:
        task = EvalTask.model_validate(
            {
                "task_id": "fixture-002",
                "category": "unanswerable",
                "question": "zzzz qqqq nonexistent terminology?",
                "expected_behavior": "abstain",
                "dataset_version": "fixture-v1",
                "expected_document_ids": [],
                "expected_evidence": [],
            }
        )
        report = await run_suite("fixture", [task], agent, settings)
        assert report.runs[0].abstained is True
        assert report.runs[0].behavior_matched is True

        payload = report.to_json()
        assert payload["task_count"] == 1
        assert "grader_versions" in payload

    async def test_claim_support_v2_is_citation_scoped_and_reports_macro_micro(
        self, store: ChunkStore, settings: Settings
    ) -> None:
        from app.agent.graph import build_graph

        scoped_agent = ResearchAgent(store, settings)
        scoped_agent.client = CitationScopedEvalClient()  # type: ignore[assignment]
        scoped_agent.graph = build_graph(
            scoped_agent.retriever,
            scoped_agent.verifier,
            scoped_agent.client,
            store,
        )
        live_settings = settings.model_copy(update={"live_model": True})
        task = EvalTask.model_validate(
            {
                "task_id": "fixture-claim-support",
                "category": "single_document_fact_lookup",
                "question": "Which state's law governs this agreement?",
                "expected_behavior": "answer",
                "dataset_version": "fixture-v1",
                "allowed_document_ids": ["doc-900"],
                "expected_document_ids": ["doc-900"],
                "expected_evidence": [
                    {
                        "document_id": "doc-900",
                        "page_number": 3,
                        "section_id": "14.2",
                        "span_text": "governed by the laws of Delaware",
                        "source": "manual",
                    }
                ],
            }
        )

        report = await run_suite(
            "fixture",
            [task],
            scoped_agent,
            live_settings,
            with_claim_support=True,
        )
        payload = report.to_json()

        assert report.grader_versions["claim_support"] == "claim_support@2"
        assert report.grader_versions["citation_coverage"] == "citation_coverage@1"
        assert payload["claim_support_aggregate"] == {
            "macro": 1.0,
            "micro": 1.0,
            "supported_claims": 1,
            "evaluated_claims": 1,
        }
        assert payload["citation_coverage_aggregate"] == {
            "macro": 1.0,
            "micro": 1.0,
            "cited_claims": 1,
            "factual_claims": 1,
        }


class TestStubClient:
    async def test_stub_cites_the_evidence_it_was_given(self, store: ChunkStore) -> None:
        from app.agent.graph import to_evidence
        from tests.conftest import make_retrieved

        chunk = store.get_chunk("doc-900-c0003")
        assert chunk is not None
        evidence = to_evidence([make_retrieved(chunk)], max_chars=10_000)
        response = await StubClient(evidence).complete("system", "user")
        assert "[doc-900, p. 3, §14.2]" in response.text

    async def test_stub_with_no_evidence_declines(self) -> None:
        response = await StubClient([]).complete("system", "user")
        assert "could not find" in response.text.lower()


class TestBehaviorMatching:
    def test_abstain_expected_and_abstained(self, store: ChunkStore) -> None:
        from app.agent.runner import ResearchResult

        task = EvalTask.model_validate(
            {
                "task_id": "t",
                "category": "unanswerable",
                "question": "q?",
                "expected_behavior": "abstain",
                "dataset_version": "v1",
                "expected_document_ids": [],
                "expected_evidence": [],
            }
        )
        result = ResearchResult(
            question="q?",
            answer="",
            status="abstained",
            citations=[],
            citation_errors=[],
            retrieved=[],
            evidence=[],
            trace=[],
            input_tokens=0,
            output_tokens=0,
        )
        assert behavior_matches(task, result) is True


class TestResearchLoopRouting:
    """The loop must stop when the evidence is already sufficient."""

    async def test_sufficient_evidence_stops_after_one_search(self, agent: ResearchAgent) -> None:
        """Regression: the router read `stop_reason` instead of the verdict boolean, so
        a question whose first search already satisfied the policy still refined and
        searched again -- wasting a search and a third of the tokens on every run."""
        result = await agent.research("aggregate liability exceed negligence gross")
        steps = [e.step for e in result.trace]

        assess = next(e for e in result.trace if e.step == "assess")
        assert "coverage" in assess.detail

        assert steps.count("search") == 1, f"expected one search, got: {steps}"
        assert "refine" not in steps

    async def test_thin_evidence_still_triggers_a_second_search(self, agent: ResearchAgent) -> None:
        """The other direction: the fix must not disable the loop entirely."""
        result = await agent.research("Delaware")
        steps = [e.step for e in result.trace]
        assert steps.count("search") >= 1
        assert steps[-1] in ("finalize", "abstain", "fail")
