"""Free, controlled answerability cases; these fixtures are not quality benchmarks."""

from __future__ import annotations

import json

import pytest

from app.agent.answerability import ANSWERABILITY_SYSTEM
from app.agent.graph import build_graph
from app.agent.llm import ModelError, ModelResponse
from app.agent.runner import ResearchAgent
from app.config import Settings
from app.ingestion.store import ChunkStore


class VerdictClient:
    model_id = "answerability-fixture"

    def __init__(self, verdict: str, *, repair: bool = False) -> None:
        self.verdict = verdict
        self.repair = repair
        self.prompts: list[tuple[str, str]] = []
        self.writer_calls = 0

    async def complete(self, system: str, user: str) -> ModelResponse:
        self.prompts.append((system, user))
        if system == ANSWERABILITY_SYSTEM:
            if self.verdict == "timeout":
                raise ModelError("Model call exceeded 60.0s")
            if self.verdict == "malformed":
                return ModelResponse(text="not JSON", input_tokens=20, output_tokens=5)
            evidence = json.loads(user)["evidence"]
            item = next((e for e in evidence if "Delaware" in e["excerpt"]), evidence[0])
            supported = self.verdict != "unanswerable"
            quote = item["excerpt"] if self.verdict != "false_quote" else "Insurance is unlimited."
            payload = {
                "verdict": "answerable" if self.verdict == "false_quote" else self.verdict,
                "supported_aspects": ["Governing law"] if supported else [],
                "missing_aspects": []
                if self.verdict in ("answerable", "false_quote")
                else ["Required cyber insurance coverage"],
                "evidence_ids": [item["evidence_id"]] if supported else [],
                "quotes": [{"evidence_id": item["evidence_id"], "quote": quote}]
                if supported
                else [],
                "reason": "The governing-law clause does not establish insurance coverage.",
            }
            return ModelResponse(text=json.dumps(payload), input_tokens=20, output_tokens=5)
        self.writer_calls += 1
        if self.repair and self.writer_calls == 1:
            text = "The law is Delaware [doc-999, p. 1]."
        else:
            text = "The governing law is Delaware [doc-900, p. 3, §14.2]."
            if self.verdict == "partial":
                text += " The supplied excerpts do not establish the required insurance coverage."
        return ModelResponse(text=text, input_tokens=30, output_tokens=12)


def controlled_agent(
    store: ChunkStore, settings: Settings, client: VerdictClient, *, gate: bool = True
) -> ResearchAgent:
    agent = ResearchAgent(store, settings, answerability_gate=gate)
    agent.client = client  # type: ignore[assignment]
    agent.graph = build_graph(
        agent.retriever, agent.verifier, client, store, answerability_gate=gate
    )
    return agent


async def test_related_but_unanswerable_evidence_stops_before_writer(
    store: ChunkStore, settings: Settings
) -> None:
    client = VerdictClient("unanswerable")
    result = await controlled_agent(store, settings, client).research(
        "What cyber insurance coverage is required under the governing law?"
    )
    assert result.retrieved, "the important case has retrieval hits"
    assert result.status == "abstained"
    assert result.failure is None
    assert client.writer_calls == 0
    assert result.model_calls == 1
    assert result.answerability is not None
    assert result.answerability.verdict == "unanswerable"
    assert "Evidence checked:" in result.answer
    assert "Required cyber insurance coverage" in result.answer
    assert result.input_tokens == 20 and result.output_tokens == 5


async def test_supported_answer_still_reaches_writer_and_citation_gate(
    store: ChunkStore, settings: Settings
) -> None:
    client = VerdictClient("answerable")
    result = await controlled_agent(store, settings, client).research("What is the governing law?")
    assert result.status == "completed"
    assert result.citations and not result.citation_errors
    assert client.writer_calls == 1
    assert result.model_calls == 2
    assert result.input_tokens == 50 and result.output_tokens == 17
    assert result.elapsed_seconds > 0
    steps = [event.step for event in result.trace]
    assert steps.index("select_evidence") < steps.index("answerability") < steps.index("write")


async def test_partial_scope_is_retained_on_citation_repair(
    store: ChunkStore, settings: Settings
) -> None:
    client = VerdictClient("partial", repair=True)
    result = await controlled_agent(store, settings, client).research(
        "What is the governing law and required insurance coverage?"
    )
    assert result.status == "completed"
    assert result.answerability is not None and result.answerability.verdict == "partial"
    assert "do not establish" in result.answer
    assert result.model_calls == 3
    writer_prompts = [user for system, user in client.prompts if system != ANSWERABILITY_SYSTEM]
    assert len(writer_prompts) == 2
    assert all("Required cyber insurance coverage" in prompt for prompt in writer_prompts)
    assert all("Answer only the supported aspects" in prompt for prompt in writer_prompts)


@pytest.mark.parametrize("verdict", ["malformed", "false_quote", "timeout"])
async def test_gate_failures_are_neither_answers_nor_correct_abstentions(
    store: ChunkStore, settings: Settings, verdict: str
) -> None:
    client = VerdictClient(verdict)
    result = await controlled_agent(store, settings, client).research("What is the governing law?")
    assert result.status == "failed" and not result.abstained
    assert result.failure == ("model_timeout" if verdict == "timeout" else "output_validation")
    assert result.model_calls == 1
    assert client.writer_calls == 0
    if verdict != "timeout":
        assert result.input_tokens == 20 and result.output_tokens == 5


async def test_baseline_switch_skips_gate_without_changing_writer(
    store: ChunkStore, settings: Settings
) -> None:
    client = VerdictClient("unanswerable")
    result = await controlled_agent(store, settings, client, gate=False).research(
        "What is the governing law?"
    )
    assert result.status == "completed"
    assert result.answerability is None
    assert result.model_calls == 1
    assert all(system != ANSWERABILITY_SYSTEM for system, _ in client.prompts)


async def test_empty_retrieval_needs_no_answerability_call(
    store: ChunkStore, settings: Settings
) -> None:
    client = VerdictClient("answerable")
    result = await controlled_agent(store, settings, client).research("zzzz qqqq")
    assert result.abstained and result.model_calls == 0
    assert not client.prompts


async def test_stub_gate_is_identified_as_a_simulation(
    store: ChunkStore, settings: Settings
) -> None:
    result = await ResearchAgent(store, settings, answerability_gate=True).research(
        "What is the governing law?"
    )
    assert result.status == "completed"
    event = next(event for event in result.trace if event.step == "answerability")
    assert event.data["simulated"] is True
    assert "does not measure answerability" in event.detail


async def test_default_preserves_measured_baseline(store: ChunkStore, settings: Settings) -> None:
    agent = ResearchAgent(store, settings)
    assert not agent.answerability_gate
    result = await agent.research("What is the governing law?")
    assert result.status == "completed" and result.answerability is None
    assert not any(event.step == "answerability" for event in result.trace)
