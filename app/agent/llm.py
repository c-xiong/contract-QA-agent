"""Model adapter: one provider, hard timeouts, and a deterministic stub.

Two implementations behind one protocol. `AnthropicClient` calls the API;
`StubClient` returns a deterministic answer built from the evidence it was given.
The stub is not a mock in the testing sense -- it is the default. Nothing in this
repository spends money unless `CRA_LIVE_MODEL=1` is set explicitly, so tests, CI,
and a first run on a fresh clone all work with no API key.

The stub also keeps the citation gate honest during development: it can be made to
emit a bad citation on demand, which is how the verifier's failure paths get
exercised without paying a model to misbehave.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.schemas.evidence import Evidence


class ModelError(RuntimeError):
    """The model call failed, timed out, or returned something unusable."""


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = "stub"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelClient(Protocol):
    model_id: str

    async def complete(self, system: str, user: str) -> ModelResponse: ...


class StubClient:
    """Deterministic stand-in for a model. No network, no key, no cost.

    It reproduces the shape of a real answer -- a sentence of prose followed by a
    citation drawn from the evidence -- without any language understanding. That is
    enough to exercise wiring, the citation gate, and the eval runner end to end,
    which is all Sprint 0 claims to do.
    """

    model_id = "stub"

    def __init__(self, evidence: list[Evidence] | None = None) -> None:
        self._evidence = evidence or []

    def with_evidence(self, evidence: list[Evidence]) -> StubClient:
        return StubClient(evidence)

    async def complete(self, system: str, user: str) -> ModelResponse:
        if not self._evidence:
            return ModelResponse(
                text=(
                    "I could not find evidence in the searched documents to answer this question."
                ),
                model_id=self.model_id,
            )

        top = self._evidence[0]
        excerpt = " ".join(top.excerpt.split())[:300]
        citation = _render_citation(top)
        text = (
            f"Based on the retrieved evidence: {excerpt} {citation}\n\n"
            f"(This answer was produced by the deterministic stub, not a model. "
            f"Set CRA_LIVE_MODEL=1 to use {Settings().model_id}.)"
        )
        return ModelResponse(
            text=text,
            input_tokens=len(user) // 4,
            output_tokens=len(text) // 4,
            model_id=self.model_id,
        )


class AnthropicClient:
    """Live model access with a timeout enforced in code.

    The timeout is `asyncio.wait_for` around the SDK call, not the SDK's own timeout
    parameter alone. Budgets and limits are deterministic code in this project
    (CLAUDE.md rule 5), and that includes not trusting a library setting to be the
    only thing standing between a hung request and a stalled eval run.
    """

    def __init__(self, settings: Settings) -> None:
        if settings.anthropic_api_key is None:
            raise ModelError(
                "CRA_LIVE_MODEL is set but ANTHROPIC_API_KEY is not. "
                "Set the key, or unset CRA_LIVE_MODEL to use the deterministic stub."
            )
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ModelError(f"anthropic SDK unavailable: {exc}") from exc

        self.model_id = settings.model_id
        self._timeout = settings.model_timeout_s
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

    async def complete(self, system: str, user: str) -> ModelResponse:
        from anthropic import APIError

        try:
            message = await asyncio.wait_for(
                self._client.messages.create(
                    model=self.model_id,
                    max_tokens=2048,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                ),
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            raise ModelError(f"Model call exceeded {self._timeout}s") from exc
        except APIError as exc:
            raise ModelError(f"Model call failed: {exc}") from exc
        except (TypeError, ValueError) as exc:
            # The SDK raises TypeError for unresolvable credentials, which is a
            # configuration problem rather than an API problem. Left unwrapped it
            # escapes the graph's ModelError handler and kills the run instead of
            # producing a classified failure the UI can display.
            raise ModelError(f"Model client misconfigured: {exc}") from exc

        parts = [block.text for block in message.content if block.type == "text"]
        if not parts:
            raise ModelError("Model returned no text content")

        return ModelResponse(
            text="\n".join(parts),
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model_id=self.model_id,
        )


def _render_citation(evidence: Evidence) -> str:
    """Render evidence in the internal citation format.

    The page component is omitted when the source has no pagination. An earlier version
    always emitted it, producing `[doc-041, p. None]` for every ContractNLI document --
    which the parser rejected as malformed, so the gate failed, so every ContractNLI task
    abstained. The symptom looked like a retrieval problem and was a formatting bug.
    """
    parts = [evidence.document_id]
    if evidence.page_number is not None:
        parts.append(f"p. {evidence.page_number}")
    if evidence.section_id:
        parts.append(f"§{evidence.section_id}")
    return f"[{', '.join(parts)}]"


def build_client(settings: Settings) -> ModelClient:
    """Return the live client if explicitly enabled, otherwise the stub."""
    if settings.live_model:
        return AnthropicClient(settings)
    return StubClient()
