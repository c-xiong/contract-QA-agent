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
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Protocol

from app.agent.answerability import ANSWERABILITY_SYSTEM, AnswerabilityDecision, EvidenceQuote
from app.config import Settings
from app.schemas.evidence import Evidence

# Output ceiling for every live call; budget preflights must reserve at least this much.
MAX_OUTPUT_TOKENS = 2048


class ModelError(RuntimeError):
    """The model call failed, timed out, or returned something unusable."""


class ProviderLimitReached(ModelError):
    """The subscription/rate limit refused a call. Runners stop and resume later instead
    of scoring the affected case as a model failure."""


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
        if system == ANSWERABILITY_SYSTEM:
            # This is a routing fixture, not a semantic answerability model. Simulating
            # an answer with evidence preserves the offline wiring check; adversarial
            # and partial verdicts are exercised by explicit test clients.
            top = self._evidence[0] if self._evidence else None
            decision = AnswerabilityDecision(
                verdict="answerable" if top else "unanswerable",
                supported_aspects=["Stub wiring fixture; semantic support is not evaluated"]
                if top
                else [],
                missing_aspects=[] if top else ["No evidence was supplied"],
                evidence_ids=[top.evidence_id] if top else [],
                quotes=[EvidenceQuote(evidence_id=top.evidence_id, quote=top.excerpt[:4000])]
                if top
                else [],
                reason="Deterministic stub; this verdict does not measure answerability.",
            )
            return ModelResponse(
                text=decision.model_dump_json(),
                input_tokens=len(user) // 4,
                output_tokens=len(decision.model_dump_json()) // 4,
                model_id=self.model_id,
            )
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
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(), max_retries=0
        )

    async def complete(self, system: str, user: str) -> ModelResponse:
        from anthropic import APIError

        try:
            message = await asyncio.wait_for(
                self._client.messages.create(
                    model=self.model_id,
                    max_tokens=MAX_OUTPUT_TOKENS,
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


# The CLI must not see the project, its CLAUDE.md, tools, MCP servers, skills or saved
# sessions; it should behave as a single system+user completion.
CLAUDE_CODE_FLAGS = (
    "--tools",
    "",
    "--setting-sources",
    "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--output-format",
    "json",
)
# Credentials removed from the child environment so the CLI authenticates with the
# subscription login rather than silently billing an API key.
_API_CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CRA_ANTHROPIC_API_KEY")
_LIMIT_MARKERS = ("usage limit", "rate limit", "rate_limit", "overloaded", "limit reached")


class ClaudeCodeClient:
    """Live model access through the local Claude Code CLI, billed to a subscription.

    DECISION: a transport over the same provider and models, not a new provider. Each call
    is one `claude -p` process with the caller's system prompt replacing Claude Code's,
    every tool disabled, user/project settings and MCP servers ignored, and an empty
    working directory, so no project file or CLAUDE.md reaches the model. Claude Code
    still adds a small fixed preamble (environment details and the account email, about
    650 input tokens) to every call; it is identical across arms and disclosed.
    Alternatives rejected: the Agent SDK or `--bare` (both need an API key, defeating
    subscription use) and an OpenAI-compatible shim (a different provider).

    The CLI has no max_tokens control, and adaptive thinking tokens count as output, so
    budget preflights for this backend declare a larger per-call output ceiling. The
    served model is checked against the requested one on every call.
    """

    def __init__(self, settings: Settings) -> None:
        executable = shutil.which(settings.claude_code_path)
        if executable is None:
            raise ModelError(
                f"CRA_MODEL_BACKEND=claude_code but `{settings.claude_code_path}` is not on PATH"
            )
        self.executable = executable
        self.model_id = settings.model_id
        self.effort = settings.claude_code_effort
        self._timeout = settings.claude_code_timeout_s
        self._workdir = tempfile.mkdtemp(prefix="cra-claude-code-")
        self._env = {k: v for k, v in os.environ.items() if k not in _API_CREDENTIAL_VARS}
        self.limit_reached = False

    def command(self, system: str) -> list[str]:
        return [
            self.executable,
            "-p",
            "--model",
            self.model_id,
            "--effort",
            self.effort,
            "--system-prompt",
            system,
            *CLAUDE_CODE_FLAGS,
        ]

    async def complete(self, system: str, user: str) -> ModelResponse:
        process = await asyncio.create_subprocess_exec(
            *self.command(system),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._workdir,
            env=self._env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(user.encode()), timeout=self._timeout
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            process.kill()
            await process.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise ModelError(f"Claude Code call exceeded {self._timeout}s") from exc
        return self.parse(stdout.decode(errors="replace"), stderr.decode(errors="replace"))

    def parse(self, stdout: str, stderr: str) -> ModelResponse:
        payload: dict[str, Any] | None = None
        for line in reversed(stdout.strip().splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("type") == "result":
                payload = candidate
                break
        if payload is None:
            detail = " ".join((stderr or stdout).split())[:200]
            self._flag_limit(detail)
            raise ModelError(f"Claude Code returned no result: {detail or 'empty output'}")
        text = str(payload.get("result") or "")
        if payload.get("is_error") or payload.get("subtype") != "success":
            status = payload.get("api_error_status")
            self._flag_limit(text, status)
            raise ModelError(f"Claude Code call failed (status {status}): {text[:200]}")
        served = sorted(payload.get("modelUsage") or {})
        if served != [self.model_id]:
            raise ModelError(f"Claude Code served {served}, not the requested {self.model_id}")
        usage = payload.get("usage") or {}
        if not text.strip():
            raise ModelError("Model returned no text content")
        return ModelResponse(
            text=text,
            input_tokens=sum(
                int(usage.get(key) or 0)
                for key in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            ),
            output_tokens=int(usage.get("output_tokens") or 0),
            model_id=self.model_id,
        )

    def _flag_limit(self, detail: str, status: Any = None) -> None:
        if status in (429, 529) or any(m in detail.lower() for m in _LIMIT_MARKERS):
            self.limit_reached = True
            raise ProviderLimitReached(f"Claude Code usage/rate limit: {detail[:200]}")


def claude_code_version(settings: Settings) -> str | None:
    """The CLI version, recorded in provenance; None when the CLI is unavailable."""
    executable = shutil.which(settings.claude_code_path)
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, timeout=30, check=False
    )
    return result.stdout.strip() or None


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
        if settings.model_backend == "claude_code":
            return ClaudeCodeClient(settings)
        return AnthropicClient(settings)
    return StubClient()
