"""A2A executor: maps one A2A task onto one V2 research run.

DECISION: this module is a protocol adapter, like app/mcp_server.py. Retrieval, tool
  policy, budgets, prompt-injection handling and citation verification all run inside
  the unchanged V2 graph via `ResearchAgent.research()`; nothing here re-decides them.
  The caller can only narrow scope with `document_ids`, never widen it.

DECISION: `input-required` is an adapter scope policy, not a model judgement.
  The V2 agent never asks clarifying questions: it abstains. The real ambiguity in this
  corpus is *which contract* a question is about (one indexed snapshot per document, so
  version ambiguity does not occur). If a request names no documents and the corpus holds
  more than one, the task asks for `document_ids` before any model call. Alternatives
  rejected: mapping the agent's abstention to input-required (an abstention is a
  correct final answer, CLAUDE.md rule 3) and adding a clarify action to the planner
  (changes the core and the frozen evaluation suites).

DECISION: `context_id` groups tasks and is recorded in the trace; it carries no
  memory between tasks (CLAUDE.md rule 6). Text in a clarification reply is ignored; only
  its `document_ids` are read, and the question is the task's first user message.

Lifecycle: working (with coarse progress) -> artifact + completed for answers and
abstentions; failed with the agent's `stop_reason` for run failures and adapter errors;
canceled on request. A failure is never reported as an abstention or an answer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from a2a.helpers import get_data_parts, get_message_text, new_data_part, new_task, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Role, TaskState
from a2a.utils.errors import InvalidParamsError

from app.agent.llm import StubClient
from app.agent.runner import ResearchAgent, ResearchResult
from app.agent.state import ResearchState
from app.api.schemas import render_citations

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000  # same bound as the inspector's stream endpoint
MAX_CLARIFICATIONS = 2
MAX_LISTED_DOCUMENTS = 100

# Human sentences for the agent's closed stop_reason vocabulary (app/agent/v2.py).
REASONS = {
    "verified": "Answer verified against cited clauses.",
    "insufficient_evidence": "The contract text does not support an answer.",
    "repeated_action_without_evidence": "Research stopped without finding supporting text.",
    "citation_verification_failed": "Citations could not be verified, so no answer is given.",
    "deadline_exhausted": "The run hit its time budget.",
    "model_budget_exhausted": "The run hit its model-call budget.",
    "draft_budget_exhausted": "The run hit its drafting budget.",
    "verification_budget_exhausted": "The run hit its verification budget.",
    "model_unavailable": "The language model was unavailable.",
    "invalid_tool_arguments": "The planner produced invalid tool arguments.",
    "invalid_document_or_version_scope": "The requested document or version is not indexed.",
    "scope_or_provenance_violation": "A tool result fell outside the permitted scope.",
    "internal_error": "Internal error; see the run trace.",
    "unknown_document_ids": "One or more document_ids are not in the corpus.",
    "scope_not_provided": "No contract was specified after repeated requests.",
}


class ContractQAExecutor(AgentExecutor):
    def __init__(self, agent: ResearchAgent):
        if agent.settings.agent_mode != "v2":
            raise ValueError("The A2A entrypoint requires agent_mode='v2' (traces, cancel).")
        self.agent = agent
        self.clarifications: dict[str, int] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if message is None or context.task_id is None or context.context_id is None:
            raise InvalidParamsError(message="A message is required")
        task = context.current_task

        if task is None:
            question = get_message_text(message).strip()
            if not question or len(question) > MAX_QUESTION_CHARS:
                raise InvalidParamsError(
                    message=f"Question text must be 1-{MAX_QUESTION_CHARS} characters"
                )
            task = new_task(
                context.task_id,
                context.context_id,
                TaskState.TASK_STATE_SUBMITTED,
                history=[message],
            )
            await event_queue.enqueue_event(task)
        elif task.status.state != TaskState.TASK_STATE_INPUT_REQUIRED:
            # The SDK queues messages per task and runs them one after another, so a
            # message sent while research runs arrives here once the task is terminal.
            # One task maps to at most one run: never start a second one.
            logger.info(
                "A2A task %s context %s: message ignored in state %s",
                context.task_id,
                context.context_id,
                TaskState.Name(task.status.state),
            )
            return
        else:
            # Continuation of an input-required task: the reply may be data-only, so the
            # question is always the task's first user message.
            first = next(m for m in task.history if m.role == Role.ROLE_USER)
            question = get_message_text(first).strip()

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        # An empty list (e.g. Enter at the demo prompt) is an unanswered ask, not a scope.
        document_ids = requested_documents(message) or None
        store = self.agent.store

        if document_ids is None and len(store.documents) > 1:
            # Counted in-process: the task store is in-memory too, so both die together.
            rounds = self.clarifications.get(context.task_id, 0)
            if rounds >= MAX_CLARIFICATIONS:
                self.clarifications.pop(context.task_id, None)
                await fail(updater, "scope_not_provided")
                return
            self.clarifications[context.task_id] = rounds + 1
            logger.info(
                "A2A task %s context %s: input-required (no document scope, ask %d)",
                updater.task_id,
                updater.context_id,
                rounds + 1,
            )
            await updater.requires_input(self.clarification(updater))
            return
        self.clarifications.pop(context.task_id, None)
        if document_ids is not None:
            unknown = [d for d in document_ids if not store.has_document(d)]
            if unknown:
                await fail(updater, "unknown_document_ids", unknown=unknown)
                return

        await updater.start_work(text(updater, "Accepted; starting research"))
        await self.run(question, document_ids, context, updater)

    async def run(
        self,
        question: str,
        document_ids: list[str] | None,
        context: RequestContext,
        updater: TaskUpdater,
    ) -> None:
        phases: asyncio.Queue[str | None] = asyncio.Queue(maxsize=16)
        drain = asyncio.create_task(self.drain(phases, updater))
        try:
            try:
                result = await self.agent.research(
                    question,
                    allowed_document_ids=document_ids,
                    progress=observer(phases),
                    external_ref={
                        "a2a.task_id": updater.task_id,
                        "a2a.context_id": updater.context_id,
                    },
                )
            except asyncio.CancelledError:
                raise  # the SDK calls cancel(); v2_states has journaled run_cancelled
            except Exception:
                logger.exception("A2A task %s: research raised", updater.task_id)
                await finish_drain(phases, drain)
                await fail(updater, "internal_error")
                return
            await finish_drain(phases, drain)
            await self.publish(result, updater)
        finally:
            drain.cancel()
            with suppress(asyncio.CancelledError):
                await drain

    async def drain(self, phases: asyncio.Queue[str | None], updater: TaskUpdater) -> None:
        while (label := await phases.get()) is not None:
            await updater.update_status(TaskState.TASK_STATE_WORKING, text(updater, label))

    async def publish(self, result: ResearchResult, updater: TaskUpdater) -> None:
        if result.status in ("completed", "abstained"):
            await updater.add_artifact(
                [new_text_part(result.answer), new_data_part(self.payload(result))],
                name="contract_answer",
            )
            note = "Answered" if result.status == "completed" else "Abstained"
            reason = REASONS.get(result.stop_reason or "", result.stop_reason or "")
            await updater.complete(text(updater, f"{note}: {reason}"))
            return
        await fail(
            updater,
            result.stop_reason or "internal_error",
            run_id=result.run_id,
            failure=result.failure,
        )

    def payload(self, result: ResearchResult) -> dict[str, Any]:
        titles = {d.document_id: d.title for d in self.agent.store.documents}
        return {
            "status": result.status,
            "run_id": result.run_id,
            "stop_reason": result.stop_reason,
            "citations": [c.model_dump(mode="json") for c in render_citations(result, titles)],
            "documents_searched": result.retrieved_document_ids,
            "model_id": self.agent.client.model_id,
            "simulated": isinstance(self.agent.client, StubClient)
            or bool(getattr(self.agent.client, "simulated", False)),
        }

    def clarification(self, updater: TaskUpdater) -> Message:
        documents = sorted(self.agent.store.documents, key=lambda d: d.document_id)
        listed = [
            {"document_id": d.document_id, "title": d.title}
            for d in documents[:MAX_LISTED_DOCUMENTS]
        ]
        return updater.new_agent_message(
            [
                new_text_part(
                    "Which contract is this question about? Reply on this task with a data "
                    'part {"document_ids": [...]}. Reply text is not added to the question.'
                ),
                new_data_part(
                    {"documents": listed, "truncated": len(documents) > MAX_LISTED_DOCUMENTS}
                ),
            ]
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # The SDK has already cancelled execute(); the run journal records run_cancelled.
        if context.task_id is None or context.context_id is None:
            return
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel(text(updater, "Canceled on request."))


def requested_documents(message: Message) -> list[str] | None:
    """`document_ids` from the message's data parts; None when absent."""
    for data in get_data_parts(message.parts):
        if isinstance(data, dict) and "document_ids" in data:
            ids = data["document_ids"]
            if isinstance(ids, list):
                return [str(d) for d in ids]
            return [str(ids)]
    return None


def observer(phases: asyncio.Queue[str | None]) -> Callable[[ResearchState], None]:
    """Synchronous, non-blocking: derive a phase label, drop it if the queue is full."""
    last: list[str] = []

    def observe(state: ResearchState) -> None:
        label = phase(state)
        if label is None or (last and last[-1] == label):
            return
        last.append(label)
        with suppress(asyncio.QueueFull):
            phases.put_nowait(label)

    return observe


def phase(state: ResearchState) -> str | None:
    status = state.get("status")
    if status == "researching":
        return "Searching clauses"
    if status == "writing":
        return "Verifying citations" if state.get("draft_answer") else "Drafting answer"
    return None


async def finish_drain(phases: asyncio.Queue[str | None], drain: asyncio.Task[None]) -> None:
    """Let queued progress go out before the terminal event (a terminal state is final)."""
    try:
        phases.put_nowait(None)
    except asyncio.QueueFull:
        drain.cancel()
    with suppress(asyncio.CancelledError):
        await drain


def text(updater: TaskUpdater, body: str) -> Message:
    return updater.new_agent_message([new_text_part(body)])


async def fail(updater: TaskUpdater, reason: str, **detail: Any) -> None:
    sentence = REASONS.get(reason, reason)
    data = {"stop_reason": reason, **{k: v for k, v in detail.items() if v is not None}}
    # IDs and the closed reason code only: runless outcomes are traced by this line.
    logger.info(
        "A2A task %s context %s: failed (%s) run_id=%s",
        updater.task_id,
        updater.context_id,
        reason,
        detail.get("run_id"),
    )
    await updater.failed(
        updater.new_agent_message(
            [new_text_part(f"Failed ({reason}): {sentence}"), new_data_part(data)]
        )
    )
