"""Minimal A2A orchestrator: discover the contract agent, delegate one question, stream it.

Shows the whole A2A flow against `python -m app.a2a`:
  1. discovery   - fetch the public Agent Card, check a skill tagged "contract" exists
  2. delegation  - send the question as a streaming task (bearer token from CRA_A2A_TOKEN)
  3. streaming   - print each status update as it arrives
  4. input-required - when the agent asks which contract, answer on the same task with
                      --clarify-with (or interactively on a TTY)
  5. artifact    - print the answer and its citations

Not a multi-agent router: one card, one skill check. Usage:
  uv run --extra a2a python scripts/a2a_orchestrator.py "<question>" --clarify-with <doc_id>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_data_parts, get_message_text, new_data_part, new_message, new_text_part
from a2a.types import AgentCard, Message, Role, SendMessageRequest, TaskState

from app.config import get_settings

TERMINAL = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}


def capable(card: AgentCard, tag: str = "contract") -> bool:
    return any(tag in skill.tags for skill in card.skills)


async def run(
    url: str,
    question: str,
    *,
    token: str,
    document_ids: list[str] | None = None,
    clarify_with: list[str] | None = None,
    http: httpx.AsyncClient | None = None,
    out: Callable[[str], None] = print,
    ask: Callable[[], str | None] | None = None,
) -> int:
    """Run the demo. Returns a process exit code (0 completed, 1 otherwise, 2 no agent)."""
    owned = http is None
    http = http or httpx.AsyncClient(timeout=180)
    # The token is installed only for this agent's task calls and removed afterwards, so a
    # caller-supplied client never carries it to discovery or to another origin.
    previous = http.headers.pop("Authorization", None)
    try:
        # 1. Discovery: the card is public, so it is fetched before any credential is sent.
        card = await A2ACardResolver(http, url).get_agent_card()
        out(f"[discover] {card.name} v{card.version}")
        for skill in card.skills:
            out(f"[discover]   skill {skill.id}: tags={list(skill.tags)}")
        out(f"[discover]   auth: {', '.join(card.security_schemes)}")
        if not capable(card):
            out("[discover] no skill tagged 'contract'; nothing to delegate")
            return 2

        http.headers["Authorization"] = f"Bearer {token}"
        client = await create_client(card, ClientConfig(httpx_client=http, streaming=True))

        parts = [new_text_part(question)]
        if document_ids:
            parts.append(new_data_part({"document_ids": document_ids}))
        message = new_message(parts, role=Role.ROLE_USER)
        out(f"[send] {question}")

        while True:
            state, task_id, context_id, prompt = await stream(client, message, out)
            if state != TaskState.TASK_STATE_INPUT_REQUIRED:
                return 0 if state == TaskState.TASK_STATE_COMPLETED else 1
            # 4. input-required: answer on the same task with document ids only.
            if prompt is not None:
                out(f"[input-required] {get_message_text(prompt)}")
                for data in get_data_parts(prompt.parts):
                    docs = data.get("documents", []) if isinstance(data, dict) else []
                    out(f"[input-required]   {len(docs)} candidate documents, e.g.")
                    for doc in docs[:5]:
                        out(f"[input-required]     {doc['document_id']}  {doc['title']}")
            reply = clarify_with
            clarify_with = None  # answer automatically once; a second ask goes to the user
            if reply is None and ask is not None:
                answer = ask()
                # Empty input counts as no answer; the agent asks again or gives up.
                reply = [answer.strip()] if answer and answer.strip() else []
            if reply is None:
                out("[input-required] no --clarify-with given; stopping")
                return 1
            out(f"[clarify] document_ids={reply}")
            message = new_message(
                [new_data_part({"document_ids": reply})],
                context_id=context_id,
                task_id=task_id,
                role=Role.ROLE_USER,
            )
    finally:
        http.headers.pop("Authorization", None)
        if previous is not None:
            http.headers["Authorization"] = previous
        if owned:
            await http.aclose()


async def stream(
    client: Any, message: Message, out: Callable[[str], None]
) -> tuple[int, str, str, Message | None]:
    """Send one message and print events until the task stops or needs input."""
    state, task_id, context_id, prompt = TaskState.TASK_STATE_UNSPECIFIED, "", "", None
    async for event in client.send_message(SendMessageRequest(message=message)):
        if event.HasField("task"):
            task_id, context_id = event.task.id, event.task.context_id
            state = event.task.status.state
            out(f"[task] {task_id} {TaskState.Name(state)}")
        elif event.HasField("status_update"):
            update = event.status_update
            task_id, context_id = update.task_id, update.context_id
            state = update.status.state
            note = get_message_text(update.status.message) if update.status.message.parts else ""
            out(f"[status] {TaskState.Name(state)} {note}".rstrip())
            if state == TaskState.TASK_STATE_INPUT_REQUIRED:
                prompt = update.status.message
            if state == TaskState.TASK_STATE_FAILED:
                for data in get_data_parts(update.status.message.parts):
                    out(f"[failed] {data}")
        elif event.HasField("artifact_update"):
            print_artifact(event.artifact_update.artifact, out)
        if state in TERMINAL or state == TaskState.TASK_STATE_INPUT_REQUIRED:
            break
    return state, task_id, context_id, prompt


def print_artifact(artifact: Any, out: Callable[[str], None]) -> None:
    out(f"[artifact] {artifact.name}")
    for part in artifact.parts:
        if part.HasField("text"):
            out(f"[answer] {part.text}")
    for data in get_data_parts(artifact.parts):
        out(
            f"[answer] status={data.get('status')} run_id={data.get('run_id')} "
            f"simulated={data.get('simulated')}"
        )
        for citation in data.get("citations", []):
            out(f"[cite] {citation.get('display')}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question")
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument("--document-id", action="append", dest="document_ids")
    parser.add_argument("--clarify-with", action="append", dest="clarify_with")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    token = get_settings().a2a_token
    if token is None:
        print("Set CRA_A2A_TOKEN (same value as the server).", file=sys.stderr)
        return 2

    def ask() -> str | None:
        return input("document_id> ") if sys.stdin.isatty() else None

    return asyncio.run(
        run(
            args.url,
            args.question,
            token=token.get_secret_value(),
            document_ids=args.document_ids,
            clarify_with=args.clarify_with,
            ask=ask,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
