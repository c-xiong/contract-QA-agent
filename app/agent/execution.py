"""One durable execution boundary for CLI/API and streaming consumers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import cast
from uuid import uuid4

from app.agent.graph import ResearchGraph
from app.agent.state import ResearchState
from app.observability.events import emit


def close_incomplete(state: ResearchState, status: str, reason: str) -> None:
    path = Path(state["trace_path"])
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    if not lines:
        return
    last = json.loads(lines[-1])
    if last["event_type"] in {"run_completed", "run_abstained", "run_failed", "run_cancelled"}:
        return
    snapshot = last["state"]
    snapshot.update(status=status, stop_reason=reason, final_answer=None, usage_known=False)
    emit(snapshot, "run_" + status, "terminal")


async def v2_states(
    graph: ResearchGraph, state: ResearchState, runs_dir: Path
) -> AsyncGenerator[ResearchState, None]:
    # Allocate identity before graph initialization so cancellation between nodes can
    # still locate the trace. External request schemas cannot supply these fields.
    state["run_id"] = uuid4().hex
    state["trace_path"] = str(runs_dir / state["run_id"] / "events.jsonl")
    status, reason = "cancelled", "consumer_disconnected"
    try:
        async for merged in graph.astream(
            state, stream_mode="values", config={"recursion_limit": 100}
        ):
            state = cast(ResearchState, merged)
            yield state
    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception:
        status, reason = "failed", "execution_interrupted"
        raise
    finally:
        # Synchronous final append must survive cancellation of the consumer itself.
        close_incomplete(state, status, reason)
