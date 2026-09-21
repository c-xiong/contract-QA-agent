"""Durable private state snapshots with a deliberately small public event surface."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.agent.tool_registry import digest


def json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def emit(state: Any, event_type: str, node: str, **data: Any) -> dict[str, Any]:
    state["last_event_sequence"] = state.get("last_event_sequence", 0) + 1
    event = {
        "run_id": state["run_id"],
        "sequence": state["last_event_sequence"],
        "event_id": f"{state['run_id']}:{state['last_event_sequence']}",
        "span_id": data.get("call_id", node),
        "parent_span": state["run_id"],
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "node": node,
        "data": json_value(data),
    }
    # DECISION: snapshots are protected local artifacts. They include source text,
    # never credentials, prompts, or hidden chain-of-thought. SSE only sees public().
    event["state"] = json_value({k: v for k, v in state.items() if k != "trace"})
    path = Path(state["trace_path"])
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def public(event: dict[str, Any]) -> dict[str, Any]:
    state = event.get("state", {})
    data = event.get("data", {})
    call_id = data.get("call_id")
    result = state.get("tool_results_by_call_id", {}).get(call_id, {})
    action = state.get("next_action", {})
    tool = data.get("tool") or result.get("tool")
    if event["node"] in ("decide", "authorize"):
        tool = action.get("tool")
    budget = state.get("budget", {})
    # Explicitly select fields: arguments, source text, prompts and state stay private.
    return {k: event[k] for k in ("run_id", "sequence", "event_type", "node")} | {
        "status": state.get("status"),
        "stop_reason": state.get("stop_reason"),
        "tool": tool,
        "call_id": call_id,
        "reason": action.get("reason") if event["node"] in ("decide", "authorize") else None,
        "tool_status": result.get("status"),
        "error_code": (result.get("error") or {}).get("code") or data.get("code"),
        "elapsed_ms": result.get("elapsed_ms", data.get("elapsed_ms")),
        "verification_ok": data.get("ok")
        if event["event_type"] == "verification_completed"
        else None,
        "tool_calls": state.get("tool_calls", 0),
        "model_calls": state.get("model_calls", 0),
        "searches_used": state.get("searches_used", 0),
        "repair_attempts": state.get("repair_attempts", 0),
        "retries": state.get("retries", 0),
        "limits": {
            key: budget.get(key)
            for key in (
                "max_tool_calls",
                "max_model_calls",
                "max_searches",
                "max_repair_attempts",
                "max_evidence_items",
                "deadline_seconds",
            )
        },
        "input_tokens": state.get("input_tokens") if state.get("usage_known", True) else None,
        "output_tokens": state.get("output_tokens") if state.get("usage_known", True) else None,
        "evidence_ids": sorted(state.get("evidence_by_id", {})),
        "evidence_refs": [
            {
                key: ref.get(key)
                for key in ("evidence_id", "doc_id", "version_id", "chunk_id", "page", "section")
            }
            for ref in state.get("evidence_by_id", {}).values()
        ],
    }


def code_hash() -> str:
    return digest(
        {
            str(p): digest(p.read_text())
            for root in ("app", "evals")
            for p in sorted(Path(root).rglob("*.py"))
        }
    )


def recover_interrupted(runs_dir: Path) -> int:
    """Mark dead-process traces failed on startup; never resume or repeat their work."""
    count = 0
    for path in runs_dir.glob("*/manifest.json"):
        manifest = json.loads(path.read_text())
        pid = manifest.get("pid")
        if not isinstance(pid, int):
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            event_path = path.with_name("events.jsonl")
            if not event_path.exists():
                continue
            lines = event_path.read_text().splitlines()
            if not lines:
                continue
            last = json.loads(lines[-1])
            if last["event_type"] in {
                "run_completed",
                "run_abstained",
                "run_failed",
                "run_cancelled",
            }:
                continue
            state = last["state"]
            state.update(
                status="failed",
                stop_reason="process_interrupted",
                final_answer=None,
                trace_path=str(event_path),
                usage_known=False,
            )
            emit(state, "run_failed", "terminal")
            count += 1
        except PermissionError:
            continue  # A process we cannot inspect must not be assumed dead.
    return count


def index_hash(directory: Path) -> str | None:
    paths = sorted(p for p in directory.rglob("*") if p.is_file())
    if not paths:
        return None
    result = hashlib.sha256()
    for path in paths:
        result.update(str(path.relative_to(directory)).encode())
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                result.update(block)
    return result.hexdigest()
