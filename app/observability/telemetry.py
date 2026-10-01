"""Optional OpenTelemetry export of V2 runs, derived from the durable local trace.

DECISION: spans are built from a run's `events.jsonl` when its terminal event is written,
  using the recorded timestamps. There is no second set of business instrumentation: the
  durable journal stays the source of record and telemetry is a projection of it. A
  failed or slow exporter therefore cannot change answers, runtime policy or local
  logging. Alternative rejected: live spans opened inside graph nodes, which would
  duplicate every emit() site and couple runtime control flow to an exporter.

DECISION: attributes come from an explicit allowlist: IDs, counts, timings, statuses,
  version identifiers and evidence locators. Questions, prompts, answers, contract text,
  tool arguments, credentials and exception messages are never exported. Self-hosting a
  backend does not relax this.

DECISION: export uses a private TracerProvider, not the global one, so spans that
  libraries emit on their own (the MCP SDK has OTel hooks) are not captured here.

Semantic conventions: OpenTelemetry GenAI conventions as of opentelemetry-semantic-
conventions 0.66b0 (`gen_ai.operation.name`, `gen_ai.request.model`,
`gen_ai.provider.name`, `gen_ai.usage.input_tokens`/`output_tokens`, `gen_ai.tool.name`,
`gen_ai.tool.call.id`, `error.type`). Project attributes use the `cra.` prefix.
Langfuse endpoint and credentials can come from typed settings (including `.env`).
Without them, endpoint and headers use the standard OTEL_EXPORTER_OTLP_* environment.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

SERVICE_NAME = "contract-research-agent"
TERMINAL = {"run_completed", "run_abstained", "run_failed", "run_cancelled"}
MARKER = "telemetry.json"
MANIFEST_FIELDS = (
    "prompt_version",
    "tool_version",
    "code_hash",
    "git_revision",
    "corpus_hash",
    "index_hash",
    "retriever",
    "config_hash",
)


class TelemetrySetupError(RuntimeError):
    """Export was requested but the optional telemetry dependencies are missing."""


class RunExporter:
    """Turns one durable run journal into a span tree and hands it to a span processor."""

    def __init__(self, tracer: Any):
        self.tracer = tracer

    def export(self, trace_path: Path) -> str | None:
        directory = trace_path.parent
        marker = directory / MARKER
        if marker.exists():
            return None  # already exported: never emit duplicate spans for one run
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        if not rows or rows[-1]["event_type"] not in TERMINAL:
            return None
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        trace_id = self._emit(rows, manifest)
        marker.write_text(json.dumps({"trace_id": trace_id, "service": SERVICE_NAME}) + "\n")
        marker.chmod(0o600)
        return trace_id

    def _emit(self, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        first, last = rows[0], rows[-1]
        final = last["state"]
        model = manifest.get("model") or "unknown"
        simulated = bool(manifest.get("simulated", True))
        root_attributes: dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "contract-research-v2",
            "gen_ai.request.model": model,
            "cra.run_id": first["run_id"],
            "cra.simulated": simulated,
            "cra.status": final.get("status") or "unknown",
            "cra.stop_reason": final.get("stop_reason") or "none",
            "cra.tool_calls": final.get("tool_calls", 0),
            "cra.model_calls": final.get("model_calls", 0),
            "cra.searches_used": final.get("searches_used", 0),
            "cra.retries": final.get("retries", 0),
            "cra.repair_attempts": final.get("repair_attempts", 0),
            "cra.evidence_items": len(final.get("evidence_by_id") or {}),
            "cra.verification_ok": bool(final.get("verification_ok")),
            "cra.usage_known": bool(final.get("usage_known", True)),
        }
        if final.get("usage_known", True):
            root_attributes["gen_ai.usage.input_tokens"] = final.get("input_tokens", 0)
            root_attributes["gen_ai.usage.output_tokens"] = final.get("output_tokens", 0)
        for field in MANIFEST_FIELDS:
            if manifest.get(field) is not None:
                root_attributes[f"cra.{field}"] = str(manifest[field])
        if not simulated:
            root_attributes["gen_ai.provider.name"] = "anthropic"

        root = self.tracer.start_span(
            "invoke_agent contract-research-v2",
            start_time=ns(first["timestamp"]),
            attributes=root_attributes,
        )
        if final.get("status") == "failed":
            root.set_status(Status(StatusCode.ERROR))
            root.set_attribute("error.type", final.get("stop_reason") or "failed")
        context = trace.set_span_in_context(root)

        def child(
            name: str, start: dict[str, Any], end: dict[str, Any], attributes: dict[str, Any]
        ) -> Any:
            span = self.tracer.start_span(
                name, context=context, start_time=ns(start["timestamp"]), attributes=attributes
            )
            return span, ns(end["timestamp"])

        tool_spans: dict[str, Any] = {}
        for index, row in enumerate(rows):
            kind, data, node = row["event_type"], row["data"], row["node"]
            if kind == "model_started":
                end = next_row(rows, index, node=node)
                attributes = {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": model,
                    "cra.node": node,
                    "cra.attempt": data.get("attempt", 0),
                    "cra.simulated": simulated,
                }
                if not simulated:
                    attributes["gen_ai.provider.name"] = "anthropic"
                if end["state"].get("usage_known", True):
                    for key in ("input_tokens", "output_tokens"):
                        delta = end["state"].get(key, 0) - row["state"].get(key, 0)
                        attributes[f"gen_ai.usage.{key}"] = max(0, delta)
                span, end_time = child(f"chat {model}", row, end, attributes)
                if end["event_type"] in TERMINAL and end["state"].get("status") == "failed":
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("error.type", end["state"].get("stop_reason") or "failed")
                span.end(end_time=end_time)
            elif kind == "tool_started":
                call_id = data.get("call_id", "")
                end = next_row(rows, index, call_id=call_id)
                tool = data.get("tool", "unknown")
                result = (end["state"].get("tool_results_by_call_id") or {}).get(call_id, {})
                attributes = {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool,
                    "gen_ai.tool.call.id": call_id,
                    "cra.attempt": data.get("attempt", 0),
                    "cra.tool_status": result.get("status") or "error",
                }
                span, end_time = child(f"execute_tool {tool}", row, end, attributes)
                code = (result.get("error") or {}).get("code") or end["data"].get("code")
                if end["event_type"] != "tool_completed":
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("error.type", code or "tool_failed")
                tool_spans[call_id] = (span, end_time)
            elif kind == "evidence_added":
                # Admission happens after the tool call; record it on the latest tool span.
                ref = (row["state"].get("evidence_by_id") or {}).get(data.get("evidence_id"), {})
                if tool_spans:
                    span, _ = list(tool_spans.values())[-1]
                    span.add_event(
                        "evidence_admitted",
                        attributes=clean(
                            {
                                "cra.evidence_id": data.get("evidence_id"),
                                "cra.doc_id": ref.get("doc_id"),
                                "cra.page": ref.get("page"),
                                "cra.section": ref.get("section"),
                            }
                        ),
                        timestamp=ns(row["timestamp"]),
                    )
            elif kind == "verification_completed":
                start = rows[index - 1] if index else row
                span, end_time = child(
                    "verify_citations",
                    start,
                    row,
                    {
                        "cra.verification_ok": bool(data.get("ok")),
                        "cra.mandatory": bool(data.get("mandatory")),
                        "cra.citation_errors": len(row["state"].get("citation_errors") or []),
                    },
                )
                if not data.get("ok"):
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("error.type", "citation_verification_failed")
                span.end(end_time=end_time)
            elif kind == "repair_started":
                root.add_event(
                    "repair_started",
                    attributes={"cra.attempt": data.get("attempt", 0)},
                    timestamp=ns(row["timestamp"]),
                )
        for span, end_time in tool_spans.values():
            span.end(end_time=end_time)
        root.end(end_time=ns(last["timestamp"]))
        return format(root.get_span_context().trace_id, "032x")


def next_row(
    rows: list[dict[str, Any]],
    index: int,
    *,
    node: str | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """The event closing a span: same node (model calls) or same call id (tools)."""
    for row in rows[index + 1 :]:
        if row["event_type"] in TERMINAL:
            return row
        if node is not None and row["node"] == node:
            return row
        if (
            call_id is not None
            and row["data"].get("call_id") == call_id
            and row["event_type"] in ("tool_completed", "tool_failed")
        ):
            return row
    return rows[-1]


def ns(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp).timestamp() * 1_000_000_000)


def clean(attributes: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attributes.items() if v is not None}


_exporter: RunExporter | None = None


def on_terminal(trace_path: Path) -> None:
    """Hook called by emit() after a terminal event is durably written."""
    if _exporter is None:
        return
    try:
        _exporter.export(trace_path)
    except Exception:  # noqa: BLE001 -- telemetry must never break the run or its log
        logger.warning("Telemetry export skipped for %s", trace_path.parent.name)


def langfuse_connection(settings: Settings | None) -> dict[str, Any]:
    """Resolve typed Langfuse settings without changing the process environment."""
    if settings is None:
        return {}
    values = (
        settings.langfuse_base_url,
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
    )
    if not any(values):
        return {}  # Generic OTLP export retains the SDK's standard environment handling.
    if not all(values):
        raise TelemetrySetupError(
            "Langfuse export requires LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, "
            "and LANGFUSE_SECRET_KEY. Fill all three or leave all three blank for generic OTLP."
        )
    assert settings.langfuse_base_url is not None
    assert settings.langfuse_public_key is not None
    assert settings.langfuse_secret_key is not None
    credentials = (
        f"{settings.langfuse_public_key.get_secret_value()}:"
        f"{settings.langfuse_secret_key.get_secret_value()}"
    )
    authorization = base64.b64encode(credentials.encode()).decode("ascii")
    return {
        "endpoint": f"{settings.langfuse_base_url}/api/public/otel/v1/traces",
        "headers": {
            "Authorization": f"Basic {authorization}",
            "x-langfuse-ingestion-version": "4",
        },
    }


def configure(
    enabled: bool, *, timeout_seconds: float = 5.0, settings: Settings | None = None
) -> bool:
    """Install the exporter once per process. Returns whether export is active.

    Disabled (the default) this imports nothing and makes no network requests.
    """
    global _exporter
    if not enabled:
        return False
    if _exporter is not None:
        return True
    connection = langfuse_connection(settings)
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise TelemetrySetupError(
            "CRA_TELEMETRY_EXPORT=1 needs the optional extra: uv sync --locked --extra otel"
        ) from exc
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    # Bounded queue and timeouts: a slow or absent backend drops spans, never blocks runs.
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(timeout=timeout_seconds, **connection),
            max_queue_size=2048,
            export_timeout_millis=int(timeout_seconds * 1000),
        )
    )
    atexit.register(provider.shutdown)
    install(RunExporter(provider.get_tracer(__name__)))
    return True


def install(exporter: RunExporter | None) -> None:
    """Install (or remove, with None) the run exporter; tests pass in-memory tracers."""
    global _exporter
    _exporter = exporter
