"""OTel projection of durable V2 runs: nesting, allowlist, idempotence, failure isolation.

Tests needing the optional `otel` extra skip without it; CI runs them in a separate job.
"""

# ruff: noqa: ASYNC240 -- tiny local fixture artifacts
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from app.agent.state import ResearchState
from app.agent.tool_registry import ToolRegistry
from app.config import Settings
from app.observability import telemetry
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk
from tests.unit.test_v2 import action, registry, run

__all__ = ["registry"]  # re-exported pytest fixture

REPO = Path(__file__).resolve().parents[2]
QUESTION = "When is notice due?"


@pytest.fixture(autouse=True)
def reset_exporter() -> Iterator[None]:
    telemetry.install(None)
    yield
    telemetry.install(None)


def test_disabled_by_default_imports_no_sdk() -> None:
    code = (
        "import sys; from app.config import Settings; from app.observability import telemetry; "
        "s = Settings(_env_file=None); "
        "assert telemetry.configure(s.telemetry_export) is False; "
        "assert telemetry._exporter is None; "
        "print(any(m.startswith('opentelemetry.sdk') or m.startswith('opentelemetry.exporter') "
        "for m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin", "CRA_LIVE_MODEL": "0"},
    )
    assert completed.stdout.strip() == "False"


def test_enabled_without_extra_is_a_clear_setup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    with pytest.raises(telemetry.TelemetrySetupError, match="--extra otel"):
        telemetry.configure(True)


def test_disabled_export_does_not_validate_or_contact_langfuse() -> None:
    settings = Settings(_env_file=None, LANGFUSE_BASE_URL="https://cloud.langfuse.com")
    assert telemetry.configure(False, settings=settings) is False
    assert telemetry._exporter is None


def test_unconfigured_langfuse_preserves_generic_otlp_environment() -> None:
    assert telemetry.langfuse_connection(Settings(_env_file=None)) == {}


@pytest.mark.parametrize(
    "values",
    [
        {"LANGFUSE_BASE_URL": "https://cloud.langfuse.com"},
        {"LANGFUSE_PUBLIC_KEY": "pk-synthetic", "LANGFUSE_SECRET_KEY": "sk-synthetic"},
        {"LANGFUSE_BASE_URL": "https://cloud.langfuse.com", "LANGFUSE_SECRET_KEY": "sk-synthetic"},
    ],
)
def test_incomplete_langfuse_credentials_fail_clearly(values: dict[str, str]) -> None:
    settings = Settings(_env_file=None, **values)
    with pytest.raises(telemetry.TelemetrySetupError) as failure:
        telemetry.configure(True, settings=settings)
    message = str(failure.value)
    assert "LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY" in message
    assert "pk-synthetic" not in message and "sk-synthetic" not in message


# --- with the optional extra -------------------------------------------------------


def memory_exporter() -> Any:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    telemetry.install(telemetry.RunExporter(provider.get_tracer("test")))
    return spans


def everything_exported(spans: list[Any]) -> str:
    values: list[Any] = []
    for span in spans:
        values += [span.name, *span.attributes.values()]
        for event in span.events:
            values += [event.name, *event.attributes.values()]
    return json.dumps(values, default=str)


async def test_run_becomes_one_nested_trace(registry: ToolRegistry, tmp_path: Path) -> None:
    spans = memory_exporter()
    state, _ = await run(
        registry, tmp_path, [action("document_search", query="notice"), action("draft")]
    )
    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.parent is None]
    assert len(roots) == 1
    root = roots[0]
    assert root.attributes["cra.run_id"] == state["run_id"]
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["cra.status"] == "completed"
    assert root.attributes["gen_ai.usage.input_tokens"] == state["input_tokens"]
    children = [s for s in finished if s.parent is not None]
    assert all(s.parent.span_id == root.context.span_id for s in children)
    assert all(s.context.trace_id == root.context.trace_id for s in children)
    names = sorted(s.name for s in children)
    assert names == sorted(
        [
            "chat scripted-v2-fixture",  # search decision
            "chat scripted-v2-fixture",  # draft decision
            "chat scripted-v2-fixture",  # writer
            "execute_tool document_search",
            "verify_citations",
        ]
    )
    tool = next(s for s in children if s.name.startswith("execute_tool"))
    assert tool.attributes["gen_ai.tool.name"] == "document_search"
    admitted = [e for e in tool.events if e.name == "evidence_admitted"]
    assert {e.attributes["cra.evidence_id"] for e in admitted} == set(state["evidence_by_id"])
    chats = [s for s in children if s.name.startswith("chat")]
    assert sum(s.attributes["gen_ai.usage.input_tokens"] for s in chats) == state["input_tokens"]
    for span in finished:
        assert span.start_time <= span.end_time
        assert root.start_time <= span.start_time and span.end_time <= root.end_time
    marker = json.loads((Path(state["trace_path"]).parent / telemetry.MARKER).read_text())
    assert marker["trace_id"] == format(root.context.trace_id, "032x")


async def test_payloads_follow_the_allowlist(registry: ToolRegistry, tmp_path: Path) -> None:
    spans = memory_exporter()
    state, _ = await run(
        registry,
        tmp_path,
        [action("document_search", query="notice"), action("draft")],
        answers=["The term is 30 days. [doc-900, p. 1, §1]"],
    )
    exported = everything_exported(spans.get_finished_spans())
    for private in (
        QUESTION,
        "Notice must be sent",  # contract text
        "The term is 30 days",  # answer
        '"notice"',  # tool arguments
        "PLANNER",
    ):
        assert private not in exported, private
    assert state["run_id"] in exported


async def test_error_paths_do_not_leak_exception_text(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    spans = memory_exporter()

    class Broken:
        name = "broken"

        async def search(
            self, query: str, *, top_k: int, filters: RetrievalFilters | None = None
        ) -> list[RetrievedChunk]:
            raise RuntimeError("provider payload sk-secret must not leak")

    registry.retriever = Broken()
    await run(registry, tmp_path, [action("document_search", query="notice")])
    finished = spans.get_finished_spans()
    exported = everything_exported(finished)
    assert "provider payload" not in exported and "sk-secret" not in exported
    tool = next(s for s in finished if s.name.startswith("execute_tool"))
    assert tool.status.status_code.name == "ERROR"
    assert tool.attributes["error.type"] == "tool_unavailable"
    root = next(s for s in finished if s.parent is None)
    assert root.status.status_code.name == "ERROR"


async def test_each_run_is_exported_once(registry: ToolRegistry, tmp_path: Path) -> None:
    spans = memory_exporter()
    state, _ = await run(
        registry, tmp_path, [action("document_search", query="notice"), action("draft")]
    )
    count = len(spans.get_finished_spans())
    assert telemetry._exporter is not None
    assert telemetry._exporter.export(Path(state["trace_path"])) is None
    assert len(spans.get_finished_spans()) == count


async def test_exporter_failure_does_not_change_the_run(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    class Exploding(telemetry.RunExporter):
        def __init__(self) -> None:
            pass

        def export(self, trace_path: Path) -> str | None:
            raise ConnectionError("collector down")

    telemetry.install(Exploding())
    state, _ = await run(
        registry, tmp_path, [action("document_search", query="notice"), action("draft")]
    )
    assert state["status"] == "completed" and state["verification_ok"]
    rows = Path(state["trace_path"]).read_text().splitlines()
    assert json.loads(rows[-1])["event_type"] == "run_completed"


# --- real OTLP/HTTP export -------------------------------------------------------------


class Receiver(BaseHTTPRequestHandler):
    bodies: ClassVar[list[tuple[str, bytes, str]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        Receiver.bodies.append(
            (self.path, self.rfile.read(length), self.headers.get("Authorization", ""))
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


def otlp_exporter(endpoint: str, timeout: float) -> Any:
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint, timeout=timeout, headers={"Authorization": "Basic dGVzdA=="}
            ),
            export_timeout_millis=int(timeout * 1000),
        )
    )
    telemetry.install(telemetry.RunExporter(provider.get_tracer("test")))
    return provider


async def run_with(registry: ToolRegistry, tmp_path: Path) -> ResearchState:
    state, _ = await run(
        registry, tmp_path, [action("document_search", query="notice"), action("draft")]
    )
    return state


@pytest.mark.loopback
async def test_otlp_http_export_reaches_a_local_collector(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    pytest.importorskip("opentelemetry.proto")
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    Receiver.bodies = []
    server = HTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = otlp_exporter(f"http://127.0.0.1:{server.server_port}/v1/traces", 5)
        state = await run_with(registry, tmp_path)
        assert provider.force_flush(10_000)
        provider.shutdown()
    finally:
        server.shutdown()
        server.server_close()
    assert Receiver.bodies and Receiver.bodies[0][0] == "/v1/traces"
    assert Receiver.bodies[0][2] == "Basic dGVzdA=="
    request = ExportTraceServiceRequest()
    request.ParseFromString(b"".join(body for _, body, _ in Receiver.bodies[:1]))
    names = [s.name for rs in request.resource_spans for ss in rs.scope_spans for s in ss.spans]
    assert "invoke_agent contract-research-v2" in names
    assert "execute_tool document_search" in names
    raw = b"".join(body for _, body, _ in Receiver.bodies)
    assert state["run_id"].encode() in raw
    assert QUESTION.encode() not in raw and b"Notice must be sent" not in raw


@pytest.mark.loopback
async def test_langfuse_dotenv_settings_reach_otlp_receiver(
    registry: ToolRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    import opentelemetry.sdk.trace as otel_trace

    providers: list[Any] = []
    provider_type = otel_trace.TracerProvider

    def capture_provider(**kwargs: Any) -> Any:
        provider = provider_type(**kwargs)
        providers.append(provider)
        return provider

    monkeypatch.setattr(otel_trace, "TracerProvider", capture_provider)
    for key in ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    # A typed Langfuse destination must override an unrelated ambient OTLP destination.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:1/wrong")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=wrong")
    Receiver.bodies = []
    server = HTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"LANGFUSE_BASE_URL=http://127.0.0.1:{server.server_port}/\n"
        "LANGFUSE_PUBLIC_KEY=public\nLANGFUSE_SECRET_KEY=secret\nCRA_TELEMETRY_EXPORT=1\n"
    )
    try:
        settings = Settings(_env_file=env_file)
        assert telemetry.configure(settings.telemetry_export, settings=settings)
        connection = telemetry.langfuse_connection(settings)
        assert connection["headers"]["x-langfuse-ingestion-version"] == "4"
        state = await run_with(registry, tmp_path)
        assert state["status"] == "completed"
        assert providers[0].force_flush(10_000)
    finally:
        for provider in providers:
            provider.shutdown()
        server.shutdown()
        server.server_close()
    assert Receiver.bodies
    assert Receiver.bodies[0][0] == "/api/public/otel/v1/traces"
    assert Receiver.bodies[0][2] == "Basic cHVibGljOnNlY3JldA=="


@pytest.mark.loopback
async def test_unreachable_collector_is_bounded_and_harmless(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # closed again: nothing listens here
    provider = otlp_exporter(f"http://127.0.0.1:{port}/v1/traces", 0.5)
    started = time.monotonic()
    state = await run_with(registry, tmp_path)
    assert state["status"] == "completed"
    provider.shutdown()
    assert time.monotonic() - started < 15
    rows = Path(state["trace_path"]).read_text().splitlines()
    assert json.loads(rows[-1])["event_type"] == "run_completed"
