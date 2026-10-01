"""Export finished V2 runs from their durable journals to an OTLP backend (e.g. Langfuse).

Needs the optional extra (`uv sync --locked --extra otel`) and Langfuse settings in
`.env` or standard OTEL_EXPORTER_OTLP_* environment variables. Each run is exported
at most once; its `telemetry.json` records the trace id for local/remote correlation.

    uv run python -m scripts.export_run_telemetry runs/<run_id> [runs/<run_id> ...]
    uv run python -m scripts.export_run_telemetry --latest 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import Settings
from app.observability import telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--latest", type=int, default=0, help="export the N newest runs")
    args = parser.parse_args()
    settings = Settings()
    directories = list(args.runs)
    if args.latest:
        journals = sorted(settings.runs_dir.glob("*/events.jsonl"), key=lambda p: p.stat().st_mtime)
        directories += [p.parent for p in journals[-args.latest :]]
    if not directories:
        raise SystemExit("Name run directories or pass --latest N.")
    telemetry.configure(True, timeout_seconds=settings.telemetry_timeout_s, settings=settings)
    exporter = telemetry._exporter
    assert exporter is not None
    for directory in directories:
        marker = directory / telemetry.MARKER
        trace_id = exporter.export(directory / "events.jsonl")
        if trace_id:
            print(f"{directory.name}: exported trace {trace_id}")
        elif marker.exists():
            previous = json.loads(marker.read_text())["trace_id"]
            print(f"{directory.name}: already exported as trace {previous}")
        else:
            print(f"{directory.name}: skipped (no terminal event yet)")
    # atexit flushes the batch processor within the configured timeout.


if __name__ == "__main__":
    main()
