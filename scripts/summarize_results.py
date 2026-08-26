"""Render the pinned artifacts in results/ as a readable Markdown report.

Generated from the JSON rather than written by hand, so the report cannot drift from
the data behind it. Re-run it after any run you promote into results/.

    uv run python scripts/summarize_results.py            # writes results/RESULTS.md
    uv run python scripts/summarize_results.py --stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RESULTS = Path("results")

# Which pinned file backs which claim. Anything else in results/ is reported as extra
# rather than silently ignored.
KNOWN = {
    "eval-hand-rerank.json": "Full eval suite",
    "experiment-a-retrieval.json": "Experiment A — retrieval comparison",
    "experiment-b-agentic.json": "Experiment B — single-pass vs agentic",
    "experiment-c-citation-gate.json": "Experiment C — citation gate ablation",
}


def provenance(payload: dict[str, Any]) -> str:
    """One line saying how far the numbers in this artifact can be trusted."""
    version = payload.get("dataset_version", "?")
    generated = payload.get("questions_generated")
    if generated or str(version).startswith("provisional-generated"):
        return (
            "**Questions were model-generated.** These figures are a pipeline check, "
            "not a capability measurement."
        )
    if "arms" in payload:
        # Retrieval-only: no model is involved at all, so there is no live/stub axis.
        return (
            "Hand-written questions, expert-annotated evidence. Retrieval only — this "
            "experiment calls no model, so it is free and fully reproducible."
        )
    model = payload.get("model_id")
    if model == "stub":
        return (
            "Ran against the deterministic stub, so retrieval and evidence assembly are "
            "real and answer quality is NOT measured."
        )
    return f"Hand-written questions, expert-annotated evidence, live model `{model}`."


def fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_eval(payload: dict[str, Any]) -> list[str]:
    out = [
        f"- suite `{payload['suite']}` · dataset `{payload['dataset_version']}`",
        f"- n = {payload['task_count']} · model `{payload['model_id']}` "
        f"· retriever `{payload['retriever']}` · k = {payload['top_k']}",
        "",
        "| grader | version | mean |",
        "|---|---|---:|",
    ]
    versions = payload.get("grader_versions", {})
    for name, mean in sorted(payload.get("means", {}).items()):
        out.append(f"| `{name}` | `{versions.get(name, '?')}` | {fmt(mean)} |")
    return out


def render_experiment_a(payload: dict[str, Any]) -> list[str]:
    excluded = payload.get("abstain_excluded", 0)
    out = [
        f"- n = {payload['task_count']} scored"
        + (f" ({excluded} abstain tasks excluded)" if excluded else "")
        + f" · k = {payload['k']} · allowlist honoured: {payload.get('respect_allowlist')}",
        f"- corpus {payload['corpus']['documents']} documents / "
        f"{payload['corpus']['chunks']:,} chunks · calls no model",
        "",
        "| arm | document recall | evidence span recall | MRR |",
        "|---|---:|---:|---:|",
    ]
    for arm in payload.get("arms", {}).values():
        m = arm["means"]
        out.append(
            f"| `{arm['name']}` | {fmt(m['document_recall'])} "
            f"| {fmt(m['evidence_span_recall'])} | {fmt(m['reciprocal_rank'])} |"
        )
    return out


def render_conditions(payload: dict[str, Any]) -> list[str]:
    conditions = payload.get("conditions", {})
    if not conditions:
        return []
    metrics = sorted({m for c in conditions.values() for m in c.get("means", {})})
    names = list(conditions)

    out = [
        f"- n = {payload['task_count']} · model `{payload.get('model_id')}` "
        f"· retriever `{payload.get('retriever')}`",
    ]
    if payload.get("fault_injection_every"):
        out.append(
            f"- **fault injection: 1 in {payload['fault_injection_every']}** — a test "
            "parameter, not an estimate of how often a model miscites"
        )
    if "unverified_rate" in payload:
        out += [
            "",
            f"- citations emitted with the gate off: **{payload['citations_emitted_ungated']}**",
            f"- failing verification: **{payload['citations_failing_verification']} "
            f"({payload['unverified_rate']:.1%})**",
            f"- failure codes: {payload.get('failure_codes') or 'none'}",
            f"- tasks affected: {len(payload.get('affected_tasks', []))}/{payload['task_count']}",
        ]

    out += [
        "",
        "| metric | " + " | ".join(f"`{n}`" for n in names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for metric in metrics:
        cells = " | ".join(fmt(conditions[n]["means"].get(metric, 0.0)) for n in names)
        out.append(f"| `{metric}` | {cells} |")
    for label, key in (("total tokens", "tokens"), ("total searches", "searches")):
        cells = " | ".join(f"{conditions[n].get(key, 0):,}" for n in names)
        out.append(f"| {label} | {cells} |")
    return out


def render(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    title = KNOWN.get(path.name, path.stem)

    # The eval report stamps `started_at`; the experiments stamp `run_at`.
    when = payload.get("run_at") or payload.get("started_at") or "unknown"
    out = [f"## {title}", "", f"`{path.name}` · run {when}", ""]
    out.append(provenance(payload))
    out.append("")

    if "grader_versions" in payload:
        out += render_eval(payload)
    elif "arms" in payload:
        out += render_experiment_a(payload)
    else:
        out += render_conditions(payload)
    out.append("")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args()

    files = sorted(RESULTS.glob("*.json"))
    if not files:
        print(f"No artifacts in {RESULTS}/.", file=sys.stderr)
        return 1

    lines = [
        "# Results",
        "",
        "Generated by `scripts/summarize_results.py` from the artifacts in this "
        "directory. Do not edit by hand — re-run the script instead, so the report "
        "cannot drift from the data behind it.",
        "",
        "Every figure below is recomputable from the JSON beside it: each artifact "
        "carries its own task count, dataset version, grader versions, per-task scores "
        "and per-category breakdown.",
        "",
    ]
    # Deterministic order: the eval first, then the experiments in letter order.
    order = list(KNOWN)
    files.sort(key=lambda p: (order.index(p.name) if p.name in order else len(order), p.name))
    for path in files:
        lines += render(path)

    text = "\n".join(lines).rstrip() + "\n"
    if args.stdout:
        print(text)
        return 0

    target = RESULTS / "RESULTS.md"
    target.write_text(text, encoding="utf-8")
    print(f"Wrote {target} ({len(files)} artifacts, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
