"""Paired V1/V2 comparison report from two standalone reports with frozen human labels.

    uv run python -m scripts.compare_v1_v2 reports/v1-standalone-hand-live.json \
        reports/v2-standalone-hand-live.json --output reports/v1v2-hand-live \
        [--v1-sealed reports/v1-standalone-hand-live.judge-frozen.sealed.jsonl \
         --v2-sealed reports/v2-standalone-hand-live.judge-frozen.sealed.jsonl]

Writes `<output>.comparison.json` and a readable `<output>.comparison.md`. Refuses to
run until both label sets are frozen against the exact report bytes. Judge verdicts,
when supplied, are reported only as evaluator agreement; they never change the outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.v2_labels import compare as judge_agreement
from scripts.v2_labels import paths as report_paths

from evals.paired import MODES, Mode, compare
from evals.v2_labels import HumanLabel, apply_adjudications, check_frozen, load_labels

DEFAULT_SEED = 20261006


def limitations(result: dict[str, Any], frozen: dict[Mode, dict[str, Any]]) -> list[str]:
    reviewer_types = {t for record in frozen.values() for t in record["reviewer_types"]}
    notes = [
        f"One trial per arm over {result['task_count']} hand-v1 development cases; hand-v1 "
        "is not held-out and the author has seen earlier outputs.",
        "Labels are "
        + ("author-reviewed" if reviewer_types == {"author"} else "from mixed reviewers")
        + "; blinding hides mode names, stop reasons and traces, but answer style, V2's "
        "fixed abstention wording, evidence lists and run status can reveal the mode.",
        "The sign test and bootstrap interval are descriptive context for a single trial, "
        "not a significance claim. Category cells have very small n.",
        "The treatment is the whole V2 design (planner, typed tools, stricter structural "
        "citation gate, routing), not agentic retrieval alone.",
        "V1 tool calls are not instrumented as registry invocations: null, never zero.",
        "Runtime and any judge share a model family; separate prompts do not imply "
        "independent errors. The judge is calibrated on V2 answers only, so V1 agreement "
        "on the calibration-split case IDs is not in-sample tuning agreement.",
    ]
    reviewers = sorted({r for record in frozen.values() for r in record["reviewers"]})
    notes.append("Recorded label reviewers (verbatim): " + "; ".join(reviewers) + ".")
    if result["comparability"]["shared_config"].get("model_backend") == "claude_code":
        versions = result["comparability"]["claude_code_version"]
        notes.append(
            "Both arms ran through the Claude Code CLI (subscription): a fixed ~650-token "
            "harness preamble per call, no max_tokens control, adaptive thinking at the "
            "pinned effort; token counts include it, latency includes process start. CLI "
            f"versions: V1 {versions['v1']}, V2 {versions['v2']}"
            + ("." if versions["v1"] == versions["v2"] else " (differ; a possible confound).")
        )
    if not result["reporting_eligible"]:
        notes.insert(0, "NOT reporting-eligible: stub or provisional inputs; wiring check only.")
    return notes


def pct(block: dict[str, Any]) -> str:
    rate = block.get("rate")
    return f"{block['count']}/{block['denominator']}" + (
        f" ({rate:.0%})" if rate is not None else ""
    )


def num(value: Any) -> str:
    if value is None:
        return "null"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def markdown(output: dict[str, Any]) -> str:
    r = output["comparison"]
    shared = r["comparability"]["shared_config"]
    allp = r["paired_task_success"]["all"]
    boot = allp["bootstrap"]
    interval = (
        "[" + ", ".join(f"{x:+.2f}" for x in boot["interval_95"]) + "]"
        if boot["interval_95"]
        else "n/a"
    )
    lines = [
        f"# V1 vs V2 paired comparison ({r['task_count']} cases, {r['trial_count']} trial)",
        "",
        "Reporting-eligible: **" + ("yes" if r["reporting_eligible"] else "no") + "**",
        "",
        "## Limitations",
        "",
        *[f"- {x}" for x in output["limitations"]],
        "",
        "## Comparability",
        "",
        "| Key | V1 | V2 |",
        "|---|---|---|",
        *[
            f"| {k} | {r['comparability']['v1_config'][k]} | {r['comparability']['v2_config'][k]} |"
            for k in r["comparability"]["differing_keys"]
        ],
        "",
        "Shared: "
        + ", ".join(
            f"{name} `{str(shared.get(key))[:12] if key.endswith(('sha256', 'hash')) else shared.get(key)}`"
            for name, key in (
                ("model", "model_id"),
                ("arm", "retrieval_arm"),
                ("live", "live_model"),
                ("dataset", "dataset_sha256"),
                ("corpus", "corpus_hash"),
                ("index", "index_hash"),
            )
        )
        + ".",
        "",
        "## Task success (human labels, paired)",
        "",
        f"V1 {pct(allp['v1_success'])}, V2 {pct(allp['v2_success'])}. "
        f"V2 wins {allp['v2_wins']}, losses {allp['v2_losses']}, ties {allp['ties']} "
        f"(both succeed {allp['both_succeed']}, both fail {allp['both_fail']}).",
        "",
        f"Descriptive only: exact sign test p = {allp['sign_test_p_two_sided']:.3f} on "
        f"{allp['v2_wins'] + allp['v2_losses']} discordant pairs; paired bootstrap difference "
        f"{num(boot['difference'])}, 95% interval {interval} "
        f"({boot['resamples']} resamples, seed {boot['seed']}).",
        "",
        "| Category | n | V1 success | V2 success | V2 W/L/T |",
        "|---|---|---|---|---|",
    ]
    for key, block in r["paired_task_success"].items():
        if key == "all":
            continue
        lines.append(
            f"| {key.removeprefix('category:')} | {block['n']} | {pct(block['v1_success'])} | "
            f"{pct(block['v2_success'])} | {block['v2_wins']}/{block['v2_losses']}/{block['ties']} |"
        )
    q = r["human_quality"]
    lines += [
        "",
        "## Answer quality by mode (human labels)",
        "",
        "| Metric | V1 | V2 |",
        "|---|---|---|",
        *[
            f"| {metric} | {pct(q['v1']['all'][metric])} | {pct(q['v2']['all'][metric])} |"
            for metric in (
                "task_success",
                "false_abstention",
                "unsupported_answer",
                "execution_failure",
                "claims_supported",
                "claims_contradicted",
            )
        ],
        "",
        "## Outcome transitions (V1 -> V2)",
        "",
        "| Transition | Cases |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in r["outcome_transitions"].items()],
        "",
        "## Paired diagnostics (V2 W/L/T over cases where both values are known)",
        "",
        "| Metric | V2 wins | V2 losses | Ties | n | Excluded |",
        "|---|---|---|---|---|---|",
        *[
            f"| {k} | {v['v2_wins']} | {v['v2_losses']} | {v['ties']} | {v['n']} | "
            f"{v['excluded_unknown']} |"
            for k, v in (
                {"human_claim_support_rate": r["paired_claim_support_rate"]}
                | r["paired_diagnostics"]
            ).items()
        ],
        "",
        "## Cost (failures included)",
        "",
        "| Metric | V1 median | V2 median | Median paired V2 - V1 (n) |",
        "|---|---|---|---|",
    ]
    cost = r["cost"]
    for key, diff in cost["paired_differences"].items():
        lines.append(
            f"| {key} | {num(cost['v1'][f'median_{key}'])} | {num(cost['v2'][f'median_{key}'])} | "
            f"{num(diff['median_v2_minus_v1'])} ({diff['n']}) |"
        )
    lines += ["", "## Primary failures", "", "| Failure | V1 | V2 |", "|---|---|---|"]
    failures = r["primary_failures"]
    for name in sorted(set(failures["v1"]) | set(failures["v2"])):
        lines.append(f"| {name} | {failures['v1'].get(name, 0)} | {failures['v2'].get(name, 0)} |")
    for title, key in (
        ("V2 wins", "v2_wins"),
        ("V2 losses", "v2_losses"),
        ("Both fail", "both_fail"),
    ):
        lines += ["", f"## {title}", ""]
        cases = r["cases"][key]
        if not cases:
            lines.append("None.")
        for c in cases:
            lines += [
                f"- **{c['task_id']}** ({c['category']}): "
                f"V1 {c['v1']['outcome']} / {c['v1']['primary_failure'] or '-'}; "
                f"V2 {c['v2']['outcome']} / {c['v2']['primary_failure'] or '-'}",
                *[f"  - {m.upper()} notes: {c[m]['notes']}" for m in MODES if c[m]["notes"]],
            ]
    lines += ["", "## Judge agreement (evaluator measurement, not the outcome)", ""]
    for mode in MODES:
        agreement = output["judge_agreement"][mode]
        if isinstance(agreement, str):
            lines.append(f"- {mode.upper()}: {agreement}")
            continue
        for name in ("calibration_in_sample", "audit_primary"):
            if name in agreement:
                a = agreement[name]
                lines.append(
                    f"- {mode.upper()} {name}: task-success agreement "
                    f"{pct(a['task_success_agreement'])}, false pass {pct(a['false_pass'])}"
                )
    return "\n".join(lines) + "\n"


def run(
    v1_path: Path, v2_path: Path, prefix: Path, sealed: dict[str, Path | None], seed: int
) -> Path:
    reports: dict[Mode, dict[str, Any]] = {}
    labels: dict[Mode, list[HumanLabel]] = {}
    frozen: dict[Mode, dict[str, Any]] = {}
    for mode, path in zip(MODES, (v1_path, v2_path), strict=True):
        data = path.read_bytes()
        p = report_paths(path)
        try:
            frozen[mode] = check_frozen(p["labels"], hashlib.sha256(data).hexdigest())
        except ValueError as exc:
            raise SystemExit(f"{mode}: {exc}") from exc
        reports[mode] = json.loads(data)
        labels[mode] = apply_adjudications(load_labels(p["labels"]), p["adjudications"])
    try:
        result = compare(reports["v1"], reports["v2"], labels, seed=seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output = {
        "sources": {
            mode: {
                "report": str(path),
                "labels_sha256": frozen[mode]["labels_sha256"],
                "report_sha256": frozen[mode]["report_sha256"],
                "adjudications": report_paths(path)["adjudications"].exists(),
            }
            for mode, path in zip(MODES, (v1_path, v2_path), strict=True)
        },
        "comparison": result,
        "judge_agreement": {
            mode: judge_agreement(path, verdicts) if (verdicts := sealed[mode]) else "not supplied"
            for mode, path in zip(MODES, (v1_path, v2_path), strict=True)
        },
        "limitations": limitations(result, frozen),
    }
    target = Path(f"{prefix}.comparison.json")
    target.write_text(json.dumps(output, indent=2) + "\n")
    target.with_suffix(".md").write_text(markdown(output))
    print(f"Wrote {target} and {target.with_suffix('.md')}.")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("v1_report", type=Path)
    parser.add_argument("v2_report", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="path prefix")
    parser.add_argument("--v1-sealed", type=Path)
    parser.add_argument("--v2-sealed", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run(
        args.v1_report,
        args.v2_report,
        args.output,
        {"v1": args.v1_sealed, "v2": args.v2_sealed},
        args.seed,
    )


if __name__ == "__main__":
    main()
