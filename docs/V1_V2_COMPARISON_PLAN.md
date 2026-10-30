# Plan: complete live evaluation of V1 and V2, and a paired comparison

Status: implemented 2026-10-06 (offline-verified; no live run yet). Purpose: understand where V2 (bounded agentic tool use)
differs from V1 (fixed retrieve → answer → verify pipeline) on the same frozen cases.
The comparison is for understanding the system, not for CV bullets; the standalone V2
evaluation remains the source of any V2 capability statement.

## What exists and what is missing

| Piece | V2 | V1 |
|---|---|---|
| Budgeted, resumable live runner with frozen config | `experiments/v2_standalone.py` | missing (only the unbudgeted, non-resumable `experiments/v2_comparison.py`) |
| Human labels bound to report hashes, freeze, adjudication | `scripts/v2_labels.py` | works on any standalone report, but labelling each mode separately is unblinded |
| Sealed offline judges (claim_support@2, answer_rubric) | `experiments/v2_judge.py` | works on any standalone report |
| Paired report over human labels | — | — |

## Design

1. **One runner for both modes.** `experiments/v2_standalone.py --mode v1|v2` (default
   `v2`, so existing commands and artifacts keep their meaning). `agent_mode` enters the
   frozen config; `planner_version` is recorded only for V2. Rows record `mode`. Default
   output `reports/{mode}-standalone-{suite}-{live|offline}.json`. V1 gets the same ledger,
   resume, failure accounting and the same disabled answerability gate.
2. **V1 preflight templates.** `v1-hand-application` (ceiling 41 × 12 = 492 attempts, the
   same cap; V1 normally uses fewer) and `v1-hand-judge` (frozen pass only:
   41 × (C + 1) = 328 attempts). Prices stay unfilled until a paid run.
3. **Blind paired labelling** (`scripts/paired_review.py`).
   - `sheet V1 V2 --output PREFIX`: checks the two reports are comparable, binds each
     report's hash (the normal `.labels.source.json`), and writes one Markdown sheet
     where each case shows answers **A** and **B** in a seeded random order, plus a
     blank `PREFIX.blind-labels.jsonl` (`HumanLabel` fields + `slot`). The A/B key goes
     to `PREFIX.blind-key.json`; the reviewer does not open it.
   - `unblind PREFIX`: validates complete blind labels and writes each report's
     `.labels.jsonl`; then the existing `v2_labels freeze` runs on each report.
   - Blinding is partial and disclosed: style, evidence lists and run status can reveal
     the mode. Mode names, stop reasons, tool traces and versions are hidden.
4. **Judges.** Unchanged protocol: calibrate on the 20 V2 calibration cases, author
   review of the answer rubric, then a frozen pass on **both** reports. The V1 frozen
   pass is an extra evaluator-agreement set, not a tuning set.
5. **Paired comparison** (`evals/paired.py` + `scripts/compare_v1_v2.py`). Requires
   both label sets frozen and comparable frozen configs (only `agent_mode` and
   `planner_version` may differ). Writes `PREFIX.comparison.json` and a readable
   `PREFIX.comparison.md`:
   - comparability table (frozen-config diff, model, arm, dataset, hashes);
   - human-label quality per mode, overall and per category (existing `quality()`);
   - paired task success: V2 win / loss / tie overall and per category, discordant
     pairs, exact two-sided sign-test p and a seeded paired-bootstrap interval for the
     success-rate difference, both labelled descriptive (single trial, n = 41);
   - outcome transition matrix (V1 outcome → V2 outcome);
   - per-case human claim support, paired;
   - deterministic paired diagnostics: evidence recall@5, trajectory recall,
     answer/abstain decision, structural gate pass;
   - cost: per-mode medians including failures, paired medians of differences for
     model calls, tokens and latency; V1 tool calls stay null, never zero;
   - every V2 win and loss listed with both labels' outcome, primary failure and notes,
     plus primary-failure counts per mode;
   - judge agreement per mode when sealed frozen verdicts are supplied;
   - limitations.
6. **Docs.** New "Paired V1/V2 comparison" section in `docs/evaluation_protocol.md`, a
   `docs/decisions.md` entry, pointers in `docs/README.md` and `evals/preflight/README.md`.

Not in scope: repeated trials (a second trial is a second pair of outputs; aggregation is
future work), new datasets, an OpenAI judge, changes to graders or prompts.

## Model backend (decided 2026-10-06: Claude Code CLI)

The author chose subscription billing. `CRA_MODEL_BACKEND=claude_code` runs every live
call as one `claude -p` process (`ClaudeCodeClient`): the caller's system prompt
replaces Claude Code's, tools/MCP/skills/settings/session persistence are off, the
working directory is empty and API credentials are stripped. Both arms use it.
Writer `claude-sonnet-5`, effort `high`; judge `claude-opus-5-5`. Disclosed effects: a
~650-token fixed preamble per call (reserved via `per_call_input_overhead_tokens`), no
max_tokens control (32k per-call output ceiling covers adaptive thinking), latency that
includes process start, and notional list-price spend. A usage/rate limit stops the run
without scoring the case; `--resume` continues (judge passes too).

## Workflow once implemented

```bash
# 1. Free wiring check, both modes
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_standalone --mode v1 --suite hand
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_standalone --mode v2 --suite hand
# 2. Fill evals/preflight/{v1,v2}-hand-{application,judge}.json from the templates
# 3. Live application runs, same arm and model (CLI backend: export
#    CRA_MODEL_BACKEND=claude_code CRA_MODEL_ID=claude-sonnet-5 first)
CRA_LIVE_MODEL=1 uv run python -m experiments.v2_standalone --mode v1 --suite hand \
    --arm rrf_hybrid_rerank --preflight evals/preflight/v1-hand-application.json \
    --output reports/v1-standalone-hand-live.json
CRA_LIVE_MODEL=1 uv run python -m experiments.v2_standalone --mode v2 --suite hand \
    --arm rrf_hybrid_rerank --preflight evals/preflight/v2-hand-application.json \
    --output reports/v2-standalone-hand-live.json
# 4. Blind labels for both, then freeze each
uv run python -m scripts.paired_review sheet reports/v1-standalone-hand-live.json \
    reports/v2-standalone-hand-live.json --output reports/v1v2-hand-live
#    ... fill reports/v1v2-hand-live.blind-labels.jsonl from reports/v1v2-hand-live.review.md
uv run python -m scripts.paired_review unblind reports/v1v2-hand-live
uv run python -m scripts.v2_labels freeze reports/v1-standalone-hand-live.json
uv run python -m scripts.v2_labels freeze reports/v2-standalone-hand-live.json
# 5. Judge: calibration on V2, author rubric review, frozen pass on both (standalone protocol)
# 6. Paired comparison
uv run python -m scripts.compare_v1_v2 reports/v1-standalone-hand-live.json \
    reports/v2-standalone-hand-live.json --output reports/v1v2-hand-live \
    [--v1-sealed ...judge-frozen.sealed.jsonl --v2-sealed ...judge-frozen.sealed.jsonl]
```

## Acceptance

- [x] Runner supports both modes; V2 default behaviour and frozen config unchanged
      apart from the added `agent_mode` key.
- [x] Blind sheet hides mode identity; unblinding refuses incomplete or duplicate labels
      and never overwrites existing label files.
- [x] Comparison refuses unfrozen labels, mismatched task sets and incomparable configs.
- [x] Every rate carries its denominator; failures remain in denominators.
- [x] Unit tests for each new path; ruff, format, mypy and offline pytest pass.
- [x] An offline end-to-end run (stub, both modes, synthetic labels in a temp dir)
      produces a comparison report, clearly marked not reporting-eligible.
