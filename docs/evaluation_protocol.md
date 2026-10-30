# V2 evaluation protocol

Scope: how to execute and review a V2 answer-quality experiment. The standalone
runner, budget ledger, frozen split and review tools are implemented. **No live V2
answer-quality run or human-reviewed V2 quality report is checked in yet.** The
answer-rubric judge still requires author review before its frozen live pass.

For the reasoning behind the steps, read [the project guide](project-guide.md).
[The documentation index](README.md) separates learning material, runtime instructions,
measurement definitions and historical reports.

Keep `hand-v1` byte-for-byte frozen: **41 = 35 expected answers + 1 expected partial
answer + 5 expected abstentions**. Its 36 evidence-bearing cases form the retrieval-only
benchmark; all 41 belong in end-to-end evaluation. Keep the 24
`provisional-generated-v2-1` cases separate. Their questions, expected dates,
capabilities and rubrics are generated pipeline fixtures. They cannot become reporting
data by removing the provisional label: replace/review content and increment the
version. The separate 119-query CUAD validation measures retrieval on development-
disjoint documents, not V2 answer quality.

Use [standalone V2 evaluation](#standalone-v2-evaluation-hand-v1) for current work.
To understand how V2 differs from V1 on the same cases, use the
[budgeted paired V1/V2 comparison](#budgeted-paired-v1v2-comparison-hand-v1), which reuses
the standalone runner, labels and judges for both arms. The
[original paired procedure](#historicaloptional-paired-v1v2-protocol) remains available
for reproducing earlier runs.

## Standalone V2 evaluation (hand-v1)

This is the primary protocol for new V2 work. It evaluates V2 alone; no V1 execution
is needed, and no V2-over-V1 claim can come from it. Stub runs remain wiring checks.
A full run retains all 41 cases, including the five expected abstentions; reference-
evidence retrieval metrics apply to the other 36 (35 full-answer and one partial-answer
case). Human answer-quality labels and execution outcomes keep failures visible.

### Frozen inputs (checked in before any live V2 output exists)

| Artifact | Content |
|---|---|
| `evals/datasets/hand/judge_split-v1.json` | 20 calibration / 21 audit case IDs, seed 20260928, rule, dataset SHA-256, balance. `scripts/build_judge_split.py --check` verifies it. |
| `evals/datasets/hand/claim_ceiling-v1.json` | `C = 7`: the maximum cited-claim count over 82 historical live hand-v1 answers (distribution 1:11, 2:25, 3:31, 4:8, 5:2, 6:4, 7:1), counted by the grader's own `extract_cited_claims`. V1-writer answers; V2 answers may be longer. |
| `evals/preflight/*.json` | Application and judge budgets, including the model's current prices, filled from the provider's price page before a paid run. |

With `C = 7`: judge ceiling `2 × 20 × 8 + 41 × 8 = 648` attempts; application ceiling
`41 × 12 = 492`; combined `1140` provider attempts, plus the token and currency caps.

### Workflow

```bash
# 1. Wiring check (free): stub planner/writer, no provider calls.
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_standalone --suite hand
# 2. Paid application run (explicit): every attempt passes through the ledger.
CRA_LIVE_MODEL=1 uv run python -m experiments.v2_standalone --suite hand \
    --arm rrf_hybrid_rerank --preflight evals/preflight/v2-hand-application.json \
    --output reports/v2-standalone-hand-live.json            # add --resume after interruption
# 3. Human labels first. Do not open *.sealed.jsonl.
uv run python -m scripts.v2_labels sheet reports/v2-standalone-hand-live.json
#    ... fill reports/v2-standalone-hand-live.labels.jsonl using the .review.md sheet ...
uv run python -m scripts.v2_labels freeze reports/v2-standalone-hand-live.json
# 4. Judge calibration (paid): round 1 sees only the 20 calibration cases.
CRA_LIVE_MODEL=1 uv run python -m experiments.v2_judge reports/v2-standalone-hand-live.json \
    --phase calibration --round 1 --preflight evals/preflight/v2-hand-judge.json
# 5. Inspect calibration agreement BEFORE deciding whether to freeze the judge.
uv run python -m scripts.v2_labels agreement reports/v2-standalone-hand-live.json \
    --sealed reports/v2-standalone-hand-live.judge-calibration-r1.sealed.jsonl
#    Review calibration disagreements/false passes and the sampled agreements.
#    Tune only against these 20 cases. Keep the 21 audit labels out of judge tuning.
#    If needed, use the one remaining calibration round within the SAME judge budget:
# CRA_LIVE_MODEL=1 uv run python -m experiments.v2_judge reports/v2-standalone-hand-live.json \
#     --phase calibration --round 2 --preflight evals/preflight/v2-hand-judge.json
# uv run python -m scripts.v2_labels agreement reports/v2-standalone-hand-live.json \
#     --sealed reports/v2-standalone-hand-live.judge-calibration-r2.sealed.jsonl
# 6. Complete author review of the answer-rubric prompt, bump its version and set
#    ANSWER_RUBRIC_AUTHOR_REVIEWED as described in evals/graders/answer_rubric.py.
#    Freeze the final judge prompts/versions; make no changes based on audit results.
# 7. Frozen judge pass (paid): all 41 cases; audit outputs/labels have not been used for tuning.
CRA_LIVE_MODEL=1 uv run python -m experiments.v2_judge reports/v2-standalone-hand-live.json \
    --phase frozen --preflight evals/preflight/v2-hand-judge.json
# 8. Only now unseal the frozen results and write the final report.
uv run python -m scripts.v2_labels report reports/v2-standalone-hand-live.json \
    --sealed reports/v2-standalone-hand-live.judge-frozen.sealed.jsonl
```

The `sheet` command creates a `.labels.jsonl` file and a `.labels.source.json`
companion containing the exact source report's SHA-256. Retain the companion with
the labels and keep the source report unchanged. Freezing rejects a report changed
since sheet creation; unsealing and final reporting reject frozen labels or sealed
verdicts bound to a changed or different report. Legacy artifacts without these
hashes are rejected.

Human labels remain frozen throughout calibration. The labeler reviews all 41 outputs
before seeing any judge verdict, but judge tuning uses only the 20 calibration cases
and their labels. Do not inspect the 21 audit verdicts or use those cases' human labels
to tune the judge. The frozen pass is the first judge exposure to the audit cases;
its final report shows audit agreement separately from calibration agreement. A
second round is optional and consumes the same cumulative judge budget, not a reset.

Freezing judge prompts is a recorded procedural commitment: each pass stores prompt
hashes and versions, but the script does not itself enforce that calibration was
inspected before the frozen pass or prevent someone from opening the audit labels.
Keep the final prompt revision fixed through the frozen pass and disclose any breach.

Enforced in code: a live run refuses to start without a matching preflight; the ledger
checks attempt limits and estimated token/cost reservations before each attempt, keeps
the estimated reservation when usage is unknown, persists after every change, and
refuses a changed preflight for an existing ledger. Known usage above a reservation
exhausts the ledger and stops subsequent attempts, including after resume;
`settled_cap_breaches` records per-call token and cumulative token/currency breaches.
The input estimate can undercount, so an in-flight attempt can exceed the declared
spend caps. `--resume` refuses a changed frozen configuration and never re-runs completed
cases; unexecuted cases are listed; execution failures are never judged and cannot be
labelled successes; claims beyond `C` are `unknown` and routed to human review; labels
must be complete and frozen before any sealed verdict is read, each unseal is logged, and
post-freeze corrections must be adjudication records; the frozen judge pass refuses to
run live until the author has reviewed the answer-rubric prompt (`TODO(author)` in
`evals/graders/answer_rubric.py`). Sealing is procedural, not cryptographic: opening a
sealed file early breaks the protocol and must be disclosed.

### Judge calibration rules

- Before any live V2 answers or judge verdicts are inspected, assign the 41 case IDs
  to 20 calibration cases and 21 frozen judge-audit cases. Use a recorded seed and
  stratify by the dataset's `category` field (nine categories), then check
  `expected_behavior` balance. Categories with an even count split evenly; for an
  odd count the extra case goes to audit, then calibration is topped up to 20 by
  seeded draw. Two-case categories (`prompt_injection`,
  `similar_document_distractors`) therefore contribute one case to each side.
  Check in the assignment and selection script/rule. This split evaluates the judge;
  it does not make the development QA set held-out.
- Disclose prior exposure. The author has already inspected earlier live hand-v1
  outputs from the historical three-arm runs, and the existing `claim_support@2`
  judge prompt was developed with that material available. The 20/21 split prevents
  leakage during this calibration only; audit results must not be described as a
  held-out or blind validation of the judge.
- Use the 20 calibration answers, covering where available correct responses,
  missing exceptions, unsupported claims, mistaken abstentions, and date errors.
  Identify constructed error examples separately from actual model outputs.
- Have a human label all 41 answers against the frozen rubric before exposing that
  answer's judge verdict to the reviewer. Audit labels are sealed from judge tuning.
  Author review is acceptable for this milestone but must be named as such.
- Separate calibration examples from a frozen judge-audit sample. Do not tune the
  judge against the audit labels and then claim an independent audit on those labels.
- Keep offline judge prompts/verdicts separate from the runtime verifier. Do not
  provide its verdict, hidden reasoning, or system version label to the judge.
- Use the existing provider initially. Disclose shared-model-family limitations;
  separate prompts do not imply independent errors.
- Record criterion-level human/judge disagreement and false passes. Review all
  observed disagreements and a recorded random sample of apparent agreements.
- If the judge misses material errors, revise and recalibrate it or use human labels
  as the reported outcome. Never treat an uncalibrated judge as ground truth.

### Authoritative labels and bounded execution

The final report's answer-quality metrics use human labels for all 41 answers.
Judge scores are reported separately as evaluator-agreement/error measurements,
with the 21-case frozen audit as the primary judge audit. Calibration agreement is
in-sample and is labeled accordingly. LLM labels never overwrite human labels.
Deterministic structural metrics and measured operational costs retain their own
sources. Corrections to human labels are explicit, versioned adjudications.

Use the following initial call ceiling, not a prediction of actual usage:

- Application run: 41 cases × at most 12 model calls = 492 application calls,
  matching the current default `max_model_calls`. Count all provider attempts and
  configure SDK retries so they cannot silently evade the experiment ceiling.
- Judge: keep the existing per-claim `claim_support@2` grader
  (`app/evidence/claim_support.py` calls the judge once per cited claim). Batching
  claims into one call changes the prompt and input structure and would be a new
  grader version (`claim_support@3`) requiring its own comparison against @2 on the
  calibration set; it is not part of this milestone. Each answer therefore costs at
  most `C + 1` judge calls: `C` claim-support calls plus one answer-rubric call,
  where `C` is a declared maximum cited-claim count per answer.
- Declare `C` in preflight before live execution, derived from claim counts in the
  existing checked-in hand-v1 run artifacts and recorded with that derivation.
  Judge ceiling: `2 × 20 × (C + 1)` for at most two calibration rounds plus
  `41 × (C + 1)` for one frozen pass. An answer with more than `C` cited claims is
  not truncated; its excess claims route to human review and are counted as such.
- Combined ceiling: 492 application attempts plus the derived judge ceiling,
  including retries. Application and judge budgets are additive, not multiplied.
  Constructed probes or additional judge calls must fit these caps or require a
  separately declared experiment budget.

Preflight must also set input/output token caps and a numeric currency cap using
the selected model's current prices. Call count alone is not a financial cap.
Before each call, reserve `ceil(prompt_chars / 2)` estimated input tokens plus the
declared output ceiling and their cost; refuse attempts whose reservation would exceed
a cap. This character estimate is not a tokenizer bound and cannot guarantee actual
spend stays below a cap on an in-flight attempt. Settle reported usage, record any
per-call or cumulative cap breach, and stop subsequent attempts after any reservation
overrun. Unknown usage is not zero: retain its estimated reservation. Persist partial results
and mark unexecuted cases explicitly. Never restart a partial run by resetting its
cumulative budget. Judge failures/overflow cases route to human review, not a pass.

## Budgeted paired V1/V2 comparison (hand-v1)

Added 2026-10-06 ([plan](V1_V2_COMPARISON_PLAN.md)). This runs the standalone workflow
twice, once per agent mode, and pairs the two reports only after both label sets are
frozen. It is for understanding where the designs differ; V2 capability statements still
come from the standalone V2 report. No live comparison has been run yet.

- **Same frozen inputs.** `experiments.v2_standalone --mode v1|v2` records `agent_mode`
  in the frozen config. Pairing refuses reports whose frozen configs differ in anything
  except `agent_mode` and `planner_version` (model, arm, budget, dataset, split, corpus,
  index), or that still have unexecuted cases.
- **Separate budgets.** Each arm has its own application preflight and ledger
  (`v1-hand-application`, `v2-hand-application`; 492 attempts each).
- **Blind labels.** `scripts.paired_review sheet` shows each case's two answers as A/B in
  a seeded order and binds both report hashes; `unblind` writes each report's labels.
  Mode names, stop reasons, traces and versions are hidden. Answer style, V2's fixed
  abstention wording, evidence lists and run status can still reveal the mode, so the
  blinding is partial and disclosed.
- **Judges.** Calibrate on the 20 V2 calibration cases exactly as in the standalone
  protocol, then run one frozen pass on each report (`v1-hand-judge`: 41 × (C + 1) = 328
  attempts). The V1 pass is an additional evaluator-agreement set, never a tuning set.
- **Report.** `scripts.compare_v1_v2` requires both label sets frozen against the exact
  report bytes. Human task success is the paired outcome: V2 win/loss/tie overall and
  per category, the outcome transition matrix, and every win, loss and shared failure
  with its labels. An exact sign test and a seeded paired bootstrap interval are
  descriptive context for one trial, not a significance claim. Deterministic diagnostics
  (evidence recall, answer/abstain decision, structural gate) and costs are paired over
  cases where both values are known; V1 tool calls stay null.

```bash
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_standalone --mode v1 --suite hand
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_standalone --mode v2 --suite hand
# Paid, after filling both application preflights with the same model and prices:
CRA_LIVE_MODEL=1 uv run python -m experiments.v2_standalone --mode v1 --suite hand \
    --arm rrf_hybrid_rerank --preflight evals/preflight/v1-hand-application.json \
    --output reports/v1-standalone-hand-live.json
CRA_LIVE_MODEL=1 uv run python -m experiments.v2_standalone --mode v2 --suite hand \
    --arm rrf_hybrid_rerank --preflight evals/preflight/v2-hand-application.json \
    --output reports/v2-standalone-hand-live.json
uv run python -m scripts.paired_review sheet reports/v1-standalone-hand-live.json \
    reports/v2-standalone-hand-live.json --output reports/v1v2-hand-live
#  ... label reports/v1v2-hand-live.blind-labels.jsonl from reports/v1v2-hand-live.review.md
uv run python -m scripts.paired_review unblind reports/v1v2-hand-live
uv run python -m scripts.v2_labels freeze reports/v1-standalone-hand-live.json
uv run python -m scripts.v2_labels freeze reports/v2-standalone-hand-live.json
# Judge steps 4-7 of the standalone workflow on the V2 report; then the V1 frozen pass:
CRA_LIVE_MODEL=1 uv run python -m experiments.v2_judge reports/v1-standalone-hand-live.json \
    --phase frozen --preflight evals/preflight/v1-hand-judge.json
uv run python -m scripts.compare_v1_v2 reports/v1-standalone-hand-live.json \
    reports/v2-standalone-hand-live.json --output reports/v1v2-hand-live \
    --v1-sealed reports/v1-standalone-hand-live.judge-frozen.sealed.jsonl \
    --v2-sealed reports/v2-standalone-hand-live.judge-frozen.sealed.jsonl
```

**Subscription backend.** The 2026-10-06 runs use the Claude Code CLI
(`CRA_MODEL_BACKEND=claude_code`, writer `CRA_MODEL_ID=claude-sonnet-5`, effort `high`;
judge passes with `CRA_MODEL_ID=claude-opus-5-5`). The checked-in
`evals/preflight/*-hand-*.json` files carry notional list prices, a 32k per-call output
ceiling (thinking is not capped by the CLI) and a 1,000-token per-call preamble
allowance. A usage/rate limit stops a run as `provider_limit_stopped` (or a judge case as
`not_judged: provider_limit`); rerun the same command with `--resume`. Interrupted cases
are never scored. The backend, effort and CLI version are recorded in provenance.

The treatment is the whole V2 design (planner, typed tools, stricter structural citation
gate, routing), not agentic retrieval alone. The two arms run sequentially, not
interleaved per case; record the run dates and disclose any provider change between them.

## Historical/optional paired V1/V2 protocol

This was the original V2 protocol and backs the checked-in offline paired reports.
It is retained for reproduction and optional comparisons. It is not a prerequisite
for the current standalone workflow. A claim that V2 improves over V1 would require
comparable live runs and reviewed outputs from both modes; stub completion rates
cannot establish that improvement. These historical commands do not provide the
standalone workflow's experiment-wide preflight ledger.

Before a live comparison, freeze model revision, corpus/index hashes, retrieval arm,
search/evidence/repair budgets and V2 ceilings. Both modes see identical cases and
allowed snapshots. The runner disables the optional V1 answerability gate. V2's stricter
citation gate and dynamic routing are part of the treatment, and should be disclosed.
V1 cases requiring missing capabilities stay in the denominator. Do not remove failures.

```bash
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_comparison --suite hand
CRA_LIVE_MODEL=0 uv run python -m experiments.v2_comparison --suite v2
# Intentional paid experiment, after dataset/rubric review:
CRA_LIVE_MODEL=1 uv run python -m experiments.v2_comparison --suite hand --arm rrf_hybrid_rerank --output reports/v2-hand-live.json
```

Each report includes provenance, per-case predictions, category and answerability subsets,
paired diagnostic wins/losses/ties, and cost medians including failed runs. No repeated
trials, held-out claim, bootstrap interval, p95 or currency estimate. The per-case
`.predictions.jsonl` is flushed after each run. The adjacent `.review.jsonl` starts blank.

An independent reviewer fills task_success, citation support counts, useful-call counts,
reviewer identity and one primary failure category per failed case. Review both arms,
including every apparent V2 win. Multiple correct capability trajectories are acceptable;
capability_order expresses only prerequisite relationships, not one exact tool sequence.

```bash
uv run python -m scripts.score_v2_review reports/v2-hand-live.json reports/v2-hand-live.review.jsonl --output reports/v2-hand-reviewed.json
```

The scorer rejects missing/duplicate cases, inconsistent denominators, unreviewed outcomes,
and successful labels on infrastructure failures. Generated/stub reports remain ineligible
for capability claims even after labels are supplied. Original semantic labels are never
inferred from citation structure or lexical coverage.

| Metric | Treatment |
|---|---|
| Task success | Manual frozen rubric; failures/cancellation always fail. |
| Citation validity | Structural delivered-citation counts and gate-pass task counts. |
| Semantic citation support | Independent supported / inspected citation counts. |
| Evidence recall | Existing span matching at k=5 and trajectory-wide; the 36 non-abstention cases (35 answer + 1 partial), excluding five cases without reference evidence. |
| Tool selection | Required-capability coverage/order diagnostic; scope/arguments and usefulness manually inspected. |
| Unnecessary calls | Manual usefulness counts and explicit retry/duplicate diagnostics. |
| Abstention | Runtime answer-versus-abstain decision plus separate false abstention; prose refusals still need manual review. |
| Cost | Median latency, model/tool calls, input/output tokens; failures included; unknown usage remains null. |

V1's tool-call count is null: its internal operations were not instrumented as V2 registry
invocations. Do not treat null as zero or claim a cross-mode tool-cost reduction from it.
Stub usage is estimated and latency describes local fixture execution. Initial retrieval
models/indexes are shared; V1 runs before V2 within each pair. Cache state/order is disclosed,
not hidden as a performance result. Production generalization cannot be inferred here.
