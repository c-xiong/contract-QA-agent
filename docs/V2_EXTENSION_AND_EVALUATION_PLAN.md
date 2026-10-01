# V2 Extension and Evaluation Plan

Scope: implementation status and acceptance criteria, not a report of answer quality.
Start with [the project guide](project-guide.md) and [the documentation index](README.md)
for the learning path and current runtime walkthroughs.

Implemented: the active V2 path and V1 reference, local read-only MCP with shared tool
policy, standalone evaluation/review infrastructure, and optional OTel export. Remaining:
a bounded live V2 evaluation, human labels and calibrated judge artifacts, author review
of the answer-rubric judge, and a publication-safe trace illustration. Manual stub and
live V2 UI smoke traces were viewed in Langfuse Cloud on 2026-10-01.
**There is no current live V2 answer-quality result.** Checked items below distinguish
implementation checks and manual smoke evidence; unchecked evaluation/reporting items
remain pending.

## Objective

Develop V2 as the single actively maintained contract research agent. Add a usable
MCP integration, establish credible answer-quality evaluation, and export
telemetry through OpenTelemetry to Langfuse.

The goal is to demonstrate integration design, enforceable execution boundaries,
measured answer quality, and reproducible failure analysis. Adding a protocol or
observability backend does not itself demonstrate better answer quality.

This document records the agreed scope, implementation status and remaining acceptance
criteria. Architecture, offline tests, a real backend demonstration and measured live
answer quality are separate milestones; completing one does not complete the others.

## Scope and sequence

1. Establish V2 as the active development path and document V1 preservation.
2. Build a local, read-only MCP server with shared server-side policy enforcement.
3. Complete the minimal OTel/Langfuse demonstration described in section 5.
4. Freeze the resulting implementation, then evaluate V2 answers and calibrate the
   offline LLM judges against human review as described in section 3.
5. Run one or two optional focused ablations when a concrete question warrants them.

Section numbers identify work packages, not execution order. The shared executor
refactor and its regression checks must finish before freezing code for live answer
evaluation. Later changes to answer behavior require a new evaluated revision; do
not attach pre-refactor measurements to the final implementation.

V1/V2 live comparison is optional. It becomes necessary if publishing a claim that
V2 improves on V1. Absolute V2 evaluation does not require maintaining two runtime
implementations.

Out of scope: authentication/SSO, multi-tenancy, mandatory cloud deployment,
Kubernetes, a new product frontend, GraphRAG, fine-tuning, conversational memory,
new model providers, and additional frameworks or databases without a demonstrated
need. Cloud deployment may be considered separately later.

## 1. V2 mainline and V1 preservation

### Work

- Direct new MCP, evaluation, and observability work toward V2.
- Record an existing V1 commit reference, dependency lockfile, reproduction commands,
  and existing result artifacts before removing any V1 runtime path.
- Keep the V1/V2 runner only while its maintenance cost is low. Do not require every
  new feature to support V1. Removing V1 runtime code is a separate, deliberate change.
- Preserve historical reports, frozen datasets, and their original limitations.
- When retiring a V1 runtime path, annotate the corresponding README headline
  tables as historical V1 results, with their revision and original limitations.
  Do not imply that those tables measure the current V2 path.
- Keep runtime and evaluation documentation aligned with implementation. The
  [standalone V2 protocol](evaluation_protocol.md#standalone-v2-evaluation-hand-v1)
  is now primary; the original paired procedure is retained as historical/optional.
  Split rules, budgets, stop conditions and labeling order live in that protocol,
  leaving status and acceptance criteria here. Do not relabel author review as
  independent review or reinterpret historical measurements as V2 quality.
- Record meaningful architecture changes in `docs/decisions.md` and concise
  `# DECISION:` comments in decision-bearing modules.

### Acceptance criteria

- [x] README identifies the active V2 path and provides one working V2 entry point.
- [x] A documented V1 reference and reproduction instructions remain available.
- [x] Existing historical artifacts and `hand-v1` remain unchanged.
- [x] README identifies pre-V2 result tables as historical V1 measurements; V1 runtime
  code remains available and has not been retired.
- [x] New features do not require maintaining duplicate V1 implementations.
- [x] Documentation distinguishes architectural properties from measured quality.
- [x] No V2 live result is compared against a V1 stub result as a quality improvement.

## 2. Local read-only MCP server

### Work

Start with local stdio transport and demonstrate use from an external MCP client.
An unauthenticated remote service is not part of this milestone.

Expose the five existing capabilities where their prerequisites can be preserved:

- `document_search`
- `retrieve_clause`
- `extract_definition`
- `calculate_date`
- `verify_citation`

Reuse the existing typed schemas and business logic. This is a policy refactor,
not merely a protocol wrapper: `app/agent/v2.py` currently owns tool/search budget
checks, counters, retries, and deadline handling around registry execution.

Introduce a shared `ToolSession` and executor with the following responsibilities:

- `ToolSession` owns immutable granted scope/snapshots, admitted evidence, retrieved
  source records, tool/search counters, and applicable execution limits. There must
  be one authoritative counter/evidence store, not independently mutable copies in
  graph state and the executor. Any graph-state view is synchronized at a defined
  execution boundary and retains existing serialization behavior.
- The executor validates scope and prerequisites, atomically checks/reserves tool
  and search capacity before dispatch, applies timeouts, executes the registry call,
  verifies returned provenance, admits evidence, and records the terminal outcome.
- A short lock protects reservation and admission; do not hold it across slow I/O.
  Each dispatched attempt consumes its reservation, including failures and retries.
  Invalid/rejected requests follow the existing graph accounting semantics. Test
  cancellation, concurrent admission, and evidence-size limits explicitly.
- Graph and MCP both invoke this executor. Caller policy supplies the applicable
  deadline and reserve rules. Preserve the graph's final-verifier reservation;
  MCP has no mandatory finalization and must not imply that it has the same reserve.
- Planning, model-call budgets, repair routing, duplicate-action decisions, and the
  graph's mandatory final-answer gate remain graph responsibilities. The refactor
  must not turn `ToolSession` into a second orchestration framework.

Preserve existing graph error/stop mappings and event semantics through a small
adapter where needed, rather than leaking MCP-specific outcomes into graph runs.

For MCP, CLI arguments or environment settings at server startup grant the document
allowlist and allowed snapshot map. Missing scope configuration fails closed; tool
arguments may only narrow this configured scope. A client cannot grant itself access.
The graph continues to obtain its run scope from the existing request path; this
refactor does not introduce user authentication or change that API contract.

The server owns evidence admission, call counters, deadlines, and output limits.
Store admitted evidence in server-owned session state. Preserve original
source coordinates, hashes, and nullable locators.

Keep date calculations linked to admitted governing evidence, preserve the explicit
calendar convention, and retain unsupported-convention errors. The server can check
arithmetic and provenance; it cannot guarantee that the external model selected the
correct contractual rule.

Reuse `verify_citation` rather than introducing a misleading `verify_answer` alias.
Its output must explicitly distinguish structural validity from semantic support.
The external client may skip this tool or ignore its result.

For this stdio-only milestone, one server process serves one client connection and
is one session. Do not add a `start_session` tool or persistent session service.
Process exit/connection closure discards session state; reconnecting launches a new
process and resets quotas. This is not an identity or global anti-abuse boundary.

Each MCP call has a configured timeout. An optional session lifetime starts at
process initialization and is disabled by default; when enabled, a call is bounded
by the smaller of its timeout and remaining session lifetime. Expired sessions
reject further calls. Unlike graph execution, an MCP session is not a QA run.
The graph retains its existing run-wide deadline unchanged.

### Guarantees and limits

| Server-enforced property | Property outside the server's control |
| --- | --- |
| Document/snapshot scope and validated arguments | External model's total calls and token spend |
| MCP tool/search budgets and bounded service execution | Overall external-agent execution deadline |
| Evidence admission and quote/locator checks | Whether the client invokes final verification |
| Limits on returned evidence and classified errors | Whether the client uses the verification result |
| Isolation of session evidence and counters | Correctness of prose generated outside the server |
| Per-session quotas within one process | Cumulative usage across reconnects/new processes |

### Acceptance criteria

- [x] All existing V2 offline regression tests pass without changing or weakening
  their assertions. Stop reasons, counter semantics, event names/order/meaning,
  provenance, and final-verifier reserve remain unchanged. Mechanical test setup
  changes, if unavoidable, are disclosed separately and cannot relax behavior.
- [x] Characterization coverage captures these behaviors before refactoring where
  existing coverage is missing; nondeterministic IDs/timestamps are normalized.
- [x] An offline MCP Python SDK client integration test starts the server over
  stdio, exercises `list_tools` and `call_tool`, checks typed success/error results,
  and terminates cleanly. It runs in CI without a model API or external client app.
- [x] Dependency locks pin the MCP SDK; the tested negotiated protocol revision and
  compatible client versions are recorded. Unsupported negotiation fails clearly.
- [x] An external client discovers and successfully invokes all five tools using
  documented setup commands and public or synthetic demonstration data.
- [x] A short reproducible walkthrough shows search, clause retrieval, and citation
  verification; definition lookup and date calculation have executable examples.
- [x] Tests reject out-of-scope document/version requests and attempted scope widening.
- [x] Tests reject fabricated, modified, and cross-session evidence references.
- [x] Tests verify call/search/output limits, deadline handling, and concurrent
  budget enforcement; failures cannot bypass or incorrectly refund the limits.
- [x] Tests preserve date-rule evidence requirements and unsupported conventions.
- [x] Structured errors remain machine-readable; protocol stdout contains no logs.
- [x] Startup grants, fail-closed missing scope, process-session quota reset, per-call
  timeout, optional session expiry, and cleanup are documented and tested.
- [x] A negative demonstration shows that the external client can skip verification;
  README does not claim the graph's final-answer gate applies to external prose.
- [x] The MCP dependency and shared enforcement design are justified in decisions.
  MCP is a main dependency for this deliverable because it is a supported entry
  point; OTel/Langfuse dependencies remain separate optional installation extras.

## 3. Standalone V2 answer evaluation

### Dataset and protocol

Begin with the existing `hand-v1` development regression set and keep it frozen:
**41 = 35 expected answers + 1 expected partial answer + 5 expected abstentions**.
The 36 non-abstention cases carry reference evidence and form retrieval-only scores;
all 41 belong in end-to-end answer evaluation. These are evaluation questions, not
ordinary software tests. Review necessary answer requirements and prohibited errors
in a separate, versioned rubric artifact; corrections must not overwrite the dataset.

Keep the 24 provisional V2 cases separate. Generated questions, expected answers,
and rubrics remain provisional until content is actually reviewed/replaced and a
new version is created. Merely removing the label is not sufficient.

Freeze the corpus/index, allowed snapshots, retrieval configuration, model revision
or recorded identifier, prompts, rubric, code revision/worktree provenance, and
execution budgets before a reporting run. Call the existing set a development
regression set, not an unseen test set.

### Evaluation dimensions

| Dimension | Method | Required distinction |
| --- | --- | --- |
| Citation validity | Deterministic source/scope/quote checks | Valid locations do not establish semantic support |
| Claim support | Offline judge against each claim's cited excerpts | Separate supported, contradicted, and insufficient evidence |
| Answer correctness and completeness | Frozen task rubric, reviewed reference evidence, judge plus human review | Detect omitted exceptions, conditions, parties, and incorrect conclusions |
| Abstention | Reviewed answerability labels plus delivered-output assessment | Separate correct abstention, false abstention, and unsupported answering |
| Reliability and resource use | Recorded terminal outcomes, latency, calls, tokens | Include failures; distinguish application costs from evaluation costs |

Retain and extend the existing `claim_support@2` methodology. A support judge sees
the claim and its actual cited excerpts. An answer judge also receives the question,
reviewed reference evidence, and task rubric, so it can detect facts omitted by the
system's retrieval. References need not prescribe one exact wording or tool sequence.

Check deterministic quantities, such as date arithmetic, with code. Assess selection
and interpretation of the governing clause separately. Prefer explicit criterion-level
verdicts to an unexplained overall 1–10 score.

### Judge calibration, labels and bounded execution

The executable protocol (frozen 20/21 split, declared claim ceiling `C`, call/token/
currency caps, labeling order, sealing and adjudication) lives in
[the standalone V2 section of the evaluation protocol](evaluation_protocol.md#standalone-v2-evaluation-hand-v1).
It was moved there when implementation began so that the two documents cannot diverge.

### Implemented infrastructure

- [x] Standalone runner, frozen configuration, incremental case saving and resume checks.
- [x] Recorded 20/21 judge split and historical claim-ceiling derivation.
- [x] Preflight templates and a cumulative call/token/currency reservation ledger.
- [x] Human review-sheet generation, label freezing, sealed judge output, logged unsealing
  and explicit post-freeze adjudication.
- [x] Offline diagnostics and human/judge reporting code retain explicit denominators;
  failures are not successful abstentions and missing usage is not zero.

These are infrastructure checks. The acceptance criteria below require actual live
outputs, filled budgets, reviewed labels and reporting artifacts.

### Reporting and acceptance criteria

- [ ] A standalone V2 runner evaluates all 41 existing cases with live model outputs,
  within a declared run budget, without requiring V1 execution.
- [ ] Dataset, rubric, judge, model, code, corpus/index, and configuration provenance
  are recorded; individual predictions are saved incrementally.
- [ ] Calibration and frozen audit records include human labels, judge verdicts,
  disagreement counts, false-pass counts, and reviewer identity/type.
- [ ] The 20/21 split was recorded before inspecting outputs; all human labels were
  produced without seeing judge verdicts. Final quality tables use human labels,
  while judge audit tables use the frozen 21-case subset separately.
- [ ] Preflight records a numeric token/currency cap and the applicable call ceilings;
  execution enforces cumulative budgets across retries and resumed partial runs.
- [ ] With this small initial suite, all live answers receive human review against
  the frozen rubric; report author-reviewed versus independently reviewed accurately.
- [ ] The report separates task success, citation validity, semantic support,
  completeness errors, abstention behavior, and execution failures.
- [ ] Each metric has an explicit denominator. Missing labels and judge failures
  remain visible and are never silently counted as correct or dropped.
- [ ] Timeouts, cancellations, and infrastructure failures are not correct abstentions.
- [ ] Report per-case records, category counts, medians and underlying latency/token
  data. Unknown usage remains null. Judge costs are separate from application costs.
- [ ] At least three actual failures or judge disagreements receive a concrete
  analysis linking the output, evidence, likely cause, and proposed improvement.
  If fewer exist, review all and state the count without inventing failures.
- [ ] Checked-in, publication-safe artifacts support every reported metric. Private
  contract contents are excluded. Small-sample and development-set limits are explicit.
- [ ] No arbitrary quality threshold is invented after seeing results. Completion
  means trustworthy measurement, not a guaranteed improvement or minimum score.

## 4. Optional focused ablations

Run an ablation only to answer a concrete question suggested by the task set or
failure analysis. Select one or two experiments, not an exhaustive feature matrix.

Examples:

- Clause retrieval enabled/disabled on cross-clause questions.
- Definition extraction enabled/disabled on definition-dependent questions.
- Offline audit of pre-gate drafts versus delivered answers to measure blocked
  unsupported content and incorrect rejection/abstention.

### Acceptance criteria

- [ ] Each experiment declares its hypothesis, case-selection rule, changed variable,
  and fixed settings before execution.
- [ ] Results preserve paired case records, failures, quality, and resource use.
- [ ] Draft-versus-delivery auditing is described as gate analysis, not automatically
  as a causal experiment measuring removal of the gate.
- [ ] Runtime safety boundaries are not disabled in the actual service for a demo.
- [ ] Findings include neutral or negative results and do not generalize beyond the
  evaluated cases. Multiple trials are added only when the intended claim needs them.

## 5. Required minimal OTel export and Langfuse demonstration

### Work

Preserve durable local JSONL traces and the existing inspector. Add one optional
telemetry adapter rather than replacing the local record or duplicating business
instrumentation.

Delivery of this adapter and a real Langfuse demonstration is required; enabling it
at runtime is optional. Keep the milestone small: one V2 run with nested retrieval,
tool/model operations, timings, terminal status, and available usage, correlated to
its durable local trace. A stub demo validates plumbing; one bounded live smoke run
validates actual model-call usage capture. Count any paid smoke run separately in
the preflight budget rather than hiding it in the evaluation allowance.

Use either Langfuse Cloud with public/synthetic demo data or an existing/self-hosted
instance. Hosting mode is not an acceptance criterion. If choosing self-hosting,
reuse pinned official deployment configuration; do not build a custom observability
platform or add a Langfuse deployment to this repository's runtime requirements.

Put OTel exporter dependencies in an explicit optional extra (and any Langfuse SDK
only if actually needed). Default `uv sync` must not install them. Import/configuration
paths work without the extra; enabling export without it yields a clear setup error.

Represent timed operations as spans and state changes as span events. Preserve
parent/child relationships across retrieval, model calls, tools, and validation.
Record errors, duration, token usage when available, and relevant version identifiers.
Pin/document the adopted GenAI semantic conventions and backend attribute mappings.

Use OTel/OTLP for trace export, with Langfuse as a demonstrated backend. Export must
be disabled by default and must not require network access for normal offline tests.
Use a documented enable flag/endpoint configuration; standard OTel settings may be
used rather than creating redundant project-specific settings.

Default to an explicit metadata allowlist, such as safe locator IDs, counts, timings,
and versions. Do not export raw contract text, prompts, answers, credentials, or
arbitrary exception strings by default. Self-hosting does not replace this policy.

Langfuse dataset synchronization, experiment/score upload, online judges, prompt
management, alerts, and additional dashboard/backend integrations are out of scope
for this milestone. They require a later concrete need. Repository artifacts remain
the reporting source of record; Langfuse is a trace viewing and debugging aid.

### Acceptance criteria

- [x] With export disabled, offline execution makes no telemetry network requests.
- [x] The default installation excludes telemetry extras and still passes its offline
  suite. A separate CI job installs the extra and validates the exporter offline.
- [x] Offline trace tests demonstrate correct nesting and correlation to a local run
  ID; this does not establish acceptance by a real Langfuse backend.
- [x] Export timeouts/failures do not alter answer correctness, runtime policy, or
  durable logging; bounded buffering/shutdown avoids indefinite blocking.
- [x] Tests show that default payloads omit raw content and secrets, including error
  paths, and that no duplicate spans are emitted for the same operation.
- [x] A local test receiver validates OTLP export independently of Langfuse; the
  [walkthrough](ui-langfuse-walkthrough.md) was used to manually verify stub and live
  V2 UI smoke traces in Langfuse Cloud on 2026-10-01.
- [ ] A short README walkthrough and publication-safe screenshot/export show an
  actual Langfuse trace correlated to a durable local run. One annotated case explains
  a failure or latency observation visible in the trace, without claiming a quality gain.
- [x] The live UI smoke run on 2026-10-01 showed three model calls, two tool calls
  including final verification, 12,918 input tokens and 404 output tokens in about
  6.8 seconds. These are one-run operational measurements, not answer-quality scores.
  Absent provider usage remains unknown. No dataset sync or online evaluation is required.

## Validation and final deliverables

For implementation changes, run the relevant behavioral tests and required checks:

```bash
CRA_LIVE_MODEL=0 uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy app evals
```

Live evaluation is an explicit paid experiment, separate from offline CI. Document
its command, frozen inputs, call/token budget, stop conditions, and partial-run handling
before execution. Normal CI must not silently enable model calls or telemetry export.

Required deliverables are the V2 mainline documentation, working MCP server and
boundary tests, standalone evaluation protocol/runner, calibrated judge artifacts,
human-reviewed live V2 report, minimal OTel export, a real Langfuse trace demonstration,
and reproducible demos. Focused ablations remain optional and do not block completion.

README and resume claims must distinguish implemented architecture, tested boundary
behavior, measured answer quality, and remaining limitations. Do not claim that MCP
or telemetry improved accuracy, that V2 outperforms V1 without a comparable live
baseline, or that author/LLM review constitutes independent expert validation.
