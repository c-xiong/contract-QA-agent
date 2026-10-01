# Practice with the UI and Langfuse

Reviewed 2026-10-01. Read [the project guide](project-guide.md) first if the 36/41 cases
or the distinction between tracing and evaluation is unclear.

## 1. Start a no-cost UI session

Run commands from the repository root. The local `.env` may already enable a paid
model, so explicitly select stub mode while learning the interface.

```bash
uv sync --locked --all-groups --extra otel
CRA_LIVE_MODEL=0 CRA_AGENT_MODE=v2 CRA_TELEMETRY_EXPORT=0 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Confirm the header says **V2**, **Stub / simulated**,
and **model: stub**. The current inspector explicitly requests V2 even though V1
remains the runtime default. Changing the retriever does not turn a stub into a live model.

These offline model-cache flags require the retrieval models to have been downloaded
already. If startup reports missing model files, omit only `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` for a first download. Keep `CRA_LIVE_MODEL=0`. If the corpus/index
is missing, follow the acquisition/ingestion commands in the [root README](../README.md#quickstart).
First model loading can take time even though no paid model API is called.

## 2. Run the insurance example

1. Click **Try insurance example**. It fills the question and selects **All documents**
   and **rrf_hybrid_rerank**. Preparing the example does not run it.
2. Click **Ask**. Watch initialize, decide, authorize, execute, write, verify, and
   terminal. The route can revisit nodes; this is a graph, not a one-pass checklist.
3. Open the evidence for **doc-011, page 38, section 11.5**. Locate the clinical-trial
   and product-liability insurance text. Preserve redacted `[***]` amounts.
4. Compare cited evidence with uncited context. A passage appearing in the evidence
   list does not mean every answer sentence cites it.
5. Inspect citation checks, budget counters, and the execution log. Note the tool
   actions, admitted evidence IDs, and the run ID. A green citation gate checks
   references and admission; it does not prove the answer is semantically correct.

The question is: "What coverage must Aimmune keep in place while the product is
being tested or sold?" In stub mode the actual retrieval and tools run locally,
but the planner and answer are simulated. Treat its output as an execution demonstration.

Repeat with BM25 and the hybrid reranker to inspect different evidence. One manual
example cannot reproduce the 72.7% benchmark: that number averages 36 frozen cases
using the specific optional `legal_query_coverage_rerank` experiment arm.

## 3. Inspect local run files

Every V2 run writes `runs/<run_id>/`:

| File | What to learn |
|---|---|
| `manifest.json` | Model, simulated flag, corpus/index/configuration and version identifiers |
| `events.jsonl` | Ordered state/action/evidence events, including the terminal outcome |
| `summary.json` | Final status, stop reason, calls, tokens, verification outcome |
| `telemetry.json` | Export correlation ID, present only after an export attempt is queued |

Check both the model name and `simulated` flag when interpreting old examples; some
historical scripted demos identify their model as `scripted-offline-demo`. They are
not real provider evaluations. Stub token usage is a fixture estimate.

To see controlled success, repair, and abstention examples without overwriting the
checked-in demonstrations:

```bash
CRA_LIVE_MODEL=0 uv run python -m scripts.v2_demo \
  --output /tmp/contract-agent-practice-traces --runs /tmp/contract-agent-practice-runs
```

An arbitrary unanswerable question in the stub UI is not a dependable semantic refusal
test. Use controlled fixtures to understand the route, then live/human-reviewed cases
to assess refusal quality. Closing a stream cancels that run; reconnecting is not
crash recovery. Evaluation-batch resume is a separate feature.

## 4. Configure a Langfuse destination

If you have not set up Langfuse, begin with [Langfuse Cloud EU](https://cloud.langfuse.com):

1. Create/sign in to your account and complete the organization setup shown by the UI.
2. Create a project, for example **contract-research-agent**.
3. Open **Project → Settings**, find/create its API keys, and keep the public key and
   secret key locally. [Official API-key instructions](https://langfuse.com/faq/all/where-are-langfuse-api-keys).
4. Configure this repository's existing exporter using the commands below. Installing
   a separate Langfuse SDK or local Langfuse backend is not needed for this Cloud path.

The repository contains the exporter, not a Langfuse service or account. A local test
receiver proves OTLP wiring but will not create a dashboard. A self-hosted deployment
is an alternative if you prefer to operate the backend yourself.

Fill these fields in the repository's gitignored `.env`, replacing the placeholders
with the two project keys:

```dotenv
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=YOUR_PUBLIC_KEY
LANGFUSE_SECRET_KEY=YOUR_SECRET_KEY
CRA_TELEMETRY_EXPORT=1
```

The base URL is the host only. For self-hosting, it might be `http://localhost:3000`.
The application adds `/api/public/otel/v1/traces` and constructs Basic authentication
plus the ingestion-version header internally. Both keys are masked in Settings;
they are never span attributes. Empty keys are allowed while export is disabled.
Restart the server after changing the file; its existing process caches settings.
The current startup command explicitly disables export, so follow section 6 to enable it.

No encoded header is needed for this configuration. Advanced generic OTLP export
through `OTEL_EXPORTER_OTLP_*` shell variables is still possible when all three
Langfuse fields are absent. Such variables do not belong in the typed application
`.env`. See [Langfuse's OpenTelemetry guide](https://langfuse.com/integrations/native/opentelemetry)
for protocol details. Keep actual keys and encoded authentication headers out of
screenshots, chat, documentation, and commits.

## 5. Export an existing run first

You can inspect real historical model-call timings/tokens without paying for another
model call. Configure the project keys above, choose an existing completed V2 directory,
then run:

```bash
uv run python -m scripts.export_run_telemetry runs/REPLACE_WITH_RUN_ID
```

Alternatively, `--latest 1` selects the newest journal by modification time, which may
be a stub example. An unfinished selected journal is skipped; it does not fall back to
an older completed run. Export itself does not call the answering model or LLM judges.
Confirm the chosen manifest before interpreting its usage as real.

This repository exports only allowlisted metadata: identifiers, evidence locators,
version hashes, counts, statuses, timings, and available token usage. Questions,
prompts, answers, source text, tool arguments, credentials, and exception messages
are excluded. Local journals still contain the detailed execution record.

## 6. Produce a new UI trace

Stop the server with Ctrl+C. With the Langfuse fields saved in `.env`,
restart with the same startup command but change **CRA_TELEMETRY_EXPORT=0 to 1**.
Keep **CRA_LIVE_MODEL=0** for a no-cost model demonstration. Run the insurance example
again and let it finish. Export occurs after the terminal event; Langfuse does not
stream this run while it is executing.

For an intentional real-model practice run, stop/restart with **CRA_LIVE_MODEL=1**
and an available `ANTHROPIC_API_KEY`. Confirm the UI says **Live** and names the actual
model before asking. This invokes paid planner/writer calls. The UI run has per-run
execution limits; the experiment currency ledger applies to evaluation scripts,
not to arbitrary UI requests. Start with one question rather than a full suite.

## 7. Read the trace

In your Langfuse project, open the traces/observations view. Find the exported trace
ID from `runs/<run_id>/telemetry.json`, or locate its `cra.run_id` attribute. Match it
to the local journal rather than selecting an unrelated recent trace.

| Trace element | Meaning |
|---|---|
| `invoke_agent contract-research-v2` | Entire run, final status, counts, version identifiers |
| `chat <model>` | One model call, its duration and known token usage |
| `execute_tool <tool>` | One dispatched tool attempt; retries produce additional spans |
| `evidence_admitted` | Source evidence identifier and location accepted by the executor |
| `verify_citations` | Structural citation-check result |
| `repair_started` | The graph started its bounded repair route |

Ask: which operation was slow, did it repeat, which tool admitted the cited evidence,
did verification fail, and was token usage known? Token counts may support the
backend's model-cost estimates when recognized; they are not an invoice or a measured
cost saving. Telemetry does not assign answer-quality scores. The separate evaluation
pipeline remains the place for human labels and offline judges.

## If a trace does not appear

- Check the optional `otel` dependencies, enabled flag, destination, and project keys in
  the same environment that started the process. Restart after changing settings.
- Verify that the chosen run has a terminal event. A currently running journal is skipped.
- Allow the background batch exporter to send. A short export script flushes on normal exit.
- Check terminal logs and the project's region/destination. Do not print authentication headers.
- A `telemetry.json` marker means spans were queued and suppresses repeat emission;
  it is not an acknowledgement that Langfuse received them. Delivery is best effort.
  Do not repeatedly re-export a marked run expecting guaranteed redelivery. After
  fixing configuration, produce a fresh stub run to test delivery.
- Unknown token usage on a failed/cancelled call is not zero usage.

Offline verification is available with `CRA_LIVE_MODEL=0 uv run pytest tests/unit/test_telemetry.py -rs`;
its OTLP tests use only a receiver on localhost.
Read [the decision log](decisions.md) for the privacy/export design and
[the evaluation protocol](evaluation_protocol.md#standalone-v2-evaluation-hand-v1)
before running a paid benchmark.

## Use one trace to test and review a real answer

Setup smoke check on 2026-10-01: both a stub trace and one real-model V2 UI trace
were visible in Langfuse Cloud. The live insurance example took about 6.8 seconds,
with three model calls and two tool calls (including verification), using 12,918
input tokens and 404 output tokens. These are one-run observations, not benchmark
results. Langfuse displayed an estimated cost of about $0.03.

A stub is a deterministic stand-in for the planner/writer. It follows scripted actions
and formats retrieved text without a model API call. Real retrieval, authorization,
tool execution, and citation checks still happen. Stub runs test that wiring; their
answers and estimated token counts cannot establish real-model quality or cost.

For one live insurance run, use this checklist:

1. Confirm the local UI says **Live** and the corresponding manifest names the real
   model. Locate the same run/trace in Langfuse; `cra.simulated` should be false.
2. In the span tree/timeline, inspect the model and tool sequence, calls, latency,
   token usage, errors, and verification outcome. These are execution checks.
3. Read the answer and source evidence in the local inspector. Does it cover both
   clinical testing and commercial distribution/sales? Does it preserve redactions
   rather than inventing coverage amounts? Does each factual claim follow from its
   own citation? These are semantic checks; Langfuse's metadata alone cannot answer them.
4. If useful, attach a manual score/comment using Langfuse's **Annotate** control while
   reviewing the local answer. Name exactly what was assessed, for example task
   success or citation support, and explain the evidence. A citation-gate flag is not
   a substitute for this review.
5. For the 41-case evaluation, use the standalone protocol and frozen human labels.
   This integration does not upload questions/answers, synchronize evaluation
   datasets, upload grader scores, or configure automatic Langfuse LLM judges.
   Do not run a whole paid suite merely to check that telemetry works.

The smoke run illustrates why these checks are separate. Its citations passed the
structural gate and its answer preserved the redacted insurance amounts. An
AI-assisted source comparison also found that the answer's notice sentence omitted
the source's qualification that a change must materially adversely affect Xencor's
rights. Record that as a review finding requiring human assessment, not a frozen
human label. A successful trace or citation gate does not guarantee complete wording.

The final meaning is simple: traces help explain **how execution behaved**; reviewing
the answer against evidence establishes **whether the answer was good**.
