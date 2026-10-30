# Design: A2A entrypoint for the contract research agent

Status: implemented. Approved in design review (Codex, round 2), then **slimmed by the
author's decision (2026-10-30)**; the slim scope below is what was built.
Author: Claude (orchestrator). Scope fixed by `brief.md`.

## Slim scope (user decision; overrides the sections below where they conflict)

Kept: card + `contract_qa` skill + streaming; executor with working/completed (cited
artifact)/failed/abstained-as-completed; adapter scope gate -> input-required with
data-only continuation; bearer auth; `progress` + `external_ref` hooks (Q1 yes) with
task_id/context_id in manifest and root span; minimal cancel; orchestrator CLI; README
commands (Q2: no Makefile); ~6-8 tests; README + decisions entry.

Dropped: TaskSlot lock / cancel-vs-complete race handling (replaced by: a send for a task
already running is rejected); stream-disconnect handling and raising-observer tests
(observer stays sync + exception-guarded, untested beyond one case); adapter journal and
`a2a_task` span for runless tasks (they are server-log lines only, stated limitation);
separate CI job (the existing jobs sync `--extra a2a`); cancel race tests.

## 0. What exists today (facts this design rests on)

| Fact | Evidence |
|---|---|
| One assembly point: `ResearchAgent(store, settings, retriever, client=...)`; one call: `await agent.research(question, allowed_document_ids=, budget=, requested_versions=) -> ResearchResult` | `app/agent/runner.py:63-152` |
| `build_agent(settings, arm)` already loads the corpus and degrades (FAISS -> BM25, live -> stub) with a note | `app/api/main.py:45-75` |
| V2 runs execute through `v2_states(graph, state, runs_dir)`, which assigns `run_id`/`trace_path`, and on `CancelledError` records `run_cancelled` in the journal | `app/agent/execution.py:33-53` |
| Terminal statuses: `completed`, `abstained`, `failed`, `cancelled`; `stop_reason` is a closed vocabulary (`insufficient_evidence`, `model_budget_exhausted`, `deadline_exhausted`, `invalid_document_or_version_scope`, `scope_or_provenance_violation`, `citation_verification_failed`, ...) | `app/agent/state.py:20-22`, `app/agent/v2.py:68-75, 198-479` |
| **The agent has no clarification state.** The planner prompt says "do not guess missing dates, versions, definitions" and resolves that by **abstaining**, not by asking. There is no LangGraph checkpointer and no thread id; each run is single-shot | `app/agent/v2.py:40-49`, `runner.py:120-129` |
| Only one snapshot per document is indexed; `requested_versions` defaults to it and a mismatch fails as `invalid_document_or_version_scope` | `app/agent/v2.py:157-198`, `tool_registry.py:137-140` |
| OTel export is V2-only, built from `runs/<run_id>/events.jsonl` + `manifest.json` at the terminal event, with an attribute allowlist (`MANIFEST_FIELDS`) | `app/observability/telemetry.py:1-24, 44-50, 81-121` |
| MCP server is a separate process (`app/mcp_server.py`, stdio) and calls the same `ToolExecutor`; it declares "protocol adapter, nothing here re-decides policy" | `app/mcp_server.py:1-20` |
| Optional OTel deps live in an extra (`otel`), with its own CI job | `pyproject.toml [project.optional-dependencies]`, `.github/workflows/ci.yml:77-85` |
| CI runs the main suite with no network (`unshare --net`) | `.github/workflows/ci.yml:52` |
| Default `agent_mode` is `v1` | `app/config.py:85` |

Two consequences shape the whole design:

1. **"Contract version is ambiguous" is not a real case in this corpus** (one snapshot
   per document). The demo must not invent one. The real, honest ambiguity is **which
   contract** a question is about when the caller gives no document scope.
2. **The agent never asks for clarification**, and the brief forbids changing its core.
   So `input-required` must come from the A2A adapter, as a deterministic pre-run scope
   policy, not from the LLM. Docs and CV must say so.

## 1. Architecture and placement

```
scripts/a2a_orchestrator.py  --(HTTP JSON-RPC + SSE, bearer token)-->  app/a2a/server.py
                                                                         |  AgentCard  (/.well-known/agent-card.json)
                                                                         |  ContractQAExecutor (app/a2a/executor.py)
                                                                         v
                                                       ResearchAgent.research()  [unchanged V2 LangGraph]
                                                                         |
                                                       ToolExecutor -> ToolRegistry (same tools MCP exposes)
```

New package `app/a2a/` (adapter only):

- `card.py` - builds the `AgentCard` from settings (pure function, testable).
- `executor.py` - `ContractQAExecutor(AgentExecutor)`: the lifecycle mapping.
- `auth.py` - bearer-token Starlette middleware.
- `server.py` - assembles routes + `uvicorn` entry; `python -m app.a2a`.

**DECISION: separate process and port (default 127.0.0.1:8100), not mounted into
`app/api/main.py`.** The FastAPI app is the read-only trace inspector with no auth;
mounting an authenticated A2A surface inside it mixes two trust levels in one app and
makes "the inspector has no auth" ambiguous. It also mirrors the MCP server, which is
its own process. Reuse is through `build_agent()` (corpus load + degradation notes), not
through the inspector app object. Alternative rejected: one FastAPI app with
`add_a2a_routes_to_fastapi` - fewer processes, but mixes auth boundaries.

**DECISION: the A2A server forces `agent_mode="v2"`.** Telemetry, durable run journals,
`run_id`, scope/version checks and cancellation journaling are V2-only (table above).
V1 has none of these, so a V1 A2A entry could not meet the tracing requirement.

## 2. SDK and dependency

- `a2a-sdk[http-server]` pinned `>=1.3,<2` in a new optional extra `a2a` (mirrors `otel`).
  Rationale: SDK pulls protobuf + google-api-core + json-rpc; base install, eval runs
  and the MCP path do not need them (CLAUDE.md rule 6).
- Implementation targets the **v1.0 API** (verified 2026-10-30 against the a2a-python
  `docs/migrations/v1_0/README.md`): protobuf types; `TaskState.TASK_STATE_*`;
  `Part(text=...)`, `Part(data=ParseDict(...))`; `AgentCard.supported_interfaces=
  [AgentInterface(protocol_binding="JSONRPC", url=...)]`; `DefaultRequestHandler(...,
  agent_card=...)`; routes from `create_agent_card_routes`, `create_jsonrpc_routes`
  composed into `Starlette(routes=...)`; client via `a2a.client.create_client`,
  responses as `StreamResponse` checked with `HasField`. Exact names are re-checked
  against the installed version before coding; any divergence is recorded in the module
  docstring. The a2a-samples LangGraph sample (0.3 API) is used for structure only.
- `InMemoryTaskStore`. No persistence (brief: out of scope). A restart loses in-flight
  and input-required tasks; documented as a limitation.
- No v0.3 compatibility mode.

## 3. Agent Card

| Field | Value |
|---|---|
| name | `Contract Research Agent` |
| description | Answers questions about commercial contracts in a fixed public corpus (CUAD/ContractNLI + labelled synthetic variants) with clause citations; abstains when evidence does not support an answer. Not legal advice. |
| version | `pyproject` project version |
| supported_interfaces | one `JSONRPC` interface at `http://{host}:{port}/` (from CLI flags) |
| capabilities | `streaming=True`, `push_notifications=False`, `extended_agent_card=False` |
| default_input_modes | `text/plain`, `application/json` |
| default_output_modes | `text/plain`, `application/json` |
| security_schemes | `bearer`: HTTP auth scheme `bearer` |
| security (requirements) | `[{bearer: []}]` |
| skills | one: `contract_qa` |

Skill `contract_qa`: description "Answer one question about one or more named contracts
in the indexed corpus; returns an answer with verified clause citations, or an explicit
abstention." Tags: `contract`, `legal`, `qa`, `citations`. Examples: two questions
taken from the existing hand suite wording style (e.g. "What is the notice period for
termination for convenience in <document>?"). Input: text plus optional data part
`{"document_ids": [...]}`. Output: text + data.

Served at the SDK's well-known path (`/.well-known/agent-card.json` per spec 1.0;
whatever `create_agent_card_routes` registers is authoritative). The card itself is
public (discovery must work before auth); only JSON-RPC routes require the token.

## 4. Executor: request -> agent -> task lifecycle

### 4.1 Input contract

Validation order matters (a clarification reply may be data-only):

1. If the message continues an existing task in `input-required` (same `task_id`),
   go to the continuation path in 4.2 **before** any question validation. The reply
   needs no text; only its `document_ids` data part is validated.
2. Otherwise it is a new task. Then, from the incoming `Message`:
- `question`: concatenated text parts. Empty -> reject with `InvalidParamsError` (no task).
- `document_ids`: optional, from a data part `{"document_ids": [...]}`. Unknown IDs ->
  task `failed` with reason `unknown_document_ids: [...]` (mirrors `/research` 400 at
  `app/api/main.py:245-250`; never silently widened to the whole corpus).
- Length cap 2000 chars (same as the SSE endpoint, `app/api/main.py:279`).

### 4.2 Scope gate -> `input-required`

**DECISION: `input-required` is an adapter scope policy, not an agent judgement.**
Rule: if the request carries no `document_ids` and the corpus holds more than one
document, the executor does not run the agent. It sets `TASK_STATE_INPUT_REQUIRED`
with an agent message: "Which contract? Reply with a data part
`{"document_ids": [...]}`", plus a data part listing the corpus documents as
`{document_id, title}`, sorted by `document_id`, capped at 100 entries with a
`truncated` flag (the shipped corpus has 56). Deterministic: no retrieval, no model call,
no ranking that could look like an answer; the gate cannot fail for retrieval reasons.

Why this rule and not others:
- It is the real ambiguity in this corpus (section 0, consequence 1) and the one a bank
  reviewer cares about: answering about the wrong contract is worse than asking.
- Alternative rejected: map the agent's `abstained` + `insufficient_evidence` to
  `input-required`. That would reinterpret a verified guardrail outcome as a request
  for more input, and an abstention is a correct final answer (CLAUDE.md rule 3).
- Alternative rejected: add a "clarify" action to the V2 planner. Changes the core
  (brief forbids) and the frozen evaluation suites.
- Alternative rejected: a version-ambiguity demo. No such case exists in the data.

Continuation: the follow-up message arrives with the same `task_id` (A2A continues an
`input-required` task). The executor recovers the original question from the task's
own history in the task store (first user message), merges `document_ids` from the
reply, and runs. No extra state store. A reply still without valid `document_ids` gets
`input-required` again (max 2 rounds, then `failed: scope_not_provided`). A reply with
unknown IDs -> `failed: unknown_document_ids` (same rule as 4.1). Text in a
clarification reply is ignored (it is not appended to the question: that would be
conversational memory by another name); this is stated in the clarification message.

**DECISION: `context_id` groups tasks; it does not carry memory.** CLAUDE.md rule 6
forbids conversational memory, and the agent has no thread/checkpointer. A new task in
the same context starts from scratch (it must name its documents again). `context_id`
and `task_id` are recorded in the run manifest for tracing (section 6). This is
"the simplest mapping" the brief allows.

### 4.3 Running and progress

Executor flow (V2 only):

1. Emit the `Task` first (SDK 1.0 requires Task before any update), then
   `working` + text "Received; checking scope".
2. Run `agent.research(question, allowed_document_ids=..., progress=cb, external_ref=...)`
   as an `asyncio.Task` held per `task_id` (for cancel).
3. Progress: `cb(state)` is a **synchronous, non-blocking** observer called on each
   merged V2 state. It only derives a phase label and does `queue.put_nowait(label)` on
   a bounded `asyncio.Queue(maxsize=16)`; on `QueueFull` it drops the update (progress
   is best-effort, the artifact is not). A separate drain task turns labels into
   `working` status events; slow SSE consumers therefore never consume the research
   deadline. Phase labels, emitted only on change: "Searching clauses",
   "Drafting answer", "Verifying citations". No question text, evidence or prompts.

### 4.3a Concurrency and terminal ownership

The adapter, not the SDK, is the authority for "one task -> at most one run, exactly one
terminal state". Per process, a dict `task_id -> TaskSlot{lock, run_task, terminal}`:

- A send for a task whose slot has a live `run_task` is rejected with the SDK's
  invalid-state error (no second run). Two concurrent clarification replies: the first
  to take `lock` resumes; the second sees `run_task` set and is rejected.
- Terminal publication (`artifact+completed`, `failed`, `canceled`) happens only
  under `lock` and only if `terminal is None`; it sets `terminal` first. So
  cancel-vs-complete is "first under the lock wins"; the loser publishes nothing.
  If cancel wins after the run finished, the journal still says `completed` - stated
  limitation, and the task data part never claims otherwise because nothing is
  published after cancel.
- The slot is removed in a `finally` after terminal publication; the same `finally`
  cancels and awaits the drain task, so it can never outlive its run.
- Stream disconnect: a dropped SSE consumer does **not** cancel the run (A2A tasks
  outlive a stream; the caller can `tasks/get` or re-subscribe, and only an explicit
  cancel request cancels). The drain task writes to the SDK event queue; if that write
  fails because the consumer is gone, the drain stops itself (progress is best-effort)
  and the run and terminal publication continue. How the installed SDK's event queue
  behaves when its consumer is closed is verified during implementation and covered by
  a test (disconnect mid-run -> run still reaches its terminal state, task store shows it).
- Whatever the SDK already guarantees about task-state transitions is verified against
  the installed version during implementation and noted; the lock is kept regardless,
  since it is cheap and the guarantee is load-bearing.

### 4.4 Terminal mapping

| Agent outcome | A2A state | Payload |
|---|---|---|
| `completed` (`stop_reason=verified`) | `artifact` then `TASK_STATE_COMPLETED` | Artifact `contract_answer`: text part = answer; data part = `{status, run_id, stop_reason, citations:[{document_id,title,section,page,quote,...}], documents_searched, model_id, simulated}` (rendered by the existing `render_citations`) |
| `abstained` | artifact then `COMPLETED` | same artifact shape, `status:"abstained"`, answer = abstention text, `citations:[]`. Abstention is a correct outcome, not a failure. |
| `failed` (any `stop_reason`, incl. budget/deadline exhaustion, scope violation, model unavailable) | `TASK_STATE_FAILED` | status message: readable reason = `stop_reason` + fixed human sentence per code; data part `{run_id, stop_reason, failure}`. No exception text, no stack trace. |
| exception escaping `research()` | `FAILED` | `stop_reason: internal_error`; logged locally with run_id. Never reported as abstained or completed. |
| cancel request | `TASK_STATE_CANCELED` | see 4.5 |
| scope gate | `TASK_STATE_INPUT_REQUIRED` | see 4.2 |

`rejected` is not used: the agent does not decline tasks by policy; abstention is an
answer.

### 4.5 Cancellation

`cancel()` cancels the held `asyncio.Task`. `v2_states` already converts
`CancelledError` into a durable `run_cancelled` journal entry (`execution.py:44-53`),
so the trace is closed consistently. The executor then emits `TASK_STATE_CANCELED`.
Cancelling a task in `input-required` just transitions it (nothing is running).
Cancelling a terminal task returns the SDK's not-cancelable error.

## 5. Changes outside `app/a2a/`: two additive, behaviour-neutral hooks

(Plus, non-runtime: `telemetry.emit_adapter_span()` from section 6, `Settings.a2a_token`,
`pyproject` extra, CI matrix from section 8, README/decisions. Nothing else.)

The executor needs streaming progress and trace correlation. `research()` today returns
only the final result. Two optional keyword arguments are added to
`ResearchAgent.research()` and nothing else:

- `progress: Callable[[ResearchState], None] | None = None` - a **synchronous**
  observer invoked inside the existing `async for merged in states` loop
  (`runner.py:122-126`). It is not awaited, so it cannot hold the generator open across
  the deadline. Any exception it raises is caught in `research()` (logged once with
  run_id, observer disabled for the rest of the run) and never propagates into
  `v2_states`, so it cannot turn a run into `cancelled`/`execution_interrupted`
  (`execution.py:39-53`). V1 path untouched. Default `None` = identical behaviour.
- `external_ref: dict[str, str] | None = None` - stored in state and copied into the V2
  manifest as `external_ref` (`v2.py:163-190`), and added to telemetry's allowlist as
  `cra.a2a.task_id` / `cra.a2a.context_id` (`telemetry.py:44-50, 109-111`). IDs only;
  no question text (respects the telemetry DECISION at `telemetry.py:9-13`).

These touch runner/manifest/telemetry metadata, not graph topology, routing, retrieval,
prompts, budgets or citation checks. Tests (section 8, test 10): a scripted V2 run gives
the same terminal state and event-type sequence (a) without hooks, (b) with a no-op
observer, (c) with an observer that raises on every call; and cancellation while the
drain task is blocked yields A2A `canceled` and journal `run_cancelled` together.
Alternative rejected: re-implementing the `v2_states` loop and `ResearchResult`
construction inside the executor - duplicates `runner.py:120-152` and drifts.
Alternative rejected: parse SSE frames from `app/api/stream.py` - couples to the
inspector's wire format.

## 6. Security and observability

**Auth (DECISION: static bearer token, fail closed).**
- `CRA_A2A_TOKEN` (`SecretStr` in `Settings`). Server refuses to start if unset. No
  "insecure" flag.
- Starlette middleware on all non-card routes: `Authorization: Bearer <token>`, compared
  with `hmac.compare_digest`; missing/wrong -> HTTP 401 with `WWW-Authenticate: Bearer`,
  no body detail. The token is never logged or put in traces.
- Card declares the scheme (section 3) so a client knows what to send.
- Binds `127.0.0.1` by default.
- Limitation stated in docs: one shared token = authentication of a caller, not per-user
  authorization or tenant isolation (multi-tenancy is out of scope by rule 6). A bank
  deployment would replace it with OAuth2 client credentials / mTLS; the card format
  already supports declaring that.

**Scope enforcement is not weakened by the new entry.** The A2A caller can only narrow
scope via `document_ids`; the V2 `initialize` scope/version check
(`v2.py:190-198`) and `ToolExecutor` authorization still apply unchanged. Prompt
injection in the question is handled by the same planner prompt and tool policy as
every other entry - the adapter adds no prompt text.

**Tracing.**
- Every A2A task that reaches the agent produces one V2 run: `runs/<run_id>/` journal +
  manifest containing `external_ref.a2a_task_id/context_id`. With
  `CRA_TELEMETRY_EXPORT=1` the existing exporter emits the root span with
  `cra.a2a.task_id` and `cra.a2a.context_id` attributes, so Langfuse search by task id
  finds the full run trace.
- The artifact data part carries `run_id`, so the orchestrator side can also correlate.
- Trace coverage, defined explicitly:
  - Accepted task that runs the agent: the V2 run trace (journal + root span) carries
    `a2a_task_id`, `context_id` and `clarification_rounds`. This is the full trace.
  - Accepted task that ends without a run (`scope_not_provided`, `unknown_document_ids`,
    canceled while `input-required`): the adapter appends one IDs-only line per
    lifecycle transition to `runs/a2a/<task_id>.jsonl` (`{ts, task_id, context_id,
    state, reason}`; 0600, same convention as the run manifest), and, if telemetry is
    on, emits one span `a2a_task` with those attributes through the same private
    provider (`telemetry.py` gains a tiny `emit_adapter_span()`; no global provider).
    Every accepted task therefore has a durable, correlatable record.
  - Not covered: requests rejected before a task exists (401, empty question). They
    are server-log lines only; they are not tasks.
- No SDK-level OTel (`a2a-sdk[telemetry]`) in this iteration: it would use the global
  tracer provider, which `telemetry.py:15-16` deliberately avoids.

## 7. Orchestrator CLI (`scripts/a2a_orchestrator.py`)

```
uv run python scripts/a2a_orchestrator.py --url http://127.0.0.1:8100 \
    "What is the termination notice period?" [--document-id ID] [--clarify-with ID]
```
1. Fetch the card (no token), print name/skills/auth scheme.
2. Skill check: proceed only if a skill tagged `contract` exists; else exit 2 with
   "no capable agent". (Single card; no routing across agents - out of scope.)
3. `create_client(card)` with the bearer token from `CRA_A2A_TOKEN`; send streaming.
4. Print each status update; on `input-required`, print the candidates and answer with
   `--clarify-with` (non-interactive, for a reproducible demo) or a stdin prompt if
   absent and stdin is a TTY; send on the same task id/context id.
5. Print final artifact (answer + citations) or failed reason; exit code 0/1.

Demo commands (README):
- serve: `uv run --extra a2a python -m app.a2a --port 8100` (forces V2 itself)
- demo: `uv run --extra a2a python scripts/a2a_orchestrator.py --url ... "<Q>"
  --clarify-with <ID>` with no document id on the first send. The README pins one
  concrete question/document pair from the shipped corpus, chosen during
  implementation by actually running it with the offline stub and confirming a cited
  artifact; the pair is recorded in README, not left as a placeholder.
  (No Makefile exists today; these are two documented `uv run` commands. See Q2.)

Offline: with `CRA_LIVE_MODEL=0` the stub client runs; the demo is labelled "stub
model" and the artifact `simulated:true`. A live demo needs the API key, as today.

## 8. Tests (all offline, in-process via `httpx.ASGITransport`; no sockets)

`tests/unit/test_a2a.py`, guarded with `pytest.importorskip("a2a")`.

Dependency/CI matrix (current CI: `ci.yml:34,43,77-85`):
- Default job (`--all-groups`, no extras): `test_a2a.py` skips; nothing outside
  `app/a2a/` and `scripts/a2a_orchestrator.py` imports `a2a` at module level, so the
  rest of the suite is unaffected.
- Lint/type job: sync changes to `--all-groups --extra a2a` so
  `mypy app evals scripts experiments` type-checks the new modules for real (no
  `ignore_missing_imports` blanket).
- New `a2a` job: `uv sync --locked --all-groups --extra a2a --extra otel` +
  `pytest tests/unit/test_a2a.py -rs`, and the job fails if any test in that file is
  skipped (`-rs` output checked, or `--strict` skip guard), so the OTel correlation
  test (9) and all others actually execute.

Fixtures: (i) small synthetic store + `ResearchAgent` with an injected scripted client
(pattern of `tests/unit/test_v2.py`) for controlled outcomes; (ii) the **real offline
stub** (`StubClient`, `v2.py:201-247`) for the end-to-end demo test, which drives the
real path the README demo uses.

1. Card: GET well-known path, 200 without token; parses as `AgentCard`; has
   `contract_qa`, `streaming=True`, bearer scheme + requirement.
2. Auth: JSON-RPC without/with wrong token -> 401; correct token -> accepted.
3. Happy path: with `document_ids` -> stream contains Task, >=1 `working`, artifact
   with >=1 citation whose `document_id` is in scope, then `completed`.
4. Input-required: no `document_ids` -> `input-required` with candidates, agent not
   invoked (scripted client call count 0); follow-up on same task id with
   `document_ids` -> `completed`. Reply without IDs twice -> `failed:scope_not_provided`.
5. Abstention: scripted abstain -> `completed`, artifact `status=abstained`, no
   citations (not `failed`, not `input-required`).
6. Injection: reuse the scenario of `test_v2.py:543` (scripted out-of-scope search)
   through A2A -> same outcome as direct `research()`: `failed`, evidence only from the
   allowed doc. The test compares A2A outcome to the direct call, so the guardrail
   equivalence is asserted, not assumed.
7. Failure: scripted budget exhaustion -> `failed` with `stop_reason` in the message;
   unknown document id -> `failed: unknown_document_ids`; client raising -> `failed:
   internal_error`, never `completed`.
8. Cancel: slow scripted client; cancel -> `canceled`; journal ends with `run_cancelled`.
9. Trace link: manifest contains `external_ref` with the task id; with the in-memory
   tracer (`telemetry.install`) root span has `cra.a2a.task_id`.
10. Characterization (section 5): no hooks / no-op observer / raising observer give the
    same terminal state and event-type sequence; cancel during blocked drain -> A2A
    `canceled` + journal `run_cancelled`.
11. Data-only continuation: reply to `input-required` with only a data part (no text)
    -> resumes with the original question (regression for review B1).
12. Concurrency: two concurrent continuations of one task -> exactly one run (one
    `run_id` dir), the other rejected; cancel racing completion -> exactly one terminal
    event published.
13. End-to-end demo, real stub: the orchestrator's `run()` function (imported, not
    subprocess) against the in-process ASGI app over the synthetic store: discovery ->
    `input-required` -> clarification with a fixed document id -> artifact with >=1
    citation and `simulated: true` -> exit code 0. If the stub cannot produce a cited
    answer on the chosen pair, the pair is changed, not the assertion.

Plus required gates: `ruff check`, `ruff format --check`, `mypy app evals`, full
offline `pytest`.

## 9. Docs

- README: "A2A entrypoint" section - the diagram in section 1 extended with MCP
  (`external MCP client -> app/mcp_server.py -> ToolExecutor`), the two commands,
  auth setup, what the demo shows, limitations (in-memory tasks, shared token,
  adapter-level clarification, stub vs live).
- Design trade-off, two sentences: tools go over MCP because they are passive, typed,
  side-effect-bounded functions the caller orchestrates; the agent goes over A2A because
  it is an opaque collaborator that owns its own plan, budgets and guardrails and
  reports a task lifecycle instead of a return value.
- docs/decisions.md: one entry for the A2A adapter (sections 1, 4.2, 5, 6).
- This design moves to `docs/a2a_design.md` once approved.

## 10. Explicitly out of scope

Second agent / Retail Support, multi-agent routing, push notifications, persistent
task store, gRPC/REST bindings, v0.3 compat, OAuth/mTLS, per-user authz, rate
limiting, conversational memory, any change to graph/prompts/retrieval/citation logic,
CV edits, eval runs over A2A (no new metric is claimed from this work).

## 11. Open questions for the user

- Q1. Section 5 adds two optional args to `research()` and two manifest/telemetry
  allowlist fields. Acceptable as "not changing core logic"? (Author: yes - metadata
  and an observer callback only, guarded by a characterization test.)
- Q2. Makefile targets vs plain `uv run` commands in README. (Author: plain commands;
  the repo has no Makefile.)
- Q3. Is adapter-level `input-required` (missing contract scope) honest enough for the
  CV, given the agent itself never asks? (Author: yes, if worded "A2A task lifecycle
  incl. input-required scope clarification", not "the agent asks clarifying questions".)
