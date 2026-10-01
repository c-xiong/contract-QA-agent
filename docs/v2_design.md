# V2 implementation

Scope: the implemented V2 runtime, its boundaries and deliberate limitations. For a
plain-language walkthrough, start with [the project guide](project-guide.md); use
[the documentation index](README.md) to find demonstrations and measured reports.
V2 is the active development path; V1 remains the runtime default and historical
reference. MCP and optional OTel export are implemented. Live V2 answer quality
has not yet been measured with the standalone human-review protocol.

## Corpus decision

Option B: retain the existing ingested corpus and index. The frozen `hand-v1` suite is
unchanged: 41 cases comprise 35 expected answers, one expected partial answer and five
expected abstentions. Retrieval-only evaluation scores the 36 evidence-bearing cases;
end-to-end evaluation keeps all 41. Historical retrieval results are separate from
live V2 answer-quality measurements; the 24 provisional V2 fixtures and the 119-query
CUAD retrieval validation also have separate artifacts and denominators. No expanded-
corpus claim.

## Execution

`graph.build_graph(agent_mode="v2")` reuses `ResearchState`, `Budget`, the existing
writer/repair prompts, model adapter, retriever, source store and citation verifier.
`V2Runtime` adds initialize → decide → authorize → execute → decide conditional routing;
draft → mandatory verify → terminal or bounded repair shares the existing writer.
There is no second retrieval framework or multi-agent topology.

The graph and local read-only MCP server share `ToolSession`/`ToolExecutor` for scope,
reservations, timeouts and original-evidence admission. The graph additionally owns
planning, model-call limits and final delivery. An external MCP client can skip or
ignore citation verification; server tool budgets do not cap that client's model calls
or token spending. See [the MCP entry point](../README.md#use-the-tools-from-an-mcp-client).

The planner emits one Pydantic `Action`; it cannot write arbitrary graph state. Python
validates schemas and scope, clamps top-k, reserves counters before dispatch, admits
original source chunks, chooses terminal state and invokes final verification. A public
`verify_citation` call never substitutes for the final gate. Empty/uncited drafts fail.
Scope violations are failures; provider outages are not evidence-based abstentions.

V1 retains its original behavior for comparison. V2 tightens two existing boundaries:
all evidence, including the first item, must fit the character budget; citation locators
must match one admitted chunk's document/page/section jointly. This still does not prove
that a natural-language claim is entailed by the source.

## Identity and provenance

Documents have no reliable legal version labels in current ingestion. `sha256:<hash>`
identifies the indexed chunks and coordinates of a snapshot. This cannot establish
which agreement supersedes another. Missing/mismatched requested versions fail closed.
Evidence IDs derive from chunk identity and original text; source locators remain nullable.
Offsets are **chunk-relative**, not invented offsets into the original PDF.

## Counters

Preserved: 3 search attempts, 5 hits/query, 24,000 admitted evidence characters, 1 repair.
Additional defaults: 12 tool invocations (including mandatory verification), 12 model
attempts (planner + writer + repair), 20 unique evidence spans, at most one retry per
failed read, and a 120-second deadline. Limits are captured before every run.

Every search retry spends another search and tool slot. Evidence counts unique admitted
spans; repair counts rewrites following failed verification. Duplicate action+arguments
without new evidence stop the loop. Invalid structured output/arguments permit one
correction in the entire run. An empty search permits at most one reformulation. A
remaining verifier slot is required before tool dispatch or drafting; repair must also
fit a model call and another verifier slot. Provider SDK retries are disabled so they
cannot silently evade accounting. Missing token usage is explicitly unknown.

Blocking retrieval runs outside the asyncio event loop so cancellation/deadlines remain
responsive. A cancelled native retrieval worker may finish its bounded read-only work;
it cannot admit evidence or finish the cancelled run. No external source writes exist.

## Durability and API

Each run has a mode-0700 directory, mode-0600 manifest and append-only fsynced JSONL
snapshots. Artifacts record code/worktree, corpus, document metadata, index fingerprint
(when available), model/prompt/tool versions, counters, results and terminal reason.
Text lives in local snapshots; SSE progress exposes IDs, status and counters only.
The final response can include evidence already authorized by the request.

There is exactly one terminal event for normal success, failure, abstention or
cancellation. SSE cleanup also handles disconnects between graph nodes. A reconnect
with `Last-Event-ID` is rejected rather than restarting. On next V2 agent startup,
traces whose recorded process is gone are marked failed, without executing them again.
This is local-process failure accounting, **not crash-resumable execution**. PID reuse
can delay detection; disk failure can prevent persistence. No distributed state store.

## Validation and remaining evaluation

The tests cover schema/scope/version rejection, arithmetic boundaries, retries,
source membership, strict budgets, empty drafts, repair exhaustion, tool-verifier bypass,
SSE redaction/cancellation, native-work deadlines and dead-process detection. Three
scripted synthetic fixtures demonstrate distinct paths. They are not live model evidence.

The historical paired runner retains every case and creates blank review forms.
The current standalone runner saves cases incrementally, enforces cumulative live-run
budgets, resumes under the same frozen configuration and prepares human review plus
judge calibration/audit. Deterministic diagnostics are available; semantic task success
and citation support require reviewed outputs. [The dated offline report](../reports/v2_results.md)
records earlier implementation validation, not current live capability or a current
test-count claim.

Remaining for the current standalone evaluation:

- Review the provisional answer-rubric judge and freeze the reporting configuration.
- Fill price/token/currency preflight and execute a bounded live V2 run on all 41
  `hand-v1` cases, including expected abstentions and execution failures.
- Human-label every output before exposing judge verdicts; calibrate on 20 cases and
  audit the frozen judge on 21. This split tests the judge, not QA generalization.
- Publish human-label quality metrics, separate judge-agreement/error metrics and
  inspected failures. Describe author review and independent review accurately.
- Review/replace the 24 generated cases under a new version before using them for any
  additional capability claims; they are not required for the initial `hand-v1` run.

[The evaluation protocol](evaluation_protocol.md#standalone-v2-evaluation-hand-v1) is
primary. A paired live comparison is optional unless claiming that V2 improves over V1.
No higher task-success claim is required to complete evaluation; a null or negative
result is valid. V2 remains opt-in while the live report is pending.
