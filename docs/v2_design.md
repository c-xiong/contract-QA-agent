# V2 implementation

## Corpus decision

Option B: retain the existing ingested corpus and index. The frozen `hand-v1` suite is
unchanged. Historical retrieval results remain V1 measurements; new V2 cases and
regression cases have separate artifacts and denominators. No expanded-corpus claim.

## Execution

`graph.build_graph(agent_mode="v2")` reuses `ResearchState`, `Budget`, the existing
writer/repair prompts, model adapter, retriever, source store and citation verifier.
`V2Runtime` adds initialize → decide → authorize → execute → decide conditional routing;
draft → mandatory verify → terminal or bounded repair shares the existing writer.
There is no second retrieval framework or multi-agent topology.

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

The paired runner retains every case and creates blank independent-review forms.
Deterministic diagnostics are available; task success, semantic support and useful calls
remain unscored until reviewed. `reports/v2_results.md` records actual validation.

Remaining before the plan's full evaluation definition of done:
- Author replacement/review of the generated new cases and rubric, with a new dataset version.
- Frozen live-model paired runs on both suites under matching retrieval/model settings.
- Independent manual labels, auditing every apparent V2 win and classifying each failure.
- A final capability report and CV wording based on those reviewed results.

No higher task-success claim is required to complete that experiment; a null or negative
result is valid. V2 remains opt-in while these steps are pending.
