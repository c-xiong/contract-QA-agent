# Contract RAG Agent: Interview Questions and Defensible Answers

Prepared against reference commit `01a04f8` and the current working-tree implementation.
This is a project-specific practice guide, not a list of questions verified from
particular employers. Answers describe the inspected code, not future completion.

## How to use this guide

Start with the [documentation index](README.md) and [project walkthrough](project-guide.md)
if the architecture is new to you. Use the [UI and Langfuse walkthrough](ui-langfuse-walkthrough.md)
to connect these answers to an execution you can inspect.

Practice each answer in 60–90 seconds: explain the problem, your implementation,
the tradeoff, the evidence, and the remaining limitation. Open the linked code and
trace one example before memorizing any numbers. Use “I implemented” only for code
you can explain; use “I would” or “the next milestone is” for proposed work.

Current status matters:

- V2 exists and is opt-in; the README still describes V1 as the default baseline.
- The local MCP server, shared tool executor, and optional OpenTelemetry exporter
  are implemented. Tests exercise MCP over stdio and OTLP against a local receiver;
  manual stub and live V2 UI smoke traces were viewed in Langfuse Cloud on 2026-10-01.
  This verifies connectivity and execution visibility, not answer quality.
- Standalone V2 execution, resume, budget ledgers, human-label freezing and judge
  comparison are implemented. A live V2 quality evaluation remains pending, and the
  new answer-rubric prompt is provisional until author review and calibration.
- Historical retrieval results are available. The V2 validation report contains
  offline/stub results, not measured live V2 answer-quality improvements.

Update completion claims using actual test and experiment artifacts. Implementation,
a successful cloud demonstration, and measured answer quality are separate milestones.

## How the four CV bullets map to the implementation

The proposed wording is in [resume-bullets.md](resume-bullets.md); it is a draft for
review, and does not change the CV source.

| Bullet | What to explain in an interview | Where to practice |
| --- | --- | --- |
| LangGraph, FastAPI/SSE and MCP | The model proposes typed actions; Python authorizes and executes them. LangGraph owns routing, FastAPI/SSE shows the actual route, and MCP lets another agent call the same tools through a local stdio interface. MCP does not orchestrate that external agent. | Questions 2–3, 21 and 30–31; [V2 design](v2_design.md). |
| Retrieval and recall | BM25 finds lexical matches, local dense embeddings find related wording, RRF combines ranks, and a cross-encoder reranks candidates. Contract query expansion and document coverage are explicit selectable improvements. Explain the denominator and distinguish retrieval from answer accuracy. | Questions 11–15 and 26–27; [retrieval results](retrieval-improvement-results.md). |
| Scope, evidence and budgets | Startup/request scope and content-hash snapshots constrain tools; admission limits constrain context; tool/model/search budgets and the final citation gate constrain the run. A real source location still does not prove a sentence is supported. | Questions 16–22; [tool contracts](tool_contracts.md). |
| OpenTelemetry and evaluation | The durable local journal becomes a metadata-only trace after termination; the OTLP backend can be Langfuse. Evaluation separately measures retrieval, citation structure, semantic support and human-reviewed answer quality; resumable experiment artifacts and a spending ledger make failures visible. | Questions 23–25 and 32; [evaluation protocol](evaluation_protocol.md) and [UI/trace walkthrough](ui-langfuse-walkthrough.md). |

For a 60-second project explanation, connect the four ideas: preserve contract
evidence, let the planner choose bounded tools, inspect the execution, and measure
whether the resulting answer is supported and complete.

## A. Project framing and architectural judgment

### 1. What did you build, and what is technically difficult about it?

**Suggested answer:** “I built a bounded contract research agent over public CUAD
and ContractNLI data, plus explicitly labeled synthetic variants. The difficult part
is preserving the relationship between a legal conclusion and its source: a clause
may depend on definitions, exceptions, or another section. My system combines scoped
retrieval, source-preserving evidence, typed tools, execution limits, and citation
checks. I evaluate retrieval separately from answer correctness because finding the
right passage does not mean the model interpreted it correctly.”

**Follow-up:** Walk through one question requiring both a main clause and an exception.
Do not claim legal expertise or production legal validation.

**Evidence:** [README](../README.md), [evaluation methodology](eval-methodology.md).

### 2. Why an agent rather than a fixed RAG pipeline?

**Suggested answer:** “A fixed pipeline is a useful baseline for direct lookup.
V2 lets the model choose whether to search, retrieve a referenced clause, resolve a
definition, calculate a date, or draft an answer. The model proposes an action;
Python controls what can execute. This adds flexibility for multi-step questions,
but also adds planner calls and failure modes. I have implemented those capabilities;
I have not yet established that live V2 improves every task over the fixed workflow.”

**Follow-up:** Which question category needs adaptive routing? What cheaper approach
would you try first? An architecture diagram is not evidence of a quality gain.

**Evidence:** [V2 design](v2_design.md), [V2 runtime](../app/agent/v2.py).

### 3. Why LangGraph, and where does correctness actually come from?

**Suggested answer:** “LangGraph expresses conditional transitions and makes the
execution path explicit. Correctness boundaries come from my schemas, authorization,
budgets, evidence admission, and final verification, not from the framework name.
State uses a TypedDict because nodes exchange partial updates; external inputs and
tool contracts use Pydantic. The tradeoff is that TypedDict does not validate state
assignments at runtime, so boundary validation and regression tests matter.”

**Follow-up:** Draw the route from decide to authorize to execute, then draft, verify,
repair, and terminal. Explain which transitions the model cannot override.

**Evidence:** [state](../app/agent/state.py), [V2 design](v2_design.md).

## B. Chunking, source structure, and truncation

### 4. How do you chunk documents? What are the exact units and defaults?

**Suggested answer:** “The default target is 512 estimated tokens, using four
characters per token, with zero overlap. That is a character-based estimate, not the
embedding model's tokenizer. Chunking respects page and detected section structure,
then uses paragraph boundaries, sentence-like boundaries including periods and
semicolons, and finally hard character splits when necessary. Very small tails can
merge back into the previous chunk, so the target is not a strict tokenizer limit.”

**Follow-up:** Why zero overlap? “It avoids automatic duplication, but it is a default
choice rather than an empirically optimal result. I would test overlap on boundary-
sensitive examples before claiming it is best.”

**Evidence:** [configuration](../app/config.py),
[chunker](../app/ingestion/chunker.py), especially defaults and split/tail logic.

### 5. Why not simply use larger chunks or overlap everywhere?

**Suggested answer:** “Larger chunks preserve more context but dilute retrieval
signals, consume more context, and may be truncated by embedding or reranking models.
Overlap reduces some boundary loss but duplicates text and can crowd the top-k list
with near-identical evidence. I would compare these choices using complete evidence
coverage, final answer quality, and cost, rather than optimizing chunk size in isolation.”

**Follow-up:** Would you change retrieval units and citation units together? Explain
why smaller retrieval windows can point back to larger immutable source chunks.

### 6. What happens if a clause is split in the middle?

**Suggested answer:** “First I distinguish text preservation from context recovery.
The splitter tries natural boundaries, but a long paragraph can still be mechanically
split, and chunks never cross pages. Continuations can retain section metadata, but
that does not make the missing neighboring text visible. V2 can retrieve original
chunks for a section using `retrieve_clause`; it returns at most five chunks and
flags truncation. That is a bounded recovery mechanism, not proof that the complete
clause has been reconstructed.”

**Follow-up:** If those five chunks are insufficient, would you paginate, expand around
a source offset, or abstain? These are possible extensions, not current guarantees.

**Evidence:** [chunker](../app/ingestion/chunker.py),
[tool registry](../app/agent/tool_registry.py), [tool contracts](tool_contracts.md).

### 7. Does your embedding model actually read the entire chunk?

**Suggested answer:** “Not necessarily. The MiniLM embedder has a 256-wordpiece input
limit, while chunks target 512 estimated tokens. The audit found 1,358 of 6,270 bodies
exceeding a 254-wordpiece body budget even before headings. I implemented an experimental
tokenizer-aware windowed embedding arm that retains original citation chunks. It uses
bounded heading text and overlapping windows. However, that change did not improve
the development retrieval benchmark, so it is not the selected default.”

**Follow-up:** Why might the fix not improve recall? Offer hypotheses, not facts:
the informative text may already be near the beginning, extra windows may introduce
ranking noise, or reranking may remain the bottleneck. Check per-case ranks.

**Evidence:** [embeddings](../app/retrieval/embeddings.py),
[retrieval experiments and truncation audit](retrieval-improvement-results.md).

### 8. How is embedding truncation different from reranker and prompt truncation?

**Suggested answer:** “They happen at different stages. Embedding truncation can
hide a relevant passage before candidate retrieval. Reranker truncation can misorder
a candidate that was successfully retrieved. Evidence-budget admission can exclude
a whole retrieved item before generation. I would instrument tokenized lengths at
each model boundary. The current reranker sends the query, heading, and chunk body
to the CrossEncoder; it does not implement explicit window-based truncation repair.
I should inspect the pinned model configuration before quoting its effective limit.”

**Follow-up:** Do not respond to every truncation problem with ‘increase overlap.’
Identify the actual stage and show where the lost information disappears.

**Evidence:** [reranker](../app/retrieval/rerank.py),
[embeddings](../app/retrieval/embeddings.py), [V2 admission](../app/agent/v2.py).

### 9. How do you handle cross-references, definitions, and exceptions?

**Suggested answer:** “I preserve section metadata and original source references.
The existing cross-reference resolver supports bounded same-document expansion;
its default depth is one with a separate four-chunk budget. V2 additionally exposes
explicit clause and definition tools for model-selected lookup. I do not conflate
the older automatic expansion path with V2's tool behavior. Definition lookup is
conservative lexical candidate retrieval, not a complete legal parser, and ambiguous
matches must remain visible.”

**Follow-up:** How do you prevent recursive reference cycles or applying a definition
from a different agreement? Explain document/snapshot scope and bounded expansion.

**Evidence:** [cross-reference resolver](../app/evidence/cross_reference.py),
[tool contracts](tool_contracts.md).

### 10. Can you reliably handle every PDF, table, or scanned agreement?

**Suggested answer:** “No. I preserve extracted source text and coordinates, but I
would not claim universal layout or OCR support. Before blaming the retriever, I
check whether extraction preserved reading order, section headings, table relationships,
and the relevant text. Source-text corpora may have no PDF page coordinate, and I
keep that field null rather than inventing a page number.”

**Follow-up:** Design an ingestion audit using source-to-text spot checks and answer
span round trips. Describe OCR or table extraction as future work unless demonstrated.

**Evidence:** [evaluation methodology](eval-methodology.md), [V2 provenance](v2_design.md).

## C. Retrieval and ranking

### 11. Why combine BM25 with dense retrieval?

**Suggested answer:** “Contract questions contain both exact lexical signals, such
as defined terms and clause references, and semantic paraphrases. BM25 and dense
retrieval address different failure modes. I combine ranked candidates with reciprocal
rank fusion and optionally rerank them with a cross-encoder. The benchmark supports
testing hybrid retrieval here, but I do not assume dense retrieval is always stronger.”

**Follow-up:** Explain RRF as a sum of reciprocal rank contributions rather than an
addition of incomparable BM25 and cosine scores. Read the implementation before
quoting its fusion constant or candidate-pool configuration.

**Evidence:** [retrieval results](retrieval-improvement-results.md),
[historical experiment definitions](eval-methodology.md).

### 12. Why use a cross-encoder after retrieving candidates?

**Suggested answer:** “The initial retrievers cheaply generate candidates; a
cross-encoder scores the query and candidate together for finer relevance ranking.
That costs more per candidate, so I bound the reranking pool. The evaluated retrieval
improvement retained a 40-candidate cross-encoder budget. Reranking cannot recover a
passage absent from its candidate pool, and it can also demote a correct candidate.”

**Follow-up:** Separate candidate recall from final top-k recall. If the answer is
already in the candidate pool, increasing that pool may not solve the failure.

**Evidence:** [reranker](../app/retrieval/rerank.py),
[retrieval failure analysis](retrieval-improvement-results.md).

### 13. What retrieval improvement can you actually defend?

**Suggested answer:** “The optional legal-query coverage reranking arm increased
document-aware evidence recall@5 from 61.1% for the previous hybrid-plus-rerank arm
to 72.7% on 36 development questions. On 119 original CUAD questions across 23
development-disjoint contracts, the corresponding scores were 70.2% and 71.8%.
The independent gain is modest and uncertain. These are evidence-location metrics,
not answer accuracy, and the optional arm did not silently replace runtime defaults.”

**Follow-up:** State percentage points, not an ambiguous ‘percent improvement.’
Explain category regressions, uncertainty, and the scoped single-document setting.

**Evidence:** [full results, ablations, and intervals](retrieval-improvement-results.md).

### 14. Does document filtering guarantee good retrieval within the allowed scope?

**Suggested answer:** “Filtering prevents returning disallowed documents, but that
is separate from recall. The dense path uses bounded overfetch followed by scope
filtering. A relevant allowed chunk can sit below the global candidate cutoff, so
scope correctness does not imply exhaustive scoped retrieval. I would measure this
before considering prefiltered search or separate indexes.”

**Follow-up:** Explain why changing the filter placement changes recall and cost.
Do not describe an allowlist as user authentication or tenant isolation.

**Evidence:** [dense retrieval](../app/retrieval/dense.py).

### 15. Why FAISS rather than a hosted vector database? How would this scale?

**Suggested answer:** “The current corpus is small and locally reproducible, so a
local index avoids service overhead. I have not established production-scale throughput.
At larger scale I would measure index memory, filtered recall, update requirements,
concurrency, and latency before choosing another storage system. A vector database
would address operational requirements; it would not automatically fix chunking or
answer quality.”

**Follow-up:** Discuss stale indexes, snapshot identity, deletion, and metadata
filtering as design requirements rather than features already implemented.

## D. Agent reliability and trust boundaries

### 16. How do you prevent an agent from looping or spending indefinitely?

**Suggested answer:** “V2 enforces limits in code: defaults are three searches,
five hits per query, twelve tool calls including mandatory verification, twelve
model attempts, one repair, twenty evidence items, 24,000 evidence characters, and
a 120-second deadline. Search retries consume another search and tool slot. Duplicate
actions without new evidence stop the loop, and SDK retries are disabled to avoid
hidden attempts. These are current defaults, not universal optimal values.”

**Follow-up:** Explain why a verifier slot is reserved and how repair consumes the
remaining budget. Distinguish application-call limits from a currency cap.

**Evidence:** [Budget](../app/agent/state.py), [counter semantics](v2_design.md).

Paid standalone experiments additionally use a persisted token/cost ledger before
provider calls. Its input reservation uses a conservative character-to-token estimate,
then settles to reported usage; unknown usage keeps its reservation. Explain it as a
spending control with an estimation assumption, not an exact tokenizer-backed guarantee
for arbitrary prompts. The ordinary UI run has application budgets, not this experiment
ledger. See [budget implementation](../evals/budget.py).

### 17. What is the difference between a tool call being valid and being useful?

**Suggested answer:** “Schema and scope checks can establish that a call is permitted
and well formed. They cannot establish that it helps answer the question. A date
calculation can be arithmetically correct but use the wrong clause. I evaluate tool
usefulness and task success separately and accept multiple valid tool trajectories
rather than requiring one exact sequence.”

**Follow-up:** Give an example of a legal but unnecessary tool call and its cost.

**Evidence:** [tool contracts](tool_contracts.md), [evaluation protocol](evaluation_protocol.md).

### 18. Does citation verification eliminate hallucinations?

**Suggested answer:** “No. My deterministic verifier checks source locations,
scope, and admission. V2 requires document, page, and section to correspond jointly
to admitted evidence. A real citation may still fail to support the sentence, or the
answer may omit an exception. That is why I separate structural validity, semantic
support, completeness, and answerability.”

**Follow-up:** Use this example: ‘30 days' notice is sufficient’ can be misleading
when the contract also prohibits termination during the first year.

**Evidence:** [V2 design](v2_design.md), [offline failures](../reports/failure_analysis.md).

### 19. What happens when verification fails or the evidence is insufficient?

**Suggested answer:** “The graph owns final delivery. A tool-level verification
call cannot replace the mandatory final gate. Failed verification can trigger one
bounded repair when resources permit; otherwise the run terminates without delivering
an unverified final answer. Insufficient evidence and infrastructure failure must
remain different outcomes. I still need semantic evaluation of whether the live
planner chooses appropriate abstention.”

**Follow-up:** A completed runtime status is not a task-success label. Explain the
stub case that completed with a valid citation on an expected-abstention question.

**Evidence:** [V2 runtime](../app/agent/v2.py), [offline failure report](../reports/failure_analysis.md).

### 20. How do you handle prompt injection in retrieved documents?

**Suggested answer:** “Contract text is data, not permission to change execution
policy. Tool inputs still pass schema and scope checks, and the planner cannot
write arbitrary state. These controls constrain actions, but do not prove the model
will ignore all malicious instructions or avoid answer contamination. I would test
adversarial source passages and distinguish rejected unauthorized actions from
semantically corrupted but structurally valid answers.”

**Follow-up:** Do not claim comprehensive prompt-injection resistance from an allowlist.
Describe a test where a source tells the model to ignore an exception or invent a citation.

### 21. What happens on cancellation, timeout, reconnect, or process crash?

**Suggested answer:** “The API streams redacted progress and records durable local
events. Cancellation stops the run; blocking retrieval is off the event loop, although
a native worker can finish its bounded read-only work after cancellation. It cannot
admit evidence into the cancelled run. Reconnect with Last-Event-ID is rejected instead
of silently starting new work. Startup can mark dead-process traces failed, but this
is failure accounting, not crash-resumable execution.”

**Follow-up:** Explain the difference between cancelling an await and forcibly
terminating native computation. Mention disk failure and PID-reuse limitations.

**Evidence:** [durability design](v2_design.md), [events](../app/observability/events.py).

### 22. How do you represent contract versions?

**Suggested answer:** “A content-derived snapshot identifier identifies the indexed
chunks and their coordinates. It supports reproducibility and rejects mismatched
requested snapshots. It does not tell me which agreement legally supersedes another.
I also preserve nullable source locators and chunk-relative offsets rather than
inventing original-document coordinates.”

**Follow-up:** How would legal precedence differ from byte/content identity?

**Evidence:** [identity and provenance](v2_design.md), [registry](../app/agent/tool_registry.py).

## E. Evaluation and failure-driven improvement

### 23. How do you evaluate a RAG agent end to end?

**Suggested answer:** “I separate retrieval coverage, citation validity, semantic
claim support, answer correctness/completeness, abstention, and operational cost.
Each answers a different question. The evaluation infrastructure includes frozen
tasks, per-case artifacts, provenance, separate failure categories, resumable V2
execution and persisted paid-run budget ledgers. The live V2 quality report is still
pending; its reported quality metrics use human-reviewed labels, with judge agreement
reported separately.”

**Dataset counts:** `hand-v1` has 41 tasks: 35 expect an answer, one expects a partial
answer, and five expect abstention. The retrieval comparison's 36 non-abstention tasks
are the 35 answer tasks plus the partial task. Its evidence-recall denominator excludes
the five abstention tasks because they have no expected evidence. End-to-end evaluation
keeps all 41, including failures. The 119 original CUAD retrieval queries and 24
provisional V2 cases are separate suites, not extra members of `hand-v1`.

**Follow-up:** Name the denominator for each metric. Exclude unanswerable questions
from evidence recall, but do not delete failed executions from end-to-end outcomes.

**Evidence:** [methodology](eval-methodology.md), [standalone protocol](evaluation_protocol.md),
[hand-v1 tasks](../evals/datasets/hand/tasks.jsonl).

### 24. Why use an LLM judge if humans review all 41 answers?

**Suggested answer:** “The first small human-reviewed run provides quality evidence
and a way to evaluate the evaluator. A calibrated judge can later help triage larger
regression runs. The checked-in split fixes twenty cases for calibration and twenty-one for a
frozen judge audit before inspecting outputs. Humans label answers without seeing
judge verdicts. Final answer-quality numbers use human labels; judge agreement and
false passes are separate results.”

**Follow-up:** What if the judge agrees with humans overall but accepts a dangerous
exception omission? Review error categories and false passes, not agreement alone.

**Status:** The split, sealed-verdict files, label freezing and agreement tooling are
implemented. Live human labels and a completed calibration result are still pending.
The sealing is procedural, not encryption: manually opening the file violates the
protocol. The frozen live judge pass also requires author review of the provisional
answer-rubric prompt.

### 25. What does the judge see, and how do you avoid circular evaluation?

**Suggested answer:** “For claim support, it sees a claim and only that claim's
cited excerpts. For answer completeness, it also needs reviewed reference evidence
and a task rubric; otherwise it cannot detect facts the retriever missed. Offline
judging is separate from runtime verification. The answer-rubric judge excludes the
verifier's decision, internal reasoning and system/version labels. Sharing a model
family still creates correlated-error risk.”

**Follow-up:** Existing sentence extraction can merge multiple propositions; discuss
how that limits sentence-level scores. A reference answer should allow equivalent wording.

**Evidence:** [claim_support@2 methodology](eval-methodology.md),
[answer-rubric judge](../evals/graders/answer_rubric.py).

### 26. Is your benchmark held out, representative, and statistically convincing?

**Suggested answer:** “The 41-case suite was used in development, so it is not held
out. The 24 generated V2 cases remain provisional. The 119-question CUAD retrieval
validation uses development-disjoint documents, but measures a narrower source-
annotated clause-extraction task. I report those scopes separately. Small samples
and single trials do not establish broad production generalization.”

**Follow-up:** A train/test split for judge calibration does not retroactively make
the underlying QA tasks unseen by the system developer.

**Evidence:** [dataset provenance](eval-methodology.md), [retrieval validation](retrieval-improvement-results.md).

### 27. Tell me about an experiment that did not work.

**Suggested answer:** “I identified a real mismatch between chunk length and the
embedder's input limit and tested windowed embeddings. It did not improve the
development benchmark: the reported evidence recall was 58.3%, versus 61.1% for the
previous hybrid-plus-rerank baseline. I kept the experiment as evidence rather than
assuming a technically reasonable fix must improve retrieval. I would investigate
per-case candidate and reranker behavior before adopting it.”

**Follow-up:** Explain what the experiment establishes and what remains only a
hypothesis. Do not present this development result as universal evidence against windows.

**Evidence:** [ablation report](retrieval-improvement-results.md).

### 28. How would you debug a wrong answer systematically?

**Suggested answer:** “I trace the first failing boundary: was the source extracted,
was the evidence present in candidate retrieval, did it survive reranking, was it
admitted into context, did the planner retrieve needed definitions, and did generation
interpret the evidence correctly? Then I inspect the gate and any repair. I change
the component responsible for the failure and rerun both the case and the relevant
regression subset.”

**Follow-up:** For missing exceptions, distinguish missing retrieval, admission loss,
and interpretation error. Increasing top-k is not the answer to all three.

### 29. Do you need to prove V2 beats V1?

**Suggested answer:** “Only if I claim comparative improvement. I can demonstrate
V2's boundary behavior with tests and its absolute answer quality with a standalone
evaluation. V1 can remain a reproducible historical reference rather than a second
actively developed product. For design questions, targeted ablations can be more
interpretable than comparing versions that changed several things at once.”

**Follow-up:** Never compare V1 stub results with V2 live outputs as a quality gain.

**Evidence:** [updated project plan](V2_EXTENSION_AND_EVALUATION_PLAN.md).

## F. MCP, observability, and engineering scope

### 30. What does MCP add beyond ordinary tool calling?

**Suggested answer:** “The local MCP server makes existing contract tools callable
from an external client through a standard interface. It does not make retrieval or
reasoning more accurate. It uses local stdio, explicit startup document grants,
optional content-hash snapshot pins, and one process per session. A shared executor
enforces tool limits, scope, provenance and evidence admission for the graph and MCP.
The adapter calls no model itself.”

**Follow-up:** Which guarantees disappear? The external model can skip verification
or ignore its result; the server cannot control that model's total spend or final prose.

**Evidence:** [MCP adapter](../app/mcp_server.py), [shared executor](../app/agent/tool_session.py),
[stdio integration tests](../tests/unit/test_mcp_server.py). Claude Code registration is
documented; the scripted SDK client is the automated external-client demonstration.

### 31. What is the riskiest part of adding MCP?

**Suggested answer:** “Moving policy out of graph nodes without changing their
behavior. I extracted one authoritative ToolSession/executor for reservations,
counters, timeouts and evidence admission, while leaving planning, duplicate-action
detection, repair and final verification in the graph. Graph state synchronizes with
the session before trace events. Characterization tests preserve the existing stop
reasons and trace behavior; the MCP adapter has stdio SDK-client integration tests.”

**Follow-up:** How do concurrent calls reserve the last slot? How are cancellation,
retry spending, session resets, and the graph's verifier reserve handled?

**Details to defend:** Reservations and admission use a lock; no lock is held across
a slow tool await. Dispatched failures, retries and cancellations keep their consumed
slots. The graph reserves one tool call for mandatory final verification; MCP reserves
none because its client may skip verification. A new MCP process starts fresh quotas.

**Evidence:** [executor](../app/agent/tool_session.py),
[characterization tests](../tests/unit/test_v2_characterization.py),
[MCP tests](../tests/unit/test_mcp_server.py).

### 32. Why Langfuse if you already have local traces?

**Suggested answer:** “Local traces provide durable, reproducible records. I implemented
optional OpenTelemetry export over OTLP/HTTP so Langfuse can display agent, model, tool
and verification spans with latency and available token usage. The exporter derives
spans from the finished journal rather than adding a second instrumentation path to
each graph node. Export is disabled by default and a collector failure cannot change
the answer. Langfuse is an interchangeable trace backend; evaluation stays in the
repository.”

**Privacy and timing:** Only allowlisted IDs, locators, counts, statuses, timings,
token usage and version hashes leave the machine. Prompts, questions, answers,
contract text, tool arguments and exception strings do not. A private tracer provider
avoids exporting unrelated library spans. The trace appears after a terminal event,
while the local SSE inspector shows progress during execution. A marker suppresses
repeat span emission; it does not prove that the remote collector received the trace.

**Follow-up:** A trace groups one run; a span measures an operation, and an event marks
an instant such as evidence admission. Use the local run ID to find `cra.run_id` in the
backend. Tests verify protobuf OTLP export to a local receiver; a Langfuse Cloud view
was manually verified for stub and live V2 UI smoke runs on 2026-10-01. The live run
showed three model calls, 12,918 input tokens and 404 output tokens in about 6.8 seconds;
these are one-run operational measurements, not a quality benchmark. Dataset sync,
score upload and online Langfuse judges are outside this implementation.

**Evidence:** [durable event layer](../app/observability/events.py),
[OpenTelemetry exporter](../app/observability/telemetry.py),
[privacy and loopback export tests](../tests/unit/test_telemetry.py),
[viewing walkthrough](ui-langfuse-walkthrough.md).

### 33. What production capabilities are intentionally missing?

**Suggested answer:** “This is a bounded local research system, not a multi-tenant
production service. It has typed interfaces, tests, CI, local container reproducibility,
and durable traces, but I do not claim authentication, distributed execution, crash
resumption, or production-scale load validation. I am prioritizing measured answer
quality and enforceable tool boundaries because those are the main learning goals.”

**Follow-up:** If asked to productionize it, first establish access requirements,
traffic, data sensitivity, availability, and cost targets before proposing infrastructure.

## Practice exercises that can guide the next iteration

1. **Boundary example:** Find a clause split across chunks/pages. Show original text,
   retrieved chunks, the missing condition, and what `retrieve_clause` actually returns.
2. **Truncation example:** Compare estimated chunk length with actual embedder tokens.
   Locate a relevant sentence beyond the model input limit; inspect candidate ranks.
3. **Ranking example:** Find a gold passage present before reranking but absent at k=5.
   Explain why a larger candidate pool alone is unlikely to help.
4. **Citation counterexample:** Write a structurally valid but incomplete answer and
   explain which check accepts it and which evaluator should reject it.
5. **Budget trace:** Walk through a failed read, retry, writer call, and final verifier.
   Account for every slot and explain which terminal reason applies at exhaustion.
6. **One-page experiment proposal:** Choose one failure, state a hypothesis, change one
   factor, predeclare cases/metrics, and describe what a negative result would teach you.

Turn a practice gap into a code change only when you can name the failing behavior,
its user impact, and a test or measurement that would establish improvement.

## Claims to avoid

| Avoid saying | Defensible wording |
| --- | --- |
| “512 tokens is optimal.” | “512 estimated tokens is the current default; optimality is unproven.” |
| “Overlap fixes truncation.” | “I distinguish chunk boundaries from model-input and context-budget truncation.” |
| “My answer accuracy is 72.7%.” | “That figure is development evidence-location recall@5.” |
| “Citations eliminate hallucinations.” | “Structural validity and semantic support are separate.” |
| “V2 is proven better.” | “V2 has stronger explicit controls; live quality is a separate measurement.” |
| “My agent resumes after crashes.” | “It records interrupted runs as failures; it does not resume execution.” |
| “The allowlist authenticates users.” | “It constrains a request or configured tool scope.” |
| “MCP enforces the external agent's final answer.” | “The server enforces tool scope and budgets; the client can skip citation verification.” |
| “Langfuse export guarantees cloud delivery.” | “Metadata-only OTLP export is locally tested and cloud viewing was manually verified for smoke runs; future delivery is not guaranteed.” |
| “The V2 judge proves answer quality.” | “The live quality run is pending; final quality metrics require frozen human labels.” |

## Background reading

These sources inform architecture discussion; they do not establish facts about this
repository or guarantee that an interviewer will ask a particular question.

- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents):
  useful vocabulary for discussing fixed workflows versus dynamic tool selection.
- [Contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval):
  an alternative way to restore chunk context. This project does not thereby implement
  generated contextual embeddings, and generated context must not become source evidence.
- [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents):
  useful questions about tool interfaces, bounded responses, and evaluation.
