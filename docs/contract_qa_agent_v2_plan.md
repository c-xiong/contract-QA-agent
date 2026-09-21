# Contract QA Agent V2: Bounded Tool-Using Agent

**Purpose.** Turn the existing LangGraph Contract QA service from a fixed retrieval workflow into a bounded, observable, tool-using agent. The output of this work is a running system, one honest evaluation, and inspectable traces. It is not a paper, not a product, and not a multi-agent framework.

**What this version is trying to prove.** That the system can select actions based on what it actually retrieved, that every action is authorized and accounted for by code rather than by the model, and that every step of a run can be reconstructed afterwards.

## 1. Preserved from V1

Do not rewrite these. Re-run their checks before claiming any V2 result.

- LangGraph orchestration with code-enforced search, evidence, and repair budgets.
- Hybrid BM25 + MiniLM/FAISS retrieval with reciprocal rank fusion and cross-encoder reranking.
- Non-LLM citation verification against document, page, and section identity, request scope, and retrieved evidence.
- Bounded repair followed by abstention when verification still fails.
- FastAPI SSE progress and evidence streaming.
- The 41-question evaluation suite, versioned graders, code/data hashes, and the existing CI (offline tests plus ingestion-to-evaluation smoke tests).

The historical numbers (56 documents, evidence-span recall@5 from 45.8% to 61.1% on 36 document-scoped questions) are V1 results to reproduce, not V2 measurements. If the corpus changes (section 2), those numbers must be recomputed and the old and new denominators reported separately.

## 2. Corpus decision (do this first)

The single biggest credibility limit of V1 is that it runs on 56 documents with hand-written questions. Settle this before building tools, because it determines the index, the evaluation set, and every number that ends up on a CV.

Two acceptable outcomes:

**Option A, expand the corpus.** Ingest a public contract collection such as CUAD (510 commercial contracts with human-annotated clause spans) or contracts pulled from EDGAR exhibits. CUAD's annotations are clause-extraction labels rather than QA pairs, so they need conversion, but they give two things that are expensive to produce by hand: a real order of magnitude, and gold evidence spans that someone else can verify. Recompute the retrieval baseline on the new corpus and report it alongside the old one.

**Option B, keep 56 documents and say so.** Acceptable if conversion cost is too high. In that case, stop describing the corpus in terms of document count anywhere, and describe the evaluation in terms of question coverage and failure modes instead.

What is not acceptable is expanding the corpus halfway, leaving the evaluation set anchored to the old documents, and reporting a mixed number.

## 3. Architecture

Keep the existing graph. Add typed tools and bounded conditional routing.

```text
Request + document scope
          |
    Initialize state, budgets, deadline
          |
    Decide next action <-----------------------+
          |                                    |
    Validate action, check scope, reserve budget
          |                                    |
    Execute typed tool -> record result -> update evidence
          |                                    |
          +--- missing evidence / recoverable error
          |
    Draft supported claims
          |
    Mandatory non-LLM citation verifier
          |
    Pass      -> final answer
    Fail      -> bounded repair -> verify again
    Exhausted -> abstain with a typed reason
```

"Decide next action" selects a tool, redrafts, or stops. It must react to observed tool results rather than run a fixed sequence for every question. Action reasons are short structured strings, not hidden chain-of-thought.

**The authorization boundary is the core of this version.** The model proposes actions. Code owns document authorization, argument validation, counters, deadlines, graph transitions, and terminal status. The model cannot widen its document scope, raise a budget, or emit a final answer that skipped the verifier. Implement this as an explicit check between "decide" and "execute", not as prompt instructions.

## 4. Tools

Typed request and response models (Pydantic) behind a small registry. All tools are read-only with respect to source contracts. Valid document and version IDs come from the request's allowed scope and repository metadata, never from a model-produced path.

Three new tools:

| Tool | Contract |
|---|---|
| `retrieve_clause(doc_id, version_id, clause_ref)` | Exact or indexed clause lookup with bounded adjacent context. Returns original text, source coordinates, outbound references, and explicit `not_found` / `ambiguous` statuses. |
| `extract_definition(term, doc_id, version_id)` | Returns definition candidates from that version as cited source text plus candidate IDs. A model paraphrase is never evidence. |
| `calculate_date(anchor_date, offset, unit, direction, convention)` | Deterministic date arithmetic. Requires an explicit calendar convention. Supports calendar days; business-day and holiday conventions return a typed limitation rather than a guess. The citations for the governing rule live in the calling state, not in the tool output. |

Two wrappers over existing components, exposed through the same registry so routing is uniform:

| Tool | Contract |
|---|---|
| `document_search(query, doc_ids, top_k)` | Wraps the existing BM25 + FAISS -> RRF -> cross-encoder pipeline. Returns bounded hits with scores, chunk IDs, versions, and source spans. `top_k` is clamped in code. |
| `verify_citation(claims, citations, evidence_ids)` | Wraps the existing non-LLM verifier. Returns per-citation checks, missing references, and scope or span mismatches. Also runs automatically as the mandatory final gate. |

Version comparison is deliberately **not** a tool. Retrieve both sides with `retrieve_clause` and let the answer stage do the comparison over cited evidence. A dedicated deterministic diff is not worth its implementation and annotation cost at this scale; revisit only if version questions show up as a distinct failure cluster.

Common response envelope:

```text
ToolResult = {
  call_id, status: ok | empty | error,
  data, evidence_refs[],
  error: {code, retryable, safe_message} | null,
  elapsed_ms, cache_hit, result_hash
}

EvidenceRef = {
  evidence_id, doc_id, version_id, chunk_id,
  page, section, start_offset, end_offset, text_hash
}
```

A computed date is derived evidence: its provenance is the cited rule plus the supplied inputs, never a fabricated source quote. Rename any of these to match existing repository conventions. Do not introduce a second retrieval framework.

## 5. State and budget invariants

Extend the current typed state rather than replacing it.

```text
RunState = {
  run_id, question, allowed_doc_ids, requested_versions,
  config_hash, model_revision, prompt_version, corpus_hash,
  next_action, current_goal,
  evidence_by_id: {evidence_id: EvidenceRef + text_or_private_ref},
  resolved_definitions, calculations,
  evidence_gaps[], draft_claims[], citations[],
  verification_results[], tool_results_by_call_id,
  counters: {searches, evidence_items, tool_calls, model_calls,
             repairs, retries, input_tokens, output_tokens},
  limits: {...}, deadline_at,
  errors[], seen_action_fingerprints[],
  status: running | answered | abstained | failed | cancelled,
  stop_reason, last_event_sequence
}
```

Counter semantics have to be written down once and then obeyed: searches count retrieval attempts including those issued inside a tool; evidence counts unique admitted spans; repairs count draft revision cycles after a verification failure; retries count as attempts; tokens from every planner, tool-calling, and repair model call count toward the run total.

Keep the existing search, evidence, and repair limits unchanged so the V1 and V2 comparison is not confounded. Add ceilings for the new degrees of freedom: a total tool-call cap, a total model-call cap, at most one retry per failed invocation, and a wall-clock deadline. Tune them once against the evaluation set, then freeze and record them in the run manifest.

Invariants:

- Evidence keeps immutable document, version, and span identity, and passes a scope check before entering state.
- Tool arguments and structured model outputs are schema-validated. Invalid arguments trigger one bounded correction, then stop.
- Duplicate action fingerprints with no new evidence are detected and stop the loop.
- A final verification slot is reserved. Budget exhaustion must never produce an unverified answer presented as complete. Stop before dispatch when the remaining budget cannot cover the action plus verification.
- Traces persist independently of SSE delivery. A dropped client connection must not silently start a second run.

No distributed state store. If the project already has checkpointing, reuse it; otherwise an interrupted process is recorded as failed and restarted explicitly. Do not imply crash-resumable execution just because tracing exists.

## 6. Tracing

An append-only JSONL event stream written locally, with a safe subset surfaced through the existing SSE endpoint. This is the highest-value part of V2 relative to its cost: it is what makes a run explainable to someone who did not write it.

| Field group | Contents |
|---|---|
| Identity | Run, event, and span IDs, parent span, monotonic sequence, wall timestamp, graph node |
| Reproducibility | Git revision, config, prompt, model, tool, and grader versions, corpus and index hashes |
| Action | Event type, selected tool, validated arguments or a redacted reference, short action reason |
| Evidence | Query, bounded hits with scores, document/version/chunk/span IDs, selected evidence, result hash |
| Resources | Start and end times, duration, token counts, call counts, remaining budgets, cache status |
| Outcome | Result summary, verification status, error code, retry and repair linkage, terminal reason |

Event types: `run_started`, `tool_started`, `tool_completed`, `tool_failed`, `evidence_added`, `draft_created`, `verification_completed`, `repair_started`, and exactly one terminal event.

Contract text stays in protected local artifacts; client-visible traces carry IDs and redacted summaries. Never log secrets or private chain-of-thought. Use a monotonic clock for durations and mark unavailable token counts as unknown rather than zero.

Add a compact per-run summary: call count, evidence sources, repairs, final verification status, latency, tokens. **Save three runs as checked-in fixtures: one clean success, one failure recovered within budget, one justified abstention.** These are the demo.

## 7. Failure handling

| Failure | Required response |
|---|---|
| Empty retrieval | Reformulate once if the budget permits, otherwise abstain for insufficient evidence |
| Missing or ambiguous clause / version | Retrieve clarifying evidence; never silently substitute another version |
| Invalid tool arguments | Reject before execution, allow one bounded structured correction |
| Transient timeout | Retry only retryable reads, within the remaining deadline, logging each attempt |
| Citation mismatch | Existing bounded repair, then reverify; abstain if still unresolved |
| Missing computation inputs | Require an explicit date and convention plus grounded clause inputs; return unsupported or ambiguous when absent |
| Repeated unproductive calls | Detect duplicate actions with no new evidence and stop |
| Prompt injection in documents | Treat retrieved text as evidence only; scope and tool policy are enforced outside the model |
| Budget or deadline exhaustion | Typed terminal reason, no unverified claims presented as an answer |
| Cancellation or provider outage | Clean up pending work and emit cancelled or failed, not an evidence-based abstention |

**Stated limit of the verifier.** Valid citations establish identity, location, and evidence membership. They do not prove that a claim follows from the text. Semantic support is measured separately (section 8) and the verifier is never described as making the system hallucination-proof.

## 8. Evaluation

One frozen evaluation set. No development/holdout split, no bootstrap intervals, no repeated-run variance analysis. At this sample size those devices cost more than the information they return, and a single well-annotated set with an honest failure table is more defensible than a thin statistical apparatus.

**Set.** Keep the existing 41 questions as a frozen regression suite (it already covers unanswerable, conflicting-version, and prompt-injection cases). Add roughly 20 to 25 new questions targeting the new capabilities: definition chains, version comparison, notice-deadline calculation, and at least a few simple lookups so that orchestration overhead on easy questions is visible rather than hidden. Freeze the rubric before running V2.

**Per-case annotation.** Question ID, allowed documents and versions, answerability, expected factual claims, gold evidence spans or acceptable alternatives, required capabilities, expected abstention or error behavior, and the grading rubric version. Annotate an acceptable partial order of capabilities, not one exact tool sequence: several correct trajectories exist.

**Protocol.** Run V1 and V2 on identical cases with the same model, corpus, index, retrieval settings, and budgets. Report deterministic offline checks separately from live-model runs. Fixed seeds where supported. Keep every run, including failures. Where a question requires a capability V1 does not have, record it as a V1 failure rather than excluding it, and say so.

**Metrics.** Report counts alongside percentages throughout, and give the denominator for each.

| Metric | Definition |
|---|---|
| Task success | Cases satisfying the frozen rubric, including required computations and support. Timeouts and errors are failures. |
| Citation validity | Structural verifier pass rate. |
| Semantic citation support | Manually judged: cited evidence actually supports the claim. Reported separately from validity. |
| Evidence recall | Gold spans recovered over gold spans, answerable cases only, using the existing span-matching rule. Preserve recall@5 and add trajectory-wide recall at the fixed evidence budget. |
| Tool-selection quality | Required capability coverage, plus useful calls over executed calls under the annotated allowed paths. Judge arguments and document scope, not just tool names. |
| Unnecessary calls | Duplicate or no-progress calls per run, separating justified retries from waste. |
| Abstention accuracy | Correct answer-versus-abstain decisions, with false abstention on answerable cases reported separately. |
| Cost | Median end-to-end latency, tool calls, model calls, and tokens per task. Include failed runs and disclose caching. |

Answerable and unanswerable subsets are reported separately. Infrastructure failures count as failures, not as correct abstentions.

**Failure taxonomy.** One primary cause per failed case, so percentages do not double-count: retrieval miss, wrong tool or arguments, missing definition, version or precedence confusion, incomplete evidence, calculation error, unsupported claim, invalid citation, unnecessary loop, failed repair, wrong abstention, scope or injection violation, infrastructure failure.

Deterministic graders cover identifiers, source spans, date arithmetic, counters, schemas, and terminal behavior. Semantic support, task success, and tool usefulness are graded manually on this set. If an LLM grader is used at all, it is versioned, calibrated against the manual labels, and all disagreements are audited. Audit every apparent V2 win by hand before reporting it.

## 9. Explicitly out of scope

Listing these matters as much as the plan itself, because each of them is a plausible-sounding way to spend weeks without improving the system.

- Multi-agent architectures: planner/executor splits, LLM verifier chains, parallel evidence workers. Record as future work and do not leave a "if time permits" opening.
- Held-out sets, bootstrap confidence intervals, repeated-run variance, p95 latency, currency cost estimates. The sample size does not support them.
- A dedicated version-diff tool (section 4).
- New frontend, general legal assistant, open-web browsing, autonomous document edits, long-term memory, distributed workers, vendor tracing deployments.
- Restructuring working V1 modules to match the file layout in section 11.

## 10. Definition of done

- [ ] V1 retrieval, verification, budgets, abstention, SSE, graders, hashes, and CI still pass.
- [ ] The three new tools have typed contracts and tests covering out-of-scope documents, missing data, wrong versions, and computation edge cases.
- [ ] At least three questions demonstrably take different tool paths driven by intermediate results, and the traces show it.
- [ ] Every run is reconstructable from its event log; counters stay bounded across retry and repair paths; no final answer bypasses the verifier under budget exhaustion or adversarial inputs.
- [ ] V1 and V2 are reported on identical cases, with the failure taxonomy, separate citation validity and semantic support numbers, and explicit denominators. Original 41-question results are reported separately from the new questions.
- [ ] README, three checked-in traces, and a short report back every claim that will appear on a CV.

Keeping the single-agent design after measuring it is a successful outcome. Higher task success is not a prerequisite for a finished, honest experiment.

## 11. Repository artifacts

Adapt paths to the existing project.

```text
docs/v2_design.md
docs/tool_contracts.md
docs/evaluation_protocol.md
app/agent/{state,graph,budgets,authorization}.py
app/tools/{registry,search,clauses,definitions,dates,citations}.py
app/observability/{events,writer}.py
eval/datasets/{existing_41,v2_tasks}.*
eval/graders/{answer,trajectory,citation,evidence,cost}.*
eval/manifests/              # case, grader, code, corpus, config hashes
tests/                       # tools, routing, scope, budgets, failures, SSE
runs/<run_id>/{manifest,events,predictions,metrics}.*
reports/{v2_results,failure_analysis}.md
examples/traces/{success,repaired,abstained}.jsonl
.github/workflows/           # extend existing offline CI and smoke checks
README.md
```

## 12. Interview talking points

- Walk through one saved trace and explain why each tool was called given what the previous call returned.
- Show the recovered-failure trace: what failed, what the budget allowed, why the run still terminated correctly.
- Explain the authorization boundary: what the model proposes versus what code decides, and what specifically prevents a model-proposed action from reading a document outside the request scope or skipping verification.
- Distinguish citation validity from semantic support, with the measured gap between the two as evidence that the distinction is real.
- Explain where dynamic routing did not help. Simple lookups that gained nothing but cost an extra model call are a better answer than a uniform improvement claim.
- Explain why multi-agent variants were left out, in cost terms.
