# Design decisions and interview notes

Updated 2026-09-16. Consolidates the historical decision log and interview questions.

## Positioning and authorship

This is a bounded retrieval workflow with explicit tools/state and measured evidence
quality. It is not an autonomous LLM planner or persistent conversation agent. Development
used coding-agent assistance. The existing hand-v1 questions and expected answers were
authored/reviewed by the project author; generated supplements remain labelled separately.

## Decisions to explain

1. **Stable manifest IDs.** Titles can repeat across contract versions. Deterministic
   filename normalization and explicit exceptions preserve annotation-to-document identity.
2. **Section-aware, page-preserving chunks.** Keep clauses and carve-outs together while
   retaining citation locators. Paragraph/line-wrap and decimal article bugs were fixed
   before the current corpus. Some context still spans page boundaries.
3. **BM25 plus local MiniLM/FAISS.** Exact legal terms and semantic similarity complement
   one another. RRF fuses ranks because raw scores are incompatible; its constant is 60.
4. **Bounded cross-encoder reranking.** It can reorder candidates, not recover a missing
   document. Cross-document failures may need candidate allocation rather than a larger model.
5. **Deterministic search policy.** Term coverage, new evidence and budgets determine
   additional searches. This transparent policy is not LLM planning or answerability.
6. **Depth-one explicit references.** A cap can be qualified elsewhere. Follow references
   inside the same document under a chunk budget; defer deeper traversal without evidence.
7. **Whole evidence items.** Deduplicate by provenance, preserve conflicting versions and
   drop whole items instead of truncating quotations without disclosure.
8. **Citation structure versus support.** A real location can still be wrong evidence.
   The deterministic gate checks provenance; claim_support@2 sees only each claim's own
   citations. Sentence-level extraction has a stable but coarse denominator.
9. **Independent citation audit.** A disabled verifier cannot grade itself. Separate raw
   output, verification and repair. Withdraw legacy 8.6%: malformed attempts appeared in
   its error count but not its denominator. New counting includes them once.
10. **Failures are not refusals.** Timeouts, invalid JSON and fabricated evidence IDs remain
    classified failures, so a broken system cannot score as correct abstention.

## Closeout decisions recorded before live evaluation

- Add one pre-write answerability node with an independent, versioned runtime prompt.
  Validate evidence IDs and verbatim quotations in Python. Partial answers name missing
  aspects. A matching quotation proves provenance, not semantic truth. Retrieval scores
  are not calibrated answerability probabilities.
- Keep runtime gate and offline judge independent. Follow the adoption rule in the
  closeout plan; retain the gate as opt-in if the measured tradeoff fails that rule.
- Hash sorted canonical chunk content rather than IDs alone. Text/boundaries can change
  without changing IDs. Record code SHA and dirty state; never invent historical provenance.
- B toggles loop and xref independently; C separates verification and repair. Both disable
  answerability. D measures that gate alone.
- Default tests use fake embedders/rerankers, temporary API corpus settings and network
  blocking. Explicit model integration tests exercise the real dependencies separately.
- Keep only this guide, evaluation methodology and the closeout checklist. They replace
  the long specification, duplicate interview questions and obsolete authoring tutorial.
- Stop after validation and CV updates. No additional memory layer, multi-agent system,
  hosted demo, vector database or product UI is needed for this portfolio scope.

## Observed closeout decision

Experiment D fails the predeclared adoption rule: zero false answers in both arms,
over-abstention 4/36 versus 11/36, and four validation failures with the new gate.
Application tokens rise from 153,062 to 264,936. Keep `CRA_ANSWERABILITY_GATE=0` as the
default; retain the implementation and tests for explicit experiments. Higher support
among fewer delivered claims does not offset the coverage loss. B also finds no clear
answer-quality advantage from more orchestration in one trial. Do not add more layers.

All four current runs, limitations and inspected failures are linked from
[the result index](../results/README.md). The CV uses only the reproduced retrieval gain
and implemented engineering features. Closeout is complete; further benchmark expansion
is outside the agreed scope.

## Five-minute walkthrough

Use the local trace inspector or `scripts/demo.py --trace`. Explain a clause reference,
the evidence budget, answerability and the citation gate. Open the retrieval comparison
and a failed answerability case. Explain why valid citations can accompany unsupported
claims, and what each controlled experiment actually permits you to claim.

## V2: bounded tool selection and durable runs

The API/CLI default stays V1 (`CRA_AGENT_MODE=v2` opts in), while the local inspector
only displays V2 and passes an explicit per-request mode. V1 remains a backend comparison
baseline; it has no inspector controls or diagram.
The model proposes a typed action; code authorizes scope, reserves budgets, admits original
source chunks and enforces final verification. Five tools share the existing retrieval and
citation components. See [V2 design](v2_design.md) and [tool contracts](tool_contracts.md).

We chose corpus option B before implementation: keep the current index and frozen hand
suite. Generated new cases are explicitly provisional. Document content hashes identify
indexed snapshots because ingestion has no trustworthy legal version labels; we do not
invent version precedence. Character budgets apply even to the first evidence item, and
V2 citations must match a joint admitted locator. This tightens structure, not entailment.

Traces persist before SSE delivery in protected local files. Disconnects cancel with a
terminal record; reconnects cannot silently restart. Dead-process runs are marked failed
at the next startup. This does not provide crash resumption. Provider SDK retries are
disabled to keep attempts accountable. The implementation and provisional fixtures were
created with coding-agent assistance; semantic review is still separate and pending.

The inspector tails the durable journal during execution, rather than waiting for a
whole graph node to return, so model/tool start events remain visible during slow calls.
Only whitelisted tool metadata, evidence locators and budget counters reach progress
events. Mode/arm switches preserve the effective live/stub setting, including degraded
startup. The existing single-file UI uses V2 progress events without new dependencies.
