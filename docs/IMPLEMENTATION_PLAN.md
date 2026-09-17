# Project closeout

Updated 2026-09-16. Replaces the superseded sprint plans and specification.

## Scope and checklist

Finish the existing bounded contract QA system. No additional agent layers, hosted
service, vector database or long-term memory are part of this closeout.

| Deliverable | Acceptance | Status |
|---|---|---|
| Documentation | Three maintained documents; accurate README links | Complete |
| Offline tests | Default pytest needs no network, corpus, model cache or API key | Complete |
| Provenance | New reports record code, corpus hash, dataset and grader versions | Complete |
| Answerability | Quote-validated answer/partial/abstain before writing | Complete; opt-in after failed adoption rule |
| Experiments | A retrieval; B loop × xref; C verify/repair; D answerability off/on | Complete |
| Results | Current artifacts, inspected failures, explicit historical limitations | Complete |
| CV | Replace the invalid 8.6% claim; compile and inspect existing variants | Complete |

## Decision rule recorded before live runs

The frozen 41-question `hand-v1` suite is unchanged. It is a development benchmark,
not an unseen generalization test. Additional generated question wrappers remain
`provisional-generated-*`, even when labels are transcribed from expert annotations;
they are supplementary pipeline checks and never headline or resume metrics.

Compare answerability off/on using identical corpus, retrieval, writer, search/evidence
budgets and citation policy. The offline judge must not read runtime verdicts. Retain
the previous adoption rule: fewer false answers, over-abstention at most 10%, required
point coverage declining by no more than 0.05, and no decline in offline claim support.
Report answerable-only support and failures separately so abstention cannot inflate the
score. Record tokens and latency either way. If the gate misses this rule, leave it
opt-in; do not tune against final outputs and call the same run validation.

## Execution and stopping point

1. Verify existing corpus artifacts; preserve frozen tasks.
2. Finish implementation, then run offline checks and stub experiments.
3. Run one bounded current-corpus live comparison per experiment, including independent
   claim support for D. Inspect at least three failures or disagreements.
4. Update README/results and the two CV project bullets. Use the existing LaTeX build.

Explicitly set `CRA_LIVE_MODEL=0` for offline commands: local `.env` may enable live
access. Low cross-document recall, depth-one references, sentence-level claim extraction
and the small development benchmark are documented limitations, not another phase.

See [methodology](eval-methodology.md), [decisions](decisions.md), and
[result artifacts](../results/README.md).

## Final acceptance evidence

- Current A–D artifacts are pinned in `results/`; [analysis](../results/ANALYSIS.md)
  includes three inspected disagreements/failures and the decision to leave the gate off.
- Default suite: **408 passed, 3 deselected**, with blocked external sockets and model
  loading. Separate cached real-model/corpus integration: **3 passed**.
- Ruff lint/format and strict mypy for `app evals scripts experiments` pass. Locked
  dependency sync succeeds. Supplemental generated ContractNLI tasks passed the 20-task
  stub comparison; their results are not used as capability evidence.
- The six existing CV variants compile into `build/` and export to `out/`. Four named
  PDFs pass ATS checks; both anonymous PDFs pass anonymity checks. All six are visually
  inspected single-page PDFs. The project bullets now use the reproduced 45.8% → 61.1%
  evidence-span recall result and remove the invalid citation-rate claim.
- Documentation is reduced to the three maintained files in this folder. Prior copies
  were backed up locally before removal. Changes remain local; no commit or push was made.

The agreed closeout is complete. Known limitations are recorded, not deferred as another
implementation phase.
