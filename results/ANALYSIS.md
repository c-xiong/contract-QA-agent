# Closeout analysis

2026-09-16; `hand-v1`, 56 documents / 6,270 chunks, one trial. This is a development
benchmark. [Raw artifacts](README.md) and [generated tables](RESULTS.md) are the evidence.

## What to keep

A reproduces evidence-span recall@5 of 45.8% with BM25 and 61.1% with RRF + reranking
on 36 document-scoped questions: +15.3 percentage points, 8 wins / 2 losses / 26 ties.
Both arms achieve only 25% on the eight cross-document questions. This supports the CV's
retrieval claim, not a claim of general contract-answering reliability.

B separates loop and cross-reference expansion in a 2×2 comparison. Required-point
coverage is 0.602, 0.628, 0.632 and 0.614 (single/no-xref, single/xref, loop/no-xref,
loop/xref). Main effects are +0.008 for loop and +0.004 for xref, with interaction -0.045.
Retrieval recall does not change. These small, stochastic differences do not justify a
headline quality claim for additional orchestration. Existing bounded controls remain
available, with their limits made explicit.

C independently audits the raw drafts and delivered answers, including malformed
citation attempts in the denominator:

| Arm | Raw invalid / attempts | Delivered valid / attempts | Completed / 41 | Repair success / triggered |
|---|---:|---:|---:|---:|
| Ungated | 23/79 | 56/59 | 33 | 0/0 |
| Verify only | 30/86 | 53/53 | 30 | 0/0 |
| Verify + repair | 19/81 | 63/63 | 33 | 1/9 |

Raw drafts differ between arms; this is not a shared-draft replay. Raw counts include
drafts from tasks eventually abstained, while delivered counts exclude abstentions.
The ungated arm still applies the writer's ordinary abstention policy. The audit's
`unsafe_answer_exposure` also flags uncited extracted sentences, so it must not be called
a semantic hallucination rate. Verification checks locations, allowlists and evidence
membership; it cannot prove claim support. One successful repair supports retaining a
bounded retry, not a broad claim that repair is reliable.

## Answerability adoption decision

The predeclared rule required fewer false answers, at most 10% over-abstention, no more
than 0.05 required-point decline, and no claim-support decline. D fails that rule.
Keep the new gate **disabled by default**, available with `CRA_ANSWERABILITY_GATE=1`.

| D measure | Off | On |
|---|---:|---:|
| Expected-abstention tasks refused | 5/5 | 5/5 |
| Explicit over-abstention on answerable tasks | 4/36 | 11/36 |
| Execution failures | 0 | 4 |
| Answerable tasks with completed status | 32/36 | 21/36 |
| Answerable required-point coverage | 0.560 | 0.519 |
| Supported / evaluated claims among delivered answerable responses | 56/92 | 38/56 |
| Conditional claim-support micro | 60.9% | 67.9% |
| Application tokens | 153,062 | 264,936 |
| Application model calls | 52 | 69 |
| Mean application elapsed seconds per task | 5.39 | 8.59 |
| Estimated application API cost | $0.409 | $0.749 |

These are runtime status counts, not a semantic classifier of baseline prose. A completed
response can contain a refusal or be incomplete. The conditional support increase covers
fewer claims and excludes abstentions/failures; it is not an overall quality improvement.
All four failures are quote-validation errors, not correct refusals.

The independent D judge used 101 calls, 91,062 input and 21,322 output tokens, with zero
judge execution errors: approximately $0.395 extra. B cost approximately $1.417 and C
$1.093 in total application tokens. Estimates use $2/M input and $10/M output, exclude
provider adjustments, and are not billing records. Latency is summed per-task application
time divided by task count; judge time is separate and concurrent runs are not a latency
benchmark. See [pricing source and definitions](../docs/eval-methodology.md#reproduction-and-cost).

## Three inspected failures / disagreements

### 1. `crossdoc-003`: missing evidence and misleading completion status

The baseline has `completed` status but explicitly says the excerpts do not support the
requested liability comparison. Its evidence includes mitigation at doc-001 p.74 §14.4
and indemnification at doc-003 p.6 §8.5, rather than the requested liability-loss clauses.
Evidence-span recall and required-point coverage are both zero; claim support is 0.333.
The enabled gate abstains because evidence is insufficient. The task expects an answer,
so that remains an over-abstention in the end-to-end benchmark. This case shows a retrieval
failure and the boundary of status-based answerability counts, not a successful baseline
answer or proof the gate improves the full system.

### 2. `lookup-010`: strict quotation validation introduces a failure

The baseline retrieves the annotated clause (span recall 1.0), completes and covers 2/3
required points. The enabled gate fails with `output_validation: quote not found in e01`.
The decision is rejected rather than silently accepted or counted as a refusal. This
protects provenance but adds a failure mode and call cost even when retrieval succeeds.
The recorded error does not establish whether the rejected verdict was semantically right.

### 3. `unsupported-001`: citation syntax blocks a supported response

Both arms abstain after one repair. The enabled gate classifies the NDA question as
answerable: confidentiality can apply without an explicit confidential marking. The
writer and repair still produce malformed citations to source-text span locations, which
the verifier rejects. Answerability does not fix this citation-format bottleneck. The
case explains why structural output failures and semantic support need separate metrics.

## Stopping point

The optional gate, controlled experiments, provenance and offline tests are implemented.
The measured tradeoffs and known weaknesses are published here, and the CV uses the
reproduced retrieval result. No further layers or paid tuning runs are part of closeout.
