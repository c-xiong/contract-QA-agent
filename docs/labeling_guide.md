# Labelling guide: blind V1/V2 review of hand-v1

You label 82 answers: each of the 41 hand-v1 cases has two answers, **A** and **B**, from
the two systems in a random order. Labels are the reported answer-quality outcome; the
LLM judge is only checked against them. The pass standard is the frozen task rubric
(required points and forbidden claims written for hand-v1), not anything in this guide.

```bash
uv run python -m scripts.paired_review label reports/v1v2-hand-live --reviewer "<your name>"
```

The tool shows the question, rubric, reference evidence, the answer, its numbered claims
and the run status, then asks the fields below. It saves after every answer; type `q`
at any prompt to stop and run the same command later to continue. Do not open
`reports/v1v2-hand-live.blind-key.json`, the source reports or any `*.sealed.jsonl`
until labelling is finished and frozen. The Markdown sheet
`reports/v1v2-hand-live.review.md` has the same content if you prefer reading it.

## Fields

| Field | What to record |
|---|---|
| `outcome` | One of: `correct_answer`, `partial_answer`, `incorrect_answer`, `unsupported_answer`, `correct_abstention`, `false_abstention`, `execution_failure`. A failed or interrupted run is always `execution_failure`, never an abstention. |
| `required_points` | For each numbered required point: `present` (stated or plainly entailed), `absent`, or `contradicted`. |
| `forbidden_claims` | For each numbered forbidden claim: `made` or `not_made`. |
| Claim counts | The tool lists the answer's sentence-level claims (`*` = carries a citation), split exactly as the judge splits them. Count how many you inspected, how many the **admitted evidence** supports, and how many it contradicts; the rest are recorded as insufficient. Judge support from the evidence the system admitted, not from your knowledge of the contract. |
| `task_success` | Your overall verdict against the task's rubric. |
| `primary_failure` | Only when the task failed: the single main cause (`retrieval_miss`, `incomplete_evidence`, `unsupported_claim`, `wrong_abstention`, ...). |
| `completeness_errors` | Optional, when the answer failed: omitted exception, omitted condition, wrong party, incorrect conclusion, other. |
| `notes` | One line on why, especially for failures and close calls. These notes are printed next to every V2 win and loss in the comparison report. |

## Practical notes

- Label A and B independently. Do not try to guess which system wrote which answer. If
  you think you can tell (for example from a fixed abstention sentence), carry on; the
  report already discloses that blinding is partial.
- An answer that abstains on a case whose expected behaviour is `answer` is a
  `false_abstention`. One that answers an expected-abstention case with contract facts
  the evidence does not support is not a correct abstention.
- Be consistent across the two answers of a case: the same omission should get the same
  verdict in A and B.

## After labelling

```bash
uv run python -m scripts.paired_review unblind reports/v1v2-hand-live
uv run python -m scripts.v2_labels freeze reports/v1-standalone-hand-live.json
uv run python -m scripts.v2_labels freeze reports/v2-standalone-hand-live.json
```

Corrections after freezing go in `*.adjudications.jsonl` records, never by editing
the frozen labels.
