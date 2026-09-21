# V2 offline failure inspection

These are three inspected deterministic-stub behaviors, not human labels for a live
model. They explain why structural verification cannot stand in for semantic success.
The complete cases and counters are in the paired report artifacts.

| Case | Observed behavior | Primary diagnostic issue | Implication |
|---|---|---|---|
| hand-v1 / unanswerable-001 | V2 returned a cited excerpt; one citation passed; runtime status completed despite expected abstention. | Wrong abstention decision | The offline writer copies evidence; a valid location does not establish answerability. |
| provisional / v2-017 | Governing evidence was retrieved, but no derived date was recorded; capability coverage 1/2 and expected prerequisite order failed. | Missing calculation | Retrieving the right clause cannot replace the requested computation. The scripted date demo tests the tool separately. |
| provisional / v2-021 | The business-day/holiday question produced a cited excerpt and completed instead of abstaining. | Unsupported computation not recognized by stub | Calendar limitation is enforced when the date tool is invoked; natural-language recognition still needs real-model and human evaluation. |

The structural gate does not guarantee entailment, complete task coverage, or correct
choice of a governing rule. Those are deliberately separate review dimensions. For a
live report, assign exactly one reviewed primary cause to every failed case from:
retrieval miss; wrong tool/arguments; missing definition; version/precedence confusion;
incomplete evidence; calculation error; unsupported claim; invalid citation;
unnecessary loop; failed repair; wrong abstention; scope/injection violation;
infrastructure failure. Do not turn outages into correct abstentions.

The new tests separately establish engineering behavior: out-of-scope tool calls never
execute, mismatched snapshots fail, ungrounded date calls are rejected, unsupported
calendars return a typed limitation, repeated no-progress actions stop, invalid citations
repair within budget, and exhausted runs cannot deliver an unverified final answer.
They do not establish that a live planner reliably chooses the right tool.
