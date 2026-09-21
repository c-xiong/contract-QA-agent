# Provisional V2 cases

24 generated pipeline-validation cases over existing indexed snapshots: 4 simple lookups,
6 definition chains, 6 snapshot comparisons, 4 calendar-date scenarios and 4 unsupported
or ambiguous requests. The source corpus is unchanged. Offsets/versions in tasks are pinned
to the current store; a mismatched store must be re-annotated, not silently substituted.

Questions, rubrics, required capabilities, expected dates and answerability decisions are
AI-generated and **not author-reviewed**. Source excerpts are copied from indexed synthetic
variants; they are not new expert QA annotations. Some capability paths have alternatives,
and the provisional expectations require human review. Do not use these results for
README capability claims or CV metrics. `hand-v1` remains a separate frozen regression set.

Regenerate only intentionally: `CRA_LIVE_MODEL=0 uv run python -m scripts.build_v2_tasks`.
The checked-in suite loads without downloading the source corpus; running it needs the
existing corpus. Follow [the evaluation protocol](../../../docs/evaluation_protocol.md)
for replacing provisional content and recording a new version.
