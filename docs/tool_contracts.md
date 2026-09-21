# V2 tool contracts

All tools are read-only. Inputs and outputs reject extra fields. The authorization node
checks the request's document and snapshot map before dispatch; returned chunks must
also equal the store's original chunks before evidence admission.

| Tool | Validated input | Result |
|---|---|---|
| document_search | query, optional doc_ids, top_k | Existing configured retrieval arm; top_k clamped to request limit and 10. No scope widening. |
| retrieve_clause | doc_id, version_id, clause_ref | Original indexed section chunks, up to five; not_found, ambiguous, truncation indicators; outbound references and source coordinates in hits. |
| extract_definition | term, doc_id, version_id | Original definition candidate chunks and stable IDs; multiple matches explicitly ambiguous. |
| calculate_date | anchor_date, nonnegative offset, days, before/after, explicit convention | Calendar-day arithmetic. Business-day/holiday requests return unsupported_convention; date overflow is typed. |
| verify_citation | answer, evidence_ids | Existing structural verification plus joint admitted-locator check. Does not prove semantic support. |

`draft` and `abstain` are proposals, not executable tools. Only code can deliver a final
answer. Date actions additionally carry `rule_evidence_ids`; admitted governing evidence
is required. Calculations preserve inputs and those source links separately from source
quotes. Correct selection/interpretation of the rule is evaluated semantically.

The shared result envelope contains call_id, ok/empty/error status, data, evidence_refs,
classified error/retryability, duration, cache_hit (currently false), and a result hash.
Evidence references contain immutable document, snapshot, chunk, nullable page/section,
chunk-relative offsets and text hash. Contract text is never replaced by model paraphrase.

Lookup is deliberately bounded. Clause lookup uses existing section IDs; repeated labels
on disjoint pages are ambiguous, and truncation does not imply the whole clause was read.
Definition matching is a conservative lexical candidate finder, not a complete legal
parser. The planner must resolve ambiguity or abstain. No hidden retrieval calls are made
inside clause/definition lookup; all search attempts go through document_search.
