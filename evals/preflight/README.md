# Budget preflights for paid V1/V2 evaluation

A live run of `experiments/v2_standalone.py` or `experiments/v2_judge.py` refuses to start
without a preflight. Copy a template, drop `.template` from the name, and fill every
`null`:

- `input_price_per_million_tokens`, `output_price_per_million_tokens`: the selected
  model's **current** prices from the provider's pricing page. Nothing is built in.
- `max_total_input_tokens`, `max_total_output_tokens`, `max_cost`: the most you are
  willing to spend. The currency cap is usually the binding one.

`max_attempts` comes from the protocol (492 application, 648 judge with `C = 7`). The V1
arm of the paired comparison has its own templates: `v1-hand-application` (492) and
`v1-hand-judge` (328: one frozen pass, no calibration rounds). Use the same model and
prices in both arms, or the comparison refuses to pair them.
`max_output_tokens_per_call` must be at least 2048, the live client's output ceiling.
Before each provider attempt the ledger reserves `ceil(prompt_chars / 2)` input tokens
plus the declared output ceiling at the declared prices and refuses the call if the
attempt limit or reservation checks would exceed a cap. The input count is a character
estimate, not a tokenizer bound: actual spend on an in-flight attempt can exceed the
reservation and the token or currency caps. Known usage settles at its reported count;
any reservation overrun exhausts the ledger and stops subsequent attempts, including
after resume. `settled_cap_breaches` lists any reported per-call token or cumulative
token/currency breaches. Unknown usage retains the estimated reservation in settled
accounting, rather than counting as zero.

The ledger file sits next to the report (`*.ledger.json`,
`*.judge-ledger.json`) and cannot be reset: a changed preflight requires a new
`experiment_id` and output path. Filled preflights are hashed into the ledger; commit
them with the report they govern.

The filled `v1-`/`v2-hand-*.json` files are for the Claude Code CLI backend: notional
list prices (subscription-billed), a 32k per-call output ceiling because the CLI cannot
cap adaptive thinking, and `per_call_input_overhead_tokens: 1000` for the CLI's fixed
per-call preamble. A run on the API backend needs its own preflight and experiment id.
