# Super synthesis benchmark protocol

Use this protocol to select server batch limits and client concurrency for a
self-hosted synthesis teacher. Results are deployment-specific: do not copy a
setting from another model revision, engine build, hardware profile, prompt
contract, or output allowance.

Raw manifests, prompts, responses, logs, and result ledgers belong under the
gitignored `state/benchmarks/` tree. Tracked documentation must not contain
course text, course slugs, workflow identifiers, machine paths, endpoint
addresses, or hashes of local artifacts.

## Series identity

Record the following in the ignored benchmark manifest. Changing any field
starts a new series:

- model repository, immutable revision, served ID, and quantization;
- inference engine image and revision;
- tensor/expert parallelism and scheduler configuration;
- maximum sequence length, scheduled-token limit, and server batch limit;
- prompt/schema contract fingerprints and reasoning mode;
- output allowance, timeout, retry policy, and client concurrency;
- hardware class and observable memory/cache capacity.

## Workload

Build a fixed, content-sanitized manifest containing representative SFT and DPO
windows from every synthesis substage. Include short, median, p95-sized, and
validator-repair cases. Replay the identical manifest and request order for
each cell. A one-shot transport benchmark measures raw capacity; a separate
production-shaped pass must exercise the real validators and bounded repairs.

Do not publish the captured prompts or responses. Store their hashes and the
manifest itself only in ignored evidence.

## Required metrics

| Metric | Definition |
|---|---|
| Prompt tokens/s | Prompt tokens from every HTTP attempt divided by cell wall time. |
| Completion tokens/s | Completion tokens from every HTTP attempt divided by cell wall time. |
| Total tokens/s | All prompt and completion tokens divided by cell wall time. |
| Accepted-pair tokens/s | Tokens in accepted SFT/DPO pairs divided by cell wall time. |
| Terminal units/s | Source chunks reaching a durable terminal disposition divided by cell wall time. |
| Request latency | Dispatch through complete response; report p50, p95, and p99 when sample size supports it. |
| Queue delay | Enqueue through server admission; report `unavailable` when admission time is not exposed. |
| TTFT | Dispatch through first streamed response token; non-streaming clients report `unavailable`. |
| Context headroom | `max_seq_len - (prompt_tokens + requested_max_tokens)`; report the minimum. |
| Batch-token headroom | Scheduled-token limit minus peak scheduled tokens. |
| KV/Mamba headroom | Peak used versus configured capacity for each state pool; never infer it from a lack of errors. |
| Unified-memory headroom | Minimum available memory from a source that accounts for model allocations. Process RSS is not a substitute. |

Keep these outcomes separate:

- `output_cap`: response ended at its output allowance;
- `context_rejected`: prompt plus allowance exceeded the served window;
- `batch_token_pressure`: scheduler queued or split the request;
- `kv_or_state_exhausted`: cache/state allocation failed or preempted;
- `transport`: timeout, disconnect, reset, abort, or non-success HTTP response;
- `parse_or_schema`: complete response violated the JSON/schema contract;
- `validator_rejection`: structurally complete response failed a quality gate.

## Matrix and stopping rule

Sweep bounded server-batch and client-concurrency candidates from low to high.
Run only one cell at a time and restore the same clean server configuration
between server-batch changes. Stop escalation immediately when a cell has:

- a transport, context, cache/state, fatal, output-cap, or schema failure;
- an engine-hang signal;
- zero scheduler/cache/memory headroom; or
- unsafe p95/p99 or queue-delay growth.

Zero failures and explicitly positive measured headroom are required for a
production candidate. `unavailable` is not zero and cannot qualify a dimension.

## Selection and validation

Among qualifying cells, choose the lowest server batch and client concurrency
on the accepted-pair-throughput plateau. Raw tokens/s alone is insufficient.
Validate the candidate with:

1. real SFT and DPO staged windows;
2. unchanged claim, objective, leakage, and promotion validators;
3. a longer soak capable of revealing rare failures;
4. a post-soak structured generation probe; and
5. stop/resume replay proving no duplicate or lost terminal work.

Record the selected value in deployment-local configuration, not as a universal
project default. `TRAINFORGE_SYNTHESIS_MAX_CONCURRENT` remains `1` when unset.
Re-run the matrix after any model, revision, engine, prompt/schema, token,
server-capacity, or hardware change.
