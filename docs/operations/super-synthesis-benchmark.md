# Synthesis concurrency benchmark

Use this methodology to choose a safe client-concurrency limit for a
self-hosted Trainforge synthesis provider. The result is deployment-specific:
repeat the benchmark whenever the model, inference engine, prompt contract,
output allowance, or hardware changes.

This guide defines a methodology, not a universal performance target. Keep raw
prompts, responses, manifests, endpoint details, logs, and result ledgers under
an ignored operator-local runtime directory. Never publish source content,
course or run identifiers, local paths, host details, or artifact hashes.

## Safety and licensing first

Benchmark only a provider whose outputs are permitted to become training data.
Provider and model posture is governed by
[`docs/LICENSING.md`](../LICENSING.md); benchmark results never override that
policy. The recommended public workflow is a license-clean local provider.

Do not weaken claim, objective, leakage, decontamination, quota, or promotion
checks to improve throughput. A rejected pair is not successful work, even when
the request completed quickly.

## Runtime contract

Trainforge resolves synthesis concurrency in this order:

1. the explicit `--max-concurrent` value;
2. `TRAINFORGE_SYNTHESIS_MAX_CONCURRENT`;
3. the sequential default, `1`.

Missing, blank, non-integer, and non-positive values resolve to `1`. Values
above the validated hard ceiling of `48` fail loudly instead of being clamped.
The `claude_session` provider is restricted to sequential operation
because its dispatch and budget ordering are not concurrency-safe.

At concurrency `1`, Trainforge does not construct a thread pool. At higher
values, generation may run concurrently, but one source-order writer remains
responsible for output JSONL, checkpoints, counters, and deduplication state.
There is no silent serial fallback after a concurrent failure.

The full flag and resume semantics are documented in
[`behavior-flags-trainforge.md`](behavior-flags-trainforge.md).

## Reproducible series

Create an ignored manifest for each benchmark series. Record enough information
to detect an invalid comparison:

- immutable model and inference-engine revisions;
- quantization and server scheduling configuration;
- context, output, timeout, and retry limits;
- synthesis contract and response-schema fingerprints;
- reasoning mode and client concurrency;
- a non-identifying hardware profile;
- the sanitized workload-manifest hash.

Changing any item starts a new series. Keep identifying values and the manifest
itself private.

## Workload design

Use a fixed, sanitized manifest of representative instruction and preference
units. Include varied prompt lengths and the validator-repair paths exercised by
the intended production synthesis contract.

Run two distinct passes:

1. A transport pass measures request capacity with a fixed request order.
2. A production-shaped pass uses the real synthesis path, validators,
   checkpointing, and bounded repairs.

Replay the same manifest and request order for every candidate. Do not compare
cells built from different prompts or acceptance requirements.

For a small preflight of the real synthesis path, use the documented pilot
command in [`full-run-playbook.md`](full-run-playbook.md#7-training-pair-pilot).
The pilot is a quality and integration check; it is not by itself a concurrency
benchmark.

## Measurements

Capture both transport efficiency and useful output:

| Measurement | Interpretation |
|---|---|
| Prompt and completion throughput | Server-reported tokens divided by cell wall time, including every transport attempt. |
| Accepted-pair throughput | Tokens or pairs that survive the production validators divided by cell wall time. |
| Terminal-unit throughput | Source units reaching a durable accepted or rejected disposition divided by cell wall time. |
| Request latency | Dispatch-to-completion latency; report percentiles only when the sample supports them. |
| Time to first token | Dispatch to the first streamed content token; report unavailable for non-streaming calls. |
| Queue delay | Enqueue to server admission when the engine exposes it; otherwise report unavailable. |
| Context headroom | Served context limit minus prompt tokens and requested output allowance. |
| Scheduler and cache headroom | Peak use compared with the corresponding server limits, using direct telemetry. |
| Memory headroom | Minimum available memory from a source that includes model allocations. |

Do not treat missing telemetry as zero use or positive headroom. Client HTTP
attempt records are the authority for request throughput and latency; server
logs describe scheduling behavior but do not automatically reveal cache or
memory capacity.

## Failure classification

Keep these outcomes separate so capacity problems are not mistaken for content
quality problems:

- `output_cap`: generation ended at its output allowance;
- `context_rejected`: prompt plus allowance exceeded the served context;
- `scheduler_pressure`: work queued, split, or preempted under load;
- `cache_or_state_exhausted`: inference state allocation failed;
- `transport`: timeout, disconnect, reset, abort, or non-success response;
- `parse_or_schema`: a complete response violated the response contract;
- `validator_rejection`: a structurally valid response failed a quality check.

Trainforge's production path is fail-loud. Do not recategorize a failed request
as a successful lower-quality result.

## Matrix and stopping rule

Start with the sequential baseline, then test bounded concurrency candidates in
ascending order. Change one capacity dimension at a time, run one cell at a
time, and restore the same clean server configuration between cells.

Stop escalation when a cell shows any of the following:

- transport, context, output-cap, cache/state, schema, or fatal failure;
- an engine hang or incomplete telemetry lifecycle;
- exhausted scheduler, cache, context, or memory headroom;
- unstable latency or queue growth relative to the lower-concurrency cells.

A production candidate requires zero benchmark failures and positive measured
headroom for every required capacity dimension. `unavailable` does not satisfy
that requirement.

## Select and verify

Choose the lowest concurrency on the accepted-pair-throughput plateau, not the
cell with the highest raw token rate. Then verify it with:

1. representative instruction and preference synthesis;
2. unchanged production validators and bounded repairs;
3. a longer soak;
4. a structured-output probe after the soak; and
5. stop/resume replay with no duplicated or lost terminal work.

Store the selected value in ignored deployment configuration. Keep the public
default unchanged. The concurrency flag's checkpoint-identity rules remain in
[`behavior-flags-trainforge.md`](behavior-flags-trainforge.md); use the public
[`pipeline invocation guide`](pipeline-invocation.md#7-graceful-stop-resume-and-checkpoints)
and [`full-run playbook`](full-run-playbook.md#8-stop-and-resume-safely) for
fresh-start decisions, checkpoints, and stop/resume recovery.
