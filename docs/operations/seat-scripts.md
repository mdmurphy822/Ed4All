# Local model seats

A **seat** is one long-lived local [vLLM](https://docs.vllm.ai) service that
serves one model. Ed4All addresses seats by logical name and resolves their
endpoints, containers, and launchers from environment registries; workflow code
does not embed deployment addresses or model paths.

## Keep deployment configuration private

Real launch scripts contain deployment-specific paths, model identifiers,
container names, device allocations, and resource limits. Store them under the
gitignored `runtime/seats/` directory. Never commit filled templates, endpoint
addresses, credentials, model caches, or service logs.

Start with the sanitized [`launch-seat.example.sh`](launch-seat.example.sh).
[`launch-super-hj.sh.example`](launch-super-hj.sh.example) demonstrates the same
contract for a separately scheduled heading-judge seat without prescribing a
model or resource profile.

## Registry contract

The lifecycle manager in `lib/vllm_container_lifecycle.py` reads three
registries:

| Environment variable | Value shape | Purpose |
|---|---|---|
| `ED4ALL_SEAT_BASE_URLS` | `<seat>=<loopback-base-url>` | Logical seat to endpoint |
| `ED4ALL_VLLM_CONTAINERS` | `<base-url>=<container>` | Endpoint to container |
| `ED4ALL_SEAT_LAUNCH_SPECS` | `<seat>=<absolute-script-path>` | Logical seat to private cold-recreate launcher |

Use operator-selected values in ignored local environment files. The examples
intentionally provide no working endpoint, path, model, or credential.

When `ED4ALL_SEAT_SCHEDULE` is enabled, each workflow phase reconciles the
declared `seats:` in `config/workflows.yaml`. It stops unneeded seats, launches
required seats, polls `/v1/models`, and runs a bounded content-coherence probe.
A configured launch spec lets the manager replace an unhealthy service with a
fresh container.

## Launcher requirements

Every private launcher must:

1. Validate all required settings before changing container state.
2. Bind the service to a loopback endpoint selected by the operator.
3. Cold-recreate its container; do not rely on a warm `docker start`.
4. Use pre-provisioned model artifacts and offline runtime settings.
5. Set deployment resource limits from private local configuration.
6. Contain no credentials. Pass secrets through an approved runtime secret
   mechanism when a selected provider requires them.

The lifecycle manager owns readiness and coherence checks. A launcher only
needs to validate its inputs and start the requested service cleanly.

## Optional tool calling

A seat used by `ed4all assistant`, or by another OpenAI-compatible caller that
sends `tools` with `tool_choice: "auto"`, must enable the tool-calling support
required by its selected model and vLLM release. Parser selection is
model-specific and belongs in the ignored private launcher; do not copy a
parser from another model without validating its chat-template grammar.

## Related documentation

- [`launch-seat.example.sh`](launch-seat.example.sh) — generic, fail-loud shell
  template.
- [`launch-super-hj.sh.example`](launch-super-hj.sh.example) — separately
  scheduled heading-judge template.
- [`seat-schedule.env.example`](seat-schedule.env.example) — registry shape and
  schedule configuration.
- [`behavior-flags.md`](behavior-flags.md) — lifecycle and schedule behavior.
- [`pipeline-invocation.md`](pipeline-invocation.md) — running, stopping, and
  resuming workflows.
