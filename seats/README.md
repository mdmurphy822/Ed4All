# `seats/` — local vLLM seat launch scripts

A **seat** is one long-lived local [vLLM](https://docs.vllm.ai) container that
serves exactly one model on a fixed loopback port. The Ed4All pipeline never
hardcodes a model or a URL — it addresses seats by a **logical name** and
resolves everything else from data-driven env registries. The scripts in this
directory are the machine-specific launch specs for one host's seat stack.

## Why these scripts are not tracked

Every real seat script carries host-specific detail — absolute HF-cache mount
paths, concrete model ids, container names, and port / GPU-utilization pins
tuned for one machine's GPU. Per the project's data-hygiene contract, that
never lands in git. `.gitignore` ignores `seats/*` except this `README.md` and
the sanitized `launch-seat.example.sh` template. Copy the template, fill the
`<PLACEHOLDERS>`, and keep your filled-in scripts local-only.

> These scripts previously lived under the operator campaign harness'
> `seats/` subdir. They now
> live here; compat symlinks remain at the old path so existing
> `ED4ALL_SEAT_LAUNCH_SPECS` values keep resolving unchanged.

## The three seat registries

A new seat is a set of registry entries, never a code change. All three are
comma-separated env vars parsed by `lib/vllm_container_lifecycle.py`:

| Env var | Shape | Maps |
|---------|-------|------|
| `ED4ALL_SEAT_BASE_URLS` | `<seat-name>=<loopback base_url>` | logical seat name → vLLM URL |
| `ED4ALL_VLLM_CONTAINERS` | `<base_url>=<container-name>` | vLLM URL → docker container |
| `ED4ALL_SEAT_LAUNCH_SPECS` | `<seat-name>=<abs path to a script here>` | logical seat name → cold-recreate launcher |

The declarative per-phase seat schedule (`ED4ALL_SEAT_SCHEDULE`, wired at each
workflow phase boundary) reconciles resident seats to a phase's `seats:`
annotation in `config/workflows.yaml`: it stops seats a phase does not need and
(cold-)starts the ones it does, then health-checks each with a liveness poll
(`/v1/models`) followed by a bounded content-coherence probe. When a seat has an
`ED4ALL_SEAT_LAUNCH_SPECS` entry, the schedule can **cold-recreate** it (via the
script here) to self-heal a mode-collapse.

## Two rules every seat script must honor

1. **Never co-resident.** Size `--gpus` / `--gpu-memory-utilization` so this
   seat and any other simultaneously-scheduled seat fit the card. The default
   small-box profile serves ONE heavy seat at a time under the GPU-lifecycle
   lease (`ED4ALL_GPU_LIFECYCLE`); only a large-unified-memory host runs several
   concurrently. Two seats that overcommit VRAM will OOM.

2. **Cold-recreate, never warm-start; always coherence-probe.** A warm
   `docker start` of a previously-stopped vLLM seat can come up live-but-
   mode-collapsed (passes `/v1/models` yet emits degenerate output). Every
   script here does `docker rm -f` + `docker run` (a fresh container), and the
   seat schedule content-coherence-probes every seat it (re)starts — a live seat
   that is incoherent is cold-recreated immediately and re-checked.

## Assistant-capable seats need OpenAI tool-calling flags

The `ed4all assistant` surface (and any caller that sends OpenAI
`tools` + `tool_choice:"auto"`) requires the seat to be launched with vLLM's
tool-calling machinery enabled. Without it, vLLM rejects the request with
**HTTP 400** (`"auto" tool choice requires --enable-auto-tool-choice`). Add
both flags to any seat that must answer tool calls:

```
--enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**Verified parser for Nemotron-3: `qwen3_coder`.** The Nemotron-3 chat
template (`chat_template.jinja`, shipped in both the Super-120B-NVFP4 and
Nano-30B-FP8 model dirs) emits tool calls in the XML grammar

```
<tool_call>
<function=NAME>
<parameter=KEY>
value
</parameter>
</function>
</tool_call>
```

which is byte-for-byte the format vLLM's `qwen3_coder` parser consumes. The
required `<tool_call>` / `</tool_call>` control tokens are atomic added-tokens
in the Nemotron-3 tokenizer (ids 14 / 15), so the parser constructs cleanly.
There is **no** `nemotron`-named tool parser in the vLLM build these seats run
(NGC `nvcr.io/nvidia/vllm:26.05.post1-py3`, vLLM `0.21.0`); `qwen3_coder` is the
correct match, not a workaround. Fallback if it ever misbehaves: `qwen3_xml`
(same grammar, streaming-oriented reimplementation, no special-token
construction guard). No chat-template override is needed — the model dir's
bundled tool-capable `chat_template.jinja` is auto-loaded.

Both `seats/launch-super.sh` and `seats/launch-nano.sh` carry these flags. The
Super seat additionally keeps its A/B-validated `--reasoning-parser nemotron_v3`
alongside the tool parser (reasoning + tool parsing coexist).

## Related

- `seats/launch-seat.example.sh` — sanitized copy-me template.
- `docs/operations/seat-schedule.env.example` — the seat-schedule env stack
  (source it alongside the run env; never enable it mid-build).
- Root `CLAUDE.md` → the `ED4ALL_SEAT_*` / `ED4ALL_VLLM_CONTAINER*` flag rows.
- `docs/operations/behavior-flags.md` — full seat-schedule / lifecycle detail.
