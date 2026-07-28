# Nemotron 3 on the DGX Spark — self-hosted serving runbook

Stand up the NVIDIA Nemotron 3 models **locally** on the DGX Spark (GB10,
`sm_121` Blackwell, 128 GB unified memory) and route each Ed4All synthesis
surface at them. This runbook is **entirely local self-hosted** — the hosted
`nvidia` API seat (`https://integrate.api.nvidia.com/v1`) stays **gated OFF**
and nothing here dispatches to the cloud (see § "Hosted seat stays off").

The endpoint rows already exist in `config/endpoints.yaml`:

| Endpoint name | Default served-model-name | Purpose | Base URL env |
|---------------|---------------------------|---------|--------------|
| `spark-super` | `nemotron-3-super-120b-a12b` | Throughput / synthesis tier | `SPARK_BASE_URL` |
| `spark-nano`  | `nemotron-3-nano-30b-a3b`   | Fast tier + interactive answer | `SPARK_BASE_URL` |
| `local`       | `nemotron-3-nano-30b-a3b` | Canonical strict OpenAI-compatible Nano seat (localhost:8000) | `LOCAL_SYNTHESIS_BASE_URL` |

Both `spark-*` rows carry `provenance_provider: local` (genuinely local:
self-hosted OSS weights on our own box under the NVIDIA Open Model License,
which permits commercial use **and** training on outputs). They share
`SPARK_BASE_URL` / `SPARK_API_KEY` and differ only in served-model-name.

---

## 0. Model inventory + HF repo IDs

> **Verification status (checked against Hugging Face, 2026-07):** the Super
> and Nano text repo IDs are confirmed. The **Omni repo id in the original
> ask was wrong** — corrected below. Precision variant matters: the **DGX
> Spark uses the NVFP4 checkpoint**, not the FP8 one (FP8 is the B200/B300
> single-GPU checkpoint).

| Role | Confirmed HF repo id | Notes |
|------|----------------------|-------|
| Super — DGX Spark **(recommended)** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 120B / 12B-active LatentMoE + MTP. NVFP4 is the Spark-validated checkpoint (per the vLLM DGX Spark playbook). |
| Super — B200/B300 | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` | The FP8 checkpoint the original ask named; fits a single B200/B300 with `--tensor-parallel-size 1`. Listed for completeness — **prefer NVFP4 on the Spark**. |
| Nano — text / LoRA base | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` at revision `cbd3fa9f933d55ef16a84236559f4ee2a0526848` | Canonical training and retrieval-generation base: roughly 30B total / 3.5B active. BF16 is required by the checked-in LoRA config; `Trainforge/training/base_models.py` pins this exact repo and revision. |
| Nano — Omni (multimodal, for SemantiK VLM) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` | **CORRECTED repo id.** The ask's `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B` does not exist — the real repo has **no `NVIDIA-` prefix** and a **`-Reasoning-<precision>` suffix**. Variants: `-Reasoning-FP8` / `-Reasoning-BF16` / `-Reasoning-NVFP4`. Unifies video/audio/image/text; use for the SemantiK VLM (OCR/vision) surface. |

### Download

```bash
# Auth once (Nemotron repos are ungated but HF_TOKEN avoids rate limits).
export HF_TOKEN=hf_...

# Super (DGX Spark: NVFP4).
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

# Nano text / LoRA base. Pin the same immutable revision as the trainer.
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --revision cbd3fa9f933d55ef16a84236559f4ee2a0526848

# Nano Omni (SemantiK VLM). VERIFY the precision your GPU/image wants.
huggingface-cli download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
```

> On the Spark, models land in `~/.cache/huggingface`. Bind-mount that into
> the vLLM container (below) so a container restart doesn't re-download.

---

## Strict OpenAI server (`local` / `spark-nano` throughput path)

The canonical deployment is a TRT-LLM or vLLM OpenAI-compatible server on
**port 8000**. Both `LOCAL_SYNTHESIS_BASE_URL` and `SPARK_BASE_URL` may point
at that seat. Ed4All has no implicit Ollama endpoint or lifecycle fallback.

### TensorRT-LLM structured-output synthesis seat

TensorRT-LLM 1.3.0rc9 parses OpenAI `response_format` even when guided
decoding is disabled. A successful flat JSON response therefore does not prove
schema enforcement. Any staged-synthesis seat must set:

```yaml
guided_decoding_backend: xgrammar
```

Readiness must confirm the startup log contains `Guided decoder initialized
with backend: ...XGRAMMAR`. Use OpenAI
`response_format.type=json_schema`; rc9 does not accept vLLM-style top-level
`guided_json`, `guided_regex`, or `guided_grammar` fields.

When `--reasoning_parser` is configured, rc9 rewrites schema guidance into a
forced reasoning-plus-content grammar. A synthesis seat requiring thinking-off
must omit that flag. The staged client fails loudly if `reasoning_content` is
returned.

Readiness requires exact validation of flat and nested object/array schemas,
enum/required/`additionalProperties: false`, an adversarial instruction, and
the production plan/SFT/DPO schemas. Every response must have
`finish_reason=stop`, with zero reasoning and zero truncation.

### Staged synthesis preconditions (operator environment setup)

Staged synthesis validates three required operator environment variables at
orchestration time, before any synthesis output or provider is created:

1. **`TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS`** — The served model's
   context window size (tokens). This is obtained from the model's HF card or
   deployment notes. Staged synthesis fails closed if missing, blank, zero, or
   non-numeric.
   
2. **`TRAINFORGE_REQUIRE_EMBEDDINGS=true`** — An embedding backend
   (sentence-transformers or HuggingFace) is required for evidence-grounded SFT
   pair selection. Staged synthesis fails closed if not explicitly set to one of
   `{1, true, yes, on}` (case-insensitive).

3. **`LOCAL_SYNTHESIS_MODEL`** — The exact model ID served at
   `LOCAL_SYNTHESIS_BASE_URL`, validated against the server's OpenAI-compatible
   `/v1/models` endpoint response by
   `Trainforge/synthesize_training.py::_preflight_local_staged_model_identity`.
   Staged synthesis fails closed if the ID is absent from the served list. This
   prevents silent model-mismatch 404s mid-phase. Read the id off `/v1/models`
   rather than guessing it: a seat launched without `--served-model-name`
   (as `seats/launch-super-trtllm.sh` is) reports its checkpoint snapshot path
   as the id.

All three must be set in your shell environment before sourcing the run-env
template. See `docs/operations/run-env.example.sh` for example declarations and
`Trainforge/CLAUDE.md` § "Opt-In Behavior Flags" for per-flag detail.

Primary references: NVIDIA TensorRT-LLM
[guided decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc9/docs/source/features/guided-decoding.md)
and the rc9
[OpenAI protocol translation](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc9/tensorrt_llm/serve/openai_protocol.py#L203-L250).

### Serve Super (DGX Spark, NVFP4)

Command shape from the vLLM DGX Spark playbook
(`vllm.ai/blog/2026-06-01-vllm-dgx-spark`) — **verify the exact flags against
that playbook for your image/driver**:

```bash
export HF_TOKEN=hf_...

docker run -d --name vllm-super --ipc=host --restart unless-stopped \
  --gpus all -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super-120b-a12b \
    --trust-remote-code \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4 \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

Notes:
- `--served-model-name nemotron-3-super-120b-a12b` **must match** the
  `spark-super` endpoint's `default_model` (or whatever you set
  `SPARK_SUPER_MODEL` to). The blog uses the shorter `nemotron-3-super`;
  we pin the endpoint-default name so `LLM_PROVIDER=spark-super` resolves
  with no extra env.
- Quantization: **no `--quantization` flag** — the NVFP4 checkpoint is
  pre-quantized.
- `--tensor-parallel-size` is **omitted** for a single Spark; the playbook
  notes `--tensor-parallel-size 2` applies only when linking two Sparks via
  ConnectX-7.
- `--max-num-seqs 4` / `--gpu-memory-utilization 0.85` balance weights + KV
  cache in the 128 GB unified pool. **Raise `--max-num-seqs`** if you want
  more concurrency headroom for the benchmark below (this is exactly the
  batching knob the benchmark's `--concurrency` exercises) — verify it fits
  memory.
- Image `vllm/vllm-openai:cu130-nightly` is the `sm_121`-validated image per
  the playbook — **verify the current tag**.

### Serve Nano (fast tier)

Same shape, different served-model-name + repo. Run it as a **second server
on a different port** if you want both resident, or swap in/out — but note
Ed4All's default `ED4ALL_GPU_LIFECYCLE=on` (below) means models load/release
per phase, so co-residency is usually unnecessary on the 119 GB box.

```bash
docker run -d --name vllm-nano --ipc=host --restart unless-stopped \
  --gpus all -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --revision cbd3fa9f933d55ef16a84236559f4ee2a0526848 \
    --served-model-name nemotron-3-nano-30b-a3b \
    --trust-remote-code \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 8 \
    --reasoning-parser nemotron_v3
```

> `--served-model-name nemotron-3-nano-30b-a3b` must match the `spark-nano`
> endpoint default (or `SPARK_NANO_MODEL`). If both servers listen on 8000
> you can only run one at a time; give each its own port and set
> `SPARK_BASE_URL` accordingly when you switch, OR serve both model names
> from a single vLLM process if your image supports multi-model serving
> (**verify** — historically vLLM is one model per server).

---

## Routing Ed4All surfaces at Nemotron

Every surface resolves its endpoint BY NAME from `config/endpoints.yaml`, so
routing is env-only — no code change.

```bash
# Point the shared Spark base URL at the local vLLM server.
export SPARK_BASE_URL=http://localhost:8000/v1
export SPARK_API_KEY=local          # vLLM ignores auth; stable placeholder

# --- Whole-pipeline default: everything at the Super seat. ---
export LLM_PROVIDER=spark-super

# --- OR per-tier (finer control): keep authoring on Super, fast bits on Nano. ---
export COURSEFORGE_PROVIDER=spark-super
export COURSEPLANNER_PROVIDER=spark-super
export TEXTBOOK_SYNTHESIS_PROVIDER=spark-super
export COURSEFORGE_OUTLINE_PROVIDER=spark-super
export COURSEFORGE_REWRITE_PROVIDER=spark-super
# (interactive answer path / classifiers can go to the cheaper Nano seat)

# --- Served-model-name overrides (only if you served a different name than
#     the endpoint default). ---
export SPARK_SUPER_MODEL=nemotron-3-super-120b-a12b
export SPARK_NANO_MODEL=nemotron-3-nano-30b-a3b

# Generic local retrieval-generation resolves to the same base seat.
export LOCAL_SYNTHESIS_BASE_URL=http://localhost:8000/v1
export LOCAL_SYNTHESIS_MODEL=nemotron-3-nano-30b-a3b
export ED4ALL_ANSWER_MODEL=nemotron-3-nano-30b-a3b

# --- SemantiK VLM (OCR/vision) at the Omni model. ---
export SEMANTIK_VLM_MODEL=nemotron-3-nano-omni-30b-a3b
export SEMANTIK_VLM_BASE_URL=http://localhost:11434   # or the vLLM port serving Omni
```

After course LoRA training, full evaluation, and promotion, set
`ED4ALL_ANSWER_MODEL` (and the server-side served model name) to the promoted
adapter ID. This is deliberately a deferred handoff: the base Nano ID remains
the default until promotion, and this runbook never invents or pre-binds an
adapter.

### GPU lifecycle stays ON

Leave `ED4ALL_GPU_LIFECYCLE=on` (the project default). The deterministic
phase-boundary lease loads each model, runs its phase, and **releases the
card** before the next stage — so on the 119 GB Spark the Super, Nano, and
Omni models **never need to co-reside**. Do NOT force them all resident to
"save load time"; the lifecycle sweep is what keeps a multi-model run inside
the memory budget. (See the root `CLAUDE.md` flag table for
`ED4ALL_GPU_LIFECYCLE` semantics.)

---

## Hosted seat stays off

The hosted `nvidia` endpoint (`https://integrate.api.nvidia.com/v1`, gated by
`NVIDIA_API_KEY`) is **not used by this runbook**. Everything here is local
self-hosted vLLM/Ollama on the Spark. Do not set `NVIDIA_API_KEY` for this
flow — the `spark-*` seats are `provenance_provider: local` precisely because
they run on our own box; routing through the cloud seat would both change the
licensing posture and defeat the point of the Spark. Keep the cloud seat
gated OFF (per the standing "NVIDIA off — pure-local" posture).

---

## Benchmark the seats

Once a server is up, measure single-stream vs batched throughput per seat.
The batched aggregate tps (and its scaling factor over a single stream) is
the number that justifies the Spark:

```bash
# Sweep all three seats at concurrency 32 (skip any seat whose server is
# down — the harness records the error and continues).
python scripts/integration/benchmark_generation_providers.py \
    --providers local spark-nano spark-super --concurrency 32

# Just the Super seat, harder fan-out, bigger decode cap:
python scripts/integration/benchmark_generation_providers.py \
    --providers spark-super --concurrency 64 --max-tokens 512
```

The harness resolves each provider through the SAME registry client the
pipeline uses (`lib.llm.endpoints.build_openai_compatible_client`), prints a
per-provider table (single tps / aggregate tps / scaling factor), and writes
a JSON report to `state/benchmarks/generation_providers_<ts>.json`. An
unreachable seat (server not up yet) is recorded as a per-provider error and
the sweep continues. Time-to-first-token is **not** reported — the shared
client does not stream tokens, so TTFT is left `null` rather than fabricated.

To raise the achievable aggregate tps, increase the vLLM server's
`--max-num-seqs` (and re-run the benchmark with a matching `--concurrency`)
until memory or the scaling curve flattens.

### Selecting synthesis concurrency

Concurrency is a property of the complete deployment and workload, not of the
model name alone. Benchmark the exact model revision, engine image, server
batch/token limits, prompt contract, and output allowance that production will
use. Store raw reports under ignored `state/benchmarks/`; never copy
course-derived prompts, responses, workflow IDs, machine paths, or local
artifact hashes into tracked documentation.

Sweep a bounded candidate set and require every cell to report:

- complete requests and terminal synthesis units;
- prompt, completion, and accepted-pair throughput;
- p50/p95 latency and queue delay when observable;
- context and scheduled-token headroom;
- KV/Mamba/unified-memory headroom, or an explicit `unavailable`;
- transport, output-cap, schema, and validator failures.

Stop escalation at the first truncation, transport failure, engine-hang signal,
or exhausted scheduler/cache headroom. Select the lowest concurrency on the
accepted-pair-throughput plateau, then validate it with production-shaped SFT
and DPO windows and a longer soak. A shallow `/health` or `/v1/models` response
does not prove generation health; retain an exact structured generation probe.

`TRAINFORGE_SYNTHESIS_MAX_CONCURRENT` defaults to `1`. Any higher value and any
request-timeout override are deployment-specific operator settings that must be
derived from the current benchmark, not copied from another machine's result.
Re-benchmark after changing the model, revision, engine, prompt/schema,
token cap, batch limit, or hardware.

---

## References

- NVIDIA NeMo training source of truth in this repo:
  `Trainforge/training/base_models.py` and
  `Trainforge/training/configs/nemotron3-nano-30b.yaml`
- Nemotron 3 Super Spark deployment guide (NVIDIA docs) — verify current URL.
- HF repos: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`,
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`,
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`,
  `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`
- Endpoint registry: `config/endpoints.yaml` (`spark-super` / `spark-nano` /
  `local` rows). Deployment-target notes: DGX Spark memory posture in the
  root `CLAUDE.md` GPU-lifecycle + big-memory flag rows.
