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
| `local`       | `qwen2.5:7b-instruct-q4_K_M` | Existing Ollama seat (localhost:11434) | `LOCAL_SYNTHESIS_BASE_URL` |

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
| Nano — text | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | 30B / 3.2B-active hybrid MoE. Fast tier + interactive answer. NVFP4 variant `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` also exists — **verify which precision your Spark image prefers**. |
| Nano — Omni (multimodal, for SemantiK VLM) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` | **CORRECTED repo id.** The ask's `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B` does not exist — the real repo has **no `NVIDIA-` prefix** and a **`-Reasoning-<precision>` suffix**. Variants: `-Reasoning-FP8` / `-Reasoning-BF16` / `-Reasoning-NVFP4`. Unifies video/audio/image/text; use for the SemantiK VLM (OCR/vision) surface. |

### Download

```bash
# Auth once (Nemotron repos are ungated but HF_TOKEN avoids rate limits).
export HF_TOKEN=hf_...

# Super (DGX Spark: NVFP4).
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

# Nano text (fast tier).
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

# Nano Omni (SemantiK VLM). VERIFY the precision your GPU/image wants.
huggingface-cli download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
```

> On the Spark, models land in `~/.cache/huggingface`. Bind-mount that into
> the vLLM container (below) so a container restart doesn't re-download.

---

## Option A — quick baseline via Ollama (`local` endpoint)

The **existing** `local` endpoint (`http://localhost:11434/v1`) already
points at Ollama. This is the fastest way to smoke a Nemotron model, but
**Ollama's batching is weak** — use it for the **interactive / answer path**,
NOT the throughput/synthesis path.

```bash
# Pull a Nemotron 3 tag (VERIFY the exact Ollama tag once it publishes —
# 'nemotron3:...' is illustrative; check `ollama list` / the Ollama library).
ollama pull nemotron3:nano          # verify tag

# Point the existing `local` endpoint's MODEL env at it (base URL unchanged).
export LOCAL_SYNTHESIS_MODEL=nemotron3:nano   # verify tag
# LOCAL_SYNTHESIS_BASE_URL stays http://localhost:11434/v1 (endpoint default)
```

No new endpoint row is needed — `local` resolves `LOCAL_SYNTHESIS_MODEL` per
its registry row. Good for a `ed4all doctor` / interactive `answer` smoke;
throughput will be single-stream-ish because Ollama does not do the
continuous batching that makes the Spark worth it.

---

## Option B — vLLM OpenAI server (`spark-super` / `spark-nano` throughput path)

This is the real throughput path. One vLLM OpenAI-compatible server on **port
8000** serves the Spark seats. `SPARK_BASE_URL=http://localhost:8000/v1`.

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
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
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

# --- SemantiK VLM (OCR/vision) at the Omni model. ---
export SEMANTIK_VLM_MODEL=nemotron-3-nano-omni-30b-a3b
export SEMANTIK_VLM_BASE_URL=http://localhost:11434   # or the vLLM port serving Omni
```

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

---

## References

- vLLM DGX Spark playbook: `vllm.ai/blog/2026-06-01-vllm-dgx-spark`
- Nemotron 3 Super Spark deployment guide (NVIDIA docs) — verify current URL.
- HF repos: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`,
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`,
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`,
  `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`
- Endpoint registry: `config/endpoints.yaml` (`spark-super` / `spark-nano` /
  `local` rows). Deployment-target notes: DGX Spark memory posture in the
  root `CLAUDE.md` GPU-lifecycle + big-memory flag rows.
