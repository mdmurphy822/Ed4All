# Resource-constrained SemantiK compatibility guide

The current deployment target is a DGX Spark-class host. This page preserves
the smaller-GPU procedure for compatibility testing and recovery environments;
it is not the recommended production topology. On a constrained GPU, Stage 6
can run fully on-device with the local **Qwen3-4B QLoRA specialists** (prose,
table, math, and gap fill) and no hosted calls.

The preferred production conversion lane is GLM-OCR and must be selected
explicitly because the code-level default remains the compatibility council:

```bash
export SEMANTIK_GLMOCR_LANE=1
export SEMANTIK_GLMOCR_BASE_URL=http://localhost:8002/v1
export SEMANTIK_GLMOCR_MODEL=glm-ocr
```

Leaving `SEMANTIK_GLMOCR_LANE` unset runs the live ModernBERT council cascade.
The page-arranger route is another flag-gated compatibility option, not part of
the preferred GLM-OCR recipe.

## Default (recommended): local 4B specialists author

Nothing is required. The Stage-6 draft tier is the local GGUF specialists, and
this is the corrected default — selecting a hosted provider no longer silently
skips them. Just run the cascade; the specialists author on-device.

## Optional: add a refine pass (still the local 4B, still on-device)

```bash
export SEMANTIK_SPECIALIST_REFINE=1   # Phase 1 (4B draft) -> Phase 2 (refine)
```

Both phases default to the **local** seat, so the refine pass runs on the same
on-device 4B plumbing — zero hosted calls. (Without this flag, only Phase 1
runs: the specialists author single-pass.)

## Model-agnostic seats (only if you later want a different Phase-2 model)

Both phases resolve independent `{provider, model}` seats; the hosted endpoint
is **opt-in only**. Leave all of these unset for the fully-local 4B setup:

| Env | Effect |
|-----|--------|
| `SEMANTIK_SPECIALIST_PHASE1_PROVIDER` / `_MODEL` | override the draft seat (default `local`) |
| `SEMANTIK_SPECIALIST_PHASE2_PROVIDER` / `_MODEL` | override the refine seat (default: single seat → `local`) |
| `SEMANTIK_SPECIALIST_BASE_URL` / `_API_KEY` | base_url / key for a non-`local` seat (a localhost server is still on-device) |
| `SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE=1` | pure-endpoint (the pre-fix behavior) |

A `local` provider uses the in-process GGUF specialist runtime; any other
provider value routes to the OpenAI-compatible runtime at
`SEMANTIK_SPECIALIST_BASE_URL`.

## Building the v3 structure dataset (lossless aligner)

```bash
cd SemantiK
../.venv/bin/python -m data.builders.build_structure_data --aligner global \
  --pair-dirs data/pairs/textbook data/pairs/arxiv data/pairs/wikipedia \
  --workers 4
# emits data/structure_dataset_v3/{train,val,test}.jsonl + coverage_report.json
```

The default aligner is `greedy` (byte-stable); `global` is the lossless
split/merge-aware path (`SEMANTIK_STRUCTURE_ALIGNER=global` is equivalent).

## Constrained-VRAM notes

- Council BERTs are released before Stage-6 (cascade phasing — one large model
  resident at a time).
- NLI grounding shares the card: `ED4ALL_NLI_DEVICE=cuda`,
  `ED4ALL_NLI_MIN_FREE_VRAM_MIB` (default 1024), `ED4ALL_NLI_EVICT_FOR_CUDA=true`
  handle contention (evict the resident local LLM, score, reload). See the root
  `CLAUDE.md`.
- Observability: `ED4ALL_VRAM_DOCTOR=1` logs a per-phase VRAM trajectory, and
  `ed4all doctor` preflights GPU fit.

These constraints describe the compatibility path. DGX Spark deployments
should use the seat schedule and GLM-OCR recipe in
`docs/operations/pipeline-invocation.md`.

## Environment

Use the repo-root `.venv` (it has `pikepdf` / `playwright` / `axe`); the system
`/usr/bin/python3` lacks the cascade's full stack.
