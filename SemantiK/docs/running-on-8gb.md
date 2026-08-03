# Running SemantiK Stage-6 locally (8GB GPU, fully on-device)

This box is a single ~8GB GPU. Stage-6 runs fully on-device on the
license-clean local **Qwen3-4B QLoRA specialists** (prose / table / math /
gap_fill) — no hosted or external calls.

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

## VRAM notes (8GB)

- Council BERTs are released before Stage-6 (cascade phasing — one large model
  resident at a time).
- NLI grounding shares the card: `ED4ALL_NLI_DEVICE=cuda`,
  `ED4ALL_NLI_MIN_FREE_VRAM_MIB` (default 1024), `ED4ALL_NLI_EVICT_FOR_CUDA=true`
  handle contention (evict the resident local LLM, score, reload). See the root
  `CLAUDE.md`.
- Observability: `ED4ALL_VRAM_DOCTOR=1` logs a per-phase VRAM trajectory, and
  `ed4all doctor` preflights GPU fit.

## Environment

Use the repo-root `.venv` (it has `pikepdf` / `playwright` / `axe`); the system
`/usr/bin/python3` lacks the cascade's full stack.
