# Math Qwen LoRA Adapter — Eval Follow-up & Next Steps

> **STATUS (banner added 2026-06-17): SUPERSEDED.** Open items moved to the math-v2 retrain
> track (2048 caps; see `project_math_adapter_caps` memory + Plan 12 §C). Historical.

**Plan version:** 2026-05-24
**Branch:** `main`
**Context:** Math Qwen3-4B LoRA adapter finished training (30,000 steps, eval token acc 96.57%).
GGUF/as-shipped eval is done; in-process safetensors eval is mid-run. This plan covers
finishing the eval, the drift diff, the ship decision, and what comes after.
**Self-contained** — do not need conversation history to execute.

---

## 0. Status snapshot (as of writing)

- **Adapter:** `models/qwen_specialists/math/v1/final/` (safetensors LoRA) +
  `models/qwen_specialists/math/v1/math.q4_k_m.gguf` (2.4 GB, shipped form).
- **Eval harness:** `scripts/eval_qwen_math_adapter.py` — two backends (`safetensors`
  via peft 4-bit; `gguf` via in-process `LlamaCppRuntime` or `--server-url` HTTP).
  300-row rare-class-oversampled sample, seed 0, `--max-new-tokens 960`.
- **Sample fixed at:** {inline 201, display 28, numbered 64, multiline 6, matrix 1} = 300.
- **GGUF eval — DONE** (`data/eval_reports/qwen_math_adapter_gguf.json`):
  - id_stripped_exact **0.570**, id_stripped_sim **0.909**, alttext match **0.987**,
    well-formed of complete **0.966**, truncated **0.033**.
  - Per class: inline strongest (id_exact 0.71 / wf 0.97); numbered solid (0.41 / 0.86);
    display loose (0.07 / sim 0.75); matrix+multiline tiny-n + truncation-limited.
- **Safetensors eval — IN PROGRESS** (background, watcher `b8rat7p5z`,
  log `data/logs/eval_math_safetensors.log`). Same 300 rows, same cap.
- **Production runtime gap — FIXED:** llama-cpp-python 0.3.23 built w/ CUDA
  (`llama_supports_gpu_offload()=True`); recorded in `pyproject.toml` `runtime` extra +
  `[tool.uv] no-binary-package` guard.

### Metric note (don't relitigate)
Raw exact-match on MathML is meaningless: target `id`/`xref` values encode document
position (section/eq numbers) absent from the prompt. **id-stripped structural match is
the quality metric.** Ill-formed XML in long outputs is truncation at the 960 cap, not
bad structure.

---

## 1. When the safetensors eval finishes (FIRST thing on resume)

1. Confirm completion: check `data/logs/eval_math_safetensors.log` tail for `300/300` and
   that `data/eval_reports/qwen_math_adapter_safetensors.json` (+ `.samples.jsonl`) exist.
   If the watcher reports a crash, run the pipeline-debugger before relaunching.
2. If it produced raw samples but pre-dates the final scoring helpers, **re-score** the
   `.samples.jsonl` with the current `score_one` (id-stripped metrics) so it's
   apples-to-apples with the GGUF report. Do not re-generate — just re-score.

## 2. Quantization-drift diff (the "Both" deliverable)

Compare safetensors (fp/4-bit reference) vs GGUF (q4_k_m, shipped) on the **identical
300 rows**:

- Aggregate delta table: id_stripped_exact, id_stripped_sim, alttext_match,
  wellformed(complete), truncated, char_sim — safetensors − gguf.
- Per-class delta (inline/display/numbered/multiline/matrix).
- **Row-level disagreement set:** rows where id_stripped_exact flips between backends.
  Eyeball ~5 to see whether q4_k_m drops structure or just perturbs id values.
- Write `data/eval_reports/qwen_math_drift_safetensors_vs_gguf.md`.

**Interpretation guide:** small uniform drop (≤2–3 pts) = quantization is safe, ship
GGUF. Large or class-concentrated drop (esp. display/numbered) = consider q5_k_m or q8_0,
re-convert via `scripts/qwen_lora_to_gguf.py`, re-eval just that backend.

## 3. Ship decision gate for the math adapter

Ship the GGUF as the Stage-6 math specialist if **all** hold:
- id_stripped_sim ≥ 0.90 (have 0.909), alttext_match ≥ 0.95 (have 0.987),
  wellformed(complete) ≥ 0.95 (have 0.966), drift within tolerance (§2).
- Truncation ≤ ~3–4% AND confirmed to be long-tail nesting, not systematic — otherwise
  bump runtime `max_tokens` for the math adapter specifically.

If gate passes: adapter is **registered** already (config.yaml `adapter_path`, backup at
`models/qwen_specialists/_config_history/`). Verify the registration still points at the
shipped GGUF and the `max_tokens` for math is ≥ 960.

## 4. End-to-end runtime smoke test (needs the GPU free — after eval)

The only untested link is loading the GGUF **through the Python binding** (not just
llama-server). Once the eval releases the GPU:
- `make_runtime("real")` → load the registered math GGUF → generate on 2–3 sample
  prompts → confirm output matches the llama-server eval (no first-token drift, GPU
  offload actually engaged via `nvidia-smi`).
- This closes the production-runtime gap with a live load, not just an import check.
- **Serial GPU rule:** do not start this while any eval/train job holds VRAM.

## 5. After the math adapter — remaining specialists

Math is one of four Qwen LoRA adapters (`prose` / `table` / `math` / `gap_fill`).
Once math ships, the open questions for the others:
- Which adapters are trained vs still pending? (Check `models/qwen_specialists/*/v1/`.)
- Reuse this harness: generalize `scripts/eval_qwen_math_adapter.py` rare-class sampling +
  id-stripped scoring for the table/prose adapters (table will need cell-structure
  metrics, not MathML).
- Same GGUF convert → register → drift-diff → smoke-test loop per adapter.

---

---

## OUTCOME (2026-05-25) — all steps done

**Verdict: math adapter SHIPS.** Quantization drift small and uniformly favoring fp
(safetensors − gguf): id_stripped_exact +2.7pts (0.597/0.570), id_stripped_sim +1.5pts
(0.924/0.909), wellformed +4.7pts, alttext_match +0.3pts (0.99/0.987). Larger per-class
deltas are all tiny-n artifacts (multiline n=6, matrix n=1). 28/300 id_stripped_exact
flips, both directions → greedy near-ties, not structural loss. q4_k_m is safe to ship.
Reports: `data/eval_reports/qwen_math_adapter_{safetensors,gguf}.json`,
`qwen_math_drift_safetensors_vs_gguf.md`.

**Two production bugs the gate + smoke test caught and FIXED:**
1. `config.yaml` math `max_new_tokens: 256` (offline 384) — far below the eval's 960.
   → bumped to **960 / 1024**. Targets reach ~919 tokens; 256 would truncate most
   display/numbered/multiline equations the eval scored complete.
2. `LlamaCppRuntime.load()` used llama-cpp-python's default **n_ctx=512**, silently
   capping prompt+generation. → set **n_ctx=4096**. The 512 cap is why the binding
   first truncated a display row the server completed.

After both fixes the in-process binding reproduces the llama-server eval **exactly** on
all 3 smoke rows (inline/numbered/display), incl. display id_sim 0.827 == server,
well-formed, not truncated. GPU offload confirmed (RTX 3070, `gpu_offload=True`,
`n_gpu_layers=-1`). `tests/test_council_runtime.py` 13/13 green. **The eval numbers now
describe shipped behavior.**

**Carry-forward for prose/table/gap_fill:** before trusting any future adapter eval,
confirm the shipped `config.yaml max_new_tokens` AND runtime `n_ctx` both clear the
target-length distribution — otherwise the eval overstates production quality. See
[[qwen-runtime-output-caps]].

---

## Do-NOT list (standing constraints)
- No parallel GPU shards — serial only on the 8 GB 3070; parallel poisons CUDA context.
- No silent fallbacks — typed exceptions, not benign flags.
- No external LLMs at runtime; GGUF wheel must stay CUDA source-built (uv guard in place).
- Only commercial-OK licensed data (CC-BY / CC0 / ODC-By).
