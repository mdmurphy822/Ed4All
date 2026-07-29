# License-clean pipeline run — opt-in deployment recipe (W-D11.F)

This file documents how to opt INTO a license-clean Ed4All pipeline run. The Anthropic-defaulted path remains the canonical / backward-compatible default — this recipe is for operators who want a corpus that is ToS-clean from end to end (so the trained SLM adapter can be redistributed without separately negotiated provider agreements).

Cross-reference: `docs/LICENSING.md` is the canonical ToS-posture document. Read it before kicking off a multi-hundred-dispatch run.

---

## TL;DR — the four env vars

| Env var | License-clean value | What it changes |
|---------|---------------------|-----------------|
| `COURSEFORGE_REWRITE_PROVIDER` | `local` | Routes the Courseforge rewrite-tier authoring surface through a local OSS server (Ollama / vLLM) instead of Anthropic. |
| `COURSEPLANNER_PROVIDER` | `local` | **W-D14**: routes the Courseforge course-outliner surface (canonical `TO-NN` / `CO-NN` objective synthesis from `textbook_structure.json`) through `Courseforge/generators/_outliner_provider.py::OutlinerProvider`. The synthesised objective text propagates to every downstream chunk's `learning_outcome_refs[]`, so this surface IS training-data exposure. Bypasses the Claude Code `course-outliner` subagent dispatch. Reuses the same `LOCAL_SYNTHESIS_*` env vars as the other local-OSS surfaces. |
| `TRAINFORGE_ASSESSMENT_PROVIDER` | `local` | **W-D15**: routes the Trainforge assessment-generator surface (assessment-question authoring grounded in course content chunks) through `Trainforge/generators/_assessment_provider.py::AssessmentGeneratorProvider`. The authored questions land in `assessments.json` and feed into the downstream `training_synthesis` instruction-pair / preference-pair surface, so this surface IS training-data exposure. Bypasses the Claude Code `assessment-generator` subagent dispatch. Reuses the same `LOCAL_SYNTHESIS_*` env vars as the other local-OSS surfaces. |
| `TRAINFORGE_TARGET_MODELS` | `local/qwen2.5-14b,together/llama-3.3-70b` (or similar — operator-chosen CSV) | Cosmetic dataset_config.json field documenting which teacher models the corpus was synthesized against; lets the LibV2 audit trail record the actual ToS-clean teachers used. |
| `ED4ALL_LLM_JUDGE_PROVIDER` | `local_nli` | Wave-102 ablation eval routes its qualitative-judge calls through a local NLI classifier (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) instead of the default `none` (no LLM judge) or `anthropic`. |
| `COURSEFORGE_BLOCK_ROUTING_PATH` | optional legacy override | The checked-in routing YAML still has per-tier Qwen choices; it is not required for the canonical shared Nemotron seat. |

Plus the prerequisites the pipeline already documents on the canonical `--provider local` path:

- `COURSEFORGE_TWO_PASS=true` — required for `COURSEFORGE_BLOCK_ROUTING_PATH` to take effect.
- `COURSEFORGE_PROVIDER=local` — routes the legacy single-pass content-generator surface; same env var the canonical `docs/LICENSING.md` recommends.
- `LOCAL_SYNTHESIS_BASE_URL` (default `http://localhost:8000/v1`) and `LOCAL_SYNTHESIS_MODEL` (default `nemotron-3-nano-30b-a3b`) — read by every local-OSS provider (synthesis, curriculum-alignment, content-generator, outline, rewrite).
- `CURRICULUM_ALIGNMENT_PROVIDER=local` — routes Trainforge teaching-role classification through the local server.

---

## Pre-flight checklist

Before kicking off a license-clean run:

- [ ] A strict OpenAI-compatible local model server is running at `http://localhost:8000/v1`. Provision it with the exact pinned command in `docs/operations/nemotron-spark-serving.md` § “Serve Nano (fast tier)”. Ed4All does not launch a server and has no implicit Ollama fallback.
- [ ] All required model pulls completed. For the recipe below the local server must serve:
  - `nemotron-3-nano-30b-a3b` (canonical NVIDIA Nano served ID; roughly 30B total / 3.5B active parameters, with deployment memory determined by engine and precision).
  - `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for the local-NLI judge — auto-downloaded by `transformers` on first call; no manual pull required.
- [ ] **`ANTHROPIC_API_KEY` is explicitly UNSET** for the run. The orchestrator's `--mode local` default uses the Claude Code session, which does not consume the env var, but several legacy code paths (`Trainforge/generators/_anthropic_provider.py`, `Trainforge/eval/qualitative_judge.py` when `ED4ALL_LLM_JUDGE_PROVIDER=anthropic`) silently route to Anthropic if the key is present. Unsetting the key is the belt-and-braces guarantee that no legacy path leaks training-data through Anthropic.
- [ ] Calibration prerequisite for `assessment_item` is met (see next section).
- [ ] `pip install -e '.[embedding]'` has run — the four statistical-tier validators wired into the two-pass router (objective_assessment_similarity, concept_example_similarity, objective_roundtrip_similarity, bloom_classifier_disagreement) need `sentence-transformers` + `transformers` + `torch`. Without these the validators emit `EMBEDDING_DEPS_MISSING` warnings and skip; for production runs set `TRAINFORGE_REQUIRE_EMBEDDINGS=true` to fail-closed instead.

---

## Calibration prerequisite for `assessment_item`

`Courseforge/CLAUDE.md` states that the `assessment_item` block "MUST stay on Anthropic to avoid silent regression". That guidance was calibrated against `claude-sonnet-4-6`; the 32B local substitute in `block_routing.license_clean.yaml` is the right SHAPE for license-clean ops but its distractor-quality + misconception-targeting performance has NOT been empirically validated against the Courseforge eval suite at the time this recipe ships.

The full calibration loop is documented inline in `Courseforge/config/block_routing.license_clean.yaml` (under the "CALIBRATION RISK" header at the top of the file). Summary:

1. Pick a representative chapter (10-20 blocks of every type).
2. Rebuild that chapter under both routings (license-clean variant vs. canonical).
3. Compare outputs against the four validators wired into the `assessment_item.validators.required` matrix:
   - `objective_assessment_similarity` (>=0.55 cosine floor).
   - `bloom_alignment`.
   - `answerability`.
   - plus the `optional` `distractor_entropy` advisory.
4. If the local-32B variant fails any required gate at a meaningfully higher rate than the Anthropic baseline, KEEP `assessment_item.rewrite` cascade pinned to the canonical `large` tier (i.e. comment out the assessment_item entry in the license-clean variant) and accept the per-block ToS hit on `assessment_item` only.

Until that calibration loop closes, courses built under the license-clean variant should NOT promote past `non_certified_archive` on the Wave-3 promotion chain (`lib/governance/course_status.py::derive_course_status`). The `course_status` enum's `certified_*` cohorts gate on `assessment_item` distractor quality through the `INSTRUCTIONAL_GATE_IDS` cohort; an uncalibrated routing hit there is a structural signal that the corpus is not yet trainable-grade.

---

## Recipe — minimum viable license-clean run

```bash
# Start the provisioned strict OpenAI-compatible TRT-LLM/vLLM seat.
# The canonical seat serves nemotron-3-nano-30b-a3b on localhost:8000.

# Belt-and-braces: explicitly unset Anthropic key.
unset ANTHROPIC_API_KEY

# Provider routing — the documented env vars.
export COURSEFORGE_REWRITE_PROVIDER=local
export COURSEPLANNER_PROVIDER=local             # W-D14 — course-outliner surface
export TRAINFORGE_ASSESSMENT_PROVIDER=local     # W-D15 — assessment-generator surface
export TRAINFORGE_TARGET_MODELS="local/nemotron-3-nano-30b-a3b"
export ED4ALL_LLM_JUDGE_PROVIDER=local_nli

# Prerequisites the canonical license-clean recipe already documents.
export COURSEFORGE_TWO_PASS=true
export COURSEFORGE_PROVIDER=local
export CURRICULUM_ALIGNMENT_PROVIDER=local
export LOCAL_SYNTHESIS_BASE_URL=http://localhost:8000/v1
export LOCAL_SYNTHESIS_MODEL=nemotron-3-nano-30b-a3b
export TRAINFORGE_REQUIRE_EMBEDDINGS=true

# Run the full pipeline.
ed4all run textbook-to-course \
  --corpus pdfs/ \
  --course-name PHYS_101 \
  --provider local
```

---

## Together fallback (no 24 GB GPU)

For deployments without a consumer 24 GB GPU, swap the `large` tier in the routing YAML to a hosted Apache-2.0 OSS model. Edit `Courseforge/config/block_routing.license_clean.yaml`'s `capability_tiers.large` block to:

```yaml
  large:
    provider: together
    model: Qwen/Qwen2.5-72B-Instruct-Turbo
    temperature: 0.4
    max_tokens: 2400
```

Then add `TOGETHER_API_KEY` to the env. Together AI's ToS explicitly permits using outputs as training data; Qwen2.5-72B is gated by the Qwen License Agreement which permits derivative training at any scale (commercial use of the model itself is gated at >100M MAU). See `docs/LICENSING.md` Synthesis providers table for the full per-model license matrix.

---

## What's NOT covered (engineering-wave gaps)

The five-env-var recipe above closes the largest training-data exposure paths in the pipeline. The Anthropic-pinned subagent surfaces have all been routed through in-process license-clean providers as of Wave W-D15:

## Conversion (SemantiK) — PDF → HTML license-clean by construction

The PDF → accessible-HTML conversion stage is **SemantiK**. There is no
Anthropic default to flip on the conversion path: SemantiK's extraction stack carries no
PyMuPDF/MuPDF (AGPL-3) or Poppler (GPL-2) and ships Apache-2.0, and its runtime
runs **fully offline** by default — the BERT council, OCR, theta, and the
Stage-6 Qwen specialists are all local. The conversion output (which is later
ingested as Trainforge training chunks) is therefore license-clean with **no
operator action required**.

**Default (fully offline):**

```bash
# No env vars needed — SemantiK runs the local GGUF council + Qwen
# specialists in-process. extraction / OCR / theta are local-only.
export SEMANTIK_SPECIALIST_PROVIDER=local   # this is already the default
```

**Optional hosted large-model quality seat.** Stage-6 specialist generation (and the
off-by-default Stage-5d structure reviewer) can be routed to a hosted large-model
endpoint for higher quality:

```bash
export SEMANTIK_SPECIALIST_PROVIDER=nvidia   # opt-in quality seat, not a dependency
export NVIDIA_API_KEY=nvapi-...
# SEMANTIK_SPECIALIST_MODEL / SEMANTIK_STRUCTURE_REVIEW_MODEL override the model;
# both resolve through NVIDIA_LARGE_MODEL → meta/llama-3.3-70b-instruct by default.
```

This hosted seat produces **structured product content** (the accessible HTML),
not a training-data corpus. The standard content → training-data caveat applies:
operators who want a fully ToS-clean training corpus should keep
`SEMANTIK_SPECIALIST_PROVIDER=local` (the default), which leaves the conversion
path with zero cloud exposure. Full flag detail + the licensing row:
`docs/operations/behavior-flags-semantik.md` and `docs/LICENSING.md`.

### Assessment-generator subagent (W-D15) — closed

W-D15 closes the assessment-generator subagent gap. `TRAINFORGE_ASSESSMENT_PROVIDER=local` (or any registered OpenAI-compatible provider) now routes assessment-question authoring through `Trainforge/generators/_assessment_provider.py::AssessmentGeneratorProvider`, mirroring the W-D14 `OutlinerProvider` pattern. The authored questions land in `assessments.json` and feed into the downstream `training_synthesis` instruction-pair / preference-pair surface — so closing this seam was the dominant remaining training-data exposure on the Trainforge assessment surface. The provider's user prompt instructs the LLM to emit an `evidence_quote` field per question per the W-D11 T11.3 grounding contract; the per-call `assessment_generator_call` decision event surfaces the dynamic `evidence_quote_emit_rate` so a post-hoc audit can replay grounding quality.

### Honest scope

This recipe documents a license-clean COURSEWARE / TRAINING-CORPUS run for every dominant code path with the W-D15 wave landed: every Anthropic-defaulted subagent surface that touches training data now has a license-clean provider seam (`COURSEFORGE_PROVIDER` for content-generator, `COURSEPLANNER_PROVIDER` for course-outliner, `TRAINFORGE_ASSESSMENT_PROVIDER` for assessment-generator, `CURRICULUM_ALIGNMENT_PROVIDER` for align_chunks). The PDF → HTML conversion surface no longer needs a seam at all: SemantiK is license-clean by construction and offline by default (`SEMANTIK_SPECIALIST_PROVIDER=local`). Operators training adapters for redistribution should still verify the calibration prerequisites for the affected block-types and document any per-block Anthropic exposure (e.g. an uncalibrated `assessment_item` block staying on Anthropic per the calibration loop above) in the corpus's audit trail.

---

## See also

- `docs/operations/pipeline-invocation.md` — the **operational** companion: per-stage invocation (stop-after / reuse / stage subcommands), the timeout knobs that actually fire (`ED4ALL_TASK_TIMEOUT_MINUTES` for a slow in-process `course_planning`), the outline-vs-rewrite naming, and the pure-local constrained-VRAM (≈8 GB) env recipe. This licensing doc covers *which* seats to pin; that one covers *how* to invoke each stage.
- `docs/LICENSING.md` — canonical ToS posture, per-provider terms, per-model license matrix.
- `Courseforge/CLAUDE.md` § "Opt-In Behavior Flags" — full env-var table for the Courseforge two-pass router.
- `Courseforge/config/block_routing.license_clean.yaml` — the sibling YAML this recipe pins via `COURSEFORGE_BLOCK_ROUTING_PATH`.
- `Trainforge/CLAUDE.md` § "Synthesis providers" — same env-var stack from the Trainforge perspective.
- `lib/governance/course_status.py::derive_course_status` — Wave-3 promotion-chain composer; the 5-value `course_status` enum a license-clean run targets.
