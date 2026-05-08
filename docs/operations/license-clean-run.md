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
| `COURSEFORGE_BLOCK_ROUTING_PATH` | `Courseforge/config/block_routing.license_clean.yaml` | Points the two-pass router at the sibling routing YAML that overrides the `large` capability tier from `claude-sonnet-4-6` to a 32B local Qwen. See "Calibration prerequisite" below. |

Plus the prerequisites the pipeline already documents on the canonical `--provider local` path:

- `COURSEFORGE_TWO_PASS=true` — required for `COURSEFORGE_BLOCK_ROUTING_PATH` to take effect.
- `COURSEFORGE_PROVIDER=local` — routes the legacy single-pass content-generator surface; same env var the canonical `docs/LICENSING.md` recommends.
- `LOCAL_SYNTHESIS_BASE_URL` (default `http://localhost:11434/v1`) and `LOCAL_SYNTHESIS_MODEL` (default `qwen2.5:14b-instruct-q4_K_M`) — read by every local-OSS provider (synthesis, curriculum-alignment, content-generator, outline, rewrite).
- `CURRICULUM_ALIGNMENT_PROVIDER=local` — routes Trainforge teaching-role classification through the local server.

---

## Pre-flight checklist

Before kicking off a license-clean run:

- [ ] Local OSS model server is running. Ollama default (`http://localhost:11434/v1`), vLLM (`http://localhost:8000/v1`), llama.cpp server (`http://localhost:8080/v1`), or LM Studio (`http://localhost:1234/v1`) all work via the OpenAI-compatible client. Set `LOCAL_SYNTHESIS_BASE_URL` if not Ollama default.
- [ ] All required model pulls completed. For the recipe below the local server must serve:
  - `qwen2.5:7b-instruct-q4_K_M` (outline tier; ~5 GB VRAM in 4-bit).
  - `qwen2.5:14b-instruct-q4_K_M` (medium-tier rewrite; ~10 GB VRAM in 4-bit).
  - `qwen2.5:32b-instruct-q4_K_M` (large-tier rewrite + assessment_item cascade; ~22 GB VRAM in 4-bit). On 16 GB hardware, fall back to Together (see "Together fallback" below).
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
# Local OSS server (Ollama default; swap base_url for vLLM / llama.cpp / LM Studio).
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull qwen2.5:32b-instruct-q4_K_M

# Belt-and-braces: explicitly unset Anthropic key.
unset ANTHROPIC_API_KEY

# Provider routing — the documented env vars.
export COURSEFORGE_REWRITE_PROVIDER=local
export COURSEPLANNER_PROVIDER=local             # W-D14 — course-outliner surface
export TRAINFORGE_ASSESSMENT_PROVIDER=local     # W-D15 — assessment-generator surface
export TRAINFORGE_TARGET_MODELS="local/qwen2.5-14b,together/llama-3.3-70b"
export ED4ALL_LLM_JUDGE_PROVIDER=local_nli
export COURSEFORGE_BLOCK_ROUTING_PATH=Courseforge/config/block_routing.license_clean.yaml

# Prerequisites the canonical license-clean recipe already documents.
export COURSEFORGE_TWO_PASS=true
export COURSEFORGE_PROVIDER=local
export CURRICULUM_ALIGNMENT_PROVIDER=local
export LOCAL_SYNTHESIS_BASE_URL=http://localhost:11434/v1
export LOCAL_SYNTHESIS_MODEL=qwen2.5:14b-instruct-q4_K_M
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

## DART (W-D13) — PDF → HTML conversion now license-clean

DART's PDF converter (`DART/pdf_converter/converter.py`) routes through the W-D13 `DART_PROVIDER` + `DART_VISION_PROVIDER` env vars. Default unset → the legacy Anthropic path stays in place (`DART_CLAUDE_MODEL` pinning `claude-sonnet-4-20250514`); set the W-D13 vars to flip the converter onto a license-clean backend. DART HTML output is later ingested as Trainforge training chunks, so closing this seam was the dominant remaining DART training-data exposure.

**License-clean DART recipe (W-D13):**

```bash
# Text-mode structure detection: route through local Ollama / vLLM.
export DART_PROVIDER=local
export LOCAL_SYNTHESIS_MODEL=qwen2.5:14b-instruct-q4_K_M  # Apache 2.0

# Vision-mode alt-text: pick ONE of the three options below.

# (a) Local vision model (cheapest, fully offline; needs ~22 GB VRAM
# for qwen2.5-vl:32b or ~16 GB for qwen2.5-vl:7b):
export DART_VISION_PROVIDER=local
export LOCAL_VISION_CAPABLE=true
# Operator picks: load a vision model into the local server.
# The resolver heuristic auto-flips when LOCAL_SYNTHESIS_MODEL contains
# vision / llava / -vl substrings, so an explicit env opt-in isn't
# required when the model identifier already names the modality.

# (b) Together AI's Llama-3.2-Vision (cloud OSS, ToS-clean for
# training-data; needs TOGETHER_API_KEY + ~$0.0006/image at 90B):
export DART_VISION_PROVIDER=together-vision
export TOGETHER_API_KEY=sk-...
# TOGETHER_VISION_MODEL=meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo (default)

# (c) Keep vision on Anthropic for now (text is license-clean; vision
# is bounded to a per-figure exposure that's smaller than running
# Sonnet over the full text payload):
export DART_VISION_PROVIDER=anthropic
```

The CLI flags `--dart-provider` and `--dart-vision-provider` override the env vars per invocation. The `AltTextGenerator` constructor raises `ValueError` IMMEDIATELY when the resolved provider is not vision-capable, so a misconfigured local-server-without-vision fails at startup, not mid-PDF.

**Calibration risk acknowledgment.** This wave SHIPS the path; the underlying calibration of vision-quality vs the Anthropic baseline (alt-text accuracy on dense scientific figures, OCR-blended diagrams, math-heavy plates) is operator-side follow-up. Recommended calibration loop, mirroring the `assessment_item` recipe under "Calibration prerequisite for `assessment_item`" above:

1. Pick a representative chapter with figure-heavy content (10-20 figures spanning charts / diagrams / photos / equations).
2. Run DART against the chapter under both routings (license-clean variant vs canonical Anthropic).
3. Compare the per-figure alt-text outputs against a hand-authored reference: assess accessibility-utility (does a screen-reader user get the figure's purpose?), specificity (does it call out the data being illustrated?), and accuracy (no hallucinated values).
4. If the local / together-vision variant fails materially more often (e.g. >2× the hand-edit rate of the Anthropic baseline on the same chapter), keep `DART_VISION_PROVIDER=anthropic` and accept the per-figure ToS hit on alt-text only — the text-mode structure detection at `DART_PROVIDER=local` still ships license-clean.

Until that calibration loop closes for a given course family, courses built under the license-clean DART variant should not promote past `non_certified_archive` on the Wave-3 promotion chain (`lib/governance/course_status.py::derive_course_status`) for vision-quality reasons. The text-mode change is structurally sound (Qwen 2.5 14B has been calibrated for the analogous Trainforge synthesis surface) but the vision surface is a fresh seam.

**Workaround if the calibration shows local vision underperforms.** Pre-convert PDFs through DART on a separate machine that has the Anthropic agreement, archive the resulting HTML, and feed the HTML directly into the textbook-to-course pipeline (skipping the `dart_conversion` phase via `--reuse-objectives` once DART has run). The Anthropic exposure is then bounded to the one-time pre-conversion step and does not leak into the synthesis phase. W-D13 makes this workaround optional rather than mandatory: an operator who's run the calibration and accepts the local-vision quality can route everything in-process.

### Assessment-generator subagent (W-D15) — closed

W-D15 closes the assessment-generator subagent gap. `TRAINFORGE_ASSESSMENT_PROVIDER=local` (or any registered OpenAI-compatible provider) now routes assessment-question authoring through `Trainforge/generators/_assessment_provider.py::AssessmentGeneratorProvider`, mirroring the W-D14 `OutlinerProvider` pattern. The authored questions land in `assessments.json` and feed into the downstream `training_synthesis` instruction-pair / preference-pair surface — so closing this seam was the dominant remaining training-data exposure on the Trainforge assessment surface. The provider's user prompt instructs the LLM to emit an `evidence_quote` field per question per the W-D11 T11.3 grounding contract; the per-call `assessment_generator_call` decision event surfaces the dynamic `evidence_quote_emit_rate` so a post-hoc audit can replay grounding quality.

### Honest scope

This recipe documents a license-clean COURSEWARE / TRAINING-CORPUS run for every dominant code path with the W-D15 wave landed: every Anthropic-defaulted subagent surface that touches training data now has a license-clean provider seam (`COURSEFORGE_PROVIDER` for content-generator, `COURSEPLANNER_PROVIDER` for course-outliner, `TRAINFORGE_ASSESSMENT_PROVIDER` for assessment-generator, `CURRICULUM_ALIGNMENT_PROVIDER` for align_chunks, `DART_PROVIDER` / `DART_VISION_PROVIDER` for DART). Operators training adapters for redistribution should still verify the calibration prerequisites for the affected block-types + DART vision quality and document any per-block Anthropic exposure (e.g. an uncalibrated `assessment_item` block staying on Anthropic per the calibration loop above) in the corpus's audit trail.

---

## See also

- `docs/LICENSING.md` — canonical ToS posture, per-provider terms, per-model license matrix.
- `Courseforge/CLAUDE.md` § "Opt-In Behavior Flags" — full env-var table for the Courseforge two-pass router.
- `Courseforge/config/block_routing.license_clean.yaml` — the sibling YAML this recipe pins via `COURSEFORGE_BLOCK_ROUTING_PATH`.
- `Trainforge/CLAUDE.md` § "Synthesis providers" — same env-var stack from the Trainforge perspective.
- `lib/governance/course_status.py::derive_course_status` — Wave-3 promotion-chain composer; the 5-value `course_status` enum a license-clean run targets.
