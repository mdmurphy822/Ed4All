# License-clean pipeline run — opt-in deployment recipe (W-D11.F)

This file documents how to opt INTO a license-clean Ed4All pipeline run. The Anthropic-defaulted path remains the canonical / backward-compatible default — this recipe is for operators who want a corpus that is ToS-clean from end to end (so the trained SLM adapter can be redistributed without separately negotiated provider agreements).

Cross-reference: `docs/LICENSING.md` is the canonical ToS-posture document. Read it before kicking off a multi-hundred-dispatch run.

---

## TL;DR — the four env vars

| Env var | License-clean value | What it changes |
|---------|---------------------|-----------------|
| `COURSEFORGE_REWRITE_PROVIDER` | `local` | Routes the Courseforge rewrite-tier authoring surface through a local OSS server (Ollama / vLLM) instead of Anthropic. |
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

# Provider routing — the four documented env vars.
export COURSEFORGE_REWRITE_PROVIDER=local
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

The four-env-var recipe above closes the largest training-data exposure paths in the pipeline. Two surfaces remain Anthropic-pinned at the time this recipe ships:

### DART (PDF → HTML conversion)

DART's PDF converter (`DART/pdf_converter/converter.py`) currently routes through Anthropic via the `DART_CLAUDE_MODEL` env var (default `claude-sonnet-4-20250514`). DART HTML output is later ingested as Trainforge training chunks, so this is a real training-data exposure. There is no `DART_PROVIDER=local` flag yet — vision-mode OSS routing via a multimodal local model (Qwen2.5-VL or Llama-3.2-Vision) is the W-D12 / W-D13 wave forthcoming.

Workaround until that wave lands: pre-convert PDFs through DART on a separate machine that has the Anthropic agreement, archive the resulting HTML, and feed the HTML directly into the textbook-to-course pipeline (skipping the dart_conversion phase via `--reuse-objectives` once DART has run). The Anthropic exposure is then bounded to the one-time pre-conversion step and does not leak into the synthesis phase.

### Course-outliner + assessment-generator subagents

Two Claude Code subagents currently dispatch through `ED4ALL_AGENT_DISPATCH=true` with no provider flag:

- `course-outliner` (Courseforge) — synthesizes canonical `TO-NN` / `CO-NN` learning objectives from textbook structure. Routes via `plan_course_structure` MCP tool. The synthesized objectives JSON is consumed by every downstream phase, so this is real training-data exposure (objective text lands in `course.json` and propagates to `chunk.learning_outcome_refs[]`).
- `assessment-generator` (Trainforge) — generates assessment questions + distractors. The questions land in `assessments.json` and feed into `instruction_pair` / `preference_pair` synthesis.

There is no `*_PROVIDER=local` flag for either subagent yet. The W-D14 / W-D15 waves will route both through the same `OpenAICompatibleClient` infrastructure as the synthesis / curriculum-alignment / content-generator surfaces.

Workaround: the `--reuse-objectives` flag on `ed4all run textbook-to-course` lets an operator hand-curate (or pre-generate via a local provider out-of-band) the synthesized objectives JSON and pin the pipeline to it, side-stepping the course-outliner dispatch entirely. No equivalent workaround exists for assessment-generator at this time.

### Honest scope

This recipe documents a license-clean COURSEWARE / TRAINING-CORPUS run for the dominant code paths. It does not yet guarantee a 100% ToS-clean trained-SLM pipeline — the two waves above must land before the trained adapter can be redistributed without per-component ToS analysis. Operators training adapters for redistribution today should bound the Anthropic-pinned surfaces (DART pre-conversion, subagent dispatch) to a separate compliance-reviewed step and document that bounding in the corpus's audit trail.

---

## See also

- `docs/LICENSING.md` — canonical ToS posture, per-provider terms, per-model license matrix.
- `Courseforge/CLAUDE.md` § "Opt-In Behavior Flags" — full env-var table for the Courseforge two-pass router.
- `Courseforge/config/block_routing.license_clean.yaml` — the sibling YAML this recipe pins via `COURSEFORGE_BLOCK_ROUTING_PATH`.
- `Trainforge/CLAUDE.md` § "Synthesis providers" — same env-var stack from the Trainforge perspective.
- `lib/governance/course_status.py::derive_course_status` — Wave-3 promotion-chain composer; the 5-value `course_status` enum a license-clean run targets.
