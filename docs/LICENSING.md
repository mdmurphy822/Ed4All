# Licensing & ToS Posture

This document is the canonical reference for Ed4All's licensing and Terms-of-Service posture across the tools and LLM models the project uses. Other docs (`CLAUDE.md`, `AGENTS.md`, `Trainforge/CLAUDE.md`) link here rather than duplicating, so this is the only file that should change when a provider's ToS or a model's license changes.

The repo-root `NOTICE` file is the short attribution summary (Apache-2.0 dependency notices + synthesis/embedding model attributions, e.g. Llama 3.3's "Built with Llama"). It deliberately points back here for the long-form posture; this file is the source of truth and `NOTICE` is the redistribution-facing acknowledgment. Keep them consistent: a new attributable dependency or model gets a `NOTICE` line and (when it selects a synthesis/embedding backend) a row in the tables below.

IMS Common Cartridge validation also consumes operator-installed XML Schema
files published by IMS Global Learning Consortium/1EdTech and W3C. Ed4All does
not distribute those third-party payloads. Operators obtain them from their
official locations, preserve their embedded copyright, IPR, license, and
distribution notices, and install them according to
`Courseforge/schemas/imscc/README.md`. This records provenance only; the
upstream notices remain the authoritative terms.

---

## Purpose

Ed4All has an asymmetry that contributors and operators must understand before running any synthesis pass:

- **Orchestration / development tools** (Claude Code, OpenAI Codex) read files, run scripts, dispatch shell commands, and generate code that gets committed. Their ToS restricts using outputs to train derivative models, but that restriction does not bind Ed4All because these tools never produce training data — they produce code, summaries, and shell invocations.
- **Synthesis providers** (Anthropic, Together AI, local OSS models) generate the paraphrased instruction / preference pairs that become training data for course-pinned SLM adapters. Their ToS layer is load-bearing because the trained model is a derivative work of those outputs. License-clean here means clean all the way through.

The two cases need different defensive postures. This file documents both.

---

## Tooling (no training-data exposure)

The choice of development tool has zero effect on the trained SLM's licensing — these tools never generate training data on this codebase. They drive scripts, edit files, and produce code-review-quality output. ToS restrictions on training-data routing are a non-issue because that routing does not happen here.

### Claude Code (Anthropic CLI)

- **Role:** Primary development assistant. Reads `CLAUDE.md`, dispatches subagents through MCP, edits source, runs tests.
- **ToS layer:** Anthropic Consumer Terms (Pro / Max sessions) — https://www.anthropic.com/legal/consumer-terms — or Anthropic Commercial Terms (API access) — https://www.anthropic.com/legal/commercial-terms.
- **What's permitted:** Generating code, prose, configuration, and tests for the project. Committing those outputs to the repository.
- **What's restricted:** Routing Claude outputs into training data for a competing or derivative AI model. Anthropic's ToS prohibits this explicitly.
- **Why this is fine for Ed4All — with one caveat:** Claude Code does not generate training data on this project, EXCEPT through the Courseforge content-generator subagent under `ED4ALL_AGENT_DISPATCH=true` when `COURSEFORGE_PROVIDER` is unset. In that configuration, the subagent's Claude Code session authors HTML prose that Trainforge ingests as training chunks — i.e. it touches training data. Setting `COURSEFORGE_PROVIDER=local` (or `together`) routes the same surface through a license-clean provider. See the Synthesis providers table below.

### OpenAI Codex (OpenAI CLI)

- **Role:** Alternate development assistant configured at `~/.codex/config.toml`. Runs scripts, summarizes pilot reports, orchestrates local model servers. See `AGENTS.md` for Codex-specific guidance.
- **ToS layer:** OpenAI Services Terms — https://openai.com/policies/services-terms/ — and OpenAI Business Terms — https://openai.com/policies/business-terms/.
- **What's permitted:** Same as Claude Code — code, prose, tooling, configuration.
- **What's restricted:** Same shape as Anthropic's — using Codex outputs to train a competing model is not permitted.
- **Why this is fine for Ed4All:** Codex's role is orchestration. The local Qwen / Together-hosted OSS model produces training data; Codex tells the shell to start the model server, runs `pilot_synthesis.py`, and summarizes the report. Codex output never lands in `instruction_pairs.jsonl`.

The single line to internalize: **the dev tool you use to write Ed4All code has no bearing on what's in the trained SLM's training corpus.** The two surfaces are isolated by design.

### Private operator material

Source corpora, generated course and training artifacts, model caches, endpoint
addresses, credentials, and run records are always operator-private and are not
tracked or distributed with Ed4All. Internal model artifacts are not shipped.
Packaging or redistributing derived weights, third-party source material, or
system-provided runtimes requires a separate operator and legal review.

---

## Synthesis providers (training-data exposure)

These are the providers that actually produce paraphrased training pairs. Each row's ToS layer + underlying model license decide whether the resulting corpus can train a derivative SLM without legal exposure.

| `--provider` flag | Default model | Model license | ToS layer | Training-data permitted | Recommended use |
|-------------------|---------------|---------------|-----------|--------------------------|-----------------|
| `anthropic` | `claude-sonnet-4-6` | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward compat / non-training synthesis only |
| `claude_session` | Active Claude Code session | Anthropic proprietary | Anthropic Consumer Terms (Pro/Max) | **No** | Backward compat only — consumer terms even more restrictive |
| `COURSEFORGE_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `ANTHROPIC_SYNTHESIS_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward-compat default for Courseforge content-generator; not recommended for training data |
| `COURSEFORGE_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for Courseforge content authoring |
| `COURSEFORGE_PROVIDER=local` | `nemotron-3-nano-30b-a3b` (via `LOCAL_SYNTHESIS_MODEL`) | NVIDIA Open Model License | N/A (your hardware) | **Yes** | **Recommended for ToS-clean Courseforge content** |
| `COURSEFORGE_OUTLINE_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `ANTHROPIC_SYNTHESIS_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Two-pass outline tier; not recommended for training data |
| `COURSEFORGE_OUTLINE_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the two-pass outline tier |
| `COURSEFORGE_OUTLINE_PROVIDER=local` | `nemotron-3-nano-30b-a3b` (via `LOCAL_SYNTHESIS_MODEL`) | NVIDIA Open Model License | N/A (your hardware) | **Yes** | **Recommended for ToS-clean outline drafting** — outline drafts are re-ingested by Trainforge as chunks, so this surface IS training-data exposure |
| `COURSEFORGE_OUTLINE_PROVIDER=<registry>` | per the `_OPENAI_COMPATIBLE_PROVIDERS` registry entry (`groq` / `fireworks` / `deepseek` / `nvidia` / future additions) | per registry entry's underlying model license | per provider's ToS | per provider | Dynamic provider — the outline tier accepts the full endpoint-registry superset; adding a registry entry surfaces here with no code edit (per-seat posture carried by the `config/endpoints.yaml` row / model choice) |
| `COURSEFORGE_REWRITE_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `ANTHROPIC_SYNTHESIS_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward-compat default for the two-pass rewrite tier; not recommended for training data |
| `COURSEFORGE_REWRITE_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the two-pass rewrite (authoring) tier |
| `COURSEFORGE_REWRITE_PROVIDER=local` | `nemotron-3-nano-30b-a3b` (via `LOCAL_SYNTHESIS_MODEL`) | NVIDIA Open Model License | N/A (your hardware) | **Yes** | **Recommended for ToS-clean rewrite authoring** — rewrite-tier published HTML is re-ingested by Trainforge as chunks, so this surface IS training-data exposure |
| `COURSEFORGE_REWRITE_PROVIDER=<registry>` | per the `_OPENAI_COMPATIBLE_PROVIDERS` registry entry (`groq` / `fireworks` / `deepseek` / `nvidia` / future additions) | per registry entry's underlying model license | per provider's ToS | per provider | Dynamic provider — the rewrite tier accepts the full endpoint-registry superset (the legacy `openai_compatible` alias collapses to `local` at constructor entry); adding a registry entry surfaces here with no code edit |
| `COURSEPLANNER_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `ANTHROPIC_SYNTHESIS_MODEL` / `COURSEPLANNER_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward-compat default for `OutlinerProvider`; not recommended for training data |
| `COURSEPLANNER_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL` / `COURSEPLANNER_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the Courseforge course-outliner surface |
| `COURSEPLANNER_PROVIDER=local` | `nemotron-3-nano-30b-a3b` (via `LOCAL_SYNTHESIS_MODEL` / `COURSEPLANNER_MODEL`) | NVIDIA Open Model License | N/A (your hardware) | **Yes** | **Recommended for ToS-clean course planning** |
| `COURSEPLANNER_PROVIDER=<registry>` | per the `_OPENAI_COMPATIBLE_PROVIDERS` registry entry (`groq` / `fireworks` / `deepseek` / future additions) | per registry entry's underlying model license | per provider's ToS | per provider | Dynamic provider — adding a new entry to the registry surfaces here without a code edit |
| `TEXTBOOK_SYNTHESIS_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `TEXTBOOK_SYNTHESIS_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward-compat default for `TextbookSynthesisProvider`; not recommended for training data |
| `TEXTBOOK_SYNTHESIS_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TEXTBOOK_SYNTHESIS_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the three-stage textbook synthesis surface |
| `TEXTBOOK_SYNTHESIS_PROVIDER=local` | `nemotron-3-nano-30b-a3b` (via `LOCAL_SYNTHESIS_MODEL` / `TEXTBOOK_SYNTHESIS_MODEL`) | NVIDIA Open Model License | N/A (your hardware) | **Yes** | **Recommended for ToS-clean three-stage textbook synthesis** — the domain-concept vocabulary propagates into chunk `concept_tags[]` and the synthesized objectives propagate into every downstream chunk's `learning_outcome_refs[]`, so this surface IS training-data exposure |
| `TEXTBOOK_SYNTHESIS_PROVIDER=<registry>` | per the `_OPENAI_COMPATIBLE_PROVIDERS` registry entry (`groq` / `fireworks` / `deepseek` / future additions) | per registry entry's underlying model license | per provider's ToS | per provider | Dynamic provider — adding a new entry to the registry surfaces here without a code edit |
| `TRAINFORGE_ASSESSMENT_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `ANTHROPIC_SYNTHESIS_MODEL` / `TRAINFORGE_ASSESSMENT_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward-compat default for `AssessmentGeneratorProvider`; not recommended for training data |
| `TRAINFORGE_ASSESSMENT_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL` / `TRAINFORGE_ASSESSMENT_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the Trainforge assessment-generator surface |
| `TRAINFORGE_ASSESSMENT_PROVIDER=local` | `nemotron-3-nano-30b-a3b` (via `LOCAL_SYNTHESIS_MODEL` / `TRAINFORGE_ASSESSMENT_MODEL`) | NVIDIA Open Model License | N/A (your hardware) | **Yes** | **Recommended for ToS-clean assessment generation.** Authored questions feed downstream `training_synthesis`, so this surface IS training-data exposure. |
| `TRAINFORGE_ASSESSMENT_PROVIDER=<registry>` | per the `_OPENAI_COMPATIBLE_PROVIDERS` registry entry (`groq` / `fireworks` / `deepseek` / future additions) | per registry entry's underlying model license | per provider's ToS | per provider | Dynamic provider — adding a new entry to the registry surfaces here without a code edit |
| `TRAINFORGE_SYNTHESIS_PROVIDER=anthropic` | n/a — path removed | Anthropic proprietary | Anthropic Commercial Terms | **No** | **Unavailable.** The Anthropic-SDK training-pair synthesis path is not shipped. `run_synthesis` fails closed with `SynthesisLicensingError` **UNCONDITIONALLY** on `provider="anthropic"`; `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true` does not unlock it. Use `local` or an operator-approved hosted OSS provider. |
| `TRAINFORGE_SYNTHESIS_PROVIDER=claude_session` | Active Claude Code session | Anthropic proprietary | Anthropic Consumer Terms (Pro/Max) | **No** | **Gated opt-in** — a SEPARATE Claude-Code-session route (NOT the removed SDK path). `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true` acknowledgment required; even-more-restricted consumer-terms path. Not recommended for training data. |
| `TRAINFORGE_SYNTHESIS_PROVIDER` (unset, pipeline run) | n/a — resolves to a provider | per resolved provider | per resolved provider | **per resolved provider** | **License-clean by default.** On a `textbook_to_course` / `course_generation` run, the workflow runner defaults this environment variable to `LLM_PROVIDER`, then `local`; an explicit value wins. |
| `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true` | n/a — acknowledgment gate | n/a | n/a | n/a | **Acknowledgment flag, not a provider selector.** Gates only the separate `claude_session` training-pair route; the unavailable `anthropic` SDK path still fails closed unconditionally. Without this acknowledgment, `run_synthesis` rejects `claude_session` before dispatch. Set it only if you hold a separate agreement permitting derivative training. |
| `TRAINFORGE_SYNTHESIS_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the Trainforge training-synthesizer surface |
| `TRAINFORGE_SYNTHESIS_PROVIDER=local` | `nemotron-3-nano-30b-a3b` (via canonical `LOCAL_SYNTHESIS_MODEL`) | NVIDIA Open Model License | N/A (your hardware) | **Yes** | **Recommended for ToS-clean training-pair synthesis.** The emitted instruction / preference pairs ARE the canonical SLM training corpus consumed by `Trainforge.train_course`, so restricted development-tool output must never author them. The provider short-circuit and licensing gate in `Trainforge/synthesis/synthesize_training.py` enforce that boundary. Staged production synthesis requires an explicit `LOCAL_SYNTHESIS_MODEL` and exact `/v1/models` identity match before dispatch; `TRAINFORGE_SYNTHESIS_MODEL` cannot substitute for it and a conflicting value fails closed. |
| `TRAINFORGE_STAGED_SYNTHESIS_V4=true` | Inherits `TRAINFORGE_SYNTHESIS_PROVIDER` and its resolved model | Inherits the selected provider/model row | Inherits the selected provider row | **Per selected provider** | Prompt-workflow selector only: evidence plan, SFT realization, and independent DPO chosen/rejected realization calls use the same configured synthesis seat. It does not select or substitute a provider/model and does not relax the provider licensing gate. |
| `--synthesis-contract micro-v1` / `TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1=true` | Inherits `TRAINFORGE_SYNTHESIS_PROVIDER` and its resolved model | Inherits the selected provider/model row | Inherits the selected provider row | **Per selected provider** | Versioned prompt-workflow selector for `ed4all.staged-synthesis-micro.v1`; it divides task, claim, realization, misconception, and rejected-answer work into audited micro calls on the same configured synthesis seat. Either entry selects it: the CLI selector resolves to the environment flag in `main()` before any provider is constructed (a conflicting ambient value exits non-zero rather than being overridden), and the environment flag on its own is the process-level routing seam used by the pipeline phase. Neither selects or substitutes a provider/model. The existing licensing gate still rejects restricted providers before dispatch, so the recommended license-clean `local` / permitted OSS provider posture is unchanged. Default off preserves legacy routing. A direct-library micro+V4 conflict fails closed; the explicit CLI resolves precedence by setting only its selected contract. Stop/resume journaling, terminal publication authority, and post-synthesis gates do not alter the provider or training-data license. |
| `together` (Llama) | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Llama 3.3 Community License | Together AI ToS | **Yes** | Hosted OSS fallback |
| `together` (Qwen) | `Qwen/Qwen2.5-72B-Instruct-Turbo` | Qwen License Agreement | Together AI ToS | **Yes** | Hosted OSS fallback |
| `together` (DeepSeek) | `deepseek-ai/DeepSeek-V3` | DeepSeek License | Together AI ToS | **Yes** (per DeepSeek License) | Hosted OSS fallback |
| `local` (Nemotron Nano) | `nemotron-3-nano-30b-a3b` (served name for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, local snapshot `cbd3fa9f933d55ef16a84236559f4ee2a0526848`) | **NVIDIA Nemotron Open Model License** (the pinned model card declares `license_name: nvidia-nemotron-open-model-license`) | N/A (your hardware) | **Yes** | **Recommended default** for license-clean corpora; training-on-outputs and adapter derivatives remain subject to the pinned Nemotron terms and preflight guard below |
| `local` (Qwen 14B) | `qwen2.5:14b-instruct-q4_K_M` | Apache 2.0 | N/A | **Yes** | Local OSS option |
| `local` (Qwen 32B) | `qwen2.5:32b-instruct-q4_K_M` | Apache 2.0 | N/A | **Yes** | Local OSS option |
| `ED4ALL_CAMPAIGN_BASE_MODEL` | `nemotron3-nano-30b` (BF16 LoRA base, not a synthesis provider) | NVIDIA Nemotron Open Model License (pinned identity — `assert_nemotron_pin` fails the build on identity drift) | N/A (your hardware, HF-offline pre-seeded snapshot) | n/a — this flag selects the model being TRAINED, not a pair-authoring teacher | Base-model selector (`lib/assistant/campaign_tools.resolve_campaign_base_model`); the value must resolve in `Trainforge/training/base_models.py::BaseModelRegistry` — unknown name is a loud error, never a fallback model. Teacher-side licensing for the pairs it trains on is governed by the SFT teacher roster below. |
| `ed4all run --base-model <name>` (CLI flag) | none — unset leaves `ED4ALL_CAMPAIGN_BASE_MODEL` > `nemotron3-nano-30b` in charge | Whatever the NAMED registry entry carries. Every `BaseModelRegistry` entry is a license-vetted base: the Nemotron entry stays under its pinned identity (`assert_nemotron_pin`), the Qwen 2.5 / SmolLM2 / Llama 3.2 / Phi 3.5 entries under their own upstream terms. The flag CANNOT introduce an unvetted base — an unrecognized name exits 2 with the supported list. | N/A (your hardware) | n/a — selects the model being TRAINED, not a pair-authoring teacher | Operator-facing sibling of the env var above and the HIGHEST-precedence input to the same resolution chain (`--base-model` > `ED4ALL_CAMPAIGN_BASE_MODEL` > registry default), consumed by `config/workflows.yaml::training`'s `inputs_from: base_model <- workflow_params.base_model` route. Validated at CLI parse time through the SAME `BaseModelRegistry.resolve` the `run_training` handler validates against — one registry, one supported-set message, no second allowlist. Governs both `ed4all run trainforge_train` and the in-build `--with-training` tail; re-pinnable on `--resume`. Teacher-side licensing for the pairs it trains on is governed by the SFT teacher roster below. |
| `local` (Qwen 72B) | `qwen2.5:72b-instruct-q4_K_M` | Qwen License Agreement | N/A | **Yes** (outputs unrestricted at any scale) | Local OSS option |
| `local` (Llama 70B) | `llama3.3:70b-instruct-q4_K_M` | Llama 3.3 Community License | N/A | **Yes** (with attribution) | Strong instruction following |
| `local` (Mistral 24B) | `mistral-small:24b-instruct-q4_K_M` | Apache 2.0 | N/A | **Yes** | Local OSS option |
| `local` (Phi-3.5 mini) | `phi3.5:3.8b-mini-instruct-q4_K_M` | MIT | N/A | **Yes** | Smallest OSS option |
| `COURSEFORGE_BLOCK_ROUTING_PATH=Courseforge/config/block_routing.license_clean.yaml` | n/a (legacy router-level YAML; its per-tier rows still select local Qwen models) | Per selected tier (the checked-in Qwen rows are Apache-2.0) | N/A on local hardware | **Yes** | Optional legacy two-pass override, not the canonical shared-seat recommendation. The canonical local curriculum, Courseforge, and Trainforge path uses `LOCAL_SYNTHESIS_MODEL=nemotron-3-nano-30b-a3b` on the strict TRT-LLM/vLLM seat and is governed by the **NVIDIA Nemotron Open Model License**. |
| `COURSEFORGE_BLOCK_ROUTING_PATH=Courseforge/config/block_routing.nvidia_large.yaml` (key from `NVIDIA_API_KEY`; base_url/model via `NVIDIA_BASE_URL` / `NVIDIA_LARGE_MODEL`) | `meta/llama-3.3-70b-instruct` (large tier only; small/medium stay local Qwen 7B/14B) | NVIDIA model / catalog license | NVIDIA hosted-API ToS | **N/A — generates COURSE CONTENT (product), not training-data corpus** | Router-level YAML sibling of the license-clean variant. The small + medium capability tiers are byte-identical local 7B/14B Qwen (Apache 2.0, loopback-only); ONLY the rewrite `large` / escalation tier routes to NVIDIA's hosted OpenAI-compatible inference API. **This row is content-generation, not synthesis:** the NVIDIA-authored HTML is published product. If that HTML is later re-ingested by Trainforge as training chunks, the standard content→training-data caveat applies (see "Courseforge content-generator shares the synthesis provider stack" below) — operators wanting a fully ToS-clean training corpus should keep this surface on a license-clean local/Together provider. |
| `SEMANTIK_SPECIALIST_PROVIDER=<endpoint>` + `SEMANTIK_SPECIALIST_MODEL` / `SEMANTIK_STRUCTURE_REVIEW_MODEL` (key from `NVIDIA_API_KEY`; base via `NVIDIA_BASE_URL`; model resolves through `NVIDIA_LARGE_MODEL`) | `meta/llama-3.3-70b-instruct` (specialist generation and optional structure review) | NVIDIA model / catalog license (Llama 3.3 Community License underneath) | NVIDIA hosted-API ToS | **N/A — generates structured product content, not training pairs** | The specialist provider defaults to a self-hosted `local` seat. Selecting a non-local endpoint routes specialist generation and optional structure review through that provider and introduces its ToS layer. Later Trainforge ingestion creates content-to-training-data exposure. |
| `SEMANTIK_VLM_PROVIDER` / `SEMANTIK_VLM_MODEL` (base via `SEMANTIK_VLM_BASE_URL`, default an operator-configured local endpoint; key from `SEMANTIK_VLM_API_KEY` — none for the local/loopback seat) | `qwen2.5vl:7b` = **Qwen2.5-VL-7B-Instruct** (optional VLM extraction source, gated by `SEMANTIK_VLM_EXTRACT`, default off) | **Apache-2.0** (the 7B VL is Apache-2.0 — NOT the Qwen-licensed 72B VL; pin the 7B) | N/A — local-by-default (loopback ollama; a non-local seat is opt-in and carries that endpoint's ToS) | **N/A — generates STRUCTURED CONTENT (product), not training-data corpus** | SemantiK's opt-in LOCAL VLM extraction seat: a per-page image → Qwen2.5-VL markdown transcription fused as a fourth extraction source (provider-agnostic OpenAI-compatible, mirroring `SEMANTIK_SPECIALIST_*` — a new provider is env config, never a subclass; the same seat serves the local ollama GPU). Default OFF and local (`qwen2.5vl:7b` on loopback ollama), so the default posture stays fully offline. **This row is content-generation, not synthesis:** the transcribed structure is published product. If that output is later re-ingested by Trainforge as training chunks, the standard content→training-data caveat applies — keep the seat local for a fully ToS-clean corpus. MiniCPM-V is excluded on license. **`SEMANTIK_REASONING_QC` (the optional reasoning-QC pass, default off) rides THIS already-licensed VLM seat** (the Apache-2.0 `qwen2.5vl:7b` default), so it needs no separate row — UNLESS its optional `SEMANTIK_REASONING_QC_MODEL` override selects a DISTINCT model/endpoint, which then requires its own Synthesis-providers row per the maintenance contract below. Flag detail: `docs/operations/behavior-flags-semantik.md`. |
| `SEMANTIK_SEMANTIC_SUBCLASS` (seat via `LOCAL_SYNTHESIS_BASE_URL` / `LOCAL_SYNTHESIS_MODEL` / `LOCAL_SYNTHESIS_API_KEY`) | `nemotron-3-nano-30b-a3b` (the existing local reviewer seat; composite-unit subclassifier, `lib/semantik/subclassifier.py`, gated by `SEMANTIK_SEMANTIC_SUBCLASS`, default off) | **NVIDIA Nemotron Open Model License** for the canonical Nano BF16 default; any explicit `LOCAL_SYNTHESIS_MODEL` override carries that model's own license | N/A — local-by-default (loopback strict OpenAI-compatible TRT-LLM/vLLM server; a non-local override carries that endpoint's ToS) | **N/A — emits a metadata LABEL (`data-semantik-subclass` / chunk `unit_subclass`), not prose training-data corpus** | SemantiK's optional composite-unit subclass pass reuses the SAME license-clean local reviewer seat as `SEMANTIK_SPECIALIST_PROVIDER=local` / the Trainforge `local` synthesis seat (env-resolved via `LOCAL_SYNTHESIS_*`, no hardcoded endpoint). It classifies each already-rendered composite unit into a kebab-case subclass label added payload-only to the published HTML; the label rides into chunks as the additive `unit_subclass` metadata field (never prose text, never a chunk-text/id change). Default OFF → no dispatch. Keep the seat local for a fully ToS-clean corpus. Flag detail: `docs/operations/behavior-flags-semantik.md`. |

| `SEMANTIK_GLMOCR_LANE` (`SEMANTIK_GLMOCR_BASE_URL`; model via `SEMANTIK_GLMOCR_MODEL`, default `glm-ocr`) | `glm-ocr` = **GLM-OCR** on an operator-configured local OpenAI-compatible endpoint; SDK `glmocr` 0.1.5 with PP-DocLayoutV3 | **MIT** model weights; **Apache-2.0** SDK and PP-DocLayoutV3 | N/A for self-hosted `local`; a non-local endpoint carries its provider's ToS | **N/A — generates accessible product content, not training pairs** | Optional, default-off whole-document extraction lane. It uses the external Poppler `pdftoppm` runtime for page rendering, then performs layout analysis, OCR, provenance capture, and accessible-HTML rendering. Later Trainforge ingestion creates content-to-training-data exposure. |
| `SEMANTIK_HEADING_JUDGE_MODEL` (`SEMANTIK_HEADING_JUDGE_BASE_URL`) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` on an operator-configured self-hosted endpoint | **NVIDIA Nemotron Open Model License** | N/A for self-hosted `local`; a hosted `nvidia` endpoint carries NVIDIA hosted-API ToS | **N/A — emits heading metadata for accessible product content** | Default-on GLM-OCR enrichment judge. Self-hosted `local` provenance remains distinct from the license-restricted hosted `nvidia` provider. A model or provider override requires its own licensing review before downstream training-data use. |
| `SEMANTIK_ALTTEXT_PROVIDER=qwen30` (`SEMANTIK_ALTTEXT_BASE_URL`; model via `SEMANTIK_ALTTEXT_MODEL`, default `qwen3-vl-30b`) | **Qwen3-VL-30B-A3B-Instruct-FP8** | **Apache-2.0** | N/A for self-hosted `local`; a non-local endpoint carries its provider's ToS | **N/A — emits accessibility product content, not training pairs** | Optional, default-off figure alt-text and caption enrichment. Endpoint addresses and credentials remain operator-private. Later Trainforge ingestion creates content-to-training-data exposure. |
| `spark-super` | `nemotron-3-super-120b-a12b` (via `SPARK_SUPER_MODEL`) | NVIDIA Open Model License (permissive — commercial use, modification, redistribution, and TRAINING ON OUTPUTS all explicitly permitted; NVIDIA disclaims ownership of outputs; the single restriction — may not be used to build a competing foundation-model-training service — does NOT bind Ed4All's course-pinned SLM training) | N/A (your hardware) | **Yes** | **Local self-hosted Nemotron — quality synthesis tier.** NOT the hosted `nvidia` API seat; these are local weights and are training-clean (see the distinction note below the table). |
| `spark-nano` | `nemotron-3-nano-30b-a3b` (via `SPARK_NANO_MODEL`) | NVIDIA Open Model License (permissive — same terms as the `spark-super` row: training-on-outputs permitted, NVIDIA disclaims output ownership, sole restriction is the competing-foundation-model-training-service bar which does NOT bind Ed4All's course-pinned SLM training) | N/A (your hardware) | **Yes** | **Local self-hosted Nemotron — high-volume / interactive tier.** NOT the hosted `nvidia` API seat; local weights, training-clean. |
| `SEMANTIK_VLM_MODEL=nemotron-3-nano-omni-30b-a3b` | `nemotron-3-nano-omni-30b-a3b` (via `SEMANTIK_VLM_MODEL`) | NVIDIA Open Model License (permissive — same terms as the `spark-super` row: training-on-outputs permitted, NVIDIA disclaims output ownership, sole restriction is the competing-foundation-model-training-service bar which does NOT bind Ed4All's course-pinned SLM training) | N/A (your hardware) | **Yes** | **Local self-hosted Nemotron-Omni — SemantiK document VLM** (per-page image → markdown transcription source seat; the local Nemotron alternative to the default `qwen2.5vl:7b` VLM row above). NOT the hosted `nvidia` API seat; local weights, training-clean. |

**Local self-hosted Nemotron ≠ the hosted `nvidia` API seat.** The three rows immediately above are self-hosted Nemotron 3 weights running on operator-managed hardware. Under the NVIDIA Open Model License their outputs are training-clean, so they are deliberately **NOT** in the license-restricted synthesis set (`MCP/core/workflow_runner.py::_LICENSE_RESTRICTED_SYNTHESIS` / `lib/diagnostics/provider.py::_LICENSE_RESTRICTED`, both `{"anthropic", "nvidia"}`) — those constants key on the hosted-provider name `nvidia`, and the self-hosted seats carry `local` provenance. The hosted `nvidia` provider (the `NVIDIA_*` cloud-API seat used by `block_routing.nvidia_large.yaml` / `SEMANTIK_SPECIALIST_PROVIDER`) **stays license-restricted and gated OFF for training data**; its exposure is the hosted-API ToS layer, not the underlying model license. Do not conflate the two: `nvidia` (hosted API, restricted) and self-hosted Super, Nano, and Nemotron-Omni seats (self-hosted weights, training-clean) are distinct seats.

---

## SFT teacher roster (course-pinned 1.5B adapter)

This section governs which model **outputs** may become instruction / preference training pairs for the commercially-shipped course-pinned SLM adapter (Qwen2.5-1.5B + LoRA), and which model **weights** may be ingested as a base checkpoint. It is the prose half of the machine-readable roster at `lib/licensing/teacher_roster.py`; the two must stay consistent (maintenance contract at the bottom of this file).

**Machine-readable source of truth + build invariants.** `lib/licensing/teacher_roster.py` encodes every row below as a `LicenseRecord` (`{name, license_spdx, license_url, verdict, obligations[], commercial_use}`) and exposes three fail-closed guards, all wired into the training preflight (`Trainforge/training/runner.py::_assert_licensing_preflight`):

1. **Export-time teacher filter** — `assert_export_licenses(pairs)` refuses to export/train a corpus if **any** pair's teacher is `barred`, unregistered, or claude/anthropic-tagged, naming the offending pair + teacher. A pair carrying no teacher signal at all (legacy shape) or a license-clean `local` / `together` provenance passes (registry-defaults-byte-identical).
2. **Per-checkpoint LICENSE assertion at ingest** — `assert_checkpoint_license(model, role="base_model")` verifies the *actual* base-weight license (Qwen 3B/72B ≠ 7-32B; GLM-4-9B ≠ 4.5; DeepSeek distills inherit their base), barring non-commercial weights for a shipped adapter.
3. **Nemotron license-pin / FAIL-BUILD-ON-RE-PIN guard** — see the Nemotron subsection below.

Per-pair provenance: synthesis stamps every pair with `generating_seat` (model id) + a `license` tag (`stamp_pair_license`), additive on the pairs' `additionalProperties:true` schemas — the coarse closed-enum `provider` field is untouched. The registry accessor `MCP/orchestrator/llm_backend.py::license_metadata_for_provider` surfaces the same posture per endpoint (inline YAML `license_*` fields win; else the roster is consulted by the endpoint's default model / provenance).

**DERIVED pairs inherit, they do not introduce, a teacher.** One pair class is not generated by a fresh model call: the reject-mined DPO negatives behind `TRAINFORGE_DPO_MINE_REJECTS` (default off) re-use two completions an already-licensed seat produced earlier in the SAME run — the accepted instruction pair as `chosen`, a rejected one for the same chunk/objective as `rejected`. No provider, model, or backend is selected, so the flag correctly carries **no Synthesis-providers row**; the licensing question it does raise is teacher attribution, and it is answered by the same `stamp_pair_license` call every other pair goes through, on the reusing row's own teacher. A mined row whose teacher classifies barred / unregistered / claude-tagged is **dropped at mining time** rather than emitted, because `assert_export_licenses` fail-closes the whole training run on such a teacher and a derived row must never be the thing that bricks a multi-hour build. Any future generator that DERIVES rows from already-generated text inherits this rule: stamp from the source row's teacher, drop rather than emit on a non-clean verdict, and no new provider row.

| Model | License (SPDX / ref) | Verdict | Obligations | Recommended role |
|---|---|---|---|---|
| **Qwen2.5-7B / 14B / 32B** | Apache-2.0 | **SAFE** | none | Drafter (primary) |
| **Qwen2.5-72B** | Qwen License | **CONDITIONAL** | Emit *"Built with Qwen" / "Improved using Qwen"* in shipped docs; flag 100M-MAU trigger | Drafter (capability tier) |
| **Qwen2.5-3B** | Qwen RESEARCH | **BARRED** | non-commercial only | — |
| **Nemotron-3-Super-120B** | NVIDIA Nemotron OML (Dec 15 2025) | **SAFE (Case B)** | none on the shipped output-trained adapter; voluntary attribution | Drafter / verifier-judge (high tier) |
| **Mistral-NeMo-12B, Mixtral-8x7B/8x22B, Mistral-Small-3-24B, Mistral-7B** | Apache-2.0 | **SAFE** | none | Drafter / paraphraser |
| **Mistral Large / Medium / Pixtral / Ministral / Codestral** | MRL / MNPL | **BARRED** | non-commercial | — |
| **Phi-3 / Phi-4 (mini)** | MIT | **SAFE** | retain MIT notice on weights only | Paraphraser / verifier |
| **GLM-4.5 / 4.6 / GLM-OCR** | MIT (verify per-repo; legacy GLM-4-9B is custom) | **SAFE** | retain MIT notice | Drafter |
| **OLMo-2-7B / 13B / 32B** | Apache-2.0 (data ODC-BY) | **SAFE** — most-defensible provenance | none | Anchor / verifier-judge |
| **DeepSeek-V3-0324 / R1** | MIT (distillation explicitly permitted) | **SAFE** | avoid the *Llama*-based R1-distill checkpoints | Drafter (frontier tier) |
| **Llama 3.x family** | Llama Community | **BARRED (as teacher)** | forces leading *"Llama-"* name on the adapter + AUP flow-down + 700M-MAU. NB: commercial use of the *weights* is permitted (with attribution), so a Llama *base checkpoint* is ingest-allowed — barred only as a teacher, whose outputs force the naming flow-down onto the corpus | — (teacher only if owner accepts the prefix) |
| **Gemma** | Gemma Terms | **CONDITIONAL / investigate** | PUP flow-down; internal Output-vs-Model-Derivative interpretation contradiction | defer — capability already covered |

Recommendation for a license-clean corpus: **Qwen2.5-32B (Apache-2.0) as the sole LLM drafting teacher**, with deterministic templating from course artifacts as the primary source. Nemotron is a SAFE Case-B alternative once the pin below is honored.

### NVIDIA Nemotron Open Model License (Dec 15 2025) — SAFE Case-B + license pin

The shipped adapter is **Case B**: a Qwen2.5-1.5B adapter trained on Nemotron-drafted pairs is NOT a *Derivative Work of the Work* (the weights) — the license's redistribution conditions trigger only on distributing *"the Work or Derivative Works of the Work,"* and NVIDIA *"does not claim ownership to any outputs generated using the Works."* There is **no distillation / synthetic-data / "train another model" restriction** anywhere in the Nemotron license (verified NOT PRESENT), and NVIDIA itself publishes *Nemotron-Post-Training-Dataset-v1/v2* (synthetic Nemotron outputs) under CC-BY-4.0 *"to train and evaluate"* other models. So no NOTICE and no attribution string is *legally required* on the shipped adapter (we carry attribution voluntarily as provenance hygiene).

**License-pin + FAIL-BUILD-ON-RE-PIN guard.** The roster pins the exact identity string `"NVIDIA Nemotron Open Model License, Dec 15 2025"`. `lib/licensing.assert_nemotron_pin()` (wired into the training preflight) **fails the build** if that identity ever drifts — in particular a re-pin to the general *NVIDIA Open Model License* (Oct 24 2025), which carries the Trustworthy-AI use-restriction, guardrail-circumvention auto-termination, and a *revocable* grant that the Nemotron license lacks. Do not update the pin without re-reading the signed license PDF and obtaining procurement sign-off on any termination / compliance hooks. Use-time hygiene: don't circumvent Nemotron guardrails during synthesis; suing NVIDIA on patent/copyright forfeits the (otherwise irrevocable) grant.

Internal classification: tag Nemotron **"source-available / NVIDIA-permissive (Apache-2.0-shaped)"**, never *"OSI open source"* (the NVIDIA license family fails the Open Source Definition).

### Anthropic / Claude outputs — PROHIBITED as training data (ToS finding, not a preference)

Claude / Anthropic outputs may **not** become training pairs. This is a Terms-of-Service finding under a conservative read, backed by three independent, current, operative sources:

- **Commercial Terms, Restrictions (§D.4, eff. 2025-06-17):** *"Customer may not… (a) access the Services to build a competing product or service, including to train competing AI models… except as expressly approved by Anthropic."* "Competing" is undefined and the only escape is Anthropic's express approval.
- **Usage Policy (eff. 2025-09-15):** prohibits *"Utilization of inputs and outputs to train an AI model (e.g., 'model scraping' or 'model distillation') without prior authorization from Anthropic."* Broader than the competition test — it bars training on outputs at all without prior auth, and names distillation.
- **Help Center, "Can I use my Outputs to train an AI model?" (2026-03-16):** the PERMITTED lane is narrow discriminative/extractive tools (sentiment, categorization, summarization); PROHIBITED is *"General purpose chatbots, Models designed for open-ended text generation, Using Outputs as training targets for models."* **There is no small-model carve-out; size is irrelevant.**

A course-pinned tutor is a generative conversational model — the prohibited "open-ended text generation" bucket — and SFT literally *is* "using Outputs as training targets." Enforcement is active (OpenAI API cutoff Aug 2025; public China-distillation accusations Feb 2026; published detection tooling). The only compliant path to revisit is **express written authorization / a bespoke enterprise agreement obtained from Anthropic BEFORE any synthesis** — never a self-serve reading; same bar the project uses to exclude GPT outputs. The `assert_export_licenses` filter fail-closes on any `provider="anthropic"` / `provider="claude_session"` / claude-tagged `generating_seat` pair. Dev tooling (Claude Code writing pipeline code) remains fine and out of scope (see "Tooling" above).

### Replay set (general-instruction mix, ~15%)

The SFT mix interleaves ~15% license-clean general instruction data to preserve instruction-following / format / refusal. Replay must be 100% permissive — no copyleft, no upstream-model-output terms.

| Dataset | License | Provenance | Status |
|---|---|---|---|
| **OASST1 / OASST2** (primary anchor) | Apache-2.0 | human-generated — cleanest possible | **ACCEPTED** |
| **FLAN-v2** | Apache-2.0 | aggregate of ~1,800 permissive academic sets (aggregate-inherited cleanliness) | **ACCEPTED** |
| **NuminaMath-TIR** | Apache-2.0 | clean math replay | **ACCEPTED** |
| **Dolly-15k** | CC-BY-SA-3.0 | Databricks human-authored | **REJECTED** — whether model weights are a derivative work of CC-BY-SA training data is legally unsettled; the conservative read would force a BY-SA-trained shipped model to itself be CC-BY-SA (share-alike), incompatible with a proprietary commercial adapter. Wrong bet for a procurement-defensible corpus. |
| **Tulu-3-SFT** | ODC-BY (top-level) | contains **No-Robots CC-BY-NC** + GPT-derived rows | **SUBSET-FILTER-ONLY** — admit only affirmatively-permissive rows (OASST/Guanaco Apache, NuminaMath Apache, FLAN-v2); drop CC-BY-NC + GPT-output rows. A dataset-level label is NOT sufficient provenance. |
| **SmolTalk** | Apache-2.0 (label) | masks a **400K Llama-3.1-405B Magpie core** | **SUBSET-FILTER-ONLY** — drop the Llama-Magpie rows; per-row provenance filtering required. |

## Embedding providers (retrieval-index embeddings)

The `ED4ALL_EMBEDDING_*` family ([`docs/operations/behavior-flags.md`](operations/behavior-flags.md))
selects the embedding backend used to build the on-device retrieval vector
index. **These embeddings are NOT training-data synthesis** — they index
existing corpus chunks for nearest-neighbor retrieval; no paraphrased
instruction/preference pairs are produced. The maintenance contract still
requires a row per provider/model-selecting flag, so they are documented here
in their own table. All candidates are Apache-2.0 / MIT; no cloud embedding
provider exists in the registry (Phase IA is local-only — no network call in the
query path, ever).

| Flag/value | Default model | Model license | ToS layer | Notes |
|------------|---------------|---------------|-----------|-------|
| `ED4ALL_EMBEDDING_PROVIDER=st` | `BAAI/bge-large-en-v1.5` | MIT | N/A (local in-process) | retrieval-index embeddings; in-process `sentence-transformers`; not training-data synthesis; current benchmark-selected default |
| `ED4ALL_EMBEDDING_PROVIDER=local-openai` | `nomic-embed-text` (Ollama) | Apache 2.0 | N/A (your hardware) | OpenAI-compatible `/v1/embeddings` against a local server (Ollama / vLLM / llama.cpp) |
| `ED4ALL_EMBEDDING_PROVIDER=fake` | deterministic hash vectors | N/A | N/A | test-only; production index load refused without `ED4ALL_EMBEDDING_ALLOW_FAKE` |
| benchmark candidate | `BAAI/bge-large-en-v1.5` | MIT | N/A (local in-process) | strong en-only baseline — selected benchmark candidate (hybrid-rrf winner; now the `st` default above) |
| benchmark candidate | `Alibaba-NLP/gte-large-en-v1.5` | Apache 2.0 | N/A (local in-process) | requires `trust_remote_code=True` (executes HF model code) — droppable candidate |
| benchmark candidate | `nomic-ai/nomic-embed-text-v1.5` | Apache 2.0 | N/A (local in-process) | also Ollama-servable; needs `search_query:` / `search_document:` task prefixes |
| benchmark candidate | `Qwen/Qwen3-Embedding-0.6B` | Apache 2.0 | N/A (local in-process) | documented stretch candidate |
| smoke baseline | `sentence-transformers/all-MiniLM-L6-v2` | Apache 2.0 | N/A (local in-process) | already cached; CI real-model smoke + floor baseline only |

The default model pin (`BAAI/bge-large-en-v1.5`, selected from the
4-model benchmark) remains re-pinnable from future benchmark results — a
one-line registry change in `lib/embedding/providers.py::_EMBEDDING_PROVIDERS`
plus an update to this table's default row.

## Reranker providers (retrieval re-ranking)

The `ED4ALL_RERANK_*` family ([`docs/operations/behavior-flags.md`](operations/behavior-flags.md)) selects
the optional cross-encoder reranker that re-scores the first-stage retrieval
candidate pool on the grounded-answer path (resolver
`lib/retrieval/reranker.py`). **This is NOT training-data synthesis** — it
re-orders already-retrieved grounded passages and emits no corpus content; the
passages' native scores are preserved verbatim. Default OFF
(`ED4ALL_RERANK_PROVIDER` unset → no client built). The maintenance contract
still requires a row per provider/model-selecting flag, so they are documented
here in their own table. All candidates are license-clean (MIT / Apache-2.0);
jina / mxbai rerankers are deliberately EXCLUDED (CC-BY-NC / non-clean).

| Flag/value | Default model | Model license | ToS layer | Notes |
|------------|---------------|---------------|-----------|-------|
| `ED4ALL_RERANK_PROVIDER=st-cross-encoder` | `BAAI/bge-reranker-base` | MIT | N/A (local in-process) | runtime Q&A retrieval re-ranking; in-process `sentence-transformers` `CrossEncoder`; offline/loopback; NOT training-data synthesis (re-orders already-retrieved grounded passages, emits no corpus content) |
| `ED4ALL_RERANK_PROVIDER=fake` | deterministic hash scores | N/A | N/A | test-only; production read path refused without `ED4ALL_RERANK_ALLOW_FAKE` (anti-poisoning) |
| benchmark candidate | `BAAI/bge-reranker-large` | MIT | N/A (local in-process) | larger bge cross-encoder |
| benchmark candidate | `BAAI/bge-reranker-v2-m3` | Apache-2.0 | N/A (local in-process) | multilingual headroom |

## Validation models (inference-only quality gates — no training-data exposure)

The validation-gate suite (`lib/validators/`, `lib/retrieval/groundedness.py`,
`lib/classifiers/`) scores pipeline artifacts with small local models. **None
of these models author content**: they emit pass/fail verdicts, entailment
probabilities, cosine similarities, and Bloom-level votes over content that
already exists. No validator output is distributed as course content and none
lands in `instruction_pairs.jsonl`, so this surface has zero training-data
exposure regardless of model license. Recorded here explicitly (rather than
implied) as part of the public dependency register.

| Model | License | Role |
|-------|---------|------|
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | MIT (Microsoft DeBERTa-v3 base + MIT fine-tune) | Active NLI entailment scorer for prose, objective, and grounding gates. The optional Bloom trivote reuses it to compare six Bloom-level hypothesis templates. That comparison is a zero-shot heuristic, not a Bloom-trained classifier. |
| Unprovisioned Ed4All Bloom-head scaffold (`lib/classifiers/bloom_deberta_heads.py`; base `microsoft/deberta-v3-base`) | MIT base; any future Ed4All fine-tune requires its own artifact provenance | Optional `ED4ALL_BLOOM_TRIVOTE_HEADS` backend. No weights ship with or are currently provisioned for Ed4All, and no training is authorized by this documentation. The operator-run trainer is not wired into a workflow. Missing, partial, or unloadable local artifacts return `None`, so the head voter explicitly abstains. With strict mode off, trivote continues with the active zero-shot DeBERTa-NLI heuristic and remaining evidence; insufficient participation emits `BLOOM_TRIVOTE_INSUFFICIENT_VOTERS`. Strict mode fails closed when no usable classifier signal exists. |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache 2.0 | Validator cosine embedder — `rewrite_source_grounding`, statistical-tier validators, `BlockFeatureCache.embed`, NLI candidate ordering |
| `BAAI/bge-large-en-v1.5` | MIT | Retrieval-index embedder (see the Embedding-providers table above — listed here for completeness because retrieval eval gates read the same index) |
| `cip29/bert-blooms-taxonomy-classifier` | **No documented license; do not bundle, download, or redistribute.** | Retired, unreliable, non-authoritative compatibility metadata. The current dispatcher never loads it and returns `unknown` for the compatibility ensemble path. |
| `distilbert-base-uncased-finetuned-sst-2-english` | Apache 2.0 | Retired, non-authoritative sentiment-model metadata. It is not a Bloom classifier, and the current dispatcher never loads it. |

**Inference runtimes (optional accelerators, system-provided):** `torch` CUDA
wheels, `onnxruntime` (MIT), and NVIDIA TensorRT (proprietary NVIDIA SLA;
free production use on NVIDIA GPUs, runtime redistribution permitted only per
its redistributable-files clauses) are DEPLOYMENT dependencies, not part of
Ed4All's code license. Posture: never vendor NVIDIA proprietary binaries into
the repo or Ed4All-built images — document them as prerequisites installed
from NVIDIA's own channels (identical to the existing CUDA/NGC posture in
`docs/operations/docker.md`), so an institution's use is governed by the
NVIDIA license they already accept for their GPU stack. Optional backends must
degrade gracefully (or fail loudly) when the runtime is absent.

## Grounded-answer provider (runtime inference — not training data)

The `ED4ALL_ANSWER_*` family ([`docs/operations/behavior-flags.md`](operations/behavior-flags.md)) selects
the local model that composes a passage-constrained answer to a learner's
question at query time. **These outputs are NOT training data** — they are
ephemeral learner answers (runtime Q&A inference), never paraphrased into the
SLM corpus. The maintenance contract still requires a row per provider/model-
selecting flag, so they are documented here.

The answer path has **no cloud arm by design** (FERPA posture, $0 marginal cost,
fully offline). Resolution reads the `_OPENAI_COMPATIBLE_PROVIDERS`
registry but enforces that the resolved `base_url` host is loopback; a non-
loopback resolution raises `AnswerProviderNotLocal`. There is no escape-hatch
env for cloud answer routing. Any future *additional local* provider entry must
land with a row here.

| Flag/value | Default model | Model license | ToS layer | Training-data permitted | Recommended use |
|------------|---------------|---------------|-----------|-------------------------|-----------------|
| `ED4ALL_ANSWER_PROVIDER=local` | `nemotron-3-nano-30b-a3b` (via `ED4ALL_ANSWER_MODEL` → `LOCAL_SYNTHESIS_MODEL`) | NVIDIA Open Model License | N/A (your hardware; loopback-enforced) | N/A — runtime Q&A inference; outputs are ephemeral learner answers, never corpus content | **Only permitted value in Phase IA.** Non-loopback resolution raises `AnswerProviderNotLocal`. After a course adapter is trained, evaluated, and promoted, `ED4ALL_ANSWER_MODEL` is the explicit binding hook; the base model remains the default until then. |
| `ED4ALL_ASSISTANT_BASE_URL` / `ED4ALL_ASSISTANT_MODEL` (the `ed4all assistant` seat) | `nemotron-3-nano` on a local vLLM seat (an operator-configured local endpoint) | NVIDIA Nemotron Open Model License | N/A (your hardware; loopback-enforced — non-loopback resolution raises `AssistantProviderNotLocal`) | N/A — runtime operator-help surface (status / run start-stop / curated help chat); outputs are ephemeral operator replies, NEVER a training-data producer and never re-ingested as corpus content | Operator-assistant chat only (`lib/assistant/`); sandboxed to a typed tool whitelist, no shell / file access. |

### Citation links (verbatim)

- BAAI/bge-m3 LICENSE (MIT): https://huggingface.co/BAAI/bge-m3/blob/main/README.md
- BAAI/bge-large-en-v1.5 LICENSE (MIT): https://huggingface.co/BAAI/bge-large-en-v1.5
- Alibaba-NLP/gte-large-en-v1.5 LICENSE (Apache 2.0): https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5
- nomic-ai/nomic-embed-text-v1.5 LICENSE (Apache 2.0): https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- Qwen3-Embedding LICENSE (Apache 2.0): https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- all-MiniLM-L6-v2 LICENSE (Apache 2.0): https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- BAAI/bge-reranker-base LICENSE (MIT): https://huggingface.co/BAAI/bge-reranker-base
- BAAI/bge-reranker-large LICENSE (MIT): https://huggingface.co/BAAI/bge-reranker-large
- BAAI/bge-reranker-v2-m3 LICENSE (Apache 2.0): https://huggingface.co/BAAI/bge-reranker-v2-m3
- Anthropic Consumer Terms: https://www.anthropic.com/legal/consumer-terms
- Anthropic Commercial Terms: https://www.anthropic.com/legal/commercial-terms
- OpenAI Services Terms: https://openai.com/policies/services-terms/
- OpenAI Business Terms: https://openai.com/policies/business-terms/
- Together AI Terms of Service: https://www.together.ai/terms-of-service
- Qwen2.5-7B-Instruct LICENSE (Apache 2.0): https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE
- Qwen2.5-14B-Instruct LICENSE (Apache 2.0): https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/LICENSE
- Qwen2.5-32B-Instruct LICENSE (Apache 2.0): https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/LICENSE
- Qwen2.5-72B-Instruct LICENSE (Qwen License Agreement): https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE
- Llama 3.3 Community License: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/LICENSE
- Mistral-Small-Instruct LICENSE (Apache 2.0): https://huggingface.co/mistralai/Mistral-Small-Instruct-2409/blob/main/LICENSE
- Phi-3.5-mini LICENSE (MIT): https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE.md
- DeepSeek-V3 LICENSE: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL

### Notes per row

- **Anthropic / Claude Session** — Outputs are restricted from training-data use under Anthropic's ToS. The pipeline keeps these providers wired for backward compatibility and for callers who have a separate written agreement with Anthropic permitting derivative training, but the **default recommended path is NOT to use them for training-data synthesis**. The critical-severity `LibV2ModelValidator::MOCK_PROVIDER_CORPUS` check fails closed on `provider="mock"` corpora; analogous operator discipline is required for `provider="anthropic"` or `provider="claude_session"` runs that intend to train.
- **Together AI** — Together's ToS explicitly permits using outputs for training-data generation; the underlying OSS model license still governs distribution of the model and any derivatives. Both layers must be cited (ToS + model license). Llama-3.3 requires attribution and a >700M-MAU special license; Qwen2.5-72B requires written permission for >100M-MAU commercial use; DeepSeek-V3 carries its own permissive license.
- **Local OSS** — Output license is the underlying model's license, full stop. Apache 2.0 (Qwen2.5-7B/14B/32B, Mistral-Small) is the cleanest: unrestricted commercial use including using outputs to train derivative models, and no attribution required for outputs (only for redistributing the model itself). Llama-3.3 requires attribution. Qwen2.5-72B's Qwen License Agreement permits outputs for derivative training at any scale but gates >100M-MAU commercial use of the model.
- **Hosted-cloud registry entries (`groq` / `fireworks` / `deepseek`)** — these endpoints carry **cloud, not local, licensing exposure**. A hosted OSS teacher is governed by both the serving provider's terms and the underlying model license. Their endpoint rows therefore map `provenance_provider` to the existing cloud value `together`, and synthesis records that value rather than misclassifying hosted output as local. This preserves the closed `Touch.provider` set (`anthropic`, `claude_session`, `deterministic`, `local`, `nvidia`, `together`) while keeping corpus provenance truthful.

### Curriculum alignment shares the synthesis provider stack

The synthesis providers above are also the LLM stack consumed by the **curriculum-alignment surface** (`Trainforge/alignment/align_chunks.py` teaching-role classification via `Trainforge/generators/providers/_curriculum_provider.py::CurriculumAlignmentProvider`). The `CURRICULUM_ALIGNMENT_PROVIDER` env var (and the `--curriculum-provider` CLI flag on `python -m Trainforge.alignment.align_chunks`) accepts the same `anthropic` / `together` / `local` values, and the `local` and `together` branches reuse the same `LOCAL_SYNTHESIS_*` / `TOGETHER_*` env vars so one local server serves both task surfaces. Curriculum alignment writes corpus metadata (the `teaching_role` field on every chunk) — i.e. it touches training data — so the same ToS calculus applies.

**Recommended setting for both surfaces is `local`**, with the canonical `nemotron-3-nano-30b-a3b` TRT-LLM/vLLM seat under the **NVIDIA Nemotron Open Model License**, for a ToS-clean corpus end-to-end. The production `Trainforge.alignment.align_chunks.main()` path resolves `CURRICULUM_ALIGNMENT_PROVIDER` before classification, so setting the environment variable redirects the curriculum-alignment surface away from Anthropic.

### Courseforge content-generator shares the synthesis provider stack

The Courseforge content-generator surface (`Courseforge/generators/_provider.py::ContentGeneratorProvider`, instantiated from `MCP/tools/pipeline_tools.py::_generate_course_content`) reuses the same provider stack and env vars as Trainforge synthesis: the `local` and `together` branches read `LOCAL_SYNTHESIS_BASE_URL` / `LOCAL_SYNTHESIS_MODEL` / `LOCAL_SYNTHESIS_API_KEY` and `TOGETHER_API_KEY` / `TOGETHER_SYNTHESIS_MODEL` respectively, and the `anthropic` branch reads `ANTHROPIC_API_KEY` / `ANTHROPIC_SYNTHESIS_MODEL` — so a single local server serves all three surfaces (synthesis, curriculum alignment, content generation). Every page authored through this provider emits one `content_generator_call` decision event with `provider`, `model`, `page_id`, and retry count, so a post-hoc audit can attribute every HTML chunk to its provider. **Content-generator short-circuit semantics:** setting `COURSEFORGE_PROVIDER` to any non-empty value (`anthropic` / `together` / `local`) overrides `ED4ALL_AGENT_DISPATCH=true` for the `content-generator` agent only — the executor falls through to the in-process provider call instead of dispatching the Claude Code subagent, while every other pipeline agent (course-outliner, oscqr-course-evaluator, etc.) keeps dispatching unchanged. **Recommended setting for ToS-clean Courseforge content is `COURSEFORGE_PROVIDER=local` with `LOCAL_SYNTHESIS_MODEL=nemotron-3-nano-30b-a3b`**, governed by the NVIDIA Nemotron Open Model License, so the authored HTML — which Trainforge later ingests as training chunks — is license-clean from end to end.

### Deterministic generators (no LLM exposure)

The provider-free programs under `Trainforge/generators/deterministic/` emit
their pairs without an LLM call. The KG-metadata, pyshacl-verified violation,
abstention, schema-translation, assessment-SFT, and graph-SFT programs are
therefore outside provider ToS analysis. Their outputs derive from project
contracts and operator-private course artifacts; selecting a paraphrase
provider does not affect their licensing posture.

---

## Decision tree

If you are building a course-pinned SLM and want a license-clean training corpus:

1. **First choice:** `--provider local` with `LOCAL_SYNTHESIS_MODEL=nemotron-3-nano-30b-a3b` (NVIDIA Open Model License) on the canonical TRT-LLM/vLLM seat. The served model is roughly 30B parameters total and 3.5B active; deployment capacity depends on the engine and precision. Review the NVIDIA license terms for the deployment.
2. **If local deployment is unavailable:** `--provider together` with a hosted Apache 2.0 OSS model (Qwen2.5-72B-Instruct-Turbo) or the default Llama-3.3-70B. Both are ToS-clean for training-data generation.
3. **Do NOT use** `--provider anthropic` or `--provider claude_session` for training data unless you have separately obtained written permission from Anthropic. Pipeline default is to route around them.
4. **Do NOT use** `--provider mock` for any corpus you intend to train on. Mock is a deterministic 30-template factory wired for plumbing tests; the `MOCK_PROVIDER_CORPUS` validator fails closed on promotion.
5. **When unsure:** read the model's `LICENSE` file on Hugging Face and the provider's current ToS before kicking off a multi-hundred-dispatch run.

---

## Pipeline guarantees

The project's posture is encoded into the validation gates and provider invariants:

- **Sentinel-phrase hardening** removed a formerly injected filler sentence from short paraphrases. Any training pair that would have carried sentinel filler now triggers a re-paraphrase or a fail-loud `SynthesisProviderError`.
- **`LocalSynthesisProvider`** (`Trainforge/generators/providers/_local_provider.py`) was added precisely so the training corpus can be license-clean from end to end. The provider speaks the OpenAI chat-completions protocol against any local server (Ollama / vLLM / llama.cpp / LM Studio); the underlying model license is the only ToS layer that applies.
- **Anthropic providers stay wired** for backward compatibility but are no longer the recommended default for training data. The `synthesis_provider_call` decision-capture event records which provider produced each pair, so a post-hoc audit can identify any rows that crossed a ToS boundary.
- **Per-call audit trail** — every provider call emits a `synthesis_provider_call` decision event with model ID, max_tokens, prompt-cache hit/miss, and retry count. The full provider × model history of a corpus is reconstructible from `runtime/training-captures/trainforge/<COURSE_CODE>/`.
- **SemantiK conversion dependencies are explicit.** The GLM-OCR path renders PDF pages through the system-provided Poppler `pdftoppm` utility before local layout analysis and OCR. Ed4All does not vendor that runtime. Packaging or redistributing Poppler, derived weights, or other system-provided components requires separate operator and legal review; this document makes no redistribution conclusion. GLM-OCR and enrichment outputs are published product content rather than training pairs, although later Trainforge ingestion gives the selected endpoint and model the usual content-to-training-data exposure.

---

## When in doubt

- Read the model's `LICENSE` file on Hugging Face. The Hugging Face URLs above are the authoritative source — license terms can change between model releases.
- Read the provider's current ToS — Anthropic, OpenAI, and Together evolve their terms, and an old reading may be stale.
- If the use case is novel (multi-modal training, fine-tuning a frontier model from another's outputs, redistribution of derived weights, hosting an adapter for paid inference), consult counsel. This document is engineering documentation, not legal advice.

---

## Maintenance contract

Any new behavior flag in `CLAUDE.md` § "Opt-In Behavior Flags" that selects an LLM provider, model ID, or synthesis backend MUST land with a corresponding row in this file's "Synthesis providers" table (or a one-line entry in "Tooling" if it doesn't touch training data). Drift between this file and the per-provider rows in `CLAUDE.md` / `Trainforge/CLAUDE.md` is a documentation bug.
