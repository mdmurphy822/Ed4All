# Licensing & ToS Posture

This document is the canonical reference for Ed4All's licensing and Terms-of-Service posture across the tools and LLM models the project uses. Other docs (`CLAUDE.md`, `AGENTS.md`, `Trainforge/CLAUDE.md`) link here rather than duplicating, so this is the only file that should change when a provider's ToS or a model's license changes.

The repo-root `NOTICE` file is the short attribution summary (Apache-2.0 dependency notices + synthesis/embedding model attributions, e.g. Llama 3.3's "Built with Llama"). It deliberately points back here for the long-form posture; this file is the source of truth and `NOTICE` is the redistribution-facing acknowledgment. Keep them consistent: a new attributable dependency or model gets a `NOTICE` line and (when it selects a synthesis/embedding backend) a row in the tables below.

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

### Internal-tooling provenance notes (one DECIDED, one PENDING — NOT the shippable course-SLM corpus)

Two provenance surfaces are **internal-tooling only**. Neither feeds the distributed course-pinned SLM training corpus (that stays the license-clean synthesis stack in the table below). The first — vendor-aligned gold LABELS for the BERT-v2 structure/relation heads — was **DECIDED 2026-07-13** (owner-approved; see the row). The second — the Nemotron-teacher head-redistribution question — remains OPEN. Both are surfaced here so a future *distribution* decision doesn't inherit an unexamined liability: in both cases the trained heads stay internal pipeline instruments, and the NC / vendor source content itself is never distributed.

| Surface | What it is | License exposure | Posture (PENDING) |
|---------|------------|------------------|-------------------|
| **Nemotron-teacher-labeled internal tooling models (BERT-v2)** — task #42 | The SemantiK council structure heads retrained as in-distribution ontology annotators on labels produced by a **Nemotron VLM teacher** (`nemotron-3-nano-omni` / Super-120B). The trained BERT-v2 classifier is therefore a derivative work of Nemotron outputs. It is a STRUCTURE-DETECTION tool inside the conversion cascade — it emits no distributed prose and never lands in `instruction_pairs.jsonl`. | NVIDIA Open Model License on the teacher outputs (permissive — training-on-outputs explicitly permitted, sole restriction is the competing-foundation-model-training-service bar, which a document-structure classifier does not touch). The self-hosted Spark Nemotron seats are already treated training-clean in the Synthesis-providers table below. | **Internal-tooling posture owner-affirmed 2026-07-13; head-redistribution still PENDING (task #42).** Under the NVIDIA Open Model License the teacher-labeled classifier reads clean. The 2026-07-13 owner decision (row below) affirms BERT-v2 as an **internal pipeline instrument** — training the structure/relation heads on the available label supply is cleared. What remains OPEN is narrowly the *distribution* question: is the trained head redistributed, or purely internal to a self-hosted build? Treat as internal-tooling-only until task #42 closes that with an explicit posture row. Not in the license-restricted synthesis set (`{"anthropic", "nvidia"}` — the HOSTED `nvidia` API seat), because the labels come from the self-hosted `local`-provenance Nemotron weights, not the hosted API. |
| **Vendor-HTML eval corpora + vendor-aligned BERT-v2 gold labels** | Publisher / vendor HTML in two internal-tooling roles: **(a)** MEASURE conversion + retrieval quality (structure-authority A/B, gold-compare); **(b)** the `gold_alignment` structure/relation LABELS (`caption_of` / `practice_of` / `solution_of` / `same_unit` / furniture + co-occurrence) that train the BERT-v2 structure + relation heads — OCR'd scan units aligned back onto the vendor HTML's OWN markup (no model call). Never republished, never re-ingested as distributed training chunks, gitignored. | Vendor terms are typically **non-commercial (NC)** and/or no-redistribution (e.g. a CC BY-NC-SA-licensed vendor textbook). | **DECIDED 2026-07-13 (owner-approved).** CC BY-NC-SA vendor-aligned gold labels ARE usable for training the BERT-v2 **internal tooling models** (the SemantiK structure/relation heads). Rationale: the trained heads are internal pipeline instruments (document-structure classifiers inside the conversion cascade) — they emit no distributed prose and never land in the course-SLM `instruction_pairs.jsonl`; the NC content itself is still **never distributed**. The `license_tag` provenance field is stamped on every gold record (the onboarding pipeline + `dataset_builder.py`), so the posture is **reversible + auditable** — re-quarantining the NC family is a one-flag change (`--exclude-license cc-by-nc`). Eval-only use (measurement fixtures) is unchanged and still permitted. Distributing any vendor-HTML-derived TEXT (as opposed to a trained-classifier weight) still requires its own cleared license; this mirrors the standing internal-tool-validation posture for licensed source material (licensed vendor content is not distributed without permission). |

---

## Synthesis providers (training-data exposure)

These are the providers that actually produce paraphrased training pairs. Each row's ToS layer + underlying model license decide whether the resulting corpus can train a derivative SLM without legal exposure.

| `--provider` flag | Default model | Model license | ToS layer | Training-data permitted | Recommended use |
|-------------------|---------------|---------------|-----------|--------------------------|-----------------|
| `anthropic` | `claude-sonnet-4-6` | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward compat / non-training synthesis only |
| `claude_session` | Active Claude Code session | Anthropic proprietary | Anthropic Consumer Terms (Pro/Max) | **No** | Backward compat only — consumer terms even more restrictive |
| `COURSEFORGE_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `ANTHROPIC_SYNTHESIS_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward-compat default for Courseforge content-generator; not recommended for training data |
| `COURSEFORGE_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for Courseforge content authoring |
| `COURSEFORGE_PROVIDER=local` | `qwen2.5:7b-instruct-q4_K_M` (via `LOCAL_SYNTHESIS_MODEL`) | Apache 2.0 (Qwen2.5-7B/14B/32B default) | N/A (your hardware) | **Yes** | **Recommended for ToS-clean Courseforge content** |
| `COURSEFORGE_OUTLINE_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `ANTHROPIC_SYNTHESIS_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Two-pass outline tier; not recommended for training data |
| `COURSEFORGE_OUTLINE_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the two-pass outline tier |
| `COURSEFORGE_OUTLINE_PROVIDER=local` | `qwen2.5:7b-instruct-q4_K_M` (via `LOCAL_SYNTHESIS_MODEL`) | Apache 2.0 (Qwen2.5-7B/14B/32B default) | N/A (your hardware) | **Yes** | **Recommended for ToS-clean outline drafting** — outline drafts are re-ingested by Trainforge as chunks, so this surface IS training-data exposure |
| `COURSEFORGE_OUTLINE_PROVIDER=<registry>` | per the W-D12 `_OPENAI_COMPATIBLE_PROVIDERS` registry entry (`groq` / `fireworks` / `deepseek` / `nvidia` / future additions) | per registry entry's underlying model license | per provider's ToS | per provider | Dynamic provider — the outline tier now accepts the full endpoint-registry superset; adding a registry entry surfaces here with no code edit (per-seat posture carried by the `config/endpoints.yaml` row / model choice) |
| `COURSEFORGE_REWRITE_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `ANTHROPIC_SYNTHESIS_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward-compat default for the two-pass rewrite tier; not recommended for training data |
| `COURSEFORGE_REWRITE_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the two-pass rewrite (authoring) tier |
| `COURSEFORGE_REWRITE_PROVIDER=local` | `qwen2.5:7b-instruct-q4_K_M` (via `LOCAL_SYNTHESIS_MODEL`) | Apache 2.0 (Qwen2.5-7B/14B/32B default) | N/A (your hardware) | **Yes** | **Recommended for ToS-clean rewrite authoring** — rewrite-tier published HTML is re-ingested by Trainforge as chunks, so this surface IS training-data exposure |
| `COURSEFORGE_REWRITE_PROVIDER=<registry>` | per the W-D12 `_OPENAI_COMPATIBLE_PROVIDERS` registry entry (`groq` / `fireworks` / `deepseek` / `nvidia` / future additions) | per registry entry's underlying model license | per provider's ToS | per provider | Dynamic provider — the rewrite tier now accepts the full endpoint-registry superset (the legacy `openai_compatible` alias collapses to `local` at constructor entry); adding a registry entry surfaces here with no code edit |
| `COURSEPLANNER_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `ANTHROPIC_SYNTHESIS_MODEL` / `COURSEPLANNER_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward-compat default for `OutlinerProvider`; not recommended for training data |
| `COURSEPLANNER_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL` / `COURSEPLANNER_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the Courseforge course-outliner surface (W-D14) |
| `COURSEPLANNER_PROVIDER=local` | `qwen2.5:7b-instruct-q4_K_M` (via `LOCAL_SYNTHESIS_MODEL` / `COURSEPLANNER_MODEL`) | Apache 2.0 (Qwen2.5-7B/14B/32B default) | N/A (your hardware) | **Yes** | **Recommended for ToS-clean course planning** — closes the W-D11.F "course-outliner subagent" gap (W-D14) |
| `COURSEPLANNER_PROVIDER=<registry>` | per the W-D12 `_OPENAI_COMPATIBLE_PROVIDERS` registry entry (`groq` / `fireworks` / `deepseek` / future additions) | per registry entry's underlying model license | per provider's ToS | per provider | Dynamic provider — adding a new entry to the registry surfaces here without a code edit per the W-D12 dynamic-references contract |
| `TEXTBOOK_SYNTHESIS_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `TEXTBOOK_SYNTHESIS_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward-compat default for `TextbookSynthesisProvider`; not recommended for training data |
| `TEXTBOOK_SYNTHESIS_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TEXTBOOK_SYNTHESIS_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the three-stage textbook synthesis surface |
| `TEXTBOOK_SYNTHESIS_PROVIDER=local` | `qwen2.5:7b-instruct-q4_K_M` (via `LOCAL_SYNTHESIS_MODEL` / `TEXTBOOK_SYNTHESIS_MODEL`) | Apache 2.0 (Qwen2.5-7B/14B/32B default) | N/A (your hardware) | **Yes** | **Recommended for ToS-clean three-stage textbook synthesis** — the Stage-3 domain-concept vocabulary propagates into chunk `concept_tags[]` and the Stage-1/2 synthesized objectives propagate into every downstream chunk's `learning_outcome_refs[]`, so this surface IS training-data exposure |
| `TEXTBOOK_SYNTHESIS_PROVIDER=<registry>` | per the W-D12 `_OPENAI_COMPATIBLE_PROVIDERS` registry entry (`groq` / `fireworks` / `deepseek` / future additions) | per registry entry's underlying model license | per provider's ToS | per provider | Dynamic provider — adding a new entry to the registry surfaces here without a code edit per the W-D12 dynamic-references contract |
| `TRAINFORGE_ASSESSMENT_PROVIDER=anthropic` | `claude-sonnet-4-6` (via `ANTHROPIC_SYNTHESIS_MODEL` / `TRAINFORGE_ASSESSMENT_MODEL`) | Anthropic proprietary | Anthropic Commercial Terms | **No** (without separate agreement) | Backward-compat default for `AssessmentGeneratorProvider`; not recommended for training data |
| `TRAINFORGE_ASSESSMENT_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL` / `TRAINFORGE_ASSESSMENT_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the Trainforge assessment-generator surface (W-D15) |
| `TRAINFORGE_ASSESSMENT_PROVIDER=local` | `qwen2.5:7b-instruct-q4_K_M` (via `LOCAL_SYNTHESIS_MODEL` / `TRAINFORGE_ASSESSMENT_MODEL`) | Apache 2.0 (Qwen2.5-7B/14B/32B default) | N/A (your hardware) | **Yes** | **Recommended for ToS-clean assessment generation** — closes the W-D11.F "assessment-generator subagent" gap (W-D15). Authored questions feed downstream `training_synthesis` so this surface IS training-data exposure. |
| `TRAINFORGE_ASSESSMENT_PROVIDER=<registry>` | per the W-D12 `_OPENAI_COMPATIBLE_PROVIDERS` registry entry (`groq` / `fireworks` / `deepseek` / future additions) | per registry entry's underlying model license | per provider's ToS | per provider | Dynamic provider — adding a new entry to the registry surfaces here without a code edit per the W-D12 dynamic-references contract |
| `TRAINFORGE_SYNTHESIS_PROVIDER=anthropic` | n/a — path removed | Anthropic proprietary | Anthropic Commercial Terms | **No** | **REMOVED (Phase 4).** The Anthropic-SDK training-pair synthesis path (`AnthropicSynthesisProvider` + its SDK transport) was deleted entirely. `run_synthesis` now fails closed with `SynthesisLicensingError` **UNCONDITIONALLY** on `provider="anthropic"` — there is NO acknowledgment-flag escape (`TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true` does NOT unlock it). Training-pair synthesis is license-clean by construction. Use `local` / `together`. |
| `TRAINFORGE_SYNTHESIS_PROVIDER=claude_session` | Active Claude Code session | Anthropic proprietary | Anthropic Consumer Terms (Pro/Max) | **No** | **Gated opt-in** — a SEPARATE Claude-Code-session route (NOT the removed SDK path). `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true` acknowledgment required; even-more-restricted consumer-terms path. Not recommended for training data. |
| `TRAINFORGE_SYNTHESIS_PROVIDER` (unset, pipeline run) | n/a — resolves to a provider | per resolved provider | per resolved provider | **per resolved provider** | **License-clean by default (Marketable-v1 D4).** On a `textbook_to_course` / `course_generation` run the workflow runner defaults this env to `LLM_PROVIDER > local` (mirrors the GUI authoring-route fill), so a CLI/GUI run routes training-pair synthesis through a license-clean provider instead of the Claude Code subagent. setdefault — an explicit value wins. |
| `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true` | n/a — acknowledgment gate | n/a | n/a | n/a | **Acknowledgment flag, not a provider selector.** Gates the `claude_session` training-pair route ONLY (the `anthropic` SDK path was removed Phase 4 and fails closed unconditionally — this flag does NOT unlock it). Absent it, `Trainforge/synthesize_training.py::run_synthesis` raises `SynthesisLicensingError` for `claude_session` before any LLM dispatch, pointing here. Set only if you hold a separate Anthropic agreement permitting derivative training. |
| `TRAINFORGE_SYNTHESIS_PROVIDER=together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (via `TOGETHER_SYNTHESIS_MODEL`) | Llama 3.3 Community License (or selected Together OSS model) | Together AI ToS | **Yes** | Hosted OSS option for the Trainforge training-synthesizer surface (Wave1-I1) |
| `TRAINFORGE_SYNTHESIS_PROVIDER=local` | `qwen2.5:7b-instruct-q4_K_M` (via `LOCAL_SYNTHESIS_MODEL`) | Apache 2.0 (Qwen2.5-7B/14B/32B default) | N/A (your hardware) | **Yes** | **Recommended for ToS-clean training-pair synthesis** — closes Finding 1 of `plans/dispatch-7-execution-inspection-2026-05.md`. The emitted instruction / preference pairs ARE the canonical SLM training corpus consumed by `Trainforge.train_course`, so Claude must NEVER author them — this short-circuit replaces `--skip-training` operator-discipline with a fail-loud env-var guard (Wave1-I1). |
| `together` (Llama) | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Llama 3.3 Community License | Together AI ToS | **Yes** | Hosted OSS fallback |
| `together` (Qwen) | `Qwen/Qwen2.5-72B-Instruct-Turbo` | Qwen License Agreement | Together AI ToS | **Yes** | Hosted OSS fallback |
| `together` (DeepSeek) | `deepseek-ai/DeepSeek-V3` | DeepSeek License | Together AI ToS | **Yes** (per DeepSeek License) | Hosted OSS fallback |
| `local` (Qwen 7B) | `qwen2.5:7b-instruct-q4_K_M` | Apache 2.0 | N/A (your hardware) | **Yes** | **Recommended default** for license-clean corpora |
| `local` (Qwen 14B) | `qwen2.5:14b-instruct-q4_K_M` | Apache 2.0 | N/A | **Yes** | Stronger paraphrase, 12 GB GPU |
| `local` (Qwen 32B) | `qwen2.5:32b-instruct-q4_K_M` | Apache 2.0 | N/A | **Yes** | Top OSS quality on 24 GB GPU |
| `ED4ALL_CAMPAIGN_BASE_MODEL` | `nemotron3-nano-30b` (BF16 LoRA base, not a synthesis provider) | NVIDIA Nemotron Open Model License (Dec 15 2025 pin — `assert_nemotron_pin` fails the build on identity drift) | N/A (your hardware, HF-offline pre-seeded snapshot) | n/a — this flag selects the model being TRAINED, not a pair-authoring teacher | Campaign Stage-B base-model selector (`lib/assistant/campaign_tools.resolve_campaign_base_model`); the value must resolve in `Trainforge/training/base_models.py::BaseModelRegistry` — unknown name is a loud error, never a fallback model. Teacher-side licensing for the pairs it trains on is governed by the SFT teacher roster below. |
| `ed4all run --base-model <name>` (CLI flag) | none — unset leaves `ED4ALL_CAMPAIGN_BASE_MODEL` > `nemotron3-nano-30b` in charge | Whatever the NAMED registry entry carries. Every `BaseModelRegistry` entry is a license-vetted base: the Nemotron entry stays under the Dec 15 2025 pin (`assert_nemotron_pin`), the Qwen 2.5 / SmolLM2 / Llama 3.2 / Phi 3.5 entries under their own upstream terms. The flag CANNOT introduce an unvetted base — an unrecognized name exits 2 with the supported list. | N/A (your hardware) | n/a — selects the model being TRAINED, not a pair-authoring teacher | Operator-facing sibling of the env var above and the HIGHEST-precedence input to the same resolution chain (`--base-model` > `ED4ALL_CAMPAIGN_BASE_MODEL` > registry default), consumed by `config/workflows.yaml::training`'s `inputs_from: base_model <- workflow_params.base_model` route. Validated at CLI parse time through the SAME `BaseModelRegistry.resolve` the `run_training` handler validates against — one registry, one supported-set message, no second allowlist. Governs both `ed4all run trainforge_train` and the in-build `--with-training` tail; re-pinnable on `--resume`. Teacher-side licensing for the pairs it trains on is governed by the SFT teacher roster below. |
| `local` (Qwen 72B) | `qwen2.5:72b-instruct-q4_K_M` | Qwen License Agreement | N/A | **Yes** (outputs unrestricted at any scale) | Highest OSS quality, A100 / multi-GPU |
| `local` (Llama 70B) | `llama3.3:70b-instruct-q4_K_M` | Llama 3.3 Community License | N/A | **Yes** (with attribution) | Strong instruction following |
| `local` (Mistral 24B) | `mistral-small:24b-instruct-q4_K_M` | Apache 2.0 | N/A | **Yes** | Faster on 16 GB GPU |
| `local` (Phi-3.5 mini) | `phi3.5:3.8b-mini-instruct-q4_K_M` | MIT | N/A | **Yes** | Smallest OSS option |
| `COURSEFORGE_BLOCK_ROUTING_PATH=Courseforge/config/block_routing.license_clean.yaml` | n/a (router-level YAML; large tier defaults to `qwen2.5:32b-instruct-q4_K_M`) | Apache 2.0 (when `large` resolves to local Qwen) | n/a (your hardware) — or Together AI ToS when `large` is swapped to a hosted Apache-2.0 OSS model | **Yes** | **Recommended for ToS-clean Courseforge two-pass** runs. Sibling YAML override that swaps the canonical `large` capability tier (`claude-sonnet-4-6`) for a local 32B Qwen. See `docs/operations/license-clean-run.md` for the 4-env-var deployment recipe and the calibration prerequisite for `assessment_item` distractor quality. |
| `COURSEFORGE_BLOCK_ROUTING_PATH=Courseforge/config/block_routing.nvidia_large.yaml` (key from `NVIDIA_API_KEY`; base_url/model via `NVIDIA_BASE_URL` / `NVIDIA_LARGE_MODEL`) | `meta/llama-3.3-70b-instruct` (large tier only; small/medium stay local Qwen 7B/14B) | NVIDIA model / catalog license | NVIDIA hosted-API ToS | **N/A — generates COURSE CONTENT (product), not training-data corpus** | Router-level YAML sibling of the license-clean variant. The small + medium capability tiers are byte-identical local 7B/14B Qwen (Apache 2.0, loopback-only); ONLY the rewrite `large` / escalation tier routes to NVIDIA's hosted OpenAI-compatible inference API. **This row is content-generation, not synthesis:** the NVIDIA-authored HTML is published product. If that HTML is later re-ingested by Trainforge as training chunks, the standard content→training-data caveat applies (see "Courseforge content-generator shares the synthesis provider stack" below) — operators wanting a fully ToS-clean training corpus should keep this surface on a license-clean local/Together provider. |
| `SEMANTIK_SPECIALIST_PROVIDER=<endpoint>` + `SEMANTIK_SPECIALIST_MODEL` / `SEMANTIK_STRUCTURE_REVIEW_MODEL` (key from `NVIDIA_API_KEY`; base via `NVIDIA_BASE_URL`; model resolves through `NVIDIA_LARGE_MODEL`) | `meta/llama-3.3-70b-instruct` (Stage-6 specialist generation + Stage-5d structure-reviewer seats; the council adapters / theta stages stay local) | NVIDIA model / catalog license (Llama 3.3 Community License underneath) | NVIDIA hosted-API ToS | **N/A — generates STRUCTURED CONTENT (product), not training-data corpus** | SemantiK endpoint seat. `SEMANTIK_SPECIALIST_PROVIDER` defaults to `local` (in-process GGUF council adapters — no network, no key); ONLY a non-local value routes the Stage-6 specialist fill + the optional Stage-5d 70B structure-reviewer (`SEMANTIK_STRUCTURE_REVIEW`, default off) to NVIDIA's hosted OpenAI-compatible API. The OCR / theta / extraction stages remain local-only. **This row is content-generation, not synthesis:** the produced HTML/structure is published product. If that output is later re-ingested by Trainforge as training chunks, the standard content→training-data caveat applies — operators wanting a fully ToS-clean training corpus should keep `SEMANTIK_SPECIALIST_PROVIDER=local`. Flag detail: `SemantiK/CLAUDE.md § Opt-In Behavior Flags`. |
| `SEMANTIK_VLM_PROVIDER` / `SEMANTIK_VLM_MODEL` (base via `SEMANTIK_VLM_BASE_URL`, default `http://localhost:11434`; key from `SEMANTIK_VLM_API_KEY` — none for the local/loopback seat) | `qwen2.5vl:7b` = **Qwen2.5-VL-7B-Instruct** (the P0 VLM extraction-source seat, gated by `SEMANTIK_VLM_EXTRACT`, default off; the council BERTs / OCR / theta stages stay local) | **Apache-2.0** (the 7B VL is Apache-2.0 — NOT the Qwen-licensed 72B VL; pin the 7B) | N/A — local-by-default (loopback ollama; a non-local seat is opt-in and carries that endpoint's ToS) | **N/A — generates STRUCTURED CONTENT (product), not training-data corpus** | SemantiK's opt-in LOCAL VLM extraction seat: a per-page image → Qwen2.5-VL markdown transcription fused as a fourth extraction source (provider-agnostic OpenAI-compatible, mirroring `SEMANTIK_SPECIALIST_*` — a new provider is env config, never a subclass; the same seat serves the local ollama GPU and a Spark endpoint later). Default OFF and local (`qwen2.5vl:7b` on loopback ollama), so the default posture stays fully offline. **This row is content-generation, not synthesis:** the transcribed structure is published product. If that output is later re-ingested by Trainforge as training chunks, the standard content→training-data caveat applies — keep the seat local for a fully ToS-clean corpus. MiniCPM-V is excluded on license (plan bake-off note). **`SEMANTIK_REASONING_QC` (the Stage-9b reasoning-QC pass, default off) rides THIS already-licensed VLM seat** (the Apache-2.0 `qwen2.5vl:7b` default), so it needs no separate row — UNLESS its optional `SEMANTIK_REASONING_QC_MODEL` override selects a DISTINCT model/endpoint, which then requires its own Synthesis-providers row per the maintenance contract below. Flag detail: `SemantiK/CLAUDE.md § Opt-In Behavior Flags`. |
| `SEMANTIK_SEMANTIC_SUBCLASS` (seat via `LOCAL_SYNTHESIS_BASE_URL` / `LOCAL_SYNTHESIS_MODEL` / `LOCAL_SYNTHESIS_API_KEY`) | `qwen2.5:7b-instruct-q4_K_M` (the EXISTING local reviewer seat; Build #23 Tier-3 composite-unit subclassifier, `lib/semantik/subclassifier.py`, gated by `SEMANTIK_SEMANTIC_SUBCLASS`, default off) | **Apache-2.0** (Qwen2.5-7B default; any `LOCAL_SYNTHESIS_MODEL` override carries that model's license) | N/A — local-by-default (loopback Ollama-style OpenAI-compatible server; a non-local override carries that endpoint's ToS) | **N/A — emits a metadata LABEL (`data-semantik-subclass` / chunk `unit_subclass`), not prose training-data corpus** | SemantiK's opt-in Tier-3 composite-unit subclass pass reuses the SAME license-clean local reviewer seat as `SEMANTIK_SPECIALIST_PROVIDER=local` / the Trainforge `local` synthesis seat (env-resolved via `LOCAL_SYNTHESIS_*`, no hardcoded endpoint). It classifies each already-rendered composite unit into a kebab-case subclass label added payload-only to the published HTML; the label rides into chunks as the additive `unit_subclass` metadata field (never prose text, never a chunk-text/id change). Default OFF → no dispatch. Keep the seat local (the byte-stable default) for a fully ToS-clean corpus. Flag detail: `SemantiK/CLAUDE.md § Opt-In Behavior Flags`. |

| `SEMANTIK_GLMOCR_LANE` (seat via `SEMANTIK_GLMOCR_BASE_URL`, default `http://localhost:8002/v1`; model via `SEMANTIK_GLMOCR_MODEL`, default `glm-ocr`) | `glm-ocr` = **GLM-OCR** served on a local vLLM seat; SDK `glmocr` 0.1.5 (PP-DocLayoutV3 layout on CPU); the owner-adopted GLM-OCR extraction lane, gated by `SEMANTIK_GLMOCR_LANE`, default off | **MIT** (GLM-OCR model weights `zai-org/GLM-OCR`); the `glmocr` SDK is **Apache-2.0**; PP-DocLayoutV3 (PaddleOCR layout) is **Apache-2.0** | N/A — local-by-default (loopback vLLM seat; a non-local seat is opt-in and carries that endpoint's ToS) | **N/A — generates STRUCTURED CONTENT (accessible HTML, product), not training-data corpus** | SemantiK's opt-in GLM-OCR whole-document extraction lane: PDF → 300-DPI renders → PP-DocLayoutV3 layout + per-region OCR on the GLM-OCR seat → deterministic transform → `region_provenance` → the license-clean `lib/semantik` adapter renders the accessible HTML. Default OFF and local. **This row is content-generation / accessibility, not synthesis:** the converted HTML is published product. If that output is later re-ingested by Trainforge as training chunks, the standard content→training-data caveat applies — keep the seat local for a fully ToS-clean corpus. Flag detail: `SemantiK/CLAUDE.md § Opt-In Behavior Flags`. |
| `SEMANTIK_ALTTEXT_PROVIDER=qwen30` (seat via `SEMANTIK_ALTTEXT_BASE_URL`, default `http://localhost:8003/v1`; model via `SEMANTIK_ALTTEXT_MODEL`, default `qwen3-vl-30b`; key from `SEMANTIK_ALTTEXT_API_KEY` — none for the local/loopback seat) | `qwen3-vl-30b` = **Qwen3-VL-30B-A3B-Instruct-FP8** (the GLM-OCR lane's WCAG alt-text / caption seat, gated by `SEMANTIK_ALTTEXT_PROVIDER`, default `off`) | **Apache-2.0** (Qwen3-VL-30B-A3B-Instruct) | N/A — local-by-default (loopback OpenAI-compatible seat; a non-local seat is opt-in and carries that endpoint's ToS) | **N/A — emits WCAG alt text / captions (accessibility metadata on figures, product), not training-data corpus** | The GLM-OCR lane's optional figure alt-text / caption generator: a bbox crop of a caption-less figure → a function-first ≤125-char WCAG 2.2 AA alt string (+ optional long_desc / caption). Default `off` inside the lane until seat provisioning is wired (the lane ships harvested captions + honest placeholders otherwise). **This row is accessibility content-generation, not synthesis.** Keep the seat local for a fully ToS-clean corpus. Flag detail: `SemantiK/CLAUDE.md § Opt-In Behavior Flags`. |
| `spark-super` | `nemotron-3-super-120b-a12b` (via `SPARK_SUPER_MODEL`) | NVIDIA Open Model License (permissive — commercial use, modification, redistribution, and TRAINING ON OUTPUTS all explicitly permitted; NVIDIA disclaims ownership of outputs; the single restriction — may not be used to build a competing foundation-model-training service — does NOT bind Ed4All's course-pinned SLM training) | N/A (your hardware) | **Yes** | **Local self-hosted Nemotron (DGX Spark) — quality synthesis tier.** NOT the hosted `nvidia` API seat; these are local weights and are training-clean (see the distinction note below the table). |
| `spark-nano` | `nemotron-3-nano-30b-a3b` (via `SPARK_NANO_MODEL`) | NVIDIA Open Model License (permissive — same terms as the `spark-super` row: training-on-outputs permitted, NVIDIA disclaims output ownership, sole restriction is the competing-foundation-model-training-service bar which does NOT bind Ed4All's course-pinned SLM training) | N/A (your hardware) | **Yes** | **Local self-hosted Nemotron (DGX Spark) — high-volume / interactive tier.** NOT the hosted `nvidia` API seat; local weights, training-clean. |
| `SEMANTIK_VLM_MODEL=nemotron-3-nano-omni-30b-a3b` | `nemotron-3-nano-omni-30b-a3b` (via `SEMANTIK_VLM_MODEL`) | NVIDIA Open Model License (permissive — same terms as the `spark-super` row: training-on-outputs permitted, NVIDIA disclaims output ownership, sole restriction is the competing-foundation-model-training-service bar which does NOT bind Ed4All's course-pinned SLM training) | N/A (your hardware) | **Yes** | **Local self-hosted Nemotron-Omni (DGX Spark) — SemantiK document VLM** (per-page image → markdown transcription source seat; the local Nemotron alternative to the default `qwen2.5vl:7b` VLM row above). NOT the hosted `nvidia` API seat; local weights, training-clean. |

**Local self-hosted Nemotron ≠ the hosted `nvidia` API seat.** The three rows immediately above are self-hosted Nemotron 3 weights running on the DGX Spark. Under the NVIDIA Open Model License their outputs are training-clean, so they are deliberately **NOT** in the license-restricted synthesis set (`MCP/core/workflow_runner.py::_LICENSE_RESTRICTED_SYNTHESIS` / `lib/diagnostics/provider.py::_LICENSE_RESTRICTED`, both `{"anthropic", "nvidia"}`) — those constants key on the hosted-provider name `nvidia`, and the Spark seats carry `local` provenance. The hosted `nvidia` provider (the `NVIDIA_*` cloud-API seat used by `block_routing.nvidia_large.yaml` / `SEMANTIK_SPECIALIST_PROVIDER`) **stays license-restricted and gated OFF for training data**; its exposure is the hosted-API ToS layer, not the underlying model license. Do not conflate the two: `nvidia` (hosted API, restricted) and `spark-super` / `spark-nano` / the local Nemotron-Omni VLM (self-hosted weights, training-clean) are distinct seats.

---

## SFT teacher roster (course-pinned 1.5B adapter)

This section governs which model **outputs** may become instruction / preference training pairs for the commercially-shipped course-pinned SLM adapter (Qwen2.5-1.5B + LoRA), and which model **weights** may be ingested as a base checkpoint. It is the prose half of the machine-readable roster at `lib/licensing/teacher_roster.py`; the two must stay consistent (maintenance contract at the bottom of this file).

**Machine-readable source of truth + build invariants.** `lib/licensing/teacher_roster.py` encodes every row below as a `LicenseRecord` (`{name, license_spdx, license_url, verdict, obligations[], commercial_use}`) and exposes three fail-closed guards, all wired into the training preflight (`Trainforge/training/runner.py::_assert_licensing_preflight`):

1. **Export-time teacher filter** — `assert_export_licenses(pairs)` refuses to export/train a corpus if **any** pair's teacher is `barred`, unregistered, or claude/anthropic-tagged, naming the offending pair + teacher. A pair carrying no teacher signal at all (legacy shape) or a license-clean `local` / `together` provenance passes (registry-defaults-byte-identical).
2. **Per-checkpoint LICENSE assertion at ingest** — `assert_checkpoint_license(model, role="base_model")` verifies the *actual* base-weight license (Qwen 3B/72B ≠ 7-32B; GLM-4-9B ≠ 4.5; DeepSeek distills inherit their base), barring non-commercial weights for a shipped adapter.
3. **Nemotron license-pin / FAIL-BUILD-ON-RE-PIN guard** — see the Nemotron subsection below.

Per-pair provenance: synthesis stamps every pair with `generating_seat` (model id) + a `license` tag (`stamp_pair_license`), additive on the pairs' `additionalProperties:true` schemas — the coarse closed-enum `provider` field is untouched. The registry accessor `MCP/orchestrator/llm_backend.py::license_metadata_for_provider` surfaces the same posture per endpoint (inline YAML `license_*` fields win; else the roster is consulted by the endpoint's default model / provenance).

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
| `ED4ALL_EMBEDDING_PROVIDER=st` | `BAAI/bge-large-en-v1.5` | MIT | N/A (local in-process) | retrieval-index embeddings; in-process `sentence-transformers`; not training-data synthesis; default re-pinned from the 2026-06-10 4-model benchmark |
| `ED4ALL_EMBEDDING_PROVIDER=local-openai` | `nomic-embed-text` (Ollama) | Apache 2.0 | N/A (your hardware) | OpenAI-compatible `/v1/embeddings` against a local server (Ollama / vLLM / llama.cpp) |
| `ED4ALL_EMBEDDING_PROVIDER=fake` | deterministic hash vectors | N/A | N/A | test-only; production index load refused without `ED4ALL_EMBEDDING_ALLOW_FAKE` |
| benchmark candidate | `BAAI/bge-large-en-v1.5` | MIT | N/A (local in-process) | strong en-only baseline (D5 benchmark arm) — **selected 2026-06-10** (hybrid-rrf winner; now the `st` default above) |
| benchmark candidate | `Alibaba-NLP/gte-large-en-v1.5` | Apache 2.0 | N/A (local in-process) | requires `trust_remote_code=True` (executes HF model code) — droppable candidate |
| benchmark candidate | `nomic-ai/nomic-embed-text-v1.5` | Apache 2.0 | N/A (local in-process) | also Ollama-servable; needs `search_query:` / `search_document:` task prefixes |
| benchmark candidate | `Qwen/Qwen3-Embedding-0.6B` | Apache 2.0 | N/A (local in-process) | documented stretch candidate |
| smoke baseline | `sentence-transformers/all-MiniLM-L6-v2` | Apache 2.0 | N/A (local in-process) | already cached; CI real-model smoke + floor baseline only |

The default model pin (`BAAI/bge-large-en-v1.5`, re-pinned 2026-06-10 from the
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
implied) per the 2026-07-18 owner request.

| Model | License | Role |
|-------|---------|------|
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | MIT (Microsoft DeBERTa-v3 base + MIT fine-tune) | NLI entailment scorer — `block_prose_entailment`, `claim_support`, `objective_entailment`, `block_objective_delivery`, grounded-answer groundedness, eval gates; also the Bloom ensemble's zero-shot member AND the zero-shot voter of the re-founded `bloom_classifier_disagreement` gate under `ED4ALL_BLOOM_TRIVOTE` (same process-singleton — no second load) |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache 2.0 | Validator cosine embedder — `rewrite_source_grounding`, statistical-tier validators, `BlockFeatureCache.embed`, NLI candidate ordering |
| `BAAI/bge-large-en-v1.5` | MIT | Retrieval-index embedder (see the Embedding-providers table above — listed here for completeness because retrieval eval gates read the same index) |
| `cip29/bert-blooms-taxonomy-classifier` | **Not stated on the model card — unverified** | **RETIRED under `ED4ALL_BLOOM_TRIVOTE` (default OFF; owner-approved 2026-07-18).** Legacy `bloom_classifier_disagreement` ensemble member; NEVER loaded on the trivote path (re-founded on the generator's own asserted level + the already-licensed DeBERTa zero-shot voter + the deterministic verb ontology). Still referenced by the flag-OFF legacy path (inference-only, no live inference — the ensemble degrades to `unknown`). Verify or drop before any redistribution that bundles weights |
| `distilbert-base-uncased-finetuned-sst-2-english` | Apache 2.0 | **RETIRED under `ED4ALL_BLOOM_TRIVOTE` (default OFF).** Legacy Bloom ensemble sentiment member (mapped onto Bloom by a low-resolution heuristic); NEVER loaded on the trivote path. Retained only for flag-OFF legacy byte-identity |

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
fully offline). Resolution reads the W-D12 `_OPENAI_COMPATIBLE_PROVIDERS`
registry but enforces that the resolved `base_url` host is loopback; a non-
loopback resolution raises `AnswerProviderNotLocal`. There is no escape-hatch
env for cloud answer routing. Any future *additional local* provider entry must
land with a row here.

| Flag/value | Default model | Model license | ToS layer | Training-data permitted | Recommended use |
|------------|---------------|---------------|-----------|-------------------------|-----------------|
| `ED4ALL_ANSWER_PROVIDER=local` | `qwen2.5:7b-instruct-q4_K_M` (via `ED4ALL_ANSWER_MODEL` → `LOCAL_SYNTHESIS_MODEL`) | Apache 2.0 | N/A (your hardware; loopback-enforced) | N/A — runtime Q&A inference; outputs are ephemeral learner answers, never corpus content | **Only permitted value in Phase IA.** Non-loopback resolution raises `AnswerProviderNotLocal`. |
| `ED4ALL_ASSISTANT_BASE_URL` / `ED4ALL_ASSISTANT_MODEL` (the `ed4all assistant` seat) | `nemotron-3-nano` on a local vLLM seat (`http://localhost:8004/v1`) | NVIDIA Nemotron Open Model License | N/A (your hardware; loopback-enforced — non-loopback resolution raises `AssistantProviderNotLocal`) | N/A — runtime operator-help surface (status / run start-stop / curated help chat); outputs are ephemeral operator replies, NEVER a training-data producer and never re-ingested as corpus content | Operator-assistant chat only (`lib/assistant/`); sandboxed to a typed tool whitelist, no shell / file access. |

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

- **Anthropic / Claude Session** — Outputs are restricted from training-data use under Anthropic's ToS. The pipeline keeps these providers wired for backward compatibility and for callers who have a separate written agreement with Anthropic permitting derivative training, but the **default recommended path is NOT to use them for training-data synthesis**. The Wave 107 critical-severity `LibV2ModelValidator::MOCK_PROVIDER_CORPUS` check fails closed on `provider="mock"` corpora; analogous operator discipline is required for `provider="anthropic"` or `provider="claude_session"` runs that intend to train.
- **Together AI** — Together's ToS explicitly permits using outputs for training-data generation; the underlying OSS model license still governs distribution of the model and any derivatives. Both layers must be cited (ToS + model license). Llama-3.3 requires attribution and a >700M-MAU special license; Qwen2.5-72B requires written permission for >100M-MAU commercial use; DeepSeek-V3 carries its own permissive license.
- **Local OSS** — Output license is the underlying model's license, full stop. Apache 2.0 (Qwen2.5-7B/14B/32B, Mistral-Small) is the cleanest: unrestricted commercial use including using outputs to train derivative models, and no attribution required for outputs (only for redistributing the model itself). Llama-3.3 requires attribution. Qwen2.5-72B's Qwen License Agreement permits outputs for derivative training at any scale but gates >100M-MAU commercial use of the model.
- **Hosted-cloud registry stubs (`groq` / `fireworks` / `deepseek`)** — the `<registry>` rows above are illustrative hosted-cloud seats in the `_OPENAI_COMPATIBLE_PROVIDERS` registry (groq / fireworks serve Llama-3.3-70B; deepseek serves `deepseek-chat`), reachable via any `*_PROVIDER=<registry>` selector. They carry **cloud (not local) licensing exposure**: the corpus is NOT license-clean-local — a hosted OSS teacher is ToS-gated (the serving provider's terms) AND the underlying model license applies, exactly like the `together` category. **W9.1 provenance-correctness fix:** these three seats were previously recorded in the endpoint registry (`config/endpoints.yaml`) with `provenance_provider: local`, which was a licensing lie — a synthesized pair authored through one of them stamped `out["provider"] = "local"`, telling a downstream licensing auditor the artifact came from an Apache-2.0 local model when it in fact came from a hosted-cloud API. Their `provenance_provider` now maps to the existing cloud `together` provenance value (`Trainforge/generators/_synthesis_provider.py` stamps the registry row's `provenance_provider`, not the raw endpoint name), so recorded corpus provenance is truthful. Mapping onto the pre-existing `together` value keeps the closed `Touch.provider` provenance set (`{anthropic, claude_session, deterministic, local, nvidia, together}`) frozen — no new provider/model row, no schema/enum churn. Regression net: `Trainforge/tests/test_synthesis_provenance_stamp.py` + `tests/test_endpoint_registry_drift_guard.py`.

### Curriculum alignment shares the synthesis provider stack

The synthesis providers above are also the LLM stack consumed by the **curriculum-alignment surface** (`Trainforge/align_chunks.py` teaching-role classification via `Trainforge/generators/_curriculum_provider.py::CurriculumAlignmentProvider`). The `CURRICULUM_ALIGNMENT_PROVIDER` env var (and the `--curriculum-provider` CLI flag on `align_chunks`) accepts the same `anthropic` / `together` / `local` values, and the `local` and `together` branches reuse the same `LOCAL_SYNTHESIS_*` / `TOGETHER_*` env vars so one local server serves both task surfaces. Curriculum alignment writes corpus metadata (the `teaching_role` field on every chunk) — i.e. it touches training data — so the same ToS calculus applies.

**Recommended setting for both surfaces is `local`** (Apache 2.0 Qwen) for a ToS-clean corpus end-to-end. Wave 137 followup wired `CURRICULUM_ALIGNMENT_PROVIDER` to fire from the production `align_chunks.main()` CLI invocation path (previously it only fired when the provider class was constructed by hand), so setting the env var in the workflow environment is now sufficient to redirect the curriculum-alignment surface away from Anthropic.

### Courseforge content-generator shares the synthesis provider stack

The Courseforge content-generator surface (`Courseforge/generators/_provider.py::ContentGeneratorProvider`, instantiated from `MCP/tools/pipeline_tools.py::_generate_course_content`) reuses the same provider stack and env vars as Trainforge synthesis: the `local` and `together` branches read `LOCAL_SYNTHESIS_BASE_URL` / `LOCAL_SYNTHESIS_MODEL` / `LOCAL_SYNTHESIS_API_KEY` and `TOGETHER_API_KEY` / `TOGETHER_SYNTHESIS_MODEL` respectively, and the `anthropic` branch reads `ANTHROPIC_API_KEY` / `ANTHROPIC_SYNTHESIS_MODEL` — so a single local server serves all three surfaces (synthesis, curriculum alignment, content generation). Every page authored through this provider emits one `content_generator_call` decision event with `provider`, `model`, `page_id`, and retry count, so a post-hoc audit can attribute every HTML chunk to its provider. **Wave-74 short-circuit semantics:** setting `COURSEFORGE_PROVIDER` to any non-empty value (`anthropic` / `together` / `local`) overrides `ED4ALL_AGENT_DISPATCH=true` for the `content-generator` agent only — the executor falls through to the in-process provider call instead of dispatching the Claude Code subagent, while every other Wave-74 agent (course-outliner, oscqr-course-evaluator, etc.) keeps dispatching unchanged. **Recommended setting for ToS-clean Courseforge content is `COURSEFORGE_PROVIDER=local`** (Apache 2.0 Qwen) so the authored HTML — which Trainforge later ingests as training chunks — is license-clean from end to end.

### Deterministic generators (no LLM exposure)

The four generators in `Trainforge/generators/` that emit deterministic pairs without any LLM call — `kg_metadata_generator.py`, `violation_generator.py` (Wave 125a, pyshacl-oracle-verified), `abstention_generator.py` (Wave 124), `schema_translation_generator.py` (Wave 125b) — are fully off-grid for ToS analysis. No provider's terms apply because no provider is invoked. Their pairs are derived from the course's pedagogy graph + property manifest + SHACL fixtures, all of which are project-internal. Pairs from these generators are licence-clean regardless of which `--provider` is selected for the paraphrase loop.

---

## Decision tree

If you are building a course-pinned SLM and want a license-clean training corpus:

1. **First choice:** `--provider local` with `LOCAL_SYNTHESIS_MODEL=qwen2.5:7b-instruct-q4_K_M` (Apache 2.0). Fits an 8 GB GPU in 4-bit. Outputs are unrestricted; training a derivative SLM on these paraphrases is fully permitted.
2. **If hardware can't run 7B locally:** `--provider together` with a hosted Apache 2.0 OSS model (Qwen2.5-72B-Instruct-Turbo) or the default Llama-3.3-70B. Both are ToS-clean for training-data generation.
3. **Do NOT use** `--provider anthropic` or `--provider claude_session` for training data unless you have separately obtained written permission from Anthropic. Pipeline default is to route around them.
4. **Do NOT use** `--provider mock` for any corpus you intend to train on. Mock is a deterministic 30-template factory wired for plumbing tests; the Wave 107 `MOCK_PROVIDER_CORPUS` validator fails closed on promotion.
5. **When unsure:** read the model's `LICENSE` file on Hugging Face and the provider's current ToS before kicking off a multi-hundred-dispatch run.

---

## Pipeline guarantees

The project's posture is encoded into the validation gates and provider invariants:

- **Wave 112 sentinel-phrase hardening** removed the `"This passage anchors the answer in the source material."` filler that previously injected on short paraphrases. Any training pair that would have carried sentinel filler now triggers a re-paraphrase or a fail-loud `SynthesisProviderError`. See `Trainforge/CLAUDE.md` § "Synthesis pipeline integrity invariants (Wave 112)".
- **Wave 113 `LocalSynthesisProvider`** (`Trainforge/generators/_local_provider.py`) was added precisely so the training corpus can be license-clean from end to end. The provider speaks the OpenAI chat-completions protocol against any local server (Ollama / vLLM / llama.cpp / LM Studio); the underlying model license is the only ToS layer that applies.
- **Anthropic providers stay wired** for backward compatibility but are no longer the recommended default for training data. The `synthesis_provider_call` decision-capture event records which provider produced each pair, so a post-hoc audit can identify any rows that crossed a ToS boundary.
- **Per-call audit trail** — every provider call emits a `synthesis_provider_call` decision event with model ID, max_tokens, prompt-cache hit/miss, and retry count. The full provider × model history of a corpus is reconstructible from `training-captures/trainforge/<COURSE_CODE>/`.
- **SemantiK conversion is license-clean by construction.** The SemantiK PDF → accessible-HTML conversion stage carries **no PyMuPDF/MuPDF (AGPL-3) and no Poppler (GPL-2)** on its extraction path — that copyleft liability is removed. The subsystem ships Apache-2.0 (`SemantiK/LICENSE`) and runs fully offline by default (`SEMANTIK_SPECIALIST_PROVIDER=local`). The only ToS layers on the conversion path are the opt-in hosted-70B quality seat and the opt-in LOCAL VLM extraction seat (`SEMANTIK_VLM_EXTRACT` → Qwen2.5-VL-7B, Apache-2.0, default off and local — only a non-local override carries endpoint ToS exposure), both documented in the Synthesis providers table above, and even those produce published product content, not training data. The default remains fully offline.

---

## When in doubt

- Read the model's `LICENSE` file on Hugging Face. The Hugging Face URLs above are the authoritative source — license terms can change between model releases.
- Read the provider's current ToS — Anthropic, OpenAI, and Together evolve their terms, and an old reading may be stale.
- If the use case is novel (multi-modal training, fine-tuning a frontier model from another's outputs, redistribution of derived weights, hosting an adapter for paid inference), consult counsel. This document is engineering documentation, not legal advice.

---

## Maintenance contract

Any new behavior flag in `CLAUDE.md` § "Opt-In Behavior Flags" that selects an LLM provider, model ID, or synthesis backend MUST land with a corresponding row in this file's "Synthesis providers" table (or a one-line entry in "Tooling" if it doesn't touch training data). Drift between this file and the per-provider rows in `CLAUDE.md` / `Trainforge/CLAUDE.md` is a documentation bug.
