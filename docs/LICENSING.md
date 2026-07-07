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
| `local` (Qwen 72B) | `qwen2.5:72b-instruct-q4_K_M` | Qwen License Agreement | N/A | **Yes** (outputs unrestricted at any scale) | Highest OSS quality, A100 / multi-GPU |
| `local` (Llama 70B) | `llama3.3:70b-instruct-q4_K_M` | Llama 3.3 Community License | N/A | **Yes** (with attribution) | Strong instruction following |
| `local` (Mistral 24B) | `mistral-small:24b-instruct-q4_K_M` | Apache 2.0 | N/A | **Yes** | Faster on 16 GB GPU |
| `local` (Phi-3.5 mini) | `phi3.5:3.8b-mini-instruct-q4_K_M` | MIT | N/A | **Yes** | Smallest OSS option |
| `COURSEFORGE_BLOCK_ROUTING_PATH=Courseforge/config/block_routing.license_clean.yaml` | n/a (router-level YAML; large tier defaults to `qwen2.5:32b-instruct-q4_K_M`) | Apache 2.0 (when `large` resolves to local Qwen) | n/a (your hardware) — or Together AI ToS when `large` is swapped to a hosted Apache-2.0 OSS model | **Yes** | **Recommended for ToS-clean Courseforge two-pass** runs. Sibling YAML override that swaps the canonical `large` capability tier (`claude-sonnet-4-6`) for a local 32B Qwen. See `docs/operations/license-clean-run.md` for the 4-env-var deployment recipe and the calibration prerequisite for `assessment_item` distractor quality. |
| `COURSEFORGE_BLOCK_ROUTING_PATH=Courseforge/config/block_routing.nvidia_large.yaml` (key from `NVIDIA_API_KEY`; base_url/model via `NVIDIA_BASE_URL` / `NVIDIA_LARGE_MODEL`) | `meta/llama-3.3-70b-instruct` (large tier only; small/medium stay local Qwen 7B/14B) | NVIDIA model / catalog license | NVIDIA hosted-API ToS | **N/A — generates COURSE CONTENT (product), not training-data corpus** | Router-level YAML sibling of the license-clean variant. The small + medium capability tiers are byte-identical local 7B/14B Qwen (Apache 2.0, loopback-only); ONLY the rewrite `large` / escalation tier routes to NVIDIA's hosted OpenAI-compatible inference API. **This row is content-generation, not synthesis:** the NVIDIA-authored HTML is published product. If that HTML is later re-ingested by Trainforge as training chunks, the standard content→training-data caveat applies (see "Courseforge content-generator shares the synthesis provider stack" below) — operators wanting a fully ToS-clean training corpus should keep this surface on a license-clean local/Together provider. |
| `SEMANTIK_SPECIALIST_PROVIDER=<endpoint>` + `SEMANTIK_SPECIALIST_MODEL` / `SEMANTIK_STRUCTURE_REVIEW_MODEL` (key from `NVIDIA_API_KEY`; base via `NVIDIA_BASE_URL`; model resolves through `NVIDIA_LARGE_MODEL`) | `meta/llama-3.3-70b-instruct` (Stage-6 specialist generation + Stage-5d structure-reviewer seats; the council adapters / theta stages stay local) | NVIDIA model / catalog license (Llama 3.3 Community License underneath) | NVIDIA hosted-API ToS | **N/A — generates DART-replacement STRUCTURED CONTENT (product), not training-data corpus** | SemantiK (the DART replacement) endpoint seat. `SEMANTIK_SPECIALIST_PROVIDER` defaults to `local` (in-process GGUF council adapters — no network, no key); ONLY a non-local value routes the Stage-6 specialist fill + the optional Stage-5d 70B structure-reviewer (`SEMANTIK_STRUCTURE_REVIEW`, default off) to NVIDIA's hosted OpenAI-compatible API. The OCR / theta / extraction stages remain local-only. **This row is content-generation, not synthesis:** the produced HTML/structure is published product. If that output is later re-ingested by Trainforge as training chunks, the standard content→training-data caveat applies — operators wanting a fully ToS-clean training corpus should keep `SEMANTIK_SPECIALIST_PROVIDER=local`. Flag detail: `SemantiK/CLAUDE.md § Opt-In Behavior Flags`. |
| `SEMANTIK_VLM_PROVIDER` / `SEMANTIK_VLM_MODEL` (base via `SEMANTIK_VLM_BASE_URL`, default `http://localhost:11434`; key from `SEMANTIK_VLM_API_KEY` — none for the local/loopback seat) | `qwen2.5vl:7b` = **Qwen2.5-VL-7B-Instruct** (the P0 VLM extraction-source seat, gated by `SEMANTIK_VLM_EXTRACT`, default off; the council BERTs / OCR / theta stages stay local) | **Apache-2.0** (the 7B VL is Apache-2.0 — NOT the Qwen-licensed 72B VL; pin the 7B) | N/A — local-by-default (loopback ollama; a non-local seat is opt-in and carries that endpoint's ToS) | **N/A — generates DART-replacement STRUCTURED CONTENT (product), not training-data corpus** | SemantiK's opt-in LOCAL VLM extraction seat: a per-page image → Qwen2.5-VL markdown transcription fused as a fourth extraction source (provider-agnostic OpenAI-compatible, mirroring `SEMANTIK_SPECIALIST_*` — a new provider is env config, never a subclass; the same seat serves the local ollama GPU and a Spark endpoint later). Default OFF and local (`qwen2.5vl:7b` on loopback ollama), so the default posture stays fully offline. **This row is content-generation, not synthesis:** the transcribed structure is published product. If that output is later re-ingested by Trainforge as training chunks, the standard content→training-data caveat applies — keep the seat local for a fully ToS-clean corpus. MiniCPM-V is excluded on license (plan bake-off note). Flag detail: `SemantiK/CLAUDE.md § Opt-In Behavior Flags`. |
| `SEMANTIK_SEMANTIC_SUBCLASS` (seat via `LOCAL_SYNTHESIS_BASE_URL` / `LOCAL_SYNTHESIS_MODEL` / `LOCAL_SYNTHESIS_API_KEY`) | `qwen2.5:7b-instruct-q4_K_M` (the EXISTING local reviewer seat; Build #23 Tier-3 composite-unit subclassifier, `lib/semantik/subclassifier.py`, gated by `SEMANTIK_SEMANTIC_SUBCLASS`, default off) | **Apache-2.0** (Qwen2.5-7B default; any `LOCAL_SYNTHESIS_MODEL` override carries that model's license) | N/A — local-by-default (loopback Ollama-style OpenAI-compatible server; a non-local override carries that endpoint's ToS) | **N/A — emits a metadata LABEL (`data-dart-subclass` / chunk `unit_subclass`), not prose training-data corpus** | SemantiK's opt-in Tier-3 composite-unit subclass pass reuses the SAME license-clean local reviewer seat as `SEMANTIK_SPECIALIST_PROVIDER=local` / the Trainforge `local` synthesis seat (env-resolved via `LOCAL_SYNTHESIS_*`, no hardcoded endpoint). It classifies each already-rendered composite unit into a kebab-case subclass label added payload-only to the published HTML; the label rides into chunks as the additive `unit_subclass` metadata field (never prose text, never a chunk-text/id change). Default OFF → no dispatch. Keep the seat local (the byte-stable default) for a fully ToS-clean corpus. Flag detail: `SemantiK/CLAUDE.md § Opt-In Behavior Flags`. |

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
- **SemantiK conversion is license-clean by construction.** The PDF → accessible-HTML conversion stage (SemantiK, the replacement for the retired DART converter) carries **no PyMuPDF/MuPDF (AGPL-3) and no Poppler (GPL-2)** on its extraction path — that copyleft liability is removed. The subsystem ships Apache-2.0 (`SemantiK/LICENSE`) and runs fully offline by default (`SEMANTIK_SPECIALIST_PROVIDER=local`). The only ToS layers on the conversion path are the opt-in hosted-70B quality seat and the opt-in LOCAL VLM extraction seat (`SEMANTIK_VLM_EXTRACT` → Qwen2.5-VL-7B, Apache-2.0, default off and local — only a non-local override carries endpoint ToS exposure), both documented in the Synthesis providers table above, and even those produce published product content, not training data. The default remains fully offline.

---

## When in doubt

- Read the model's `LICENSE` file on Hugging Face. The Hugging Face URLs above are the authoritative source — license terms can change between model releases.
- Read the provider's current ToS — Anthropic, OpenAI, and Together evolve their terms, and an old reading may be stale.
- If the use case is novel (multi-modal training, fine-tuning a frontier model from another's outputs, redistribution of derived weights, hosting an adapter for paid inference), consult counsel. This document is engineering documentation, not legal advice.

---

## Maintenance contract

Any new behavior flag in `CLAUDE.md` § "Opt-In Behavior Flags" that selects an LLM provider, model ID, or synthesis backend MUST land with a corresponding row in this file's "Synthesis providers" table (or a one-line entry in "Tooling" if it doesn't touch training data). Drift between this file and the per-provider rows in `CLAUDE.md` / `Trainforge/CLAUDE.md` is a documentation bug.
