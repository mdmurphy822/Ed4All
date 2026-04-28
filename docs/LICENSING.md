# Licensing & ToS Posture

This document is the canonical reference for Ed4All's licensing and Terms-of-Service posture across the tools and LLM models the project uses. Other docs (`CLAUDE.md`, `AGENTS.md`, `Trainforge/CLAUDE.md`) link here rather than duplicating, so this is the only file that should change when a provider's ToS or a model's license changes.

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
- **Why this is fine for Ed4All:** Claude Code does not generate training data on this project. Training-data synthesis routes through the dedicated providers in `Trainforge/generators/` (see next section). Code Claude writes for the orchestrator does not become a training example.

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

### Citation links (verbatim)

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

---

## When in doubt

- Read the model's `LICENSE` file on Hugging Face. The Hugging Face URLs above are the authoritative source — license terms can change between model releases.
- Read the provider's current ToS — Anthropic, OpenAI, and Together evolve their terms, and an old reading may be stale.
- If the use case is novel (multi-modal training, fine-tuning a frontier model from another's outputs, redistribution of derived weights, hosting an adapter for paid inference), consult counsel. This document is engineering documentation, not legal advice.

---

## Maintenance contract

Any new behavior flag in `CLAUDE.md` § "Opt-In Behavior Flags" that selects an LLM provider, model ID, or synthesis backend MUST land with a corresponding row in this file's "Synthesis providers" table (or a one-line entry in "Tooling" if it doesn't touch training data). Drift between this file and the per-provider rows in `CLAUDE.md` / `Trainforge/CLAUDE.md` is a documentation bug.
