# AGENTS.md — Codex Project Context for Ed4All

This file is the OpenAI Codex CLI counterpart to `CLAUDE.md`. Codex reads it
at the start of every session in this project. The deep-dive reference is
still `CLAUDE.md` and the per-component guides under
`DART/CLAUDE.md`, `Courseforge/CLAUDE.md`, `Trainforge/CLAUDE.md`,
`LibV2/CLAUDE.md`, and `schemas/ONTOLOGY.md`. **Do not duplicate them here
— link.**

The headline reason this file exists: **Codex is the agent of choice for
running paraphrase synthesis against a local model server** (Ollama / vLLM /
llama.cpp), bypassing the ToS-restricted Anthropic provider and the paid
Together AI provider. Section 3 is the load-bearing section.

---

## 1. Project overview

Ed4All is a hybrid orchestrator that turns textbooks into accessible online
courses, then trains course-pinned small language models on the resulting
corpora. Four components, one CLI:

| Component | Purpose | Deep dive |
|-----------|---------|-----------|
| **DART** | PDF → accessible HTML (multi-source synthesis, WCAG 2.2 AA, source-block provenance) | `DART/CLAUDE.md` |
| **Courseforge** | HTML → modular course content + IMSCC packaging for LMS import | `Courseforge/CLAUDE.md` |
| **Trainforge** | IMSCC → RAG corpus + assessments + instruction/preference training pairs | `Trainforge/CLAUDE.md` |
| **LibV2** | Course archive + post-import training stage (QLoRA adapters, model cards) | `LibV2/CLAUDE.md` |

### Unified CLI

```bash
ed4all run <workflow> --corpus <PATH> --course-name <NAME> [--mode local|api]
ed4all run textbook-to-course --corpus textbook.pdf --course-name PHYS_101
ed4all run rag_training       --corpus course.imscc --course-name CHEM_101
ed4all run trainforge_train   --course-code <slug>  --base-model qwen2.5-1.5b
```

Three modes:
- `--mode local` (default): Claude Code session as LLM; subagent dispatch
- `--mode api`: Anthropic SDK direct (needs `ANTHROPIC_API_KEY`)
- `trainforge_train` workflow: post-import LibV2 stage; trains a QLoRA
  adapter against the course's RAG + training-specs

### Top-level layout (abridged)

```
Ed4All/
├── DART/         # PDF → HTML
├── Courseforge/  # HTML → modules → IMSCC
├── Trainforge/   # IMSCC → RAG + assessments + training pairs
│   ├── synthesize_training.py
│   ├── generators/   # provider implementations (anthropic, claude_session, together, local)
│   └── scripts/pilot_synthesis.py
├── LibV2/        # course archive + training stage
├── MCP/          # FastMCP server, orchestrator core, hardening, IPC
├── cli/          # `ed4all` entry point
├── config/       # workflows.yaml, agents.yaml
├── lib/          # validators, ontology, decision_capture
├── schemas/      # JSON schemas (knowledge, models, taxonomies, context/SHACL)
├── plans/        # wave plans (e.g., 2026-04-28-pipeline-hardening-rebuild-train.md)
└── state/        # GENERATION_PROGRESS.md, runs/, checkpoints/
```

---

## 2. Working conventions

These are the invariants. Violate any one of them and CI rejects the change.

**Branching**
- Current branch: `dev-v0.3.0`. **Do NOT merge to `main`** without
  explicit operator authorization. Do not push without operator
  authorization.
- Commits are wave-based: `Wave NNN — short description`. TDD discipline:
  red test → minimal patch → green → commit. One concern per commit.

**Validation gates (do not bypass)**
- Source of truth: `config/workflows.yaml::validation_gates`. Every phase
  has gate(s) wired against `lib/validators/*`.
- A failing gate means the underlying artifact is wrong. Fix the root
  cause; do not lower the threshold or downgrade the severity.
- The full gate matrix is documented in `CLAUDE.md` § "Active Gates".

**Decision capture (mandatory at every LLM call site)**
- Use `lib/decision_capture.py::DecisionCapture`. Emit at least one event
  per call (per batch when batched).
- Required fields: `decision_type`, `decision`, `rationale` (≥20 chars,
  must reference *dynamic* signals — block IDs, page numbers, model ID,
  max_tokens, confidence distributions — not boilerplate).
- Every new LLM call site needs a regression test asserting the capture
  fires. See precedents in `CLAUDE.md` § "LLM call-site instrumentation".

**File-by-file editing**
- One agent per file. No shared writes. Use file locking for state files.
- Maximum 10 simultaneous task dispatches per batch. Wait for the batch
  before starting the next.

**Synthesis invariants (Wave 112 — currently live)**
- No sentinel filler ("[paraphrase]", "TODO", "Lorem ipsum") in any
  emitted training pair. Fail loud.
- No empty fields. Schema validation runs on every emit.
- Length-clamped paraphrase outputs (per-template min/max char bounds).
- Decision-capture emitted on every provider call.
- Wave 113 layered four post-synthesis gates on top: `synthesis_quota`,
  `min_edge_count`, `synthesis_diversity`, `property_coverage`. All in
  `lib/validators/`.

---

## 3. Local Model Paraphrase Generation — primary contribution

This section is why the file exists. Read it carefully before running any
synthesis pass.

### 3.1 Why local

| Provider | ToS for training data | Cost | Air-gapped | Hardware |
|----------|-----------------------|------|------------|----------|
| `--provider anthropic` | **Restricted** — Claude outputs cannot train derivative models | Per-token | No | None |
| `--provider claude_session` | **Restricted** — same ToS | Subscription | No | None |
| `--provider together` | Clean (OSS models, Together's ToS) | Per-token | No | None |
| `--provider local` | **Clean** — your hardware, your weights | Free per call | **Yes** | 24GB+ GPU |

**Use `--provider local` for every paraphrase pass that produces training
data.** It is fully reproducible (same model + same seed = same output),
zero ToS exposure, zero per-call cost, and zero network dependency.
Trade-off: ~15–30 min one-time setup and a 24GB+ GPU (more for larger
models).

### 3.2 Recommended local-server stacks

Pick one. All four expose an OpenAI-compatible `/v1/chat/completions`
endpoint, which is what `LocalSynthesisProvider` speaks.

| Stack | Best for | Setup |
|-------|----------|-------|
| **Ollama** | Easiest. systemd service, automatic GPU detection, model registry. 24GB+ GPU. | install: `curl -fsSL https://ollama.com/install.sh \| sh` <br> start: `ollama serve` (or auto via systemd) <br> pull: `ollama pull qwen2.5:32b-instruct-q4_K_M` <br> endpoint: `http://localhost:11434/v1` |
| **vLLM** | Production. Fastest throughput, tensor parallelism for multi-GPU. | install: `pip install vllm` <br> start: `vllm serve Qwen/Qwen2.5-32B-Instruct --quantization awq --port 8000` <br> endpoint: `http://localhost:8000/v1` |
| **llama.cpp server** | CPU-only / edge. GGUF quantized models. | build llama.cpp from source <br> start: `./build/bin/llama-server -m model.gguf --port 8080` <br> endpoint: `http://localhost:8080/v1` |
| **LM Studio** | GUI. Model swap by clicking. Good for interactive tinkering. | desktop app → "Local Server" tab <br> endpoint: `http://localhost:1234/v1` |

### 3.3 Recommended models (Ollama tags shown; vLLM uses HF repo IDs)

| Model | Size (4-bit) | License | Hardware | Notes |
|-------|--------------|---------|----------|-------|
| `qwen2.5:32b-instruct-q4_K_M` | ~18 GB | Apache 2.0 | 24 GB GPU | **Top recommendation for RDF/SHACL paraphrase.** Strong technical-text rewriting, training-permitted, fits a single RTX 3090/4090/L4. |
| `qwen2.5:72b-instruct-q4_K_M` | ~40 GB | Apache 2.0 | A100 40GB / L40S / 2× 24GB w/ tensor parallel | Highest quality OSS option. |
| `llama3.3:70b-instruct-q4_K_M` | ~40 GB | Llama 3.3 (training-permitted) | A100 40GB / 2× 24GB | Very strong on instruction following. |
| `mistral-small:24b-instruct-q4_K_M` | ~14 GB | Apache 2.0 | 16 GB GPU | Faster, slightly lower quality. |
| `qwen2.5:14b-instruct-q4_K_M` | ~8 GB | Apache 2.0 | 12 GB GPU | Iteration speed; use for pilot only. |

For RDF/SHACL/ontology paraphrase work, default to **qwen2.5:32b** unless
hardware forces a smaller model.

### 3.4 Wiring a local server into Trainforge

The sibling worker is landing `LocalSynthesisProvider` in
`Trainforge/generators/_local_provider.py`. By the time you read this it
exists. Selected via `--provider local`. Speaks the OpenAI chat-completions
protocol against `LOCAL_SYNTHESIS_BASE_URL`.

```bash
# Terminal 1 — start the local model server
ollama serve
# or
# vllm serve Qwen/Qwen2.5-32B-Instruct --quantization awq --port 8000

# Terminal 2 — pull a model (Ollama only; vLLM downloads on first request)
ollama pull qwen2.5:32b-instruct-q4_K_M

# Terminal 3 — env vars + Trainforge invocation
export LOCAL_SYNTHESIS_BASE_URL="http://localhost:11434/v1"   # Ollama default
export LOCAL_SYNTHESIS_MODEL="qwen2.5:32b-instruct-q4_K_M"

# Pilot run first (always)
python Trainforge/scripts/pilot_synthesis.py \
    --corpus LibV2/courses/<course-slug> \
    --course-code <course-slug> \
    --provider local \
    --max-pairs 30 \
    --seed 11

# Full corpus rebuild (only after the pilot is clean)
python -m Trainforge.synthesize_training \
    --corpus-dir LibV2/courses/<course-slug>/corpus \
    --course-code <course-slug> \
    --provider local \
    --seed 11
```

Endpoint defaults by stack:
- Ollama → `http://localhost:11434/v1`
- vLLM → `http://localhost:8000/v1`
- llama.cpp → `http://localhost:8080/v1`
- LM Studio → `http://localhost:1234/v1`

### 3.5 Operational notes

**Model-cache warm-start.** The first request after `ollama serve` boot
incurs a 10–30 s model-load delay (weights pulled from disk into VRAM).
Smoke-test before kicking off a multi-hundred-dispatch corpus rebuild:

```bash
curl -X POST $LOCAL_SYNTHESIS_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:32b-instruct-q4_K_M",
    "messages": [{"role":"user","content":"Paraphrase: SHACL is a constraint language."}],
    "max_tokens": 50
  }'
```

A 200 with a non-empty paraphrase = ready. A 404 = model not pulled. A
connection refused = server not started.

**vLLM tensor-parallel for multi-GPU.** Two 24 GB GPUs can host a 70B
model:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --quantization awq \
    --port 8000
```

**Concurrency.** Ollama serializes by default; vLLM batches requests
internally. For maximum throughput on a corpus rebuild, prefer vLLM.

**Determinism.** Pass `--seed 11` (or whatever seed the wave plan
specifies) on every run. The provider should forward the seed to the
server (vLLM accepts `seed` in the request body; Ollama respects
`options.seed`). Same model + same seed + same prompts = byte-identical
training pairs.

### 3.6 Verification protocol before a full rebuild

1. **Smoke-test the server.** `curl` against `/chat/completions` (above).
   200 with paraphrase = good.
2. **Pilot.** `pilot_synthesis.py --provider local --max-pairs 30`.
   Read `pilot_report.md`. Confirm:
   - Zero sentinel-phrase leaks (Wave 112 invariant)
   - All target property surface forms covered (Wave 113
     `property_coverage` gate)
   - Diversity within bounds (Wave 113 `synthesis_diversity` gate:
     top-3 templates ≤60% of pairs, single template ≤35%, distinct
     templates ≥8)
   - Length-clamped outputs, no empties
3. **Full corpus** only after the pilot is clean. Re-run the four Wave
   113 gates post-rebuild via the standard validation phase.

### 3.7 Quality calibration

Local 32B models are roughly 85–90% as strong as Claude Sonnet on
technical paraphrase tasks. If post-rebuild gates fail:
- First escalation: 70B model (qwen2.5:72b or llama3.3:70b) with
  tensor-parallel.
- Second escalation: `--provider together` for a hosted larger OSS model
  (still ToS-clean, just paid + networked).
- Do **not** fall back to `--provider anthropic` or `claude_session` for
  training-data synthesis — those are ToS-restricted for this use.

---

## 4. Trainforge architecture map

Just enough skeleton to navigate. Full details in `Trainforge/CLAUDE.md`.

**Entry point:** `Trainforge/synthesize_training.py` — picks a provider via
`--provider`. CLI surface mirrors `Trainforge.process_course` for the RAG
half.

**Providers** (`Trainforge/generators/`):

| File | Provider | ToS | Use case |
|------|----------|-----|----------|
| `_anthropic_provider.py` | Anthropic API | Restricted | Non-training synthesis only |
| `_claude_session_provider.py` | Claude Code session bridge | Restricted | Non-training synthesis only |
| `_together_provider.py` | Together AI hosted OSS | Clean | Paid, networked alternative |
| `_local_provider.py` | Local OpenAI-compatible server | **Clean** | **Default for paraphrase / training-data work** |

**Shared invariants for all providers:**
- Length-clamp outputs (per-template min/max chars)
- Fail-loud on empty/short responses (no sentinel filler)
- Emit one `synthesis_provider_call` decision-capture event per call,
  with rationale referencing model ID, max_tokens, prompt block ID
- Schema-validate every emitted training pair before write

**Wave 113 gates** (post-synthesis, all under `lib/validators/`):

| Gate | Validator | Purpose |
|------|-----------|---------|
| `synthesis_quota` | `synthesis_quota.py` | Minimum pairs per chunk / property |
| `min_edge_count` | `min_edge_count.py` | Pre-synthesis: ≥100 edges, ≥4 edge types, ≥50 nodes |
| `synthesis_diversity` | `synthesis_diversity.py` | Top-3 templates ≤60%, single ≤35%, distinct ≥8 |
| `property_coverage` | (Wave 109+) | Every target property surface form covered |

These are the gates that decide whether a corpus is shippable to the
training stage. Pilot small, gate hard, then go big.

**Where the synthesized pairs land:**
`LibV2/courses/<course-slug>/training_specs/*.jsonl` — consumed by the
post-import training stage (`ed4all run trainforge_train`).

---

## 5. When NOT to use Codex for this work

Codex is good at scoped, scriptable work. It is not a drop-in for Claude
Code on this project. Be honest about the lines.

**Use Codex for:**
- Pure code refactors with clear before/after
- Writing tests (especially decision-capture regression tests)
- Small feature additions in a single component
- **Running validators and pilot syntheses end-to-end** — `python
  Trainforge/scripts/pilot_synthesis.py --provider local …`, watch the
  output, summarize `pilot_report.md`. This is the load-bearing use case
  this file enables.
- Reading and summarizing CLAUDE.md sections on demand
- Local-model server orchestration (start/stop/swap models)

**Avoid Codex for:**
- Anything that dispatches subagents through MCP. Trainforge's
  `LocalDispatcher` and the textbook-to-course phase workers depend on
  the parent agent's MCP tooling, which is Claude Code-specific. Codex
  can run scripts, but cross-agent dispatch is not its lane.
- Live ToS interpretation. If a question of "can we use this model's
  output for training" comes up, defer to the operator. Do not improvise.
- Full Wave 113 manual review. The operator must read `pilot_report.md`
  and the gate outputs and make the go/no-go call. Codex summarizes;
  operator decides.
- Wave 114 (training stage). That requires a GPU + manual eval review +
  promotion-ledger update. Out of scope for this file.

---

## 6. Licensing context

Read `docs/LICENSING.md` before kicking off any training-data synthesis run. Codex's posture in this project is shaped by an asymmetry between **dev tooling** and **synthesis providers**:

- **Codex's role is orchestration.** It runs scripts (`pilot_synthesis.py`, `synthesize_training.py`), reads files, dispatches shell commands, summarizes pilot reports. It does NOT generate training data.
- **The local Qwen / Together-hosted OSS model generates training data**, not Codex. That separation matters because the trained SLM is a derivative work of the synthesis provider's outputs — not Codex's.
- **OpenAI's Services Terms** (https://openai.com/policies/services-terms/) and **Business Terms** (https://openai.com/policies/business-terms/) restrict using Codex outputs to train competing models. Same restriction shape as Anthropic's Consumer/Commercial Terms on Claude Code. For Ed4All's use case this is a non-issue — Codex output is code and shell invocations, not training data.
- **The pipeline routes training-data synthesis through `--provider local` or `--provider together` by design**, not Anthropic. This keeps the training corpus license-clean from end to end. Wave 113's `LocalSynthesisProvider` exists for exactly this reason.

**One-line per default model:**
- `--provider local` default: `qwen2.5:7b-instruct-q4_K_M` (Apache 2.0). Outputs are unrestricted; training a derivative SLM on these paraphrases is fully permitted, no attribution required for outputs.
- `--provider together` default: `meta-llama/Llama-3.3-70B-Instruct-Turbo` (Llama 3.3 Community License + Together AI ToS). Outputs permitted for training-data use under both layers; >700M-MAU commercial use of the model itself requires attribution + special license.
- `--provider anthropic` and `--provider claude_session`: outputs restricted from training-data use. Stay wired for backward compat; **not** the recommended default.

If a question of "can we use this model's output for training" comes up, defer to the operator. Do not improvise. Full table + URLs in `docs/LICENSING.md`.

---

## 7. Operator handoff

If the operator's intent is unclear at any point, **ask**. Auto mode is
not a license to guess on training-data work — bad pairs poison the
adapter and the regression won't surface until eval.

**Reference plan for the current effort:**
`plans/2026-04-28-pipeline-hardening-rebuild-train.md`

**Out of scope for this AGENTS.md:**
- Wave 114 (the actual SLM training run). Requires GPU, manual eval
  matrix review, promotion-ledger update. Operator-driven.
- Schema changes to `schemas/models/model_card.schema.json` or
  `model_pointers.schema.json`. Wave-plan territory.
- Any `git push` or merge to `main`. Operator authorization required.

**In scope for Codex on this branch:**
1. Stand up a local model server (Ollama or vLLM)
2. Pull/serve a recommended model
3. Run `pilot_synthesis.py --provider local`
4. Summarize `pilot_report.md` + the Wave 113 gate outputs
5. On operator green-light, run the full corpus rebuild
6. Commit results as a new `Wave NNN — …` commit on `dev-v0.3.0`

That's the end-to-end. Stop at any failed gate and surface the failure
verbatim — do not paper over it.
