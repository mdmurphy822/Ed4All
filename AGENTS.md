# AGENTS.md — Agent Instructions for Ed4All

Tool-agnostic entry point for any coding agent or agent harness working in this
repository. It defines the **agent surfaces** (what agents exist, where they are
declared, how they are dispatched) and the **invariants** every agent must
respect.

`CLAUDE.md` is the Claude-specific deep dive. Per-subsystem detail lives in the
subsystem `CLAUDE.md` files. **This file links; it does not duplicate.**
Duplicated tables rot — if a number, path, or flag name appears in a subsystem
guide, read it there rather than copying it here.

| Subsystem | Purpose | Deep dive |
|-----------|---------|-----------|
| **SemantiK** | PDF / HTML → accessible HTML (WCAG 2.2 AA, source-block provenance) | `SemantiK/CLAUDE.md` |
| **Courseforge** | Accessible HTML → modular course content + IMSCC packaging | `Courseforge/CLAUDE.md` |
| **Trainforge** | IMSCC → RAG corpus + assessments + instruction/preference pairs | `Trainforge/CLAUDE.md` |
| **LibV2** | Course archive + post-import training stage (adapters, model cards) | `LibV2/CLAUDE.md` |
| Ontology / schemas | Canonical shapes and taxonomies | `schemas/ONTOLOGY.md` |

---

## 1. Where agents are declared

There are **three distinct agent surfaces**. They are not interchangeable, and
conflating them is the most common source of confusion in this repo.

### 1.1 Pipeline agents — `config/agents.yaml`

The workflow registry. **24 agents** under the top-level `agents:` key, plus a
`projects:` map (4 entries: `semantik`, `courseforge`, `trainforge`, `libv2`,
each pointing at a subsystem path + its `CLAUDE.md`) and a `fallback_config:`
block.

Each agent entry uses only these keys: `source`, `type`, `capabilities`,
`max_instances`, `fallback_agents`.

`source` points at a Markdown agent spec. Most resolve to
`Courseforge/agents/*.md` or `Trainforge/agents/*.md`; `semantik-converter`
points at `SemantiK/CLAUDE.md`. Two agents — `textbook-stager` and
`semantik-chunker` — are **implemented in-code with no `.md` spec**, and the
YAML says so inline.

Registered agents, grouped by the directory their spec lives in:

| Group | Agents |
|-------|--------|
| Courseforge specs (`Courseforge/agents/`) | `course-outliner`, `requirements-collector`, `content-generator`, `brightspace-packager`, `oscqr-course-evaluator`, `quality-assurance`, `semantik-automation-coordinator`, `imscc-intake-parser`, `content-analyzer`, `accessibility-remediation`, `content-quality-remediation`, `intelligent-design-mapper`, `remediation-validator`, `textbook-ingestor`, `source-router` |
| Trainforge specs (`Trainforge/agents/`) | `assessment-extractor`, `rag-indexer`, `assessment-generator`, `assessment-validator` |
| Non-`.md` source | `semantik-converter` (→ `SemantiK/CLAUDE.md`), `libv2-archivist` (→ `LibV2/tools/libv2/importer.py`), `training-synthesizer` (→ `Trainforge/synthesize_training.py`) |
| In-code, no `source` at all | `textbook-stager`, `semantik-chunker` |

One spec ships **without** a `config/agents.yaml` entry:
`Trainforge/agents/pedagogy-graph-builder.md`. That agent is referenced directly
from `config/workflows.yaml` (`concept_extraction` phase) and routed in
`MCP/core/executor.py`, so it works — but it will not appear if you enumerate
agents from the registry alone.

Adding a pipeline agent means: write the `.md` spec (or document the in-code
implementation), add the `config/agents.yaml` entry, and wire the phase in
`config/workflows.yaml`. A spec file with no registry entry does nothing.

### 1.2 Dispatch routing — `MCP/core/executor.py`

Agent name alone does not always determine the tool that runs. Nine phases
route by **phase name** rather than agent name via
`MCP/core/executor.py::_PHASE_TOOL_MAPPING`; that mapping cannot be inferred
from YAML. Validator-only phases declare `agents: []` and get a synthesized
virtual `phase-handler` task only when the phase appears in that map. Two of
the nine (`training` → `run_training`, `evaluation` → `run_evaluation`) also
sit in a deterministic-tool set keyed on the resolved tool name, which forces
in-process execution under `ED4ALL_AGENT_DISPATCH` — the subagent fork happens
before the registry lookup and cannot produce an adapter. The canonical list of
those nine phases is in `CLAUDE.md` § "Phase-name dispatch override" — read it
there.

Pipeline-internal tools registered in
`MCP/tools/pipeline_tools.py::_build_tool_registry` but deliberately **not**
decorated `@mcp.tool()` are reachable from workflow phases only, never from an
external MCP client.

### 1.3 Review subagents — `.claude/agents/*.md`

Ten Claude Code subagent definitions, each with YAML frontmatter (`name`,
`description`, `tools`). These are **code-review and audit agents**, entirely
separate from the pipeline registry — they never appear in
`config/agents.yaml` and never run as a workflow phase.

| Subagent | Reviews |
|----------|---------|
| `chunk-emission-reviewer` | Trainforge chunk emission, `lib/ontology/` reuse, `chunk_v4` schema conformance |
| `decision-capture-reviewer` | `DecisionCapture` wiring on new LLM call sites |
| `doc-sanitation-reviewer` | Tracked-doc hygiene, hardcoded slugs, index/count drift |
| `plan-coherence-reviewer` | `plans/*.md` amendments vs git history and live code |
| `semantik-chunk-interrogator` | Converted HTML / `chunks.jsonl` for conversion defects |
| `shacl-shape-reviewer` | SHACL shapes under `schemas/context/` and `lib/validators/shacl/` |
| `slm-evaluator` | Post-training adapter evaluation, promote/hold/reject |
| `training-data-auditor` | Pre-training audit of instruction/preference pairs |
| `training-monitor` | Live diagnostics for in-progress training runs |
| `validation-gate-reviewer` | New/changed validators and their `config/workflows.yaml` wiring |

---

## 2. Invariants

Non-negotiable. CI and the validation gates enforce most of them.

**Branching.** Work happens on the current development branch (currently
`dev-v0.3.1`). Do **not** merge to `main`, `git push`, or open a PR without
explicit operator authorization.

**Validation gates.** Source of truth is
`config/workflows.yaml::validation_gates`; per-gate detail in
`docs/validation/gates.md`. A failing gate means the artifact is wrong — fix
the root cause. Never lower a threshold or downgrade a severity to get green.

**No silent degradation.** Engineer the intended mode to work. A loud failure
is always preferred to a fallback that quietly produces degraded output. Do not
introduce — or document — a silent fallback as if it were intended behavior.
Parse-with-fallback on *operator env input* is the one sanctioned exception and
has its own established vocabulary ("garbage / non-positive → default").

**Decision capture.** Every LLM call site wires
`lib/decision_capture.py::DecisionCapture` and emits at least one event per call
(per batch when batched). `rationale` must be ≥20 chars and interpolate
**dynamic** signals — block IDs, page numbers, model ID, `max_tokens`,
confidence distributions — never static boilerplate. Every new call site lands
with a regression test asserting the capture fires. Canonical shape:
`schemas/events/decision_event.schema.json`; precedents:
`docs/architecture/decision-capture.md`.

**File ownership and batching.** One agent per file; no shared writes; file
locking for state files. Maximum 10 simultaneous task dispatches per batch, and
wait for the whole batch before starting the next.

**Data hygiene in tracked files.** No course-data references, hardcoded course
slugs, absolute local paths, machine hostnames, or LAN IPs in tracked code or
docs. Referencing a past run ID inside a regression-test docstring as provenance
is acceptable; a slug baked into non-test logic is not.

**Behavior flags.** Every flag defaults to a byte-identical legacy path unless
explicitly justified. A new flag lands with its row in the owning subsystem's
flag table (or `docs/operations/behavior-flags.md` for the root-owned
cross-cutting prefixes). Any flag that selects an LLM provider, model ID, or
synthesis backend **must** also land a row in `docs/LICENSING.md` — drift there
is a documentation bug.

**Stop-on-command.** Long-running stages poll a run-scoped stop sentinel at unit
boundaries so `ed4all stop` checkpoints instead of losing work. New long-running
call sites land with a resume sidecar, a stop check, and a test. Semantics:
`docs/operations/pipeline-invocation.md` § 7.

---

## 3. Running the pipeline

```bash
ed4all run <workflow> --corpus <PATH> --course-name <NAME> [--mode local|api]
```

`--mode local` (default) uses the current agent session as the LLM and
dispatches phase workers as subagents; `--mode api` uses the Anthropic SDK
directly and needs `ANTHROPIC_API_KEY`. Workflow list, stage subcommands,
per-stage flags, timeout knobs, and the resume/stop runbook are in `CLAUDE.md`
§ Quick Start and `docs/operations/pipeline-invocation.md`.

Note for non-Claude harnesses: phase workers under `--mode local` depend on MCP
subagent dispatch. A harness without that capability can run scripts and
validators directly, but cannot drive `--mode local` cross-agent dispatch.

---

## 4. Training-data synthesis — licensing posture

Read `docs/LICENSING.md` before starting any synthesis pass. It is the
authoritative source; the summary below exists to prevent a wrong first move.

The project separates two surfaces with different exposure:

- **Development tooling** (whatever agent harness you are) produces code, prose,
  and shell invocations. It does **not** produce training data, so its ToS has
  no bearing on the trained model's licensing.
- **Synthesis providers** produce the instruction/preference pairs that become
  training data. The trained model is a derivative work of *those* outputs, so
  the provider's ToS and the underlying model license decide shippability.

Provider posture for `Trainforge/synthesize_training.py` (`--provider`):

| Provider | Status for training-pair synthesis |
|----------|-----------------------------------|
| `local` | **Default recommendation.** OpenAI-compatible local server; license-clean with an Apache-2.0 model. |
| `together` | Clean hosted OSS fallback (paid, networked). |
| `mock` | Template factory — plumbing tests only. Produces template-recognizer adapters, never a shippable corpus. |
| `claude_session` | Still wired, but gated behind the explicit `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS` acknowledgment. Not recommended for training data. |
| `anthropic` | **Fails closed unconditionally.** Still listed in the CLI `choices` for a clear error message, but `run_synthesis` rejects it outright — training-pair synthesis is license-clean by construction. |

Post-synthesis validators live under `lib/validators/`: `synthesis_quota.py`,
`min_edge_count.py`, `synthesis_diversity.py`, `property_coverage.py`. Their
thresholds and phase wiring are in `docs/validation/gates.md` and
`config/workflows.yaml` — read them there rather than trusting a copy.

Synthesized pairs land under
`LibV2/courses/<course-slug>/training_specs/*.jsonl` and are consumed by the
post-import training stage (`ed4all run trainforge_train`).

**Pilot before a full rebuild.** `Trainforge/scripts/ops/pilot_synthesis.py` runs a
bounded pass (`--max-pairs`, default 50) so gate failures surface on a handful
of pairs instead of a full corpus. Read
the pilot report and the gate output before going wide. If a gate fails, the
escalation path is a larger/stronger license-clean model — never a fallback to a
ToS-restricted provider, and never a threshold change.

---

## 5. Escalate rather than guess

If operator intent is unclear, **ask**. This applies with particular force to:

- **Licensing judgment calls.** "Can this model's output train a derivative?" is
  an operator decision. Do not improvise.
- **Go/no-go on a corpus.** An agent summarizes gate output; the operator makes
  the call.
- **Training runs.** GPU-bound, need manual eval-matrix review and a
  promotion-ledger update. Operator-driven.
- **Schema changes** under `schemas/models/`. Plan territory, not ad-hoc edits.
- **Anything touching `main`.** Authorization required.

Stop at the first failed gate and surface the failure verbatim. Do not paper
over it.
