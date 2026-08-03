# Decision capture

> Canonical event shape: `schemas/events/decision_event.schema.json`. Helper: `lib/decision_capture.py::DecisionCapture`. Root `CLAUDE.md § Decision Capture` carries the rule sentences; this file carries the contract detail, the storage layout, and the real call-site precedents with their regression tests.

## The contract

Every LLM decision in the pipeline is logged as one JSONL event. The schema declares six **required** fields:

| Field | Source |
|---|---|
| `run_id` | `DecisionCapture` instance state |
| `timestamp` | stamped at `log_decision` time |
| `operation` | passed in, or inferred from `decision_type` via `lib/constants.py` |
| `decision_type` | caller — validated against the schema enum (see below) |
| `decision` | caller — the choice actually made |
| `rationale` | caller — **minimum 20 characters** |

`_build_record` (`lib/decision_capture.py`) fills a wider envelope around those six: `event_id`, `seq`, `course_id`, `module_id`, `artifact_id`, `task_id`, `tool`, `phase`, `alternatives_considered`, `context`, `confidence`, `is_default`, `ml_features`, `inputs_ref`, `prompt_ref`, `outputs`, `outcome`, and a `metadata` block carrying `rationale_length`, `quality_level`, and the quality-gate verdict.

### Rationale minimum and quality tiers

The 20-character floor is enforced in `lib/validation.py::validate_decision` (`if len(rationale) < 20: issues.append(…)`). Separately, `lib/quality.py::assess_decision_quality` grades every record against `QUALITY_THRESHOLDS` in `lib/constants.py`:

| Tier | Rationale min length | Also requires |
|---|---:|---|
| `exemplary` | 100 | `inputs_ref` **and** `alternatives_considered` |
| `proficient` | 50 | `inputs_ref` **or** `alternatives_considered` |
| `developing` | 20 | — |
| `inadequate` | 0 | — |

`_build_record` then calls `check_quality_acceptable(quality_level, minimum_level="proficient")`. A record below `proficient` is still written, but is stamped `metadata.quality_gate_passed = false` with a reason and logged as a warning — the mechanism that keeps thin rationales out of the training corpus without losing the audit row.

### Where `decision_type` is validated

`decision_type` is **not** a free string. `lib/decision_capture.py` builds `ALLOWED_DECISION_TYPES` at module-import time by reading the `decision_type` enum out of `schemas/events/decision_event.schema.json` — **243 values** at the time of writing — with a small hardcoded tuple as fallback if the schema cannot be loaded. The schema, not a Python constant, is the single source of truth. **Adding a decision type is a schema edit plus its first production use site, in the same change.**

Two environment variables govern what happens to an invalid record, and they compose:

| `VALIDATE_DECISIONS` | `DECISION_VALIDATION_STRICT` | Behavior |
|---|---|---|
| unset / falsey | (any) | validation is a no-op |
| **default (`"true"`)** | unset | warn-only: issues attach to `metadata.validation_issues`, record IS written |
| truthy | `"true"` | fail-closed: `ValueError` raised, record NOT written |

Note the default: `VALIDATE_DECISIONS` defaults to `"true"` in `lib/constants.py`, so validation runs on every capture out of the box — it just does not block. Ship a new type in the schema *before* the code that emits it, or a strict-mode run will drop the event.

## Storage layout

A single `DecisionCapture` writes the same rows to up to three sinks:

1. **Canonical LibV2 catalog store** — `LibV2/catalog/<COURSE_CODE>/training/<tool>/phase_<phase>/decisions_<session_id>.jsonl`, resolved by `lib/libv2_storage.py::LibV2Storage.get_training_capture_path`. Honors `ED4ALL_LIBV2_ROOT`.
2. **Legacy mirror** — `runtime/training-captures/<tool>/<COURSE_CODE>/phase_<phase>/decisions_<session_id>.jsonl`. Root overridable via `ED4ALL_TRAINING_CAPTURES_DIR` (which does **not** follow `ED4ALL_LIBV2_ROOT` — the mirror lives at the project root, so it needs its own knob; the test-isolation autouse fixture points it at a tmp dir so a pytest run does not grow the real tree).
3. **Run-scoped stream** — `<run_context.decisions_path>/decisions_<tool>_<session_id>.jsonl`, written only when a hardening run context is active. `lib/replay_engine.py` reads this sink back from `runtime/state/runs/<RUN_ID>/decisions/`.

**Phase names are normalized:** `phase.replace("_", "-")`, and `phase=None` routes to `phase_unknown/` rather than crashing (tool-level captures such as the orchestrator's `phase_start` fire before a phase has been selected). So a capture constructed with `phase="semantik_conversion"` lands in a directory named `phase_semantik-conversion`.

Observed `<tool>` roots under `runtime/training-captures/`: `courseforge`, `libv2`, `orchestrator`, `pipeline`, `semantik`, `textbook-pipeline`, `trainforge`. The set is **not closed** — `tool` is a free constructor argument, not an enum. (A `runtime/training-captures/decisions/` directory also exists but is **not** a `<tool>` root: it holds loose JSONL files directly, not the `<tool>/<COURSE_CODE>/phase_<phase>/` shape this layout describes.)

### Write buffering

`ED4ALL_CAPTURE_BUFFER` (default off) coalesces the per-decision write+flush+fsync across the mirrors into one batched write every `ED4ALL_CAPTURE_BUFFER_ROWS` rows (default 50; garbage or ≤0 → 50). The buffer drains on flush, close, and `atexit`. The worst case is losing the last N buffered *telemetry* rows on a hard kill — captures are advisory records, not pipeline state.

## Example wiring

```python
from lib.decision_capture import DecisionCapture

capture = DecisionCapture(
    course_code="INT_101",
    phase="content-generator",
    tool="courseforge",
    streaming=True,
)

capture.log_decision(
    decision_type="content_structure",
    decision="Use 6-week modular structure",
    rationale="Aligns with competency-based approach and allows flexible pacing for diverse learners",
    alternatives_considered=[
        "8-week linear: Too rigid for self-paced learning",
        "4-week intensive: Insufficient depth for foundational content",
    ],
)
```

## LLM call-site instrumentation contract

Every LLM call site MUST wire a `DecisionCapture` and emit at least one decision per call (per-batch when batched). **Static boilerplate rationales are forbidden** — the rationale must interpolate dynamic signals specific to that call (block ids, page numbers, model + `max_tokens`, `finish_reason`, confidence distributions, tallies) so the capture is replayable post-hoc. `tests/decision_capture/test_boilerplate_rationale_detector.py` exists to police exactly this. **A regression test MUST assert the capture fires on the call path.**

The contract applies to *new model calls*. A module that only reads a report the harness already wrote correctly wires no capture — `lib/governance/procurement_evidence.py` and `lib/aggregators/build_cost.py` are the documented examples.

### Precedent call sites

Each row below was verified: the emitting module exists, emits the named `decision_type`, and the named test file exists.

| Call site | `decision_type` | Regression coverage |
|---|---|---|
| `lib/decision_capture.py::SemantiKDecisionCapture` (factory `create_semantik_capture`) | `structure_detection` (per structure decision), `alt_text_generation` (per figure) | — (base helper) |
| `SemantiK/semantik_structure/figure_captioner.py` — the omni cascade's Stage-6b SmolVLM2 captioner | `alt_text_generation` | `SemantiK/semantik_structure/tests/test_figure_captioner_capture.py` |
| `SemantiK/semantik_structure/glmocr/heading_judge.py` — the GLM-OCR lane's heading-level judge | `structure_review` with a `heading_level_judge=True` discriminator | `SemantiK/semantik_structure/tests/test_heading_judge.py::test_decision_capture_fires_with_dynamic_rationale`, `::test_capture_failure_never_breaks_judge` |
| `MCP/tools/pipeline_tools.py::_emit_structure_review_capture` (conversion seam, called at the `_run_semantik_v2_conversion` site) | `structure_review`, one per converted doc | `lib/semantik/tests/test_structure_review_bridge.py` |
| `MCP/tools/pipeline_tools.py::_emit_block_resegment_capture` (same seam) | `block_resegment`, one per converted doc when the re-partition pass fired | `lib/semantik/tests/test_structure_review_bridge.py` |
| `lib/retrieval/groundedness.py::score_groundedness` (`ED4ALL_GROUNDEDNESS_COMPUTATIONAL`) | `groundedness_computational_check` | `lib/tests/test_groundedness.py` |
| `Trainforge/generators/assessment_generator.py` (`TRAINFORGE_COGNITIVE_TASK_TYPE`) | `cognitive_task_type_detection` | `Trainforge/tests/test_assessment_generator_capture_wiring.py` |
| `LibV2/tools/libv2/evaluation/model_bridge.py::run_fresh_eval` | `fresh_eval_invocation` | `LibV2/tools/libv2/tests/test_model_eval_bridge.py` |
| `Trainforge/generators/_curriculum_provider.py` (consumed by `Trainforge/align_chunks.py::classify_teaching_roles`) | `curriculum_alignment_call` | `Trainforge/tests/test_curriculum_alignment_provider.py` |
| `Trainforge/generators/_local_provider.py`, `_together_provider.py`, `_claude_session_provider.py` (all over `_base_synthesis_provider.py`) | `synthesis_provider_call` | `Trainforge/tests/test_local_synthesis_provider.py`, `test_together_synthesis_provider.py`, `test_claude_session_provider.py`, `test_base_synthesis_provider.py`, `test_synthesis_provider.py` |
| `Trainforge/generators/_openai_compatible_client.py::OpenAICompatibleClient` | `llm_chat_call`, one per call when wired with a capture | `Trainforge/tests/test_openai_compatible_client.py` |
| `MCP/orchestrator/llm_backend.py::OpenAICompatibleBackend` | `llm_chat_call`, with the dynamic `provider_name` interpolated into rationale and decision string | `lib/tests/test_llm_backend.py` |
| `MCP/orchestrator/llm_backend.py::OpenAICompatibleBackend.complete_sync(images=…)` | `llm_chat_call` with an `images_count` extra distinguishing vision calls | `MCP/tests/test_llm_backend_vision.py`, `Trainforge/tests/test_openai_compatible_client_vision.py` |

### Notes on individual rows

**HTML wire contract.** The SemantiK path is the source of the capture factory named in the table (`SemantiKDecisionCapture` / `create_semantik_capture`). `lib/semantik/adapter.py` emits `data-semantik-*` attributes (`data-semantik-block-id`, `data-semantik-source`, `data-semantik-pages`, …), and the **CURIE** source-id form is minted as `semantik:{slug}#{block_id}`. `lib/validators/source_refs.py` additionally yields the legacy prefix for every id on the READ side, so freshly-emitted and unmigrated corpora both resolve against the staging manifest.

**Block-resegment capture.** It resolves audit rows off *both* cascade arms — the in-process result dict's top-level `block_resegment` key and the cross-venv bridge (`SemantiK/scripts/run_cascade_json.py`). It is a deterministic-pass capture (the resegment ops are deterministic-first), so it fires even with no LLM op-proposal layer; the rationale interpolates merge/split/regroup tallies, the fused-title-split count, the folded-region tally, the merged-unit semantic-class set, a bounded source-id sample, and `conservation_verified`.

**Groundedness capture threading.** The capture is threaded on **both** surfaces — the eval path (`lib/retrieval/grounded_eval.py`) and the production answer path (`lib/retrieval/grounded_answer.py`, which forwards `capture=capture` into `score_groundedness`). Best-effort: a capture failure never aborts scoring.

**Provider registry, not subclasses.** `OpenAICompatibleBackend` is a single class wrapping `Trainforge.generators._openai_compatible_client.OpenAICompatibleClient`. New providers plug in by appending a row to `config/endpoints.yaml`; `MCP/orchestrator/llm_backend.py::_OPENAI_COMPATIBLE_PROVIDERS` is a legacy-shaped projection of that YAML registry, resolved by name. Adding a provider is a registry-entry change, **not** a new call site or a subclass. `lib/tests/test_llm_backend.py::TestProviderRegistry::test_provider_registry_extension_pattern` pins that contract.

**Vision path ownership.** The vision content-block translation lives one layer below the backend, at `OpenAICompatibleClient._attach_vision_blocks`; the backend forwards `images=` after a `vision_capable` gate that raises `RuntimeError` for a non-vision provider entry. The generic vision backend remains available to any OpenAI-compatible vision consumer, but the conversion stage's figure alt-text is owned by SemantiK's own captioner paths, not by this backend.

**Anthropic synthesis is not a live precedent.** `Trainforge/generators/_anthropic_provider.py` exists and emits `synthesis_provider_call`, but `Trainforge/synthesize_training.py` carries `_REMOVED_SYNTHESIS_PROVIDERS = frozenset({"anthropic"})` and fails closed on it unconditionally for training-pair synthesis. Do not cite it as the pattern to copy; use the local / together / claude-session providers above.

## Known instrumentation gap

`SemantiK/semantik_structure/glmocr/alttext.py` — the GLM-OCR lane's Qwen3-VL alt-text seat — is an LLM call site that wires **no** `DecisionCapture`. `heading_judge.py` is the only module in `SemantiK/semantik_structure/glmocr/` that references `DecisionCapture`. The equivalent call site on the legacy omni cascade (`figure_captioner.py`) *is* instrumented and has a regression test, so the contract is satisfied on that path but not on the lane. This is a real gap against the instrumentation law, recorded here rather than papered over.
