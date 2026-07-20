# Decision Capture — long-form notes

> Canonical decision-event shape: `schemas/events/decision_event.schema.json`. Helper: `lib/decision_capture.py::DecisionCapture`. Root `CLAUDE.md § Decision Capture` carries the rule sentences; this file carries call-site precedents and example output paths.

## Output Locations

Captures land under `training-captures/<tool>/<COURSE_CODE>/phase_<phase>/decisions_*.jsonl`, where `<tool>`
and `<phase>` are the `DecisionCapture` constructor arguments. The root is overridable via
`ED4ALL_TRAINING_CAPTURES_DIR`.

```
training-captures/
├── semantik/{COURSE_CODE}/                 # SemantiK conversion captures
│   └── phase_semantik_conversion/
├── courseforge/{COURSE_CODE}/
│   ├── phase_input-research/
│   ├── phase_content-generator/
│   └── phase_brightspace-packager/
└── trainforge/{COURSE_CODE}/
    ├── phase_content-analysis/
    ├── phase_question-generation/
    └── phase_validation/
```

Other `<tool>` roots appear as their call sites fire (`libv2`, `orchestrator`, `pipeline`,
`textbook-pipeline`, and a legacy `dart` root left by captures written before the SemantiK rename). The set is
not closed — `<tool>` is a free constructor argument, not an enum.

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

## Where `decision_type` is validated

`decision_type` is **not** a free string. `lib/decision_capture.py` builds `ALLOWED_DECISION_TYPES` at
module-import time by reading the `decision_type` enum out of `schemas/events/decision_event.schema.json`
(243 values as of 2026-07-20), with a small hardcoded tuple as the fallback if the schema cannot be loaded.
The schema — not a Python constant — is the single source of truth, so **adding a decision type is a schema
edit plus its first production use site**, in the same change.

`DECISION_VALIDATION_STRICT` governs what happens on an unknown type: unset, the event is warned about;
truthy, the capture fails closed. Ship a new type in the schema before the code that emits it, or a strict-mode
run will drop the event.

## LLM call-site instrumentation contract

Every LLM call site MUST wire up a `DecisionCapture` instance and emit at least one decision per call (per-batch when batched). Static boilerplate rationales are forbidden — rationale must interpolate dynamic signals specific to the call (block IDs, image hashes, page numbers, model + max_tokens, confidence distributions, etc.) so captures are replayable post-hoc. A regression test MUST assert that the capture fires on the call path.

### Precedent call sites + regression tests

- Conversion structure / alt-text captures: `lib/decision_capture.py::SemantiKDecisionCapture` (factory `create_semantik_capture`) → one `structure_detection` capture per structure decision and one `alt_text_generation` capture per figure. It labels captures with tool `semantik` and phase `semantik_conversion`. `DARTDecisionCapture` / `create_dart_capture` survive as **deprecated module-level aliases** of the same class and factory (scheduled for removal); new call sites must use the SemantiK names. Note the asymmetry that trips readers: the *capture* naming was migrated, but the `data-dart-*` HTML provenance attributes and the `dart:{slug}#{block_id}` CURIE form remain the **preserved wire contract** — SemantiK is the engine that emits them now, and renaming them would break every downstream resolver.
- SemantiK Stage-5d structure-reviewer bridge: `MCP/tools/pipeline_tools.py::_emit_structure_review_capture` (hooked in the `_run_semantik_v2_conversion` conversion seam) → one `structure_review` capture per converted doc, best-effort (a capture failure logs a warning, never breaks conversion). Regression coverage: `lib/semantik/tests/test_structure_review_bridge.py`.
- SemantiK Stage-5e block-resegment bridge: `MCP/tools/pipeline_tools.py::_emit_block_resegment_capture` (same conversion seam, section 2c) → one `block_resegment` capture per converted doc when the Stage-5e re-partition pass fired, best-effort. It resolves the audit rows off BOTH cascade arms — the in-process result-dict top-level `block_resegment` key AND the cross-venv bridge (`SemantiK/scripts/run_cascade_json.py::_build_bridge_dict` now forwards it via `_resolve_block_resegment`, with the same-touch `second_pass_verify` arm forwarded alongside). Deterministic-pass capture (the resegment ops are deterministic-first, so it fires even with no LLM op-proposal layer); dynamic rationale interpolates the merge / split / regroup op tallies, the fused-title-split count, the folded-region tally, the merged-unit `semantic_class` set, a bounded `source_ids` sample, and `conservation_verified`. Regression coverage: `lib/semantik/tests/test_structure_review_bridge.py` § 3c (`test_block_resegment_capture_row_emitted_dynamic_rationale`; flag-off skip asserted by `test_block_resegment_capture_skips_cleanly_when_off`).
- Grounded-answer computational-groundedness check (`ED4ALL_GROUNDEDNESS_COMPUTATIONAL`): `lib/retrieval/groundedness.py::score_groundedness` → one `groundedness_computational_check` capture per answer with ≥1 computational claim WHEN a capture is threaded. The capture is now threaded on BOTH surfaces — the eval path (`lib/retrieval/grounded_eval.py`) AND the production answer path (`lib/retrieval/grounded_answer.py::_score_groundedness` forwards `capture=capture` into `score_groundedness(...)`). Best-effort: a capture failure never aborts scoring. Regression coverage: `lib/tests/test_groundedness.py::test_capture_fires_when_flag_on` (+ `test_computational_check_absent_when_flag_off` for the default-off byte-identical contract).
- Trainforge cognitive-task-type detection (`TRAINFORGE_COGNITIVE_TASK_TYPE`): `Trainforge/generators/assessment_generator.py` → one `cognitive_task_type_detection` capture per tagged question, emitted only when the flag is on and `lib/ontology/cognitive_task.py::detect_cognitive_task_type` matched a canonical task verb on the stem (no match → no field, no capture). Rationale interpolates the detected verb, the stem prefix, `question_id`, the objective, and `bloom_level`. Regression coverage: `Trainforge/tests/test_assessment_generator_capture_wiring.py::test_cognitive_task_type_flag_on_tags_and_captures` (+ `test_cognitive_task_type_flag_off_byte_identical` for the default-off no-capture / byte-identical `to_dict()` contract).
- LibV2 fresh-eval bridge: `LibV2/tools/libv2/model_eval_bridge.py::run_fresh_eval` → one `fresh_eval_invocation` capture per fresh adapter eval (`libv2 models eval <slug> <model_id> --fresh` / `libv2 eval run <slug> <model_id>`), best-effort (a capture failure logs a warning, never fails the eval). Rationale interpolates `model_id`, `course_slug`, the base repo, the eval profile, `smoke`, the gen knobs, `replace`, and the output report name. Regression coverage: `LibV2/tools/libv2/tests/test_model_eval_bridge.py::test_fresh_eval_capture_fires` (+ `test_fresh_eval_capture_failure_never_fails_eval`). The `fresh_eval_invocation` value is now present in `schemas/events/decision_event.schema.json`'s `decision_type` enum, so it survives `DECISION_VALIDATION_STRICT=true` (the earlier schema gap is closed).
- Trainforge synthesis provider: `Trainforge/generators/_anthropic_provider.py` → one `synthesis_provider_call` capture per call (see `Trainforge/tests/test_anthropic_synthesis_provider.py`).
- Trainforge curriculum-alignment provider: `Trainforge/generators/_curriculum_provider.py` (consumed by `Trainforge/align_chunks.py::classify_teaching_roles`) → one `curriculum_alignment_call` capture per teaching-role classification (see `Trainforge/tests/test_curriculum_alignment_provider.py`).
- Trainforge OpenAI-compatible HTTP client: `Trainforge/generators/_openai_compatible_client.py` → one `llm_chat_call` capture per call when wired with a capture; surface used by future task providers that compose the client directly (see `Trainforge/tests/test_openai_compatible_client.py`).
- Orchestrator generic OpenAI-compatible backend (Wave W-D12): `MCP/orchestrator/llm_backend.py::OpenAICompatibleBackend` → one `llm_chat_call` capture per call, with the dynamic `provider_name` constructor arg interpolated into the rationale (`provider=<name>`) and the decision string (`provider_name=<name>`, host-only `base_url`). The backend is a single class wrapping `Trainforge.generators._openai_compatible_client.OpenAICompatibleClient`; new providers (Together, Groq, Fireworks, DeepSeek, Mistral, hosted Gemini-via-OpenAI-shim, ...) plug in by appending a registry entry to `_OPENAI_COMPATIBLE_PROVIDERS` — adding a provider is a registry-entry change, NOT a new call site or subclass. Regression coverage: `lib/tests/test_llm_backend.py::TestDecisionCaptureProviderName::test_decision_capture_emits_provider_name` (interpolated provider name) and `lib/tests/test_llm_backend.py::TestProviderRegistry::test_provider_registry_extension_pattern` (load-bearing dynamic-extension contract).
- Orchestrator vision-mode OpenAI-compatible backend (Wave W-D13): `MCP/orchestrator/llm_backend.py::OpenAICompatibleBackend.complete_sync(images=...)` → same `llm_chat_call` capture as above, with the `images_count` extra threaded into the decision-string `extras` so the audit trail distinguishes vision calls from text-only calls without the rationale branching on the value. The vision-content-block translation lives ONE layer down at `Trainforge/generators/_openai_compatible_client.py::OpenAICompatibleClient._attach_vision_blocks`; the backend just forwards `images=` through after a vision-capability gate (`RuntimeError` when the provider entry's `vision_capable` flag is `False`). This generic vision backend remains available for any OpenAI-compatible vision consumer; the figure alt-text path for the conversion stage is now owned by SemantiK's local Stage-6b figure captioner (SmolVLM2) rather than the retired DART vision converter. Regression coverage: `MCP/tests/test_llm_backend_vision.py` (vision-capable forwarding + non-vision-capable RuntimeError), `Trainforge/tests/test_openai_compatible_client_vision.py` (content-block payload shape).
