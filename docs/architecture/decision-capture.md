# Decision Capture — long-form notes

> Canonical decision-event shape: `schemas/events/decision_event.schema.json`. Helper: `lib/decision_capture.py::DecisionCapture`. Root `CLAUDE.md § Decision Capture` carries the rule sentences; this file carries call-site precedents and example output paths.

## Output Locations

```
training-captures/
├── dart/{COURSE_CODE}/                     # SemantiK conversion captures (path name preserved)
│   └── decisions_{PDF_NAME}_{TIMESTAMP}.jsonl
├── courseforge/{COURSE_CODE}/
│   ├── phase_input-research/
│   ├── phase_content-generator/
│   └── phase_brightspace-packager/
└── trainforge/{COURSE_CODE}/
    ├── phase_content-analysis/
    ├── phase_question-generation/
    └── phase_validation/
```

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

Every LLM call site MUST wire up a `DecisionCapture` instance and emit at least one decision per call (per-batch when batched). Static boilerplate rationales are forbidden — rationale must interpolate dynamic signals specific to the call (block IDs, image hashes, page numbers, model + max_tokens, confidence distributions, etc.) so captures are replayable post-hoc. A regression test MUST assert that the capture fires on the call path.

### Precedent call sites + regression tests

- Conversion structure / alt-text captures: `lib/decision_capture.py::DARTDecisionCapture` (`create_dart_capture`) → one `structure_detection` capture per structure decision and one `alt_text_generation` capture per figure. The `dart`-named capture tool + `data-dart-*` provenance vocabulary are the **preserved wire contract** — SemantiK is the engine that drives them now (the legacy DART converter is retired).
- SemantiK Stage-5d structure-reviewer bridge: `MCP/tools/pipeline_tools.py::_emit_structure_review_capture` (hooked in the `_run_semantik_v2_conversion` conversion seam) → one `structure_review` capture per converted doc, best-effort (a capture failure logs a warning, never breaks conversion). Regression coverage: `lib/semantik/tests/test_structure_review_bridge.py`.
- Trainforge synthesis provider: `Trainforge/generators/_anthropic_provider.py` → one `synthesis_provider_call` capture per call (see `Trainforge/tests/test_anthropic_synthesis_provider.py`).
- Trainforge curriculum-alignment provider: `Trainforge/generators/_curriculum_provider.py` (consumed by `Trainforge/align_chunks.py::classify_teaching_roles`) → one `curriculum_alignment_call` capture per teaching-role classification (see `Trainforge/tests/test_curriculum_alignment_provider.py`).
- Trainforge OpenAI-compatible HTTP client: `Trainforge/generators/_openai_compatible_client.py` → one `llm_chat_call` capture per call when wired with a capture; surface used by future task providers that compose the client directly (see `Trainforge/tests/test_openai_compatible_client.py`).
- Orchestrator generic OpenAI-compatible backend (Wave W-D12): `MCP/orchestrator/llm_backend.py::OpenAICompatibleBackend` → one `llm_chat_call` capture per call, with the dynamic `provider_name` constructor arg interpolated into the rationale (`provider=<name>`) and the decision string (`provider_name=<name>`, host-only `base_url`). The backend is a single class wrapping `Trainforge.generators._openai_compatible_client.OpenAICompatibleClient`; new providers (Together, Groq, Fireworks, DeepSeek, Mistral, hosted Gemini-via-OpenAI-shim, ...) plug in by appending a registry entry to `_OPENAI_COMPATIBLE_PROVIDERS` — adding a provider is a registry-entry change, NOT a new call site or subclass. Regression coverage: `lib/tests/test_llm_backend.py::TestDecisionCaptureProviderName::test_decision_capture_emits_provider_name` (interpolated provider name) and `lib/tests/test_llm_backend.py::TestProviderRegistry::test_provider_registry_extension_pattern` (load-bearing dynamic-extension contract).
- Orchestrator vision-mode OpenAI-compatible backend (Wave W-D13): `MCP/orchestrator/llm_backend.py::OpenAICompatibleBackend.complete_sync(images=...)` → same `llm_chat_call` capture as above, with the `images_count` extra threaded into the decision-string `extras` so the audit trail distinguishes vision calls from text-only calls without the rationale branching on the value. The vision-content-block translation lives ONE layer down at `Trainforge/generators/_openai_compatible_client.py::OpenAICompatibleClient._attach_vision_blocks`; the backend just forwards `images=` through after a vision-capability gate (`RuntimeError` when the provider entry's `vision_capable` flag is `False`). This generic vision backend remains available for any OpenAI-compatible vision consumer; the figure alt-text path for the conversion stage is now owned by SemantiK's local Stage-6b figure captioner (SmolVLM2) rather than the retired DART vision converter. Regression coverage: `MCP/tests/test_llm_backend_vision.py` (vision-capable forwarding + non-vision-capable RuntimeError), `Trainforge/tests/test_openai_compatible_client_vision.py` (content-block payload shape).
