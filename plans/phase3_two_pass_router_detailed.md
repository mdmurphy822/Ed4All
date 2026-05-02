# Phase 3 Detailed Execution Plan — Two-Pass Router + Outline/Rewrite Tiers

Refines `/home/user/Ed4All/plans/phase3_two_pass_router.md` (post-Round-7 amendment, commit `9b6a5e4`) into atomic, individually-verifiable subtasks. Each subtask has a unique deterministic verification command. The execution worker should NOT need to re-explore the codebase. Mirrors the granularity of `/home/user/Ed4All/plans/phase2_intermediate_format_detailed.md`.

---

## Investigation findings (locked)

- **Phase 1 + Phase 2 deliverables are landed in tree.** `Courseforge/generators/_provider.py::ContentGeneratorProvider.generate_page` (`:278-389`) returns a `Block` (not `str`); `Block.touched_by` already carries one `Touch(tier="outline", purpose="draft")` per provider call (`:380-388`). The provider routes via `COURSEFORGE_PROVIDER` (env constant at `:107`, `SUPPORTED_PROVIDERS = ("anthropic", "together", "local")` at `:109`).
- **`Block` dataclass** at `Courseforge/scripts/blocks.py:223-291` is frozen, carries `validation_attempts: int = 0` and `escalation_marker: Optional[str] = None` per Phase 2 Subtask 3 (post-feedback). `_ESCALATION_MARKERS` is a frozenset at `:105-111` containing `{"outline_budget_exhausted", "structural_unfixable", "validator_consensus_fail"}`. `Block.with_touch` at `:303-309` is the immutable-append helper Phase 3 must use to record tier provenance.
- **`OpenAICompatibleClient`** at `Trainforge/generators/_openai_compatible_client.py:74-621` is the shared HTTP client. The `extra_payload` arg on `chat_completion` (`:209-211`) flows through `_post_with_retry` to the wire body untouched (`:252-256`). Wave-113 `json_mode=True` (`:138-149`) already injects BOTH `format: "json"` and `response_format: {"type": "json_object"}` for grammar-aware backends — Phase 3 plumbs additional grammar-payload fields (`grammar`, `guided_json`, `guided_grammar`, `guided_regex`, `format` as a JSON-Schema dict for Ollama 0.5+) THROUGH `extra_payload` without modifying the client. `_extract_json_lenient` at `:425-492` is the canonical lenient parser for 7B-class drift.
- **Phase 1 wire-in surface** is `MCP/tools/_content_gen_helpers.py:1854-1907` inside `_build_content_modules_dynamic`. The provider call returns a Block; the consumer reads `block.content` + `block.key_terms` (`:1869-1889`). Phase 3 widens this to dispatch via `CourseforgeRouter.route_all(blocks)` when `COURSEFORGE_TWO_PASS=true`.
- **Pipeline entry point** is `MCP/tools/pipeline_tools.py::_generate_course_content` (`:2668-3019`). The provider is instantiated at `:2816-2836` gated on `_courseforge_provider_env`. The router instantiation slots in alongside (when `COURSEFORGE_TWO_PASS=true` is set).
- **Wave-74 short-circuit** at `MCP/core/executor.py:824-887` already routes `content-generator` through the in-process path when `COURSEFORGE_PROVIDER` is set; Phase 3's new env var `COURSEFORGE_TWO_PASS` joins the same boolean trigger (set EITHER var → in-process). No executor change needed because `COURSEFORGE_TWO_PASS=true` implies an in-process router; the operator's intent is to use the local pipeline.
- **`config/workflows.yaml::textbook_to_course::content_generation`** is at `:594-672` (under the `course_generation` workflow `:39-65` is the older slimmer form — Phase 3 splits BOTH workflows where present). Existing validation_gates: `content_structure` (warning), `source_refs` (critical), `content_grounding` (critical). The split adds `content_generation_outline` (with the four promoted gates) BEFORE the rewrite phase.
- **Decision-event schema enum** at `schemas/events/decision_event.schema.json:51-138` — `phase` enum is at `:53` (currently 24 values; alphabetised by Wave 22+); `decision_type` enum is at `:63-137` (alphabetised). Phase 3 adds 4 `decision_type` values + 2 `phase` values; insertion respects alphabetical order.
- **`GateResult` dataclass** at `MCP/hardening/validation_gates.py:48-78` carries `passed: bool`, `score: Optional[float]`, `issues: List[GateIssue]`, plus audit fields. **No `action` field today** — Phase 3 adds it as `Optional[str]` defaulting to `None`, and the router treats `None` as `"pass"` on success / `"block"` on failure for back-compat.
- **Validator entry surface.** `lib/validators/page_objectives.py:69::PageObjectivesValidator.validate(self, inputs: Dict[str, Any]) -> GateResult`. Same shape on `source_refs.py:76`, `curie_anchoring.py:154`, `content_type.py`. Phase 3's Block-input validators reuse the same `validate(inputs)` signature; the router passes `{"blocks": [...]}` in the inputs dict.
- **CourseforgeRouter is net-new** — no existing `Courseforge/router/` directory. Verified via `find Courseforge/ -name "router*" -type d`. The package is born here.
- **No existing `block_routing.yaml`** — verified via `find Ed4All -name "block_routing*"`. Schema + loader are net-new.
- **`Trainforge/generators/_local_provider.py:78` `LOCAL_SYNTHESIS_MODEL` default and `:127-150` `_LOCAL_INSTRUCTION_SYSTEM_PROMPT` + `DEFAULT_LOCAL_KIND_BOUNDS`** are the precedents Phase 3 mirrors for `OutlineProvider` (terse system prompts; per-block-type kind bounds).
- **CLAUDE.md flag table** is at `CLAUDE.md:728-731`. Existing rows: `COURSEFORGE_EMIT_BLOCKS` (`:729`), `COURSEFORGE_PROVIDER` (`:730`), `CURRICULUM_ALIGNMENT_PROVIDER` (`:731`). Phase 3 inserts new rows alphabetically between `COURSEFORGE_EMIT_BLOCKS` and `COURSEFORGE_PROVIDER` (`COURSEFORGE_BLOCK_ROUTING_PATH`, `COURSEFORGE_OUTLINE_*`, `COURSEFORGE_REWRITE_*`, `COURSEFORGE_TWO_PASS`).

---

## Pre-resolved decisions

1. **`_BaseLLMProvider` extraction.** Extract the shared HTTP / dispatch / decision-capture skeleton from `Courseforge/generators/_provider.py` into a new `Courseforge/generators/_base.py::_BaseLLMProvider` class. Phase 1's `ContentGeneratorProvider` becomes a thin subclass that overrides only `_render_user_prompt` (page-authoring) and the `_SYSTEM_PROMPT` constant. `OutlineProvider` and `RewriteProvider` are sibling subclasses sharing the same HTTP plumbing. Rationale: keeps Phase 1's `COURSEFORGE_PROVIDER` env var byte-stable (constructor signature unchanged); Phase 3's new env vars (`COURSEFORGE_OUTLINE_*` / `COURSEFORGE_REWRITE_*`) are read by the new subclasses' `__init__` overrides, NOT by `_BaseLLMProvider`.
2. **Two-pass workflow integration.** Split `content_generation` in `config/workflows.yaml::textbook_to_course` (`:594-672`) AND `course_generation` (`:39-65`) into `content_generation_outline` + `inter_tier_validation` + `content_generation_rewrite`. Gate the split behind `COURSEFORGE_TWO_PASS=true` via a new YAML attribute `enabled_when_env: COURSEFORGE_TWO_PASS=true` on the new phases; the legacy `content_generation` phase gains `enabled_when_env: COURSEFORGE_TWO_PASS!=true`. `MCP/core/workflow_runner.py::_should_skip_phase` (`:1150-1169`) gains a single 5-line addition that honours `enabled_when_env`. Default unset → legacy path runs unchanged.
3. **`block_routing.yaml` location + schema.** Default `Courseforge/config/block_routing.yaml`. Override path via `COURSEFORGE_BLOCK_ROUTING_PATH`. Optional file — when missing, env vars + tier defaults govern. JSON Schema at `schemas/courseforge/block_routing.schema.json` (Draft 2020-12, `additionalProperties: false`).
4. **Per-block-type defaults.** Outline tier: `qwen2.5:7b-instruct-q4_K_M` for ALL block types (matches the local-provider default at `Trainforge/generators/_local_provider.py:78`). Rewrite tier: `claude-sonnet-4-6` (Anthropic) for `prereq_set` / `assessment_item` / `misconception` (multi-step reasoning); `qwen2.5:14b-instruct-q4_K_M` (local) for everything else. Document as starting points subject to Phase 4 calibration. Encoded in `Courseforge/router/policy.py::DEFAULT_BLOCK_ROUTING` as a frozen Python dict (loaded as a fallback if YAML is absent).
5. **`GateResult.action` field.** Extend `MCP/hardening/validation_gates.py::GateResult` (`:48-78`) with `action: Optional[str] = None`. New Phase-3/4 validators emit `action="regenerate" | "escalate" | "block"` per Phase 4 §1 mapping. Existing validators that don't set `action` are treated by the router as `"pass"` on success and `"block"` on failure — back-compat preserved.
6. **Self-consistency dispatch parallelism.** Sequential N-candidate generation (not parallel asyncio) for the first cut. Rationale: simpler to test, easier to capture decisions, latency penalty minor at N=3 for outline-tier 7B (~3-5s/candidate). Parallelism is a later optimisation; the router method signature accepts `parallel: bool = False` so opt-in is one kwarg flip.
7. **Decision-capture event types.** Phase 3 adds 4 `decision_type` enum values (alphabetically sorted): `block_escalation`, `block_outline_call`, `block_rewrite_call`, `block_validation_action`. Plus 2 `phase` enum values: `courseforge-content-generator-outline`, `courseforge-content-generator-rewrite`.
8. **Phase 1 ContentGeneratorProvider deprecation.** None. The class continues to exist and behave identically when `COURSEFORGE_TWO_PASS` is unset. Phase 3 introduces `OutlineProvider` + `RewriteProvider` as siblings; existing tests stay green.
9. **Environment-variable inventory** (8 new vars + the existing `COURSEFORGE_PROVIDER`):
   - `COURSEFORGE_TWO_PASS` (master gate; default `false`)
   - `COURSEFORGE_OUTLINE_PROVIDER` (default `local`)
   - `COURSEFORGE_REWRITE_PROVIDER` (default `anthropic`)
   - `COURSEFORGE_OUTLINE_MODEL` (default `qwen2.5:7b-instruct-q4_K_M`)
   - `COURSEFORGE_REWRITE_MODEL` (default `claude-sonnet-4-6`)
   - `COURSEFORGE_OUTLINE_N_CANDIDATES` (default `3`)
   - `COURSEFORGE_OUTLINE_REGEN_BUDGET` (default `3`)
   - `COURSEFORGE_OUTLINE_GRAMMAR_MODE` (`gbnf|json_schema|json_object|none`; autodetect when unset)
   - `COURSEFORGE_BLOCK_ROUTING_PATH` (override default `Courseforge/config/block_routing.yaml`)
10. **Constrained-decoding payload field per provider** (selected in `OutlineProvider._build_grammar_payload`):
    - llama.cpp: `grammar: <gbnf-string>`
    - Ollama 0.5+: `format: <json-schema-dict>` (full schema, NOT just the legacy `"json"` token)
    - Ollama legacy: `format: "json"` (Wave-113 default)
    - vLLM: `extra_body: {guided_grammar: <gbnf>, guided_json: <schema>, guided_regex: <pattern>}`
    - Together: `response_format: {type: "json_schema", json_schema: {...}}`
    - Anthropic: no sample-time grammar; falls back to JSON-mode-only + lenient parse + remediation retry
11. **Inter-tier gate phase: separate workflow phase, not in-process inside outline phase.** Reason: per-phase decision capture (already established convention); cleaner failure isolation; Phase 5's `--escalated-only` CLI flag becomes a one-line workflow-runner check on `phase_outputs[inter_tier_validation]`.

---

## Atomic subtasks

Estimated total LOC: ~3,800 across all subtasks (200 base extraction + 350 OutlineProvider + 300 RewriteProvider + 350 Router + 200 policy loader + 250 self-consistency + 200 regen budget/escalation + 250 inter-tier gates + 200 validator-action + 300 workflow integration + 250 grammar plumbing + 700 tests + 250 docs).

### A. Workflow integration (5 subtasks)

#### Subtask 1: Add `COURSEFORGE_TWO_PASS` recognition in `MCP/core/workflow_runner.py`
- **Files:** `/home/user/Ed4All/MCP/core/workflow_runner.py:1150-1169` (extend `_should_skip_phase`)
- **Depends on:** none
- **Estimated LOC:** ~25
- **Change:** Extend `_should_skip_phase` to honour an `enabled_when_env: "<VAR>=<VALUE>" | "<VAR>!=<VALUE>"` attribute on `WorkflowPhase`. Read the env var (using `os.environ.get`); compare against the predicate; return `True` when the predicate is unsatisfied. Falls through to the existing optional-phase logic when `enabled_when_env` is absent. Add a sibling helper `_eval_enabled_when_env(predicate: str) -> bool` returning the boolean. Predicate grammar: `"<NAME>=<truthy_value>"` (true when env equals value), or `"<NAME>!=<value>"` (true when env does not equal). Truthy value match is case-insensitive equality; the literal `true` matches `1`/`true`/`yes`/`on` (mirrors `Courseforge/scripts/blocks.py::_EMIT_BLOCKS_TRUTHY` at `:40`).
- **Verification:** `python -c "from MCP.core.workflow_runner import WorkflowRunner; import os; os.environ['COURSEFORGE_TWO_PASS']='true'; r=WorkflowRunner.__new__(WorkflowRunner); from types import SimpleNamespace; p=SimpleNamespace(name='content_generation',optional=False); p.enabled_when_env='COURSEFORGE_TWO_PASS!=true'; assert r._should_skip_phase(p, {}) == True"` exits 0.

#### Subtask 2: Extend `WorkflowPhase` dataclass schema to accept `enabled_when_env`
- **Files:** `/home/user/Ed4All/MCP/core/workflow_runner.py` (the `WorkflowPhase` dataclass — search `class WorkflowPhase` for line)
- **Depends on:** Subtask 1
- **Estimated LOC:** ~10
- **Change:** Add `enabled_when_env: Optional[str] = None` field. Update the YAML loader (typically in `MCP/core/workflow_loader.py` or wherever `WorkflowPhase(**phase_dict)` is constructed) to pass through the new key when present. Schema-validate via the existing JSON Schema for `config/workflows.yaml` (e.g. `schemas/config/workflows.schema.json`) by adding `enabled_when_env: {type: "string"}` to the phase property set.
- **Verification:** `python -c "from MCP.core.workflow_runner import WorkflowPhase; p=WorkflowPhase(name='x',agents=[],parallel=False,max_concurrent=1,depends_on=[],timeout_minutes=1,description='t', enabled_when_env='COURSEFORGE_TWO_PASS=true'); assert p.enabled_when_env=='COURSEFORGE_TWO_PASS=true'"` exits 0.

#### Subtask 3: Split `content_generation` in `config/workflows.yaml::textbook_to_course`
- **Files:** `/home/user/Ed4All/config/workflows.yaml:594-672`
- **Depends on:** Subtasks 1, 2
- **Estimated LOC:** ~120
- **Change:** Add `enabled_when_env: "COURSEFORGE_TWO_PASS!=true"` to the existing `content_generation` phase (line `:594`) so it skips when the new flag is on. Insert THREE new sibling phases AFTER it, each gated `enabled_when_env: "COURSEFORGE_TWO_PASS=true"`:
  - `content_generation_outline` — `agents: [content-generator]`, `parallel: true`, `max_concurrent: 10`, `depends_on: [course_planning, source_mapping, staging]`, `timeout_minutes: 30`, outputs `[blocks_outline_path, project_id, weeks_prepared]`. NO validation_gates (those run in the separate inter-tier phase).
  - `inter_tier_validation` — `agents: []` (no agent — runs Python validators only), `parallel: false`, `depends_on: [content_generation_outline]`, `timeout_minutes: 5`, outputs `[blocks_validated_path, blocks_failed_path]`. Validation_gates: `outline_curie_anchoring`, `outline_content_type`, `outline_page_objectives`, `outline_source_refs` (all severity critical with `behavior: {on_fail: warn, on_error: warn}` initially — promoted to block in a follow-up wave).
  - `content_generation_rewrite` — `agents: [content-generator]`, `parallel: true`, `max_concurrent: 10`, `depends_on: [inter_tier_validation]`, `timeout_minutes: 90`, outputs `[content_paths, page_paths, content_dir, blocks_final_path]`. Validation_gates: `content_grounding` (critical, on_fail: block) — moved from the legacy phase.
- **Verification:** `python -c "import yaml; d=yaml.safe_load(open('config/workflows.yaml')); wf=next(w for w in d['workflows'] if w['name']=='textbook_to_course'); names=[p['name'] for p in wf['phases']]; assert 'content_generation_outline' in names and 'inter_tier_validation' in names and 'content_generation_rewrite' in names"` exits 0.

#### Subtask 4: Mirror the split into `config/workflows.yaml::course_generation`
- **Files:** `/home/user/Ed4All/config/workflows.yaml:39-65`
- **Depends on:** Subtask 3
- **Estimated LOC:** ~60
- **Change:** Same structural change as Subtask 3, applied to the slimmer `course_generation` workflow. Use the same three new phase names + same `enabled_when_env` predicates. The `course_generation` workflow's `packaging` phase (`depends_on: [content_generation]` at `:64`) gains an alternate predicate via a new YAML key `depends_on_one_of: [[content_generation], [content_generation_rewrite]]` consumed by `_dependencies_met` — OR, simpler, leave `depends_on: [content_generation]` unchanged and add a sibling `depends_on_when_env: COURSEFORGE_TWO_PASS=true: [content_generation_rewrite]`. Choose the simpler form (a single key); update `_dependencies_met` in workflow_runner to honour it (~10 LOC).
- **Verification:** `python -c "import yaml; d=yaml.safe_load(open('config/workflows.yaml')); wf=next(w for w in d['workflows'] if w['name']=='course_generation'); names=[p['name'] for p in wf['phases']]; assert 'content_generation_outline' in names and 'inter_tier_validation' in names"` exits 0.

#### Subtask 5: Add `_LEGACY_PHASE_OUTPUT_KEYS` entries for the three new phases
- **Files:** `/home/user/Ed4All/MCP/core/workflow_runner.py:60-130` (the `_LEGACY_PHASE_OUTPUT_KEYS` mapping)
- **Depends on:** Subtask 3
- **Estimated LOC:** ~15
- **Change:** Add three entries: `"content_generation_outline": ["blocks_outline_path", "project_id", "weeks_prepared"]`, `"inter_tier_validation": ["blocks_validated_path", "blocks_failed_path"]`, `"content_generation_rewrite": ["content_paths", "page_paths", "content_dir", "blocks_final_path"]`. Plus add per-phase param-routing entries (the `_LEGACY_PHASE_INPUT_ROUTES` analogue if present in the file): `content_generation_rewrite::blocks_validated_path` ← `phase_outputs.inter_tier_validation.blocks_validated_path`. Keeps the workflow-runner cross-reference integrity check (`_validate_inputs_from`) green when the new phases run.
- **Verification:** `python -c "from MCP.core.workflow_runner import _LEGACY_PHASE_OUTPUT_KEYS; assert 'content_generation_outline' in _LEGACY_PHASE_OUTPUT_KEYS and 'blocks_outline_path' in _LEGACY_PHASE_OUTPUT_KEYS['content_generation_outline']"` exits 0.

#### Subtask 6: Add `_courseforge_two_pass_enabled()` helper to `MCP/tools/pipeline_tools.py`
- **Files:** `/home/user/Ed4All/MCP/tools/pipeline_tools.py:2700-2800` (near `_generate_course_content`)
- **Depends on:** none
- **Estimated LOC:** ~10
- **Change:** Module-level helper `def _courseforge_two_pass_enabled() -> bool` reading `os.environ.get("COURSEFORGE_TWO_PASS","").strip().lower() in {"1","true","yes","on"}`. Mirror Phase 2's `_courseforge_emit_blocks_enabled` style. Used by Subtasks 39-41 (Phase 1 wire-in update).
- **Verification:** `python -c "import os; os.environ['COURSEFORGE_TWO_PASS']='true'; from MCP.tools.pipeline_tools import _courseforge_two_pass_enabled; assert _courseforge_two_pass_enabled()==True"` exits 0.

### B. Decision-event enum + schema (2 subtasks)

#### Subtask 7: Add 4 `decision_type` + 2 `phase` enum values to `decision_event.schema.json`
- **Files:** `/home/user/Ed4All/schemas/events/decision_event.schema.json:53` (phase enum) and `:63-137` (decision_type enum)
- **Depends on:** none
- **Estimated LOC:** ~6
- **Change:** Insert into `phase` enum (alphabetically): `"courseforge-content-generator-outline"`, `"courseforge-content-generator-rewrite"`. Insert into `decision_type` enum (alphabetically): `"block_escalation"`, `"block_outline_call"`, `"block_rewrite_call"`, `"block_validation_action"`. Update the `description` string at `:138` to reference the four new values. Maintain alphabetical ordering invariant.
- **Verification:** `python -c "import json; d=json.load(open('schemas/events/decision_event.schema.json')); e=d['properties']['decision_type']['enum']; assert all(v in e for v in ['block_escalation','block_outline_call','block_rewrite_call','block_validation_action']); ph=d['properties']['phase']['enum']; assert 'courseforge-content-generator-outline' in ph and 'courseforge-content-generator-rewrite' in ph"` exits 0.

#### Subtask 8: Add regression test for the strict-mode validator's frozen enum cache
- **Files:** create `/home/user/Ed4All/lib/tests/test_phase3_decision_event_enums.py`
- **Depends on:** Subtask 7
- **Estimated LOC:** ~40
- **Change:** Test that `lib.decision_capture.DecisionCapture.log_decision(decision_type="block_outline_call", ...)` does NOT raise under `DECISION_VALIDATION_STRICT=true`. Three cases per new value: dict-shaped decision, missing-rationale fail-loud, well-formed pass. Reuse the fixture pattern from existing decision-capture tests (`grep -rln "DECISION_VALIDATION_STRICT" lib/tests` for precedent).
- **Verification:** `pytest lib/tests/test_phase3_decision_event_enums.py -v` reports ≥6 PASSED.

### C. `_BaseLLMProvider` extraction (4 subtasks)

#### Subtask 9: Create `Courseforge/generators/_base.py::_BaseLLMProvider` skeleton
- **Files:** create `/home/user/Ed4All/Courseforge/generators/_base.py`
- **Depends on:** none
- **Estimated LOC:** ~250
- **Change:** Extract everything from `Courseforge/generators/_provider.py` that's NOT page-authoring-specific into a new abstract base class:
  - `__init__(*, provider, model, api_key, base_url, capture, max_tokens, temperature, client, anthropic_client, env_provider_var, default_provider, default_model_anthropic, default_model_together, default_model_local, default_base_url_local, supported_providers, system_prompt)` — all the per-tier knobs become constructor args; `OutlineProvider` and `RewriteProvider` pass tier-specific defaults.
  - `_dispatch_call(user_prompt: str) -> Tuple[str, int]` — moved from `:429-450`.
  - `_call_anthropic(user_prompt: str) -> Tuple[str, int]` — moved from `:452-500`.
  - `_last_capture_id() -> str` — moved from `:506-548`.
  - `_emit_decision(*, decision_type: str, decision: str, rationale: str)` — generic version of `:550-592`; subclasses build the rationale string.
  - Abstract method `_render_user_prompt(...) -> str` (subclass-specific).
  - Abstract method `_emit_per_call_decision(*, ..., raw_text, retry_count) -> None` (subclass picks the decision_type).
- **Verification:** `python -c "from Courseforge.generators._base import _BaseLLMProvider; assert hasattr(_BaseLLMProvider, '_dispatch_call') and hasattr(_BaseLLMProvider, '_call_anthropic')"` exits 0.

#### Subtask 10: Refactor `ContentGeneratorProvider` to extend `_BaseLLMProvider`
- **Files:** `/home/user/Ed4All/Courseforge/generators/_provider.py:140-601`
- **Depends on:** Subtask 9
- **Estimated LOC:** ~80 (deletion + thin override)
- **Change:** `ContentGeneratorProvider(BaseLLMProvider)`. Override only: `_render_user_prompt` (the page-context prompt at `:395-427`) and `generate_page` (the public entry that constructs the Block at `:278-389`). Delete `_dispatch_call` / `_call_anthropic` / `_last_capture_id` / `_emit_decision` from this file (now inherited). The `__init__` becomes a 1-line `super().__init__(env_provider_var=ENV_PROVIDER, default_provider=DEFAULT_PROVIDER, system_prompt=_SYSTEM_PROMPT, ..., **kwargs)`. Re-export `ENV_PROVIDER` / `DEFAULT_PROVIDER` / `SUPPORTED_PROVIDERS` constants from this module (back-compat for callers grepping for them).
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py -v` reports all PASS (regression — Phase 1 + 2 tests must stay green byte-for-byte).

#### Subtask 11: Add `Courseforge/generators/tests/test_base_llm_provider.py`
- **Files:** create `/home/user/Ed4All/Courseforge/generators/tests/test_base_llm_provider.py`
- **Depends on:** Subtask 10
- **Estimated LOC:** ~80
- **Change:** Tests for the extracted base: `test_dispatch_call_routes_to_anthropic_for_anthropic_provider`, `test_dispatch_call_routes_to_oa_client_for_local_provider`, `test_emit_decision_includes_required_fields`, `test_last_capture_id_format_when_capture_present`, `test_last_capture_id_falls_back_to_in_memory_when_capture_none`. Reuse the `httpx.MockTransport` fixture pattern from `Courseforge/tests/test_content_generator_provider.py`.
- **Verification:** `pytest Courseforge/generators/tests/test_base_llm_provider.py -v` reports ≥5 PASSED.

#### Subtask 12: Wire `Courseforge/generators/__init__.py` re-exports
- **Files:** `/home/user/Ed4All/Courseforge/generators/__init__.py`
- **Depends on:** Subtasks 9, 10
- **Estimated LOC:** ~10
- **Change:** Add `from Courseforge.generators._base import _BaseLLMProvider`. Re-export at top-level so callers can `from Courseforge.generators import _BaseLLMProvider, ContentGeneratorProvider`. Adds OutlineProvider + RewriteProvider re-exports too (forward-declared for Subtasks 13-22).
- **Verification:** `python -c "from Courseforge.generators import ContentGeneratorProvider, _BaseLLMProvider; assert _BaseLLMProvider is not None"` exits 0.

### D. OutlineProvider class (8 subtasks)

#### Subtask 13: Create `Courseforge/generators/_outline_provider.py::OutlineProvider` skeleton
- **Files:** create `/home/user/Ed4All/Courseforge/generators/_outline_provider.py`
- **Depends on:** Subtask 9
- **Estimated LOC:** ~120
- **Change:** Module docstring describing the outline tier per Phase 3 §2.1. Constants: `ENV_PROVIDER = "COURSEFORGE_OUTLINE_PROVIDER"`, `ENV_MODEL = "COURSEFORGE_OUTLINE_MODEL"`, `ENV_GRAMMAR_MODE = "COURSEFORGE_OUTLINE_GRAMMAR_MODE"`, `DEFAULT_PROVIDER = "local"`, `DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"`, `_DEFAULT_MAX_TOKENS = 1200`, `_DEFAULT_TEMPERATURE = 0.0`, `SUPPORTED_PROVIDERS = ("anthropic", "together", "local", "openai_compatible")`. Class `OutlineProvider(_BaseLLMProvider)` extending the base. Method stubs: `generate_outline(self, block: Block, *, source_chunks: List[Dict[str, Any]], objectives: List[Dict[str, Any]]) -> Block`, `_render_user_prompt`, `_build_grammar_payload(block_type: str) -> Dict[str, Any]`, `_outline_kind_bounds() -> Dict[str, Tuple[int,int]]`. Methods raise `NotImplementedError` for now.
- **Verification:** `python -c "from Courseforge.generators._outline_provider import OutlineProvider, ENV_PROVIDER; assert ENV_PROVIDER=='COURSEFORGE_OUTLINE_PROVIDER'"` exits 0.

#### Subtask 14: Implement `_OUTLINE_KIND_BOUNDS` per-block-type bounds table
- **Files:** `/home/user/Ed4All/Courseforge/generators/_outline_provider.py`
- **Depends on:** Subtask 13
- **Estimated LOC:** ~40
- **Change:** Module-level constant `_OUTLINE_KIND_BOUNDS: Dict[str, Dict[str, Tuple[int,int]]]` keyed on `block_type` (every value in `BLOCK_TYPES`), each value is `{"key_claims": (min,max), "section_skeleton": (min,max), "summary_chars": (min,max)}`. Mirrors `Trainforge/generators/_local_provider.py:145-150::DEFAULT_LOCAL_KIND_BOUNDS`. Defaults per type: `objective` → `{key_claims:(1,3), section_skeleton:(0,0), summary_chars:(40,200)}`; `concept` → `{key_claims:(1,5), section_skeleton:(1,3), summary_chars:(80,400)}`; `assessment_item` → `{key_claims:(1,2), section_skeleton:(1,2), summary_chars:(60,300)}`; etc. Document the table values are starting points subject to Phase 4 calibration.
- **Verification:** `python -c "from Courseforge.generators._outline_provider import _OUTLINE_KIND_BOUNDS; assert 'objective' in _OUTLINE_KIND_BOUNDS and _OUTLINE_KIND_BOUNDS['objective']['key_claims']==(1,3)"` exits 0.

#### Subtask 15: Implement `OutlineProvider.__init__` with per-tier env vars
- **Files:** `/home/user/Ed4All/Courseforge/generators/_outline_provider.py`
- **Depends on:** Subtask 14
- **Estimated LOC:** ~50
- **Change:** Subclass `__init__` calls `super().__init__(env_provider_var="COURSEFORGE_OUTLINE_PROVIDER", default_provider="local", system_prompt=_OUTLINE_SYSTEM_PROMPT, ...)`. Resolves model from `kwargs.get("model") or os.environ.get("COURSEFORGE_OUTLINE_MODEL") or DEFAULT_MODEL`. Resolves `n_candidates: int = 3` from `os.environ.get("COURSEFORGE_OUTLINE_N_CANDIDATES")` or kwarg. Resolves `regen_budget: int = 3` from `os.environ.get("COURSEFORGE_OUTLINE_REGEN_BUDGET")`. Resolves `grammar_mode: str` from `COURSEFORGE_OUTLINE_GRAMMAR_MODE` (default `None` → autodetect by provider+base_url). Stores all as instance attrs.
- **Verification:** `python -c "import os; os.environ['COURSEFORGE_OUTLINE_PROVIDER']='local'; os.environ['COURSEFORGE_OUTLINE_N_CANDIDATES']='5'; from Courseforge.generators._outline_provider import OutlineProvider; p=OutlineProvider(); assert p._n_candidates==5 and p._provider=='local'"` exits 0.

#### Subtask 16: Author `_OUTLINE_SYSTEM_PROMPT` (terse — ≤80 words per Phase 3 §2.1)
- **Files:** `/home/user/Ed4All/Courseforge/generators/_outline_provider.py`
- **Depends on:** Subtask 13
- **Estimated LOC:** ~25
- **Change:** Module-level `_OUTLINE_SYSTEM_PROMPT` literal: "You are an outline-tier draft generator for Courseforge blocks. Emit a structurally-correct JSON outline carrying: block_id, block_type, content_type, bloom_level, objective_refs[], curies[], key_claims[], section_skeleton[], source_refs[], structural_warnings[]. PRESERVE every CURIE and source_id verbatim from the input. Do NOT add facts not in the supplied source_chunks. Do NOT generate prose — generate the structural skeleton only. Output ONLY the JSON object — no preamble, no markdown, no commentary." Mirrors `Trainforge/generators/_local_provider.py:127-142` terseness.
- **Verification:** `python -c "from Courseforge.generators._outline_provider import _OUTLINE_SYSTEM_PROMPT; assert len(_OUTLINE_SYSTEM_PROMPT.split()) <= 80 and 'JSON' in _OUTLINE_SYSTEM_PROMPT"` exits 0.

#### Subtask 17: Implement per-block-type user prompt construction in `_render_user_prompt`
- **Files:** `/home/user/Ed4All/Courseforge/generators/_outline_provider.py`
- **Depends on:** Subtask 16
- **Estimated LOC:** ~70
- **Change:** `_render_user_prompt(*, block: Block, source_chunks: List[Dict], objectives: List[Dict]) -> str` returns a JSON-shaped user prompt: a header line (`Block ID: <id>; Type: <type>`), the source_chunks block (each chunk's id + body, truncated at 1200 chars), the objectives block (id + statement), the target schema (per-block-type — built from `_OUTLINE_KIND_BOUNDS[block_type]`), and an explicit "RESPOND ONLY WITH A JSON OBJECT containing: block_id, block_type, content_type, bloom_level, objective_refs, curies, key_claims, section_skeleton, source_refs, structural_warnings". Per-block-type variations: `assessment_item` adds "stem and answer must reference the listed objective_refs"; `prereq_set` adds "list prerequisitePages explicitly".
- **Verification:** `python -c "from Courseforge.generators._outline_provider import OutlineProvider; from Courseforge.scripts.blocks import Block; b=Block(block_id='p#objective_x_0',block_type='objective',page_id='p',sequence=0,content='Define X'); p=OutlineProvider.__new__(OutlineProvider); s=p._render_user_prompt(block=b, source_chunks=[{'id':'c1','body':'X is a thing'}], objectives=[{'id':'TO-01','statement':'Define X'}]); assert 'Block ID' in s and 'TO-01' in s and 'JSON OBJECT' in s.upper()"` exits 0.

#### Subtask 18: Implement `_build_grammar_payload` per-provider-mechanism dispatch
- **Files:** `/home/user/Ed4All/Courseforge/generators/_outline_provider.py`
- **Depends on:** Subtask 15
- **Estimated LOC:** ~80
- **Change:** Method `_build_grammar_payload(self, block_type: str) -> Dict[str, Any]` returns the per-call `extra_payload` dict the OpenAICompatibleClient (Subtask 21) will merge into the request body. Dispatch on `(self._provider, self._base_url, self._grammar_mode)`:
  - `mode=="gbnf" or (provider in {"local","openai_compatible"} and base_url contains "llama" or "lmstudio")` → return `{"grammar": <gbnf-string>}` from `_BLOCK_TYPE_GBNF[block_type]` (a module-level dict; each value is a GBNF grammar string per Phase 3 §2.1.1).
  - `mode=="json_schema" or provider=="ollama-0.5"` → return `{"format": <schema_dict>}` (full Ollama 0.5 JSON Schema).
  - `provider=="together"` → return `{"response_format": {"type": "json_schema", "json_schema": {"name": f"OutlineBlock_{block_type}", "schema": <schema_dict>, "strict": True}}}`.
  - `provider=="vllm"` (detected by base_url) → return `{"extra_body": {"guided_json": <schema_dict>}}`.
  - Anthropic / unrecognised → return `{}` and rely on Wave-113 `json_mode=True` (set on the OpenAICompatibleClient).
- **Verification:** `python -c "import os; os.environ['COURSEFORGE_OUTLINE_PROVIDER']='local'; os.environ['COURSEFORGE_OUTLINE_GRAMMAR_MODE']='gbnf'; from Courseforge.generators._outline_provider import OutlineProvider; p=OutlineProvider(); g=p._build_grammar_payload('objective'); assert 'grammar' in g"` exits 0.

#### Subtask 19: Implement minimal per-block-type JSON Schema dicts (`_BLOCK_TYPE_JSON_SCHEMAS`)
- **Files:** `/home/user/Ed4All/Courseforge/generators/_outline_provider.py`
- **Depends on:** Subtask 18
- **Estimated LOC:** ~80
- **Change:** Module-level `_BLOCK_TYPE_JSON_SCHEMAS: Dict[str, Dict[str, Any]]` keyed on every `block_type` in `BLOCK_TYPES`. Each value is a Draft 2020-12 schema requiring: `block_id`, `block_type` (const matching the key), `content_type`, `bloom_level` (enum), `objective_refs` (array of strings), `curies` (array of strings, each matching `^[a-z][a-z0-9]*:[A-Za-z0-9_-]+$`), `key_claims` (array of strings, length per `_OUTLINE_KIND_BOUNDS`), `section_skeleton` (array of objects), `source_refs` (array of `{sourceId,role}`), `structural_warnings` (array of strings; default `[]`). `additionalProperties: false`. Per-block-type variations: `assessment_item` requires `stem` + `answer_key`; `prereq_set` requires `prerequisitePages: array`.
- **Verification:** `python -c "import jsonschema; from Courseforge.generators._outline_provider import _BLOCK_TYPE_JSON_SCHEMAS; assert all(b in _BLOCK_TYPE_JSON_SCHEMAS for b in ['objective','concept','assessment_item','prereq_set']); jsonschema.Draft202012Validator.check_schema(_BLOCK_TYPE_JSON_SCHEMAS['objective'])"` exits 0.

#### Subtask 20: Implement `OutlineProvider.generate_outline` (single-candidate path)
- **Files:** `/home/user/Ed4All/Courseforge/generators/_outline_provider.py`
- **Depends on:** Subtasks 17, 18, 19
- **Estimated LOC:** ~100
- **Change:** `generate_outline(self, block: Block, *, source_chunks, objectives) -> Block` (single candidate; self-consistency loop is Subtask 30). Steps: build user_prompt via `_render_user_prompt`; build `extra_payload` via `_build_grammar_payload(block.block_type)`; call `self._dispatch_call(user_prompt, extra_payload=extra_payload)` (extends `_BaseLLMProvider._dispatch_call` to accept `extra_payload` kwarg — see Subtask 21); apply `OpenAICompatibleClient._extract_json_lenient` to the response text; validate against `_BLOCK_TYPE_JSON_SCHEMAS[block.block_type]`; on parse-fail, retry up to `MAX_PARSE_RETRIES=3` with a remediation hint appended ("Your previous output failed JSON Schema validation — return ONLY a JSON object matching {schema}"); on exhaustion raise `OutlineProviderError(code="outline_exhausted")`. Return a new Block via `dataclasses.replace(block, content=parsed_outline_dict)` plus a Touch entry `Touch(tier="outline", purpose="draft", model=self._model, provider=self._provider, ...)` appended via `block.with_touch(touch)`.
- **Verification:** Stub-level unit test (full integration test in Subtask 53). `python -c "from Courseforge.generators._outline_provider import OutlineProvider, OutlineProviderError; assert hasattr(OutlineProvider, 'generate_outline') and OutlineProviderError.__name__=='OutlineProviderError'"` exits 0.

### E. RewriteProvider class (7 subtasks)

#### Subtask 21: Extend `_BaseLLMProvider._dispatch_call` to accept `extra_payload` kwarg
- **Files:** `/home/user/Ed4All/Courseforge/generators/_base.py`
- **Depends on:** Subtask 9
- **Estimated LOC:** ~15
- **Change:** Update `_dispatch_call(self, user_prompt: str, *, extra_payload: Optional[Dict[str, Any]] = None) -> Tuple[str, int]`. When `provider != "anthropic"`, pass `extra_payload=extra_payload` to `self._oa_client.chat_completion(...)` — but since the Phase-1 path uses `_post_with_retry` directly (`:448` of `_provider.py`), update that direct call site too: merge `extra_payload` into the `payload` dict before the POST. When `provider=="anthropic"`, `extra_payload` is ignored (Anthropic SDK doesn't accept arbitrary OpenAI fields).
- **Verification:** `python -c "from Courseforge.generators._base import _BaseLLMProvider; import inspect; sig=inspect.signature(_BaseLLMProvider._dispatch_call); assert 'extra_payload' in sig.parameters"` exits 0.

#### Subtask 22: Create `Courseforge/generators/_rewrite_provider.py::RewriteProvider` skeleton
- **Files:** create `/home/user/Ed4All/Courseforge/generators/_rewrite_provider.py`
- **Depends on:** Subtask 9
- **Estimated LOC:** ~80
- **Change:** Constants `ENV_PROVIDER = "COURSEFORGE_REWRITE_PROVIDER"`, `ENV_MODEL = "COURSEFORGE_REWRITE_MODEL"`, `DEFAULT_PROVIDER = "anthropic"`, `DEFAULT_MODEL = "claude-sonnet-4-6"`, `_DEFAULT_MAX_TOKENS = 2400`, `_DEFAULT_TEMPERATURE = 0.4`, `SUPPORTED_PROVIDERS = ("anthropic", "together", "local", "openai_compatible")`. Class `RewriteProvider(_BaseLLMProvider)`. Method stubs `generate_rewrite(self, block: Block, *, source_chunks, objectives) -> Block`, `_render_user_prompt`, `_render_escalated_user_prompt`. Module-level `_REWRITE_SYSTEM_PROMPT` (full Pattern-22 prevention contract from `Courseforge/agents/content-generator.md` — port the existing system prompt verbatim from `Courseforge/generators/_provider.py:119-132` plus an additional "Outline is structurally correct but generated by a smaller model. PRESERVE: factual claims (verbatim), CURIEs (verbatim), objective refs, source refs. REWRITE: for pedagogical depth, scaffolding, examples, voice. DO NOT add facts not in the outline's key_claims or in the source chunks." paragraph).
- **Verification:** `python -c "from Courseforge.generators._rewrite_provider import RewriteProvider, _REWRITE_SYSTEM_PROMPT; assert 'PRESERVE' in _REWRITE_SYSTEM_PROMPT and 'pedagogical depth' in _REWRITE_SYSTEM_PROMPT"` exits 0.

#### Subtask 23: Implement `RewriteProvider.__init__`
- **Files:** `/home/user/Ed4All/Courseforge/generators/_rewrite_provider.py`
- **Depends on:** Subtask 22
- **Estimated LOC:** ~30
- **Change:** Subclass `__init__` calls `super().__init__(env_provider_var="COURSEFORGE_REWRITE_PROVIDER", default_provider="anthropic", default_model_anthropic="claude-sonnet-4-6", default_model_together="meta-llama/Llama-3.3-70B-Instruct-Turbo", default_model_local="qwen2.5:14b-instruct-q4_K_M", system_prompt=_REWRITE_SYSTEM_PROMPT, max_tokens=2400, temperature=0.4, ...)`. No grammar plumbing for the rewrite tier — output is HTML, not JSON.
- **Verification:** `python -c "import os; os.environ['COURSEFORGE_REWRITE_PROVIDER']='together'; from Courseforge.generators._rewrite_provider import RewriteProvider; p=RewriteProvider(api_key='x'); assert p._provider=='together'"` exits 0.

#### Subtask 24: Implement `_render_user_prompt` for the rewrite tier (consumes Block.content as outline)
- **Files:** `/home/user/Ed4All/Courseforge/generators/_rewrite_provider.py`
- **Depends on:** Subtask 23
- **Estimated LOC:** ~60
- **Change:** `_render_user_prompt(*, block: Block, source_chunks, objectives) -> str`. The block's `content` field at this point is the outline dict produced by OutlineProvider. Sections: `Outline (structurally correct, pedagogical-depth missing): <json.dumps(block.content)>`; `Source chunks (cite via source_refs): <chunks>`; `Objectives: <ids+statements>`; `Block-type-specific output contract`: per Block.block_type, append the HTML attribute contract (e.g. `objective` → "emit `<li>` carrying `data-cf-objective-id`, `data-cf-bloom-level`, `data-cf-bloom-verb`"; `flip_card_grid` → "emit `<div class=\"flip-card-grid\">` with per-card `<div class=\"flip-card\">`"). The block-type→HTML-shape map mirrors `Courseforge/scripts/blocks.py::Block.to_html_attrs`. Final instruction: "Author the rendered HTML body for this block now. Emit ONLY the HTML — no preamble, no markdown, no commentary."
- **Verification:** `python -c "from Courseforge.generators._rewrite_provider import RewriteProvider; from Courseforge.scripts.blocks import Block; b=Block(block_id='x',block_type='objective',page_id='p',sequence=0,content={'key_claims':['c1'],'curies':['sh:NodeShape']}); p=RewriteProvider.__new__(RewriteProvider); s=p._render_user_prompt(block=b,source_chunks=[],objectives=[]); assert 'sh:NodeShape' in s and 'data-cf-objective-id' in s"` exits 0.

#### Subtask 25: Implement `_render_escalated_user_prompt` for budget-exhausted blocks
- **Files:** `/home/user/Ed4All/Courseforge/generators/_rewrite_provider.py`
- **Depends on:** Subtask 24
- **Estimated LOC:** ~40
- **Change:** When `block.escalation_marker is not None`, the rewrite tier switches to a richer prompt template per Phase 3 §3.7: "The outline tier could not produce a valid {block_type} after {n} attempts (marker={escalation_marker}). Synthesize from scratch using {source_chunks} and {objective_refs}, preserving CURIEs verbatim: {curies_list}. Do not introduce facts outside the supplied source chunks." Returns the same HTML-output contract appended. Branches on the marker: `outline_budget_exhausted` → "the outline contains a partial draft you can reference"; `outline_skipped_by_policy` → "no outline was generated; create from scratch"; `validator_consensus_fail` → "the outline contained semantic violations the validators flagged".
- **Verification:** `python -c "from Courseforge.generators._rewrite_provider import RewriteProvider; from Courseforge.scripts.blocks import Block; b=Block(block_id='x',block_type='concept',page_id='p',sequence=0,content={'curies':['rdf:type']},escalation_marker='outline_budget_exhausted'); p=RewriteProvider.__new__(RewriteProvider); s=p._render_escalated_user_prompt(block=b,source_chunks=[],objectives=[]); assert 'outline_budget_exhausted' in s and 'rdf:type' in s"` exits 0.

#### Subtask 26: Implement `RewriteProvider.generate_rewrite` with CURIE-preservation gate
- **Files:** `/home/user/Ed4All/Courseforge/generators/_rewrite_provider.py`
- **Depends on:** Subtasks 24, 25
- **Estimated LOC:** ~100
- **Change:** `generate_rewrite(self, block: Block, *, source_chunks, objectives) -> Block`. Branches on `block.escalation_marker`: non-None → calls `_render_escalated_user_prompt`; None → `_render_user_prompt`. Dispatch via `self._dispatch_call(user_prompt)`; capture HTML response. Apply CURIE-preservation gate: `outline_curies = block.content.get("curies", [])` (when content is an outline dict); for each curie in outline_curies, assert `curie in html_response` (substring match); on miss, build a remediation prompt ("Your previous output dropped CURIE {curie}. PRESERVE every CURIE verbatim. Re-emit the HTML.") and retry up to `MAX_PARSE_RETRIES=2`. Direct port of `Trainforge/generators/_local_provider.py:548-583::_missing_preserve_tokens` + `_append_preserve_remediation` patterns. On exhaustion raise `RewriteProviderError(code="rewrite_curie_drop", missing_curies=[...])`. On success, return a new Block via `dataclasses.replace(block, content=html_response)` plus a Touch entry `Touch(tier="rewrite", purpose="pedagogical_depth", ...)`.
- **Verification:** `python -c "from Courseforge.generators._rewrite_provider import RewriteProvider, RewriteProviderError; assert hasattr(RewriteProvider, 'generate_rewrite') and RewriteProviderError.__name__=='RewriteProviderError'"` exits 0.

#### Subtask 27: Add `Courseforge/generators/tests/test_rewrite_provider.py` baseline tests
- **Files:** create `/home/user/Ed4All/Courseforge/generators/tests/test_rewrite_provider.py`
- **Depends on:** Subtask 26
- **Estimated LOC:** ~140
- **Change:** Tests via `httpx.MockTransport`: `test_default_rewrite_provider_is_anthropic_when_env_unset`, `test_env_var_selects_provider`, `test_unknown_provider_raises_value_error`, `test_generate_rewrite_calls_anthropic_path_for_anthropic_provider`, `test_curie_preservation_gate_fires_remediation_on_drop`, `test_curie_preservation_exhaustion_raises_rewrite_curie_drop`, `test_escalated_block_uses_richer_prompt`, `test_rewrite_appends_touch_with_tier_rewrite`. Reuse `Trainforge/tests/test_curriculum_alignment_provider.py` fixture pattern.
- **Verification:** `pytest Courseforge/generators/tests/test_rewrite_provider.py -v` reports ≥8 PASSED.

### F. CourseforgeRouter class (5 subtasks)

#### Subtask 28: Create `Courseforge/router/` package + `BlockProviderSpec` dataclass
- **Files:** create `/home/user/Ed4All/Courseforge/router/__init__.py` (empty), `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtasks 13, 22
- **Estimated LOC:** ~80
- **Change:** Module docstring describing the router. Import `OutlineProvider`, `RewriteProvider`, `Block`, `Touch`. Define `@dataclass(frozen=True) class BlockProviderSpec` with fields `block_type: str`, `tier: Literal["outline","rewrite"]`, `provider: Literal["anthropic","together","local","openai_compatible"]`, `model: str`, `base_url: Optional[str] = None`, `api_key_env: Optional[str] = None`, `temperature: float = 0.0`, `max_tokens: int = 2400`, `extra_payload: Dict[str, Any] = field(default_factory=dict)`, `escalate_immediately: bool = False`. `__post_init__` validates `tier` and `provider` against allowed sets.
- **Verification:** `python -c "from Courseforge.router.router import BlockProviderSpec; s=BlockProviderSpec(block_type='objective',tier='outline',provider='local',model='qwen2.5:7b'); assert s.tier=='outline'"` exits 0.

#### Subtask 29: Implement `CourseforgeRouter.__init__` + provider-resolution helpers
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtask 28
- **Estimated LOC:** ~100
- **Change:** `class CourseforgeRouter` with `__init__(*, policy=None, outline_provider=None, rewrite_provider=None, capture=None, deterministic_gates=None, statistical_filter=None, n_candidates=None, regen_budget=None) -> None`. Store as instance attrs; lazy-instantiate `outline_provider` / `rewrite_provider` on first use (so a router constructed without these still works for `route_all` calls when the YAML policy provides per-block-type dispatch). Implement `_resolve_spec(self, block: Block, tier: str, **overrides) -> BlockProviderSpec` per Phase 3 §3.3: per-call kwargs win → block_routing.yaml entry for `(block_type, tier)` → tier-default env vars → `COURSEFORGE_PROVIDER` final fallback → hardcoded default. The hardcoded defaults table is a module-level `_HARDCODED_DEFAULTS: Dict[Tuple[str, str], BlockProviderSpec]` (entry per `(block_type, tier)`).
- **Verification:** `python -c "from Courseforge.router.router import CourseforgeRouter; r=CourseforgeRouter(); from Courseforge.scripts.blocks import Block; b=Block(block_id='x',block_type='objective',page_id='p',sequence=0,content='c'); s=r._resolve_spec(b,'outline'); assert s.tier=='outline' and s.block_type=='objective'"` exits 0.

#### Subtask 30: Implement `CourseforgeRouter.route` (per-block, per-tier)
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtask 29
- **Estimated LOC:** ~80
- **Change:** `route(self, block: Block, *, tier: Literal["outline","rewrite"], source_chunks=None, objectives=None, **overrides) -> Block`. Resolves spec via `_resolve_spec`; instantiates the provider on demand (cache by spec key); dispatches to `outline_provider.generate_outline(block, ...)` or `rewrite_provider.generate_rewrite(block, ...)`; emits a `block_outline_call` or `block_rewrite_call` decision-capture event with the resolved `(provider, model, base_url, policy_source, prompt_hash_12, token_usage)` per Phase 3 §9.2. Pre-fires the outline tier `escalate_immediately` short-circuit: if `tier=="outline" and spec.escalate_immediately`, skip the outline call entirely, set `block.escalation_marker="outline_skipped_by_policy"`, return without LLM dispatch.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_route_dispatches_to_outline_provider_for_outline_tier -v` PASSES (test added in Subtask 32).

#### Subtask 31: Implement `CourseforgeRouter.route_all` (full two-pass over a Block list)
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtask 30
- **Estimated LOC:** ~70
- **Change:** `route_all(self, blocks: List[Block], *, source_chunks_by_block_id=None, objectives=None) -> List[Block]`. Two-pass: (a) outline tier per block (with self-consistency in Subtask 33); (b) inter-tier validation chain (Subtask 36); (c) rewrite tier per block that passed validation. Failed blocks are NOT dispatched to rewrite — they're persisted with `status="failed"` for re-execution. Returns the full list including failed blocks (ordering preserved).
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_route_all_runs_two_pass -v` PASSES.

#### Subtask 32: Add `Courseforge/router/tests/test_router.py` baseline tests
- **Files:** create `/home/user/Ed4All/Courseforge/router/tests/test_router.py` and `__init__.py`
- **Depends on:** Subtasks 30, 31
- **Estimated LOC:** ~180
- **Change:** Tests: `test_resolve_spec_per_call_kwargs_win_over_yaml_and_env`, `test_resolve_spec_yaml_overrides_env_var`, `test_resolve_spec_env_var_overrides_default`, `test_unknown_provider_raises_value_error`, `test_route_dispatches_to_outline_provider_for_outline_tier`, `test_route_dispatches_to_rewrite_provider_for_rewrite_tier`, `test_route_all_runs_two_pass`, `test_route_all_excludes_failed_outline_from_rewrite`, `test_per_block_type_dispatch_uses_yaml_override_when_present`, `test_route_emits_block_outline_call_decision_event` (mock capture), `test_escalate_immediately_short_circuits_outline_tier`. Reuse `httpx.MockTransport` fixture pattern.
- **Verification:** `pytest Courseforge/router/tests/test_router.py -v` reports ≥11 PASSED.

### G. block_routing.yaml schema + loader (4 subtasks)

#### Subtask 33: Author `schemas/courseforge/block_routing.schema.json`
- **Files:** create `/home/user/Ed4All/schemas/courseforge/block_routing.schema.json`
- **Depends on:** none
- **Estimated LOC:** ~120
- **Change:** Draft 2020-12 schema. `$id: "https://ed4all.dev/ns/courseforge/v1/BlockRouting.schema.json"`. Top-level required: `["version"]`. `additionalProperties: false`. Properties: `version: {const: 1}`, `defaults: {type:"object", properties: {outline: <SpecRef>, rewrite: <SpecRef>}}`, `blocks: {type:"object", patternProperties: {"^(objective|concept|example|...|recap)$": {type:"object", properties: {outline: <SpecRef>, rewrite: <SpecRef>, n_candidates: {type:"integer",minimum:1}, escalate_immediately: {type:"boolean"}}}}}`, `overrides: {type:"array", items: {type:"object", required:["block_id"], properties:{block_id:{type:"string"}, outline: <SpecRef>, rewrite: <SpecRef>}}}`. `$defs/Spec` is `{type:"object", properties: {provider:{enum:["anthropic","together","local","openai_compatible"]}, model:{type:"string"}, base_url:{type:"string"}, api_key_env:{type:"string"}, temperature:{type:"number"}, max_tokens:{type:"integer"}}}`.
- **Verification:** `python -c "import json,jsonschema; s=json.load(open('schemas/courseforge/block_routing.schema.json')); jsonschema.Draft202012Validator.check_schema(s)"` exits 0.

#### Subtask 34: Author `Courseforge/router/policy.py::load_block_routing_policy`
- **Files:** create `/home/user/Ed4All/Courseforge/router/policy.py`
- **Depends on:** Subtask 33
- **Estimated LOC:** ~120
- **Change:** Module-level `_DEFAULT_POLICY_PATH = Path("Courseforge/config/block_routing.yaml")`. `@dataclass(frozen=True) class BlockRoutingPolicy` with `defaults: Dict[str, BlockProviderSpec]` (keyed `outline`/`rewrite`), `blocks: Dict[str, Dict[str, BlockProviderSpec]]` (keyed by block_type then tier), `overrides: List[Dict]`. `load_block_routing_policy(path: Optional[Path] = None) -> BlockRoutingPolicy`: resolves path from arg or `COURSEFORGE_BLOCK_ROUTING_PATH` env or `_DEFAULT_POLICY_PATH`; absent file → returns empty policy with INFO log; loads YAML; validates against `block_routing.schema.json`; constructs frozen policy. `match_block_id_glob(block_id: str, pattern: str) -> bool`: Python `fnmatch` glob match. Method on `BlockRoutingPolicy::resolve(block_id, block_type, tier) -> Optional[BlockProviderSpec]` walking overrides → blocks[block_type][tier] → defaults[tier] → None.
- **Verification:** `python -c "from Courseforge.router.policy import load_block_routing_policy; p=load_block_routing_policy(); assert p is not None"` exits 0.

#### Subtask 35: Author skeleton `Courseforge/config/block_routing.yaml`
- **Files:** create `/home/user/Ed4All/Courseforge/config/block_routing.yaml`
- **Depends on:** Subtask 34
- **Estimated LOC:** ~50
- **Change:** Minimal example YAML matching the schema: `version: 1`, `defaults` block with outline (provider=local, model=qwen2.5:7b-instruct-q4_K_M, base_url=http://localhost:11434/v1, temperature=0.0, max_tokens=1200) and rewrite (provider=anthropic, model=claude-sonnet-4-6, temperature=0.4, max_tokens=2400). `blocks` map with three entries demonstrating the per-block-type override surface: `assessment_item.rewrite` flips to anthropic + claude-sonnet-4-6 (with operator-facing rationale comment), `flip_card.outline` keeps local 7B (with rationale), `prereq_set.outline.escalate_immediately: true` (with rationale).
- **Verification:** `python -c "import yaml,jsonschema,json; y=yaml.safe_load(open('Courseforge/config/block_routing.yaml')); s=json.load(open('schemas/courseforge/block_routing.schema.json')); jsonschema.validate(y,s)"` exits 0.

#### Subtask 36: Add `Courseforge/router/tests/test_policy.py`
- **Files:** create `/home/user/Ed4All/Courseforge/router/tests/test_policy.py`
- **Depends on:** Subtasks 34, 35
- **Estimated LOC:** ~100
- **Change:** Tests: `test_load_returns_empty_policy_when_file_absent`, `test_load_validates_against_schema`, `test_resolve_walks_overrides_first`, `test_resolve_falls_through_to_defaults`, `test_resolve_returns_none_when_no_match`, `test_block_id_glob_match_supports_star`, `test_invalid_yaml_raises`, `test_env_var_overrides_default_path`. Use `tmp_path` fixture for file-system test setup.
- **Verification:** `pytest Courseforge/router/tests/test_policy.py -v` reports ≥8 PASSED.

### H. Self-consistency dispatch (4 subtasks)

#### Subtask 37: Implement `CourseforgeRouter.route_with_self_consistency`
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtask 30
- **Estimated LOC:** ~100
- **Change:** Method `route_with_self_consistency(self, block: Block, *, n_candidates: Optional[int]=None, validators: Optional[List]=None, source_chunks=None, objectives=None) -> Block`. Resolves `n_candidates` from arg → policy.blocks[type].n_candidates → env var → 3. Sequential loop `for i in range(n)`: dispatch one outline candidate via `route(block, tier="outline")`; run the validator chain (Subtask 38) on the candidate; if all pass, return the block (with `Touch.purpose="self_consistency_winner"` and the `winning_candidate_index=i` audit field on the touch). If all candidates fail every validator, return the LAST candidate with `validation_attempts=n` and don't set `escalation_marker` (Subtask 41 handles that). Records per-candidate failure distribution into a local dict for the audit event.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_route_with_self_consistency_returns_first_passer -v` PASSES.

#### Subtask 38: Implement validator chain ordering helper
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtask 37
- **Estimated LOC:** ~50
- **Change:** Method `_run_validator_chain(self, block: Block, validators: List) -> Tuple[bool, List[GateResult]]`. Cheapest-first ordering per Phase 3 §3.6: grammar/JSON Schema (always passes — already enforced sample-time, listed for shape), then SHACL (Phase 4 seam), then CURIE resolution, then embedding similarity (Phase 4), then round-trip check (Phase 4). For Phase 3 only the first three are wired; later validators are no-op shims for forward-compat. Returns `(all_passed, [GateResult per validator])`. Stops at first fail when `fast_fail=True` (default); collects all when `fast_fail=False`.
- **Verification:** Stub-level: `python -c "from Courseforge.router.router import CourseforgeRouter; r=CourseforgeRouter(); assert hasattr(r, '_run_validator_chain')"` exits 0.

#### Subtask 39: Add decision-capture metadata for self-consistency to `block_outline_call` event
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py` (extend `_emit_block_outline_call` or equivalent)
- **Depends on:** Subtasks 37, 38
- **Estimated LOC:** ~25
- **Change:** Extend the `ml_features` payload of every `block_outline_call` event with `n_candidates_requested: int`, `winning_candidate_index: Optional[int]` (None when all failed), `failed_candidate_count: int`, `validator_failure_distribution: Dict[str, int]`. Per Phase 3 §3.6.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_self_consistency_emits_decision_metadata -v` PASSES.

#### Subtask 40: Add `Courseforge/router/tests/test_self_consistency.py`
- **Files:** create `/home/user/Ed4All/Courseforge/router/tests/test_self_consistency.py`
- **Depends on:** Subtask 39
- **Estimated LOC:** ~120
- **Change:** Tests: `test_first_candidate_passes_returns_immediately`, `test_third_candidate_passes_after_two_fail`, `test_all_candidates_fail_returns_last_with_validation_attempts_n`, `test_n_candidates_resolves_from_env_var`, `test_n_candidates_resolves_from_policy_block_override`, `test_decision_event_includes_winning_candidate_index`, `test_validator_failure_distribution_is_aggregated`. Use a stub `OutlineProvider` that returns canned outputs in sequence.
- **Verification:** `pytest Courseforge/router/tests/test_self_consistency.py -v` reports ≥7 PASSED.

### I. Regeneration budget + escalation (5 subtasks)

#### Subtask 41: Implement regen-budget tracking in `route_with_self_consistency`
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtask 37
- **Estimated LOC:** ~40
- **Change:** Inside the self-consistency loop, increment `block.validation_attempts` on every failed-validator pass via `dataclasses.replace(block, validation_attempts=block.validation_attempts+1)`. Resolve `regen_budget` from arg → policy.blocks[type].regen_budget → env var `COURSEFORGE_OUTLINE_REGEN_BUDGET` → instance attr → 3. When `block.validation_attempts >= regen_budget`, exit the loop early: set `block = dataclasses.replace(block, escalation_marker="outline_budget_exhausted")` and break. Return the escalated block.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_regen_budget_exhausted_sets_escalation_marker -v` PASSES.

#### Subtask 42: Implement `escalate_immediately` short-circuit in `route`
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtask 30
- **Estimated LOC:** ~25
- **Change:** Already covered partially by Subtask 30; finalise here. When `tier=="outline"` AND the resolved spec carries `escalate_immediately=True`, skip ALL outline dispatch (no LLM call), set `block.escalation_marker="outline_skipped_by_policy"`, append a Touch entry with `tier="outline"` and `purpose="skipped_by_policy"`, emit a `block_escalation` decision-capture event, return immediately.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_escalate_immediately_skips_outline_dispatch_entirely -v` PASSES.

#### Subtask 43: Wire `block_escalation` decision-capture event emission
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtasks 41, 42
- **Estimated LOC:** ~30
- **Change:** Helper `_emit_block_escalation(self, block: Block, *, marker: str, attempts: int, n_candidates: int) -> None`. Emits one `decision_type="block_escalation"` event with rationale ≥20 chars: `f"Block {block.block_id} (block_type={block.block_type}) escalated to rewrite tier with marker={marker} after {attempts} validation attempts across {n_candidates} candidates. Outline tier exhausted regen budget; rewrite tier will receive an enriched prompt with full source chunks + objective refs to author from scratch."` `ml_features` includes `block_id`, `block_type`, `marker`, `attempts`, `n_candidates`.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_block_escalation_event_emitted_on_budget_exhaustion -v` PASSES.

#### Subtask 44: Add `Courseforge/router/tests/test_regen_budget.py`
- **Files:** create `/home/user/Ed4All/Courseforge/router/tests/test_regen_budget.py`
- **Depends on:** Subtask 43
- **Estimated LOC:** ~150
- **Change:** Tests: `test_validation_attempts_increments_on_failure`, `test_budget_exhaustion_sets_outline_budget_exhausted_marker`, `test_budget_exhaustion_emits_block_escalation_event`, `test_per_block_type_regen_budget_overrides_env`, `test_escalate_immediately_skips_outline_entirely`, `test_escalate_immediately_sets_outline_skipped_by_policy`, `test_escalated_block_routes_to_rewrite_with_richer_prompt`, `test_validation_attempts_persists_through_block_replace`. Use stub `OutlineProvider` that always returns failing outputs.
- **Verification:** `pytest Courseforge/router/tests/test_regen_budget.py -v` reports ≥8 PASSED.

#### Subtask 45: Document the regen-budget + escalation contract in code comments
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py` (extend module docstring)
- **Depends on:** Subtask 44
- **Estimated LOC:** ~20
- **Change:** Module docstring section "## Regeneration budget + escalation" describing: validation_attempts increment per fail; budget resolution order; escalation_marker semantics for `outline_budget_exhausted` vs `outline_skipped_by_policy`; rewrite-tier prompt branching; cross-link to Phase 3 §3.7 and Phase 5 `--escalated-only`.
- **Verification:** `grep -c "regeneration budget\|escalation_marker" Courseforge/router/router.py` returns ≥3.

### J. Validator action signal (4 subtasks)

#### Subtask 46: Extend `MCP/hardening/validation_gates.py::GateResult` with `action` field
- **Files:** `/home/user/Ed4All/MCP/hardening/validation_gates.py:48-78`
- **Depends on:** none
- **Estimated LOC:** ~10
- **Change:** Add field `action: Optional[Literal["pass","regenerate","escalate","block"]] = None`. Update the `to_dict()` method to include `action` (default `None` stays `None` in the dict for back-compat). Add a class-method helper `derive_default_action(passed: bool, action: Optional[str]) -> str`: returns `action` when set; else returns `"pass"` if `passed` else `"block"`. The router calls this helper to interpret legacy validators.
- **Verification:** `python -c "from MCP.hardening.validation_gates import GateResult; r=GateResult(gate_id='x',validator_name='v',validator_version='1',passed=True); assert r.action is None; r2=GateResult.__class__.derive_default_action(False, None) if hasattr(GateResult,'derive_default_action') else 'block'"` exits 0.

#### Subtask 47: Add `block_validation_action` decision-event emit in router
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtask 46
- **Estimated LOC:** ~40
- **Change:** Helper `_emit_block_validation_action(self, block: Block, *, gate_id: str, action: str, score: Optional[float], issues: List[GateIssue]) -> None`. Emits one `decision_type="block_validation_action"` event per validator that returns a non-pass action. Rationale ≥20 chars interpolating gate_id, action, score, top-3 issues. Emitted from `_run_validator_chain` (Subtask 38) when `result.action in {"regenerate","escalate","block"}`.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_block_validation_action_event_emitted_on_regenerate -v` PASSES.

#### Subtask 48: Implement validator-action consumption in `route_with_self_consistency`
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py`
- **Depends on:** Subtasks 37, 47
- **Estimated LOC:** ~50
- **Change:** In the self-consistency loop, after running the validator chain on a candidate, dispatch on the highest-priority action across all validator results: `block` > `escalate` > `regenerate` > `pass`. Per Phase 4 §1.5 mapping: `block` → mark block as `status="failed"`, exclude from rewrite, return immediately. `escalate` → set `escalation_marker="validator_consensus_fail"`, exit loop, return for rewrite tier. `regenerate` → continue the self-consistency loop (Subtask 41 handles the budget). `pass` → return the candidate.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_validator_block_action_marks_block_failed -v` PASSES.

#### Subtask 49: Add `Courseforge/router/tests/test_validator_action.py`
- **Files:** create `/home/user/Ed4All/Courseforge/router/tests/test_validator_action.py`
- **Depends on:** Subtask 48
- **Estimated LOC:** ~150
- **Change:** Tests: `test_legacy_validator_passed_true_treated_as_pass_action`, `test_legacy_validator_passed_false_treated_as_block_action`, `test_validator_emitting_regenerate_triggers_self_consistency_retry`, `test_validator_emitting_escalate_skips_remaining_outline_retries`, `test_validator_emitting_block_marks_block_failed`, `test_block_validation_action_event_includes_gate_id_and_action`, `test_action_priority_block_over_escalate_over_regenerate_over_pass`, `test_multiple_validators_emit_separate_events`. Use stub validators returning canned `GateResult` instances.
- **Verification:** `pytest Courseforge/router/tests/test_validator_action.py -v` reports ≥8 PASSED.

### K. Inter-tier gate seam (4 subtasks)

#### Subtask 50: Create `Courseforge/router/inter_tier_gates.py` with adapter shims
- **Files:** create `/home/user/Ed4All/Courseforge/router/inter_tier_gates.py`
- **Depends on:** Subtask 46
- **Estimated LOC:** ~180
- **Change:** Module exposes 4 inter-tier validators wrapping the existing `lib/validators/`:
  - `BlockCurieAnchoringValidator(blocks: List[Block]) -> GateResult` — ports the gate logic from `lib/validators/curie_anchoring.py:154` accepting a Block list. Reads `block.content.get("curies", [])` from the outline dict; verifies non-empty; returns `action="regenerate"` on miss.
  - `BlockContentTypeValidator(blocks) -> GateResult` — wraps `lib/validators/content_type.py`; reads `block.content.get("content_type")`; verifies it's in the canonical taxonomy; returns `action="regenerate"` on miss.
  - `BlockPageObjectivesValidator(blocks) -> GateResult` — adapter; reads `(block.objective_ids, block.content.get("key_claims",[]))`; calls the existing LO-specificity check; returns `action="block"` on miss (LO coverage is structural).
  - `BlockSourceRefValidator(blocks, *, manifest_path) -> GateResult` — adapter; reads `block.source_refs`; calls the existing `_resolve_against_manifest`; returns `action="block"` on miss (sourceId not in manifest is structural).
- **Verification:** `python -c "from Courseforge.router.inter_tier_gates import BlockCurieAnchoringValidator, BlockContentTypeValidator, BlockPageObjectivesValidator, BlockSourceRefValidator; assert all([BlockCurieAnchoringValidator, BlockContentTypeValidator, BlockPageObjectivesValidator, BlockSourceRefValidator])"` exits 0.

#### Subtask 51: Generalise `lib/validators/curie_anchoring.py` to accept Block-list input
- **Files:** `/home/user/Ed4All/lib/validators/curie_anchoring.py:154`
- **Depends on:** Subtask 50
- **Estimated LOC:** ~50
- **Change:** Add a new method `_validate_blocks(self, blocks: List[Block]) -> GateResult` parallel to the existing `_validate_pairs(pairs)` (the instruction-pair entry from Wave 135c). The new method reads `(block.content.get("curies",[]), block.content.get("key_claims",[]))` from each Block's outline dict; applies the same CURIE-resolution check the existing path uses; returns `action="regenerate"` (per Phase 4 §1 mapping) on miss. The existing instruction-pair entry stays unchanged.
- **Verification:** `pytest lib/tests/test_curie_anchoring.py -v` PASSES (regression — existing tests stay green); add ONE new test `test_validate_blocks_returns_regenerate_action_on_curie_miss`.

#### Subtask 52: Wire inter-tier validators into the workflow as `inter_tier_validation` phase
- **Files:** `/home/user/Ed4All/config/workflows.yaml` (the `inter_tier_validation` phase from Subtask 3)
- **Depends on:** Subtasks 50, 51
- **Estimated LOC:** ~30
- **Change:** Add `validation_gates` to the `inter_tier_validation` phase YAML, each pointing at the Block-input adapters from Subtask 50: `outline_curie_anchoring` → `Courseforge.router.inter_tier_gates.BlockCurieAnchoringValidator`, `outline_content_type` → `BlockContentTypeValidator`, `outline_page_objectives` → `BlockPageObjectivesValidator`, `outline_source_refs` → `BlockSourceRefValidator`. All severity critical, all `behavior: {on_fail: warn, on_error: warn}` initially (promoted to block in a follow-up wave per Phase 4 §6 calibration policy).
- **Verification:** `python -c "import yaml; d=yaml.safe_load(open('config/workflows.yaml')); wf=next(w for w in d['workflows'] if w['name']=='textbook_to_course'); ph=next(p for p in wf['phases'] if p['name']=='inter_tier_validation'); assert len(ph['validation_gates'])==4"` exits 0.

#### Subtask 53: Add `Courseforge/router/tests/test_inter_tier_gates.py`
- **Files:** create `/home/user/Ed4All/Courseforge/router/tests/test_inter_tier_gates.py`
- **Depends on:** Subtasks 50, 52
- **Estimated LOC:** ~140
- **Change:** Tests: one per adapter — `test_block_curie_anchoring_passes_when_curies_present`, `test_block_curie_anchoring_returns_regenerate_when_missing`, `test_block_content_type_validates_against_taxonomy`, `test_block_page_objectives_returns_block_action_when_objective_unmatched`, `test_block_source_ref_returns_block_action_when_sourceid_unknown`. Use minimal Block fixtures.
- **Verification:** `pytest Courseforge/router/tests/test_inter_tier_gates.py -v` reports ≥5 PASSED.

### L. Constrained-decoding payload extension (3 subtasks)

#### Subtask 54: Confirm `OpenAICompatibleClient.extra_payload` accepts grammar fields
- **Files:** `/home/user/Ed4All/Trainforge/generators/_openai_compatible_client.py:209-256`
- **Depends on:** none (read-only verification)
- **Estimated LOC:** ~5 (defensive comment update only)
- **Change:** No code change — the client already accepts arbitrary `extra_payload` keys (`:252-256`) and Wave-113 already injects `format` / `response_format` for `json_mode=True`. Add a docstring paragraph to `chat_completion` (`:209`) explicitly listing the per-provider grammar fields the router routes through (`grammar`, `format` as schema dict, `guided_grammar`, `guided_json`, `guided_regex`, `extra_body`, `response_format` for json_schema mode). Confirms by inspection that no client surgery is needed.
- **Verification:** `python -c "import inspect; from Trainforge.generators._openai_compatible_client import OpenAICompatibleClient; sig=inspect.signature(OpenAICompatibleClient.chat_completion); assert 'extra_payload' in sig.parameters"` exits 0.

#### Subtask 55: Plumb `extra_payload` through `_BaseLLMProvider._dispatch_call` to direct `_post_with_retry` call
- **Files:** `/home/user/Ed4All/Courseforge/generators/_base.py` (the moved `_dispatch_call`)
- **Depends on:** Subtasks 9, 21
- **Estimated LOC:** ~20
- **Change:** The Phase-1 `_dispatch_call` builds the payload manually (mirrors `Courseforge/generators/_provider.py:442-447`) and calls `self._oa_client._post_with_retry(payload)`. Update the moved version to merge `extra_payload or {}` into the payload BEFORE the POST. This is the load-bearing wire-through that makes Subtask 18's grammar payloads reach the wire.
- **Verification:** `pytest Courseforge/generators/tests/test_base_llm_provider.py::test_dispatch_call_merges_extra_payload_into_request_body -v` PASSES.

#### Subtask 56: Add `Courseforge/generators/tests/test_constrained_decoding_payload.py`
- **Files:** create `/home/user/Ed4All/Courseforge/generators/tests/test_constrained_decoding_payload.py`
- **Depends on:** Subtasks 18, 55
- **Estimated LOC:** ~120
- **Change:** Tests using `httpx.MockTransport` capturing the request body: `test_grammar_payload_for_local_with_gbnf_mode_includes_grammar_field`, `test_grammar_payload_for_ollama_json_schema_mode_includes_format_dict`, `test_grammar_payload_for_together_includes_response_format_json_schema`, `test_grammar_payload_for_vllm_includes_extra_body_guided_json`, `test_grammar_payload_for_anthropic_falls_back_to_json_mode_only`, `test_grammar_mode_env_var_overrides_autodetect`. Each test asserts the exact field name + presence in the captured POST body.
- **Verification:** `pytest Courseforge/generators/tests/test_constrained_decoding_payload.py -v` reports ≥6 PASSED.

### M. Tests — outline + integration (4 subtasks)

#### Subtask 57: Add `Courseforge/generators/tests/test_outline_provider.py`
- **Files:** create
- **Depends on:** Subtask 20
- **Estimated LOC:** ~180
- **Change:** Tests via `httpx.MockTransport`: `test_default_outline_provider_is_local_when_env_unset`, `test_env_var_selects_provider`, `test_outline_kind_bounds_per_block_type`, `test_outline_user_prompt_includes_block_id_and_objectives`, `test_outline_user_prompt_includes_per_block_type_schema_directive`, `test_lenient_json_extraction_recovers_from_markdown_fence`, `test_outline_invalid_json_after_max_retries_raises_outline_exhausted`, `test_outline_validates_against_block_type_json_schema`, `test_outline_appends_touch_with_tier_outline`, `test_outline_failure_emits_decision_event`. Mirror `Trainforge/tests/test_curriculum_alignment_provider.py` patterns.
- **Verification:** `pytest Courseforge/generators/tests/test_outline_provider.py -v` reports ≥10 PASSED.

#### Subtask 58: Add `tests/integration/test_courseforge_two_pass_end_to_end.py`
- **Files:** create
- **Depends on:** Subtasks 31, 52
- **Estimated LOC:** ~200
- **Change:** End-to-end test fixture: 1 mini course with 2 weeks, 4 block types. Mock outline provider returns canned outline JSON; mock rewrite provider returns canned HTML. Run the three new workflow phases via `WorkflowRunner.run_workflow(..., workflow_id="textbook_to_course")` with `COURSEFORGE_TWO_PASS=true`. Assert: every emitted block has `touched_by` entries for both tiers (outline + rewrite); every CURIE in the outline survives to the final HTML; all decision-capture events validate against `decision_event.schema.json` (strict mode); the `inter_tier_validation` phase outputs `blocks_validated_path` and `blocks_failed_path`; failed blocks do NOT have a rewrite-tier Touch.
- **Verification:** `pytest tests/integration/test_courseforge_two_pass_end_to_end.py -v` reports ≥1 PASSED.

#### Subtask 59: Add regression test for legacy single-pass mode (`COURSEFORGE_TWO_PASS=false`)
- **Files:** create `/home/user/Ed4All/tests/integration/test_courseforge_legacy_single_pass.py`
- **Depends on:** Subtask 58
- **Estimated LOC:** ~80
- **Change:** Run `course_generation` workflow against a fixture with `COURSEFORGE_TWO_PASS` unset (and explicitly with `=false`); assert the legacy `content_generation` phase runs (verify via `phase_outputs.content_generation` non-empty) and the new phases (`content_generation_outline` etc.) are skipped. Assert byte-stable (or whitespace-tolerant) match against a pre-Phase-3 golden output snapshot.
- **Verification:** `pytest tests/integration/test_courseforge_legacy_single_pass.py -v` PASSES.

#### Subtask 60: Add strict-schema decision-event test
- **Files:** extend `/home/user/Ed4All/tests/integration/test_courseforge_two_pass_end_to_end.py`
- **Depends on:** Subtask 58
- **Estimated LOC:** ~30
- **Change:** Test `test_all_phase3_decision_events_pass_strict_schema_validation`: with `DECISION_VALIDATION_STRICT=true`, run the integration workflow; assert every emitted decision event passes the strict `decision_event.schema.json` validator. Closes the regression class Wave 120 Phase A re-fixed for the curriculum surface.
- **Verification:** `DECISION_VALIDATION_STRICT=true pytest tests/integration/test_courseforge_two_pass_end_to_end.py::test_all_phase3_decision_events_pass_strict_schema_validation -v` PASSES.

### N. Phase 1 wire-in update (3 subtasks)

#### Subtask 61: Wire `CourseforgeRouter` into `MCP/tools/pipeline_tools.py::_generate_course_content`
- **Files:** `/home/user/Ed4All/MCP/tools/pipeline_tools.py:2812-2836`
- **Depends on:** Subtasks 6, 31
- **Estimated LOC:** ~60
- **Change:** When `_courseforge_two_pass_enabled()` is true, instantiate a `CourseforgeRouter` instead of (or in addition to) the legacy `ContentGeneratorProvider`. Pass `capture=capture` and `policy=load_block_routing_policy()`. Store the router on a local variable `content_router`. Pass `content_router` to `_cgh.build_week_data` (Subtask 62 widens the signature). When `_courseforge_two_pass_enabled()` is false, the legacy `content_provider = ContentGeneratorProvider(capture=capture)` path runs unchanged (preserves Phase 1 behavior).
- **Verification:** `pytest MCP/tests/test_pipeline_tools.py -v -k "two_pass" 2>&1 | head -20` reports new test PASSED.

#### Subtask 62: Update `MCP/tools/_content_gen_helpers.py::_build_content_modules_dynamic` to consume `content_router`
- **Files:** `/home/user/Ed4All/MCP/tools/_content_gen_helpers.py:1854-1907`
- **Depends on:** Subtask 61
- **Estimated LOC:** ~80
- **Change:** Add new optional kwarg `content_router: Optional[CourseforgeRouter] = None` to `_build_content_modules_dynamic` and `build_week_data`. When `content_router` is supplied, build a list of Block stubs (one per (page, block_type) pair the page would emit) with empty `content`; call `content_router.route_all(blocks, source_chunks_by_block_id=..., objectives=...)`; harvest the rewritten Block list; consume `block.content` (now HTML, since rewrite tier ran) into `sections[0]["paragraphs"]`. The legacy `content_provider.generate_page` path stays as the `elif content_provider is not None` branch — exactly as today (`:1854-1907`).
- **Verification:** `pytest MCP/tools/tests/test_content_gen_helpers.py -v -k "router or two_pass"` reports the new tests PASSED.

#### Subtask 63: Add `MCP/tools/tests/test_content_gen_helpers_two_pass.py`
- **Files:** create
- **Depends on:** Subtask 62
- **Estimated LOC:** ~120
- **Change:** Tests: `test_build_content_modules_dispatches_to_router_when_two_pass_enabled`, `test_build_content_modules_falls_back_to_legacy_provider_when_router_none`, `test_router_dispatched_blocks_carry_two_touches_outline_and_rewrite`, `test_failed_outline_blocks_excluded_from_rewrite_in_helper`. Stub the router; assert the helper's I/O contract.
- **Verification:** `pytest MCP/tools/tests/test_content_gen_helpers_two_pass.py -v` reports ≥4 PASSED.

### O. Documentation (2 subtasks)

#### Subtask 64: Add Phase 3 outline + rewrite section to `Courseforge/CLAUDE.md`
- **Files:** `/home/user/Ed4All/Courseforge/CLAUDE.md` (after Phase 2 Block section, ~line 250)
- **Depends on:** Subtasks 28, 31 deliverables (router, providers)
- **Estimated LOC:** ~80
- **Change:** New section `### Phase 3: outline-rewrite two-pass router`. Describe the architecture (outline tier 7B local → inter-tier validators → rewrite tier configurable cloud); cross-link `Courseforge/router/router.py`, `_outline_provider.py`, `_rewrite_provider.py`, `policy.py`. Brief description of `block_routing.yaml` schema with one example. Brief description of `validation_attempts` + `escalation_marker` per-block fields and their interaction with `route_with_self_consistency`. Explicit mention of feature flag (`COURSEFORGE_TWO_PASS=true` opt-in; default off; legacy `content_generation` phase runs when off).
- **Verification:** `grep -c "Phase 3: outline-rewrite\|content_generation_outline\|escalation_marker" Courseforge/CLAUDE.md` returns ≥3.

#### Subtask 65: Add 8 new env-var rows to root `CLAUDE.md` flag table
- **Files:** `/home/user/Ed4All/CLAUDE.md:728` (insert before `COURSEFORGE_PROVIDER` row at `:730`)
- **Depends on:** Subtasks 6, 13, 22, 33
- **Estimated LOC:** ~80
- **Change:** Insert 8 rows alphabetically (between `COURSEFORGE_EMIT_BLOCKS` and `COURSEFORGE_PROVIDER`):
  - `COURSEFORGE_BLOCK_ROUTING_PATH` (overrides `Courseforge/config/block_routing.yaml` default; cross-link to `schemas/courseforge/block_routing.schema.json`)
  - `COURSEFORGE_OUTLINE_GRAMMAR_MODE` (`gbnf|json_schema|json_object|none`; autodetect from provider)
  - `COURSEFORGE_OUTLINE_MODEL` (default `qwen2.5:7b-instruct-q4_K_M`)
  - `COURSEFORGE_OUTLINE_N_CANDIDATES` (default 3; self-consistency budget)
  - `COURSEFORGE_OUTLINE_PROVIDER` (default `local`)
  - `COURSEFORGE_OUTLINE_REGEN_BUDGET` (default 3; budget exhausted → escalation)
  - `COURSEFORGE_REWRITE_MODEL` (default `claude-sonnet-4-6`)
  - `COURSEFORGE_REWRITE_PROVIDER` (default `anthropic`)
  - `COURSEFORGE_TWO_PASS` (master gate; default `false`; opt-in)
  Each row with full prose + cross-links matching the density of the existing `COURSEFORGE_PROVIDER` row.
- **Verification:** `grep -cE "COURSEFORGE_(TWO_PASS|OUTLINE_PROVIDER|OUTLINE_MODEL|OUTLINE_N_CANDIDATES|OUTLINE_REGEN_BUDGET|OUTLINE_GRAMMAR_MODE|REWRITE_PROVIDER|REWRITE_MODEL|BLOCK_ROUTING_PATH)" CLAUDE.md` returns ≥9.

### P. Operator-facing smoke (1 subtask)

#### Subtask 66: End-to-end operator smoke command sequence
- **Files:** N/A — this is the final operator-facing verification (run after all other subtasks land)
- **Depends on:** Subtasks 1-65
- **Estimated LOC:** N/A (operator runbook)
- **Change:** Operator-facing runbook documented in `Courseforge/CLAUDE.md` (Subtask 64) and the "Final smoke test" section below. The full sequence exercises: (1) the router-resolution unit suite; (2) the constrained-decoding payload tests; (3) the integration test suite under `COURSEFORGE_TWO_PASS=true`; (4) the legacy regression suite under `COURSEFORGE_TWO_PASS=false`; (5) a real ed4all-CLI invocation against a fixture course; (6) provenance audit on the resulting `touchedBy[]` chain.
- **Verification:** See "Final smoke test" section.

---

## Execution sequencing

**Strict-serial within categories:**
- A: 1 → 2 → 3 → 4 → 5 → 6
- B: 7 → 8
- C: 9 → 10 → 11 → 12
- D: 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20
- E: 21 → 22 → 23 → 24 → 25 → 26 → 27
- F: 28 → 29 → 30 → 31 → 32
- G: 33 → 34 → 35 → 36
- H: 37 → 38 → 39 → 40
- I: 41 → 42 → 43 → 44 → 45
- J: 46 → 47 → 48 → 49
- K: 50 → 51 → 52 → 53
- L: 54 → 55 → 56
- M: 57 → 58 → 59 → 60
- N: 61 → 62 → 63
- O: 64 → 65
- P: 66

**Wave plan (Phase 3 is split into three waves):**

### Wave N — Foundational classes + workflow split (default OFF; legacy unchanged)

Subtasks: A (1-6) + B (7-8) + C (9-12) + D (13-20) + E (21-27) + F (28-32) + L (54-56). Total: 32 subtasks.

Parallelisable batches inside Wave N:
- **Day 1-2 batch (parallelisable):** A (1-6) + B (7-8) (workflow + schema; independent surfaces).
- **Day 3 batch (sequential):** C (9-12) (`_BaseLLMProvider` extraction).
- **Day 4-6 batch (parallelisable):** D (13-20) and E (21-27) — once `_BaseLLMProvider` lands, OutlineProvider and RewriteProvider can be authored in parallel by two engineers.
- **Day 7 batch (parallelisable):** F (28-32) and L (54-56) — router skeleton + grammar plumbing tests; independent.

### Wave N+1 — Router policy + self-consistency + regen budget + validators + tests (still default OFF)

Subtasks: G (33-36) + H (37-40) + I (41-45) + J (46-49) + K (50-53) + M (57-60). Total: 24 subtasks.

Parallelisable batches inside Wave N+1:
- **Day 1 batch (parallelisable):** G (33-36) and J (46-49) — policy loader + GateResult action field; independent.
- **Day 2-3 batch (sequential):** H (37-40) → I (41-45) (self-consistency must land before regen budget builds on it).
- **Day 4 batch (parallelisable):** K (50-53) and M (57-60) — inter-tier gates + outline-provider tests; independent.

### Wave N+2 — Phase 1 wire-in flip, docs, smoke (the moment two-pass becomes user-facing)

Subtasks: N (61-63) + O (64-65) + P (66). Total: 6 subtasks.

Parallelisable batches inside Wave N+2:
- **Day 1 batch (parallelisable):** N (61-63) and O (64-65) — wire-in update + docs; independent.
- **Day 2 batch (sequential):** P (66) — final smoke after everything else lands.

**Migration rollout strategy** (mirrors Phase 2 pattern):
1. **Wave N** — Land subtasks 1-32. `COURSEFORGE_TWO_PASS=false` by default. Legacy `content_generation` phase runs unchanged. New router + providers are inert (no caller wires them yet). All unit tests + L-category constrained-decoding tests green.
2. **Wave N+1** — Land subtasks 33-60. Default still `false`. The new phases (`content_generation_outline` + `inter_tier_validation` + `content_generation_rewrite`) exist in YAML but skip due to the `enabled_when_env` predicate. Operators can opt-in to `true` to test the new path. All integration tests green when run with the flag on.
3. **Wave N+2** — Land subtasks 61-66. Default still `false`. The Phase 1 wire-in is updated to dispatch through the router when the flag is set; otherwise the byte-stable Phase 1 path runs. After this wave, an operator who sets `COURSEFORGE_TWO_PASS=true` gets the full two-pass deliverable with zero further setup.
4. **Phase 3 followup (NOT this plan):** flip `COURSEFORGE_TWO_PASS` default to `true` after one wave's smoke run on a real course (e.g. `rdf-shacl-551-2`); promote the four inter-tier validators from `behavior: warn` to `behavior: block` after Phase 4 calibration data lands; drop the legacy `content_generation` phase.

---

## Final smoke test

A single end-to-end verification an operator runs to prove Phase 3 landed:

```bash
# 1. Run the unit + integration test suite for the new router + providers:
pytest Courseforge/generators/tests/test_base_llm_provider.py \
       Courseforge/generators/tests/test_outline_provider.py \
       Courseforge/generators/tests/test_rewrite_provider.py \
       Courseforge/generators/tests/test_constrained_decoding_payload.py \
       Courseforge/router/tests/test_router.py \
       Courseforge/router/tests/test_policy.py \
       Courseforge/router/tests/test_self_consistency.py \
       Courseforge/router/tests/test_regen_budget.py \
       Courseforge/router/tests/test_validator_action.py \
       Courseforge/router/tests/test_inter_tier_gates.py \
       MCP/tools/tests/test_content_gen_helpers_two_pass.py -v

# 2. Decision-event strict-schema test:
DECISION_VALIDATION_STRICT=true pytest \
  lib/tests/test_phase3_decision_event_enums.py \
  tests/integration/test_courseforge_two_pass_end_to_end.py::test_all_phase3_decision_events_pass_strict_schema_validation -v

# 3. Legacy regression — TWO_PASS off byte-stable:
unset COURSEFORGE_TWO_PASS
pytest tests/integration/test_courseforge_legacy_single_pass.py -v
pytest Courseforge/scripts/tests/ -v   # Phase 2 regression suite stays green

# 4. End-to-end with the new path on, against a fixture course:
export COURSEFORGE_TWO_PASS=true
export COURSEFORGE_OUTLINE_PROVIDER=local
export LOCAL_SYNTHESIS_BASE_URL=http://localhost:11434/v1
export COURSEFORGE_OUTLINE_MODEL=qwen2.5:7b-instruct-q4_K_M
export COURSEFORGE_REWRITE_PROVIDER=together
export TOGETHER_API_KEY=...   # operator-supplied
ed4all run textbook_to_course --course-code DEMO_303 --weeks 1

# 5. Verify the three new phases ran:
ls Courseforge/exports/PROJ-DEMO_303-*/training-captures/courseforge/DEMO_303/ \
  | grep -E "phase_courseforge-content-generator-(outline|rewrite)"   # both present

# 6. Verify per-block touch chain has BOTH outline + rewrite tiers:
jq -r '.[] | select(.block_type=="objective") | .touched_by | map(.tier) | join(",")' \
  Courseforge/exports/PROJ-DEMO_303-*/03_content_development/blocks_final.json \
  | sort -u   # expect "outline,rewrite" or "outline,validation,rewrite"

# 7. Verify decision-event volume is sane (no explosion):
wc -l training-captures/courseforge/DEMO_303/phase_courseforge-content-generator-outline/decisions_*.jsonl
# expect: ~1 outline_block_call event per (page × block) — for a 1-week × ~7 pages × ~10 blocks corpus, ~70 events.

# 8. Verify per-tier provider routing actually fired:
jq -r 'select(.decision_type=="block_outline_call") | .ml_features.provider' \
  training-captures/courseforge/DEMO_303/phase_courseforge-content-generator-outline/decisions_*.jsonl | sort -u
# expect: "local"
jq -r 'select(.decision_type=="block_rewrite_call") | .ml_features.provider' \
  training-captures/courseforge/DEMO_303/phase_courseforge-content-generator-rewrite/decisions_*.jsonl | sort -u
# expect: "together" (or "anthropic" depending on per-block-type override)

# 9. Verify the inter-tier validators fired:
jq -r 'select(.decision_type=="block_validation_action") | .ml_features.gate_id' \
  training-captures/courseforge/DEMO_303/phase_courseforge-content-generator-outline/decisions_*.jsonl | sort -u
# expect: at least the four wired gate ids (or empty if every block passed all gates first time)

# 10. Verify the regen-budget primitive: take a misbehaving fixture page known to fail outline-tier,
# confirm `validation_attempts` increments and (if exhausted) `escalation_marker` is set:
jq -r '.[] | select(.escalation_marker != null) | "\(.block_id): \(.escalation_marker) (attempts=\(.validation_attempts))"' \
  Courseforge/exports/PROJ-DEMO_303-*/03_content_development/blocks_final.json
# expect: zero or more lines depending on corpus difficulty; per-line format proves the field semantics.

# 11. Verify per-block-type policy override works:
cat Courseforge/config/block_routing.yaml   # shows e.g. assessment_item.rewrite.provider=anthropic
jq -r 'select(.decision_type=="block_rewrite_call" and .ml_features.block_type=="assessment_item") | .ml_features.provider' \
  training-captures/courseforge/DEMO_303/phase_courseforge-content-generator-rewrite/decisions_*.jsonl | sort -u
# expect: "anthropic" (matches the YAML override)
```

**Acceptance criteria:** all `pytest` invocations PASS; commands 4-11 return non-empty as documented; commands 5-9 prove both tiers fired with correct provider routing; command 10 proves the regen-budget + escalation primitive is wired; command 11 proves the per-block-type YAML override surface is wired.

---

### Critical Files for Implementation
- `/home/user/Ed4All/Courseforge/router/router.py` (NEW — `CourseforgeRouter` + `BlockProviderSpec`)
- `/home/user/Ed4All/Courseforge/generators/_outline_provider.py` (NEW — `OutlineProvider` + per-block-type GBNF/JSON-Schema map)
- `/home/user/Ed4All/Courseforge/generators/_rewrite_provider.py` (NEW — `RewriteProvider` + escalated-prompt branch)
- `/home/user/Ed4All/Courseforge/generators/_base.py` (NEW — `_BaseLLMProvider` extracted skeleton)
- `/home/user/Ed4All/Courseforge/router/policy.py` (NEW — `block_routing.yaml` loader + `BlockRoutingPolicy`)
- `/home/user/Ed4All/Courseforge/router/inter_tier_gates.py` (NEW — Block-input adapters for the 4 promoted validators)
- `/home/user/Ed4All/config/workflows.yaml` (split `content_generation` into 3 phases gated on `COURSEFORGE_TWO_PASS=true`)
- `/home/user/Ed4All/MCP/hardening/validation_gates.py` (extend `GateResult` with `action: Optional[str]`)
- `/home/user/Ed4All/MCP/tools/_content_gen_helpers.py` (wire `CourseforgeRouter` into `_build_content_modules_dynamic` when flag set)
- `/home/user/Ed4All/schemas/events/decision_event.schema.json` (add 4 `decision_type` + 2 `phase` enum values)
