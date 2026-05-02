# Phase 3.5 Detailed Execution Plan — Post-Rewrite Validation + Symmetric Gates + Phase 3a Env-Var Fixes

Refines `plans/courseforge_architecture_roadmap.md` §2 (row Phase 3.5) and §3.6/3.7/3.8 into atomic, individually-verifiable subtasks. Mirrors `plans/phase3_two_pass_router_detailed.md` granularity. **Wave-N premise: Phase 3 is ~75% landed (Worker I in flight on `Courseforge/router/router.py` for regen-budget Subtasks 41-45). Pull-rebase before reading router state.**

---

## Investigation findings (locked)

- **Router self-consistency loop is in-tree at `Courseforge/router/router.py:816-1041`**. `route_with_self_consistency` already increments `cumulative_attempts` (`:981-985`) and stamps `escalation_marker="outline_budget_exhausted"` on budget exhaustion (`:992-1003`). The post-failure path does NOT yet inject failure context into the next prompt — every retry re-rolls with the same prompt. Phase 3.5 wires `_append_remediation_for_gates(prompt, failures)` into this loop between iterations.
- **`_DEFAULT_OUTLINE_REGEN_BUDGET = 3` at `router.py:103`**. Phase 3.5 bumps to `10` and adds a sibling `_DEFAULT_REWRITE_REGEN_BUDGET = 10`. Env-var hooks already exist (`_ENV_OUTLINE_REGEN_BUDGET = "COURSEFORGE_OUTLINE_REGEN_BUDGET"` at `:88`); add `_ENV_REWRITE_REGEN_BUDGET = "COURSEFORGE_REWRITE_REGEN_BUDGET"`.
- **CURIE-preservation gate is in `Courseforge/generators/_rewrite_provider.py`** with a direct port of `Trainforge/generators/_local_provider.py:548-583::_missing_preserve_tokens` + `_append_preserve_remediation`. The `_REWRITE_SYSTEM_PROMPT` is at `_rewrite_provider.py:129-148`; `MAX_PARSE_RETRIES = 2` at `:114`. Phase 3.5 generalizes the remediation builder out of this provider into a shared module.
- **`inter_tier_gates.py` does not exist yet** in tree — verified. Phase 3 plan Subtask 50 lands four `Block*Validator` adapters there. Phase 3.5 makes them shape-discriminating (block.content as outline-tier dict OR rewrite-tier HTML string).
- **`block_routing.yaml` is at `Courseforge/config/block_routing.yaml`**, version 1; `defaults.outline.model="qwen2.5:7b-instruct-q4_K_M"` and `defaults.rewrite.model="claude-sonnet-4-6"` are hardcoded literals. Phase 3a env-var fix: read `COURSEFORGE_OUTLINE_MODEL` / `COURSEFORGE_REWRITE_MODEL` before falling back to these defaults.
- **`workflows.yaml::textbook_to_course` already carries `content_generation_outline` (`:757`), `inter_tier_validation` (`:790`), `content_generation_rewrite` (`:860`)**. Each gated `enabled_when_env: "COURSEFORGE_TWO_PASS=true"`. The legacy `content_generation` phase carries `enabled_when_env: "COURSEFORGE_TWO_PASS!=true"` at `:678`. Phase 3.5 inserts a NEW phase `post_rewrite_validation` between `content_generation_rewrite` and `packaging`. The `packaging` phase's `depends_on_when_env_value: [content_generation_rewrite]` (Phase 3 contract) bumps to `[post_rewrite_validation]`.
- **`Block.content` type is `Union[str, Dict[str, Any]]`** per `Courseforge/scripts/blocks.py:223-291`. Outline tier emits dict; rewrite tier emits HTML string. The four `Block*Validator` shape-adapter logic dispatches on `isinstance(block.content, dict)`.
- **`Block.touched_by` is a tuple of Touch entries**; `Block.with_touch(touch)` is the immutable-append helper at `blocks.py:303-309`. Touch tier values pre-Phase-3.5 are `{"outline","rewrite"}` (verify via `grep "_TIER_VALUES" Courseforge/scripts/blocks.py`); Phase 3.5 adds `"outline_val"` and `"rewrite_val"`.
- **Decision-event `block_validation_action` enum exists** per Phase 3 Subtask 7 (`schemas/events/decision_event.schema.json::decision_type`). Phase 3.5 extends the event's `ml_features` payload with `tier="outline"|"rewrite"` field; schema does NOT pin `additionalProperties: false` on `ml_features`, so no schema change.
- **`MCP/core/workflow_runner.py` honours `enabled_when_env`** per Phase 3 Subtask 1 (`:1150-1169`). Phase 3.5 reuses the same predicate grammar.
- **DART files have hardcoded model `claude-sonnet-4-20250514`** at: `DART/pdf_converter/claude_processor.py:228`, `alt_text_generator.py:42`, `cli.py:99-100`, `converter.py:69`. (NOT in this plan — those are Phase 3c, bundled into Phase 6.)

---

## Pre-resolved decisions

1. **Symmetric-validation phase position.** New `post_rewrite_validation` phase between `content_generation_rewrite` and `packaging` in BOTH `textbook_to_course` AND `course_generation` workflows. Gated `enabled_when_env: "COURSEFORGE_TWO_PASS=true"`. The `packaging.depends_on_when_env_value` predicate flips from `[content_generation_rewrite]` to `[post_rewrite_validation]`.
2. **Same validator classes, shape-discriminating adapters.** The four `Block*Validator` classes in `Courseforge/router/inter_tier_gates.py` (Phase 3 deliverable) gain a `_validate_one_block(block: Block) -> List[GateIssue]` helper that branches on `isinstance(block.content, dict)` (outline tier) vs `isinstance(block.content, str)` (rewrite tier). For HTML inputs, the validator scans the rendered HTML via the existing `lib/ontology/curie_extraction.py::extract_curies_from_html` (regex-based), or for content_type a regex extractor of `data-cf-content-type=...` attributes, or for objective refs a regex extractor of `data-cf-objective-id`.
3. **Remediation builder location.** New module `Courseforge/router/remediation.py` exposing `_append_remediation_for_gates(prompt: str, failures: List[GateResult]) -> str` (general) and `_append_preserve_remediation(prompt: str, missing_tokens: List[str]) -> str` (preserve-token specialization, used by RewriteProvider). The RewriteProvider's existing `_append_preserve_remediation` and `_missing_preserve_tokens` helpers are RE-EXPORTED from `remediation.py` and the local copies in `_rewrite_provider.py` reduce to one-line `from Courseforge.router.remediation import ...`.
4. **Per-failure-mode remediation directives table.** Module-level constant `_REMEDIATION_DIRECTIVES_BY_GATE_ID: Dict[str, str]` in `remediation.py`. Keys: `"outline_curie_anchoring"`, `"outline_content_type"`, `"outline_page_objectives"`, `"outline_source_refs"`, `"rewrite_curie_anchoring"`, `"rewrite_content_type"`, `"rewrite_page_objectives"`, `"rewrite_source_refs"`. Values are short imperative directives — e.g. `outline_curie_anchoring → "Preserve every CURIE verbatim. Re-emit the JSON object including all source-declared CURIEs in 'curies'."` Cross-tier overlap is intentional — both outline and rewrite share remediation copy when the failure mode is the same.
5. **Regen-budget bump.** `_DEFAULT_OUTLINE_REGEN_BUDGET = 3 → 10` at `router.py:103`. New `_DEFAULT_REWRITE_REGEN_BUDGET = 10` constant. New env var `COURSEFORGE_REWRITE_REGEN_BUDGET` resolved by the same precedence chain `_resolve_regen_budget` already implements.
6. **Per-block-type override surface in `block_routing.yaml`.** Schema (`schemas/courseforge/block_routing.schema.json`) already accommodates per-block-type `regen_budget` (Worker G's fast-lookup map). Phase 3.5 adds `regen_budget_rewrite` as a sibling field; `BlockRoutingPolicy.regen_budget_rewrite_by_block_type` parallels `regen_budget_by_block_type`.
7. **Symmetric failure escalation policy (per roadmap §6.7 recommendation).** Re-roll → escalate → fail-closed. Implementation: `route_with_self_consistency` first calls `_run_validator_chain` with the inter-tier gates; on failure injects remediation and retries up to budget; on budget exhaustion sets `escalation_marker="validator_consensus_fail"` and returns. The rewrite-tier symmetric path uses a sibling method `route_rewrite_with_remediation` with the same shape but the rewrite gate set + `outline_skipped_by_policy` not applicable.
8. **Phase 3a env-var fix targets.** Three files: `Courseforge/config/block_routing.yaml`, `Courseforge/generators/_rewrite_provider.py`, `Courseforge/router/router.py`. Phase 3a fix is "read tier-default env var BEFORE the hardcoded literal." For `block_routing.yaml` this is a Python-side change in `Courseforge/router/policy.py::load_block_routing_policy`: when `defaults.outline.model` is the hardcoded sentinel literal AND `COURSEFORGE_OUTLINE_MODEL` is set, the env var wins. Same shape for rewrite. For `_rewrite_provider.py`: existing `__init__` already reads `os.environ.get(ENV_MODEL)`, but `DEFAULT_MODEL = "claude-sonnet-4-6"` at `:96` is the hardcoded fallback path — verify the env-var-first chain is wired correctly. For `router.py`: `_HARDCODED_DEFAULTS` table at `:213-260` populates from constants, but `_read_tier_env` already reads env vars first — verify `_resolve_spec` precedence is env > hardcoded (it is, per `:374-380`).
9. **Touch chain extension.** Phase 3.5 adds two new tier values `"outline_val"` and `"rewrite_val"` to `Courseforge/scripts/blocks.py::_TIER_VALUES`. Each gate that fires (passing OR failing) appends a Touch via `block.with_touch(Touch(tier="outline_val"|"rewrite_val", purpose="validation_pass"|"validation_fail", ...))`. SHACL shape `BlockShape` (`schemas/context/courseforge_v1.shacl.ttl::BlockShape`) and JSON Schema `$defs.Touch.tier` enum need expanding accordingly.
10. **Decision-event `block_validation_action` extended with `tier` field.** No schema change; `ml_features` is open. Router emit at `_emit_block_validation_action` interpolates `tier="outline"|"rewrite"` per call.

---

## Atomic subtasks

Estimated total LOC: ~1,800 (300 remediation module + 250 rewrite_val phase wiring + 200 shape-discriminating adapters + 120 Phase 3a env-var fixes + 250 router-side remediation injection + 200 Touch tier expansion + 150 schema/SHACL updates + 350 tests + 100 docs).

### A. Remediation module extraction (4 subtasks)

#### Subtask 1: Create `Courseforge/router/remediation.py` skeleton
- **Files:** create `/home/user/Ed4All/Courseforge/router/remediation.py`
- **Depends on:** none
- **Estimated LOC:** ~80
- **Change:** Module docstring describing the generalized remediation builder per roadmap §3.8. Imports `GateResult`, `GateIssue` from `MCP.hardening.validation_gates`. Define module-level `_REMEDIATION_DIRECTIVES_BY_GATE_ID: Dict[str, str]` with 8 keys (4 outline gates + 4 rewrite gates) — each value a short imperative directive (~80 chars) per pre-resolved decision #4. Stub functions `_append_remediation_for_gates(prompt: str, failures: List[GateResult]) -> str`, `_append_preserve_remediation(prompt: str, missing_tokens: List[str], in_keys: tuple = ("body",)) -> str`, `_missing_preserve_tokens(content: Any, tokens: List[str], in_keys: tuple) -> List[str]`. Bodies raise `NotImplementedError` for now.
- **Verification:** `python -c "from Courseforge.router.remediation import _append_remediation_for_gates, _REMEDIATION_DIRECTIVES_BY_GATE_ID; assert 'outline_curie_anchoring' in _REMEDIATION_DIRECTIVES_BY_GATE_ID and 'rewrite_curie_anchoring' in _REMEDIATION_DIRECTIVES_BY_GATE_ID"` exits 0.

#### Subtask 2: Implement `_append_remediation_for_gates(prompt, failures)`
- **Files:** `/home/user/Ed4All/Courseforge/router/remediation.py`
- **Depends on:** Subtask 1
- **Estimated LOC:** ~70
- **Change:** Implementation: for each `GateResult` in `failures` whose `action != "pass"`, look up the directive in `_REMEDIATION_DIRECTIVES_BY_GATE_ID[gate_result.gate_id]` (fall back to a generic "Re-emit correctly per the {validator_name} contract" when not found). Build a single appended block: `"\n\nYour previous attempt failed validation:\n- [<gate_id>] <issue.message>\n  Correct by: <directive>\n..."` per failure. Truncate each issue.message at 200 chars to keep prompt size bounded. Returns `prompt + appended_block`.
- **Verification:** `python -c "from Courseforge.router.remediation import _append_remediation_for_gates; from MCP.hardening.validation_gates import GateResult, GateIssue; r=GateResult(gate_id='outline_curie_anchoring',validator_name='cv',validator_version='1',passed=False,issues=[GateIssue(severity='critical',code='CURIE_DROPPED',message='sh:NodeShape was dropped')],action='regenerate'); out=_append_remediation_for_gates('original prompt', [r]); assert 'sh:NodeShape was dropped' in out and 'CURIE' in out and 'failed validation' in out"` exits 0.

#### Subtask 3: Port `_missing_preserve_tokens` + `_append_preserve_remediation` from RewriteProvider
- **Files:** `/home/user/Ed4All/Courseforge/router/remediation.py`, `/home/user/Ed4All/Courseforge/generators/_rewrite_provider.py`
- **Depends on:** Subtask 2
- **Estimated LOC:** ~60 (move) + ~10 (re-export shim)
- **Change:** Move the existing `_missing_preserve_tokens` and `_append_preserve_remediation` static methods from `_rewrite_provider.py` into `remediation.py` as module-level functions. Adapt the signature to accept `content: Any` (str or dict) — when str, search for tokens in the string body; when dict, search in `content.get("body","")` or each key in `in_keys` per the Trainforge precedent at `_local_provider.py:548-583`. Replace the `_rewrite_provider.py` definitions with `from Courseforge.router.remediation import _missing_preserve_tokens, _append_preserve_remediation`. Existing tests in `Courseforge/generators/tests/test_rewrite_provider.py` must stay green byte-for-byte.
- **Verification:** `pytest Courseforge/generators/tests/test_rewrite_provider.py::test_curie_preservation_gate_fires_remediation_on_drop -v` PASSES (regression).

#### Subtask 4: Add `Courseforge/router/tests/test_remediation.py`
- **Files:** create `/home/user/Ed4All/Courseforge/router/tests/test_remediation.py`
- **Depends on:** Subtasks 2, 3
- **Estimated LOC:** ~120
- **Change:** Tests: `test_append_remediation_for_gates_emits_one_block_per_failure`, `test_append_remediation_for_gates_uses_directive_table_lookup`, `test_append_remediation_for_gates_falls_back_to_generic_directive`, `test_append_remediation_for_gates_truncates_long_issue_messages`, `test_append_preserve_remediation_emits_token_list`, `test_missing_preserve_tokens_dict_content_searches_in_keys`, `test_missing_preserve_tokens_str_content_searches_full_body`, `test_remediation_for_pass_action_is_noop`. Reuse the `GateResult` fixture pattern from existing validator tests.
- **Verification:** `pytest Courseforge/router/tests/test_remediation.py -v` reports ≥8 PASSED.

### B. Inter-tier-gate shape discrimination (5 subtasks)

#### Subtask 5: Verify `Courseforge/router/inter_tier_gates.py` exists or create skeleton
- **Files:** verify or create `/home/user/Ed4All/Courseforge/router/inter_tier_gates.py`
- **Depends on:** none (or: depends on Phase 3 Subtask 50 if that already landed)
- **Estimated LOC:** ~250 (only if new)
- **Change:** Phase 3 detailed plan Subtask 50 specifies the four `Block*Validator` classes here. **Pull-rebase first**; if Phase 3 Subtask 50 has landed, this subtask is a no-op (skip-and-document). If not landed, port the four-validator scaffolding per Phase 3 Subtask 50's spec.
- **Verification:** `python -c "from Courseforge.router.inter_tier_gates import BlockCurieAnchoringValidator, BlockContentTypeValidator, BlockPageObjectivesValidator, BlockSourceRefValidator; assert all([BlockCurieAnchoringValidator, BlockContentTypeValidator, BlockPageObjectivesValidator, BlockSourceRefValidator])"` exits 0.

#### Subtask 6: Implement shape-discriminating dispatch on `BlockCurieAnchoringValidator`
- **Files:** `/home/user/Ed4All/Courseforge/router/inter_tier_gates.py`
- **Depends on:** Subtask 5
- **Estimated LOC:** ~50
- **Change:** Add helper `_extract_curies(block: Block) -> List[str]`: when `isinstance(block.content, dict)` returns `block.content.get("curies", [])`; when `isinstance(block.content, str)` returns `lib.ontology.curie_extraction.extract_curies_from_html(block.content)` (existing helper). The validator's `validate(inputs)` walks the per-block list and emits `action="regenerate"` on miss.
- **Verification:** `pytest Courseforge/router/tests/test_inter_tier_gates.py::test_curie_anchoring_handles_dict_content_outline_tier -v` and `::test_curie_anchoring_handles_str_content_rewrite_tier -v` both PASS.

#### Subtask 7: Implement shape-discriminating dispatch on `BlockContentTypeValidator`
- **Files:** `/home/user/Ed4All/Courseforge/router/inter_tier_gates.py`
- **Depends on:** Subtask 6
- **Estimated LOC:** ~50
- **Change:** Helper `_extract_content_type(block: Block) -> Optional[str]`: when dict, returns `block.content.get("content_type")`; when str, regex-extracts the first `data-cf-content-type="<value>"` attribute. Validator emits `action="regenerate"` when extracted value is not in the canonical 8-value taxonomy (`schemas/taxonomies/content_type.schema.json::SectionContentType`).
- **Verification:** `pytest Courseforge/router/tests/test_inter_tier_gates.py::test_content_type_handles_html_attribute_extraction -v` PASSES.

#### Subtask 8: Implement shape-discriminating dispatch on `BlockPageObjectivesValidator` and `BlockSourceRefValidator`
- **Files:** `/home/user/Ed4All/Courseforge/router/inter_tier_gates.py`
- **Depends on:** Subtask 7
- **Estimated LOC:** ~80
- **Change:** Helper `_extract_objective_refs(block: Block) -> List[str]`: dict path returns `block.content.get("objective_refs", [])`; str path regex-extracts `data-cf-objective-id="<id>"` matches. Helper `_extract_source_refs(block: Block) -> List[Dict]`: dict path returns `block.content.get("source_refs", [])`; str path regex-extracts `data-cf-source-ids="<ids>"` and synthesizes `[{"sourceId": id, "role": "contributing"}]` per id. Both validators emit `action="block"` on miss (structural — LO ID + sourceId are not soft failures).
- **Verification:** `pytest Courseforge/router/tests/test_inter_tier_gates.py::test_page_objectives_extracts_html_data_cf_objective_id -v` and `::test_source_ref_extracts_html_data_cf_source_ids -v` both PASS.

#### Subtask 9: Add `Courseforge/router/tests/test_inter_tier_gates_shape_dispatch.py`
- **Files:** create `/home/user/Ed4All/Courseforge/router/tests/test_inter_tier_gates_shape_dispatch.py`
- **Depends on:** Subtask 8
- **Estimated LOC:** ~180
- **Change:** Tests covering each of the 4 validators × 2 tiers × 2 outcomes (pass / fail) = 16 cases. Plus regression: `test_legacy_dict_content_path_unchanged_byte_stable` proving the outline-tier (dict) path emits identical GateResult shape as Phase 3.
- **Verification:** `pytest Courseforge/router/tests/test_inter_tier_gates_shape_dispatch.py -v` reports ≥16 PASSED.

### C. `post_rewrite_validation` workflow phase (4 subtasks)

#### Subtask 10: Add `post_rewrite_validation` phase to `config/workflows.yaml::textbook_to_course`
- **Files:** `/home/user/Ed4All/config/workflows.yaml:860-911` (between `content_generation_rewrite` and `packaging`)
- **Depends on:** Subtask 9
- **Estimated LOC:** ~60
- **Change:** Insert new phase entry after `content_generation_rewrite`:
  - `name: post_rewrite_validation`
  - `agents: []` (Python-only — runs validators in-process)
  - `parallel: false`
  - `depends_on: [content_generation_rewrite]`
  - `timeout_minutes: 10`
  - `enabled_when_env: "COURSEFORGE_TWO_PASS=true"`
  - `outputs: [blocks_validated_path, blocks_failed_path]`
  - `validation_gates`: 4 entries — `rewrite_curie_anchoring` → `BlockCurieAnchoringValidator`, `rewrite_content_type` → `BlockContentTypeValidator`, `rewrite_page_objectives` → `BlockPageObjectivesValidator`, `rewrite_source_refs` → `BlockSourceRefValidator`. All severity critical, all `behavior: {on_fail: warn, on_error: warn}` initially (promoted to `block` after one wave's calibration).
- Update `packaging.depends_on_when_env_value: [content_generation_rewrite]` (`:911`) → `[post_rewrite_validation]` so packaging waits for post-rewrite validation when two-pass is on.
- **Verification:** `python -c "import yaml; d=yaml.safe_load(open('config/workflows.yaml')); wf=next(w for w in d['workflows'] if w['name']=='textbook_to_course'); ph=next((p for p in wf['phases'] if p['name']=='post_rewrite_validation'),None); assert ph is not None and len(ph['validation_gates'])==4"` exits 0.

#### Subtask 11: Mirror `post_rewrite_validation` into `course_generation` workflow
- **Files:** `/home/user/Ed4All/config/workflows.yaml:113-138`
- **Depends on:** Subtask 10
- **Estimated LOC:** ~50
- **Change:** Same insertion in the slimmer `course_generation` workflow. Update `packaging.depends_on_when_env_value: [content_generation_rewrite]` → `[post_rewrite_validation]`.
- **Verification:** `python -c "import yaml; d=yaml.safe_load(open('config/workflows.yaml')); wf=next(w for w in d['workflows'] if w['name']=='course_generation'); names=[p['name'] for p in wf['phases']]; assert 'post_rewrite_validation' in names"` exits 0.

#### Subtask 12: Add `_LEGACY_PHASE_OUTPUT_KEYS` entry for `post_rewrite_validation`
- **Files:** `/home/user/Ed4All/MCP/core/workflow_runner.py` (the `_LEGACY_PHASE_OUTPUT_KEYS` mapping)
- **Depends on:** Subtask 11
- **Estimated LOC:** ~5
- **Change:** Add `"post_rewrite_validation": ["blocks_validated_path", "blocks_failed_path"]` to the mapping. Mirrors the Phase 3 `inter_tier_validation` entry.
- **Verification:** `python -c "from MCP.core.workflow_runner import _LEGACY_PHASE_OUTPUT_KEYS; assert 'post_rewrite_validation' in _LEGACY_PHASE_OUTPUT_KEYS"` exits 0.

#### Subtask 13: Add `MCP/tools/pipeline_tools.py` post-rewrite-validation invocation
- **Files:** `/home/user/Ed4All/MCP/tools/pipeline_tools.py` (search for `_run_inter_tier_validation` for sibling pattern)
- **Depends on:** Subtask 12
- **Estimated LOC:** ~80
- **Change:** New helper `async def _run_post_rewrite_validation(*, blocks_final_path, project_id, capture, **_)`: loads the rewrite-tier blocks JSONL, instantiates the four `Block*Validator` classes, runs each over the block list, persists `blocks_validated_path` (passing) and `blocks_failed_path` (failing) per the workflow phase outputs contract. Follows the same shape as the Phase 3 `_run_inter_tier_validation`. Decision-capture: emit one `block_validation_action` event per failed validator with `ml_features.tier="rewrite"`.
- **Verification:** `pytest MCP/tests/test_pipeline_tools.py -v -k "post_rewrite_validation" 2>&1 | head -10` reports new test PASSED.

### D. Touch chain extension to `outline_val` / `rewrite_val` (4 subtasks)

#### Subtask 14: Extend `Courseforge/scripts/blocks.py::_TIER_VALUES` with `outline_val` + `rewrite_val`
- **Files:** `/home/user/Ed4All/Courseforge/scripts/blocks.py` (search for `_TIER_VALUES` constant, near `:97-105`)
- **Depends on:** none
- **Estimated LOC:** ~5
- **Change:** Add `"outline_val"` and `"rewrite_val"` to the tier validation set. Update `Touch.__post_init__` validation pass-through.
- **Verification:** `python -c "from Courseforge.scripts.blocks import Touch; t=Touch(model='m',provider='deterministic',tier='rewrite_val',timestamp='2026-05-02T00:00:00Z',decision_capture_id='x',purpose='validation_pass'); assert t.tier=='rewrite_val'"` exits 0.

#### Subtask 15: Extend `schemas/context/courseforge_v1.shacl.ttl::TouchShape` enum
- **Files:** `/home/user/Ed4All/schemas/context/courseforge_v1.shacl.ttl` (search for `TouchShape` block; `tier` property has `sh:in (...)` enum)
- **Depends on:** Subtask 14
- **Estimated LOC:** ~3
- **Change:** Add `"outline_val"` and `"rewrite_val"` to the SHACL `sh:in` list on the `tier` property of `TouchShape`.
- **Verification:** `python -c "from rdflib import Graph; g=Graph(); g.parse('schemas/context/courseforge_v1.shacl.ttl', format='turtle'); s=g.serialize(format='turtle'); assert 'outline_val' in s and 'rewrite_val' in s"` exits 0.

#### Subtask 16: Extend `schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs.Touch.tier` enum
- **Files:** `/home/user/Ed4All/schemas/knowledge/courseforge_jsonld_v1.schema.json` (search for `"tier"` inside `$defs.Touch`)
- **Depends on:** Subtask 15
- **Estimated LOC:** ~3
- **Change:** Add `"outline_val"` and `"rewrite_val"` to the `$defs.Touch.properties.tier.enum` array.
- **Verification:** `python -c "import json; d=json.load(open('schemas/knowledge/courseforge_jsonld_v1.schema.json')); e=d['\$defs']['Touch']['properties']['tier']['enum']; assert 'outline_val' in e and 'rewrite_val' in e"` exits 0.

#### Subtask 17: Wire `outline_val` / `rewrite_val` Touch emission in router validators
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py` (in `_run_validator_chain` and post-rewrite-validation hook)
- **Depends on:** Subtask 14
- **Estimated LOC:** ~40
- **Change:** After each validator runs (passing OR failing), append a `Touch(tier="outline_val"|"rewrite_val", purpose="validation_pass"|"validation_fail", ...)` to the block via `block.with_touch(touch)`. The tier value is determined by the calling phase (router-side `_run_validator_chain` for outline-tier dispatch sets `tier="outline_val"`; the post-rewrite phase invokes a parallel `_run_post_rewrite_validator_chain` that sets `tier="rewrite_val"`).
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_validator_chain_appends_outline_val_touch -v` PASSES.

### E. Router-side remediation injection (5 subtasks)

#### Subtask 18: Wire `_append_remediation_for_gates` into `route_with_self_consistency`
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py:912-1003` (the candidate loop)
- **Depends on:** Subtasks 2, 3
- **Estimated LOC:** ~50
- **Change:** Inside the candidate loop after each failed `_run_validator_chain` call: build a remediated user-prompt suffix via `_append_remediation_for_gates(prompt="", failures=gate_results)`, store the suffix on a per-loop variable. The next `route(block, tier="outline", ...)` call needs to receive the remediation suffix; widen `route()` and the OutlineProvider's `generate_outline` to accept a `remediation_suffix: Optional[str] = None` kwarg appended to `_render_user_prompt`. The OutlineProvider's existing prompt construction (Phase 3 Subtask 17, `_render_user_prompt`) gets a final `if remediation_suffix: out += "\n\n" + remediation_suffix` line.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_self_consistency_injects_remediation_between_iterations -v` PASSES.

#### Subtask 19: Mirror remediation injection into rewrite-tier symmetric path
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py` (new method `route_rewrite_with_remediation`)
- **Depends on:** Subtask 18
- **Estimated LOC:** ~120
- **Change:** New method `route_rewrite_with_remediation(self, block, *, n_candidates=None, regen_budget=None, validators=None, source_chunks=None, objectives=None, **overrides) -> Block`. Mirrors `route_with_self_consistency` but dispatches `tier="rewrite"`. Resolves `regen_budget` from `_resolve_rewrite_regen_budget` (new helper paralleling `_resolve_regen_budget` but reading `_ENV_REWRITE_REGEN_BUDGET`). Inside the loop, after each failed `_run_validator_chain` (with the four `rewrite_*` gate validators), call `_append_remediation_for_gates` and pass the suffix to the rewrite provider. On budget exhaustion stamps `escalation_marker="validator_consensus_fail"` and breaks.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_route_rewrite_with_remediation_retries_with_feedback -v` PASSES.

#### Subtask 20: Bump `_DEFAULT_OUTLINE_REGEN_BUDGET` to 10 + add `_DEFAULT_REWRITE_REGEN_BUDGET = 10`
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py:103`
- **Depends on:** none
- **Estimated LOC:** ~10
- **Change:** Constant change. Add new constant `_DEFAULT_REWRITE_REGEN_BUDGET = 10` and matching `_ENV_REWRITE_REGEN_BUDGET = "COURSEFORGE_REWRITE_REGEN_BUDGET"`. Add `_resolve_rewrite_regen_budget(self, block, override) -> int` method paralleling `_resolve_regen_budget`.
- **Verification:** `python -c "from Courseforge.router.router import _DEFAULT_OUTLINE_REGEN_BUDGET, _DEFAULT_REWRITE_REGEN_BUDGET; assert _DEFAULT_OUTLINE_REGEN_BUDGET == 10 and _DEFAULT_REWRITE_REGEN_BUDGET == 10"` exits 0.

#### Subtask 21: Extend `block_routing.schema.json` with `regen_budget_rewrite` field
- **Files:** `/home/user/Ed4All/schemas/courseforge/block_routing.schema.json`
- **Depends on:** Subtask 20
- **Estimated LOC:** ~15
- **Change:** Add `regen_budget_rewrite: {type: "integer", minimum: 1}` to the per-block-type properties block (sibling to existing `regen_budget`). Update `Courseforge/router/policy.py::BlockRoutingPolicy` to expose `regen_budget_rewrite_by_block_type: Dict[str, int]`.
- **Verification:** `python -c "import json,jsonschema; s=json.load(open('schemas/courseforge/block_routing.schema.json')); jsonschema.Draft202012Validator.check_schema(s); assert 'regen_budget_rewrite' in str(s)"` exits 0.

#### Subtask 22: Add `Courseforge/router/tests/test_remediation_injection.py`
- **Files:** create `/home/user/Ed4All/Courseforge/router/tests/test_remediation_injection.py`
- **Depends on:** Subtasks 18, 19
- **Estimated LOC:** ~180
- **Change:** Tests: `test_outline_remediation_injects_curie_drop_directive`, `test_outline_remediation_injects_content_type_directive`, `test_outline_budget_10_iterations_then_escalation_marker`, `test_rewrite_remediation_injects_directive_per_failure`, `test_rewrite_budget_exhaustion_sets_validator_consensus_fail_marker`, `test_rewrite_remediation_per_block_type_override_via_yaml`, `test_outline_val_touch_appended_after_validation`, `test_rewrite_val_touch_appended_after_post_rewrite_validation`. Stub providers + validators returning canned outputs.
- **Verification:** `pytest Courseforge/router/tests/test_remediation_injection.py -v` reports ≥8 PASSED.

### F. Phase 3a env-var fixes (3 subtasks)

#### Subtask 23: Fix `Courseforge/router/policy.py::load_block_routing_policy` env-var-first model resolution
- **Files:** `/home/user/Ed4All/Courseforge/router/policy.py`
- **Depends on:** none
- **Estimated LOC:** ~30
- **Change:** When loading the YAML, after building the `defaults.outline.model` and `defaults.rewrite.model` values, check whether `os.environ.get("COURSEFORGE_OUTLINE_MODEL")` (resp. `COURSEFORGE_REWRITE_MODEL`) is set non-empty; if so, override the YAML's `defaults.outline.model` (resp. `rewrite`). Per-block-type overrides in the YAML still win over the env var (operator-explicit > tier-default). Emit one `decision_type="model_resolution_env_override"` audit event when an override fires.
- **Verification:** `python -c "import os; os.environ['COURSEFORGE_OUTLINE_MODEL']='qwen2.5:14b-instruct-q4_K_M'; from Courseforge.router.policy import load_block_routing_policy; p=load_block_routing_policy(); assert p.defaults['outline'].model=='qwen2.5:14b-instruct-q4_K_M'"` exits 0.

#### Subtask 24: Verify `Courseforge/generators/_rewrite_provider.py` env-var-first chain (audit-only)
- **Files:** `/home/user/Ed4All/Courseforge/generators/_rewrite_provider.py:96` (the `DEFAULT_MODEL = "claude-sonnet-4-6"` literal)
- **Depends on:** none
- **Estimated LOC:** ~20 (defensive — confirm `__init__` reads env-var BEFORE the hardcoded constant)
- **Change:** Verify `__init__` resolves model via `kwargs.get("model") or os.environ.get(ENV_MODEL) or DEFAULT_MODEL`. If currently `DEFAULT_MODEL or env`, swap to env-first. Add an inline comment marking this as the Phase 3a env-var-first contract. Audit `_provider.py` and `_outline_provider.py` for parallel env-first compliance and document any deviation.
- **Verification:** `python -c "import os; os.environ['COURSEFORGE_REWRITE_MODEL']='custom-model-x'; from Courseforge.generators._rewrite_provider import RewriteProvider; p=RewriteProvider(api_key='k'); assert p._model=='custom-model-x'"` exits 0.

#### Subtask 25: Audit `Courseforge/router/router.py::_resolve_spec` precedence chain
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py:328-395` (the resolution chain)
- **Depends on:** Subtask 24
- **Estimated LOC:** ~10 (audit + commentary)
- **Change:** Add an explicit comment block at `_resolve_spec` documenting the four-layer chain (per-call kwargs > YAML policy > env vars > hardcoded defaults). Add an audit test verifying that `COURSEFORGE_OUTLINE_MODEL=foo` overrides the hardcoded default but loses to a YAML entry.
- **Verification:** `pytest Courseforge/router/tests/test_router.py::test_phase3a_env_var_overrides_hardcoded_default -v` and `::test_phase3a_yaml_wins_over_env_var -v` both PASS.

### G. Decision-event extensions (2 subtasks)

#### Subtask 26: Extend `block_validation_action` event with `tier` field in `ml_features`
- **Files:** `/home/user/Ed4All/Courseforge/router/router.py` (the `_emit_block_validation_action` helper, Phase 3 Subtask 47)
- **Depends on:** Subtask 13
- **Estimated LOC:** ~10
- **Change:** Add `tier: Literal["outline", "rewrite"]` parameter to the helper signature; thread it into the `ml_features` payload. Update both call sites (the outline-tier `_run_validator_chain` call site sets `tier="outline"`; the post-rewrite-tier path sets `tier="rewrite"`).
- **Verification:** `pytest Courseforge/router/tests/test_validator_action.py::test_block_validation_action_event_includes_tier_field -v` PASSES.

#### Subtask 27: Add `block_validation_action::tier` regression test under strict-mode
- **Files:** create `/home/user/Ed4All/lib/tests/test_phase3_5_decision_event_tier_field.py`
- **Depends on:** Subtask 26
- **Estimated LOC:** ~50
- **Change:** Test that `DECISION_VALIDATION_STRICT=true` accepts `block_validation_action` events carrying `ml_features.tier="outline"` and `tier="rewrite"`.
- **Verification:** `DECISION_VALIDATION_STRICT=true pytest lib/tests/test_phase3_5_decision_event_tier_field.py -v` reports ≥2 PASSED.

### H. Documentation + smoke (3 subtasks)

#### Subtask 28: Update `Courseforge/CLAUDE.md` with Phase 3.5 section
- **Files:** `/home/user/Ed4All/Courseforge/CLAUDE.md`
- **Depends on:** Subtasks 13, 19
- **Estimated LOC:** ~80
- **Change:** New section `### Phase 3.5: symmetric validation + remediation`. Cross-link `Courseforge/router/remediation.py`, `Courseforge/router/inter_tier_gates.py` (shape-discriminating section), the `post_rewrite_validation` workflow phase, the `outline_val`/`rewrite_val` Touch tier values, and the bumped regen budgets (`_DEFAULT_*_REGEN_BUDGET = 10`).

#### Subtask 29: Add `COURSEFORGE_REWRITE_REGEN_BUDGET` row to root `CLAUDE.md` flag table
- **Files:** `/home/user/Ed4All/CLAUDE.md` (insert in the existing alphabetical `COURSEFORGE_*` block)
- **Depends on:** Subtask 20
- **Estimated LOC:** ~10
- **Change:** Single row matching the density of `COURSEFORGE_OUTLINE_REGEN_BUDGET` documenting the new env var + bumped default 10.

#### Subtask 30: End-to-end smoke command sequence
- **Files:** runbook
- **Depends on:** Subtasks 1-29
- **Verification:** See "Final smoke test" below.

---

## Execution sequencing

- Wave 3.5-N1 (Foundation): A (1-4) + B (5-9) + F (23-25). Parallelisable.
- Wave 3.5-N2 (Workflow + Touch + Router wiring): C (10-13) + D (14-17) + E (18-22).
- Wave 3.5-N3 (Events + Docs + Smoke): G (26-27) + H (28-30).

---

## Final smoke test

```bash
pytest Courseforge/router/tests/test_remediation.py \
       Courseforge/router/tests/test_inter_tier_gates_shape_dispatch.py \
       Courseforge/router/tests/test_remediation_injection.py \
       MCP/tests/test_pipeline_tools.py -k post_rewrite_validation \
       lib/tests/test_phase3_5_decision_event_tier_field.py -v

DECISION_VALIDATION_STRICT=true pytest \
  tests/integration/test_courseforge_two_pass_end_to_end.py -v

# Verify post_rewrite_validation phase fired:
ls Courseforge/exports/PROJ-DEMO_303-*/training-captures/courseforge/DEMO_303/ \
  | grep "phase_courseforge-post-rewrite-validation"

# Verify both outline_val + rewrite_val Touches present:
jq -r '.[] | .touched_by[].tier' \
  Courseforge/exports/PROJ-DEMO_303-*/03_content_development/blocks_validated.json \
  | sort -u    # expect: local, outline, outline_val, rewrite, rewrite_val
```

---

### Critical Files for Implementation
- `/home/user/Ed4All/Courseforge/router/remediation.py` (NEW)
- `/home/user/Ed4All/Courseforge/router/inter_tier_gates.py` (NEW or extended — shape-discriminating adapters)
- `/home/user/Ed4All/Courseforge/router/router.py` (extend with `route_rewrite_with_remediation` + bumped budgets + remediation injection)
- `/home/user/Ed4All/config/workflows.yaml` (insert `post_rewrite_validation` phase)
- `/home/user/Ed4All/Courseforge/scripts/blocks.py` + schemas (Touch tier expansion)
