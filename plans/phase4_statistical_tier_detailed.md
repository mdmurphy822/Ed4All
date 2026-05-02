# Phase 4 Detailed Execution Plan — Statistical Tier (BERT Ensemble + Embedding Validators + SHACL Wire-Up + Phase 3b Env-Var Fixes)

Refines `plans/phase4_statistical_tier.md` (high-level) and `plans/courseforge_architecture_roadmap.md` §3.2 into atomic subtasks. **Depends on:** Phase 3 (router seam + GateResult.action contract), Phase 3.5 (symmetric-validation surface + remediation builder).

---

## Investigation findings (locked)

- **`lib/embedding/` does not exist** — verified via `find /home/user/Ed4All/lib/embedding`. Phase 4 creates the package.
- **`Trainforge/eval/key_term_precision.py:66-71`** is the existing precedent for sentence-transformers loading. Mirrors lazy-import + try/except ImportError pattern.
- **SHACL runner exists at `/home/user/Ed4All/lib/validators/shacl_runner.py`** (576 LOC). `jsonld_payloads_to_graph` at `:207`, `run_shacl` at `:291`. Phase 4 reuses these.
- **`schemas/context/courseforge_v1.shacl.ttl`** carries 8 NodeShapes (verified at `:345-388`). Phase 4 wires `outline_shacl` to validate Block-derived JSON-LD against this file.
- **`lib/classifiers/` does not exist**. Phase 4 creates `lib/classifiers/bloom_bert_ensemble.py`.
- **`Trainforge/align_chunks.py:621` and `:1243`** carry hardcoded `claude-haiku-4-5-20251001`. **`Trainforge/process_course.py:4982`** carries `target_models = ["claude-opus-4-6", "claude-sonnet-4-6"]`. **`Trainforge/process_course.py:5296`** carries `llm_model="claude-haiku-4-5-20251001"`. These are Phase 3b env-var fixes.
- **GateResult.action contract** is in tree per Phase 3 Subtask 46 — verified at `MCP/hardening/validation_gates.py:48-78`. Phase 4 validators emit `action="regenerate"` for soft semantic faults.
- **Decision-event enum** at `schemas/events/decision_event.schema.json` already accepts the Phase-3 enum values; Phase 4 adds 4 new values: `statistical_validation_pass`, `statistical_validation_fail`, `bert_ensemble_disagreement`, `bert_ensemble_dispersion_high`.
- **`MCP/orchestrator/llm_backend.py`** has `DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"` at `:48`. Phase 3c — bundled with Phase 6, NOT Phase 4 — handles this.
- **Round-trip validator (Phase 4 §5)** dispatches one LLM call per LO via the rewrite-tier router. The router-resolved model is the same as the rewrite tier, avoiding cross-model drift.
- **`pyproject.toml`** at `/home/user/Ed4All/pyproject.toml` has `[project.optional-dependencies]` extras: `dart`, `server`, `dev`. Phase 4 adds `embedding` extra carrying `sentence-transformers>=2.5.0,<4.0.0`, `transformers>=4.49,<4.50`, `numpy`. The `training` extra remains separate (Trainforge-side).

---

## Pre-resolved decisions

1. **BERT ensemble member selection v1 (per roadmap §6.5).** 1 domain-tuned + 2 distilbert variants:
   - `kabir5297/bloom_taxonomy_classifier` (domain-tuned)
   - `distilbert-base-uncased-finetuned-sst-2-english` (general semantic — repurposed via final-layer prompt)
   - `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (NLI-style — emits Bloom via paraphrase entailment)
   - SHA-pin all three by recording `revision: <git_sha>` in `lib/classifiers/bloom_bert_ensemble.py::_DEFAULT_ENSEMBLE_MEMBERS`.
2. **k-reranker aggregation.** Confidence-weighted majority vote with dispersion penalty:
   - For each block, each member emits `(level, confidence)` pair.
   - Aggregate: `score[level] = sum(confidence for member where member.level == level)`.
   - Winner = `argmax(score)`.
   - Dispersion: `entropy(normalized_scores)`. When `entropy > _DISPERSION_THRESHOLD` (default 0.7), emit `decision_type="bert_ensemble_dispersion_high"` even when winner is unanimous.
3. **No few-shot LLM-as-judge.** Per roadmap §3.2 explicit decision.
4. **Embedding model.** `all-MiniLM-L6-v2` (matches existing `Trainforge/eval/key_term_precision.py:69`). 90 MB, ~5 ms/sentence on CPU. Optional override via `EMBEDDING_MODEL_NAME` env var.
5. **Embedding validator threshold defaults (placeholders, calibrated in Subtask 30).**
   - `objective_assessment_similarity.min_cosine = 0.55`
   - `concept_example_similarity.min_cosine = 0.50`
   - `objective_roundtrip_similarity.min_cosine = 0.70`
6. **Outline-SHACL severity** initially warning; promotion to critical after Wave N+1 calibration.
7. **Symmetric-validator surface.** Each Phase 4 gate fires on BOTH `inter_tier_validation` AND `post_rewrite_validation` workflow phases per Phase 3.5's symmetric contract. Each validator's `validate(inputs)` accepts both outline-tier dicts and rewrite-tier HTML strings via the same shape-discrimination helpers Phase 3.5 added to `inter_tier_gates.py`.
8. **Decision-event extension.** 4 new `decision_type` enum values: `statistical_validation_pass`, `statistical_validation_fail`, `bert_ensemble_disagreement`, `bert_ensemble_dispersion_high`. Plus 2 `phase` enum values: `courseforge-statistical-validation`, `courseforge-bert-ensemble`.
9. **Calibration script location.** `scripts/calibrate_phase4_thresholds.py` reads a holdout corpus from `LibV2/courses/<slug>/eval/phase4_holdout.jsonl`, runs the four gates, computes precision/recall at threshold sweeps, persists `calibrated_thresholds.yaml`.

---

## Atomic subtasks

Estimated total LOC: ~3,500 (350 embedding package + 600 BERT ensemble + 400 SHACL adapter + 800 4 embedding validators + 250 calibration script + 250 workflow integration + 250 phase 3b env fix + 600 tests + 150 docs).

### A. Embedding infrastructure (5 subtasks)

#### Subtask 1: Create `lib/embedding/` package skeleton
- **Files:** create `/home/user/Ed4All/lib/embedding/__init__.py`, `/home/user/Ed4All/lib/embedding/sentence_embedder.py`, `/home/user/Ed4All/lib/embedding/_math.py`
- **Depends on:** none
- **Estimated LOC:** ~150
- **Change:** `_math.py` exposes `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float` (port of `Trainforge/eval/key_term_precision.py:74-80`). `sentence_embedder.py` exposes `class SentenceEmbedder` with lazy-loaded model (`_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"`); `try_load_embedder() -> Optional[SentenceEmbedder]` returns None when extras missing.
- **Verification:** `python -c "from lib.embedding.sentence_embedder import SentenceEmbedder, try_load_embedder; assert SentenceEmbedder is not None"` exits 0.

#### Subtask 2: Implement `EmbeddingCache` LRU
- **Files:** `/home/user/Ed4All/lib/embedding/sentence_embedder.py`
- **Depends on:** Subtask 1
- **Estimated LOC:** ~80
- **Change:** `class EmbeddingCache` keyed on `sha256(text)`. Persists to `state/embedding_cache.jsonl`, one row per `{hash, vector}`. Loaded once per run; appended on miss. LRU bound `_MAX_CACHE_ENTRIES = 100_000`.
- **Verification:** `pytest lib/embedding/tests/test_sentence_embedder.py::test_cache_persists_across_runs -v` PASSES.

#### Subtask 3: Add `embedding` extras to `pyproject.toml`
- **Files:** `/home/user/Ed4All/pyproject.toml:[project.optional-dependencies]`
- **Depends on:** Subtask 1
- **Estimated LOC:** ~10
- **Change:** Add `embedding = ["sentence-transformers>=2.5.0,<4.0.0", "numpy>=1.24.0", "transformers>=4.49,<4.50", "torch>=2.0.0"]`.
- **Verification:** `pip install -e .[embedding] --dry-run 2>&1 | head -5` reports 4 packages would install.

#### Subtask 4: Add fallback policy on missing extras
- **Files:** `/home/user/Ed4All/lib/embedding/sentence_embedder.py`
- **Depends on:** Subtasks 1, 2
- **Estimated LOC:** ~40
- **Change:** When `sentence-transformers` is not importable, `try_load_embedder()` returns None; validators that use it emit a `severity="warning"` GateIssue `code="EMBEDDING_DEPS_MISSING"` and `passed=True`. Strict-mode opt-in via `TRAINFORGE_REQUIRE_EMBEDDINGS=true` flips to critical. Mirrors `lib/validators/shacl_runner.py:557-576`.
- **Verification:** `python -c "from lib.embedding.sentence_embedder import try_load_embedder; e=try_load_embedder(); assert e is None or hasattr(e, 'encode')"` exits 0.

#### Subtask 5: Add `lib/embedding/tests/test_sentence_embedder.py`
- **Files:** create `/home/user/Ed4All/lib/embedding/tests/test_sentence_embedder.py`
- **Depends on:** Subtasks 1-4
- **Estimated LOC:** ~120
- **Change:** Tests: `test_encode_returns_unit_vectors`, `test_cosine_similarity_perfect_match_is_1`, `test_cache_hit_returns_cached_vector`, `test_cache_persists_across_runs`, `test_try_load_embedder_returns_none_when_extras_missing` (mocked import), `test_strict_mode_raises_when_extras_missing`. Uses temp_path fixture for cache file.
- **Verification:** `pytest lib/embedding/tests/test_sentence_embedder.py -v` reports ≥6 PASSED (skip when extras missing).

### B. SHACL outline validator wire-up (4 subtasks)

#### Subtask 6: Create `lib/validators/courseforge_outline_shacl.py`
- **Files:** create or refresh `/home/user/Ed4All/lib/validators/courseforge_outline_shacl.py`
- **Depends on:** none
- **Estimated LOC:** ~180
- **Change:** Class `CourseforgeOutlineShaclValidator` implementing `validate(inputs: Dict[str, Any]) -> GateResult`. Inputs: `blocks_path` (JSONL of Block-derived JSON-LD payloads) or `blocks: List[dict]` directly. Builds RDF graph via `lib.validators.shacl_runner.jsonld_payloads_to_graph`. Calls `lib.validators.shacl_runner.run_shacl(SHAPES_PATH, graph)` against `schemas/context/courseforge_v1.shacl.ttl`. Projects violations to `GateIssue` via `ShaclViolation.to_gate_issue()`. Returns `GateResult(action="regenerate")` on warning-severity violations and `action="block"` on critical.
- **Verification:** `python -c "from lib.validators.courseforge_outline_shacl import CourseforgeOutlineShaclValidator; v=CourseforgeOutlineShaclValidator(); assert hasattr(v, 'validate')"` exits 0.

#### Subtask 7: Wire `outline_shacl` gate into `inter_tier_validation` workflow phase
- **Files:** `/home/user/Ed4All/config/workflows.yaml:790-859` (the `inter_tier_validation` phase)
- **Depends on:** Subtask 6
- **Estimated LOC:** ~15
- **Change:** Append a 5th gate to `inter_tier_validation::validation_gates`:
  ```yaml
  - gate_id: outline_shacl
    validator: lib.validators.courseforge_outline_shacl.CourseforgeOutlineShaclValidator
    severity: warning
    threshold:
      max_critical_issues: 0
    behavior: {on_fail: warn, on_error: warn}
  ```
- **Verification:** `python -c "import yaml; d=yaml.safe_load(open('config/workflows.yaml')); ph=next(p for p in d['workflows'][next((i for i,w in enumerate(d['workflows']) if w['name']=='textbook_to_course'))]['phases'] if p['name']=='inter_tier_validation'); gates=[g['gate_id'] for g in ph['validation_gates']]; assert 'outline_shacl' in gates"` exits 0.

#### Subtask 8: Mirror `outline_shacl` into `post_rewrite_validation` phase (symmetric)
- **Files:** `/home/user/Ed4All/config/workflows.yaml` (the `post_rewrite_validation` phase Phase 3.5 added)
- **Depends on:** Subtask 7
- **Estimated LOC:** ~15
- **Change:** Mirror gate as `rewrite_shacl` (gate_id distinct for decision-event filtering) with the same validator + severity warning. The validator's shape-discrimination handles the rewrite-tier HTML input via `_extract_block_jsonld_from_html` (per Subtask 13 of Phase 3.5's plan).

#### Subtask 9: Add `lib/validators/tests/test_courseforge_outline_shacl.py`
- **Files:** create `/home/user/Ed4All/lib/validators/tests/test_courseforge_outline_shacl.py`
- **Depends on:** Subtasks 6, 7
- **Estimated LOC:** ~150
- **Change:** Tests: `test_passes_well_formed_outline_blocks`, `test_critical_violation_returns_action_block`, `test_warning_violation_returns_action_regenerate`, `test_handles_str_content_via_html_extraction`, `test_handles_dict_content_directly`, `test_no_violations_returns_pass_action`. Uses fixture blocks built per `Block.to_jsonld_entry`.
- **Verification:** `pytest lib/validators/tests/test_courseforge_outline_shacl.py -v` reports ≥6 PASSED.

### C. Three embedding validators (10 subtasks — 3 validators × 3 subtasks + 1 shared)

#### Subtask 10: Create `lib/validators/objective_assessment_similarity.py`
- **Files:** create `/home/user/Ed4All/lib/validators/objective_assessment_similarity.py`
- **Depends on:** Subtask 4
- **Estimated LOC:** ~150
- **Change:** Class `ObjectiveAssessmentSimilarityValidator` with `validate(inputs)`. Reads `inputs["blocks"]`. For each `assessment_item` block, embed `block.content["stem"] + " " + block.content["answer_key"]`; for each declared `objective_ref`, embed the objective statement; compute cosine similarity. Emits `action="regenerate"` when `min(per-pair cosines) < threshold`. Skips when extras missing per Subtask 4.
- **Verification:** `pytest lib/validators/tests/test_objective_assessment_similarity.py -v` reports ≥4 PASSED.

#### Subtask 11: Create `lib/validators/concept_example_similarity.py`
- **Files:** create `/home/user/Ed4All/lib/validators/concept_example_similarity.py`
- **Depends on:** Subtask 4
- **Estimated LOC:** ~150
- **Change:** Class `ConceptExampleSimilarityValidator`. For each `example` block, embed `block.content["body"]`; for each `concept_ref` (the concept this example illustrates), embed the concept slug + definition; compute cosine. Threshold default 0.50 (examples are deliberately diverse phrasings).
- **Verification:** `pytest lib/validators/tests/test_concept_example_similarity.py -v` reports ≥4 PASSED.

#### Subtask 12: Create `lib/validators/objective_roundtrip_similarity.py`
- **Files:** create `/home/user/Ed4All/lib/validators/objective_roundtrip_similarity.py`
- **Depends on:** Subtask 4
- **Estimated LOC:** ~200
- **Change:** Class `ObjectiveRoundtripSimilarityValidator`. For each `objective` block: dispatch a paraphrase request via the rewrite-tier router (`router.route(block, tier="rewrite", overrides={"prompt_template": "Paraphrase preserving meaning"})`); embed both original and paraphrase; cosine. Threshold 0.70 (paraphrase of identical content should be tight). Skips on rewrite dispatch failure (warning).
- **Verification:** `pytest lib/validators/tests/test_objective_roundtrip_similarity.py -v` reports ≥4 PASSED.

#### Subtask 13: Wire 3 embedding gates into `inter_tier_validation` phase
- **Files:** `/home/user/Ed4All/config/workflows.yaml`
- **Depends on:** Subtasks 10-12
- **Estimated LOC:** ~50
- **Change:** Add 3 gates with severity warning; behavior `{on_fail: warn, on_error: warn}`.
- **Verification:** `python -c "import yaml; d=yaml.safe_load(open('config/workflows.yaml')); ph=next(p for p in d['workflows'][next(i for i,w in enumerate(d['workflows']) if w['name']=='textbook_to_course')]['phases'] if p['name']=='inter_tier_validation'); ids=[g['gate_id'] for g in ph['validation_gates']]; assert 'objective_assessment_similarity' in ids and 'concept_example_similarity' in ids and 'objective_roundtrip_similarity' in ids"` exits 0.

#### Subtask 14: Mirror 3 embedding gates into `post_rewrite_validation` phase
- **Files:** `/home/user/Ed4All/config/workflows.yaml`
- **Depends on:** Subtask 13
- **Estimated LOC:** ~50
- **Change:** Sibling gates on the symmetric phase.

#### Subtask 15-19: Per-validator test files and integration tests (~5 subtasks, ~100 LOC each)
- Each validator tested in isolation; one integration test asserting end-to-end firing through the workflow phase.

### D. BERT ensemble + k-reranker (8 subtasks)

#### Subtask 20: Create `lib/classifiers/` package + `bloom_bert_ensemble.py` skeleton
- **Files:** create `/home/user/Ed4All/lib/classifiers/__init__.py`, `/home/user/Ed4All/lib/classifiers/bloom_bert_ensemble.py`
- **Depends on:** Subtask 3
- **Estimated LOC:** ~120
- **Change:** Module-level `_DEFAULT_ENSEMBLE_MEMBERS: List[Dict[str, str]]` carrying name + revision SHA per pre-resolved decision #1 (3 members). Class `BloomBertEnsemble` with `__init__(self, members=None)`, `classify(self, text: str) -> Dict[str, Any]` returning `{"winner_level": str, "winner_score": float, "dispersion": float, "per_member": List[(level, confidence)]}`.
- **Verification:** `python -c "from lib.classifiers.bloom_bert_ensemble import BloomBertEnsemble, _DEFAULT_ENSEMBLE_MEMBERS; assert len(_DEFAULT_ENSEMBLE_MEMBERS) == 3 and all('revision' in m for m in _DEFAULT_ENSEMBLE_MEMBERS)"` exits 0.

#### Subtask 21: Implement member loading + lazy-instantiation
- **Files:** `/home/user/Ed4All/lib/classifiers/bloom_bert_ensemble.py`
- **Depends on:** Subtask 20
- **Estimated LOC:** ~100
- **Change:** `_load_members(self) -> List[BertClassifier]`. Lazy-import `transformers.AutoModelForSequenceClassification` and `AutoTokenizer`. SHA-pin via `revision=member["revision"]`. Cache in `~/.cache/ed4all/bert_ensemble/`. Emits `decision_type="bert_ensemble_member_loaded"` per member.
- **Verification:** `python -c "from lib.classifiers.bloom_bert_ensemble import BloomBertEnsemble; e=BloomBertEnsemble(); assert hasattr(e, '_load_members')"` exits 0.

#### Subtask 22: Implement k-reranker confidence-weighted majority + dispersion penalty
- **Files:** `/home/user/Ed4All/lib/classifiers/bloom_bert_ensemble.py`
- **Depends on:** Subtask 21
- **Estimated LOC:** ~80
- **Change:** `_aggregate(self, per_member: List[Tuple[str, float]]) -> Tuple[str, float, float]`. Score per level = sum of confidence votes. Winner = argmax. Dispersion = `entropy(normalized_scores)` (scipy.stats.entropy or hand-rolled).
- **Verification:** `python -c "from lib.classifiers.bloom_bert_ensemble import BloomBertEnsemble; e=BloomBertEnsemble(); w,s,d=e._aggregate([('apply',0.9),('apply',0.85),('analyze',0.6)]); assert w=='apply' and s>0.5"` exits 0.

#### Subtask 23: Create `lib/validators/bloom_classifier_disagreement.py`
- **Files:** create `/home/user/Ed4All/lib/validators/bloom_classifier_disagreement.py`
- **Depends on:** Subtask 22
- **Estimated LOC:** ~150
- **Change:** Class `BloomClassifierDisagreementValidator`. For each `objective` or `assessment_item` block, classify the block content; compare ensemble winner vs `block.bloom_level`. Emit `action="regenerate"` on mismatch (`bert_ensemble_disagreement` event) OR when `dispersion > 0.7` (`bert_ensemble_dispersion_high` event).
- **Verification:** `python -c "from lib.validators.bloom_classifier_disagreement import BloomClassifierDisagreementValidator; v=BloomClassifierDisagreementValidator(); assert hasattr(v, 'validate')"` exits 0.

#### Subtask 24: Wire `bloom_classifier_disagreement` gate into both validation phases
- **Files:** `/home/user/Ed4All/config/workflows.yaml`
- **Depends on:** Subtask 23
- **Estimated LOC:** ~30
- **Change:** Add gate to both `inter_tier_validation` and `post_rewrite_validation` phases.

#### Subtask 25: Author `lib/classifiers/tests/test_bloom_bert_ensemble.py`
- **Files:** create
- **Depends on:** Subtask 22
- **Estimated LOC:** ~150
- **Change:** Tests: `test_unanimous_high_confidence_returns_winner`, `test_split_vote_resolves_via_confidence_weighting`, `test_dispersion_high_when_split_vote_3_distinct_levels`, `test_member_failure_falls_through_silently_with_warning`, `test_sha_pinning_recorded_in_decision_event`. Uses HuggingFace test stubs.
- **Verification:** `pytest lib/classifiers/tests/test_bloom_bert_ensemble.py -v` reports ≥5 PASSED.

#### Subtask 26: Add 4 decision_event enum values
- **Files:** `/home/user/Ed4All/schemas/events/decision_event.schema.json`
- **Depends on:** Subtask 23
- **Estimated LOC:** ~6
- **Change:** Insert alphabetically: `bert_ensemble_disagreement`, `bert_ensemble_dispersion_high`, `statistical_validation_pass`, `statistical_validation_fail`. Plus `phase` enum: `courseforge-statistical-validation`, `courseforge-bert-ensemble`.
- **Verification:** `python -c "import json; d=json.load(open('schemas/events/decision_event.schema.json')); e=d['properties']['decision_type']['enum']; assert all(v in e for v in ['bert_ensemble_disagreement','bert_ensemble_dispersion_high','statistical_validation_pass','statistical_validation_fail'])"` exits 0.

#### Subtask 27: VRAM/CPU budget regression test
- **Files:** create `/home/user/Ed4All/lib/classifiers/tests/test_ensemble_resource_budget.py`
- **Depends on:** Subtask 21
- **Estimated LOC:** ~80
- **Change:** Test that classify-50-blocks with the 3-member ensemble completes in <5s on CPU (proxy for <50ms/block target). Skip when extras missing.
- **Verification:** `pytest lib/classifiers/tests/test_ensemble_resource_budget.py -v` PASSES.

### E. Threshold calibration script (3 subtasks)

#### Subtask 28: Create `scripts/calibrate_phase4_thresholds.py`
- **Files:** create `/home/user/Ed4All/scripts/calibrate_phase4_thresholds.py`
- **Depends on:** Subtasks 10-12, 23
- **Estimated LOC:** ~250
- **Change:** CLI accepting `--course-slug`, `--gate {objective_assessment,concept_example,objective_roundtrip,bert_ensemble}`, `--sweep-from`, `--sweep-to`, `--steps`. Reads holdout corpus from `LibV2/courses/<slug>/eval/phase4_holdout.jsonl`; runs each gate at each threshold; computes precision / recall / F1 per threshold; writes `LibV2/courses/<slug>/eval/calibrated_thresholds.yaml`.
- **Verification:** `python scripts/calibrate_phase4_thresholds.py --help` exits 0.

#### Subtask 29: Add temperature scaling for BERT ensemble
- **Files:** `/home/user/Ed4All/scripts/calibrate_phase4_thresholds.py`, `/home/user/Ed4All/lib/classifiers/bloom_bert_ensemble.py`
- **Depends on:** Subtask 28
- **Estimated LOC:** ~80
- **Change:** Add temperature parameter `T` to softmax in `BloomBertEnsemble._aggregate`. Calibration script tunes T per member to minimize ECE on holdout. Persists to `calibrated_thresholds.yaml::ensemble_temperatures`.
- **Verification:** `pytest lib/classifiers/tests/test_bloom_bert_ensemble.py::test_temperature_scaling_reduces_ece -v` PASSES.

#### Subtask 30: Add dispersion threshold tuning
- **Files:** `/home/user/Ed4All/scripts/calibrate_phase4_thresholds.py`
- **Depends on:** Subtask 29
- **Estimated LOC:** ~50
- **Change:** Sweep dispersion threshold from 0.3-1.0 against holdout disagreement labels; pick threshold maximizing F1.

### F. Phase 3b env-var fixes (3 subtasks)

#### Subtask 31: Fix `Trainforge/align_chunks.py` hardcoded model
- **Files:** `/home/user/Ed4All/Trainforge/align_chunks.py:621,1243`
- **Depends on:** none
- **Estimated LOC:** ~25
- **Change:** Replace `llm_model: str = "claude-haiku-4-5-20251001"` with `llm_model: str = None` and resolve to `os.environ.get("TRAINFORGE_ALIGN_CHUNKS_MODEL", "claude-haiku-4-5-20251001")` inside the function. Update CLI arg default similarly. Add `TRAINFORGE_ALIGN_CHUNKS_MODEL` env var doc to `Trainforge/CLAUDE.md` and root `CLAUDE.md`.
- **Verification:** `python -c "import os; os.environ['TRAINFORGE_ALIGN_CHUNKS_MODEL']='custom'; from Trainforge.align_chunks import _resolve_align_model; assert _resolve_align_model()=='custom'"` exits 0 (after helper extraction).

#### Subtask 32: Fix `Trainforge/process_course.py` hardcoded `target_models`
- **Files:** `/home/user/Ed4All/Trainforge/process_course.py:4982,5296`
- **Depends on:** Subtask 31
- **Estimated LOC:** ~25
- **Change:** Replace `"target_models": ["claude-opus-4-6", "claude-sonnet-4-6"]` with `_resolve_target_models()` reading `TRAINFORGE_TARGET_MODELS` (CSV). Replace `:5296` `llm_model="claude-haiku-4-5-20251001"` with env-var resolution.
- **Verification:** env-var override test PASSES.

#### Subtask 33: Add 2 env-var rows to root `CLAUDE.md` flag table
- **Files:** `/home/user/Ed4All/CLAUDE.md`
- **Depends on:** Subtasks 31, 32
- **Estimated LOC:** ~10

### G. Documentation + smoke (3 subtasks)

#### Subtask 34: Update `Courseforge/CLAUDE.md` and `Trainforge/CLAUDE.md` with Phase 4 sections
#### Subtask 35: Add embedding/BERT extras + ensemble member rationale to docs
#### Subtask 36: End-to-end smoke command sequence

---

## Execution sequencing

- 4-N1: A (1-5) + F (31-33) parallelisable; B (6-9) sequentially after A.
- 4-N2: C (10-19), D (20-27) parallelisable.
- 4-N3: E (28-30) → G (34-36).

---

## Final smoke test

```bash
pytest lib/embedding/tests/ \
       lib/validators/tests/test_courseforge_outline_shacl.py \
       lib/validators/tests/test_objective_assessment_similarity.py \
       lib/validators/tests/test_concept_example_similarity.py \
       lib/validators/tests/test_objective_roundtrip_similarity.py \
       lib/classifiers/tests/test_bloom_bert_ensemble.py -v

DECISION_VALIDATION_STRICT=true pytest tests/integration/ -k phase4 -v

# Calibrate against the rdf-shacl-551-2 holdout:
python scripts/calibrate_phase4_thresholds.py --course-slug rdf-shacl-551-2 \
  --gate objective_assessment --sweep-from 0.3 --sweep-to 0.8 --steps 11

# Verify ensemble dispersion event fires on a misbehaving fixture:
jq -r 'select(.decision_type=="bert_ensemble_dispersion_high") | .ml_features.dispersion' \
  training-captures/courseforge/DEMO_303/phase_courseforge-bert-ensemble/decisions_*.jsonl
```

---

### Critical Files for Implementation
- `/home/user/Ed4All/lib/embedding/sentence_embedder.py` (NEW)
- `/home/user/Ed4All/lib/classifiers/bloom_bert_ensemble.py` (NEW)
- `/home/user/Ed4All/lib/validators/courseforge_outline_shacl.py` (NEW)
- `/home/user/Ed4All/lib/validators/objective_assessment_similarity.py` (NEW)
- `/home/user/Ed4All/lib/validators/objective_roundtrip_similarity.py` (NEW)
- `/home/user/Ed4All/scripts/calibrate_phase4_thresholds.py` (NEW)
- `/home/user/Ed4All/config/workflows.yaml` (wire 5 gates × 2 phases)
