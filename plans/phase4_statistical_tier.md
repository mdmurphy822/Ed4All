# Phase 4 Plan — Statistical Tier Validators + SHACL Wire-Up

**Status:** plan only. **Depends on:** Phase 2 (Block dataclass + intermediate format), Phase 3 (router + outline/rewrite tiers + inter-tier gate seam).

## 1. Validator Inventory — Net-New vs. Upgrade

| Validator (gate_id) | Status | Phase 4 Scope |
|---|---|---|
| `outline_shacl` | wire-up | Run existing `schemas/context/courseforge_v1.shacl.ttl` (8 NodeShapes) against outline-tier Block JSON-LD via `lib/validators/shacl_runner.py:291`. |
| `objective_assessment_similarity` | new | Cosine sim ≥ threshold between every assessment-Block stem+key and each declared `objective_ids[]`. |
| `concept_example_similarity` | new | Cosine sim ≥ threshold between every example-Block body and the concept it tags. |
| `bloom_classifier_disagreement` | **DEFER** | Recommended deferred; document seam (see §3). |
| `objective_roundtrip_similarity` | new | LLM paraphrases the objective; cosine sim with the original ≥ threshold. Detects unstable / vacuous objective wording. |
| `assessment_objective_alignment` | parallel sibling, NOT swap | The existing validator at `lib/validators/assessment_objective_alignment.py:38` is a **structural ID-match** check (Wave 24), not Jaccard semantic similarity. Phase 4 lands `objective_assessment_similarity` as a *parallel semantic gate*; keep `assessment_objective_alignment` as the structural floor. The "Jaccard fallback" in the audit-finding refers to fallback inside the new semantic validator, not to this one. |

This contradicts the "swap Jaccard for embedding" framing in the original brief — see Open Question #1.

## 2. Embedding Infrastructure

**Module location.** New `lib/embedding/` package with `lib/embedding/sentence_embedder.py`. Avoids stuffing torch into `lib/validators/`. Mirrors existing eval-tier pattern at `Trainforge/eval/key_term_precision.py:66-71`.

**Surface (single source of truth):**
- `SentenceEmbedder.encode(texts: list[str]) -> np.ndarray` — lazy-loads `sentence-transformers`'s `all-MiniLM-L6-v2` on first call; caches the model on the instance.
- `cosine_similarity(a, b) -> float` — pure-Python cosine identical to `Trainforge/eval/key_term_precision.py:74-80`. Promote to a shared helper at `lib/embedding/_math.py`.
- `EmbeddingCache` — LRU keyed on `sha256(text)`; persists per-run to `state/embedding_cache.jsonl` (one JSONL row = `{hash, vector}`). Enables resume across `content_generation` retries.
- `try_load_embedder() -> SentenceEmbedder | None` — returns None when extras missing.

**Model choice.** Stay with `all-MiniLM-L6-v2`. Reasons: (1) already used at `Trainforge/eval/key_term_precision.py:69`; (2) 90 MB — fits CI; (3) inference ~5 ms/sentence on CPU — acceptable per §8 budget. Larger (mpnet-base, bge-small) is open-question candidate later; not blocking Phase 4.

**Fallback policy.** When `sentence-transformers` is not importable:
- Default behavior: emit a single `severity="warning"` GateIssue `code="EMBEDDING_DEPS_MISSING"` and `passed=True`. Mirrors the SHACL-deps-missing pattern at `lib/validators/shacl_runner.py:557-576`. Rationale: Phase 4 ships at warning-severity overall; CI without extras must not block.
- Strict-mode opt-in via env var `TRAINFORGE_REQUIRE_EMBEDDINGS=true` flips to a `passed=False` critical issue. Useful for production runs where ops want to refuse promotion of an unembedded course.
- **No Jaccard fallback as the default scoring path inside the new validators.** Jaccard correlates poorly with semantic similarity at the thresholds we'll be tuning to (~0.55–0.75) and would silently dilute the gate. Only `Trainforge/eval/key_term_precision.py:245` retains Jaccard as a quality-degraded fallback because it predates Phase 4.

**Initial threshold guidance** (warning-severity at first; numbers below are starting points, not load-bearing):
- `objective_assessment_similarity.min_cosine`: **0.55**.
- `concept_example_similarity.min_cosine`: **0.50** (examples are deliberately diverse phrasings).
- `objective_roundtrip_similarity.min_cosine`: **0.70** (paraphrase of identical content should be tight).

These thresholds are *placeholders* until §6 calibration runs.

## 3. DistilBERT Bloom Classifier — Recommendation: **DEFER**

Three reasons:
1. **No off-the-shelf, license-clean Bloom classifier** dominates HF Hub. Some education-flavored sentence-classifier checkpoints exist but none have provenance + eval comparable to the `lib/ontology/bloom.py:232` regex+lookup we already trust as a baseline.
2. Fine-tuning our own is multi-week; out of scope for Phase 4.
3. The verb-lookup classifier at `lib/ontology/bloom.py:232-257` is high-precision/low-recall (deterministic, traceable). It's a *floor*, not a ceiling. Phase 4 should land embedding-similarity gates (which are *orthogonal*, not redundant with Bloom) and revisit Bloom-classifier-disagreement once we have ground-truth disagreement-rate data from production.

**Plug-in seam.** Reserve `bloom_classifier_disagreement` as the future gate id. Add `lib/ontology/bloom.py::detect_bloom_level_with_classifier(text, classifier=None)` signature pre-emptively in Phase 4 (no body change yet) so the future plug can land additive. Document as deferred in `CLAUDE.md::Active Gates` with a "Phase 5 candidate" footnote.

## 4. SHACL Wire-Up

**Where it runs.** Insert as an `outline_shacl` gate at the end of the Phase-3 router's outline tier, *before* the rewrite tier dispatches. Per the Phase 3 plan's "inter-tier gate seam," this is the natural spot: outline tier produces the Block list, gate validates it, rewrite tier proceeds only if SHACL conformant.

**Adapter (Block list → JSON-LD → RDF):**
- New `lib/validators/courseforge_outline_shacl.py::CourseforgeOutlineShaclValidator`.
- Inputs: `blocks_path` (JSONL of Block dataclass-derived JSON-LD payloads — one Block per line, contract from Phase 2). Optionally accept `blocks: list[dict]` directly for in-memory paths.
- Re-uses `lib/validators/shacl_runner.py:207::jsonld_payloads_to_graph` to materialize the RDF graph (handles `@context` injection identically to the existing `PageObjectivesShaclValidator`).
- Calls `run_shacl(SHAPES_DIR.parents[2] / "schemas/context/courseforge_v1.shacl.ttl", graph)` at `lib/validators/shacl_runner.py:291`. **Note:** the existing runner's `SHAPES_DIR` lookup is keyed on `lib/validators/shacl/{gate_id}.ttl`; for the courseforge top-level shape file we pass the absolute path explicitly (the runner accepts it — `lib/validators/shacl_runner.py:323`).
- Severity routing: `sh:Violation→critical`, `sh:Warning→warning`, `sh:Info→info` (already implemented at `lib/validators/shacl_runner.py:102`). Project violations to `GateIssue` via the existing `ShaclViolation.to_gate_issue()` at `lib/validators/shacl_runner.py:152`.

**Gate severity.** **Warning** for the first 1-2 wave windows. The 8 NodeShapes (`CourseModuleShape`, `LearningObjectiveShape`, `SectionShape`, `BloomDistributionShape`, `MisconceptionShape`, `TargetedConceptShape`, `ChunkShape`, `TypedEdgeShape`) have never been run against Block-derived JSON-LD; expect false-positives during initial calibration. Promote to critical once threshold drift settles. Mirrors the `page_objectives_shacl` pattern at `config/workflows.yaml:649-657`.

**Decision-capture event.** New `decision_type="statistical_validation_pass"` and `"statistical_validation_fail"` enum values added to `schemas/events/decision_event.schema.json:63`. SHACL violations also emit a `decision_type="shacl_outline_violation"` (separate enum value) carrying the `(focus_node, source_shape, severity, message)` tuple in the rationale string — operators auditing a downstream rewrite-tier divergence can replay outline-tier shape violations.

**workflows.yaml diff sketch:**
```yaml
# textbook_to_course::content_generation, after content_grounding gate (~line 678):
- gate_id: outline_shacl
  validator: lib.validators.courseforge_outline_shacl.CourseforgeOutlineShaclValidator
  severity: warning   # promote to critical after Wave N+2 calibration
  threshold:
    max_critical_issues: 0
  behavior:
    on_fail: warn
    on_error: warn
  description: Phase 4 — runs courseforge_v1.shacl.ttl against outline-tier Block JSON-LD before rewrite tier.
```

## 5. Round-Trip Validators — Concretized

The proposal mentioned "round-trip validators" without specifying. Phase 4 ships exactly one, and seams the second:

**(a) `objective_roundtrip_similarity` (ship in Phase 4, warning).** For each LearningObjective Block:
1. Take the objective's full text.
2. Call the rewrite-tier model with prompt template "Paraphrase this learning objective preserving meaning: {text}". Use the same router-resolved model the rewrite tier uses (avoids cross-model drift).
3. Embed both, compute cosine.
4. If `cosine < 0.70` (placeholder), emit warning. Catches LLM-rewritten objectives that have drifted in *meaning* (e.g., the verb level changed but the wording obscured it).

**(b) `assessment_intent_roundtrip` (deferred to Phase 5; document seam).** Take the assessment item, ask the model "What objective does this question test?", embed the response, compare to declared `objective_ids[]` text. Higher cost (one LLM call per item) → defer until budget is justified by data.

Both share `lib/embedding/sentence_embedder.py` for embedding; both fire as warning-severity in Phase 4.

## 6. Threshold Tuning Strategy

Validators are useless without sensible thresholds. Concrete plan:

1. **Ship at warning severity.** Phase 4 lands all four new gates with `behavior.on_fail: warn`. They cannot block any promotion in this window.
2. **Calibration script.** New `scripts/calibrate_phase4_thresholds.py`:
   - Iterates over every LibV2 course at `LibV2/courses/<slug>/`.
   - Runs each Phase 4 validator over the Block-derived intermediate format (Phase 2 output).
   - Persists raw cosine distributions per validator to `state/phase4_calibration_<slug>_<gate_id>.jsonl`.
   - Aggregates p50/p90/p95/p99 across courses; prints suggested critical thresholds.
3. **Calibration target sample size.** Minimum 5 courses, ideally 10+. The existing rdf-shacl-551-2 corpus alone is insufficient — its LO style is atypical.
4. **Promotion criterion.** A gate is promoted from warning → critical when:
   - Empirical p99 of the cosine distribution is at least 0.05 above the proposed critical threshold (false-positive ceiling).
   - At least 2 courses' worth of warning-severity production runs have shipped without operator complaints.
5. **Drift watchdog.** New per-run `state/phase4_thresholds_observed.jsonl` records actual p50/p99 each run; an out-of-band script alerts when drift exceeds ±0.05 from the calibrated baseline.

## 7. Decision-Capture Events

New `decision_type` enum values in `schemas/events/decision_event.schema.json:63`:

| New `decision_type` | Required rationale fields |
|---|---|
| `statistical_validation_pass` | `validator_name`, `score` (cosine or aggregate), `threshold`, `block_id` |
| `statistical_validation_fail` | `validator_name`, `score`, `threshold`, `block_id`, `severity` |
| `shacl_outline_violation` | `focus_node`, `source_shape`, `result_severity`, `message`, `path` |
| `embedding_deps_missing` | `validator_name`, `fallback_taken` (bool) |

Events fire from each new validator's `validate()` body, immediately before returning `GateResult`. Use the existing `MCP.hardening.validation_gates.GateIssue` shape for the issue field; capture the rationale via the project-standard `lib.trainforge_capture` pattern documented at root `CLAUDE.md::Decision Capture Protocol`.

## 8. Performance Budget

**Estimate per typical course.**
- ~500 chunks × 1 embedding/chunk = 500 chunk embeds.
- ~50 LOs × 1 embed each = 50 LO embeds.
- ~50 assessment items × 1 stem embed + 1 objective-text re-embed (cached) = 50 net new embeds.
- ~80 example Blocks × 1 embed = 80 embeds.
- Total ~680 fresh embeds + ~100 cosine ops at <1 µs each.
- At 5 ms/embed CPU on `all-MiniLM-L6-v2` → ~3.4 s wall.
- With GPU available → ~0.3 s wall.

**Round-trip validator cost.** `objective_roundtrip_similarity` adds ~50 LLM calls per course. At ~5s/call on `local` provider → ~4 min wall. This is the dominant cost; warrants `--skip-roundtrip` flag for fast iteration.

**Caching strategy.**
- Per-block `content_hash → embedding` cache at `state/embedding_cache.jsonl`. Re-runs of the same course see ~0 new embeds.
- Roundtrip results keyed on `(model_id, prompt_template_hash, content_hash)` at `state/roundtrip_cache.jsonl`. Mirrors the Wave 107 `claude_session` cache pattern documented in `Trainforge/CLAUDE.md`.

## 9. Validation-Gate Wiring (workflows.yaml)

Add to `config/workflows.yaml::textbook_to_course::content_generation::validation_gates` (after `content_grounding`, ~line 678):

```yaml
- gate_id: outline_shacl
  validator: lib.validators.courseforge_outline_shacl.CourseforgeOutlineShaclValidator
  severity: warning
  threshold: { max_critical_issues: 0 }
  behavior: { on_fail: warn, on_error: warn }

- gate_id: objective_assessment_similarity
  validator: lib.validators.objective_assessment_similarity.ObjectiveAssessmentSimilarityValidator
  severity: warning
  threshold: { min_cosine: 0.55, max_critical_issues: 0 }
  behavior: { on_fail: warn, on_error: warn }

- gate_id: concept_example_similarity
  validator: lib.validators.concept_example_similarity.ConceptExampleSimilarityValidator
  severity: warning
  threshold: { min_cosine: 0.50, max_critical_issues: 0 }
  behavior: { on_fail: warn, on_error: warn }

- gate_id: objective_roundtrip_similarity
  validator: lib.validators.objective_roundtrip_similarity.ObjectiveRoundtripSimilarityValidator
  severity: warning
  threshold: { min_cosine: 0.70, max_critical_issues: 0 }
  behavior: { on_fail: warn, on_error: warn }
```

`objective_assessment_similarity` is wired on `content_generation` (where assessment items are generated) AND duplicated on `trainforge_assessment` (where the Phase 3 router may generate additional assessments). Same gate id, same validator, different inputs.

CLAUDE.md `Active Gates` table receives 4 new rows under `textbook_to_course::content_generation`. Phase 4 documentation footnote: "warning-severity during calibration; promote to critical per §6 plan."

## 10. Testing Plan

- **Unit tests** under `lib/tests/test_phase4_*.py` per new validator: fixture Block JSON with hand-crafted high-similarity / low-similarity / edge-case pairs. Mirror `lib/validators/tests/` structure.
- **Embedding test** at `lib/tests/test_embedding_cache.py`: cache hit/miss, sha256 keying, JSONL roundtrip, deps-missing fallback.
- **Integration test** at `lib/tests/test_phase4_content_generation_integration.py`: full content_generation phase with all 4 gates wired; uses minimal 1-week fixture course; asserts gate-fire decision events appear.
- **Performance test** at `lib/tests/test_phase4_throughput.py`: assert 500-chunk corpus completes embedding pass in ≤10 s (CI-CPU budget). Skipped on `TRAINFORGE_REQUIRE_EMBEDDINGS=false`.
- **Regression test**: legacy course (no Phase 2 Block intermediate format) — validators must `passed=True` with `code="LEGACY_NO_BLOCKS_SKIP"` warning issue (pattern from `lib/validators/property_coverage.py` no-op-on-missing-manifest behavior).
- **SHACL regression**: existing `lib/validators/tests/test_shacl_runner.py` already pins runner behavior; new `test_courseforge_outline_shacl.py` adds shape-level assertions with synthetic violation inputs.

## 11. Sequencing

1. Land `lib/embedding/sentence_embedder.py` + cache + lazy-load.
2. Add `decision_type` enum values to `schemas/events/decision_event.schema.json`.
3. Land `objective_assessment_similarity` + unit tests.
4. Land `concept_example_similarity` + unit tests.
5. Land `objective_roundtrip_similarity` + unit tests + roundtrip cache.
6. Land `CourseforgeOutlineShaclValidator` + unit tests.
7. Wire all 4 gates into `config/workflows.yaml` at warning severity.
8. Land `scripts/calibrate_phase4_thresholds.py`.
9. Update `CLAUDE.md::Active Gates` and per-validator docstrings.
10. Run calibration over ≥5 LibV2 courses; record observed thresholds.
11. (Future wave) Promote stable gates from warning → critical.

## 12. Risks & Rollback

1. **Embedding-model availability in CI.** `sentence-transformers` adds ~500 MB of torch+models to test envs. **Mitigation**: gate behind the existing `[training]` extra (or new `[embedding]` extra); CI without extras gets the warning fallback path. Rollback: revert workflows.yaml gate registrations.
2. **Threshold drift / false positives.** Initial cosines may be calibrated against atypical corpora. **Mitigation**: warning-severity ships first; `state/phase4_thresholds_observed.jsonl` tracks drift; calibration script re-runnable.
3. **Round-trip cost dominates wall time.** ~4 min added per course on `local` provider. **Mitigation**: per-objective roundtrip cache; `--skip-roundtrip` flag; deferred-to-Phase-5 framing for `assessment_intent_roundtrip`.
4. **Dependency bloat from torch.** Already present via `[training]` extra (`pyproject.toml`). **Mitigation**: keep embedding fully optional; CPU-only `all-MiniLM-L6-v2` ships without bitsandbytes. Hard rollback: remove `lib/embedding/` package; warnings vanish; no critical gates regress.

## 13. Open Questions for the User

1. **Brief contradicts repo state on `assessment_objective_alignment` upgrade.** The existing validator at `lib/validators/assessment_objective_alignment.py:38` is a **structural** ID-match (Wave 24), not a Jaccard semantic check. Should Phase 4 (a) keep it as-is and add `objective_assessment_similarity` as a parallel semantic gate (my plan); or (b) actually retrofit it with Jaccard→embedding internally? Plan currently chooses (a).
2. **Phase 2 Block contract.** The plan assumes Block dataclass-derived JSON-LD lands at `outputs.blocks_path` per phase. The exact field name + JSONL vs JSON-document layout needs the Phase 2 spec to lock the validator input schema.
3. **Phase 3 router gate seam location.** "Inter-tier gate seam" was referenced but not specified. Is it: (a) a single global gate-runner between outline-tier and rewrite-tier, or (b) tier-aware gates fired by the router itself? Plan currently assumes (a).
4. **Embedding cache scope.** Per-run vs per-course-persistent? Plan assumes per-course persistent (`LibV2/courses/<slug>/state/embedding_cache.jsonl`). Per-run is safer but slower.
5. **Threshold-promotion authority.** Who decides when a warning gate becomes critical? Plan suggests an explicit follow-up wave with operator sign-off.
6. **Round-trip provider routing.** Should `objective_roundtrip_similarity` use the same provider as the rewrite-tier model (preferred for fairness), or a fixed reference model (preferred for stability across rewrite-tier swaps)? Plan currently chooses the rewrite-tier model.
7. **Closed-world overlay interaction.** Should `outline_shacl` honor `TRAINFORGE_SHACL_CLOSED_WORLD` (per `lib/validators/shacl_runner.py:86`)? Default behavior in plan: yes — same flag, same overlay, same semantics.

### Critical Files for Implementation

- `/home/user/Ed4All/lib/validators/shacl_runner.py`
- `/home/user/Ed4All/Trainforge/eval/key_term_precision.py`
- `/home/user/Ed4All/lib/validators/assessment_objective_alignment.py`
- `/home/user/Ed4All/config/workflows.yaml`
- `/home/user/Ed4All/schemas/context/courseforge_v1.shacl.ttl`
