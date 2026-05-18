# Active Validation Gates

> Source of truth: `config/workflows.yaml::validation_gates`. This file is a human-readable mirror with per-validator Wave-reference paragraphs. Per-flag rows below show the phase at which each gate fires; severity in parentheses (`critical` when unmarked).


| Workflow | Phase | Gate | Validator |
|----------|-------|------|-----------|
| `course_generation` | `content_generation` | `content_structure` | ContentStructureValidator |
| `course_generation` | `inter_tier_validation` | `outline_curie_anchoring` | BlockCurieAnchoringValidator |
| `course_generation` | `inter_tier_validation` | `outline_content_type` | BlockContentTypeValidator |
| `course_generation` | `inter_tier_validation` | `outline_page_objectives` | BlockPageObjectivesValidator |
| `course_generation` | `inter_tier_validation` | `outline_assessment_item_payload` | BlockAssessmentItemPayloadValidator |
| `course_generation` | `post_rewrite_validation` | `rewrite_curie_anchoring` | BlockCurieAnchoringValidator |
| `course_generation` | `post_rewrite_validation` | `rewrite_content_type` | BlockContentTypeValidator |
| `course_generation` | `post_rewrite_validation` | `rewrite_page_objectives` | BlockPageObjectivesValidator |
| `course_generation` | `post_rewrite_validation` | `rewrite_source_refs` | BlockSourceRefValidator |
| `course_generation` | `post_rewrite_validation` | `rewrite_assessment_item_payload` | BlockAssessmentItemPayloadValidator |
| `course_generation` | `post_rewrite_validation` | `rewrite_html_shape` | RewriteHtmlShapeValidator |
| `course_generation` | `post_rewrite_validation` | `rewrite_source_grounding` | RewriteSourceGroundingValidator |
| `course_generation` | `packaging` | `imscc_structure` | IMSCCValidator |
| `course_generation` | `packaging` | `page_objectives` | PageObjectivesValidator |
| `course_generation` | `packaging` | `page_objectives_shacl` | PageObjectivesShaclValidator (warning) |
| `course_generation` | `validation` | `wcag_compliance` | WCAGValidator |
| `course_generation` | `validation` | `oscqr_score` | OSCQRValidator (warning) |
| `intake_remediation` | `parsing` | `imscc_parse` | IMSCCParseValidator |
| `intake_remediation` | `validation` | `wcag_compliance` | WCAGValidator |
| `batch_dart` | `multi_source_synthesis` | `dart_markers` | DartMarkersValidator |
| `batch_dart` | `validation` | `wcag_aa_compliance` | WCAGValidator |
| `textbook_to_course` | `dart_conversion` | `dart_markers` | DartMarkersValidator |
| `textbook_to_course` | `chunking` | `chunkset_manifest` | ChunksetManifestValidator (warning) |
| `textbook_to_course` | `objective_extraction` | `textbook_outline_enrichment` | TextbookOutlineValidator (warning — three-stage textbook synthesis Stage-1 gate; audits the `semantic_outline` + `draft_terminal_objectives` keys the `TextbookSynthesisProvider` folds into `textbook_structure.json` when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. Floor checks: `semantic_outline.themes[]` non-empty; every theme `chapter_ids[]` resolves against `chapters[].id`; `draft_terminal_objectives[]` non-empty with each draft TO carrying a non-empty `statement` + a valid `bloom_level`. GateIssue codes: `OUTLINE_THEMES_EMPTY`, `OUTLINE_THEME_CHAPTER_UNRESOLVED`, `OUTLINE_DRAFT_TO_EMPTY`, `OUTLINE_DRAFT_TO_MALFORMED`. Skips-with-pass when the enrichment keys are absent — default-off runs (`TEXTBOOK_SYNTHESIS_PROVIDER` unset) are unaffected, mirroring the `ABCD_MISSING` graceful-degrade contract. Warning-severity day-1 per the standing calibration-debt pattern; promote to critical after a clean-corpus calibration run) |
| `textbook_to_course` | `concept_extraction` | `concept_graph` | ConceptGraphValidator (warning) |
| `textbook_to_course` | `concept_extraction` | `domain_concept_vocabulary` | DomainConceptVocabularyValidator (warning — three-stage textbook synthesis Stage-3 gate; audits the `domain_concept_vocabulary.json` sibling of `concept_graph_semantic.json` produced by the per-chapter `TextbookSynthesisProvider.synthesize_concepts` pass when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. Floor checks: `>= _MIN_CONCEPTS` (default 8 — matches `ConceptGraphValidator`'s ≥10-node floor with co-occurrence-filter headroom); every concept carries a non-empty `canonical`; `concept_count == len(concepts)`. The `chapter_synthesis_failures` count is surfaced informationally (per-chapter failure isolation, not a fail). GateIssue codes: `VOCAB_THIN`, `VOCAB_CONCEPT_NO_CANONICAL`, `VOCAB_COUNT_MISMATCH`, `VOCAB_PARTIAL_COVERAGE` (warning-only). Skips-with-pass when the vocabulary artifact is absent — default-off runs are unaffected. Warning-severity day-1 per the calibration-debt pattern) |
| `textbook_to_course` | `course_planning` | `abcd_verb_alignment` | AbcdObjectiveValidator (warning) |
| `textbook_to_course` | `course_planning` | `chapter_objective_coverage` | ChapterObjectiveCoverageValidator (warning — three-stage textbook synthesis Stage-2 gate; audits the per-chapter `CO-NN` objectives + reconciled `terminal_objectives[]` produced by `TextbookSynthesisProvider.synthesize_chapter_objectives` + `reconcile_terminal_objectives` when `TEXTBOOK_SYNTHESIS_PROVIDER` is set. Floor checks: every `chapters[].id` with non-empty text produced ≥1 CO (cross-checks `chapter_synthesis_failures`); reconciled `terminal_objectives[]` non-empty. GateIssue codes: `CHAPTER_OBJ_MISSING`, `CHAPTER_OBJ_PARTIAL_COVERAGE`, `RECONCILED_TO_EMPTY`. Skips-with-pass when the synthesis enrichment is absent — default-off runs fall through to the deterministic `synthesize_objectives_from_topics` path and are unaffected. Warning-severity day-1 per the calibration-debt pattern; surfaces per-chapter degradation so a partial LLM outage is never silent) |
| `textbook_to_course` | `course_planning` | `objective_source_refs` | ObjectiveSourceRefValidator (warning — Wave 1.6 W1.6.C; resolves each synthesized LO's `source_refs[]` against the union of `textbook_structure.chapters[*].id` ∪ `chapters[*].sections[*].id` ∪ DART chunkset universe; `require_to_attribution=false` day-1) |
| `textbook_to_course` | `content_generation` | `content_structure` | ContentStructureValidator (warning) |
| `textbook_to_course` | `content_generation` | `source_refs` | PageSourceRefValidator |
| `textbook_to_course` | `content_generation` | `content_grounding` | ContentGroundingValidator |
| `textbook_to_course` | `inter_tier_validation` | `outline_curie_anchoring` | BlockCurieAnchoringValidator |
| `textbook_to_course` | `inter_tier_validation` | `outline_content_type` | BlockContentTypeValidator |
| `textbook_to_course` | `inter_tier_validation` | `outline_page_objectives` | BlockPageObjectivesValidator |
| `textbook_to_course` | `inter_tier_validation` | `outline_source_refs` | BlockSourceRefValidator |
| `textbook_to_course` | `inter_tier_validation` | `outline_assessment_item_payload` | BlockAssessmentItemPayloadValidator |
| `textbook_to_course` | `inter_tier_validation` | `outline_shacl` | CourseforgeOutlineShaclValidator (warning) |
| `textbook_to_course` | `inter_tier_validation` | `outline_objective_assessment_similarity` | ObjectiveAssessmentSimilarityValidator (warning) |
| `textbook_to_course` | `inter_tier_validation` | `outline_concept_example_similarity` | ConceptExampleSimilarityValidator (warning) |
| `textbook_to_course` | `inter_tier_validation` | `outline_objective_roundtrip_similarity` | ObjectiveRoundtripSimilarityValidator (warning) |
| `textbook_to_course` | `inter_tier_validation` | `outline_bloom_classifier_disagreement` | BloomClassifierDisagreementValidator (warning) |
| `textbook_to_course` | `inter_tier_validation` | `outline_block_objective_delivery` | BlockObjectiveDeliveryValidator (warning — Wave 1.7 W1.7.C tri-axis NLI / Bloom-gap / verb check; stamps `objectiveAlignment[]` per Block per `$defs.ObjectiveAlignment` in `schemas/knowledge/courseforge_jsonld_v1.schema.json`) |
| `textbook_to_course` | `inter_tier_validation` | `outline_assessment_retrieval_grounding` | AssessmentRetrievalGroundingValidator (warning; calibration-gated severity flip — see `lib/governance/calibration_gate.py::resolve_severity_flip`) |
| `textbook_to_course` | `inter_tier_validation` | `outline_distractor_plausibility` | DistractorPlausibilityValidator (warning) |
| `textbook_to_course` | `inter_tier_validation` | `outline_distractor_misconception_alignment` | DistractorMisconceptionAlignmentValidator (warning) |
| `textbook_to_course` | `inter_tier_validation` | `outline_distractor_structural` | DistractorStructuralValidator (warning) |
| `textbook_to_course` | `inter_tier_validation` | `outline_padded_distractor` | PaddedDistractorValidator |
| `textbook_to_course` | `inter_tier_validation` | `outline_instructional_depth` | InstructionalDepthValidator (warning) |
| `textbook_to_course` | `inter_tier_validation` | `outline_bloom_structural_enforcement` | BloomStructuralEnforcementValidator (warning) |
| `textbook_to_course` | `content_generation_rewrite` | `content_grounding` | ContentGroundingValidator |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_curie_anchoring` | BlockCurieAnchoringValidator |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_content_type` | BlockContentTypeValidator |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_page_objectives` | BlockPageObjectivesValidator |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_source_refs` | BlockSourceRefValidator |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_assessment_item_payload` | BlockAssessmentItemPayloadValidator |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_html_shape` | RewriteHtmlShapeValidator |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_source_grounding` | RewriteSourceGroundingValidator |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_shacl` | CourseforgeOutlineShaclValidator (warning) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_objective_assessment_similarity` | ObjectiveAssessmentSimilarityValidator (warning) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_concept_example_similarity` | ConceptExampleSimilarityValidator (warning) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_objective_roundtrip_similarity` | ObjectiveRoundtripSimilarityValidator (warning) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_bloom_classifier_disagreement` | BloomClassifierDisagreementValidator (warning) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_block_objective_delivery` | BlockObjectiveDeliveryValidator (warning — Wave 1.7 W1.7.C tri-axis post-rewrite mirror; stamps post-rewrite `objectiveAlignment[]` per Block; mirrors `outline_block_objective_delivery` shape) |
| `textbook_to_course` | `post_rewrite_validation` | `claim_support` | ClaimSupportValidator (warning — Wave 2 W2.F per-claim NLI entailment vs cited chunks; fires UNSUPPORTED_CLAIM >20% and CONTRADICTED_CLAIM >5%; stamps `key_claims[].outcome` shape per Block per `chunk_v4.schema.json::key_claims` `oneOf` projection — Wave 1.5 W1.5.A migration contract preserves legacy List[str] back-compat. Wave W-D11 T11.1 adds substring + char_span enforcement against the optional `evidence_quote` / `evidence_char_span` fields on each structured claim entry: emits `EVIDENCE_QUOTE_NOT_SUBSTRING` when the quote is not a verbatim substring of any cited chunk, `EVIDENCE_CHAR_SPAN_MISMATCH` when `chunk_text[start:end] != evidence_quote`, and reserves `EVIDENCE_QUOTE_MISSING` for the future threshold-gated coverage warning (T11.5/T11.6). Aggregate counters surfaced on `GateResult.metadata`: `evidence_quote_coverage_rate`, `evidence_quote_substring_fail_rate`, `evidence_quote_char_span_mismatch_rate` (always present, 0.0 default). Day-1 warning-severity per the W1.7.C / W4.A / W4.C calibration-deferred contract — evidence-quote findings do NOT contribute to the regenerate signal and `passed` stays True; future critical flip via per-block-type thresholds once per-corpus baselines are calibrated) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_assessment_retrieval_grounding` | AssessmentRetrievalGroundingValidator (warning; calibration-gated severity flip — see `lib/governance/calibration_gate.py::resolve_severity_flip`) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_distractor_plausibility` | DistractorPlausibilityValidator (warning) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_distractor_misconception_alignment` | DistractorMisconceptionAlignmentValidator (warning) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_distractor_structural` | DistractorStructuralValidator (warning) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_padded_distractor` | PaddedDistractorValidator |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_instructional_depth` | InstructionalDepthValidator (warning) |
| `textbook_to_course` | `post_rewrite_validation` | `rewrite_bloom_structural_enforcement` | BloomStructuralEnforcementValidator (warning) |
| `textbook_to_course` | `packaging` | `imscc_structure` | IMSCCValidator (warning) |
| `textbook_to_course` | `packaging` | `page_objectives` | PageObjectivesValidator |
| `textbook_to_course` | `packaging` | `page_objectives_shacl` | PageObjectivesShaclValidator (warning) |
| `textbook_to_course` | `imscc_chunking` | `chunkset_manifest` | ChunksetManifestValidator (warning) |
| `textbook_to_course` | `trainforge_assessment` | `imscc_input_valid` | IMSCCValidator (pre-assessment) |
| `textbook_to_course` | `trainforge_assessment` | `assessment_quality` | AssessmentQualityValidator (critical at corpus-wide level for back-compat; per-question-type diversity sub-checks fire warning-severity day-1 via `_PER_QUESTION_TYPE_THRESHOLDS` in `lib/validators/assessment.py`. Calibration plan: rebuild rdf-shacl-551-2 with W5 fields → tune per-type thresholds → flip warning → critical in a micro-wave. Mirrors W1.7.C `_PER_BLOCK_TYPE_ENTAILMENT_FLOOR` + W4.C `_PER_PAIR_KIND_ENTAILMENT_FLOOR` patterns.) |
| `textbook_to_course` | `trainforge_assessment` | `assessment_objective_alignment` | AssessmentObjectiveAlignmentValidator |
| `textbook_to_course` | `trainforge_assessment` | `trainforge_padded_distractor` | PaddedDistractorValidator |
| `textbook_to_course` | `training_synthesis` | `synthesis_quota` | SynthesisQuotaValidator (warning) |
| `textbook_to_course` | `training_synthesis` | `min_edge_count` | MinEdgeCountValidator |
| `textbook_to_course` | `training_synthesis` | `synthesis_diversity` | SynthesisDiversityValidator |
| `textbook_to_course` | `training_synthesis` | `property_coverage` | PropertyCoverageValidator (no-ops on courses without a property manifest) |
| `textbook_to_course` | `training_synthesis` | `synthesis_leakage` | SynthesisLeakageValidator (fails closed at >5% verbatim chunk leakage) |
| `textbook_to_course` | `training_synthesis` | `curie_anchoring` | CurieAnchoringValidator (binary per-pair anchoring sentinel, default min_pair_anchoring_rate=0.95; supersedes the deprecated `lib/validators/curie_preservation.py::CuriePreservationValidator` shim — Wave 135c→135d migration; shim removal target Wave 137. Operators using `curie_preservation` in custom workflows: rename to `curie_anchoring` and update threshold key from `min_mean_retention` (0.40) to `min_pair_anchoring_rate` (0.95)) |
| `textbook_to_course` | `training_synthesis` | `pair_claim_support` | PairClaimSupportValidator (warning — Wave 4 W4.A per-claim NLI entailment vs cited chunks; day-1 warning per Wave 1.5/1.6/1.7 contract, flips to critical once per-corpus baseline is calibrated; stamps `per_claim_support[]` per pair per `instruction_pair.schema.json:113-187`. Wave 9 TIGHT extends `outcome` enum with `dart_disagreement` value sourced from `dart_source_check` sub-stamp; gate emits aggregate-rate `DART_DISAGREEMENT_RATE_HIGH` warning at >5%. Wave W-D11 T11.2 mirrors the block-surface evidence-quote enforcement on the pair surface: emits `EVIDENCE_QUOTE_NOT_SUBSTRING` when the structured claim's `evidence_quote` is not a verbatim substring of any chunk in the cited universe, `EVIDENCE_CHAR_SPAN_MISMATCH` when `chunk_text[start:end] != evidence_quote`, and reserves `EVIDENCE_QUOTE_MISSING` for the future threshold-gated coverage warning. Aggregate counters surfaced on `GateResult.metadata`: `evidence_quote_coverage_rate`, `evidence_quote_substring_fail_rate`, `evidence_quote_char_span_mismatch_rate` (always present, 0.0 default). Day-1 warning-severity per the calibration-deferred contract — evidence-quote findings do NOT contribute to the regenerate signal and `passed` stays True; future critical flip via per-pair-kind thresholds once per-corpus baselines are calibrated) |
| `textbook_to_course` | `training_synthesis` | `pair_lo_refs` | PairLearningOutcomeRefsValidator (Wave 4 W4.B — per-pair learning_outcome_refs subset check; critical because phantom-LO is structural; mirrors AssessmentObjectiveAlignmentValidator severity; stamps `pair_lo_resolution.{declared_los,chunk_los,phantom_los}` per `instruction_pair.schema.json:188-209`. Wave 8 deterministic-template paths additionally stamp `pair_lo_resolution.skipped: "deterministic_template"` — currently rejected by `additionalProperties: false`; W-D1 P0.1 is the schema fix.) |
| `textbook_to_course` | `training_synthesis` | `pair_objective_delivery` | PairObjectiveDeliveryValidator (warning — Wave 4 W4.C MEDIUM per-pair-per-objective tri-axis NLI entailment + Bloom-gap + verb cooccurrence delivery audit; mirrors BlockObjectiveDeliveryValidator on the pair surface; day-1 warning per Wave 1.5/1.6/1.7 contract, flips to critical once per-corpus baseline is calibrated; stamps `pair_objective_alignment[]` per pair per `instruction_pair.schema.json:211-268`; aggregate `pair_objective_alignment_pass_rate` carried at line 269-273) |
| `textbook_to_course` | `training_synthesis` | `pair_promotion` | TrainingPairPromotionValidator |
| `textbook_to_course` | `libv2_archival` | `libv2_manifest` | LibV2ManifestValidator |
| `textbook_to_course` | `libv2_archival` | `packet_integrity_strict` | PacketIntegrityValidator |
| `textbook_to_course` | `libv2_archival` | `semantic_graph_rule_output` | SemanticGraphRuleOutputValidator (warning) |
| `textbook_to_course` | `libv2_archival` | `kg_quality_report` | KGQualityValidator (critical, thresholds 0.95/0.95/0.95/0.5) |
| `textbook_to_course` | `libv2_archival` | `chunkset_drift` | ChunksetDriftValidator (warning — DART vs. IMSCC chunkset drift; sidecar `drift_report.json`) |
| `rag_training` | `assessment_generation` | `assessment_quality` | AssessmentQualityValidator (critical at corpus-wide level for back-compat; per-question-type diversity sub-checks fire warning-severity day-1 via `_PER_QUESTION_TYPE_THRESHOLDS` in `lib/validators/assessment.py`. Calibration plan: rebuild rdf-shacl-551-2 with W5 fields → tune per-type thresholds → flip warning → critical in a micro-wave. Mirrors W1.7.C `_PER_BLOCK_TYPE_ENTAILMENT_FLOOR` + W4.C `_PER_PAIR_KIND_ENTAILMENT_FLOOR` patterns.) |
| `rag_training` | `assessment_generation` | `bloom_alignment` | BloomAlignmentValidator (warning) |
| `rag_training` | `assessment_generation` | `leak_check` | LeakCheckValidator |
| `rag_training` | `assessment_generation` | `outcome_ref_integrity` | LeakCheckValidator (warning) |
| `rag_training` | `assessment_generation` | `content_fact_check` | ContentFactValidator (warning) |
| `rag_training` | `assessment_generation` | `question_quality` | QuestionQualityValidator |
| `rag_training` | `validation` | `final_quality` | FinalQualityValidator |
| `trainforge_train` | `post_training_validation` | `eval_gating` | EvalGatingValidator (fails closed on regression / yes-bias / no-bias / source-match drop; also surfaces `EVAL_CONTENT_TYPE_ROLE_ALIGNMENT_LOW` warning-severity GateIssue when `content_type_role_alignment_summary.alignment_rate` < 0.70 — internal warning code, not a separately-registered gate. Wave 7 W7.D iterates each W3.F metric's `per_question_type` bucket and emits a per-type warning code per relevant bucket below its floor: `EVAL_ANSWERABLE_PER_TYPE_BELOW_THRESHOLD`, `EVAL_SINGLE_CORRECT_PER_TYPE_BELOW_THRESHOLD`, `EVAL_DISTRACTOR_ENTROPY_PER_TYPE_BELOW_THRESHOLD`, `EVAL_BLOOM_ALIGNMENT_PER_TYPE_BELOW_THRESHOLD`, `EVAL_PLACEHOLDER_PER_TYPE_ABOVE_THRESHOLD`, `EVAL_SOURCE_SUPPORT_PER_TYPE_BELOW_THRESHOLD` — all warning-severity day-1, calibration-deferred per plan §5; non-relevant + deps-missing buckets are skipped) |
| `trainforge_train` | `post_training_validation` | `family_completeness` | FamilyCompletenessValidator (fails closed when any CURIE family is partially complete; family clusters declared in `schemas/training/family_map.<family>.yaml`) |
