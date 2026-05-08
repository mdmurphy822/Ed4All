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
| `textbook_to_course` | `concept_extraction` | `concept_graph` | ConceptGraphValidator (warning) |
| `textbook_to_course` | `course_planning` | `abcd_verb_alignment` | AbcdObjectiveValidator (warning) |
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
| `textbook_to_course` | `post_rewrite_validation` | `claim_support` | ClaimSupportValidator (warning — Wave 2 W2.F per-claim NLI entailment vs cited chunks; fires UNSUPPORTED_CLAIM >20% and CONTRADICTED_CLAIM >5%; stamps `key_claims[].outcome` shape per Block per `chunk_v4.schema.json::key_claims` `oneOf` projection — Wave 1.5 W1.5.A migration contract preserves legacy List[str] back-compat) |
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
| `textbook_to_course` | `training_synthesis` | `pair_claim_support` | PairClaimSupportValidator (warning — Wave 4 W4.A per-claim NLI entailment vs cited chunks; day-1 warning per Wave 1.5/1.6/1.7 contract, flips to critical once per-corpus baseline is calibrated; stamps `per_claim_support[]` per pair per `instruction_pair.schema.json:113-187`. Wave 9 TIGHT extends `outcome` enum with `dart_disagreement` value sourced from `dart_source_check` sub-stamp; gate emits aggregate-rate `DART_DISAGREEMENT_RATE_HIGH` warning at >5%) |
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
