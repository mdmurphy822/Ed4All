# Cross-cutting behavior flags

This registry documents root-owned environment controls that affect more than
one Ed4All subsystem. Subsystem-owned flags remain in the adjacent
[Courseforge](behavior-flags-courseforge.md),
[SemantiK](behavior-flags-semantik.md), and
[Trainforge](behavior-flags-trainforge.md) registries.

The live resolver and workflow configuration are authoritative if a release and
this guide disagree. Never lower a validation threshold to make an artifact
pass. Provider or model changes also require review against
[Licensing and ToS posture](../LICENSING.md).

## Reading the registry

Unless a row says otherwise, boolean flags accept `1`, `true`, `yes`, or `on`
case-insensitively. "Parse-with-fallback" means malformed operator input returns
to the documented default; it never authorizes silent degradation of artifact
quality. Values that alter artifacts or generation identity may invalidate
resume state. Operational-only controls are identified in their rows.

Machine-specific service endpoints, launch commands, and capacity settings
belong in ignored local configuration. Use [Seat management](seat-scripts.md),
[Launch a model service](launch-seat.example.sh), and the portable
[seat-schedule template](seat-schedule.env.example) instead of embedding a host
recipe here.

## Governance and dispatch

| Flag | Default | Current contract |
|---|---|---|
| `DECISION_VALIDATION_STRICT` | unset | Fails closed on unknown `decision_type` values in decision captures. Source: `lib/decision_capture.py`. |
| `ED4ALL_CAPTURE_BUFFER` | unset (off) | Decision-capture write buffering. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/decision_capture.py`. |
| `ED4ALL_CAPTURE_BUFFER_ROWS` | `50` | Satellite of `ED4ALL_CAPTURE_BUFFER` — the buffered-row batch size. Invalid or out-of-range values use the documented default. Source: `lib/decision_capture.py`. |
| `ED4ALL_BLOCK_QUALITY_RUBRIC` | unset (off; auto-on for pipeline runs) | Gates the eight-dimension 0-3 block-quality scoring pass. Source: `lib/validators/block_quality_rubric.py`, `lib/validators/anatomy_slot_presence.py`. |
| `ED4ALL_BLOCK_BODY_CHAR_CEILING` | `200` (global override) / per-type default | Load body ceiling. Invalid or out-of-range values use the documented default. Source: `lib/validators/content.py`, `lib/validators/_block_rubric_helpers.py`. |
| `ED4ALL_BLOCK_QUALITY_SHADOW` | unset (off; auto-on for pipeline runs) | Runs block-quality measurement without rendering rubric output or changing gate verdicts. Truthy tokens enable measurement; false or unrecognized values leave it off. Source: `lib/validators/block_quality_rubric.py`. |
| `ED4ALL_COVERAGE_FLOOR` | `0.80` | 1 source→objective coverage floor for the post-loop `PromotionChainAggregator`. Invalid or out-of-range values use the documented default. Source: `lib/aggregators/promotion_chain_report.py`. |
| `ED4ALL_COVERAGE_DROP_STRICT` | unset (off) | Makes a `COVERAGE_DROP` signal blocking instead of advisory. Truthy tokens enable strict gating; false or unrecognized values leave it advisory. Source: `lib/aggregators/promotion_chain_report.py`. |
| `ED4ALL_KG_REAL_FLOORS` | unset (off) | Gate for recomputing the `completeness` + `accuracy` KG-quality scores before the existing per-dimension floor check. Source: `lib/validators/kg_quality.py`. |
| `MCP_ORCHESTRATOR_LLM_MODEL` | `claude-opus-4-7` | Pins the Anthropic model ID used by `MCP/orchestrator/llm_backend.py::DEFAULT_ANTHROPIC_MODEL`; per-run `LLM_MODEL` keeps higher precedence. Source: `MCP/orchestrator/llm_backend.py`. |
| `LOCAL_DISPATCHER_ALLOW_STUB` | unset | Permits `LocalDispatcher` to emit a stubbed `PhaseOutput` when no `agent_tool` is wired. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `MCP/orchestrator/api_dispatcher.py`. |
| `ED4ALL_CLOUD_RATE_LIMIT` | unset (off) | Hosted-large build profile SETUP — master switch for the shared cloud-seat admission gate. Source: `lib/llm/rate_limiter.py`, `Trainforge/generators/providers/_openai_compatible_client.py`. |
| `ED4ALL_AGENT_DISPATCH` | unset | Routes subagent-classified agents through `dispatcher.dispatch_task` instead of in-process tool registry. Source: `MCP/core/executor.py`. |
| `ED4ALL_AGENT_TIMEOUT_SECONDS` | `1800` | Per-task subagent dispatch mailbox timeout. Source: `MCP/core/executor.py`. |
| `ED4ALL_ALIGNMENT_VERB_TRIPLE` | unset (off) | Requires objective, instructional-block, and assessment verbs to form a consistent alignment triple. Unset leaves this validation axis off. Source: `lib/validators/`. |

## Grounded answers

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_ANSWER_PROVIDER` | `local` | Selects the grounded-answer backend (runtime Q&A inference) from the `_OPENAI_COMPATIBLE_PROVIDERS` registry. Source: `docs/LICENSING.md`. |
| `ED4ALL_ANSWER_MODEL` | per-provider | Model ID override for the answer backend. Source: `lib/retrieval/`. |
| `ED4ALL_ANSWER_TIMEOUT_SECONDS` | `120` | Answer-client HTTP timeout (long passages, slow local GPU). Invalid or out-of-range values use the documented default. Source: `lib/retrieval/`. |
| `ED4ALL_ANSWER_NUM_CTX` | `4096` | Sets the token window used to build grounded-answer prompts. Use a positive integer that matches the serving endpoint; invalid or non-positive values use `4096`. Source: `lib/retrieval/_prompts.py`. |
| `ED4ALL_ANSWER_CITATION_PRUNE` | `shadow` | Three-valued (`off` / `shadow` / `on`) governor of the claim-attribution citation prune + add pass at answer-composition time. Accepted modes are `off`, `shadow`, and `on`; invalid values use the documented default. Source: `lib/retrieval/citation_attribution.py`. |
| `ED4ALL_ANSWER_PRUNE_MIN_OVERLAP` | `0.25` | Float support threshold for the PRUNE decision (`citation_attribution.resolve_min_overlap`; also the `prune_min_overlap` kwarg). Invalid or out-of-range values use the documented default. Source: `lib/retrieval/`. |
| `ED4ALL_ANSWER_ADD_MIN_SHINGLE` | `0.50` | Sets the minimum lexical-shingle support required to add a citation. Values must be within the resolver’s supported range; invalid values use `0.50`. Source: `lib/retrieval/citation_attribution.py`. |
| `ED4ALL_ANSWER_NLI_ADD` | `off` | Controls NLI-assisted citation addition. Accepted modes are `off`, `shadow`, and `on`; invalid values use `off`. Source: `lib/retrieval/citation_attribution.py`. |
| `ED4ALL_ANSWER_ASSESSMENT_GUARD` | unset (`off`) | Three-valued (`off` / `shadow` / `on`) governor of the L2 assessment-aware answering guard on the learner ask path — "the tutor won't do your homework". Accepted modes are `off`, `shadow`, and `on`; invalid values use the documented default. Source: `lib/retrieval/assessment_guard.py`, `gui/services/answer_service.py`. |
| `ED4ALL_GOLD_ROUNDTRIP_ENGINE` | `lexical` | Retrieval engine for the OPT-IN gold-candidate round-trip prescreen filter. Accepted values follow `{lexical, semantic, hybrid-rrf}`. Source: `lib/retrieval/gold_authoring.py`. |
| `ED4ALL_ANSWER_ASSESSMENT_GUARD_THRESHOLD` | `0.75` | Float match floor for the L2 assessment guard. Invalid or out-of-range values use the documented default. Source: `lib/retrieval/assessment_guard.py`. |
| `ED4ALL_ANSWER_EXCLUDE_CHUNK_TYPES` | unset (off) | Comma-separated, lower-cased chunk types removed from every grounded-answer retrieval pass, including completeness and library-wide retrieval. Source: `lib/retrieval/grounded_answer.py`. |
| `ED4ALL_ANSWER_ANCHOR_CONTAINMENT` | `0.85` | Sets the minimum source-anchor containment score accepted by grounded-answer citation checks. Invalid or out-of-range values use `0.85`. Source: `lib/retrieval/grounded_answer.py`. |
| `ED4ALL_ANSWER_COMPLETENESS_RECHECK` | `on` | Checks a generated answer for unanswered parts before citation validation and retries composition when grounded material is available. `off` disables the check. Source: `lib/retrieval/answer_completeness.py`. |
| `ED4ALL_ANSWER_LIBRARY_WIDE` | unset (off) | Wide grounded ask. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/retrieval/library_wide.py`. |
| `ED4ALL_ANSWER_INTENT_ROUTE` | unset (off) | Route bias on the grounded-answer path. Source: `lib/retrieval/query_augment.py`, `lib/retrieval/grounded_answer.py`. |
| `ED4ALL_ANSWER_MULTITURN` | unset (off) | Turn antecedent query rewrite. Source: `lib/retrieval/query_augment.py`. |
| `ED4ALL_ANSWER_DECOMPOSE` | unset (off) | Part question decomposition. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/retrieval/query_augment.py`. |
| `ED4ALL_ANSWER_HYDE` | unset (off) | Embedding (HyDE) retrieval arm. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/retrieval/query_augment.py`. |
| `ED4ALL_ANSWER_GRAPH_EXPAND` | unset (off) | Graph passage expansion on the grounded-answer path. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/retrieval/graph_expand.py`, `lib/retrieval/grounded_answer.py`. |
| `ED4ALL_ANSWER_GRAPH_EXPAND_MAX` | `4` | Reachable neighbors appended by `ED4ALL_ANSWER_GRAPH_EXPAND`. Invalid or out-of-range values use the documented default. Source: `lib/retrieval/graph_expand.py`. |
| `ED4ALL_ANSWER_COMPLETENESS_RERETRIEVE` | unset (off) | Retrieves additional evidence for grounded question parts that the completeness check finds unanswered. Unset reuses the original passages. Source: `lib/retrieval/grounded_answer.py`. |
| `ED4ALL_ANSWER_HEDGE_TIER` | unset (off) | Adds calibrated hedging when retrieval confidence falls within the configured margin. Unset emits no hedge. Source: `lib/retrieval/grounded_answer.py`. |
| `ED4ALL_ANSWER_HEDGE_MARGIN` | `0.15` | Hedge band width for `ED4ALL_ANSWER_HEDGE_TIER`. Invalid or out-of-range values use the documented default. Source: `lib/retrieval/grounded_answer.py`. |

## Assistant and automated operation

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_ASSISTANT_BASE_URL` | `<local-endpoint>` | Base URL of the LOCAL Nemotron nano vLLM seat behind the `ed4all assistant` operator-assistant chat. Source: `lib/assistant/client.py`, `lib/assistant/tools.py`. |
| `ED4ALL_ASSISTANT_MODEL` | `nemotron-3-nano` | Model ID the assistant seat expects. Source: `lib/assistant/client.py`. |
| `ED4ALL_ASSISTANT_AUTOSTART` | unset (off) | Starts the configured assistant seat when `ed4all assistant` finds it unavailable. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; other values leave it off. Startup failures surface to the operator. Source: `lib/assistant/engine.py`, `lib/vllm_container_lifecycle.py`. |
| `ED4ALL_ASSISTANT_MAX_TOKENS` | `1024` | Per-reply generation cap for the assistant seat. Invalid or out-of-range values use the documented default. Source: `lib/assistant/client.py`, `docs/LICENSING.md`. |
| `ED4ALL_ASSISTANT_TIMEOUT_SECONDS` | `120` | Per-request HTTP timeout (s) for the assistant seat. Invalid or out-of-range values use the documented default. Source: `lib/assistant/client.py`. |
| `ED4ALL_ASSISTANT_SEAT` | `<logical-seat>` | Logical registry seat name the assistant autostart/hint path targets. Source: `lib/assistant/client.py`. |
| `ED4ALL_ASSISTANT_DEBUG_ON_FAILURE` | unset (off) | When truthy, the ignored local batch driver records a validated pointer to the latest failed workflow for assistant debug mode. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/assistant/debug_context.py`. |
| `ED4ALL_ASSISTANT_SEAT_PRIORITY` | `<logical-seat>`,`<logical-seat>` | Ordered comma-separated list of LOGICAL registry seat names the `ed4all assistant` DYNAMIC seat resolver walks. Source: `lib/assistant/client.py`. |
| `ED4ALL_PILOT_SCHEDULE` | unset (off) | Enables scheduling actions in the ignored local batch monitor. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: operator-local monitor. |
| `ED4ALL_CAMPAIGN_BASE_MODEL` | `nemotron3-nano-30b` | Base-model selector for local batch training. Source: `docs/LICENSING.md`, `Trainforge/training/base_models.py`. |
| `ED4ALL_CAMPAIGN_DIR` | ignored local operations directory | Root of the ignored local batch-operation workspace. Source: `lib/paths.py`. |
| `ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE` | unset (off; auto-on for pipeline runs) | "True full course" archival-completeness strict-mode gate (resolver `lib/validators/libv2/course_completeness.py::resolve_require_full_course`; gate `course_completeness` wired. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/validators/libv2/course_completeness.py`. |

## Generation, curriculum, and packaging

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_RERANK_PROVIDER` | unset (off) | Cross-encoder reranker over the first-stage retrieval candidate pool on the grounded-answer path (resolver `lib/retrieval/reranker.py::resolve_reranker_provider`; hook. Source: `lib/retrieval/reranker.py`, `lib/retrieval/grounded_answer.py`. |
| `ED4ALL_BLOCK_ANATOMY` | unset (off) | Slot anatomy contract emit gate. Source: `Courseforge/scripts/blocks.py`. |
| `ED4ALL_BLOCK_A11Y` | unset (off) | Block WCAG 2.2 AA + UDL emit gate. Source: `lib/generation/block_a11y.py`, `Courseforge/scripts/blocks.py`. |
| `ED4ALL_CALLOUT_TYPED` | unset (off) | FR--03 typed B12 callout emit + gate flag (renderer reader `Courseforge/scripts/rendering/generate_course.py::_callout_typed_enabled`; gate. Source: `Courseforge/scripts/rendering/generate_course.py`, `lib/validators/callout_structure.py`. |
| `ED4ALL_COS_PER_WEEK_CAP` | `0` (auto) | Caps course objectives assigned to each week; `0` selects the automatic uncapped slice. Source: `Courseforge/scripts/rendering/generate_course.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_WEEK_TO_GROUPS` | unset (off) | Override for the per-week `"Week N"` `chapter_objectives` groups persisted into `synthesized_objectives.json`. Source: `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_CONCEPT_COVERAGE` | unset (off; auto-on for pipeline runs) | Produces the concept-coverage aggregate report. Source: `lib/aggregators/concept_coverage.py`. |
| `ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT` | `on` (default ON) | Per-window resume-checkpoint sidecar for the `concept_extraction` phase's Stage-3 concept-synthesis pass. Source: `MCP/tools/pipeline_tools.py`, `lib/generation/llm_checkpoint.py`. |
| `ED4ALL_INTELLIGENCE_RUBRIC` | unset (off; auto-on for pipeline runs) | Produces the deterministic course capability report. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/aggregators/intelligence_level.py`. |
| `ED4ALL_CONTENT_PAGE_PER_CO` | unset (off) | Page-per-CO content-emit gate (resolver `lib/generation/content_page_budget.py::content_page_per_co_enabled`; call-site wrapper `MCP/tools/pipeline_tools.py::_content_page_per_co_enabled`,. Source: `lib/generation/content_page_budget.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_CONTENT_PAGE_NUM_CTX` | `4096` (→ `ED4ALL_ANSWER_NUM_CTX` → 4096) | Authoring serving-window token budget for the page-per-CO per-page chunk cap. Invalid or out-of-range values use the documented default. Source: `lib/generation/content_page_budget.py`. |
| `ED4ALL_CONTENT_PAGE_MAX_CHUNKS` | `5` | Hard top-K ceiling on chunks kept per CO page for the page-per-CO cap. Invalid or out-of-range values use the documented default. Source: `lib/generation/content_page_budget.py`. |
| `ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED` | unset (off) | Satellite of `ED4ALL_CONTENT_PAGE_PER_CO` — TRUE one-HTML-page-per-CO (resolver `lib/generation/content_page_budget.py::content_page_per_co_uncapped`, consumed by. Source: `lib/generation/content_page_budget.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_CHUNK_ROLE_DIVERSIFY` | unset (off; auto-on for pipeline runs) | Deterministically varies chunk ordering by block role during two-pass outline emission so co-located blocks do not all receive the same evidence order. Source: `lib/generation/content_page_budget.py`. |
| `ED4ALL_COURSE_IDENTITY_DEDUP` | unset (off) | BRAIN guard (resolver `lib/course_identity.py::resolve_course_identity` / `course_identity_dedup_enabled` / `cleanup_empty_skeletons`). Source: `lib/course_identity.py`, `docs/LICENSING.md`. |
| `ED4ALL_EMBEDDING_PROVIDER` | `st` | Selects the retrieval-index embedding backend from `lib/embedding/providers.py::_EMBEDDING_PROVIDERS` (`st` in-process sentence-transformers / `local-openai` local `/v1/embeddings` server /. Source: `lib/embedding/providers.py`. |
| `ED4ALL_EMBEDDING_MODEL` | per-provider | Overrides the model ID registered for the selected embedding provider. Source: `lib/embedding/providers.py`. |
| `ED4ALL_EMBEDDING_BASE_URL` | `<local-endpoint>` | Sets the endpoint for the `local-openai` embedding provider. Source: `lib/embedding/providers.py`. |
| `ED4ALL_EMBEDDING_API_KEY` | `local` | Sets the optional bearer token for the `local-openai` embedding provider. Source: `lib/embedding/providers.py`. |
| `ED4ALL_EMBEDDING_DEVICE` | `cuda` | Torch device for the in-process `st` provider — index builds, query encoding, and the validator-tier embedder. Source: `lib/embedding/providers.py`, `lib/embedding/sentence_embedder.py`. |
| `ED4ALL_EMBEDDING_DTYPE` | `fp32` | Encoder compute precision for the in-process `st` provider. Source: `lib/embedding/providers.py`, `docs/LICENSING.md`. |
| `ED4ALL_EMBEDDING_BATCH_SIZE` | `16` | Encode batch size for the embedding client (replay parameter, recorded in the index manifest). Source: `lib/embedding/providers.py`. |
| `ED4ALL_EMBEDDING_CLIENT_CACHE` | `1` (on) | Process-level resident embedding-client cache. Invalid or out-of-range values use the documented default. Source: `lib/embedding/providers.py`. |
| `ED4ALL_EMBEDDING_CLIENT_CACHE_MAX` | `2` | Satellite of `ED4ALL_EMBEDDING_CLIENT_CACHE` — max resident clients held by the LRU (`resolve_client_cache_max`). Invalid or out-of-range values use the documented default. Source: `docs/LICENSING.md`. |
| `ED4ALL_EMBEDDING_ALLOW_FAKE` | unset | Permits indexes created with the fake embedding provider to load in production read paths. Unset rejects them. Source: `lib/embedding/providers.py`. |
| `ED4ALL_EMBED_BATCH_TUNE` | unset (off) | Builder-C entailment-gate embed-throughput knob. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/embedding/sentence_embedder.py`, `lib/validators/feature_cache.py`. |
| `ED4ALL_EMBED_PERSIST_CACHE` | unset (off) | Builder-C disk-persisted embedding cache. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/embedding/sentence_embedder.py`. |
| `ED4ALL_EMBED_FP16` | unset (off) | Validator-tier only — this is NOT the index-build fp16 knob. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/embedding/sentence_embedder.py`, `docs/LICENSING.md`. |
| `ED4ALL_EVAL_CROSS_COURSE_NEGATIVES` | unset (off) | Additive eval arm. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/retrieval/cross_course_negatives.py`. |
| `ED4ALL_GATE_ADVISORY` | unset | Makes supported validation gates advisory. Unset preserves each gate’s configured severity. Source: `Trainforge/training/runner.py`. |
| `ED4ALL_GENERATION_TECHNIQUE` | `C5` | C0..C5 generation-technique selector resolved by `lib/generation/technique_modes.py::resolve_technique_mode`. Source: `lib/generation/technique_modes.py`. |
| `ED4ALL_GENERATION_CHECKPOINT` | `on` (default ON) | FAMILY flag for the fingerprinted LLM unit-checkpoint sidecars. Source: `lib/generation/llm_checkpoint.py`. |
| `ED4ALL_DYNAMIC_BLOCK_PLAN` | unset (off) | Uses the model-backed page block planner. Unset uses the static page-type plan. Source: `MCP/core/workflow_runner.py`. |
| `ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL` | per-provider | Model-ID override for the `ED4ALL_DYNAMIC_BLOCK_PLAN` planner. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER` | `nvidia` | SEAT selector. Source: `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_PLANNER_BLOOM_CLIMB` | unset (off) | Orders planned blocks from lower- to higher-order Bloom demands. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_PLANNER_LIFECYCLE` | unset (off) | Adds missing opening and consolidation blocks to a terminal-objective sequence. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_PLANNER_SPACING` | unset (off) | Spacing pass. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_PLANNER_BLOOM_CEILING` | unset (off) | Keeps planned block demands within each objective’s Bloom ceiling. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_PLANNER_FADING` | unset (off) | FR-INT-01 B08 guided-practice fading-sequence planner pass. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_WORKED_EXAMPLE_FLOOR` | unset (off; auto-on for pipeline runs) | DENSITY floor for the dynamic block planner. Source: `lib/generation/block_planner.py`, `lib/ontology/bloom.py`. |
| `ED4ALL_BLOOM_SPREAD_FLOOR` | unset (off; auto-on for pipeline runs) | Floor for the dynamic block planner. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_TRIANGLE_FLOOR` | unset (off; auto-on for pipeline runs) | Triangle) per-CO floor. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_RETRIEVAL_INTERLEAVE` | unset (off; auto-on for pipeline runs) | Interleaved retrieval) per-content-page floor. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_HOME` | unset (repo-relative) | Relocatable data root. Source: `lib/paths.py`. |
| `ED4ALL_IMSCC_MODULE_TITLES` | unset (off) | Uses terminal-objective text for IMSCC module titles. Unset uses the standard module-title formatter. Source: `Courseforge/scripts/cartridge/package_multifile_imscc.py`. |
| `ED4ALL_KEY_TERMS_PAGE` | unset (off; auto-on for pipeline runs) | Emits a deterministic key-terms page for each terminal objective. Source: `lib/generation/key_terms.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_NEW_BLOCK_TYPES` | unset (off) | Aligned pedagogical block types — `hook` (B02 activation), `multimedia` (B04, the mandatory time-based-media stack), `worked_example` (B05, subgoal labels + per-step Why + fade-state),. Source: `lib/generation/block_planner.py`, `Courseforge/scripts/rendering/generate_course.py`. |
| `ED4ALL_REFLECTION_CALIBRATION` | unset (off) | Calibrates reflection blocks to the associated objective and evidence. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; other values leave it off. Source: `lib/generation/reflection_calibration.py`, `Courseforge/scripts/blocks.py`. |
| `ED4ALL_REASONING_THINKING_OFF` | unset (off) | Disables reasoning-token output on the NVIDIA Nemotron-3 REASONING models ("detailed thinking off") for every COMPOSED OpenAI-compatible seat (Together / Local / Curriculum / Courseforge. Source: `Trainforge/generators/providers/_openai_compatible_client.py`. |
| `ED4ALL_RECALL_SELF_CHECK` | unset (off) | Free-recall / cloze self-check variant gate (recognition->recall retrieval-practice). Source: `lib/generation/`. |
| `ED4ALL_MISCONCEPTION_RICH` | unset (off) | Named subject-specific misconception + productive-failure (predict->reveal->reconcile) gate for the B03/B12 `misconception` block (resolver. Source: `lib/generation/misconception_rich.py`, `Courseforge/scripts/blocks.py`. |
| `ED4ALL_MAYER_CTML` | unset (off) | Mayer CTML (Cognitive Theory of Multimedia Learning) 12-principles structural check enriching the UDL/multimedia surface. Source: `lib/validators/mayer_ctml.py`. |
| `ED4ALL_BLOOM_DISTRIBUTION` | unset (off; auto-on for pipeline runs) | Course-level Bloom-distribution-vs-target-curve gate. Source: `lib/validators/bloom_distribution.py`. |
| `ED4ALL_BLOOM_DISTRIBUTION_TARGET` | unset (canonical default) | Operator override for the target Bloom curve consumed by `BloomDistributionValidator`. Invalid or out-of-range values use the documented default. Source: `lib/validators/bloom_distribution.py`. |
| `ED4ALL_BLOOM_DISTRIBUTION_TOLERANCE` | `0.20` | Float L1-deviation tolerance for the `BLOOM_DISTRIBUTION_OFF_TARGET` decision. Invalid or out-of-range values use the documented default. Source: `lib/validators/bloom_distribution.py`. |
| `ED4ALL_BLOOM_DISTRIBUTION_MIN_LOS` | `6` | Small-N objective floor for `BloomDistributionValidator`. Invalid or out-of-range values use the documented default. Source: `lib/validators/bloom_distribution.py`. |
| `ED4ALL_BLOOM_TRIVOTE` | unset (off) | Enables the alternate three-signal Bloom disagreement check in `lib/validators/bloom/classifier_disagreement.py::_validate_trivote`; resolved at validation time by. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/validators/bloom/classifier_disagreement.py`, `lib/classifiers/bloom_zero_shot.py`. |
| `ED4ALL_BLOOM_LADDER` | unset (off) | Bloom-ladder initiative keystone gate — makes the full per-objective Bloom ladder (every rung at/below the objective's OWN synthesized `bloom_level`, each carrying permitted `block_types` +. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/ontology/bloom_ladder.py`, `lib/generation/bloom_ladder_blocks.py`. |
| `ED4ALL_BLOOM_TRIVOTE_HEADS` | unset (off) | Optional, unprovisioned backend swap for voter 2 of `ED4ALL_BLOOM_TRIVOTE`. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/classifiers/bloom_zero_shot.py`. |
| `ED4ALL_BLOOM_HEADS_DIR` | `models/bloom_classifiers` | Local-only path used exclusively by `ED4ALL_BLOOM_TRIVOTE_HEADS`. Source: `lib/classifiers/bloom/`. |
| `ED4ALL_HARVEST_BLOOM_LABELS` | unset (off) | Post-build Bloom-label harvest hook in `MCP/core/workflow_runner.py` (best-effort, runs post-loop; any failure logs a warning and NEVER alters `final_status`). Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `MCP/core/workflow_runner.py`, `lib/bloom_labels/harvester.py`. |
| `ED4ALL_PREREQ_SEQUENCING` | unset (off) | Prerequisite-DAG-driven content sequencing. Source: `lib/generation/prereq_sequencer.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_PREREQ_TRANSITIVE_REDUCTION` | unset (off) | Deterministic stdlib DFS transitive reduction of the projected TO→TO prereq graph. Source: `lib/generation/prereq_sequencer.py`. |
| `ED4ALL_PREREQ_CENTRALITY_TIEBREAK` | unset (off) | Break for the TO topological sort. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/generation/prereq_sequencer.py`. |
| `ED4ALL_PREREQ_CENTRALITY_METHOD` | `in_degree` | Satellite selecting the centrality method consumed by `ED4ALL_PREREQ_CENTRALITY_TIEBREAK`. Invalid or out-of-range values use the documented default. Source: `lib/generation/prereq_sequencer.py`. |
| `ED4ALL_KG_PREREQ_HEALTH` | unset (off; auto-on for pipeline runs) | Adds prerequisite-graph health signals to knowledge-graph validation. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/validators/kg_quality.py`. |
| `ED4ALL_RICHER_VISUAL_SYSTEM` | unset (off) | Enables the richer course-page visual system. Truthy tokens enable it; false or unrecognized values leave it off. Source: `Courseforge/scripts/rendering/generate_course.py`. |

## Runtime paths and model execution

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_LIBV2_ROOT` | `<repo>/LibV2/` | Absolute path to the LibV2 root directory. Source: `lib/paths.py`. |
| `ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS` | `60` at the client; `300` at the content-generation providers | Per-request HTTP timeout (float seconds) for local content-generation LLM calls so a long authoring response is not capped at 60s. Source: `Trainforge/generators/providers/_openai_compatible_client.py`. |
| `ED4ALL_LLM_OMIT_OLLAMA_FORMAT` | unset (off) | Strict-OpenAI request-shape compatibility switch in `Trainforge/generators/providers/_openai_compatible_client.py`. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `Trainforge/generators/providers/_openai_compatible_client.py`, `docs/LICENSING.md`. |
| `ED4ALL_REASONING_LOW_EFFORT` | unset (off) | Enables the supported low-effort reasoning chat-template mode for compatible OpenAI-style model servers. Source: `lib/llm/`. |
| `ED4ALL_MAILBOX_BASE_DIR` | `<repo>/state/mailbox/` | Orchestrator task-mailbox base directory. Source: `lib/paths.py`. |
| `ED4ALL_NLI_DEVICE` | `cpu` | Torch device for the in-process NLI classifier. Source: `lib/classifiers/nli_classifier.py`. |
| `ED4ALL_NLI_MIN_FREE_VRAM_MIB` | `1024` | Free-device-memory floor, in MiB, checked before placing NLI on CUDA. Invalid or out-of-range values use the documented default. Source: `lib/classifiers/nli_classifier.py`. |
| `ED4ALL_NLI_EVICT_FOR_CUDA` | `true` (on) | Governs the VRAM-contention resolution strategy when `ED4ALL_NLI_DEVICE` resolves to cuda + cuda is available BUT free VRAM is below `ED4ALL_NLI_MIN_FREE_VRAM_MIB` (a resident local ollama. Source: `lib/llm/vram_reclaim.py`. |
| `ED4ALL_NLI_MICROBATCH` | unset (off) | Batches concurrent NLI scoring requests into bounded microbatches. Truthy tokens enable it; false or unrecognized values leave it off. Source: `lib/classifiers/nli_microbatch.py`, `Courseforge/router/router.py`. |
| `ED4ALL_NLI_MICROBATCH_MAX_PAIRS` | `64` | Satellite of `ED4ALL_NLI_MICROBATCH` — the max number of (premise, hypothesis) pairs the scorer coalesces into one drain / batched forward pass. Invalid or out-of-range values use the documented default. Source: `lib/classifiers/nli_microbatch.py`. |
| `ED4ALL_NLI_MICROBATCH_WINDOW_MS` | `10` | Satellite of `ED4ALL_NLI_MICROBATCH` — the collection window in milliseconds. Invalid or out-of-range values use the documented default. Source: `lib/classifiers/nli_microbatch.py`. |
| `ED4ALL_NLI_MICROBATCH_VALIDATORS` | unset (off) | (REPURPOSED from the earlier in-process thread dispatcher) — shards the STANDALONE `block_prose_entailment` gate's per-block NLI entailment scoring across a spawn `ProcessPoolExecutor`. Source: `lib/validators/block_prose_entailment.py`. |
| `ED4ALL_NLI_VALIDATORS_PROCS` | `4` | Persistent spawned-process count for validator NLI scoring. Invalid or out-of-range values use the documented default. Source: `lib/validators/block_prose_entailment.py`. |
| `ED4ALL_NLI_VALIDATORS_FACTORY` | unset (→ `default_nli_factory`) | Unset / blank → `default_nli_factory` (→ `NliClassifier.get_or_load`, the production singleton loader). Source: `lib/classifiers/nli.py`. |
| `ED4ALL_VALIDATION_CHECKPOINT` | `on` (site flag under `ED4ALL_GENERATION_CHECKPOINT`) | Writes resumable per-block checkpoints for `block_prose_entailment`. Falsey tokens disable this site while leaving other generation checkpoints unchanged. Source: `lib/generation/llm_checkpoint.py`, `lib/validators/block_prose_entailment.py`. |
| `ED4ALL_VALIDATION_FEATURE_CACHE` | unset (off) | Shares computed block features across validators within one gate run. Truthy tokens enable it; other values leave it off. Source: `lib/validators/feature_cache.py`, `MCP/core/executor.py`. |
| `ED4ALL_NLI_CROSSBLOCK` | unset (off) | Enables single-process concurrent NLI scoring across blocks in `block_prose_entailment`. Truthy tokens enable it; other values leave it off. Source: `lib/classifiers/nli_microbatch.py`. |
| `ED4ALL_NLI_CROSSBLOCK_THREADS` | `16` | Thread count for cache-miss blocks sent to the shared coalescing NLI dispatcher. Invalid or out-of-range values use the documented default. Source: `lib/classifiers/nli_microbatch.py`. |
| `ED4ALL_NLI_BUCKET_BATCHING` | unset (off) | Token-length bucketed batching for the NLI forward pass. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/classifiers/nli_classifier.py`. |
| `ED4ALL_NLI_BUCKET_BATCH` | `256,128,64,32` | Satellite of `ED4ALL_NLI_BUCKET_BATCHING` — per-bucket forward-pass batch sizes as a comma list aligned to the `<=128 / <=256 / <=384 / <=512`-token buckets, shortest first. Invalid or out-of-range values use the documented default. Source: `lib/classifiers/nli_classifier.py`, `runtime/scratchpad/nli_bucket_microbench.py`. |

## Validation and objective planning

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_OBJECTIVE_REVIEW_PROVIDER` | unset (off) | Grounding-safe objective-review pass gate. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_review.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_REVIEW_MODEL` | per-provider | Model-ID override for the objective-review pass. Source: `lib/objectives/objective_review.py`. |
| `ED4ALL_OBJECTIVE_CHUNK_RELEVANCE_FLOOR` | `0.30` | Dedup union prune. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_CITATION_RESELECT` | unset (off; auto-on for pipeline runs) | Deterministic post-hoc CO citation RE-SELECTION pass for stage-2 objective synthesis. Source: `lib/objectives/citation_reselect.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_RESELECT_EXERCISE_DEMOTE` | on (when reselect is on) | Exercise-chunk DEMOTION inside the `ED4ALL_OBJECTIVE_CITATION_RESELECT` ranking. Source: `lib/objectives/citation_reselect.py`, `docs/LICENSING.md`. |
| `ED4ALL_OBJECTIVE_RESELECT_KEEP_ORIGINAL` | on (when reselect is on) | Keep-original UNION guard inside the `ED4ALL_OBJECTIVE_CITATION_RESELECT` re-selection. Falsey tokens disable it; other values retain the enabled default. Source: `lib/objectives/citation_reselect.py`. |
| `ED4ALL_OBJECTIVE_SANITIZE_CITATIONS` | on (default) | Removes objective citations that do not resolve against the available chunk identifiers before objectives are written. Falsey tokens disable sanitation. Source: `lib/objectives/citation_sanitize.py`. |
| `ED4ALL_OBJECTIVE_ENTAILMENT_MATH_FOLD` | unset (off) | Math-representation FOLDING of both premise (cited chunk text) and hypothesis (LO statement) before NLI scoring inside the `objective_entailment` gate. Source: `lib/validators/objective_entailment.py`, `lib/semantik/math_fold.py`. |
| `ED4ALL_OBJECTIVE_DEDUP_THRESHOLD` | `0.88` | §4.2 cosine clustering threshold for the in-synthesis objective-dedup pass. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_DISTINCT_SKILL_SPLIT` | unset (off) | Prevents objective deduplication from merging distinct skills. Source: `lib/objectives/objective_dedup.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_DEDUP_LEXICAL` | unset (off; auto-on for pipeline runs) | CROSS-WINDOW lexical-dedup SECOND PASS. Source: `lib/objectives/objective_dedup.py`. |
| `ED4ALL_OBJECTIVE_DEDUP_LEXICAL_COSINE` | `0.78` | Centroid-cosine floor for a lexical merge EDGE. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`, `docs/LICENSING.md`. |
| `ED4ALL_OBJECTIVE_DEDUP_LEXICAL_JACCARD` | `0.60` | Best-grounded skill-signature Jaccard floor for a lexical merge EDGE. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`, `docs/LICENSING.md`. |
| `ED4ALL_OBJECTIVE_SPECIFICITY` | unset (off; auto-on for pipeline runs) | Gate for the CO-statement specificity/vacuity validator. Source: `lib/validators/objective_specificity.py`. |
| `ED4ALL_OBJECTIVE_WINDOW_PER_SECTION` | unset (off) | Vendor-depth per-SECTION stage-2 map units. Source: `lib/objectives/chunk_window.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_WINDOW_MAX_CANDIDATES` | unset (`1`–`3`) / `12` in per-section mode | Sets the objective-candidate budget for each synthesis window. Positive integers override the mode default; invalid or non-positive values use that default. Source: `Courseforge/generators/outline/_textbook_synthesis_provider.py`. |
| `ED4ALL_OBJECTIVE_SEED_SANITIZE` | unset (off) | Exercise-apparatus SEED SANITATION. Source: `lib/objectives/chunk_window.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_SYNTHESIS_SKELETON` | unset (off) | Structure-aware objective synthesis — inject a compact CONTEXT-ONLY document HEADING SKELETON of the window's chapter into the Stage-2 per-window synthesis prompt so TO/CO derivation is. Source: `Courseforge/generators/outline/_textbook_synthesis_provider.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_SOURCE_BACKFILL` | unset (off) | Adds grounded objectives for content-bearing chunks left uncovered after deduplication. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/objectives/source_backfill.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_BACKFILL_COVERAGE_TARGET` | `1.0` | Minimum content-bearing chunk coverage targeted by objective source backfill. Invalid or out-of-range values use the documented default. Source: `lib/objectives/source_backfill.py`. |
| `ED4ALL_OBJECTIVE_BLOOM_RELEVEL` | unset (off; auto-on for pipeline runs) | Recomputes objective Bloom labels from their observable verbs. Source: `lib/objectives/bloom_relevel.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT` | unset (off) | Adds grounded higher-order objectives when the configured distribution floor is unmet. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/objectives/bloom_complement.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MIN_SHARE` | `0.15` | Sets the minimum share of analyze, evaluate, and create objectives before Bloom complementation runs. Invalid values use `0.15`. Source: `lib/objectives/bloom_complement.py`. |
| `ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MAX` | `8` | Caps objectives added by Bloom complementation; `0` records signals without adding objectives. Invalid or negative values use `8`. Source: `lib/objectives/bloom_complement.py`. |
| `ED4ALL_OBJECTIVE_LIBRARY_EXEMPLARS` | unset (off) | Library EXEMPLARS master gate. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/objectives/library_exemplars.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_LIBRARY_EXEMPLAR_LIMIT` | `8` | K cap on surfaced exemplars. Invalid or out-of-range values use the documented default. Source: `lib/objectives/library_exemplars.py`. |
| `ED4ALL_OBJECTIVE_LIBRARY_EXEMPLAR_MIN_OVERLAP` | `0.05` | Jaccard floor for an exemplar to be surfaced. Invalid or out-of-range values use the documented default. Source: `lib/objectives/library_exemplars.py`. |
| `ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE` | `5` | K cap on cited chunks per MERGED objective. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT` | `on` (default ON) | Per-window resume-checkpoint sidecar for the stage-2 objective-synthesis pass. Source: `MCP/tools/pipeline_tools.py`, `lib/generation/llm_checkpoint.py`. |
| `ED4ALL_PLANNING_GATE_RETRIES` | `0` (off) | Sets the number of course-planning retries after a critical validation failure. Use a non-negative integer; invalid values use `0`. Task failures still stop the retry loop. Source: `MCP/core/workflow_runner.py`. |
| `ED4ALL_PLANNING_REROLL_SALT` | unset (runner-managed) | Per-attempt re-roll salt for the `ED4ALL_PLANNING_GATE_RETRIES` course_planning gate-retry loop — NOT an operator knob: `WorkflowRunner._retry_course_planning_gates` sets it to `attempt-N`. Source: `Courseforge/generators/outline/_textbook_synthesis_provider.py`. |
| `ED4ALL_PLANNING_REROLL_FEEDBACK` | unset (runner-managed) | Carries a compact digest of blocking planning-gate issues into a runner-managed retry. Operators should not set it; unset means no retry feedback. Source: `MCP/core/workflow_runner.py`. |
| `ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES` | unset (off) | Closed for the `archive_to_libv2` objectives→`objectives.json` plumbing. Source: `MCP/tools/pipeline_tools.py`. |

## Pipeline structure and provenance

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_PRODUCTION` | `0` | When `1`, enables production-mode FastMCP server settings. Source: `MCP/server.py`. |
| `ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE` | unset (off; auto-on for pipeline runs) | Gate-side provenance resolution for the `block_prose_entailment` NLI gate. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/validators/block_prose_entailment.py`. |
| `ED4ALL_RESEGMENT_COLLAPSED` | `1` | Re-segments a collapsed single-chapter structure into contiguous chapter groups. Source: `lib/semantic_structure_extractor/resegment.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_RESEGMENT_SECTIONS_PER_CHAPTER` | `13` | Target sections-per-pseudo-chapter (`resegment_collapsed_structure`). Invalid or out-of-range values use the documented default. Source: `docs/LICENSING.md`. |
| `ED4ALL_ROOT` | auto-detect | Absolute path to the Ed4All project root. Source: `lib/paths.py`. |
| `ED4ALL_RUN_ID` | generated | Per-run identifier consumed by every artifact emitter. Source: `MCP/core/workflow_runner.py`. |
| `ED4ALL_SKIP_ABLATION` | unset | When set, skips the post-training ablation pass. Source: `Trainforge/training/runner.py`. |
| `ED4ALL_STAGE_MODE` | `symlink` | How the `staging` phase materialises staged HTML (`copy` / `symlink` / `hardlink`). Source: `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_STATE_RUNS_DIR` | `<repo>/state/runs/` | State-runs directory. Source: `lib/paths.py`. |
| `ED4ALL_STRUCTURE_EXTRACT_GUARDS` | unset (off) | Enables guarded chapter and section assembly for DPUB-ARIA article structures. Source: `lib/semantic_structure_extractor/semantic_structure_extractor.py`, `lib/semantic_structure_extractor/tests/test_structure_extract_guards.py`. |
| `ED4ALL_STRUCTURE_OUTLINE_ANCHOR` | on-when-guards-on | Aligns extracted sections to the declared outline when structure guards are enabled. Source: `lib/semantic_structure_extractor/semantic_structure_extractor.py`. |
| `ED4ALL_TO_BACKLINK_FLOOR` | `0.45` cosine / `0.10` token | Weak-link floor for the deterministic CO→TO backlink. Source: `lib/ontology/lo_backlink.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_TO_BACKLINK_REASSIGN` | unset (off) | Reassigns weak course-objective backlinks using validator-aligned scoring. Source: `lib/ontology/lo_backlink.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_TO_CLUSTER_K` | `0` (auto) | Target-K for bottom-up TO derivation Ward agglomerative clustering. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_TO_CLUSTER_THRESHOLD` | `0.50` | Clustering threshold — now governs ONLY the no-sklearn single-link FALLBACK path for bottom-up TO derivation. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`, `docs/LICENSING.md`. |
| `ED4ALL_TO_COS_PER_CLUSTER` | `6` | Divisor for bottom-up TO derivation. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`, `docs/LICENSING.md`. |
| `ED4ALL_TO_CLUSTER_GUARDS` | unset (off) | Consolidates undersized terminal-objective clusters after clustering. Source: `lib/objectives/objective_dedup.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_TO_OUTLIER_MIN_SIZE` | `3` | A) min-cluster-size floor for the CONSOLIDATE pass. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`, `docs/LICENSING.md`. |
| `ED4ALL_TO_OUTLIER_ABSORB_FLOOR` | `0.20` | A) "has a clear home" centroid-cosine floor for the OUTLIER-absorb decision. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`. |
| `ED4ALL_TO_MERGE_NEAR_DUP` | unset (off) | Merges terminal-objective clusters whose centroid similarity meets the configured floor. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/objectives/objective_dedup.py`. |
| `ED4ALL_TO_MERGE_COSINE` | `0.85` | A) centroid-cosine merge floor for the near-duplicate-TO merge. Invalid or out-of-range values use the documented default. Source: `lib/objectives/objective_dedup.py`. |
| `ED4ALL_TO_ALLOW_SINGLETON_TO` | unset (off → singleton TOs dissolved) | Anti-hallucinated-TO backstop — the OPT-OUT for the unconditional `dissolve_singletons` pass. Source: `lib/objectives/objective_dedup.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_TO_MIN_CLUSTERS` | `3` | Tiny-course floor for the `dissolve_singletons` backstop. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/objectives/objective_dedup.py`. |
| `ED4ALL_TO_SOURCE_GROUNDING` | unset (off; auto-on for pipeline runs) | Objective source-grounding validator. Source: `lib/validators/terminal_objective_source_grounding.py`. |
| `ED4ALL_TO_CHAPTER_ANCHOR` | unset (off) | CHAPTER-ANCHORED terminal-objective derivation. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `MCP/tools/pipeline_tools.py`, `lib/objectives/chapter_anchor.py`. |
| `ED4ALL_TO_CHAPTER_ANCHOR_REORDER` | on-when-master-on | Reorders component objectives by their chapter anchor when `ED4ALL_TO_CHAPTER_ANCHOR` is active. Source: `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_TO_CHAPTER_MIN_MODULES` | `2` | To chapter min modules. Invalid or out-of-range values use the documented default. Source: `MCP/tools/pipeline_tools.py`, `docs/LICENSING.md`. |
| `ED4ALL_TO_CHAPTER_MIN_CO_COVERAGE` | `0.80` | Sets the minimum cited-course-objective coverage required for chapter-anchored terminal objectives. Invalid or out-of-range values use the documented default. Source: `MCP/tools/pipeline_tools.py`, `docs/LICENSING.md`. |
| `ED4ALL_TRAINING_CAPTURES_DIR` | `<repo>/training-captures/` | Sets the secondary decision-capture output directory. Source: `lib/paths.py`, `lib/decision_capture.py`. |
| `ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM` | unset (off) | Controls validator handling of CUDA out-of-memory errors. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/llm/oom.py`, `MCP/hardening/validation_gates.py`. |
| `ED4ALL_CALIB_EXTRA_CORPORA` | unset (off) | Adds comma-separated roots to calibration-artifact discovery. Unset performs only standard discovery; unreadable roots are skipped and reported. Source: `scripts/harness/calibration_harness.py`. |

## Resource and service lifecycle

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_VRAM_DOCTOR` | unset (off) | Enables best-effort device-memory and resident-model snapshots before and after workflow phases. Source: `lib/llm/vram_doctor.py`. |
| `ED4ALL_GPU_LIFECYCLE` | `on` | Controls model-release sweeps at successful phase and SemantiK stage boundaries. Falsey tokens disable it; other values retain the enabled default. Source: `lib/gpu_lifecycle.py`, `SemantiK/semantik_structure/gpu_lifecycle.py`. |
| `ED4ALL_BIG_MEMORY_MIN_MIB` | `49152` | Total-device-memory threshold, in MiB, for advisory `ed4all doctor` service-concurrency checks. Invalid or out-of-range values use the documented default. Source: `lib/diagnostics/gpu_profile.py`. |
| `ED4ALL_VLLM_CONTAINER_LIFECYCLE` | unset (off) | Enables best-effort start-at-need and workflow-end release of registered local model-service containers. Source: `lib/vllm_container_lifecycle.py`. |
| `ED4ALL_VLLM_CONTAINERS` | unset (`{}`) | Comma-separated `base_url=container` registry used by container lifecycle. Source: `lib/vllm_container_lifecycle.py`. |
| `ED4ALL_SEAT_SCHEDULE` | unset (off) | Enables per-phase reconciliation of logical model services declared in `config/workflows.yaml`. Source: `lib/vllm_container_lifecycle.py`. |
| `ED4ALL_SEAT_BASE_URLS` | unset (`{}`) | Comma-separated `logical_name=base_url` registry for scheduled services. Source: `lib/vllm_container_lifecycle.py`. |
| `ED4ALL_SEAT_LAUNCH_SPECS` | unset (`{}`) | Per-service launch-spec registry used for cold recreation after a failed coherence probe. Source: `lib/vllm_container_lifecycle.py`. |
| `ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS` | `1200` | Positive liveness-wait ceiling for scheduled local model services. Invalid or out-of-range values use the documented default. Source: `lib/vllm_container_lifecycle.py`. |
| `ED4ALL_SEAT_COHERENCE_ATTEMPTS` | `3` | Positive count of bounded content-coherence probes after liveness succeeds. Invalid or out-of-range values use the documented default. Source: `lib/vllm_container_lifecycle.py`. |
| `ED4ALL_LLM_TTFT_METER` | unset (off) | Uses streaming completions to record time to first token in LLM usage events. Truthy tokens enable metering; false or unrecognized values use non-streaming requests. Source: `lib/llm/`. |

## Planner, assessment, chunking, and retrieval

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_PLANNER_INTERLEAVE` | unset (off) | CO practice interleaving for the dynamic block planner. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_PLANNER_FAR_TRANSFER` | unset (off) | Transfer floor for the dynamic block planner. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_PLANNER_DUAL_CODING` | unset (off) | Coding floor for the dynamic block planner. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_PLANNER_INTEGRATION` | unset (off) | Integration floor for the dynamic block planner. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_PLANNER_INTEGRATION_MAX_OVERLAP` | `0.10` | Satellite of `ED4ALL_PLANNER_INTEGRATION` — the token-overlap floor below which a CO is weak-membership. Invalid or out-of-range values use the documented default. Source: `lib/generation/block_planner.py`, `lib/ontology/lo_backlink.py`. |
| `ED4ALL_PLANNER_INTEGRATION_MAX_PER_WEEK` | `2` | Satellite of `ED4ALL_PLANNER_INTEGRATION` — hard cap on integrated-practice injections per week. Invalid or out-of-range values use the documented default. Source: `lib/generation/block_planner.py`. |
| `ED4ALL_PLANNER_CROSS_WEEK_RETRIEVAL` | unset (off) | Week cumulative retrieval for the dynamic block planner — the ONE pass that needs prior-week context (the per-week planner is stateless), so it is a standalone helper. Source: `lib/generation/block_planner.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_FAQ_PAGE` | unset (off; auto-on for pipeline runs) | Week "FAQ" page gate (canonical resolver `lib/generation/faq_page.py::resolve_faq_page`; builder `build_faq_blocks`; renderer. Source: `lib/generation/faq_page.py`, `Courseforge/scripts/rendering/generate_course.py`. |
| `ED4ALL_FAQ_MAX_PER_PAGE` | `12` | Hard cap on FAQ cards emitted per week page. Invalid or out-of-range values use the documented default. Source: `lib/generation/faq_page.py`. |
| `ED4ALL_OBJECTIVE_OBSERVABLE_VERB` | unset (off) | Checks objectives without ABCD metadata for non-observable main verbs. Unset skips this additional scan. Source: `lib/validators/abcd_objective.py`. |
| `ED4ALL_OBJECTIVE_INFER_BLOOM` | unset (off) | When an ABCD-bearing LO has a null `bloom_level`, infer the level from the declared `behavior.verb` (reverse `BLOOMS_VERBS` `_VERB_TO_LEVEL` map; verbs unique across the 6 levels) instead. Source: `lib/validators/abcd_objective.py`, `docs/LICENSING.md`. |
| `ED4ALL_CHUNK_COVERAGE_FLOOR` | unset (off) | Coverage gate floor on the existing `chunkset_manifest` gate. Invalid or out-of-range values use the documented default. Source: `lib/validators/chunkset_manifest.py`. |
| `ED4ALL_MIN_CHUNKS` | unset (off) | Sets the minimum accepted chunk count. Unset, zero, invalid, or negative values disable this check. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_GATE` | unset (off) | Master switch for the pre-synthesis chunk + structure HEALTH gate. Source: `lib/validators/chunk_health.py`. |
| `ED4ALL_KEYTERM_DEF_QUALITY` | unset (off; auto-on for pipeline runs) | Keyterm def quality. Source: `lib/validators/key_terms_definition_quality.py`. |
| `ED4ALL_PAGE_EST_MINUTES` | unset (off) | Emits estimated reading time in page metadata and accessible HTML attributes. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; other values leave it off. Source: `lib/generation/content_page_budget.py`, `Courseforge/scripts/rendering/generate_course.py`. |
| `ED4ALL_PAGE_WPM` | `200` | Reading-speed (words/minute) divisor for the `ED4ALL_PAGE_EST_MINUTES` estimate. Invalid or out-of-range values use the documented default. Source: `lib/generation/content_page_budget.py`. |
| `ED4ALL_PAGE_INTERACTION_MINUTES` | `1.0` | Per-interaction minute cost for the `ED4ALL_PAGE_EST_MINUTES` estimate. Invalid or out-of-range values use the documented default. Source: `lib/generation/content_page_budget.py`. |
| `ED4ALL_GROUNDEDNESS_COMPUTATIONAL` | unset (off) | Exempt computational sentences on the grounded-answer path. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/retrieval/groundedness.py`. |
| `ED4ALL_GROUNDEDNESS_FRONTIER` | unset (off) | Frontier-batched early-stop stage-1 for the NLI groundedness scorer. Source: `lib/retrieval/groundedness.py`, `lib/retrieval/candidate_order.py`. |
| `ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH` | `4` | Sets the number of candidate passages scored per active claim in each frontier round. Invalid or non-positive values use `4`. Source: `lib/retrieval/`. |
| `ED4ALL_EMBED_OVERFLOW_GUARD` | `1` (on; report only) | Detects records beyond the embedding model's serving window and pins the model sequence limit. Source: `lib/embedding/providers.py`. |
| `ED4ALL_EMBED_OVERFLOW_SPLIT` | unset (off) | Splits overflow records when the overflow guard is on. Source: `lib/embedding/providers.py`. |
| `ED4ALL_EMBED_MAX_SEQ_TOKENS` | `512` | Sets the embedding serving-window ceiling when overflow protection is active. Invalid or out-of-range values use the documented default. Source: `lib/embedding/providers.py`. |
| `ED4ALL_CHUNK_CODE_SPLIT` | unset (off) | Enables code-fence-aware splitting for long code blocks. Source: `Trainforge/chunker/chunker.py`. |
| `ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR` | `0` (off) | Merges chunk fragments smaller than this floor. Invalid or out-of-range values use the documented default. Source: `Trainforge/chunker/chunker.py`. |
| `ED4ALL_CHUNK_SECTION_HARD_BREAK` | unset (off) | Prevents small sections from merging across section boundaries. Truthy tokens enable it; other values leave it off. Source: `Trainforge/chunker/chunker.py`. |
| `ED4ALL_CHUNK_SUBSECTION_BREAK` | unset (off) | Allows subsection headings to end a chunk after the configured word floor is reached. Truthy tokens enable it; other values leave it off. Source: `Trainforge/chunker/chunker.py`. |
| `ED4ALL_CHUNK_SUBSECTION_MIN_WORDS` | `250` | Satellite of `ED4ALL_CHUNK_SUBSECTION_BREAK` (`resolve_chunk_subsection_min_words`) — the accumulation floor in words a buffer must reach before a sub-section heading is allowed to flush it. Invalid or out-of-range values use the documented default. Source: `Trainforge/chunker/chunker.py`. |
| `ED4ALL_CHUNK_LO_HEURISTIC` | unset (off) | Heuristically links otherwise unlinked chunks to learning objectives after exact matching. Source: `lib/ontology/lo_heuristic_link.py`. |
| `ED4ALL_CROSS_COURSE_DEDUP` | unset (off) | Removes repeated boilerplate during multi-course batch imports. Source: `Trainforge/chunker/cross_course_dedup.py`. |
| `ED4ALL_CHUNK_DEDUP` | unset (off) | WITHIN-package exact-normalized duplicate elimination inside `Trainforge.chunker.chunker.chunk_content`. Source: `Trainforge/chunker/cross_course_dedup.py`, `docs/LICENSING.md`. |
| `ED4ALL_CHUNK_DEDUP_MIN_TOKENS` | `8` | Satellite of `ED4ALL_CHUNK_DEDUP` (`resolve_chunk_dedup_min_tokens`) — the exact-normalized token floor below which a repeated unit is NEVER dropped, protecting short-but-legitimate. Invalid or out-of-range values use the documented default. Source: `docs/LICENSING.md`. |
| `ED4ALL_HTML_PARSE_WORKERS` | `10` | Sets the process count for HTML parsing. `0` and `1` select serial parsing; invalid or negative values use `10`. Source: `Trainforge/`. |
| `ED4ALL_HTML_PARSE_START_METHOD` | `spawn` | Start method for that parse pool (`_resolve_html_parse_start_method`). Source: `docs/LICENSING.md`. |
| `ED4ALL_HTML_ASSET_REJECT` | `1` (on) | Rejects discovered files whose byte signature is not HTML before decoding or parsing. Falsey tokens disable the check; invalid values retain the enabled default. Source: `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_RETRIEVAL_BLAS_THREADS` | `8` | BLAS thread-team cap scoped to the semantic search GEMV ONLY. Invalid or out-of-range values use the documented default. Source: `LibV2/tools/libv2/retrieval/vector_index.py`, `docs/LICENSING.md`. |
| `ED4ALL_RETRIEVAL_TOPK_LEGACY` | unset (off) | Selects the full-sort top-k implementation instead of the partition-based implementation. Both preserve deterministic score and chunk-ID ordering. Source: `LibV2/tools/libv2/retrieval/vector_index.py`. |
| `ED4ALL_WITH_ASSESSMENT_SFT` | unset (off) | Includes verified assessment-derived examples in synthesized SFT data. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `Trainforge/synthesis/synthesize_training.py`, `Trainforge/generators/deterministic/assessment_sft_generator.py`. |
| `ED4ALL_WITH_GRAPH_SFT` | unset (off) | Includes graph-grounded examples in synthesized SFT data. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `Trainforge/synthesis/synthesize_training.py`, `Trainforge/generators/deterministic/graph_sft_generator.py`. |
| `ED4ALL_ASSESSMENT_APPLY_ARM` | unset (off) | Generates and verifies application-level assessment items. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `Trainforge/generators/assessment/generator.py`. |
| `ED4ALL_ASSESSMENT_APPLY_ARM_MAX` | `4` | Satellite of `ED4ALL_ASSESSMENT_APPLY_ARM` (`_apply_arm_max`) — bounded per-quiz apply-arm LLM draft budget. Invalid or out-of-range values use the documented default. Source: `Trainforge/generators/`. |
| `ED4ALL_ASSESSMENT_ITEM_TRIVOTE` | unset (off) | Checks asserted assessment Bloom levels against independent voters. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/validators/assessment_item_writing.py`. |
| `ED4ALL_ASSESSMENT_NUMERIC_RECOVERY` | unset (off) | Allows numeric fill-in-the-blank extraction from verified worked-example regions. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `Trainforge/generators/assessment/content_extractor.py`. |
| `ED4ALL_ASSESSMENT_APPARATUS_STRICT` | unset (off; auto-on for pipeline runs) | Widened GENERIC apparatus marker set for the assessment harvest paths. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `Trainforge/generators/assessment/content_extractor.py`. |
| `ED4ALL_ASSESSMENT_CLEAN_PROSE` | unset (off; auto-on for pipeline runs) | Prose-only MINING VIEW for the `assessment_synthesis` phase. Source: `lib/assessment/source_prose.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_ASSESSMENT_ITEM_BANK` | unset (off; auto-on for pipeline runs) | QTI 1.2 item-BANK sidecar. Source: `MCP/tools/pipeline_tools.py`, `Courseforge/scripts/cartridge/qti_emitter.py`. |
| `ED4ALL_ASSESSMENT_ITEMS_PER_OBJECTIVE` | `1` | Expansive item-bank SCALING knob. Source: `MCP/tools/pipeline_tools.py`, `docs/LICENSING.md`. |
| `ED4ALL_TRAINFORGE_ASSESSMENT_HARVEST` | unset (off) | Dependency inversion for the `trainforge_assessment` phase. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_DISCUSSION_GROUNDING_NLI` | unset (off) | Grounded NLI arm for the `discussion_assignment_grounded` validator. Truthy tokens (`1`, `true`, `yes`, `on`) enable it; false or unrecognized values leave it off. Source: `lib/validators/discussion_assignment_grounding.py`. |
| `ED4ALL_EVAL_COMPOSER_PROVIDER` | unset (absent) | Selects the optional evaluation answer-composition provider. Source: `lib/retrieval/answer_backend.py`, `docs/LICENSING.md`. |
| `ED4ALL_EVAL_COMPOSER_MODEL` | per-provider | Satellite of `ED4ALL_EVAL_COMPOSER_PROVIDER` (`ENV_EVAL_COMPOSER_MODEL`) — model-ID override for the diagnostic composer seat. Source: `Trainforge/generators/`. |

## Root-owned satellites

These live controls were historically described inside their master rows. They
are listed separately so defaults and fallback behavior remain searchable.

### Orchestrator timeouts and configuration

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_TASK_TIMEOUT_MINUTES` | `60` | Positive per-task timeout used when no explicit timeout is supplied. Invalid or out-of-range values use the documented default. Source: `MCP/core/executor.py`, `MCP/core/tests/test_executor_timeout_graceful.py`. |
| `ED4ALL_BATCH_TIMEOUT_MINUTES` | `30` | Positive whole-phase fallback when the phase has no YAML `batch_timeout_minutes`; explicit per-phase YAML wins. Invalid or out-of-range values use the documented default. Source: `MCP/core/executor.py`, `MCP/tests/test_phase_batch_timeout_plumbing.py`. |
| `ED4ALL_ENDPOINTS_PATH` | tracked `config/endpoints.yaml` | Path override for the endpoint registry. Source: `lib/paths.py`. |
| `ED4ALL_TOKEN_STATS_EXPORT` | unset | Optional path to a private aggregate token-statistics export consumed by `scripts/ops/update_development_tokens.py`. Source: `scripts/ops/update_development_tokens.py`. |
| `ED4ALL_PRIVATE_TOKEN_FILE` | unset | Optional private-token vocabulary read by `ci/guards/repository_policy.py`. Source: `ci/guards/repository_policy.py`. |

### Shared request admission

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_CLOUD_RPM` | unset (axis off) | Positive requests-per-minute ceiling used only when `ED4ALL_CLOUD_RATE_LIMIT` is enabled. Invalid or out-of-range values use the documented default. Source: `lib/llm/rate_limiter.py`. |
| `ED4ALL_CLOUD_TPM` | unset (axis off) | Positive tokens-per-minute ceiling used only when `ED4ALL_CLOUD_RATE_LIMIT` is enabled. Invalid or out-of-range values use the documented default. Source: `lib/llm/rate_limiter.py`. |
| `ED4ALL_CLOUD_MAX_CONCURRENCY` | unset (axis off) | Positive concurrent-request ceiling used only when `ED4ALL_CLOUD_RATE_LIMIT` is enabled. Invalid or out-of-range values use the documented default. Source: `lib/llm/rate_limiter.py`, `lib/llm/tests/test_rate_limiter.py`. |

### Retrieval reranking

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_RERANK_MODEL` | provider default | Overrides the registered reranker model. Source: `lib/retrieval/reranker.py`. |
| `ED4ALL_RERANK_CANDIDATE_POOL` | `30` | Positive first-stage candidate count. Invalid or out-of-range values use the documented default. Source: `lib/retrieval/reranker.py`. |
| `ED4ALL_RERANK_DEVICE` | `cpu` | Device passed to the registered reranker implementation. Source: `lib/retrieval/reranker.py`. |
| `ED4ALL_RERANK_BATCH_SIZE` | `16` | Positive cross-encoder batch size. Invalid or out-of-range values use the documented default. Source: `lib/retrieval/reranker.py`. |
| `ED4ALL_RERANK_ALLOW_FAKE` | unset (off) | Truthiness permits the deterministic fake reranker for tests. Source: `lib/retrieval/reranker.py`, `lib/tests/test_reranker.py`. |

### Chunk-health thresholds

All threshold overrides below are no-ops while `ED4ALL_CHUNK_HEALTH_GATE` is
off. Non-numeric or non-positive input resolves to the listed default. The
validator and tests are the source of truth:
`lib/validators/chunk_health.py` and
`lib/validators/tests/test_chunk_health_validator.py`.

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_CHUNK_HEALTH_CHAPTER_RATIO_FAIL` | `3.0` | Critical chapter-to-source ratio. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_CHAPTER_RATIO_WARN` | `1.5` | Warning chapter-to-source ratio. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_SECTIONS_PER_CHAPTER` | `15` | Expected sections per chapter used by structure-expansion checks. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_APPARATUS_FAIL` | `0.30` | Critical apparatus-heading share. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_APPARATUS_WARN` | `0.10` | Warning apparatus-heading share. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_MIN_CHUNKS` | `30` | Thin-chunkset warning floor. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_INSTRUCTIONAL_FAIL` | `0.20` | Critical instructional-content share. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_INSTRUCTIONAL_WARN` | `0.40` | Warning instructional-content share. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_TINY_WORDS` | `20` | Tiny-chunk word boundary. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_MEGA_WORDS` | `1500` | Oversized-chunk word boundary. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_MOJIBAKE_RATE` | `0.10` | Mojibake warning rate. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_NUMDUMP_RATE` | `0.15` | Numeric-dump warning rate. Source: `lib/validators/chunk_health_validator.py`. |
| `ED4ALL_CHUNK_HEALTH_WORKED_EXAMPLE_INSTRUCTIONAL` | on | Counts worked examples carrying solution markers as instructional. Falsey tokens disable it; other values retain the enabled default. Source: `lib/validators/chunk_health_validator.py`. |

### Shell service guard

| Flag | Default | Current contract |
|---|---|---|
| `ED4ALL_GPU_MAX_USED_MB` | `1500` | Used-device-memory ceiling for `scripts/ops/gpu_guard.sh`. Source: `scripts/ops/gpu_guard.sh`. |
| `ED4ALL_GPU_POLL_SECONDS` | `15` | Positive polling interval for the shell service guard. Source: `scripts/ops/gpu_guard.sh`. |
| `ED4ALL_GPU_TIMEOUT_SECONDS` | `0` | Maximum shell-guard wait in seconds; `0` means no timeout. Source: `scripts/ops/gpu_guard.sh`. |
| `ED4ALL_GPU_LOCK` | system temporary lock path | Override for the inter-process lock used by the shell service guard. Source: `scripts/ops/gpu_guard.sh`. |
| `ED4ALL_GPU_TASK` | `gpu-task` | Human-readable task label shown by the shell service guard; `--task` takes precedence. Source: `scripts/ops/gpu_guard.sh`. |

## Separately owned controls

- Courseforge owns `ED4ALL_REWRITE_FIT_WINDOW`, `ED4ALL_REWRITE_NUM_CTX`,
  `ED4ALL_REWRITE_TRUNCATION_TRIPWIRE`, and
  `ED4ALL_REWRITE_MAX_ESCALATION_SHARE`; see
  [Courseforge behavior flags](behavior-flags-courseforge.md).
- Trainforge owns `ED4ALL_ASSESSMENT_PROSE_PROVIDER` and
  `ED4ALL_ASSESSMENT_DIVERSIFIED`; see
  [Trainforge behavior flags](behavior-flags-trainforge.md).
- GUI binding and authentication controls—`ED4ALL_GUI_HOST`,
  `ED4ALL_GUI_PORT`, `ED4ALL_GUI_LEARNER`, `ED4ALL_GUI_MODE`, and
  `ED4ALL_GUI_TOKEN`—are documented in [`gui/README.md`](../../gui/README.md).
- Test-only controls remain beside their owning tests:
  `ED4ALL_A11Y_SMOKE_OLLAMA`, `ED4ALL_ARCHIVE_FIXTURE_SLUG`,
  `ED4ALL_STUDY_PACK_FIXTURE_SLUG`, `ED4ALL_MOODLE_IMAGE`,
  `ED4ALL_MOODLE_BOOT_TIMEOUT`, and `ED4ALL_MOODLE_SMOKE_IMSCC`.

## Related runtime variables

`LLM_MODE`, `LLM_PROVIDER`, and `LLM_MODEL` are CLI routing inputs documented in
the root [`CLAUDE.md`](../../CLAUDE.md). GUI binding and authentication
variables are documented in [`gui/README.md`](../../gui/README.md). Logging
verbosity is controlled by `ED4ALL_LOG_LEVEL` and changes console output only.
Test-only environment controls remain documented beside their tests and are not
production behavior flags.

For stop, checkpoint, and resume behavior, see
[Pipeline invocation](pipeline-invocation.md). Validation ownership and current
gate severities live in [Validation gates](../validation/gates.md).
