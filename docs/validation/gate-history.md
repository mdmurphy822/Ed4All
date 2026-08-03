# Validation-Gate Landing History

Per-wave changelog of validation-gate additions, demotions, and severity
flips. Each entry is the narrative that accompanied a gate landing, relocated
verbatim from the root `CLAUDE.md` "Active Gates" section so that file stays
focused on the current authoritative counts.

> **Authoritative current counts live elsewhere.** The running subtotals inside
> each landing note below were correct *at the moment that wave landed* and were
> not re-derived as later waves added gates — so an intermediate subtotal here
> (e.g. the IB7 note's "→101 warning") will NOT match the present total. The
> single source of truth for the current per-workflow critical/warning/total
> counts is the summary table in `CLAUDE.md` § "Active Gates", re-derived from
> `config/workflows.yaml`, plus the per-gate detail in `docs/validation/gates.md`.
> (Do not restate the aggregate here — an inline number drifts. The prior
> restatement "62 critical / 105 warning / 167 total" was already stale by the
> time flip-wave-2 landed; consult `CLAUDE.md` for the live count.) The deltas
> below are kept for per-wave provenance only; do not sum them to recover the
> current total.

---

## Landing notes

> IB6 landing (eight-dimension quality-rubric scoring capstone): five new
> block-quality gates added in **warning** at BOTH `inter_tier_validation` and
> `post_rewrite_validation` in `course_generation` + `textbook_to_course`
> (+10 warning each) — `block_cognitive_load` (IB6.4 per-block <=~200-char body
> ceiling), `anatomy_slot_presence` (IB6.2 six-slot presence), `interaction_
> feedback` (IB6.3 universal feedback presence/elaboration), `block_quality_
> rubric` (IB6.1 the 8-dim 0-3 scorer), `qa_checklist` (IB6.7 15-point QA).
> Each gate's validator no-ops (`passed=True` + info issue) when the keystone
> flag `ED4ALL_BLOCK_QUALITY_RUBRIC` is unset, so default-off runs are
> byte-stable even with the gates wired. The IB6.5 B01 both-axes
> (cognitive-process × knowledge-type) check rides on the existing
> `abcd_verb_alignment` gate at `course_planning` (no new gate). The hard-gate
> critical flip (mean>=2.0 floors + Accessibility=0 block-fail) is DEFERRED
> behind `# TODO(calibration)` markers until the anchored 0-3 scale is
> calibrated on >=2 corpora. Counts re-derived from `config/workflows.yaml`:
> `course_generation` 15→25 warning, `textbook_to_course` 59→69 warning,
> Total 77→97 warning / 143→163 total.

> IB7 landing (planner pedagogy — Bloom-climb / lifecycle / spacing / type-range):
> two new advisory→gate validators added in **warning** at BOTH
> `inter_tier_validation` and `post_rewrite_validation` in `course_generation`
> + `textbook_to_course` (+2 warning effective per workflow; only the fired
> phase counts per run, mirroring the W3 manifest-completeness wording) —
> `retrieval_presence` (IB7.5b: every content-bearing module carries ≥1 SPACED
> low-stakes retrieval block, `self_check_question`/`reflection_prompt`, not the
> page's first block) + `bloom_type_range` (IB7.6c: a block whose target Bloom
> exceeds its catalog `bloom_ceiling` is flagged — the advisory `bloom_fit`
> becomes a gate). Both warning-day-1 with `# TODO(calibration)` markers (the
> WS3/W4 deferred-flip pattern; IB3 is the single documented fastest-flip
> exception, this is NOT IB3). The PLANNER side of IB7 (the
> `ED4ALL_PLANNER_BLOOM_CLIMB` / `_LIFECYCLE` / `_SPACING` / `_BLOOM_CEILING`
> passes in `lib/generation/block_planner.py`) makes the planner *produce*
> framework-shaped sequences rather than relying on these validators to *catch*
> violations; all four planner flags default OFF (byte-stable). Counts
> re-derived from `config/workflows.yaml`: `course_generation` 25→27 warning,
> `textbook_to_course` 69→71 warning, Total 97→101 warning / 163→167 total.
> (NOTE: this "→101 warning / 167 total" was the running subtotal at IB7's
> landing; later landings below — counted here in reverse-chronological order —
> brought the current authoritative warning total to 105 / 167 total. See the
> reconciliation note at the top of this file.)

> W4 SHADOW landing (NLI grounding gates): `rewrite_source_grounding` DEMOTED
> critical → warning in `course_generation` + `textbook_to_course`
> (post_rewrite_validation); two NLI gates added in **shadow/warning** —
> `block_prose_entailment` (post_rewrite_validation, both workflows) +
> `objective_entailment` (course_planning, textbook_to_course). The
> calibration-gated critical flip (which also promotes `claim_support` to
> critical) is DEFERRED — see the `# TODO(calibration)` markers in
> `config/workflows.yaml` and the operator-local `w4-nli-grounding-gate` plan §4.
> The count table is re-derived again at the critical-flip landing.

> WS3 landing (CO↔TO semantic-alignment detection gate): `co_terminal_alignment`
> added in **warning** at `textbook_to_course::course_planning` (+1 warning).
> Recomputes `cosine(co.statement, assigned_to.statement)` per chapter
> objective to close the structural-roll-up silent-pass loophole that
> `terminal_objective_coverage` (a roll-up check only) cannot detect.
> DETECTION not cure (WS1's bottom-up TO derivation is the cure); fires on
> ~80%+ of COs on the broken run, so it lands warning day-1. The
> calibration-gated critical flip (which also promotes
> `CO_TERMINAL_WEAK_LINK_RATE_HIGH` to critical) is DEFERRED — see the
> `# TODO(calibration)` marker at the gate in `config/workflows.yaml`; flips
> to critical only after WS1 proves the recomputed weak-link rate ≤0.10 on
> ≥2 corpora. The count table is re-derived again at the critical-flip landing.

> WS6a landing (source→objective coverage-audit gate): `source_coverage`
> added in **warning** at `textbook_to_course::course_planning` (+1 warning).
> The symmetric companion to `co_terminal_alignment` — embeds each
> content-bearing textbook section and asserts ≥1 synthesized objective
> (CO or TO) covers it above a cosine floor (default 0.45), flagging
> `SOURCE_SECTION_UNCOVERED` + the aggregate `SOURCE_COVERAGE_LOW`. A section
> no objective covers is source material downstream content generation never
> authors a page for. MEASUREMENT guardrail: on the real corpus 135/135
> content sections are covered at floor 0.45 (mean cosine 0.73), so it passes
> clean — the gate guards OTHER corpora. The calibration-gated critical flip
> (which calibrates the floor up toward ~0.55 on ≥2 corpora and promotes
> `SOURCE_COVERAGE_LOW` to critical) is DEFERRED — see the
> `# TODO(calibration)` marker at the gate in `config/workflows.yaml`. The
> count table is re-derived again at the critical-flip landing.

> W3 manifest-completeness landing: the `manifest_completeness` gate
> (per-block synthesis-manifest RESOLUTION) added in **shadow/warning** at
> BOTH `content_generation` (single-pass) and `post_rewrite_validation`
> (two-pass) in `course_generation` + `textbook_to_course` (+2 warning each;
> only one of the two phases fires per run via `enabled_when_env`). Day-1
> warning until a live 7B run proves the rewrite-tier emission hook
> (`_emit_block_synthesis_manifest`, gated by `COURSEFORGE_EMIT_BLOCKS`); flip
> to critical at the `# TODO(integration)` markers in `config/workflows.yaml`.

> Numeric-literal grounding landing (math-fabrication control): the
> `numeric_literal_grounding` gate added in **shadow/warning** at
> `post_rewrite_validation` in BOTH `course_generation` + `textbook_to_course`
> (+1 warning each). It is the fabrication control for NUMERIC / math content
> the number-blind NLI gate (`block_prose_entailment`) cannot provide —
> established this session (the operator-local `content-block-quality-2026-06` plan
> iters 5/5b): DeBERTa-v3-mnli scores the fabrication `-40/88 = -5/11` (absent
> from all 72 real chunks) ABOVE every grounded math claim, and
> `groundedness._is_computational` EXEMPTS such claims. The gate cross-checks
> each prose fraction's `num/denom` pair against the cited source under
> OCR-tolerant containment (`lib/validators/numeric_literal_grounding.py`,
> reusing the `ED4ALL_ANSWER_NLI_ADD` numeric precedent in
> `lib/retrieval/citation_attribution.py`). Measured CLEAN on a real
> algebra corpus (grounded blocks 0% source-absent including
> OCR-flattened + computed results; the 2 fabrication blocks flag; zero false
> positives). Day-1 warning; flip to critical at the `# TODO(calibration)`
> markers in `config/workflows.yaml` after a ≥2-corpus FP-rate measurement
> (the per-block `source_absent_ratio` is on every decision event so the flip
> can move to a ratio floor if a later corpus shows computed-result FPs).

> W10 landing (assessment surface — QTI quizzes + assignments + discussions):
> a new pre-packaging `assessment_synthesis` phase added BEFORE `packaging` in
> BOTH `course_generation` + `textbook_to_course` (plan
> the operator-local `w10-assessments-qti-discussions-2026-06` plan §2.4 Option A).
> Validator-only phase (`agents: []`) routed by phase NAME to
> `run_assessment_synthesis` via `_PHASE_TOOL_MAPPING`. Three gates wired on it
> (+2 critical, +1 warning per workflow): `qti_well_formed`
> (`lib.validators.qti_well_formed.QtiWellFormedValidator`, **critical**/block —
> QTI 1.2 well-formedness + XSD conformance + answer-key resolution; input
> router feeds `inputs["qti_dir"] = <export>/06_assessments`) +
> `assessment_objective_alignment` (REUSE of the existing critical grounding
> validator on the new PRODUCT-assessment surface so synthesized
> quiz/assignment/discussion `objective_id`s must each resolve to a chunk's
> `learning_outcome_refs[]` ∪ the synthesized objectives' id set) +
> `discussion_assignment_grounded` (**warning** day-1 — per-TYPE grounding
> attribution for assignment/discussion items; no dedicated per-type validator
> class exists yet, so it is wired to the SAME
> `AssessmentObjectiveAlignmentValidator` at warning, mirroring the WS3
> deferred-flip pattern — see the `# TODO(validator)` marker at the gate in
> `config/workflows.yaml`; flip the validator path + severity to critical when
> the per-type validator lands). The count table above is re-derived from
> `config/workflows.yaml`: `course_generation` 15→17 critical / 8→9 warning,
> `textbook_to_course` 37→39 critical / 49→50 warning, Total 62→66 critical /
> 60→62 warning. The QTI emitter, the `run_assessment_synthesis` handler, and
> the Studio renderers land in sibling waves (Phases 1–6 of the plan); this
> landing is the workflow + gate wiring (Phase 5) only.

> IB3 landing (constructive-alignment verb-triple keystone): two NEW gates
> wired at BOTH `inter_tier_validation` + `post_rewrite_validation` in BOTH
> `course_generation` + `textbook_to_course` (+4 warning each):
> `outline_anchored_rubric` / `rewrite_anchored_rubric`
> (`lib.validators.alignment.anchored_rubric.AnchoredRubricValidator`) +
> `outline_triangle_completeness` / `rewrite_triangle_completeness`
> (`lib.validators.alignment.triangle_completeness.TriangleCompletenessValidator`).
> A third behavior — the IB3.2 verb-triple EQUALITY axis
> (`BLOCK_OBJECTIVE_VERB_TRIPLE_MISMATCH` + the `alignment_cap_at_1` signal) —
> rides the already-wired `block_objective_delivery` gate rows, and the IB3.3
> evidence-form check (`ASSESSMENT_EVIDENCE_FORM_TOO_LOW`) rides the existing
> critical `assessment_objective_alignment` gate rows; both are gated by the
> default-OFF `ED4ALL_ALIGNMENT_VERB_TRIPLE` flag inside the validators (no new
> gate row). All five behaviors are **warning day-1** but carry the
> roadmap-documented **ACCELERATED fast-flip** `# TODO(calibration)` marker:
> unlike the multi-wave deferral the other gates take, the verb-triple
> mismatch flips to critical after ONE ≥2-corpus FP measurement clears (a
> Create objective checked by a recall MCQ is a malformed block, not a style
> choice — the false-negative cost exceeds the false-positive cost). The flip
> also waits until IB1's `interaction` slot is populated (else the verb signal
> is prose-heuristic). The count table above is re-derived from
> `config/workflows.yaml`: `course_generation` 9→13 warning,
> `textbook_to_course` 50→54 warning, Total 62→70 warning / 128→136 total.

> IB4 landing (per-block WCAG 2.2 AA + UDL multiple-means contracts): moves
> WCAG 2.2 AA from a page-level retrofit to per-block-type contracts, gates the
> data-only chunk WCAG fields, closes the missing `textbook_to_course`
> packaging WCAG gate, and adds UDL multiple-means coverage fields + a
> validator. THREE new gates wired (all **warning day-1** with a deferred
> `# TODO(calibration)` critical-flip marker): `udl_coverage`
> (`lib.validators.udl_coverage.UdlCoverageValidator`) at
> `inter_tier_validation` + `post_rewrite_validation` in BOTH
> `course_generation` (+2 warning) and `textbook_to_course` (+2 warning);
> `chunk_wcag_status` (`lib.validators.chunk_wcag_status.ChunkWcagStatusValidator`)
> at `chunking` + `imscc_chunking` in `textbook_to_course` (+2 warning); and
> `wcag_compliance` (reusing an existing WCAG validator) at
> `textbook_to_course` `packaging` (+1 warning). The IB4.1 per-block
> WCAG sub-check (`REWRITE_BLOCK_A11Y_CONTRACT`) is a WARNING sub-issue of the
> existing critical `rewrite_html_shape` gate (NOT a new gate → no count
> change) and no-ops when `ED4ALL_BLOCK_A11Y` is unset; the per-block
> `Accessibility=0` hard-fail rollup (framework 6.4) is DEFERRED to IB6. The
> three new `Block` UDL fields (`n_representations` / `response_formats` /
> `engagement_affordance`) are Optional, hash-EXCLUDED, and emitted to
> HTML/JSON-LD only behind the default-OFF `ED4ALL_BLOCK_A11Y` flag (byte-stable
> when off, mirroring the IB1 anatomy posture). UNLIKE IB3, IB4 takes the
> standard multi-wave deferred-flip (IB3 is the roadmap's single documented
> fastest-flip exception). The count table above is re-derived from
> `config/workflows.yaml`: `course_generation` 13→15 warning,
> `textbook_to_course` 54→59 warning, Total 70→77 warning / 136→143 total.

> B15 landing (Resources block type — closes the last framework catalog gap):
> adds the `resources` Ed4All block type (B15 Resources / Further Reading), the
> only canonical B-code that previously had no Ed4All primary. ONE new gate
> wired in **warning** at `post_rewrite_validation` in BOTH `course_generation`
> (+1 warning) and `textbook_to_course` (+1 warning): `resource_link_purpose`
> (`lib.validators.resource_link_purpose.ResourceLinkPurposeValidator`) — the
> WCAG 2.4.4 Link Purpose check that flags any link in a `resources` block whose
> text is non-descriptive (bare URL / "click here" / "read more" / empty) as
> `RESOURCE_LINK_PURPOSE_UNCLEAR` (`action=regenerate`). The validator no-ops
> (`passed=True` + an info issue) when `ED4ALL_NEW_BLOCK_TYPES` is unset — the
> same flag that makes the `resources` type selectable (planner content-shape
> nudge) + renderable (`_render_resources_section` returns `""` when off), so
> default runs are byte-stable. Warning-day-1 with a deferred
> `# TODO(calibration)` critical-flip marker (the standard multi-wave deferred-
> flip; IB3 is the roadmap's single documented fastest-flip exception, this is
> NOT IB3). The count table is re-derived from `config/workflows.yaml`:
> `course_generation` 29→30 warning, `textbook_to_course` 73→74 warning, Total
> 105→107 warning / 167→169 total.

> IB6.6 rollup-GATE landing (framework FR-07/13, §6.5 — the block→module→course
> quality rollup finally GATES): the `BlockQualityRollupAggregator` already
> computed `course_pass` / per-block-fails (the 3 hard gates — Accessibility=0→
> block fail, assessment-Bloom<objective→Alignment cap-1, interaction-without-
> feedback→Feedback+Coherence cap-1 — plus the mean≥2.0 AND per-dimension
> min-floor paths) but ran ONLY as a best-effort post-loop aggregator that wrote
> `block_quality_rollup_report.json` and never touched `final_status`. ONE new
> gate wired in **warning** at `post_rewrite_validation` (two-pass) in BOTH
> `course_generation` (+1 warning) and `textbook_to_course` (+1 warning):
> `block_quality_rollup`
> (`lib.validators.block_quality_rollup.BlockQualityRollupValidator`). The gate
> is SELF-SUFFICIENT — within a single phase the in-flight sibling
> `block_quality_rubric` GateResult is not yet visible through `phase_outputs`
> (the `_gate_results` chain is stashed only AFTER the phase completes), so the
> gate scores the rewrite-tier `blocks` itself by delegating to the canonical
> IB6.1 `BlockQualityRubricValidator` (the single owner of the anchored 0-3
> scoring logic — no second scorer), then feeds the scored rows to the
> `BlockQualityRollupAggregator` (the single owner of the rollup math) and
> returns a GateResult enumerating `COURSE_QUALITY_ROLLUP_FAIL` /
> `BLOCK_ACCESSIBILITY_GATE_FAIL` / `BLOCK_QUALITY_ROLLUP_BELOW_FLOOR` /
> `MODULE_QUALITY_ROLLUP_FAIL`. The builder reuses the rewrite-tier
> `_build_block_input_rewrite` shim (`MCP/hardening/gate_input_routing.py`). The
> whole path no-ops (`passed=True` + a `RUBRIC_DISABLED` info issue, no scoring,
> no rollup) when the keystone `ED4ALL_BLOCK_QUALITY_RUBRIC` flag is unset →
> default-off runs are byte-identical. **Warning-day-1**: the gate computes
> `course_pass` but returns `passed=True` unconditionally so it does NOT yet
> change `final_status`; the `# TODO(calibration)` critical-flip (flip the YAML
> row to `severity: critical` + `behavior.on_fail: block` AND set
> `passed=course_pass` in the validator so a failing rollup BLOCKS promotion,
> FR-07/13) is DEFERRED until `scripts/harness/calibration_harness.py` confirms the
> rubric gates' FP rate on ≥2 corpora (the anchored 0-3 scale must be calibrated
> before the mean/min-floor hard gates can block early runs; standard multi-wave
> deferred-flip — IB3 is the roadmap's single documented fastest-flip exception,
> this is NOT IB3). The count table is re-derived from `config/workflows.yaml`:
> `course_generation` 30→31 warning, `textbook_to_course` 74→75 warning, Total
> 107→109 warning / 169→171 total.

> FR-A11Y-02 / FR-COURSE-02 / FR-COURSE-03 landing (framework a11y +
> course-structure gates): THREE new validators wired in **warning** across the
> two two-pass workflows (+3 warning each → +6 warning total). (1)
> `interactive_a11y`
> (`lib.validators.interactive_a11y.InteractiveA11yValidator`) at
> `post_rewrite_validation` in BOTH `course_generation` (+1) +
> `textbook_to_course` (+1) — interaction-block WCAG 2.1.1/2.5.7
> (`DRAG_ONLY_NO_KEYBOARD`) + 1.4.1 (`COLOR_ONLY_SIGNALING`); reuses the
> rewrite-tier `_build_block_input_rewrite` builder; rides `ED4ALL_BLOCK_A11Y`
> (strict no-op + byte-stable when the flag is unset, mirroring IB4's per-block
> a11y emit). (2) `block_sequence_order`
> (`lib.validators.block_sequence_order.BlockSequenceOrderValidator`) at
> `inter_tier_validation` in BOTH `course_generation` (+1) +
> `textbook_to_course` (+1) — cloned from `RetrievalPresenceValidator` (same
> `inputs['blocks']` shape + no-flag posture); flags worked_example(B05)→
> guided-practice(B08) out of order (`WORKED_BEFORE_GUIDED_OUT_OF_ORDER`), a
> B07 check massed against same-objective exposition (`CHECK_NOT_SPACED`), and a
> B09 scenario opening a TO (`SCENARIO_OPENS_TO`); reuses
> `_build_block_input_outline`. (3) `cumulative_assessment`
> (`lib.validators.cumulative_assessment.CumulativeAssessmentValidator`) at
> `assessment_synthesis` in BOTH `course_generation` (+1) +
> `textbook_to_course` (+1) — flags `CUMULATIVE_RETRIEVAL_TOO_NARROW` when the
> final graded assessment (B14) spans < 2 terminal objectives while the course
> defines ≥ 4 TOs; strict no-op when < 4 TOs; new `_build_cumulative_assessment`
> builder surfaces `{assessments_path, synthesized_objectives_path}` (mirrors
> `_build_assessment_objective_alignment`). All three are **warning-day-1** with
> a `# TODO(calibration)` deferred critical-flip (standard multi-wave
> deferred-flip — IB3 is the roadmap's single documented fastest-flip exception,
> these are NOT IB3). No new behavior flags introduced (FR-A11Y-02 reuses
> `ED4ALL_BLOCK_A11Y`; the two course gates run warning-day-1 unconditionally,
> mirroring `retrieval_presence`). The count table is re-derived from
> `config/workflows.yaml`: `course_generation` 31→34 warning,
> `textbook_to_course` 75→78 warning, Total 109→115 warning / 171→177 total.

> FR-A11Y-03 landing (typed B12 callouts — redundant non-color coding): one new
> validator added in **warning** at `post_rewrite_validation` in BOTH two-pass
> workflows (+1 each) — `callout_structure`
> (`lib.validators.callout_structure.CalloutStructureValidator`). Flags
> `CALLOUT_NO_KIND` (untyped callout), `CALLOUT_COLOR_ONLY` (a `callout-kind-*`
> color/border class with no visible label+icon row — WCAG 1.4.1),
> `CALLOUT_BODY_OVERFLOW` (reuses the ~200-char `resolve_body_char_ceiling`
> cognitive-load ceiling), and `CALLOUT_MOTION` (motion without a
> `prefers-reduced-motion` guard); reuses the rewrite-tier
> `_build_block_input_rewrite` builder. Introduces the new `ED4ALL_CALLOUT_TYPED`
> flag — the same flag that makes the `generate_course.py` callout renderer emit
> the redundant visible LABEL + icon + per-kind border (never color-only); strict
> no-op + byte-stable (passed=True + a `CALLOUT_STRUCTURE_DISABLED` info issue)
> when unset. The new `Block.callout_kind` field is Optional-default-None +
> hash-EXCLUDED (the `compute_content_hash` payload is a fixed 5-key allowlist).
> Warning-day-1 with a `# TODO(calibration)` deferred critical-flip (standard
> multi-wave deferred-flip — IB3 is the roadmap's single documented fastest-flip
> exception, this is NOT IB3). The sibling FR-INT-05 change (B07 per-option
> misconception feedback) added NO new gate — it escalates the existing
> `interaction_feedback` gate's already-threaded
> `distractor_misconception_alignment` signal to a WARNING
> (`DISTRACTOR_NO_MISCONCEPTION_FEEDBACK`) and is byte-stable behind
> `ED4ALL_BLOCK_QUALITY_RUBRIC`. The count table is re-derived from
> `config/workflows.yaml`: `course_generation` 34→35 warning,
> `textbook_to_course` 78→79 warning, Total 115→117 warning / 177→179 total.

> FR-COURSE-01 landing (course-level §6.5 EMERGENT-quality QA gate): one new
> validator added in **warning** at `post_rewrite_validation` in BOTH
> `course_generation` (+1) + `textbook_to_course` (+1) — `course_level_qa`
> (`lib.validators.course_level_qa.CourseLevelQaValidator`). The course-scope
> companion to the per-block IB6 rubric/rollup gates: it COMPOSES already-
> computed REAL signals (NEVER re-scores blocks) — the block→module→course
> rollup (reconstructed self-sufficiently from the canonical IB6.1 rubric
> scorer exactly as `BlockQualityRollupValidator` does) + the per-page block-
> type distribution of the rewrite-tier `blocks` + the OPTIONAL 06_assessments
> manifest — into the framework §6.5 A-F course-level gaps no per-block gate can
> capture. KEYSTONE is Section F, the interaction MIX (OSCQR rubric item 34): a
> course must span ≥2 of {student-content (exposition/check), student-student
> (discussion B10), student-instructor (graded/feedback B14 or a graded
> assessments-manifest item)} → `COURSE_INTERACTION_MIX_NARROW`. Also
> `COURSE_NO_INTEGRATION_CLOSE` (the last content module lacks a summative
> B13/B14/recap/checklist close), `COURSE_TO_COVERAGE_GAP` (a terminal
> objective touched by zero authored block — pure id resolution, no embeddings,
> only emitted when the objectives universe resolves), and
> `COURSE_RETRIEVAL_RHYTHM_THIN` (< half of content-bearing modules carry a
> retrieval/check block — advisory course-level companion to `retrieval_presence`
> / `block_sequence_order`, no recompute of the per-module spacing logic).
> Anti-fabrication: every flag is grounded in a real signal in the input
> surface; an unresolvable signal yields a structured skip, not an invented
> verdict. New `_build_course_level_qa` builder — the BROADEST
> post_rewrite_validation builder — surfaces `{blocks,
> synthesized_objectives_path?, assessments_path?}` (reuses
> `_build_block_input_rewrite` for the block set, `_resolve_objectives_path` for
> the TO universe, and `_locate` for the optional manifest). No new behavior
> flag: REUSES `ED4ALL_BLOCK_QUALITY_RUBRIC` (the rollup it composes already
> rides that flag), so it no-ops byte-stable (passed=True + a `RUBRIC_DISABLED`
> info issue) when the flag is unset. Wired warning-day-1 with a
> `# TODO(calibration)` deferred critical-flip on `COURSE_INTERACTION_MIX_NARROW`
> (standard multi-wave deferred-flip — IB3 is the roadmap's single documented
> fastest-flip exception, this is NOT IB3). The count table is re-derived from
> `config/workflows.yaml`: `course_generation` 35→36 warning,
> `textbook_to_course` 79→80 warning, Total 117→119 warning / 179→181 total.

---

> FR-INT-04 / FR-INT-03 landing (B10 three-move discussion protocol + B11
> reflection predict-then-reveal calibration). **FR-INT-04** adds ONE new gate —
> `b10_protocol` (`lib.validators.b10_protocol.B10ProtocolValidator`) — in
> **warning** at `post_rewrite_validation` in BOTH `course_generation` +
> `textbook_to_course` (+1 warning each). It flags `DISCUSSION_PROTOCOL_COLLAPSED`
> when a `discussion_prompt` block ships a single prompt with no required peer-
> response and no synthesize move (the framework's B10 contract is post →
> respond → synthesize). Rides `ED4ALL_NEW_BLOCK_TYPES` — strict no-op +
> byte-stable (`B10_PROTOCOL_DISABLED` info issue) when the flag is unset (the
> same flag the B10 three-move render scaffold rides). Day-1 warning with a
> `# TODO(calibration)` deferred critical-flip (WS3/W4 deferred-flip pattern;
> IB3 is the single documented fastest-flip exception, this is NOT IB3). The
> `discussion_prompt` catalog entry also gains `bloom_ceiling: create` (a real
> discussion drives the synthesize move to Create) and two additive hash-excluded
> Block fields (`discussion_protocol` / `discussion_bloom_verb`).
>
> **FR-INT-03** adds NO new gate row — it RIDES two existing gates behind the new
> `ED4ALL_REFLECTION_CALIBRATION` flag (default OFF). When the flag is on, the
> `interaction_feedback` gate (IB6.3) gains a `REFLECTION_NO_CAPTURE` arm (a B11
> reflection that only asks — no predict-then-reveal capture + no calibration
> feedback) and the `anatomy_slot_presence` gate (IB6.2) gains an
> `ANATOMY_REFLECTION_NO_BENCHMARK` arm (a B11 feedback slot carrying no
> calibration benchmark). Three additive hash-excluded Block fields
> (`prediction_prompt` / `reveal_content` / `calibration_feedback`) + a
> `<details>` predict-then-reveal render scaffold (byte-stable off). Both arms
> are warning-day-1 and no-op byte-stable when the flag (or the parent
> `ED4ALL_BLOCK_QUALITY_RUBRIC`) is unset.
>
> The count table is re-derived from `config/workflows.yaml`:
> `course_generation` 36→37 warning, `textbook_to_course` 80→81 warning,
> Total 119→121 warning / 181→183 total.

> **FR-INT-02 + FR-INT-06 landing (B08 first-class guided_practice + B09
> case/scenario mode + mandatory debrief):** two NEW gates wired in **warning**
> at `post_rewrite_validation` in BOTH `course_generation` + `textbook_to_course`
> (+2 warning each) — `b08_sequence`
> (`lib.validators.b08_sequence.B08SequenceValidator`, FR-INT-02: the new
> first-class `guided_practice` B08 type should FOLLOW a worked_example and carry
> a reused `fade_state` (worked/completion/independent); flags
> `B08_PRACTICE_NOT_AFTER_WORKED` + `B08_PRACTICE_NO_FADE_STATE`) +
> `b09_debrief` (`lib.validators.b09_debrief.B09DebriefValidator`, FR-INT-06: a
> B09 `scenario` must end with a debrief in its transition/consolidate slot;
> flags `SCENARIO_DEBRIEF_MISSING`). Both gates ride `ED4ALL_NEW_BLOCK_TYPES`
> (strict no-op + byte-stable, passed=True + an `*_DISABLED` info issue, when the
> flag is unset) and are warning-day-1 with deferred `# TODO(calibration)`
> critical-flip markers (WS3/W4 deferred-flip pattern; IB3 is the single
> documented fastest-flip exception, this is NOT IB3).
>
> **FR-INT-02** is a FULL new-block-type landing: `guided_practice` joins
> `Courseforge/scripts/blocks.py::BLOCK_TYPES` (29→30), gains a catalog entry
> (`framework_block: B08`, `bloom_ceiling: create`), a `DEFAULT_BLOCK_ROUTING`
> validator-matrix entry, an `_OUTLINE_KIND_BOUNDS` entry, the `blockType` schema
> enum, the `_EXPECTED_PRIMARY` framework-map (`B08`), and a deterministic
> `_render_guided_practice` faded-scaffold renderer (REUSES the existing
> `fade_state` field; byte-stable off). **FR-INT-06** adds an Optional
> hash-excluded `scenario_mode` Block field (case/scenario/branching), a
> mode-by-Bloom planner pass (`lib/generation/block_planner.py::
> _resolve_scenario_modes`, gated `ED4ALL_DYNAMIC_BLOCK_PLAN`, identity-no-op
> off), and a `_render_scenario` debrief scaffold (byte-stable off).
>
> The count table is re-derived from `config/workflows.yaml`:
> `course_generation` 37→39 warning, `textbook_to_course` 81→83 warning,
> Total 121→125 warning / 183→187 total.

### Cycle-3 C3-2 — discussion_assignment_grounded stand-in → real validator

> **C3-2** replaces the `discussion_assignment_grounded` gate's
> `# TODO(validator)` stand-in with a dedicated validator. The gate previously
> pointed at `AssessmentObjectiveAlignmentValidator` — the SAME class the
> critical `assessment_objective_alignment` gate uses — so it merely re-ran the
> aggregate objective-coverage check already enforced critically elsewhere and
> contributed ZERO independent signal. The new
> `lib.validators.discussion_assignment_grounding.DiscussionAssignmentGroundingValidator`
> isolates the B10 discussion + assignment items and audits each ONE AT A TIME:
> an item is grounded iff its explicit `source_chunk_ids` resolve to real chunks
> OR its `objective_id` is present in some chunk's `learning_outcome_refs[]`
> (∪ the synthesized objectives' id set — the W5.E union surface reused from the
> alignment validator; NO new model load). Flags `DISCUSSION_UNGROUNDED` /
> `ASSIGNMENT_UNGROUNDED` (warning). Items come from the upstream
> `discussion_items` / `assignment_items` lists when surfaced, else are
> reconstructed from the `06_assessments` manifest (+ per-item XML for the
> objective id the persisted manifest drops). Graceful-skips (passed=True) when
> chunks or items are absent — anti-fabrication. Builder
> `MCP/hardening/gate_input_routing.py::_build_discussion_assignment_grounding`
> reuses the alignment builder's `{assessments_path, chunks_path,
> synthesized_objectives_path}` resolution.
>
> **No count change** — this is a validator-PATH REPOINT of an existing
> warning gate in BOTH `course_generation` and `textbook_to_course`, not a new
> gate. Severity stays `warning` (deferred `# TODO`-style critical-flip after a
> ≥2-corpus FP measurement; WS3/W4 deferred-flip pattern; NOT IB3). The count
> table is unchanged: 62 critical / 125 warning / 187 total.

### Cycle-3 C3-6 — course-level cross-week distributed-practice (spacing) gate

> **C3-6** adds `lib.validators.cross_week_spacing.CrossWeekSpacingValidator`
> (gate id `cross_week_spacing`) — the COURSE-level companion to the
> within-module IB7.5a `ED4ALL_PLANNER_SPACING` planner pass. The framework's
> §6.5 distributed-practice / spacing axis wants a substantive concept/objective
> REVISITED across weeks, not massed into one. The gate COMPOSES existing
> signals (no model load): for each substantive concept it builds the distinct
> `week_NN`-set its blocks appear in (objective ids preferred, `key_terms`
> concept tags as fallback) and flags `CONCEPT_MASSED_SINGLE_WEEK` when a
> concept with ≥3 blocks AND ≥1 retrieval/check is taught + assessed entirely
> within one week with no spaced revisit. It is DISTINCT from `course_level_qa`'s
> `COURSE_RETRIEVAL_RHYTHM_THIN` (per-module presence of checks, not per-concept
> cross-week distribution). Anti-fabrication: graceful structured-skip on
> unresolvable week info (`WEEK_INFO_UNRESOLVABLE`) or a single-week course
> (`SINGLE_WEEK_COURSE`); reuses the `ED4ALL_BLOCK_QUALITY_RUBRIC` flag (no-op +
> byte-stable off). Wired warning-day-1 at `post_rewrite_validation` in BOTH the
> `course_generation` and `textbook_to_course` two-pass workflows (mirrors
> `course_level_qa`'s wiring); builder `_build_cross_week_spacing` reuses the
> broadest post_rewrite Block surface (`_build_block_input_rewrite`). Severity
> `warning` with a `# TODO(calibration)` deferred critical-flip after a
> ≥2-corpus FP measurement (WS3/W4 deferred-flip pattern; NOT IB3).
>
> The count table is re-derived from `config/workflows.yaml` (+1 warning gate ×
> 2 workflows): `course_generation` 39→40 warning, `textbook_to_course` 83→84
> warning, Total 125→127 warning / 187→189 total.

### W2 Defect B — `objective_specificity` (CO vacuity gate) — 2026-07-06

Added the opt-in `objective_specificity` gate at `textbook_to_course::course_planning`
(after `objective_entailment`), severity `warning` / `on_fail: warn` / `on_error: warn`,
gated behind `ED4ALL_OBJECTIVE_SPECIFICITY` (default OFF → byte-identical skip-with-pass).
It closes a real silent-pass loophole: `objective_entailment` scores an objective's
TRUTH, but nothing scored whether a statement names a concrete teachable skill, so a
vacuous CO ("Apply various techniques to solve real-world problems") passed every
course_planning gate (~22 such COs on a real 2026-07-06 TO/CO review). Three
deterministic, embedding-free checks over each CO statement — V1 content-residual floor
(`OBJECTIVE_VACUOUS`), V2 vague-object + thin residual (`OBJECTIVE_GENERIC_OBJECT`), V3
source-token recall vs cited chunk text (`OBJECTIVE_UNANCHORED_STATEMENT`) — plus the
`OBJECTIVE_VACUOUS_RATE_HIGH` headline. All reuse the shared
`objective_dedup._skill_keyphrase_tokens` residual minus the new shared
`lib/objectives/filler_lexicon.py::filler_tokens` domain-agnostic filler lexicon
(`schemas/taxonomies/objective_filler_lexicon.json`). Registered via the existing
`_build_chapter_objective_coverage_inputs` builder (no new builder). Warning day-1 with a
`# TODO(calibration)` deferred critical-flip after a ≥2-corpus FP measurement (WS3/W4
deferred-flip pattern; NOT IB3). The companion Defect E cross-window lexical-dedup pass
(`ED4ALL_OBJECTIVE_DEDUP_LEXICAL` + two satellite floors) landed in the same wave but adds
NO gate (it operates inside `objective_dedup.dedup_candidates`).

The count table is re-derived from `config/workflows.yaml` (+1 warning gate on the
`textbook_to_course` workflow only; `course_generation` has no `course_planning` phase):
`textbook_to_course` 70→71 warning / 128→129 total, Total 100→101 warning / 194→195 total.

---

### flip-wave-2 — 7 calibration-validated warning→critical promotions + WCAG dead-ref repair — 2026-07-07

The second calibration flip wave (successor to the 17-gate wave that promoted
`anatomy_slot_presence` / `callout_structure` et al.). The calibration harness
(`scripts/harness/calibration_harness.py`) was re-run against the recorded calibration
evidence — 5 distinct internal calibration corpora discovered dynamically (no
slug pinned), 22 contributing runs, ≥2-corpora precondition satisfied. Seven
gate families measured `flip_ready` (≥2 corpora, pooled AND worst-corpus
fire-rate inside the documented band) and survived a NEW **flag-off no-op audit**
(below), so they were promoted `severity: warning → critical` /
`on_fail: warn → block` / `on_error: warn → fail_closed`, mirroring the prior
wave's declarative pattern (the validators stay warning-day-1 `passed=True`, so
default-off runs remain byte-stable — the flip reclassifies the gate into the
critical gate-ledger column, it does not newly block clean runs):

- `udl_coverage` (IB4.5) — pooled 0.0000 / 2 corpora — `course_generation` +
  `textbook_to_course` `post_rewrite_validation`.
- `interaction_feedback` (IB6.3) — pooled 0.036 / 2 corpora — both workflows.
- `block_quality_rubric` (IB6.1) — pooled 0.018 / 2 corpora — both workflows.
- `key_terms_definition_quality` (W1.5) — pooled 0.0000 / 2 corpora — both.
- `mayer_ctml` — pooled ~0.090 / 2 corpora — both.
- `recall_self_check_format` — pooled 0.0000 / 2 corpora — both.
- `misconception_productive_failure` — pooled 0.0000 / 2 corpora —
  `textbook_to_course` only (not wired in `course_generation`).

**No-op audit (harness hardening).** The four 0.0000-fire families could in
principle owe their clean rate to flag-off skip-passes counted as clean passes
(these validators return `passed=True` + a single unattributed `*_DISABLED` info
issue when their flag is unset). The harness now detects that skip signature in
the `02_validation_report/report.json` phase-level section
(`_is_block_skip_marker`: a BLOCK-scoped gate that passed, took no action, and
logged only unattributed issues) and excludes those units from the fire-rate
denominator (`no_op_evaluated`, surfaced per family) so an unexercised flag-off
gate can never present as flip-ready. Re-run confirmed all seven ran flag-ON in
both corpora (a flag-off run would have emitted the skip marker; the measured
reports carry genuine `issue_count` with zero no-op inflation).

**Out-of-band families NOT flipped** (blocker is a manual FP audit / generator
fix, NOT a missing corpus — comments retagged from the obsolete "awaiting 2nd
corpus / harness" framing): IB3 verb-triple/anchored-rubric/triangle pooled 0.32
(worst 0.51); `block_cognitive_load` 0.31; `qa_checklist` 0.28;
`retrieval_presence` 1.00; `source_coverage` 0.75; `bloom_distribution` /
`cross_week_spacing` / `course_level_qa` 1.00; `rewrite_block_objective_delivery`
0.27; `block_quality_rollup` 0.50. **Zero-observation families** (gate never
exercised — needs ≥2 flag-on / assessment-emitting runs): `course_completeness`,
`cumulative_assessment`, `difficulty_provenance`, `objective_specificity`,
`prerequisite_sequencing`, `synthesized_quiz_distractor`,
`terminal_objective_source_grounding`, `wcag_compliance`; plus `chunk_wcag_status`
+ `co_terminal_alignment` at a single corpus.

**WCAG dead-ref repair.** The `textbook_to_course` `packaging` `wcag_compliance`
gate pointed at a WCAG validator module retired in the SemantiK migration;
under `on_error: warn` the gate silently
degraded to a warning every run and the validator never executed. Repaired to
the live `lib.validators.wcag.WCAGValidator` (the class `course_generation`
packaging already uses); a new regression test
(`lib/tests/test_gate_validator_imports.py`) imports every gate's validator
dotted path so a dead ref fails CI instead of silently warn-degrading. Gate
stays warning day-1 (it now actually runs — measure before any flip).

The count table is re-derived from `config/workflows.yaml`. flip-wave-2 is a pure
warning→critical reclassification (no gates added/removed), +13 critical / −13
warning overall: `course_generation` 30→36 critical / 27→21 warning (57 total,
unchanged); `textbook_to_course` 58→65 critical / 71→64 warning (129 total,
unchanged); Total 94→107 critical / 101→88 warning / 195 total (unchanged).

### Track-L (L2) — `cartridge_conformance` post-packaging gate wired — 2026-07-19

Wired the previously-unwired `CartridgeConformanceValidator`
(`lib/validators/cartridge_conformance.py`, landed `ed8d1c32`) as a gate after
the `packaging` phase in BOTH `course_generation` and `textbook_to_course`. It is
the full-cartridge strict CC/QTI conformance guarantee on the BUILT `.imscc`
(opens the assembled ZIP: manifest well-formedness + CC-version, namespace-routed
XSD conformance, resource-type enumeration, in-zip href integrity, answer_key
resource presence) — the LMS-agnostic portability contract that complements
`qti_well_formed` (which validates only the loose `06_assessments/*.xml`
PRE-packaging). Builder reuses `_build_imscc` (`inputs["imscc_path"]`), registered
in `MCP/hardening/gate_input_routing.py`.

Warning day-1 (`severity: warning` / `on_fail: warn` / `on_error: fail_closed`,
per the `qti_well_formed` precedent) — the validator's own GateIssue severities
stay honest (critical for `CARTRIDGE_XSD_INVALID` / `_RESOURCE_TYPE_UNKNOWN` /
`_HREF_UNRESOLVED` / `_ANSWER_KEY_UNREFERENCED`; warning for the lxml/XSD-missing
graceful-degrade codes), so a later critical-flip is a config-only change. Until
the L1 IMS/QTI XSD pack is vendored (a fetch-window item), the XSD arm
graceful-degrades to `CARTRIDGE_XSD_MISSING`/`_LXML_MISSING` warnings while the
programmatic checks still run.

> The count table is re-derived from `config/workflows.yaml` (+1 warning gate per
> workflow): `course_generation` 28→29 warning / 62→63 total, `textbook_to_course`
> 71→72 warning / 135→136 total, Total 102→104 warning / 206→208 total.

### Book-1 canary keystone — `block_prose_stutter` at post_rewrite_validation — 2026-07-22

Landed the deterministic prose-stutter validator
(`lib/validators/prose_stutter.py::ProseStutterValidator`) as a
`post_rewrite_validation` gate in BOTH `course_generation` and
`textbook_to_course`. Root cause: the rewrite tier's local model ships
phrase-repetition slop ("GET and HEAD are safe and HEAD are safe and
idempotent"-shaped duplication) that NLI entailment cannot catch — a stuttered
sentence still entails its source — and ~150-190 of 613 canary prose chunks
carried it into the retrieval corpus. Five pure-text rules over segment-scoped,
HTML-stripped prose (element boundaries become hard newlines; hidden
`data-cf-curie` spans + `pre`/`code` bodies stripped): adjacent_repeat,
window_repeat, near_adjacent_repeat, echo_word, label_dup. Builder reuses
`_build_block_input_rewrite` (blocks only), registered in
`MCP/hardening/gate_input_routing.py`.

ENFORCEMENT is not the gate: `_run_content_generation_rewrite` threads the same
validator into the rewrite router's per-candidate chain
(`route_rewrite_with_remediation` + `route_rewrite_batch` — previously
`validators=[]`), so a stuttered candidate fires `action="regenerate"` and the
best-of-N loop re-rolls it at authoring time. The gate is the observability
net; warning day-1 (`on_fail: warn` / `on_error: warn`) per the WS3/W4
deferred-flip convention (NOT IB3). Calibration on the canary imscc chunkset:
243/613 prose chunks flagged (the canary heuristic flagged ~247), all 5
owner-confirmed stutter chunks caught verbatim, 0.9% (7/765) flag rate on the
clean-path QTI assessment population, ~87% hand-audited precision on flagged
prose (residual dominated by chunk-boundary fusion artifacts that cannot occur
on block HTML).

> The count table is re-derived from `config/workflows.yaml` (+1 warning gate per
> two-pass workflow): `course_generation` 29→30 warning / 63→64 total,
> `textbook_to_course` 72→73 warning / 136→137 total, Total 104→106 warning /
> 208→210 total.

### Bloom-ladder initiative WI-21 — `bloom_ladder_ceiling` + `rejected_claim_entailment` gate wiring — 2026-07-29

Wires the two Bloom-ladder-initiative validators built ahead of this landing
(WI-15's `lib.validators.bloom_ladder_ceiling.BloomLadderCeilingValidator`,
WI-16's `lib.validators.pair.rejected_claim_entailment.RejectedClaimEntailmentValidator`)
into `config/workflows.yaml` + `MCP/hardening/gate_input_routing.py`. Both are
**warning day-1**, gated by their own default-OFF flags/graceful-degrade
contracts, and never block a build.

- `bloom_ladder_ceiling` — wired at BOTH `inter_tier_validation` +
  `post_rewrite_validation` in BOTH `course_generation` and
  `textbook_to_course` (+4 warning). One shared builder
  (`_build_bloom_ladder_ceiling`) threads `content_generation_rewrite`'s
  `blocks_final_path` + the canonical `synthesized_objectives.json` path
  regardless of which of the two phases fires: on a normal full run the
  rewrite tier hasn't emitted yet at `inter_tier_validation` time, so that
  seam structurally skips (`required_missing=['blocks_final_path']`) and the
  gate runs for real only at `post_rewrite_validation`; the `courseforge-
  validate` stage subcommand pre-populates the rewrite tier from disk and
  runs both phases in one pass, so wiring both seams gives that subcommand
  full ladder coverage too. The validator itself structured-skips (INFO
  `BLOOM_LADDER_DISABLED`) when `ED4ALL_BLOOM_LADDER` is unset — byte-stable
  off.
- `rejected_claim_entailment` — wired at `textbook_to_course::training_synthesis`
  alongside the existing pair-tier gates (+1 warning; `course_generation` has
  no `training_synthesis` phase). Reuses the existing
  `_build_training_synthesis` builder verbatim (its `preference_pairs_path` +
  chunk-window resolution is exactly this validator's input contract) — no
  new builder needed, just the registry entry. Flags a
  `source="misconception"` preference pair whose `rejected` completion is
  itself NLI-entailed by its cited source chunk (entailment ≥0.70); graceful-
  degrades on missing NLI/embedding extras unless
  `TRAINFORGE_REQUIRE_EMBEDDINGS` is set.

The count table is re-derived from `config/workflows.yaml`:
`course_generation` 31→33 warning / 64→66 total, `textbook_to_course`
75→78 warning / 139→142 total, Total 109→114 warning / 212→217 total
(critical unchanged at 103).

### Bloom-ladder initiative addendum AD-02 — `dpo_yield_projection` gate wiring — 2026-07-29

Wires `lib.validators.dpo_yield_projection.DpoYieldProjectionValidator` — the
DPO-yield famine guard built alongside this landing — into
`config/workflows.yaml` + `MCP/hardening/gate_input_routing.py`. Warning
day-1, gated by its own structured-skip contract, never blocks a build.

- `dpo_yield_projection` — wired at `textbook_to_course::training_synthesis`
  alongside the existing `synthesis_quota` gate (+1 warning; `course_generation`
  has no `training_synthesis` phase). New builder `_build_dpo_yield_projection`
  threads `chunks_path` (required — `imscc_chunks_path` / `chunks_path` /
  `semantik_chunks_path` precedence, mirroring `_build_training_synthesis`) +
  `objectives_path` (`_resolve_objectives_path`, shared with
  `_build_bloom_ladder_ceiling`) + an optional `min_dpo_pairs` override.
  Projects the same admission chain the real synthesis run exercises
  (`resolve_chunk_misconceptions` → `focus_chunk_on_canonical_objective` +
  `pair_eligibility` → `micro_preference_eligibility` →
  `is_dpo_editorial_record`, every predicate imported from the real
  Trainforge modules rather than reimplemented) and fires
  `DPO_YIELD_PROJECTION_BELOW_FLOOR` when the projected admissible count is
  below `min_dpo_pairs` (default 50). Structured-skips on missing
  `chunks_path` / `objectives_path`, and on a legacy corpus (zero recovered
  misconception cards AND `ED4ALL_BLOOM_LADDER` unset) so pre-initiative
  corpora never see a spurious famine warning. The same projection core
  (`lib.validators.dpo_yield_projection.project_dpo_yield`) backs the
  standalone `python -m Trainforge.scripts.model_dpo_yield` CLI, so an
  operator can run the identical projection ahead of a manual synthesis
  invocation, not just inside the gated pipeline.

The count table is re-derived from `config/workflows.yaml`:
`textbook_to_course` 78→79 warning / 142→143 total, Total 114→115 warning /
217→218 total (`course_generation` unchanged at 33 warning / 66 total;
critical unchanged at 103).
