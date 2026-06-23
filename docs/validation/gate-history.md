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
> counts is the summary table in `CLAUDE.md` § "Active Gates" (re-derived from
> `config/workflows.yaml`: **62 critical / 105 warning / 167 total**) plus the
> per-gate detail in `docs/validation/gates.md`. The deltas below are kept for
> per-wave provenance only; do not sum them to recover the current total.

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
> `config/workflows.yaml` and `plans/finegrain/w4-nli-grounding-gate.md` §4.
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
> established this session (`plans/finegrain/content-block-quality-2026-06.md`
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
> `plans/finegrain/w10-assessments-qti-discussions-2026-06.md` §2.4 Option A).
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
> `wcag_compliance` (`DART.pdf_converter.wcag_validator.WCAGValidator`, reused
> as-is) at `textbook_to_course` `packaging` (+1 warning). The IB4.1 per-block
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
> FR-07/13) is DEFERRED until `scripts/calibration_harness.py` confirms the
> rubric gates' FP rate on ≥2 corpora (the anchored 0-3 scale must be calibrated
> before the mean/min-floor hard gates can block early runs; standard multi-wave
> deferred-flip — IB3 is the roadmap's single documented fastest-flip exception,
> this is NOT IB3). The count table is re-derived from `config/workflows.yaml`:
> `course_generation` 30→31 warning, `textbook_to_course` 74→75 warning, Total
> 107→109 warning / 169→171 total.
