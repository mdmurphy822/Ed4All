# GPT Feedback v3 — per-claim rebuttal

**Branch:** `dev-v0.3.0`
**HEAD SHA:** `af1744ca47fdb6f070d07a662d4643638a362c08`
**Date:** 2026-05-08
**Purpose:** Adjudicate the 17 claims in `plans/GPTFEEDBACK3.txt` against the actual `dev-v0.3.0` tree, pin commit-SHA evidence, and scope the small residue worth absorbing.

---

## TL;DR

GPT reviewed a stale snapshot. Of **17 distinct claims**, **13 are
REJECT-as-already-shipped** (every "missing" validator, aggregator, and
gate is on disk + wired in `config/workflows.yaml`); **3 are
ACCEPT-as-already-shipped** because the recommendation IS the live design
(W3 placeholder filter via `SkippedItem`, W5 aggregate
`*_validation_report.json`, W4 evidence-spans concept); **1 is
ACCEPT-as-deferred**: literal verbatim-quote substring storage on
`evidence_quote` is genuinely missing — tracked as the **W-D11** wave (§5).
The contested validator package ships and reorganized in `2a5556e` (W-D7
split foundation) and `5235b3c` (W-D10 subpackage reorg with shim
re-exports). I spot-checked 5 REJECT-targeted claims by filesystem grep —
all mapped to extant code.

---

## Per-claim adjudication

| # | GPT claim (verbatim excerpt) | Status | Evidence (file:line) | Commit |
|---|------------------------------|--------|----------------------|--------|
| 1 | "this snapshot has no `lib/validators/` package" (line 20) | REJECT | `lib/validators/__init__.py` plus **60** `*.py` siblings (`ls lib/validators/*.py | wc -l = 60`) | `2a5556e`, `5235b3c` |
| 2 | "`lib.validators.content.ContentStructureValidator` ... missing" (line 14) | REJECT | `lib/validators/content.py`; wired as `content_structure` gate in `config/workflows.yaml` | `2a5556e` |
| 3 | "`lib.validators.imscc.IMSCCValidator` ... missing" (line 15) | REJECT | `lib/validators/imscc.py`; wired as `imscc_structure` gate (`course_generation::packaging`, `intake_remediation::parsing`) | `2a5556e` |
| 4 | "`lib.validators.assessment.AssessmentQualityValidator` ... missing" (line 16) | REJECT | `lib/validators/assessment.py`; wired on `trainforge_assessment` + `rag_training::assessment_generation` | `2a5556e` |
| 5 | "`lib.validators.bloom.BloomAlignmentValidator` ... missing" (line 17) | REJECT | `lib/validators/bloom_alignment.py` (shim re-exporting `lib.validators.bloom.alignment` per W-D10 reorg) | `5235b3c` |
| 6 | "`lib.validators.assessment.FinalQualityValidator` ... missing" (line 18) | REJECT | `lib/validators/assessment.py::FinalQualityValidator`; wired as `final_quality` on `rag_training::validation` | `2a5556e` |
| 7 | "validator interface mismatch will bite you" (line 26) | REJECT | All `lib/validators/*.py` adopt the canonical `validate(inputs) -> GateResult` signature; the workflow runner normalises artifact dicts at the gate-runner walk; WCAG validator wrapped at `MCP/hardening/validation_gates.py` | `2a5556e` |
| 8 | "Trainforge/generators/assessment_generator.py still contains 'Correct answer based on content'" (line 50) | ACCEPT-shipped | Fallback branches no longer emit those literals. `assessment_generator.py:138-143` adds `SkippedItem`; `:431-438` logs refusal; `:594-614` raises `assessment_template_skip` instead of shipping a placeholder. The literals live ONLY in docstrings / rationale strings now | follow-on |
| 9 | "Trainforge/validation/ only contains __init__.py" (line 64) | REJECT | Conflates surface location. Trainforge validation lives at `lib/validators/` as the canonical home (the unification GPT's W1 recommends — already done): 60 modules, including every "missing" gate from items 1-6 plus `pair_claim_support`, `pair_lo_refs`, `pair_objective_delivery` | `2a5556e`, `5235b3c` |
| 10 | "LeakChecker ... is not a complete gate adapter" (line 80) | REJECT | `lib/validators/synthesis_leakage.py:148` declares `SynthesisLeakageValidator` with canonical `validate(inputs) -> GateResult`; wired as `synthesis_leakage` on `textbook_to_course::training_synthesis` (critical) | pre-existing |
| 11 | "no single courseforge_validation_report.json" (line 114) | ACCEPT-shipped | `lib/aggregators/courseforge_validation_report.py:271` → `CourseforgeValidationReport`; emits `<project>/courseforge_validation_report.json` (schema 1.1) with `passed`, `blocking_failures`, `warnings`, `per_block_results[]`, `final_promotion_decision` | superseded by `6c56767` |
| 12 | "TrainForge ... does not prove [chunks entailed]" (line 152) | REJECT | Two NLI validators: `lib/validators/claim_support.py:348::ClaimSupportValidator` (block-level, post_rewrite) and `pair/claim_support.py::PairClaimSupportValidator` (pair-level, training_synthesis); both stamp per-claim `outcome` per chunk_v4 / instruction_pair schemas | W2.F + W4.A |
| 13 | "no answerability validator" (line 67) | REJECT | `lib/validators/assessment.py::AssessmentQualityValidator` checks answerability via stem-clarity + correct-answer rules; per-question issue codes surface in `quality_report.json::assessments.per_question_issues[]` | `2a5556e` |
| 14 | "no distractor plausibility validator" (line 67) | REJECT | Three modules: `lib/validators/distractor_plausibility.py`, `distractor_misconception_alignment.py`, `distractor_structural.py` plus `padded_distractor.py`; all four wired on `inter_tier_validation` + `post_rewrite_validation` + `trainforge_assessment` | pre-existing waves |
| 15 | "no objective alignment validator" (line 72) | REJECT | `lib/validators/assessment_objective_alignment.py:116::AssessmentObjectiveAlignmentValidator` (critical, fails closed when any question's `objective_id` is uncovered by chunks' `learning_outcome_refs`) | pre-existing |
| 16 | "future `ThetaValidator` could compute ... grounding coherence, validator agreement, hallucination risk" (line 848) | DEFER (not REJECT) | Inputs already aggregated in `lib/aggregators/promotion_chain_report.py:158::PromotionChainAggregator` per-arrow rows (`source_coverage`, `validator_set[]`, `passed`, `warnings_count`) plus `lib/governance/course_status.py::derive_course_status` 5-branch composer. Composing a single `theta_score` is mechanical once W-D11 calibration lands | `6c56767` |
| 17 | "evidence_spans ... no evidence span → fail" (line 216) | ACCEPT-deferred → W-D11 | Schema-side: `key_claims[].evidence_quote` and `per_claim_support[].evidence_quote` are not yet additive optional fields on `chunk_v4.schema.json` / `instruction_pair.schema.json`. The chunk-ID-level grounding ships (item 12); the substring-level surface is the genuine forward gap | scheduled W-D11 |

Spot-checks I ran directly (not trusting any triage): items 1, 8, 10, 11, 14 — all five mapped to extant evidence at the cited file:line.

---

## What we accept

**(a) Verbatim evidence-quote substring storage IS missing.** Item 17 is
the one substantive forward signal. Chunk-ID-level grounding ships (item
12), but the literal substring the entailment fires against is not
persisted alongside the outcome — audit replays depend on re-running the
NLI classifier. W-D11 (§5) closes this.

**(b) Docstring placeholder strings invite reviewer confusion.** GPT's
grep hits on `"Correct answer based on content"` at
`Trainforge/generators/assessment_generator.py:599-609` are real — but
they are rationale-string interpolations explaining what the path
**refuses to emit**, not strings that ship into pairs. A docs-cleanup
micro-PR should reframe those so the next reviewer doesn't false-positive.

**(c) `course_status` cohort table publication.** The 5-branch cohort
table (`ACCESSIBILITY_GATE_IDS` / `INSTRUCTIONAL_GATE_IDS` /
`TRAINABLE_GATE_IDS`) is documented in root `CLAUDE.md` § "Aggregators"
but the review didn't surface it. A standalone
`docs/governance/course_status.md` would help future reviewers. Low-prio.

---

## §5 — W-D11 forward sketch

Verbatim evidence-quote substring storage will land as **Wave W-D11**
(7 commits, T11.0–T11.6) extending `key_claims[].evidence_quote` and
`per_claim_support[].evidence_quote` as **additive optional fields** on
`chunk_v4.schema.json` and `instruction_pair.schema.json` (legacy corpora
stay valid). Producers (`ClaimSupportValidator`,
`PairClaimSupportValidator`) stamp the matched substring at NLI-positive
emit time; consumers light up substring-level traceback for free. No new
gate day-1; once coverage stabilises, an `evidence_quote_coverage`
warning gate flips to critical via the standard calibration path.

---

## §6 — ThetaValidator (item 16)

DEFER, not REJECT. Every input a prospective `ThetaValidator` would
compose is already aggregated in
`lib/aggregators/promotion_chain_report.py` per-arrow rows
(`source_coverage`, `warnings_count`, `validator_set[]`, `passed`) plus
`lib/governance/course_status.py::derive_course_status`'s cohort table.
Composing a single `theta_score: float` is mechanical (weighted sum over
per-arrow `source_coverage * passed_indicator`). The right time to ship
is **post W-D11**, when calibration has a real signal floor — until
verbatim quotes anchor the NLI substring traceback, `theta_score` would
integrate noise. No wave number committed.

---

## How to re-review

1. Pin to `dev-v0.3.0` HEAD: `git checkout af1744ca47fdb6f070d07a662d4643638a362c08`.
2. Confirm validator package: `ls lib/validators/*.py | wc -l` (expect ≥ 60).
3. Confirm aggregators: `ls lib/aggregators/*.py` (expect 4: `courseforge_validation_report`, `coverage_map`, `promotion_chain_report`, `trainforge_assessment_quality_report`).
4. Run the gate-runner walk tests: `pytest lib/validators/tests/ -q` and `pytest schemas/tests/ -q`.
5. Inspect the placeholder kill-switch: `Trainforge/generators/assessment_generator.py:138-143` (SkippedItem dataclass), `:594-614` (per-question dispatch).
6. Re-read `plans/GPTFEEDBACK3.txt` against this evidence.

The signal-to-noise was 1 ACCEPT-deferred / 17 — a low yield reflecting
the snapshot lag, not a defective review process. The one real gap (W-D11
verbatim quotes) is scheduled.

---

*End of rebuttal. Source: `plans/GPTFEEDBACK3.txt`. Adjudicated against
`dev-v0.3.0` HEAD `af1744ca47fdb6f070d07a662d4643638a362c08` on
2026-05-08.*
