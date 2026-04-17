# Ed4All Versioning and Roadmap

This document is the honest characterisation of what Ed4All delivers today, what it does not, and what v1.0 is expected to deliver. It exists because the WCAG_201 knowledge package — the first real end-to-end artifact the pipeline produced — surfaced nine concrete issues that are best understood as *diagnostic signal from v0.1.0*, not as product failures.

The branch that shipped this file (`claude/fix-package-quality-FyMue`) moves Ed4All out of "v0.1.0 prototype" mode and into "v0.1.x with honest self-evaluation." The follow-up branch flips strict mode on and promotes two workflow gates from warning to critical (see §Severity flip trigger below).

---

## §1 What v0.1.0 delivers

End-to-end pipeline: **DART → Courseforge → Trainforge → LibV2**.

- **Accessible HTML** — semantic structure, proper heading hierarchy, WCAG 2.2 AA target, alt text on all images.
- **Structured courses** — IMSCC packages with course/terminal/chapter objectives, Bloom-level metadata on every learning objective, JSON-LD metadata on every HTML page.
- **Knowledge-domain language graphs** — chunked corpus with per-chunk concept tags, Bloom's level, content-type labels, key terms, misconceptions, and outcome references. A co-occurrence concept graph derived from those tags.
- **Basic quality metrics** — `quality_report.json` reports per-chunk compliance (size, tags, HTML presence, Bloom coverage, outcome coverage).

This is enough to pitch the pipeline. It is *not* enough to ship AI-ready training data that an NSF reviewer or an agency CTO could evaluate on its own metrics without additional scrutiny.

---

## §2 Known v0.1.0 limitations — the nine diagnostic signals

These are artifacts the WCAG_201 assessment surfaced. Each is framed as what v0.1.0 is designed to expose, not a bug that slipped through:

1. **Footer contamination in ~69% of chunks.** Courseforge emits the copyright notice in the page body rather than a template region. Trainforge's extractor consumes everything inside `<body>`. The fix is ownership: Courseforge moves the notice into a `data-cf-role="template-chrome"` region and Trainforge skips that role. (§2.3 of the implementation plan; see `Trainforge/rag/boilerplate_detector.py` for the defensive layer shipped in this branch.)

2. **Broken learning-outcome references (~60% unresolvable).** The chunker emits week-scoped IDs (`w01-co-02`) but `course.json` only stores flat IDs (`co-02`). Resolving this is a **schema decision**, not a regex fix: this branch commits to Courseforge emitting both forms, Trainforge storing flat IDs in `learning_outcome_refs` and week-scoped IDs in a new `pedagogical_scope_refs` field. Orphan week-scoped IDs (legacy content, drift) are preserved with `parent_id: null` so the defect surfaces in metrics rather than disappearing. (§2.1.)

3. **Mis-scoped `follows_chunk` (~68% of chain links cross lesson boundaries).** Now reset at every lesson and module boundary; violations are reported as `integrity.follows_chunk_boundary_violations`. (§4.3.)

4. **Concept graph is a tag co-occurrence graph, not a semantic knowledge graph.** The README previously overclaimed. Fixed in this branch: edges carry `relation_type: "co-occurs"` (forward-compatible for v1.0's typed extractor), pedagogy / logistics tags are partitioned into a separate `pedagogy_graph.json`, and a `concept_graph_semantic.json` filename is reserved for the v1.0 extractor. (§2.2, §3.1.)

5. **Quality report dishonesty.** `html_preservation_rate: 1.0` while 62% of chunks had unclosed `<div>` tags; `learning_outcome_refs_coverage: 1.0` while 60% of refs were unresolvable. Both measured field presence, not correctness. Metrics rewritten to measure real structure and resolution; new `methodology` block in the report documents semantics; `metrics_semantic_version: 2` constant tells downstream consumers to re-baseline. (§1.1–1.4.)

6. **Half-populated enrichment fields.** `bloom_level` 87%, `key_terms` 53%, `misconceptions` 54%, `content_type_label` 53%. **Investigation-first in this branch** (see §4 below). Fallback helpers exist in `process_course.py` but are deliberately unwired until the investigation concludes.

7. **SC name drift.** Sixteen WCAG success criteria appeared under inconsistent names (`Contrast Minimum`, `Contrast Minimum, Level AA`, `Contrast (Minimum)` …). Canonicalisation now applied to chunk text, key-terms metadata, misconceptions, and **concept tags** — the last is where retrieval sharpness actually lives, because the graph fragments otherwise. (§4.5.)

8. **Factual inaccuracies.** The WCAG_201 content claimed "87 success criteria" (W3C says 86) and enumerated "29+29+17+4" (sums to 79). A new `ContentFactValidator` (§4.6) flags both numeric-claim mismatches and internal-arithmetic contradictions. Warning-only today.

9. **`leak_check` only inspected Q/A leakage.** Extended to detect corpus-wide boilerplate repetition (§4.7).

---

## §3 Pipeline self-trust and strict mode

Honest metrics are necessary but not sufficient. A pipeline that computes accurate scores and then writes the artifact anyway still lets bad packages ship. This branch adds a *refuse-to-write* integrity gate:

- `CourseProcessor(strict_mode=True, ...)` — off by default in v0.1.x, on by default in v1.0.
- When strict mode is on, `_assert_integrity(report)` raises `PipelineIntegrityError` if any of the following hold:
  - `integrity.broken_refs` is non-empty
  - `integrity.follows_chunk_boundary_violations` is non-empty
  - `len(html_balance_violations) / total_chunks > 0.05`
- The CLI exposes this as `--strict`.

### Severity flip trigger

The gates `outcome_ref_integrity` and `content_fact_check` ship in `config/workflows.yaml` at `severity: warning`. The follow-up PR flips them to `critical` and turns on `strict_mode=True` by default. The flip is contingent on one event, not a calendar date:

> The synthetic `Trainforge/tests/fixtures/mini_course_clean/` fixture runs green in CI with `metrics.footer_contamination_rate == 0`, `integrity.broken_refs == []`, and `integrity.factual_inconsistency_flags == []`.

No single run forces the decision unless this criterion is explicit. Name it. Ship it.

---

## §4 §4.4a enrichment-coverage investigation (result slot)

**Status at time of writing:** investigation deferred — see the implementation plan (`claude/fix-package-quality-FyMue` branch, plan file). The §4.4a investigation asks which of four hypotheses dominates the 47% enrichment miss rate:

- **H1** JSON-LD `sections` keyed by a heading that doesn't match post-merge `section_heading`.
- **H2** JSON-LD `sections` is genuinely empty on many pages.
- **H3** `content_type_label` short-circuit in `_extract_section_metadata` produces half-populated chunks.
- **H4** The "no sections" code path in `_chunk_content` never invokes `_extract_section_metadata` at all.

The investigation MUST complete before any fallback helper (`derive_bloom_from_verbs`, `extract_key_terms_from_html`, `extract_misconceptions_from_text`) is wired into `_create_chunk`. If the root cause is structural (H1/H3/H4), the fix is at the source, not in fallback regex. If the root cause is H2, fallbacks are appropriate.

The helpers exist in `Trainforge/process_course.py` at module scope and are unit-tested. They will be deleted if unused after the investigation concludes — dead code masking a fixable bug is worse than a known gap.

---

## §5 v1.0 roadmap

Ordered by what the work currently on the v1.0 branch list looks like:

1. **§4.4a investigation complete** — this is the next concrete blocking item.
2. **Typed-edge concept extractor** → writes `concept_graph_semantic.json` alongside the existing co-occurrence graph. Edges carry `relation_type` ∈ {`prerequisite`, `is-a`, `related-to`, `co-occurs`}.
3. **Strict mode on by default.** See §Severity flip trigger.
4. **`outcome_ref_integrity` and `content_fact_check` promoted to `critical`.**
5. **Dual outcome-ID contract shipped.** Courseforge emits both flat and week-scoped IDs with explicit parent links; Trainforge consumes both fields.
6. **Template-chrome footer separation.** Courseforge stops emitting copyright in the page body.
7. **Enrichment coverage resolved.** Per §4.4a outcome.
8. **SC canonicalisation extended** to every SC mentioned in WCAG 2.2, not just the handful currently in the variant table.
9. **`ContentFactValidator` regex table broadened** with domain-specific content-fact rules on contact with real curricula.

---

## §6 v1.0 exit criteria (explicit)

v1.0 is not a marketing milestone. It is a concrete set of conditions all of which must hold:

- **(a) Self-trust.** `mini_course_clean/` runs green with `strict_mode=True` and zero integrity violations.
- **(b) Domain-agnostic validation.** The pipeline has been run against **≥3 distinct domain corpora** with no new defect classes surfaced. One is WCAG_201; the other two must be outside accessibility (for example: a STEM textbook and a humanities textbook). The author should deliberately pick domains they are *less* expert in, to stress-test whether the defects surfaced in v0.1.0 are universal pipeline issues or WCAG-specific artifacts. This is the single test that tells us whether Ed4All generalises.
- **(c) Severity flip completed.** Both `outcome_ref_integrity` and `content_fact_check` at `critical`. `strict_mode=True` default.
- **(d) README matches reality.** No paragraph overclaims what the graph or the metrics deliver. The reverse is fine — selling short is always safer than the claim/reality gap a sophisticated reviewer will immediately notice.

---

## §7 Grant-narrative framing (NSF TechAccess "AI-Ready America")

Ed4All is the strongest portfolio piece under the NSF TechAccess framing. DART is one stage of it; the other three stages (Courseforge, Trainforge, LibV2) are the differentiator. The WCAG_201 package is a genuine demonstration of the pipeline working end-to-end on a domain that sits directly inside Ed4All's accessibility mission.

For a proposal, the claim structure is:

- **Here is v0.1.0 output** (the WCAG_201 package and its defects).
- **Here is the defect analysis** (the nine signals documented in §2).
- **Here is the v0.1.0 → v1.0 roadmap** (this document).
- **Here is v1.0 output on the same domain** (the regenerated WCAG_201 package once v1.0 is shipped).
- **Here are the measured deltas** (footer contamination, outcome ref integrity, graph fragmentation, quality-report trustworthiness).

That structure is stronger than any "here's a pipeline we built" narrative because it demonstrates a *method*: measure, characterise, remediate, re-measure. Funded proposals reward method.

### Paired before/after artifacts — bound follow-up task

Before the pipeline moves past v0.1.x, snapshot:

1. The v0.1.0 WCAG_201 artifact as shipped.
2. The v0.1.0 WCAG_201 `quality_report.json` with both its original self-scores and the honest METRICS_SEMANTIC_VERSION=2 re-scoring for the same corpus.
3. A v1.0 WCAG_201 regeneration once the v1.0 branch ships.
4. A side-by-side delta table: per-metric change with honest math.

This task is flagged explicitly because if it isn't captured *before* the pipeline advances, the v0.1.0 artifact becomes impossible to regenerate cleanly and the grant narrative loses its strongest evidence. Tracked on the v1.0 branch.
