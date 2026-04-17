# Ed4All Versioning and Roadmap

This document is the honest characterisation of what Ed4All delivers today, what it does not, and what v1.0 is expected to deliver. It exists because the first real end-to-end knowledge package the pipeline produced (an accessibility-domain corpus, held locally and not shipped in this repo) surfaced nine concrete issues that are best understood as *diagnostic signal from v0.1.0*, not as product failures.

The branch that shipped this file (`claude/fix-package-quality-FyMue`) moved Ed4All out of "v0.1.0 prototype" mode and into "v0.1.x with honest self-evaluation." The follow-up branch flips strict mode on and promotes two workflow gates from warning to critical (see §Severity flip trigger below).

**v0.2.0 status (development branch `dev-v0.2.0`):** the workers-A-through-K cohort on `dev-v0.2.0` delivers a substantial chunk of the v1.0 roadmap ahead of the formal v1.0 release. See §5a below for the mapping from v1.0 promises to the v0.2.0 artifacts that fulfilled them. v1.0 itself remains defined by the §6 exit criteria, all of which must hold before the version number moves.

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

These are artifacts the v0.1.0 real-domain assessment surfaced. Each is framed as what v0.1.0 is designed to expose, not a bug that slipped through:

1. **Footer contamination in ~69% of chunks.** Courseforge emits the copyright notice in the page body rather than a template region. Trainforge's extractor consumes everything inside `<body>`. The fix is ownership: Courseforge moves the notice into a `data-cf-role="template-chrome"` region and Trainforge skips that role. (§2.3 of the implementation plan; see `Trainforge/rag/boilerplate_detector.py` for the defensive layer shipped in this branch.)

2. **Broken learning-outcome references (~60% unresolvable).** The chunker emits week-scoped IDs (`w01-co-02`) but `course.json` only stores flat IDs (`co-02`). Resolving this is a **schema decision**, not a regex fix: this branch commits to Courseforge emitting both forms, Trainforge storing flat IDs in `learning_outcome_refs` and week-scoped IDs in a new `pedagogical_scope_refs` field. Orphan week-scoped IDs (legacy content, drift) are preserved with `parent_id: null` so the defect surfaces in metrics rather than disappearing. (§2.1.)

3. **Mis-scoped `follows_chunk` (~68% of chain links cross lesson boundaries).** Now reset at every lesson and module boundary; violations are reported as `integrity.follows_chunk_boundary_violations`. (§4.3.)

4. **Concept graph is a tag co-occurrence graph, not a semantic knowledge graph.** The README previously overclaimed. Fixed in v0.1.x: edges carry `relation_type: "co-occurs"` (forward-compatible for the typed extractor), pedagogy / logistics tags are partitioned into a separate `pedagogy_graph.json`, and a `concept_graph_semantic.json` filename was reserved for a later typed extractor. The typed extractor itself shipped in v0.2.0 (Worker F); the reserved filename is now populated with typed edges carrying `relation_type` ∈ {`prerequisite`, `is-a`, `related-to`, `co-occurs`}, `confidence`, and `provenance`. (§2.2, §3.1, §5a.)

5. **Quality report dishonesty.** `html_preservation_rate: 1.0` while 62% of chunks had unclosed `<div>` tags; `learning_outcome_refs_coverage: 1.0` while 60% of refs were unresolvable. Both measured field presence, not correctness. Metrics rewritten to measure real structure and resolution; new `methodology` block in the report documents semantics; `metrics_semantic_version: 2` constant tells downstream consumers to re-baseline. (§1.1–1.4.)

6. **Half-populated enrichment fields.** `bloom_level` 87%, `key_terms` 53%, `misconceptions` 54%, `content_type_label` 53%. **Investigation-first in this branch** (see §4 below). Fallback helpers exist in `process_course.py` but are deliberately unwired until the investigation concludes.

7. **SC name drift.** Sixteen WCAG success criteria appeared under inconsistent names (`Contrast Minimum`, `Contrast Minimum, Level AA`, `Contrast (Minimum)` …). Canonicalisation now applied to chunk text, key-terms metadata, misconceptions, and **concept tags** — the last is where retrieval sharpness actually lives, because the graph fragments otherwise. (§4.5.)

8. **Factual inaccuracies.** The v0.1.0 real-domain content made a domain-specific numeric claim that conflicted with authoritative sources and contained an internal arithmetic contradiction. A new `ContentFactValidator` (§4.6) flags both numeric-claim mismatches and internal-arithmetic contradictions. Warning-only today.

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

The gates `outcome_ref_integrity` and `content_fact_check` ship in `config/workflows.yaml` at `severity: warning`. The follow-up PR flips them to `critical` and turns on `strict_mode=True` by default. The flip is contingent on **two** events together, not a calendar date:

> 1. **Synthetic floor.** `Trainforge/tests/fixtures/mini_course_clean/` runs green in CI with `metrics.footer_contamination_rate == 0`, `integrity.broken_refs == []`, and `integrity.factual_inconsistency_flags == []`.
>
> 2. **Real-domain floor.** A clean v1.0 regeneration of the v0.1.0 baseline corpus (or another real domain corpus, see §6(b)) produces a `quality_report.json` with the same three integrity invariants holding. The `archive/v0.1.0-baseline/` snapshot exists so this regeneration has a comparator.

The synthetic floor proves the code paths work; the real-domain floor proves the architecture handles the messiness fixtures can't simulate. Either alone is a weaker bar than the NSF narrative implies — both must hold. The follow-up PR cannot cite "CI green" alone as justification for the flip.

---

## §4 §4.4a enrichment-coverage investigation (result slot)

**Status at time of writing:** investigation deferred — see the implementation plan (`claude/fix-package-quality-FyMue` branch, plan file). The §4.4a investigation asks which of four hypotheses dominates the 47% enrichment miss rate:

- **H1** JSON-LD `sections` keyed by a heading that doesn't match post-merge `section_heading`.
- **H2** JSON-LD `sections` is genuinely empty on many pages.
- **H3** `content_type_label` short-circuit in `_extract_section_metadata` produces half-populated chunks.
- **H4** The "no sections" code path in `_chunk_content` never invokes `_extract_section_metadata` at all.
- **H5** The JSON-LD parser silently fails on edge cases (malformed JSON, unexpected schema variants, encoding quirks) and the chunker treats the parse failure as "metadata absent" rather than "metadata present but unreadable." Distinguished from H2 because the fix is in the parser, not the source.

The investigation MUST complete before any fallback helper (`derive_bloom_from_verbs`, `extract_key_terms_from_html`, `extract_misconceptions_from_text`) is wired into `_create_chunk`. If the root cause is structural (H1/H3/H4/H5), the fix is at the source or in the parser, not in fallback regex. If the root cause is H2, fallbacks are appropriate.

The helpers exist in `Trainforge/process_course.py` at module scope and are unit-tested. They will be deleted if unused after the investigation concludes — dead code masking a fixable bug is worse than a known gap.

---

## §4b Architectural decisions — explicit deferrals on this branch

The v1 plan committed to "ownership: both" for footer contamination — Courseforge moves copyright into a `data-cf-role="template-chrome"` region, Trainforge skips that role *and* runs an n-gram defensive layer. Likewise, "Courseforge emits both" was the dual outcome-ID decision, requiring `course.json` to carry course-level + week-scoped IDs with parent links.

This branch ships **only the Trainforge half of both decisions.** The Courseforge-side template change and dual-emission are not in this commit. That is a real drift from the plan, and the right move is to acknowledge it in writing rather than leave it as an unspoken gap.

| Decision | Trainforge side (this PR) | Courseforge side (follow-up) |
|---|---|---|
| Footer ownership | n-gram detector strips repeated spans; metric reports contamination rate | Move copyright out of page body into `<footer data-cf-role="template-chrome">`; add a selector-based skip in Trainforge so role-tagged chrome is dropped before n-gram detection runs |
| Outcome-ID contract | `learning_outcome_refs` holds course-level IDs; `pedagogical_scope_refs` holds week-scoped IDs with `parent_id` (orphans preserved with `parent_id: null`) | Emit both forms in `course.json` with explicit parent links so orphan counts stay zero on healthy content |

**Follow-up branch:** the v1.0 work that completes both halves is owned by the same maintainer (`mdmurphy822`) and lives on a branch named `claude/courseforge-template-chrome-and-dual-ids` (to be created). Until that branch ships:

- The "ownership: both" entry in the v0 plan's decision table is *partially fulfilled*, not retracted.
- The Trainforge defensive layer is **load-bearing**: on a small corpus or against novel template chrome, the n-gram threshold may not fire and footer contamination will leak through. The metric will surface the leak; nothing will refuse to write it. This is acceptable for v0.1.x but is the principal reason `strict_mode=True` is not on by default.
- Selector-based skip for `[data-cf-role="template-chrome"]` is **not present** in this PR. When Courseforge starts emitting the role attribute, this skip must land in `Trainforge/process_course.py` (in or alongside `_detect_corpus_boilerplate`) in the same PR as the Courseforge template change.

### What this means for the severity flip

The "real-domain floor" requirement in §3 (Severity flip trigger) cannot be satisfied until the Courseforge-side work is done. A v1.0 real-domain regeneration with Courseforge still emitting body-embedded copyright will keep the n-gram detector load-bearing, and the strict-mode integrity gate would be operating on top of a defensive layer rather than a clean source. The severity flip is therefore implicitly blocked on the Courseforge follow-up — that should be made explicit in the follow-up PR description.

---

## §5a v0.2.0 — what shipped on `dev-v0.2.0`

The `dev-v0.2.0` branch is the consolidation point for Workers A through K plus two post-merge follow-ups (Worker L anonymization, the dev-branch README rewrite). The v0.2.0 bump is an *intermediate* release that fulfils a sizeable subset of the §5 v1.0 roadmap without yet satisfying all the §6 v1.0 exit criteria; the version number moves from v0.1.x to v0.2.0 because the pipeline now carries capabilities that go meaningfully beyond the v0.1.x self-trust scaffolding.

Concrete shape on `dev-v0.2.0`:

- **Cross-worker contracts documented** — `docs/architecture/ADR-001-pipeline-shape.md` (Worker A) names the chunk-schema, metrics-semantic-version, and fixture-naming contracts every concurrent worker shares, so the B/D/E `v4` schema bump and the B-owned `METRICS_SEMANTIC_VERSION` bump happen once per release train instead of racing.
- **Flow metrics in `quality_report.json`** (Worker B, `METRICS_SEMANTIC_VERSION` 3→4). Five new observability metrics surface silent parser→chunk metadata drops: `content_type_label_coverage`, `key_terms_coverage`, `key_terms_with_definitions_rate`, `misconceptions_present_rate`, `interactive_components_rate`. Two attach `integrity.*` chunk-ID lists for targeted follow-up. See `docs/metrics/flow-metrics.md`.
- **Training-pair synthesis (SFT + DPO)** (Worker C). `Trainforge/synthesize_training.py` emits instruction pairs per chunk with a schema committed at `schemas/instruction_pair.schema.json`, a deterministic-template path for the mock provider, and full decision-capture trails.
- **Per-chunk summaries + retrieval benchmark** (Worker D). `CHUNK_SCHEMA_VERSION` goes to `v4`. Every chunk carries a 40-400 char extractive `summary`; chunks with key-terms also carry a `retrieval_text` field (summary + key terms). `Trainforge/rag/retrieval_benchmark.py` exercises recall@k across the `text`, `summary`, and `retrieval_text` variants on the `mini_course_summaries` fixture.
- **Chunk provenance (audit trail)** (Worker E). Every chunk carries `source.html_xpath` and `source.char_span` so Section 508 / ADA Title II buyers can round-trip `chunk.text` to its source IMSCC HTML. Invariants (span non-overflow, multi-part disjointness/contiguity) are locked in `Trainforge/tests/test_provenance.py`; opt-in end-to-end tests run against any locally regenerated corpus via `TRAINFORGE_PROVENANCE_CORPUS`.
- **Typed-edge concept graph** (Worker F). The `concept_graph_semantic.json` filename that v0.1.x *reserved* is now populated: rule-based inference from co-occurrence, typed-LO proximity, and optional LLM extraction produces typed edges (`prerequisite`, `is-a`, `related-to`, `co-occurs`) with `confidence` and `provenance`. The existing `concept_graph.json` remains the authoritative untyped graph.
- **Cross-package concept index** (Worker G). `libv2 cross-index` aggregates every course's `graph/concept_graph.json` (and, when present, the typed semantic graph) into `LibV2/catalog/cross_package_concepts.json` — a navigation layer that answers "given concept X, which other courses in this repo cover it?" Freshness checked by `lib/libv2_fsck.py`. The catalog file is intentionally not tracked in git (see §5b on anonymization); users regenerate it locally on demand.
- **Per-week `learningObjectives` specificity** (Worker H). `Courseforge/scripts/generate_course.py` takes `--objectives <canonical-registry.json>`; each week's emitted JSON-LD now references only canonical CO/TO IDs declared for that week's chapter range. Closes the LO-fanout defect where `outcome_reverse_coverage` collapsed to 0.143. `validate_page_objectives.py` locks the invariant.
- **Packager pre-build LO-validation gate** (Worker I). IMSCC packaging now validates the full LO JSON-LD contract before tarring, so a course that regressed on Worker H's fix cannot ship an IMSCC package silently.
- **LibV2 reference retrieval** (Worker J). ADR-002 names the scope line: `libv2 retrieve` / `libv2 retrieval-eval`, rationale payload, three metadata-aware boost functions (concept-graph overlap, LO match, prereq coverage), `ChunkFilter` with eleven v4 metadata fields, and structured tokenization that preserves `sc-1.4.3`/`aria-labelledby`-style slugs. Opt-in rationale payload is back-compat-pinned (`TestWorkerJBackCompat`). See `docs/libv2/reference-retrieval.md`.
- **Anonymization policy enforced** (Workers K + L). The repository no longer ships example-course slugs, course-specific codes, or per-course retrieval data. Verified by a repo-wide grep: zero tracked occurrences of the example-course strings the real-domain v0.1.0 corpus used. `.gitignore` is tightened so course subtrees stay under the user's control. The retrieval-eval contract is exercised by a three-chunk synthetic fixture in-test (`LibV2/tools/libv2/tests/test_eval_harness_retrieval.py`); users curate their own gold queries against their own loaded courses using the workflow in `docs/libv2/reference-retrieval.md`.

**What v0.2.0 still does NOT include (see §5 and §6):**

- Strict mode is not default-on. The Courseforge template-chrome separation (§4b) is still deferred; the defensive n-gram boilerplate stripper in Trainforge remains load-bearing.
- Severity flip for `outcome_ref_integrity` and `content_fact_check` is still pending both the synthetic floor and the real-domain floor (§3 Severity flip trigger).
- The §4.4a enrichment-coverage investigation has not concluded; the fallback helpers in `Trainforge/process_course.py` remain unwired.
- Domain-agnostic validation (§6(b)) — "run against ≥3 distinct domain corpora with no new defect classes" — has not been completed.
- SC canonicalisation still covers the variant table, not every WCAG 2.2 SC (§5 item 8).

## §5b Anonymization policy (v0.2.0)

As of v0.2.0, the repository's public tree is course-agnostic. All example-course references in docs, scripts, schemas, and tests use generic placeholders (`SAMPLE_101`, `sample-course`, `<your-course-slug>`, `sample_course_chunk_00042`). The following artifacts are intentionally **not** tracked in git and live only in the user's local checkout:

- `LibV2/courses/<slug>/` course subtrees (`corpus/chunks.jsonl`, `graph/`, `retrieval/gold_queries.jsonl`, `retrieval/README.md`, `quality/`, ...).
- `LibV2/catalog/cross_package_concepts.json` (regenerated on demand via `libv2 cross-index`).

The reference-retrieval contract is still fully exercisable — `LibV2/tools/libv2/tests/test_eval_harness_retrieval.py` builds a three-chunk synthetic course with a two-query gold set inside `tmp_path`, so `evaluate_retrieval` is regression-tested end-to-end without any tracked per-course data. `docs/libv2/reference-retrieval.md` documents how users curate their own per-course gold queries.

## §5 v1.0 roadmap

Ordered by what the work currently on the v1.0 branch list looks like. Items marked "(shipped in v0.2.0)" are implemented on `dev-v0.2.0`; they remain on this list because the §6 exit criteria have not all been met, i.e., the pipeline as a whole has not yet passed the domain-agnostic + self-trust bar that makes v1.0 stand behind the roadmap.

1. **§4.4a investigation complete** — this is the next concrete blocking item.
2. **Typed-edge concept extractor** (shipped in v0.2.0, Worker F). `concept_graph_semantic.json` is populated; edges carry `relation_type` ∈ {`prerequisite`, `is-a`, `related-to`, `co-occurs`} plus `confidence` and `provenance`. Remaining on the v1.0 path because v1.0 expects this to be the default retrieval surface; in v0.2.0 it is additive to the untyped graph.
3. **Strict mode on by default.** See §Severity flip trigger. Not yet default in v0.2.0.
4. **`outcome_ref_integrity` and `content_fact_check` promoted to `critical`.** Not yet promoted in v0.2.0.
5. **Dual outcome-ID contract shipped.** Courseforge emits both flat and week-scoped IDs with explicit parent links; Trainforge consumes both fields. Courseforge side not yet shipped in v0.2.0 (partial — the Worker-H per-week specificity work is a different defect on the same code path and does ship).
6. **Template-chrome footer separation.** Courseforge stops emitting copyright in the page body. Not yet shipped in v0.2.0; n-gram defensive layer remains load-bearing.
7. **Enrichment coverage resolved.** Per §4.4a outcome. Investigation not yet complete in v0.2.0.
8. **SC canonicalisation extended** to every SC mentioned in WCAG 2.2, not just the handful currently in the variant table.
9. **`ContentFactValidator` regex table broadened** with domain-specific content-fact rules on contact with real curricula.

---

## §6 v1.0 exit criteria (explicit)

v1.0 is not a marketing milestone. It is a concrete set of conditions all of which must hold:

- **(a) Self-trust.** `mini_course_clean/` runs green with `strict_mode=True` and zero integrity violations.
- **(b) Domain-agnostic validation.** The pipeline has been run against **≥3 distinct domain corpora** with no new defect classes surfaced. One is the original v0.1.0 real-domain corpus (accessibility); the other two must be outside that domain (for example: a STEM textbook and a humanities textbook). The author should deliberately pick domains they are *less* expert in, to stress-test whether the defects surfaced in v0.1.0 are universal pipeline issues or domain-specific artifacts. This is the single test that tells us whether Ed4All generalises.
- **(c) Severity flip completed.** Both `outcome_ref_integrity` and `content_fact_check` at `critical`. `strict_mode=True` default.
- **(d) README matches reality.** No paragraph overclaims what the graph or the metrics deliver. The reverse is fine — selling short is always safer than the claim/reality gap a sophisticated reviewer will immediately notice.

---

## §7 Grant-narrative framing (NSF TechAccess "AI-Ready America")

Ed4All is the strongest portfolio piece under the NSF TechAccess framing. DART is one stage of it; the other three stages (Courseforge, Trainforge, LibV2) are the differentiator. The v0.1.0 real-domain corpus (an accessibility-mission course, held locally, not shipped in this repo) is a genuine demonstration of the pipeline working end-to-end on a domain that sits directly inside Ed4All's mission.

For a proposal, the claim structure is:

- **Here is v0.1.0 output** (the real-domain package and its defects).
- **Here is the defect analysis** (the nine signals documented in §2).
- **Here is the v0.1.0 → v1.0 roadmap** (this document).
- **Here is v1.0 output on the same domain** (the regenerated package once v1.0 is shipped).
- **Here are the measured deltas** (footer contamination, outcome ref integrity, graph fragmentation, quality-report trustworthiness).

That structure is stronger than any "here's a pipeline we built" narrative because it demonstrates a *method*: measure, characterise, remediate, re-measure. Funded proposals reward method.

### Paired before/after artifacts — archive scaffold shipped, population owed

This branch ships an empty `archive/v0.1.0-baseline/` scaffold with an `ARCHIVE_README.md` that names what must go there. The scaffold exists in the tree so the obligation is structural, not a todo on someone's list. The artifact itself was not present in the environment this branch was developed in, so the scaffold is empty pending action by the repo owner (`mdmurphy822`).

Before the pipeline moves past v0.1.x, the maintainer must populate the scaffold with:

1. The v0.1.0 real-domain artifact as shipped (full Trainforge output dir tree — manifest.json, course.json, corpus/, graph/, pedagogy/, quality/, training_specs/).
2. The original v0.1.0 `quality_report.json` exactly as it was emitted (the dishonest scores).
3. Optional: a `quality_report_rescored_v2.json` produced by re-running the v0.1.x self-trust metrics against the same unchanged chunks. Same input, two metric generations, side-by-side comparator.

The v1.0 regeneration and delta table follow once v1.0 ships and are tracked on that branch.

The reason this can't be deferred to "the v1.0 branch will produce both": once the chunker, the metrics, the canonicalisation, the orphan rule, and the pedagogy graph split are all on `main` (which they are after this PR merges), regenerating the v0.1.0 artifact byte-for-byte becomes structurally impossible. Either the maintainer holds a copy outside this checkout and commits it, or the `archive/v0.1.0-baseline/ARCHIVE_README.md` fallback (rebuild from commit `18c6613`) is invoked, with the divergence documented.

---

## §8 Cross-worker coordination (schema, metrics, branch policy)

§1–§7 above describe v0.1.0 shape and the v0.1.x → v1.0 path. §8 describes how multiple concurrent workers (the `worker-*` branch family) keep shared constants and shared files from racing. The operational detail lives in [`docs/architecture/ADR-001-pipeline-shape.md`](docs/architecture/ADR-001-pipeline-shape.md) and [`docs/contributing/workers.md`](docs/contributing/workers.md); this section is the canonical top-level pointer.

### §8.1 Chunk-schema-version policy

The chunk object carries a `schema_version` string; `manifest.json` carries a matching `chunk_schema_version`. The current implied value is `"v3"`. Workers B, D, and E each add chunk fields and therefore all require the same bump to `"v4"`.

- The bump is **batched** across B/D/E on a shared rebase branch `chunk-schema-v4`.
- No worker bumps `CHUNK_SCHEMA_VERSION` independently. One bump per release train.
- Full protocol: see ADR-001 Contract 1.

### §8.2 `METRICS_SEMANTIC_VERSION` ownership

`METRICS_SEMANTIC_VERSION` lives at `Trainforge/process_course.py:58`. It is owned by the **base pass** and governs the `metrics` block in `quality_report.json`.

- Worker B owns the v3 → v4 bump (adds five flow metrics).
- Subsequent bumps are coordinated through the append-only decision log at the bottom of ADR-001.
- The alignment pass does NOT bump this constant. Alignment declares which base version it was computed against via `alignment.base_metrics_semantic_version`; downstream readers compare that integer against `metrics_semantic_version` to detect a stale re-run.
- Full protocol: see ADR-001 Contract 2.

### §8.3 Worker-coordination branch protocol

- Branch names: `worker-<letter>/<slug>`. PR label: `worker-<letter>`.
- Workers never share branches except the `chunk-schema-v4` rebase point for B/D/E.
- Shared test fixtures under `Trainforge/tests/fixtures/` follow the `mini_course_<purpose-slug>` naming lock. Every new fixture ships a `README.md`.
- Full protocol: see ADR-001 Contracts 4 and 5.
