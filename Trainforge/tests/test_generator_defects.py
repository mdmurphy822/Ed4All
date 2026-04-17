"""Regression tests for the nine defects documented in VERSIONING.md.

Each test class targets one defect. Where possible, tests exercise the
pipeline function directly against small inline fixtures; the shared HTML
fixtures under ``fixtures/mini_course_*`` cover flows that need real parser
input. Helpers that don't need HTML use inline strings so a failure names
exactly one defect class.

Tests are intentionally small and deterministic — no IMSCC zip construction,
no network, no fixtures larger than a handful of lines per file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
CLEAN_DIR = FIXTURE_DIR / "mini_course_clean"
DEFECTIVE_DIR = FIXTURE_DIR / "mini_course_defective"
EDGE_DIR = FIXTURE_DIR / "mini_course_edge"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _chunk(**overrides):
    """Build a minimal chunk dict with sensible defaults."""
    base = {
        "id": "mini_chunk_00001",
        "chunk_type": "explanation",
        "text": "Sample chunk text.",
        "html": "<p>Sample chunk text.</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "MINI_101",
            "module_id": "m1",
            "module_title": "Module 1",
            "lesson_id": "w01",
            "lesson_title": "Week 1",
            "resource_type": "page",
            "section_heading": "Intro",
            "position_in_module": 0,
        },
        "concept_tags": [],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 10,
        "word_count": 3,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Defect 1 — Footer contamination
# ---------------------------------------------------------------------------

class TestBoilerplateDetector:
    def test_footer_detected_across_pages(self):
        from Trainforge.rag.boilerplate_detector import detect_repeated_ngrams

        footer = "Copyright 2026 ACME FIX ME Learning Systems All rights reserved content educational use"
        docs = [f"Page {i} unique preamble. {footer}" for i in range(4)]

        spans = detect_repeated_ngrams(docs, n=10, min_doc_frac=0.5)

        assert any("ACME" in s and "FIX" in s for s in spans), spans

    def test_strip_removes_span(self):
        from Trainforge.rag.boilerplate_detector import strip_boilerplate

        text = "Body content here. Copyright 2026 ACME FIX ME Learning Systems boilerplate trailing."
        cleaned, removed = strip_boilerplate(text, ["Copyright 2026 ACME FIX ME Learning Systems"])

        assert removed == 1
        assert "ACME" not in cleaned
        assert "Body content here." in cleaned

    def test_contamination_rate_counts_chunks(self):
        from Trainforge.rag.boilerplate_detector import contamination_rate

        footer = "ACME FIX ME"
        chunks = [
            {"text": f"chunk {i} text {footer}"} for i in range(3)
        ] + [
            {"text": "clean chunk"}
        ]
        rate = contamination_rate(chunks, [footer])
        assert rate == pytest.approx(0.75)

    def test_empty_inputs_return_empty(self):
        from Trainforge.rag.boilerplate_detector import detect_repeated_ngrams

        assert detect_repeated_ngrams([]) == []
        assert detect_repeated_ngrams(["only one doc"], n=3, min_doc_frac=0.5) == []


# ---------------------------------------------------------------------------
# Defect 2 — Broken outcome refs (referential integrity)
# ---------------------------------------------------------------------------

class TestOutcomeReferentialIntegrity:
    def test_broken_ref_listed_in_report(self):
        from Trainforge.process_course import CourseProcessor

        chunks = [
            _chunk(id="c1", learning_outcome_refs=["co-01"]),
            _chunk(id="c2", learning_outcome_refs=["w99-co-99"]),
            _chunk(id="c3", learning_outcome_refs=["co-02", "co-99"]),
        ]
        valid = {"co-01", "co-02", "w01-co-01", "w01-co-02"}

        broken = CourseProcessor._collect_broken_refs(chunks, valid)

        assert {(b["chunk_id"], b["ref"]) for b in broken} == {
            ("c2", "w99-co-99"),
            ("c3", "co-99"),
        }

    def test_lo_coverage_counts_only_resolving_refs(self):
        from Trainforge.process_course import CourseProcessor

        chunks = [
            _chunk(id="c1", learning_outcome_refs=["co-01"]),            # resolves
            _chunk(id="c2", learning_outcome_refs=["w99-co-99"]),        # broken
            _chunk(id="c3", learning_outcome_refs=[]),                    # empty
        ]
        valid = {"co-01"}

        rate = CourseProcessor._resolving_lo_coverage(chunks, valid)

        assert rate == pytest.approx(1 / 3)


class TestOrphanWeekScopedRefs:
    def test_orphan_week_scoped_id_preserved_with_null_parent(self):
        from Trainforge.align_chunks import partition_outcome_refs

        chunks = [
            _chunk(id="c1", learning_outcome_refs=["w05-co-99", "co-01"]),
        ]
        parent_map = {"w01-co-01": "co-01"}
        course_level = {"co-01"}

        orphans = partition_outcome_refs(chunks, parent_map, course_level)

        assert orphans == 1
        scope = chunks[0]["pedagogical_scope_refs"]
        assert len(scope) == 1
        assert scope[0]["id"] == "w05-co-99"
        assert scope[0]["parent_id"] is None
        assert scope[0]["status"] == "orphan"
        # Course-level IDs untouched
        assert "co-01" in chunks[0]["learning_outcome_refs"]

    def test_resolved_week_scoped_id_carries_parent(self):
        from Trainforge.align_chunks import partition_outcome_refs

        chunks = [
            _chunk(id="c1", learning_outcome_refs=["w01-co-01"]),
        ]
        parent_map = {"w01-co-01": "co-01"}
        course_level = {"co-01"}

        orphans = partition_outcome_refs(chunks, parent_map, course_level)

        assert orphans == 0
        scope = chunks[0]["pedagogical_scope_refs"]
        assert scope[0]["parent_id"] == "co-01"
        assert scope[0]["status"] == "resolved"
        # Parent is also promoted into learning_outcome_refs.
        assert "co-01" in chunks[0]["learning_outcome_refs"]


# ---------------------------------------------------------------------------
# Defect 3 — follows_chunk lesson-scoped
# ---------------------------------------------------------------------------

class TestFollowsChunkBoundaries:
    def test_violations_detected(self):
        from Trainforge.process_course import CourseProcessor

        chunks = [
            _chunk(id="a", source={**_chunk()["source"], "lesson_id": "w01"}),
            _chunk(id="b", follows_chunk="a",
                   source={**_chunk()["source"], "lesson_id": "w02"}),
            _chunk(id="c", follows_chunk="b",
                   source={**_chunk()["source"], "lesson_id": "w02"}),
        ]
        violations = CourseProcessor._follows_chunk_violations(chunks)
        assert len(violations) == 1
        assert violations[0]["chunk_id"] == "b"
        assert violations[0]["reason"] == "cross_lesson"

    def test_no_violations_for_in_lesson_chain(self):
        from Trainforge.process_course import CourseProcessor

        chunks = [
            _chunk(id="a", source={**_chunk()["source"], "lesson_id": "w01"}),
            _chunk(id="b", follows_chunk="a",
                   source={**_chunk()["source"], "lesson_id": "w01"}),
        ]
        assert CourseProcessor._follows_chunk_violations(chunks) == []


# ---------------------------------------------------------------------------
# Defect 4 — Concept / pedagogy graph split
# ---------------------------------------------------------------------------

class TestConceptGraphPartition:
    def test_pedagogy_tags_excluded_from_concept_graph(self):
        from Trainforge.process_course import CourseProcessor

        proc = CourseProcessor.__new__(CourseProcessor)
        chunks = [
            _chunk(id="c1", concept_tags=["behaviorism", "apply", "cognitivism"]),
            _chunk(id="c2", concept_tags=["behaviorism", "analyze"]),
            _chunk(id="c3", concept_tags=["cognitivism", "scaffolding"]),
        ]
        graph = proc._generate_concept_graph(chunks)

        node_ids = {n["id"] for n in graph["nodes"]}
        assert "apply" not in node_ids
        assert "analyze" not in node_ids
        assert "behaviorism" in node_ids
        assert all(edge.get("relation_type") == "co-occurs" for edge in graph["edges"])

    def test_pedagogy_graph_captures_pedagogy_tags(self):
        from Trainforge.process_course import CourseProcessor

        proc = CourseProcessor.__new__(CourseProcessor)
        chunks = [
            _chunk(id="c1", concept_tags=["apply", "analyze", "behaviorism"]),
            _chunk(id="c2", concept_tags=["apply", "behaviorism"]),
        ]
        ped = proc._generate_pedagogy_graph(chunks)
        node_ids = {n["id"] for n in ped["nodes"]}
        assert "apply" in node_ids
        assert "behaviorism" not in node_ids


# ---------------------------------------------------------------------------
# Defect 5 — Quality report honesty
# ---------------------------------------------------------------------------

class TestQualityReportHonesty:
    def test_html_balance_check_catches_unclosed_div(self):
        from Trainforge.process_course import CourseProcessor

        assert CourseProcessor._html_is_well_formed("<p>hi</p>") is True
        assert CourseProcessor._html_is_well_formed("<div><p>hi</p>") is False
        assert CourseProcessor._html_is_well_formed("") is False
        assert CourseProcessor._html_is_well_formed("<br/><hr>plain") is True

    def test_metrics_semantic_version_is_written(self):
        from Trainforge.process_course import METRICS_SEMANTIC_VERSION, CourseProcessor

        proc = CourseProcessor.__new__(CourseProcessor)
        proc.stats = {"total_words": 100, "total_chunks": 1}
        proc._boilerplate_spans = []
        proc._valid_outcome_ids = {"co-01"}
        proc._factual_flags = []
        proc.MIN_CHUNK_SIZE = 100
        proc.MAX_CHUNK_SIZE = 800
        chunks = [
            _chunk(word_count=120, html="<p>ok</p>", learning_outcome_refs=["co-01"]),
        ]
        report = proc._generate_quality_report(chunks)
        assert report["metrics_semantic_version"] == METRICS_SEMANTIC_VERSION
        assert "methodology" in report
        assert report["integrity"]["broken_refs"] == []


class TestStrictMode:
    def _build_processor(self, *, strict_mode: bool):
        from Trainforge.process_course import CourseProcessor

        proc = CourseProcessor.__new__(CourseProcessor)
        proc.strict_mode = strict_mode
        proc.stats = {"total_chunks": 10}
        return proc

    def test_strict_mode_raises_on_broken_refs(self):
        from Trainforge.process_course import PipelineIntegrityError

        proc = self._build_processor(strict_mode=True)
        report = {
            "integrity": {
                "broken_refs": [{"chunk_id": "c1", "ref": "w99-co-99"}],
                "follows_chunk_boundary_violations": [],
                "html_balance_violations": [],
            }
        }
        with pytest.raises(PipelineIntegrityError):
            proc._assert_integrity(report)

    def test_strict_mode_passes_on_clean_report(self):
        proc = self._build_processor(strict_mode=True)
        report = {
            "integrity": {
                "broken_refs": [],
                "follows_chunk_boundary_violations": [],
                "html_balance_violations": [],
            }
        }
        proc._assert_integrity(report)  # must not raise

    def test_non_strict_mode_never_raises(self):
        proc = self._build_processor(strict_mode=False)
        report = {
            "integrity": {
                "broken_refs": [{"chunk_id": "c1", "ref": "w99-co-99"}],
                "follows_chunk_boundary_violations": [{"chunk_id": "c2"}],
                "html_balance_violations": [{"chunk_id": "c3", "unclosed_tags": ["div"]}] * 20,
            }
        }
        proc._assert_integrity(report)  # must not raise


# ---------------------------------------------------------------------------
# Defect 6 — Enrichment fall-through fallbacks (helpers; not wired in this PR)
# ---------------------------------------------------------------------------

class TestEnrichmentHelpers:
    def test_bloom_derived_from_verbs(self):
        from Trainforge.process_course import derive_bloom_from_verbs

        text = "Evaluate, critique, and justify the design choices made in this lesson."
        assert derive_bloom_from_verbs(text) == "evaluate"

    def test_bloom_returns_none_on_empty_text(self):
        from Trainforge.process_course import derive_bloom_from_verbs

        assert derive_bloom_from_verbs("") is None

    def test_key_terms_from_bold_tags(self):
        from Trainforge.process_course import extract_key_terms_from_html

        html = ("<p>The term <strong>scaffolding</strong> refers to structured learning support. "
                "A <dfn>rubric</dfn> is a scoring guide.</p>")
        terms = extract_key_terms_from_html(html)
        assert any(t["term"].lower() == "scaffolding" for t in terms)
        assert any(t["term"].lower() == "rubric" for t in terms)

    def test_misconception_patterns_detected(self):
        from Trainforge.process_course import extract_misconceptions_from_text

        text = (
            "Common mistake: assuming Bloom's levels are strictly hierarchical. "
            "Students often think that 'apply' must follow 'understand' linearly."
        )
        found = extract_misconceptions_from_text(text)
        assert len(found) >= 1


# ---------------------------------------------------------------------------
# Defect 7 — SC name canonicalization (text + tags)
# ---------------------------------------------------------------------------

class TestSCCanonicalization:
    def test_contrast_minimum_variants_normalized_in_text(self):
        from Trainforge.rag.wcag_canonical_names import canonicalize_sc_references

        variants = [
            "Contrast Minimum",
            "Contrast Minimum, Level AA",
            "Contrast Minimum, 4.5:1 for normal text",
        ]
        canonical = [canonicalize_sc_references(v) for v in variants]
        assert all("Contrast (Minimum)" in c for c in canonical), canonical

    def test_keyboard_trap_variants_normalized(self):
        from Trainforge.rag.wcag_canonical_names import canonicalize_sc_references

        for variant in ["No Keyboard Trap", "No Keyboard Trap , Level A", "No Keyboard Trap, Level A"]:
            assert "No Keyboard Trap" in canonicalize_sc_references(variant)

    def test_canonicalize_sc_tag_collapses_drift(self):
        from Trainforge.rag.wcag_canonical_names import canonicalize_sc_tag

        assert canonicalize_sc_tag("contrast-minimum-level-aa") == "contrast-minimum"
        assert canonicalize_sc_tag("no-keyboard-trap-level-a") == "no-keyboard-trap"
        assert canonicalize_sc_tag("some-other-tag") == "some-other-tag"


# ---------------------------------------------------------------------------
# Defect 8 — Factual accuracy
# ---------------------------------------------------------------------------

class TestContentFactValidator:
    def test_87_sc_flagged(self):
        from lib.validators.content_facts import ContentFactValidator

        flags = ContentFactValidator().check_text("WCAG 2.2 contains 87 success criteria.")
        assert any(f["claim"] == "wcag_2_2_sc_count" and f["observed"] == 87 for f in flags)

    def test_86_sc_passes(self):
        from lib.validators.content_facts import ContentFactValidator

        flags = ContentFactValidator().check_text("WCAG 2.2 contains 86 success criteria.")
        assert not any(f["claim"] == "wcag_2_2_sc_count" for f in flags)

    def test_arithmetic_contradiction_flagged(self):
        from lib.validators.content_facts import ContentFactValidator

        text = "WCAG 2.2 has 87 success criteria: Perceivable (29), Operable (29), Understandable (17), Robust (4)."
        flags = ContentFactValidator().check_text(text)
        assert any(f["claim"] == "wcag_2_2_sc_arithmetic" for f in flags)

    def test_historical_wcag_20_claim_suppressed(self):
        from lib.validators.content_facts import ContentFactValidator

        # WCAG 2.0 historically shipped with 61 SC. Mentioning that here
        # should not flag against the WCAG 2.2 expected value of 86.
        text = "WCAG 2.0 historically had 61 success criteria across four principles."
        flags = ContentFactValidator().check_text(text)
        assert not any(f["claim"] == "wcag_2_2_sc_count" for f in flags)

    def test_previously_keyword_suppresses(self):
        from lib.validators.content_facts import ContentFactValidator

        text = "The spec previously contained 50 success criteria; today it lists 86."
        flags = ContentFactValidator().check_text(text)
        # Neither the historical "50" nor the present-tense "86" should flag.
        assert not any(f["claim"] == "wcag_2_2_sc_count" for f in flags)

    def test_section_508_count_still_flags_when_wrong(self):
        from lib.validators.content_facts import ContentFactValidator

        text = "There are 99 applicable WCAG 2.0 A and AA SC under Section 508."
        flags = ContentFactValidator().check_text(text)
        # The Section 508 expected count is 38; suppressor must not blanket-skip it.
        assert any(f["claim"] == "section_508_sc_count" for f in flags)

    def test_arithmetic_suppressed_under_historical_framing(self):
        from lib.validators.content_facts import ContentFactValidator

        text = "WCAG 2.0 used to have 25 success criteria across 4 principles: 12, 8, 4, 1."
        flags = ContentFactValidator().check_text(text)
        assert not any(f["claim"] == "wcag_2_2_sc_arithmetic" for f in flags)


# ---------------------------------------------------------------------------
# Defect 9 — leak_check corpus-wide boilerplate
# ---------------------------------------------------------------------------

class TestLeakCheckerBoilerplate:
    def test_reports_boilerplate_above_threshold(self):
        from lib.leak_checker import LeakChecker

        footer = "ACME FIX ME Learning Systems copyright 2026 all rights reserved educational"
        chunks = [{"id": f"c{i}", "text": f"page {i} body. {footer}"} for i in range(5)]
        reports = LeakChecker().check_corpus_boilerplate(chunks, n=10, threshold=0.10)
        assert any("ACME" in (r.matched_text or "") for r in reports)

    def test_no_report_when_below_threshold(self):
        from lib.leak_checker import LeakChecker

        chunks = [{"id": f"c{i}", "text": f"unique page {i} body."} for i in range(5)]
        reports = LeakChecker().check_corpus_boilerplate(chunks, n=10, threshold=0.10)
        assert reports == []


# ---------------------------------------------------------------------------
# Shared fixture sanity checks
# ---------------------------------------------------------------------------

class _NullCapture:
    """Stub DecisionCapture — records calls so tests can assert on them."""

    def __init__(self):
        self.calls = []

    def log_decision(self, **kwargs):
        self.calls.append(kwargs)


def _bare_processor():
    """A CourseProcessor skipped through __init__, hydrated with only the
    attributes the methods-under-test read. Keeps these unit tests free of
    IMSCC fixtures, filesystem, and DecisionCapture side-effects.
    """
    from collections import defaultdict
    from Trainforge.process_course import CourseProcessor

    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "MINI_101"
    proc.capture = _NullCapture()
    proc.stats = {
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": defaultdict(int),
        "difficulty_distribution": defaultdict(int),
    }
    proc._all_concept_tags = set()
    proc.domain_concept_seeds = []
    proc.objectives = None
    proc._valid_outcome_ids = set()
    proc._boilerplate_spans = []
    proc._factual_flags = []
    proc.MIN_CHUNK_SIZE = 50
    proc.MAX_CHUNK_SIZE = 800
    return proc


# ---------------------------------------------------------------------------
# Bloom-level fallback (every chunk gets a level)
# ---------------------------------------------------------------------------

class TestBloomLevelFallback:
    def _item(self, **kw):
        base = {
            "module_id": "m1",
            "module_title": "Module 1",
            "item_id": "l1",
            "title": "Lesson 1",
            "resource_type": "page",
            "key_concepts": [],
            "learning_objectives": [],
        }
        base.update(kw)
        return base

    def test_verb_heuristic_fallback(self):
        proc = _bare_processor()
        text = "Evaluate the design, critique the rationale, and justify your reasoning."
        chunk = proc._create_chunk(
            chunk_id="c1", text=text, html="<p>" + text + "</p>",
            item=self._item(), section_heading="H", chunk_type="explanation",
        )
        assert chunk["bloom_level"] == "evaluate"
        assert chunk["bloom_level_source"] == "verbs"

    def test_default_when_no_signal(self):
        proc = _bare_processor()
        chunk = proc._create_chunk(
            chunk_id="c1", text="A quiet paragraph with no taxonomy verbs.",
            html="<p>A quiet paragraph with no taxonomy verbs.</p>",
            item=self._item(), section_heading="H", chunk_type="explanation",
        )
        assert chunk["bloom_level"] == "understand"
        assert chunk["bloom_level_source"] == "default"

    def test_authoritative_source_does_not_get_source_tag(self):
        """When JSON-LD supplies a level, schema stays back-compat
        (no bloom_level_source field)."""
        proc = _bare_processor()
        item = self._item(
            courseforge_metadata={
                "learningObjectives": [{"id": "co-01", "bloomLevel": "analyze"}],
                "sections": [],
            }
        )
        chunk = proc._create_chunk(
            chunk_id="c1", text="Plain text.", html="<p>Plain text.</p>",
            item=item, section_heading="Missing", chunk_type="explanation",
        )
        assert chunk["bloom_level"] == "analyze"
        assert "bloom_level_source" not in chunk


# ---------------------------------------------------------------------------
# Concept tag pollution filter (Bloom verbs never leak through)
# ---------------------------------------------------------------------------

class TestConceptTagPollutionFilter:
    def test_bloom_verbs_dropped_from_key_concepts(self):
        proc = _bare_processor()
        item = {"key_concepts": ["apply", "aria-labelledby", "create", "landmark"]}
        tags = proc._extract_concept_tags("Plain body text.", item)
        assert "aria-labelledby" in tags
        assert "landmark" in tags
        assert "apply" not in tags
        assert "create" not in tags

    def test_objective_codes_dropped(self):
        proc = _bare_processor()
        item = {"key_concepts": ["co-01", "to-03", "w04-co-02", "accessibility"]}
        tags = proc._extract_concept_tags("Sample.", item)
        assert tags == ["accessibility"]

    def test_logistics_scaffolding_dropped(self):
        proc = _bare_processor()
        item = {"key_concepts": ["initial-post", "replies", "due", "skip-link"]}
        tags = proc._extract_concept_tags("Sample.", item)
        assert tags == ["skip-link"]


# ---------------------------------------------------------------------------
# Domain concept seeds (text-based extraction of per-course vocabulary)
# ---------------------------------------------------------------------------

class TestDomainConceptSeeds:
    def test_compile_builds_word_boundary_patterns(self):
        from Trainforge.process_course import compile_domain_concept_seeds

        seeds = compile_domain_concept_seeds([
            {"id": "pour", "aliases": ["POUR", "perceivable operable"]},
        ])
        assert len(seeds) == 1
        tag, patterns = seeds[0]
        assert tag == "pour"
        assert any(p.search("POUR principles apply everywhere") for p in patterns)
        # Must not match substring inside longer word.
        assert not any(p.search("downpour") for p in patterns)

    def test_seed_matched_in_text(self):
        from Trainforge.process_course import compile_domain_concept_seeds

        proc = _bare_processor()
        proc.domain_concept_seeds = compile_domain_concept_seeds([
            {"id": "aria", "aliases": ["ARIA", "WAI-ARIA"]},
            {"id": "pour", "aliases": ["POUR"]},
        ])
        tags = proc._extract_concept_tags(
            "ARIA roles complement the POUR principles.", {"key_concepts": []}
        )
        assert "aria" in tags
        assert "pour" in tags

    def test_seed_ignored_when_not_present(self):
        from Trainforge.process_course import compile_domain_concept_seeds

        proc = _bare_processor()
        proc.domain_concept_seeds = compile_domain_concept_seeds([
            {"id": "aria", "aliases": ["ARIA"]},
        ])
        tags = proc._extract_concept_tags("No special vocabulary here.", {"key_concepts": []})
        assert "aria" not in tags


# ---------------------------------------------------------------------------
# JSON-LD keyTerms merged into concept_tags
# ---------------------------------------------------------------------------

class TestKeyTermsMergedIntoConceptTags:
    def test_key_terms_surface_as_tags(self):
        proc = _bare_processor()
        item = {
            "module_id": "m1", "module_title": "Module 1",
            "item_id": "l1", "title": "Lesson 1", "resource_type": "page",
            "key_concepts": [], "learning_objectives": [],
            "courseforge_metadata": {
                "sections": [{
                    "heading": "Focus Management",
                    "contentType": "explanation",
                    "bloomRange": ["apply"],
                    "keyTerms": [
                        {"term": "Focus Indicator", "definition": "Visible outline."},
                        {"term": "Skip Link", "definition": "Bypass to main."},
                    ],
                }],
                "learningObjectives": [],
            },
        }
        chunk = proc._create_chunk(
            chunk_id="c1",
            text="Content about focus management.",
            html="<p>Content about focus management.</p>",
            item=item, section_heading="Focus Management", chunk_type="explanation",
        )
        assert "focus-indicator" in chunk["concept_tags"]
        assert "skip-link" in chunk["concept_tags"]

    def test_key_terms_still_filtered_against_non_concepts(self):
        proc = _bare_processor()
        item = {
            "module_id": "m1", "module_title": "Module 1",
            "item_id": "l1", "title": "Lesson 1", "resource_type": "page",
            "key_concepts": [], "learning_objectives": [],
            "courseforge_metadata": {
                "sections": [{
                    "heading": "Intro",
                    "contentType": "explanation",
                    "bloomRange": ["apply"],
                    "keyTerms": [
                        {"term": "Apply"},  # Bloom verb
                        {"term": "ARIA role"},
                    ],
                }],
                "learningObjectives": [],
            },
        }
        chunk = proc._create_chunk(
            chunk_id="c1", text="Body.", html="<p>Body.</p>",
            item=item, section_heading="Intro", chunk_type="explanation",
        )
        assert "aria-role" in chunk["concept_tags"]
        assert "apply" not in chunk["concept_tags"]


# ---------------------------------------------------------------------------
# Uncovered outcomes surfaced in quality_report
# ---------------------------------------------------------------------------

class TestUncoveredOutcomesInQualityReport:
    def test_uncovered_ids_listed_and_issue_emitted(self):
        proc = _bare_processor()
        proc.stats["total_words"] = 300
        proc._valid_outcome_ids = {"co-01", "co-02", "co-03", "co-04"}
        chunks = [
            _chunk(id="c1", word_count=120, html="<p>a</p>",
                   learning_outcome_refs=["co-01"], concept_tags=["aria", "pour"]),
            _chunk(id="c2", word_count=120, html="<p>b</p>",
                   learning_outcome_refs=["co-02"], concept_tags=["landmark", "aria"]),
        ]
        report = proc._generate_quality_report(chunks)
        assert report["integrity"]["uncovered_outcomes"] == ["co-03", "co-04"]
        assert report["metrics"]["outcome_reverse_coverage"] == 0.5
        assert any("have zero resolving chunks" in i for i in report["validation"]["issues"])

    def test_full_reverse_coverage_passes(self):
        proc = _bare_processor()
        proc.stats["total_words"] = 120
        proc._valid_outcome_ids = {"co-01"}
        chunks = [
            _chunk(id="c1", word_count=120, html="<p>a</p>",
                   learning_outcome_refs=["co-01"], concept_tags=["aria", "pour"]),
        ]
        report = proc._generate_quality_report(chunks)
        assert report["integrity"]["uncovered_outcomes"] == []
        assert report["metrics"]["outcome_reverse_coverage"] == 1.0


# ---------------------------------------------------------------------------
# Pedagogy model (module sequence, Bloom progression, prereq chain)
# ---------------------------------------------------------------------------

class TestPedagogyModelRichness:
    def _mk(self, chunk_id, module_id, module_title, bloom, tags, prereqs,
            los=(), position=0):
        return _chunk(
            id=chunk_id, bloom_level=bloom, concept_tags=list(tags),
            prereq_concepts=list(prereqs), learning_outcome_refs=list(los),
            source={
                "course_id": "MINI_101",
                "module_id": module_id,
                "module_title": module_title,
                "lesson_id": module_id,
                "lesson_title": module_title,
                "resource_type": "page",
                "section_heading": "H",
                "position_in_module": position,
            },
        )

    def test_thin_summary_when_no_chunks(self):
        proc = _bare_processor()
        summary = proc._build_pedagogy_summary()
        assert summary["instructional_approach"] == "competency-based"
        assert "module_sequence" not in summary

    def test_module_sequence_ordered_by_week(self):
        proc = _bare_processor()
        chunks = [
            self._mk("a", "week_02_foo", "Week 2", "apply", ["aria"], []),
            self._mk("b", "week_01_foo", "Week 1", "understand", ["pour"], []),
            self._mk("c", "week_03_foo", "Week 3", "evaluate", ["landmark"], []),
        ]
        summary = proc._build_pedagogy_summary(chunks)
        weeks = [m["week_num"] for m in summary["module_sequence"]]
        assert weeks == [1, 2, 3]

    def test_bloom_progression_counts_per_module(self):
        proc = _bare_processor()
        chunks = [
            self._mk("a", "week_01_foo", "Week 1", "understand", ["pour"], []),
            self._mk("b", "week_01_foo", "Week 1", "apply", ["pour"], []),
            self._mk("c", "week_02_foo", "Week 2", "evaluate", ["aria"], []),
        ]
        summary = proc._build_pedagogy_summary(chunks)
        assert summary["bloom_progression"]["week_01_foo"]["understand"] == 1
        assert summary["bloom_progression"]["week_01_foo"]["apply"] == 1
        assert summary["bloom_progression"]["week_02_foo"]["evaluate"] == 1

    def test_prerequisite_chain_valid_order(self):
        proc = _bare_processor()
        chunks = [
            # Week 1 defines 'pour'
            self._mk("a", "week_01_foo", "Week 1", "understand", ["pour"], []),
            # Week 2 uses 'pour' as prereq → valid chain
            self._mk("b", "week_02_foo", "Week 2", "apply", ["aria"], ["pour"]),
        ]
        summary = proc._build_pedagogy_summary(chunks)
        chain_concepts = {e["concept"] for e in summary["prerequisite_chain"]}
        assert "pour" in chain_concepts
        assert summary["prerequisite_violations"] == []

    def test_prerequisite_violation_detected(self):
        proc = _bare_processor()
        chunks = [
            # Week 1 uses 'aria' as prereq BEFORE it is defined anywhere visible
            self._mk("a", "week_01_foo", "Week 1", "understand", [], ["aria"]),
            # Week 2 finally defines 'aria'
            self._mk("b", "week_02_foo", "Week 2", "apply", ["aria"], []),
        ]
        summary = proc._build_pedagogy_summary(chunks)
        violations = {v["concept"] for v in summary["prerequisite_violations"]}
        assert "aria" in violations


class TestFixtures:
    def test_clean_fixture_present(self):
        assert (CLEAN_DIR / "course_objectives.json").exists()
        assert any((CLEAN_DIR / "source_html").glob("*.html"))

    def test_defective_fixture_present(self):
        assert (DEFECTIVE_DIR / "course_objectives.json").exists()
        assert (DEFECTIVE_DIR / "source_html" / "week_01_overview.html").exists()

    def test_edge_fixture_has_orphan_ref(self):
        data = json.loads((EDGE_DIR / "course_objectives.json").read_text())
        existing_ws = {
            ws.lower()
            for ch in data["chapter_objectives"]
            for obj in ch["objectives"]
            for ws in obj.get("week_scoped_ids", [])
        }
        html = (EDGE_DIR / "source_html" / "week_05_orphan_ref.html").read_text()
        assert "w05-co-99" in html.lower()
        assert "w05-co-99" not in existing_ws

    def test_defective_fixture_objectives_have_dual_ids(self):
        data = json.loads((DEFECTIVE_DIR / "course_objectives.json").read_text())
        for ch in data["chapter_objectives"]:
            for obj in ch["objectives"]:
                assert "week_scoped_ids" in obj
                assert any(ws.startswith("w") for ws in obj["week_scoped_ids"])
