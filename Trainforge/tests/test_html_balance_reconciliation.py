"""Wave 82 Phase D3 tests for HTML balance metric reconciliation.

The rdf-shacl-551 audit found ``quality_report.json`` claiming 205/295
chunks failed HTML balance, while an independent HTMLParser recount
found only 116/295. The 89-chunk discrepancy traced to
``_html_is_well_formed`` returning ``False`` for empty/whitespace-only
HTML — conflating "no HTML to check" with "unclosed tags".

Wave 82 makes empty HTML well-formed by vacuity. The reconciled metric
counts only chunks with actually-unbalanced markup.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from Trainforge.process_course import CourseProcessor


def _bare_processor() -> CourseProcessor:
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.MIN_CHUNK_SIZE = 100
    proc.MAX_CHUNK_SIZE = 800
    proc._valid_outcome_ids = set()
    proc._boilerplate_spans = []
    proc._factual_flags = []
    # _generate_quality_report reads stats["total_words"] for averaging.
    proc.stats = {"total_words": 1000, "total_chunks": 0}
    return proc


# ---------------------------------------------------------------------------
# Empty HTML — the audit's reconciliation case
# ---------------------------------------------------------------------------


class TestEmptyHtmlNotCountedAsViolation:
    def test_empty_string_is_well_formed(self):
        assert CourseProcessor._html_is_well_formed("") is True

    def test_whitespace_only_is_well_formed(self):
        for ws in ["   ", "\n", "\t", "  \n\t  "]:
            assert CourseProcessor._html_is_well_formed(ws) is True, f"failed: {ws!r}"

    def test_unclosed_tag_still_unbalanced(self):
        # Real unbalanced HTML still fails — the fix only changes the
        # empty-html semantics.
        assert CourseProcessor._html_is_well_formed("<section>foo") is False

    def test_section_only_open_unbalanced(self):
        # The audit's specific failure mode (unclosed <section>).
        assert CourseProcessor._html_is_well_formed("<section><p>x</p>") is False


# ---------------------------------------------------------------------------
# Quality report metric reconciles with the HTMLParser recount
# ---------------------------------------------------------------------------


class TestQualityReportHtmlBalanceReconciliation:
    def _chunk(self, cid: str, html: str) -> Dict[str, Any]:
        # Minimum shape needed by _generate_quality_report.
        return {
            "id": cid,
            "html": html,
            "text": "x" * 200,  # in-range word count for size compliance
            "word_count": 200,
            "concept_tags": [],
            "learning_outcome_refs": [],
        }

    def test_audit_reproducer_empty_html_not_counted(self):
        """Mix of well-formed, empty, and unbalanced chunks."""
        proc = _bare_processor()
        chunks = [
            self._chunk("c1", "<p>well-formed</p>"),
            self._chunk("c2", ""),               # empty — should NOT count
            self._chunk("c3", "   "),            # whitespace — should NOT count
            self._chunk("c4", "<section>oops"),  # unclosed — counts
        ]
        report = proc._generate_quality_report(chunks)
        violations = report["integrity"]["html_balance_violations"]
        violation_ids = {v["chunk_id"] for v in violations}
        # Only c4 (actual unclosed tag) appears in violations.
        assert violation_ids == {"c4"}

    def test_pre_wave82_inflated_count_no_longer_reproduces(self):
        """Mirrors rdf-shacl-551's "89 inflated empties" subpattern."""
        proc = _bare_processor()
        # 5 well-formed, 3 empty, 2 unbalanced. Pre-Wave-82 metric: 5
        # violations (3 empties + 2 unbalanced). Post-Wave-82: 2.
        chunks = (
            [self._chunk(f"good_{i}", "<p>good</p>") for i in range(5)]
            + [self._chunk(f"empty_{i}", "") for i in range(3)]
            + [self._chunk(f"bad_{i}", "<div>oops") for i in range(2)]
        )
        report = proc._generate_quality_report(chunks)
        violations = report["integrity"]["html_balance_violations"]
        assert len(violations) == 2
        assert all(v["chunk_id"].startswith("bad_") for v in violations)

    def test_html_preservation_metric_excludes_empties_correctly(self):
        proc = _bare_processor()
        chunks = [
            self._chunk("good", "<p>x</p>"),
            self._chunk("empty", ""),
            self._chunk("bad", "<div>"),
        ]
        report = proc._generate_quality_report(chunks)
        # 2 of 3 are well-formed (good + empty). bad is the only violator.
        # Metric lives at report["metrics"]["html_preservation_rate"] per
        # the v4 metrics_semantic_version shape.
        assert report["metrics"]["html_preservation_rate"] == pytest.approx(2 / 3, abs=0.01)
