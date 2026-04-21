"""Wave 26 — BloomAlignmentValidator strict-mode tests.

Pre-Wave-26 bug: ``bloom.py:122-134`` counted ``detect_bloom_level(stem)
== None`` as aligned. 30 verb-less stems all like "Which best describes
Structural?" scored 1.0 alignment — a pedagogically null assessment
passed the gate.

Wave 26 fix: verb-less stems count as UNALIGNED by default; legacy
behavior is preserved behind ``permissive_mode=True``.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.bloom import BloomAlignmentValidator  # noqa: E402


def _q(qid: str, stem: str, declared="understand"):
    return {
        "question_id": qid,
        "stem": stem,
        "bloom_level": declared,
    }


def test_all_verbless_stems_score_zero_strict_mode():
    """30 verb-less stems (the OLSR_201 shape) should score 0.0 in strict
    mode — previously scored 1.0."""
    questions = [_q(f"q-{i:03d}", "<p>Something about Structural</p>") for i in range(30)]
    result = BloomAlignmentValidator().validate({
        "assessment_data": {"questions": questions},
        "min_alignment_score": 0.7,
    })
    assert result.score == 0.0, result.score
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert codes.count("VERB_LESS_STEM") == 30


def test_verbful_stems_score_aligned():
    """All stems have a Bloom verb; all declared levels match detected —
    score must be 1.0."""
    questions = [
        _q("q-001", "<p>Explain mitosis</p>", declared="understand"),
        _q("q-002", "<p>Describe photosynthesis</p>", declared="understand"),
        _q("q-003", "<p>Compare aerobic and anaerobic respiration</p>", declared="understand"),
    ]
    result = BloomAlignmentValidator().validate({
        "assessment_data": {"questions": questions},
        "min_alignment_score": 0.7,
    })
    assert result.score == 1.0, result.score
    assert result.passed
    codes = [i.code for i in result.issues]
    assert "VERB_LESS_STEM" not in codes


def test_mixed_2_verbful_3_verbless_score_04():
    """2 verbful + 3 verb-less → 2/5 = 0.4. Below min 0.7 → not passed."""
    questions = [
        _q("q-001", "<p>Explain this concept</p>", declared="understand"),
        _q("q-002", "<p>Describe the process clearly</p>", declared="understand"),
        _q("q-003", "<p>Widget-a</p>"),
        _q("q-004", "<p>Widget-b</p>"),
        _q("q-005", "<p>Widget-c</p>"),
    ]
    result = BloomAlignmentValidator().validate({
        "assessment_data": {"questions": questions},
        "min_alignment_score": 0.7,
    })
    assert abs(result.score - 0.4) < 1e-6, result.score
    assert not result.passed


def test_permissive_mode_preserves_legacy_behavior():
    """permissive_mode=True: 30 verb-less stems score 1.0 (pre-Wave-26
    behavior preserved for back-compat fixtures)."""
    questions = [_q(f"q-{i:03d}", "<p>Something about X</p>") for i in range(30)]
    result = BloomAlignmentValidator().validate({
        "assessment_data": {"questions": questions},
        "min_alignment_score": 0.7,
        "permissive_mode": True,
    })
    assert result.score == 1.0
    assert result.passed
    # In permissive mode the VERB_LESS_STEM diagnostic is NOT emitted.
    codes = [i.code for i in result.issues]
    assert "VERB_LESS_STEM" not in codes


def test_bloom_mismatch_still_emits_warning():
    """A stem with a detectable Bloom verb that DOESN'T match the
    declared level still emits BLOOM_MISMATCH (legacy behavior)."""
    questions = [
        _q("q-001", "<p>Compare A and B</p>", declared="remember"),  # analyze verb
        _q("q-002", "<p>Explain concept</p>", declared="understand"),
    ]
    result = BloomAlignmentValidator().validate({
        "assessment_data": {"questions": questions},
        "min_alignment_score": 0.5,
    })
    codes = [i.code for i in result.issues]
    assert "BLOOM_MISMATCH" in codes


def test_verbless_single_stem_diagnostic_includes_excerpt():
    """Per-question VERB_LESS_STEM diagnostic must include a stem
    excerpt so a reviewer can locate the broken question."""
    questions = [_q("q-007", "<p>Widget something structural</p>")]
    result = BloomAlignmentValidator().validate({
        "assessment_data": {"questions": questions},
        "min_alignment_score": 0.5,
    })
    verbless = [i for i in result.issues if i.code == "VERB_LESS_STEM"]
    assert len(verbless) == 1
    assert "q-007" in verbless[0].message
    assert "Widget" in verbless[0].message
