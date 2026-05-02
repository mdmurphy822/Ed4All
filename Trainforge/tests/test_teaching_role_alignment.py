"""Wave 138a Phase A — TeachingRoleAlignmentEvaluator regression tests.

Mirrors the property_eval test pattern: write a small fixture
chunks.jsonl into ``tmp_path``, run the evaluator, assert against the
frozen return shape from
``plans/eval-driven-teaching-role-alignment/plan.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.teaching_role_alignment import TeachingRoleAlignmentEvaluator


def _write_chunks(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    """Materialize ``rows`` as JSONL at ``path`` and return the path."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _chunk(
    chunk_id: str,
    *,
    content_type_label: Any,
    teaching_role: Any,
) -> Dict[str, Any]:
    """Minimal chunk shape for the evaluator. The evaluator only reads
    two fields, so the rest of the chunk schema is irrelevant."""
    return {
        "chunk_id": chunk_id,
        "content_type_label": content_type_label,
        "teaching_role": teaching_role,
    }


def test_evaluate_perfect_alignment(tmp_path: Path) -> None:
    """5 assessment chunks all carry teaching_role='assess' →
    actual_expected_share=1.0, mismatch=False."""
    chunks = _write_chunks(
        tmp_path / "chunks.jsonl",
        [
            _chunk(f"chunk_{i:03d}", content_type_label="assessment",
                   teaching_role="assess")
            for i in range(5)
        ],
    )
    result = TeachingRoleAlignmentEvaluator(chunks).evaluate()

    entry = result["content_type_role_alignment"]["assessment"]
    assert entry["total_chunks"] == 5
    assert entry["role_distribution"] == {"assess": 5}
    assert entry["dominant_role"] == "assess"
    assert entry["expected_role"] == "assess"
    assert entry["expected_share"] == 1.0
    assert entry["actual_expected_share"] == 1.0
    assert entry["mismatch"] is False
    assert entry["skipped_below_threshold"] is False

    summary = result["summary"]
    assert summary["total_content_types"] == 1
    assert summary["content_types_with_expected_mode"] == 1
    assert summary["mismatched_content_types"] == []
    assert summary["alignment_rate"] == 1.0


def test_evaluate_perfect_mismatch(tmp_path: Path) -> None:
    """10 real_world_scenario chunks all carry teaching_role='reinforce'
    (expected: 'transfer') → mismatch=True,
    actual_expected_share=0.0."""
    chunks = _write_chunks(
        tmp_path / "chunks.jsonl",
        [
            _chunk(f"chunk_{i:03d}",
                   content_type_label="real_world_scenario",
                   teaching_role="reinforce")
            for i in range(10)
        ],
    )
    result = TeachingRoleAlignmentEvaluator(chunks).evaluate()

    entry = result["content_type_role_alignment"]["real_world_scenario"]
    assert entry["total_chunks"] == 10
    assert entry["expected_role"] == "transfer"
    assert entry["actual_expected_share"] == 0.0
    assert entry["mismatch"] is True
    assert entry["skipped_below_threshold"] is False
    assert entry["dominant_role"] == "reinforce"

    summary = result["summary"]
    assert summary["mismatched_content_types"] == ["real_world_scenario"]
    assert summary["alignment_rate"] == 0.0


def test_evaluate_skips_below_min_chunks(tmp_path: Path) -> None:
    """Below-threshold buckets emit skipped_below_threshold=True with
    mismatch=False regardless of the underlying distribution. The
    share statistic is meaningless on N<5."""
    chunks = _write_chunks(
        tmp_path / "chunks.jsonl",
        [
            # 4 real_world_scenario chunks (below the default threshold
            # of 5). All carry the WRONG role; the evaluator MUST still
            # return mismatch=False because the bucket is too small.
            _chunk(f"chunk_{i:03d}",
                   content_type_label="real_world_scenario",
                   teaching_role="reinforce")
            for i in range(4)
        ],
    )
    result = TeachingRoleAlignmentEvaluator(chunks).evaluate()

    entry = result["content_type_role_alignment"]["real_world_scenario"]
    assert entry["total_chunks"] == 4
    assert entry["skipped_below_threshold"] is True
    assert entry["mismatch"] is False
    assert entry["actual_expected_share"] == 0.0  # still computed
    assert entry["expected_role"] == "transfer"

    summary = result["summary"]
    # Skipped buckets count toward content_types_with_expected_mode AND
    # toward passing_with_rule (they're not failing — just
    # uninformative). alignment_rate stays at 1.0.
    assert summary["content_types_with_expected_mode"] == 1
    assert summary["mismatched_content_types"] == []
    assert summary["alignment_rate"] == 1.0


def test_evaluate_no_rule_for_content_type(tmp_path: Path) -> None:
    """Content types absent from the expected_modes table emit
    expected_role=None, expected_share=None, mismatch=None — they're
    surfaced for visibility but don't participate in alignment_rate."""
    chunks = _write_chunks(
        tmp_path / "chunks.jsonl",
        [
            _chunk(f"chunk_{i:03d}",
                   content_type_label="example",
                   teaching_role="elaborate")
            for i in range(8)
        ],
    )
    result = TeachingRoleAlignmentEvaluator(chunks).evaluate()

    entry = result["content_type_role_alignment"]["example"]
    assert entry["expected_role"] is None
    assert entry["expected_share"] is None
    assert entry["actual_expected_share"] is None
    assert entry["mismatch"] is None
    assert entry["skipped_below_threshold"] is False
    assert entry["total_chunks"] == 8
    assert entry["dominant_role"] == "elaborate"

    summary = result["summary"]
    assert summary["total_content_types"] == 1
    assert summary["content_types_with_expected_mode"] == 0
    # No content types had a rule → vacuously aligned.
    assert summary["alignment_rate"] == 1.0


def test_evaluate_empty_chunks_returns_empty_dict_with_zeroed_summary(
    tmp_path: Path,
) -> None:
    """Empty chunks.jsonl → empty content_type_role_alignment dict and
    summary with zeros (alignment_rate=1.0 by the vacuously-true rule)."""
    chunks = _write_chunks(tmp_path / "chunks.jsonl", [])
    result = TeachingRoleAlignmentEvaluator(chunks).evaluate()

    assert result["content_type_role_alignment"] == {}
    summary = result["summary"]
    assert summary["total_content_types"] == 0
    assert summary["content_types_with_expected_mode"] == 0
    assert summary["mismatched_content_types"] == []
    assert summary["alignment_rate"] == 1.0


def test_evaluate_custom_expected_modes_merges_over_defaults(
    tmp_path: Path,
) -> None:
    """The expected_modes constructor kwarg must merge OVER the default
    table: caller-supplied entries override / extend, defaults remain
    for unmentioned keys."""
    chunks = _write_chunks(
        tmp_path / "chunks.jsonl",
        [
            # 6 procedure chunks all 'elaborate' — caller declares
            # procedure -> elaborate as the expected mode.
            _chunk(f"chunk_p{i:03d}",
                   content_type_label="procedure",
                   teaching_role="elaborate")
            for i in range(6)
        ] + [
            # 6 assessment chunks all 'assess' — falls under the
            # DEFAULT rule, must still be evaluated.
            _chunk(f"chunk_a{i:03d}",
                   content_type_label="assessment",
                   teaching_role="assess")
            for i in range(6)
        ],
    )
    result = TeachingRoleAlignmentEvaluator(
        chunks,
        expected_modes={
            "procedure": {"expected_role": "elaborate", "min_share": 0.70},
        },
    ).evaluate()

    proc_entry = result["content_type_role_alignment"]["procedure"]
    assert proc_entry["expected_role"] == "elaborate"
    assert proc_entry["mismatch"] is False
    assert proc_entry["actual_expected_share"] == 1.0

    asmt_entry = result["content_type_role_alignment"]["assessment"]
    assert asmt_entry["expected_role"] == "assess"
    assert asmt_entry["mismatch"] is False

    summary = result["summary"]
    assert summary["content_types_with_expected_mode"] == 2
    assert summary["alignment_rate"] == 1.0


def test_evaluate_alignment_rate_aggregate(tmp_path: Path) -> None:
    """Mixed pass/fail buckets: alignment_rate must be
    passing_with_rule / total_with_rule."""
    rows = []
    # PASS: 6 assessment chunks all 'assess' (expected: 1.0).
    rows.extend([
        _chunk(f"chunk_a{i:03d}", content_type_label="assessment",
               teaching_role="assess")
        for i in range(6)
    ])
    # PASS: 6 summary chunks all 'synthesize' (expected: 0.70).
    rows.extend([
        _chunk(f"chunk_s{i:03d}", content_type_label="summary",
               teaching_role="synthesize")
        for i in range(6)
    ])
    # FAIL: 10 real_world_scenario chunks all 'reinforce' (expected:
    # 'transfer' at 0.70).
    rows.extend([
        _chunk(f"chunk_r{i:03d}", content_type_label="real_world_scenario",
               teaching_role="reinforce")
        for i in range(10)
    ])
    # FAIL: 8 definition chunks 25% 'introduce' (expected: 0.70).
    rows.extend([
        _chunk(f"chunk_d{i:03d}", content_type_label="definition",
               teaching_role="introduce" if i < 2 else "elaborate")
        for i in range(8)
    ])

    chunks = _write_chunks(tmp_path / "chunks.jsonl", rows)
    result = TeachingRoleAlignmentEvaluator(chunks).evaluate()

    summary = result["summary"]
    assert summary["content_types_with_expected_mode"] == 4
    assert sorted(summary["mismatched_content_types"]) == [
        "definition", "real_world_scenario",
    ]
    # 2 of 4 buckets pass → 0.5.
    assert summary["alignment_rate"] == 0.5


def test_evaluate_dominant_role_alphabetical_tiebreak(tmp_path: Path) -> None:
    """Determinism check: when two roles are tied for top count,
    dominant_role must pick the alphabetically-earlier one."""
    chunks = _write_chunks(
        tmp_path / "chunks.jsonl",
        [
            # 3 'elaborate' + 3 'reinforce' under 'application'.
            # 'elaborate' < 'reinforce' alphabetically — must win.
            _chunk("chunk_001", content_type_label="application",
                   teaching_role="reinforce"),
            _chunk("chunk_002", content_type_label="application",
                   teaching_role="elaborate"),
            _chunk("chunk_003", content_type_label="application",
                   teaching_role="reinforce"),
            _chunk("chunk_004", content_type_label="application",
                   teaching_role="elaborate"),
            _chunk("chunk_005", content_type_label="application",
                   teaching_role="reinforce"),
            _chunk("chunk_006", content_type_label="application",
                   teaching_role="elaborate"),
        ],
    )
    result = TeachingRoleAlignmentEvaluator(chunks).evaluate()

    entry = result["content_type_role_alignment"]["application"]
    assert entry["dominant_role"] == "elaborate"
    assert entry["role_distribution"]["elaborate"] == 3
    assert entry["role_distribution"]["reinforce"] == 3


def test_evaluate_skips_chunks_with_null_content_type_label(
    tmp_path: Path,
) -> None:
    """Chunks where content_type_label is None must be silently
    skipped — there's no bucket to add them to, and a synthetic
    'unlabeled' bucket would inflate the summary counts."""
    rows = [
        # 5 real chunks with a labeled content_type.
        _chunk(f"chunk_d{i:03d}", content_type_label="definition",
               teaching_role="introduce")
        for i in range(5)
    ] + [
        # 3 chunks with null content_type_label — must be ignored.
        _chunk("chunk_n001", content_type_label=None,
               teaching_role="elaborate"),
        _chunk("chunk_n002", content_type_label=None,
               teaching_role="reinforce"),
        _chunk("chunk_n003", content_type_label=None,
               teaching_role=None),
    ]

    chunks = _write_chunks(tmp_path / "chunks.jsonl", rows)
    result = TeachingRoleAlignmentEvaluator(chunks).evaluate()

    # Only the labeled bucket appears.
    assert list(result["content_type_role_alignment"].keys()) == ["definition"]
    entry = result["content_type_role_alignment"]["definition"]
    assert entry["total_chunks"] == 5
    summary = result["summary"]
    assert summary["total_content_types"] == 1
