"""Wave 91 Action E: tests for SynthesisDiversityValidator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.synthesis_diversity import (  # noqa: E402
    DEFAULT_MAX_SINGLE_SHARE,
    DEFAULT_MAX_TOP3_SHARE,
    DEFAULT_MIN_DISTINCT_TEMPLATES,
    DEFAULT_MIN_TOTAL_PAIRS,
    SynthesisDiversityValidator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: Iterable[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def _balanced_corpus(template_ids: List[str], per_template: int = 20) -> List[dict]:
    out = []
    for t in template_ids:
        for i in range(per_template):
            out.append({
                "template_id": t,
                "chunk_id": f"c-{t}-{i}",
                "prompt": f"...{t}...",
                "completion": f"...completion {i}...",
            })
    return out


# ---------------------------------------------------------------------------
# Pass paths
# ---------------------------------------------------------------------------


def test_passes_on_balanced_corpus(tmp_path):
    """8+ distinct templates, well below dominance ceilings."""
    template_ids = [
        "remember.explanation", "understand.explanation", "apply.example",
        "analyze.procedure", "evaluate.comparison", "create.example",
        "remember.procedure", "apply.comparison",
    ]
    path = _write_jsonl(
        tmp_path / "instruction_pairs.jsonl",
        _balanced_corpus(template_ids, per_template=15),
    )
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    assert result.passed is True
    # Score = 1 - top3_share. With 8 templates evenly distributed
    # top3_share = 3/8 = 0.375, score = 0.625.
    assert 0.6 < result.score < 0.65


def test_passes_on_high_diversity_large_corpus(tmp_path):
    template_ids = [f"template.{i}" for i in range(20)]
    path = _write_jsonl(
        tmp_path / "instruction_pairs.jsonl",
        _balanced_corpus(template_ids, per_template=10),
    )
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    assert result.passed is True


# ---------------------------------------------------------------------------
# Critical-fail paths
# ---------------------------------------------------------------------------


def test_fails_on_single_template_dominance(tmp_path):
    """One template > max_single_share triggers critical fail."""
    records = []
    # 50% from template-A
    for i in range(80):
        records.append({"template_id": "template.A", "chunk_id": f"a-{i}"})
    # spread across 8 others to satisfy distinct-templates floor
    for i, t in enumerate(["B", "C", "D", "E", "F", "G", "H", "I"]):
        for j in range(10):
            records.append({"template_id": f"template.{t}", "chunk_id": f"{t}-{j}"})
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "SINGLE_TEMPLATE_DOMINANCE" in codes


def test_fails_on_top3_template_dominance(tmp_path):
    """Top-3 share > max_top3_share triggers critical fail without any
    single template breaching its individual ceiling."""
    records = []
    # Three templates each ~25% (75% top-3) — none above 35% individual.
    for t in ("A", "B", "C"):
        for i in range(30):
            records.append({
                "template_id": f"template.{t}", "chunk_id": f"{t}-{i}",
            })
    # 8 other templates with small representation to hit the
    # min_distinct_templates floor.
    for t in ("D", "E", "F", "G", "H", "I", "J", "K"):
        for i in range(2):
            records.append({
                "template_id": f"template.{t}", "chunk_id": f"{t}-{i}",
            })
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "TOP3_TEMPLATE_DOMINANCE" in codes
    # No single template breaches 35% individually (each ~28%).
    assert "SINGLE_TEMPLATE_DOMINANCE" not in codes


def test_fails_on_low_distinct_templates(tmp_path):
    """< min_distinct_templates triggers critical fail."""
    template_ids = [f"template.{i}" for i in range(7)]  # one short of floor
    records = _balanced_corpus(template_ids, per_template=20)
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "LOW_DISTINCT_TEMPLATES" in codes


# ---------------------------------------------------------------------------
# Warning-only path
# ---------------------------------------------------------------------------


def test_low_total_pair_count_warning_does_not_block(tmp_path):
    """Small but well-distributed corpus warns but doesn't critical-fail."""
    template_ids = [f"template.{i}" for i in range(8)]
    records = _balanced_corpus(template_ids, per_template=2)  # 16 pairs total
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    assert result.passed is True
    codes = {i.code for i in result.issues}
    assert "LOW_TOTAL_PAIR_COUNT" in codes
    # Severity must be warning, not critical.
    severities = {i.severity for i in result.issues if i.code == "LOW_TOTAL_PAIR_COUNT"}
    assert severities == {"warning"}


# ---------------------------------------------------------------------------
# Threshold overrides
# ---------------------------------------------------------------------------


def test_threshold_overrides(tmp_path):
    """Caller can override every threshold."""
    template_ids = [f"template.{i}" for i in range(3)]
    records = _balanced_corpus(template_ids, per_template=20)
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
        "min_distinct_templates": 3,
        "max_top3_share": 1.0,
        "max_single_share": 1.0,
        "min_total_pairs": 1,
    })
    assert result.passed is True


# ---------------------------------------------------------------------------
# Missing / malformed handling
# ---------------------------------------------------------------------------


def test_missing_inputs_fails():
    result = SynthesisDiversityValidator().validate({})
    assert result.passed is False
    assert {"MISSING_INPUTS"} == {i.code for i in result.issues}


def test_missing_file_fails(tmp_path):
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(tmp_path / "nonexistent.jsonl"),
    })
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "INSTRUCTION_PAIRS_NOT_FOUND" in codes


def test_malformed_jsonl_lines_warned_not_blocked(tmp_path):
    """Malformed lines emit a warning but don't block; well-formed
    pairs are still counted toward diversity."""
    template_ids = [f"template.{i}" for i in range(8)]
    records = _balanced_corpus(template_ids, per_template=15)
    path = tmp_path / "instruction_pairs.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
        fh.write("not valid json\n")
        fh.write("\n")
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    assert result.passed is True
    codes = {i.code for i in result.issues}
    assert "MALFORMED_JSONL_LINE" in codes
