"""Wave 105 — prefix-bigram diversity tests for SynthesisDiversityValidator.

The Wave 91 template-id checks live in
``Trainforge/tests/test_synthesis_diversity_validator.py``. This file
holds the Wave 105 tests for the second-layer prefix-bigram check that
catches template-collapse at the COMPLETION text level (the gap that
masked the rdf-shacl-551-2 corpus's 80% "the treatment ..." dominance).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.synthesis_diversity import (  # noqa: E402
    DEFAULT_MAX_PREFIX_TOP1_SHARE,
    DEFAULT_MAX_PREFIX_TOP3_SHARE,
    SynthesisDiversityValidator,
)


def _write_jsonl(path: Path, records: Iterable[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def _diverse_template_ids(n: int = 12) -> List[str]:
    return [f"template.{i}" for i in range(n)]


def test_critical_fail_when_top1_prefix_dominates(tmp_path):
    """All completions starting with "the core idea" trips top-1
    prefix-bigram dominance even when template_id is well distributed."""
    template_ids = _diverse_template_ids(10)
    records = []
    for i, tid in enumerate(template_ids):
        for j in range(15):
            records.append({
                "template_id": tid,
                "completion": (
                    "The core idea that drives this treatment "
                    f"is concept-{i}-{j}."
                ),
            })
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    assert result.passed is False
    codes = {i.code for i in result.issues}
    # top-1 dominance MUST fire because every completion shares the
    # same first 2 words (`the core`).
    assert "PREFIX_BIGRAM_TOP1_DOMINANCE" in codes
    # top-3 dominance also fires because the only bigram is dominant.
    assert "PREFIX_BIGRAM_TOP3_DOMINANCE" in codes


def test_critical_fail_when_top3_prefix_dominates(tmp_path):
    """Three different prefixes covering 100% of pairs trips top-3
    even if no single one breaches the top-1 threshold."""
    template_ids = _diverse_template_ids(10)
    records = []
    # Three prefix bigrams roughly equally distributed (each ~33%).
    prefixes = [
        "The treatment of",
        "The central concept",
        "The defining feature",
    ]
    for i, tid in enumerate(template_ids):
        for j in range(15):
            prefix = prefixes[j % 3]
            records.append({
                "template_id": tid,
                "completion": f"{prefix} is something concrete here.",
            })
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "PREFIX_BIGRAM_TOP3_DOMINANCE" in codes


def test_passes_when_completions_are_diverse(tmp_path):
    """A corpus with diverse opening words passes both prefix checks."""
    template_ids = _diverse_template_ids(10)
    diverse_openings = [
        "RDF is", "SHACL validates", "OWL adds", "SPARQL queries",
        "Validation rules", "A NodeShape", "Property paths", "Constraints define",
        "Triples model", "Schemas declare", "Assertions hold", "Inference draws",
        "An ontology", "Subclasses inherit", "Domains restrict", "Ranges constrain",
        "Reasoners derive", "Closure rules", "Open world", "Datatype literals",
    ]
    records = []
    for i, tid in enumerate(template_ids):
        for j in range(15):
            opening = diverse_openings[(i * 15 + j) % len(diverse_openings)]
            records.append({
                "template_id": tid,
                "completion": f"{opening} thing-{i}-{j} here.",
            })
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    codes = {i.code for i in result.issues}
    assert "PREFIX_BIGRAM_TOP1_DOMINANCE" not in codes
    assert "PREFIX_BIGRAM_TOP3_DOMINANCE" not in codes


def test_prefix_check_skipped_when_completion_field_missing(tmp_path):
    """When records carry no completion text, prefix-bigram check is
    silent (the volume warning still fires for small corpora)."""
    template_ids = _diverse_template_ids(10)
    records = [
        {"template_id": t, "chunk_id": f"c-{i}"}
        for i, t in enumerate(template_ids * 15)
    ]
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    codes = {i.code for i in result.issues}
    assert "PREFIX_BIGRAM_TOP1_DOMINANCE" not in codes
    assert "PREFIX_BIGRAM_TOP3_DOMINANCE" not in codes


def test_prefix_check_thresholds_overridable(tmp_path):
    """Caller can raise the prefix thresholds to allow a tighter
    corpus without flagging."""
    template_ids = _diverse_template_ids(10)
    records = []
    for i, tid in enumerate(template_ids):
        for j in range(15):
            records.append({
                "template_id": tid,
                "completion": f"The same opening {i}-{j}.",
            })
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    # With both prefix thresholds raised above the observed share, the
    # dominance check is effectively disabled. Top-1 share is 1.0 in
    # this fixture (every completion shares the same opening words).
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
        "max_prefix_top1_share": 1.5,
        "max_prefix_top3_share": 1.5,
    })
    codes = {i.code for i in result.issues}
    assert "PREFIX_BIGRAM_TOP1_DOMINANCE" not in codes
    assert "PREFIX_BIGRAM_TOP3_DOMINANCE" not in codes


def test_prefix_check_uses_default_thresholds(tmp_path):
    """Sanity-check the default thresholds are wired correctly."""
    assert DEFAULT_MAX_PREFIX_TOP1_SHARE == pytest.approx(0.15)
    assert DEFAULT_MAX_PREFIX_TOP3_SHARE == pytest.approx(0.30)


def test_template_collapse_at_response_level_with_diverse_template_ids(tmp_path):
    """Empirical regression test: 11 distinct template_ids but one
    prefix bigram dominates ALL completions. The Wave 91 check would
    pass (template_ids are well distributed); the Wave 105 check must
    catch the response-level collapse."""
    records = []
    template_ids = [
        "remember.explanation", "understand.explanation", "apply.example",
        "analyze.procedure", "evaluate.comparison", "create.example",
        "remember.procedure", "apply.comparison", "remember.example",
        "understand.example", "analyze.example",
    ]
    for tid in template_ids:
        for i in range(20):
            records.append({
                "template_id": tid,
                "completion": (
                    "The treatment of this concept covers something "
                    f"specific to {tid}/{i}."
                ),
            })
    path = _write_jsonl(tmp_path / "instruction_pairs.jsonl", records)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(path),
    })
    # Template-id checks alone pass (11 templates, balanced).
    template_codes = {
        "LOW_DISTINCT_TEMPLATES",
        "TOP3_TEMPLATE_DOMINANCE",
        "SINGLE_TEMPLATE_DOMINANCE",
    }
    fired_codes = {i.code for i in result.issues}
    # No template-id issue fires
    assert not (template_codes & fired_codes)
    # But the prefix-bigram check critical-fails the validator
    assert result.passed is False
    assert "PREFIX_BIGRAM_TOP1_DOMINANCE" in fired_codes
