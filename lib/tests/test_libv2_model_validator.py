"""Wave 89 — LibV2ModelValidator tests.

Mirrors ``test_libv2_manifest_validator.py``: critical-severity
checks (JSON parse, schema match, weights integrity, pedagogy hash
resolution) must block; warning-severity gaps (missing eval scores,
missing license, malformed HF repo) surface but never block.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

from lib.validators.libv2_model import LibV2ModelValidator


# ---------------------------------------------------------------------- #
# Fixtures                                                                #
# ---------------------------------------------------------------------- #


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _build_course(
    tmp_path: Path,
    *,
    slug: str = "tst-101",
    pedagogy_artifact: str = "graph/pedagogy_graph.json",
    pedagogy_payload: bytes = b'{"nodes": [], "edges": []}',
) -> Tuple[Path, str]:
    """Build a minimal LibV2 course skeleton.

    Returns (course_dir, pedagogy_sha256).
    """
    course_dir = tmp_path / "courses" / slug
    course_dir.mkdir(parents=True)
    # Create scaffold dirs (just enough to host pedagogy)
    (course_dir / "graph").mkdir()
    (course_dir / "pedagogy").mkdir()
    (course_dir / "models").mkdir()

    artifact_path = course_dir / pedagogy_artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(pedagogy_payload)
    return course_dir, _sha256_bytes(pedagogy_payload)


def _make_card(
    *,
    pedagogy_hash: str,
    weights_meta: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Return a fully-populated valid model card dict."""
    h = "a" * 64
    card: Dict[str, Any] = {
        "model_id": "qwen2-5-1-5b-tst-101-v1",
        "course_slug": "tst-101",
        "base_model": {
            "name": "qwen2.5-1.5b",
            "revision": "main",
            "huggingface_repo": "Qwen/Qwen2.5-1.5B",
        },
        "adapter_format": "safetensors",
        "training_config": {
            "seed": 42,
            "learning_rate": 2e-4,
            "epochs": 3,
            "lora_rank": 16,
            "lora_alpha": 32,
            "max_seq_length": 2048,
            "batch_size": 4,
        },
        "provenance": {
            "chunks_hash": h,
            "pedagogy_graph_hash": pedagogy_hash,
            "instruction_pairs_hash": h,
            "preference_pairs_hash": h,
            "concept_graph_hash": h,
            "vocabulary_ttl_hash": h,
            "holdout_graph_hash": h,
        },
        "created_at": "2026-04-26T18:30:00Z",
        "eval_scores": {
            "faithfulness": 0.83,
            "coverage": 0.91,
            "baseline_delta": 0.12,
            # Wave 102: scoring_commit + tolerance_band are required
            # whenever eval_scores is present.
            "scoring_commit": "f" * 40,
            "tolerance_band": {
                "accuracy": 0.0,
                "faithfulness": 0.05,
                "hallucination_rate": 0.05,
                "source_match": 0.0,
            },
        },
        "license": "apache-2.0",
    }
    if weights_meta is not None:
        card["weights"] = weights_meta  # noqa: E501 — extra; will trip schema
    card.update(overrides)
    return card


@pytest.fixture
def good_model_dir(tmp_path: Path):
    """A well-formed model dir under tmp_path that should pass validation.

    Uses an in-card 'weights' block carrying the synthetic adapter's
    sha256 + size. NOTE: the canonical schema has additionalProperties=
    false at top level — including ``weights`` in the persisted card
    would trip schema validation. The fixture writes the validation
    metadata into the card only when `with_weights_meta=True` is
    passed via parametrisation; the default fixture writes the
    canonical (schema-clean) card and lets the validator rely on the
    default ``adapter.safetensors`` filename + presence.
    """
    course_dir, pedagogy_hash = _build_course(tmp_path)
    model_id = "qwen2-5-1-5b-tst-101-v1"
    model_dir = course_dir / "models" / model_id
    model_dir.mkdir(parents=True)

    # Write a synthetic adapter (any bytes; presence is what matters)
    weights_bytes = b"safetensors-fake-bytes" * 100
    weights_path = model_dir / "adapter.safetensors"
    weights_path.write_bytes(weights_bytes)

    card = _make_card(pedagogy_hash=pedagogy_hash)
    card_path = model_dir / "model_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")

    return card_path, model_dir, course_dir, pedagogy_hash, weights_path


# ---------------------------------------------------------------------- #
# Happy path                                                              #
# ---------------------------------------------------------------------- #


def test_valid_model_passes(good_model_dir):
    card_path, model_dir, course_dir, _, _ = good_model_dir
    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert result.passed, (
        f"Valid model card should pass. Got issues: "
        f"{[(i.severity, i.code) for i in result.issues]}"
    )
    critical = [i for i in result.issues if i.severity == "critical"]
    assert not critical, f"Unexpected critical issues: {[i.code for i in critical]}"


def test_course_dir_derived_from_model_card_path(good_model_dir):
    """When course_dir is omitted, validator derives it from model_card_path."""
    card_path, _, _, _, _ = good_model_dir
    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
    })
    # Should still pass — derivation walks model_dir.parent.parent
    assert result.passed, (
        f"Derivation should yield a passing run. Issues: "
        f"{[(i.severity, i.code) for i in result.issues]}"
    )


# ---------------------------------------------------------------------- #
# Critical: input + parse failures                                        #
# ---------------------------------------------------------------------- #


def test_missing_model_card_path():
    result = LibV2ModelValidator().validate({})
    assert not result.passed
    assert any(i.code == "MISSING_MODEL_CARD_PATH" for i in result.issues)


def test_nonexistent_model_card_path(tmp_path):
    result = LibV2ModelValidator().validate({
        "model_card_path": str(tmp_path / "nope.json"),
    })
    assert not result.passed
    assert any(i.code == "MODEL_CARD_NOT_FOUND" for i in result.issues)


def test_corrupt_json_fails_critical(tmp_path):
    p = tmp_path / "model_card.json"
    p.write_text("{not valid json", encoding="utf-8")
    result = LibV2ModelValidator().validate({
        "model_card_path": str(p),
    })
    assert not result.passed
    assert any(i.code == "INVALID_JSON" for i in result.issues)


# ---------------------------------------------------------------------- #
# Critical: schema violations                                             #
# ---------------------------------------------------------------------- #


def test_schema_violation_when_missing_required_key(good_model_dir):
    card_path, model_dir, course_dir, _, _ = good_model_dir
    card = json.loads(card_path.read_text(encoding="utf-8"))
    del card["model_id"]
    card_path.write_text(json.dumps(card), encoding="utf-8")

    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert not result.passed
    assert any(i.code == "SCHEMA_VIOLATION" for i in result.issues)


def test_schema_violation_invalid_adapter_format(good_model_dir):
    card_path, model_dir, course_dir, _, _ = good_model_dir
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["adapter_format"] = "ggml"   # not in enum
    card_path.write_text(json.dumps(card), encoding="utf-8")

    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert not result.passed
    assert any(i.code == "SCHEMA_VIOLATION" for i in result.issues)


def test_schema_violation_missing_provenance_block(good_model_dir):
    card_path, model_dir, course_dir, _, _ = good_model_dir
    card = json.loads(card_path.read_text(encoding="utf-8"))
    del card["provenance"]
    card_path.write_text(json.dumps(card), encoding="utf-8")

    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert not result.passed
    assert any(i.code == "SCHEMA_VIOLATION" for i in result.issues)


# ---------------------------------------------------------------------- #
# Critical: weights file integrity                                        #
# ---------------------------------------------------------------------- #


def test_missing_weights_file_fails_critical(good_model_dir):
    card_path, model_dir, course_dir, _, weights_path = good_model_dir
    weights_path.unlink()

    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert not result.passed
    assert any(i.code == "MISSING_WEIGHTS" for i in result.issues)


# ---------------------------------------------------------------------- #
# Critical: pedagogy hash resolution                                      #
# ---------------------------------------------------------------------- #


def test_pedagogy_graph_not_found_when_no_pedagogy_artifact(tmp_path):
    """Course dir with no pedagogy artifact at any of the candidate paths fails."""
    course_dir = tmp_path / "courses" / "bare-101"
    course_dir.mkdir(parents=True)
    # Note: no graph/pedagogy_graph.json, no pedagogy/pedagogy_model.json
    (course_dir / "models").mkdir()
    model_dir = course_dir / "models" / "m1"
    model_dir.mkdir()
    (model_dir / "adapter.safetensors").write_bytes(b"x" * 32)

    card = _make_card(pedagogy_hash="a" * 64)
    card_path = model_dir / "model_card.json"
    card_path.write_text(json.dumps(card), encoding="utf-8")

    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert not result.passed
    assert any(i.code == "PEDAGOGY_GRAPH_NOT_FOUND" for i in result.issues)


def test_pedagogy_hash_mismatch_fails_critical(good_model_dir):
    card_path, model_dir, course_dir, pedagogy_hash, _ = good_model_dir
    # Tamper: rewrite pedagogy artifact so hash diverges
    pedagogy_path = course_dir / "graph" / "pedagogy_graph.json"
    pedagogy_path.write_bytes(b'{"tampered": true}')

    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert not result.passed
    assert any(i.code == "PEDAGOGY_HASH_MISMATCH" for i in result.issues)


def test_pedagogy_hash_resolves_via_pedagogy_dir_fallback(tmp_path):
    """When graph/pedagogy_graph.json is absent but pedagogy/pedagogy_model.json
    is present and hashes match, validation passes."""
    course_dir, pedagogy_hash = _build_course(
        tmp_path,
        pedagogy_artifact="pedagogy/pedagogy_model.json",
    )
    model_id = "fallback-v1"
    model_dir = course_dir / "models" / model_id
    model_dir.mkdir()
    (model_dir / "adapter.safetensors").write_bytes(b"x" * 32)
    card = _make_card(pedagogy_hash=pedagogy_hash)
    card_path = model_dir / "model_card.json"
    card_path.write_text(json.dumps(card), encoding="utf-8")

    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert result.passed, (
        f"Fallback to pedagogy/pedagogy_model.json should pass. Issues: "
        f"{[(i.severity, i.code) for i in result.issues]}"
    )


# ---------------------------------------------------------------------- #
# Warning advisories — must never block                                   #
# ---------------------------------------------------------------------- #


def test_eval_scores_missing_warns_never_blocks(good_model_dir):
    card_path, model_dir, course_dir, _, _ = good_model_dir
    card = json.loads(card_path.read_text(encoding="utf-8"))
    del card["eval_scores"]
    card_path.write_text(json.dumps(card), encoding="utf-8")

    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert result.passed, "EVAL_SCORES_MISSING must never be critical"
    eval_issues = [i for i in result.issues if i.code == "EVAL_SCORES_MISSING"]
    assert eval_issues
    assert eval_issues[0].severity == "warning"


def test_license_missing_warns_never_blocks(good_model_dir):
    card_path, model_dir, course_dir, _, _ = good_model_dir
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["license"] = None
    card_path.write_text(json.dumps(card), encoding="utf-8")

    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert result.passed, "LICENSE_NOT_DECLARED must never be critical"
    issues = [i for i in result.issues if i.code == "LICENSE_NOT_DECLARED"]
    assert issues and issues[0].severity == "warning"


def test_license_empty_string_warns(good_model_dir):
    """Empty/whitespace license string still warns."""
    card_path, model_dir, course_dir, _, _ = good_model_dir
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["license"] = "   "
    card_path.write_text(json.dumps(card), encoding="utf-8")

    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert result.passed
    assert any(i.code == "LICENSE_NOT_DECLARED" for i in result.issues)


def test_huggingface_repo_pattern_warning_when_invalid_in_card_dict():
    """Hand-roll a card with a malformed HF repo and run validator
    without committing it through the schema check.

    We want the HF_REPO_PATTERN_INVALID warning specifically. Since
    the canonical schema uses ``^[\\w-]+/[\\w.-]+$`` for the repo,
    a malformed value also trips a SCHEMA_VIOLATION (critical) —
    so this test asserts the warning emits *alongside* the schema
    violation. Both fire; warning is informational regardless of
    the pass/fail outcome from the schema critical.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        course_dir, pedagogy_hash = _build_course(tdp)
        model_dir = course_dir / "models" / "bad-repo"
        model_dir.mkdir()
        (model_dir / "adapter.safetensors").write_bytes(b"x" * 32)
        card = _make_card(pedagogy_hash=pedagogy_hash)
        card["base_model"]["huggingface_repo"] = "no-slash"
        card_path = model_dir / "model_card.json"
        card_path.write_text(json.dumps(card), encoding="utf-8")

        result = LibV2ModelValidator().validate({
            "model_card_path": str(card_path),
            "model_dir": str(model_dir),
            "course_dir": str(course_dir),
        })
        # Warning must fire (the pattern check is independent of
        # the schema critical that also fires for the same field).
        assert any(i.code == "HF_REPO_PATTERN_INVALID" for i in result.issues), (
            f"Expected HF_REPO_PATTERN_INVALID, got "
            f"{[i.code for i in result.issues]}"
        )


# ---------------------------------------------------------------------- #
# Smoke: identifying name + version                                       #
# ---------------------------------------------------------------------- #


def test_validator_metadata():
    v = LibV2ModelValidator()
    assert v.name == "libv2_model"
    assert v.version == "1.0.0"
