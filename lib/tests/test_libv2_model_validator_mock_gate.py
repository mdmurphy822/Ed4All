"""Wave 107 — LibV2ModelValidator must reject mock-provider training corpora.

A model card whose underlying ``training_specs/instruction_pairs.jsonl``
first row carries ``provider: "mock"`` is the regression class behind
the rdf-shacl-551-2 template-recognizer adapter. The validator must
fail closed so an accidental mock corpus cannot promote.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import pytest

from lib.validators.libv2_model import LibV2ModelValidator


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_minimal_course(
    tmp_path: Path,
    *,
    instruction_provider: str,
    slug: str = "tst-mock-gate",
) -> Tuple[Path, Path, Path]:
    """Return (course_dir, model_dir, card_path) for a minimal LibV2 layout
    whose instruction_pairs.jsonl first row has the requested provider."""
    course_dir = tmp_path / "courses" / slug
    course_dir.mkdir(parents=True)
    (course_dir / "graph").mkdir()
    (course_dir / "training_specs").mkdir()
    (course_dir / "models").mkdir()

    pedagogy_payload = b'{"nodes": [], "edges": []}'
    pedagogy_path = course_dir / "graph" / "pedagogy_graph.json"
    pedagogy_path.write_bytes(pedagogy_payload)
    pedagogy_hash = _sha256_bytes(pedagogy_payload)

    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    inst_path.write_text(
        json.dumps({
            "prompt": "p", "completion": "c", "provider": instruction_provider,
            "chunk_id": "chunk_001", "lo_refs": ["TO-01"],
        }) + "\n",
        encoding="utf-8",
    )
    pref_path = course_dir / "training_specs" / "preference_pairs.jsonl"
    pref_path.write_text(
        json.dumps({
            "prompt": "p", "chosen": "c", "rejected": "r",
            "provider": instruction_provider,
        }) + "\n",
        encoding="utf-8",
    )

    model_id = "qwen2-5-1-5b-tst-mock-gate-v1"
    model_dir = course_dir / "models" / model_id
    model_dir.mkdir(parents=True)

    weights_path = model_dir / "adapter.safetensors"
    weights_path.write_bytes(b"safetensors-fake-bytes" * 100)

    h = "a" * 64
    card = {
        "model_id": model_id,
        "course_slug": slug,
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
        "created_at": "2026-04-28T18:30:00Z",
        "eval_scores": {
            "faithfulness": 0.83,
            "coverage": 0.91,
            "baseline_delta": 0.12,
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
    card_path = model_dir / "model_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")

    return course_dir, model_dir, card_path


def test_validator_critical_fails_when_instruction_pairs_first_row_is_mock(
    tmp_path: Path,
) -> None:
    course_dir, model_dir, card_path = _build_minimal_course(
        tmp_path, instruction_provider="mock"
    )
    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    assert result.passed is False
    critical = [i for i in result.issues if i.severity == "critical"]
    assert any(
        "mock" in i.message.lower() and "provider" in i.message.lower()
        for i in critical
    ), [i.message for i in critical]


def test_validator_passes_when_instruction_pairs_first_row_is_claude_session(
    tmp_path: Path,
) -> None:
    course_dir, model_dir, card_path = _build_minimal_course(
        tmp_path, instruction_provider="claude_session"
    )
    result = LibV2ModelValidator().validate({
        "model_card_path": str(card_path),
        "model_dir": str(model_dir),
        "course_dir": str(course_dir),
    })
    critical = [i for i in result.issues if i.severity == "critical"]
    assert not any(
        "mock" in i.message.lower() and "provider" in i.message.lower()
        for i in critical
    ), [i.message for i in critical]
