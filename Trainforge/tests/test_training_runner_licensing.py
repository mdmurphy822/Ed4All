"""SFT-C S6 — TrainingRunner licensing-preflight regression net.

The runner is the training-data export/ingest path: instruction /
preference pairs become a shipped adapter here. The preflight must
fail-closed on a barred / claude-tagged teacher and stay byte-identical
(pass) for a license-clean corpus. Offline: dry_run, no GPU, no model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.licensing import LicenseGuardError
from Trainforge.training.runner import TrainingRunner


def _build_course(
    tmp_path: Path,
    *,
    slug: str = "lic-101",
    instruction_pairs: str,
) -> Path:
    libv2_root = tmp_path / "courses"
    course_dir = libv2_root / slug
    (course_dir / "training_specs").mkdir(parents=True)
    (course_dir / "graph").mkdir(parents=True)
    (course_dir / "imscc_chunks").mkdir(parents=True)

    (course_dir / "imscc_chunks" / "chunks.jsonl").write_text(
        json.dumps({"id": "c1", "text": "x"}) + "\n", encoding="utf-8"
    )
    (course_dir / "graph" / "pedagogy_graph.json").write_text(
        '{"nodes": [], "edges": []}', encoding="utf-8"
    )
    (course_dir / "graph" / "concept_graph_semantic.json").write_text(
        '{"concepts": []}', encoding="utf-8"
    )
    (course_dir / "graph" / "courseforge_v1.vocabulary.ttl").write_text(
        "@prefix : <http://example.com/> .\n", encoding="utf-8"
    )
    (course_dir / "training_specs" / "instruction_pairs.jsonl").write_text(
        instruction_pairs, encoding="utf-8"
    )
    # Admissible (editorial_or_misconception) rows clearing the default
    # min_dpo_pairs=50. An EMPTY preference file is not a neutral fixture
    # default: under the shipped dpo_fail_hard=true it describes a course the
    # runner must REFUSE to train, so a licensing test built on one would be
    # asserting against a run that never legitimately starts.
    (course_dir / "training_specs" / "preference_pairs.jsonl").write_text(
        "".join(
            json.dumps({
                "prompt": f"Which statement about licensing is correct? ({i})",
                "chosen": "The roster records each teacher's licence verdict.",
                "rejected": "The roster records each teacher's release date.",
                "chunk_id": "c1",
                "source": "misconception",
                "misconception_id": f"mc_{i:016x}",
                "provider": "local",
                "model": "nemotron-3-nano-30b-a3b",
            }) + "\n"
            for i in range(50)
        ),
        encoding="utf-8",
    )
    (course_dir / "training_specs" / "dataset_config.json").write_text(
        '{"format": "instruction-following", "statistics": {}}', encoding="utf-8"
    )
    return libv2_root


def _runner(libv2_root: Path, slug: str = "lic-101") -> TrainingRunner:
    return TrainingRunner(
        course_slug=slug,
        base_model="qwen2.5-1.5b",
        dry_run=True,
        libv2_root=libv2_root,
    )


def test_clean_local_corpus_passes_preflight(tmp_path: Path):
    pairs = "\n".join(
        [
            json.dumps({"prompt": "a", "completion": "b", "provider": "local"}),
            json.dumps({"prompt": "c", "completion": "d"}),  # legacy, no provider
        ]
    )
    libv2_root = _build_course(tmp_path, instruction_pairs=pairs)
    result = _runner(libv2_root).run()
    assert result.model_card_path.exists()


def test_claude_tagged_pair_fails_preflight(tmp_path: Path):
    pairs = json.dumps(
        {"prompt": "a", "completion": "b", "provider": "claude_session", "chunk_id": "c9"}
    )
    libv2_root = _build_course(tmp_path, instruction_pairs=pairs)
    with pytest.raises(LicenseGuardError, match="Claude/Anthropic"):
        _runner(libv2_root).run()


def test_barred_generating_seat_fails_preflight(tmp_path: Path):
    pairs = json.dumps(
        {
            "prompt": "a",
            "completion": "b",
            "provider": "local",
            "generating_seat": "meta-llama/Llama-3.3-70B",
            "chunk_id": "c1",
        }
    )
    libv2_root = _build_course(tmp_path, instruction_pairs=pairs)
    with pytest.raises(LicenseGuardError, match="llama"):
        _runner(libv2_root).run()
