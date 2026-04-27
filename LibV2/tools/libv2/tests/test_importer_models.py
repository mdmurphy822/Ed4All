"""Wave 93 — tests for ``LibV2/tools/libv2/importer.py`` model surface.

Covers:

- ``_COPIED_SUBDIRS`` includes ``models``.
- ``import_model`` validates the card; fails loud on bad cards.
- ``import_model`` writes ``_pointers.json`` correctly when
  ``promote=True`` and ``promote=False``.
- Demotes the previous current correctly when promoting a new model.
- Updates ``CourseManifest.slm_processing`` correctly on import.
- ``list_course_models`` / ``promote_model`` / ``get_model_eval_report``
  helpers behave for the empty / single / multi-model cases.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Project root resolution for direct test invocation
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from LibV2.tools.libv2.importer import (  # noqa: E402
    _COPIED_SUBDIRS,
    _read_pointers_file,
    _validate_model_pointers,
    get_model_eval_report,
    import_model,
    list_course_models,
    promote_model,
)
from LibV2.tools.libv2.validator import ValidationError  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixtures                                                                 #
# ---------------------------------------------------------------------- #


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


_HASH64 = "a" * 64


def _make_card(
    *,
    model_id: str = "qwen2-5-1-5b-tst-101-3a4f8c92",
    course_slug: str = "tst-101",
    pedagogy_hash: str = _HASH64,
    eval_scores: Dict[str, Any] | None = None,
    license_str: str | None = "apache-2.0",
    base_name: str = "qwen2.5-1.5b",
    created_at: str = "2026-04-26T18:30:00Z",
) -> Dict[str, Any]:
    card = {
        "model_id": model_id,
        "course_slug": course_slug,
        "base_model": {
            "name": base_name,
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
            "chunks_hash": _HASH64,
            "pedagogy_graph_hash": pedagogy_hash,
            "instruction_pairs_hash": _HASH64,
            "preference_pairs_hash": _HASH64,
            "concept_graph_hash": _HASH64,
            "vocabulary_ttl_hash": _HASH64,
            "holdout_graph_hash": _HASH64,
        },
        "created_at": created_at,
    }
    if eval_scores is not None:
        card["eval_scores"] = eval_scores
    if license_str is not None:
        card["license"] = license_str
    return card


def _build_repo_with_course(
    tmp_path: Path,
    *,
    slug: str = "tst-101",
    pedagogy_payload: bytes = b'{"nodes": [], "edges": []}',
    write_manifest: bool = True,
) -> tuple[Path, Path, str]:
    """Return (repo_root, course_dir, pedagogy_sha256)."""
    repo_root = tmp_path / "libv2-repo"
    course_dir = repo_root / "courses" / slug
    course_dir.mkdir(parents=True)
    (course_dir / "graph").mkdir()
    (course_dir / "pedagogy").mkdir()
    pedagogy_path = course_dir / "graph" / "pedagogy_graph.json"
    pedagogy_path.write_bytes(pedagogy_payload)

    if write_manifest:
        manifest = {
            "libv2_version": "1.2.0",
            "slug": slug,
            "import_timestamp": "2026-04-20T18:26:45.000000",
            "sourceforge_manifest": {
                "sourceforge_version": "test",
                "export_timestamp": "2026-04-20T18:00:00",
                "course_id": slug,
                "course_title": "Test Course",
            },
            "classification": {
                "division": "STEM",
                "primary_domain": "general",
                "secondary_domains": [],
                "subdomains": [],
                "topics": [],
                "subtopics": [],
            },
            "content_profile": {
                "total_chunks": 0,
                "total_tokens": 0,
                "total_concepts": 0,
                "language": "en",
                "difficulty_distribution": {},
                "chunk_type_distribution": {},
            },
        }
        (course_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    return repo_root, course_dir, _sha256_bytes(pedagogy_payload)


def _build_run_dir(
    tmp_path: Path,
    card: Dict[str, Any],
    *,
    name: str = "run-out",
    weights_bytes: bytes = b"safetensors-fake-bytes" * 100,
    write_eval_report: bool = True,
    write_decision_log: bool = True,
) -> Path:
    """Build a Trainforge-runner-style run dir."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    (run_dir / "model_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8"
    )
    (run_dir / "adapter.safetensors").write_bytes(weights_bytes)
    if write_eval_report:
        (run_dir / "eval_report.json").write_text(
            json.dumps({
                "faithfulness": 0.83,
                "coverage": 0.91,
                "baseline_delta": 0.12,
                "profile": "generic",
                "per_tier": {"faithfulness": {"accuracy": 0.83}},
            }, indent=2),
            encoding="utf-8",
        )
    if write_decision_log:
        (run_dir / "training_run.jsonl").write_text(
            '{"event": "run_complete"}\n', encoding="utf-8"
        )
    return run_dir


# ---------------------------------------------------------------------- #
# _COPIED_SUBDIRS                                                          #
# ---------------------------------------------------------------------- #


def test_copied_subdirs_includes_models():
    """``models`` must be in the canonical copied-subdir list (Wave 93)."""
    assert "models" in _COPIED_SUBDIRS, (
        f"_COPIED_SUBDIRS must include 'models'; got {_COPIED_SUBDIRS!r}"
    )


def test_copied_subdirs_includes_legacy_set():
    """The legacy v0.2.0 set still mirrors so existing imports keep working."""
    legacy = {"corpus", "graph", "pedagogy", "training_specs", "quality"}
    assert legacy.issubset(set(_COPIED_SUBDIRS))


# ---------------------------------------------------------------------- #
# import_model — happy path                                                #
# ---------------------------------------------------------------------- #


def test_import_model_no_promote_writes_target_and_no_pointers(tmp_path):
    repo_root, course_dir, pedagogy_hash = _build_repo_with_course(tmp_path)
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card)

    target = import_model(
        course_slug="tst-101",
        run_dir=run_dir,
        repo_root=repo_root,
        promote=False,
    )

    expected = course_dir / "models" / card["model_id"]
    assert target == expected
    assert (target / "model_card.json").exists()
    assert (target / "adapter.safetensors").exists()
    assert (target / "eval_report.json").exists()
    assert (target / "training_run.jsonl").exists()

    # No promotion → no _pointers.json should have been written
    pointers_path = course_dir / "models" / "_pointers.json"
    assert not pointers_path.exists()


def test_import_model_promote_writes_pointers_and_history(tmp_path):
    repo_root, course_dir, pedagogy_hash = _build_repo_with_course(tmp_path)
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card)

    import_model(
        course_slug="tst-101",
        run_dir=run_dir,
        repo_root=repo_root,
        promote=True,
        promoted_by="test:user",
    )

    pointers_path = course_dir / "models" / "_pointers.json"
    assert pointers_path.exists()
    pointers = json.loads(pointers_path.read_text(encoding="utf-8"))
    assert pointers["current"] == card["model_id"]
    assert len(pointers["history"]) == 1
    entry = pointers["history"][0]
    assert entry["model_id"] == card["model_id"]
    assert entry["promoted_at"]
    assert entry["promoted_by"] == "test:user"
    assert entry["demoted_at"] is None


def test_import_model_promote_then_promote_new_demotes_previous(tmp_path):
    repo_root, course_dir, pedagogy_hash = _build_repo_with_course(tmp_path)

    # First model
    card1 = _make_card(
        pedagogy_hash=pedagogy_hash,
        model_id="qwen2-5-1-5b-tst-101-aaaa1111",
    )
    run1 = _build_run_dir(tmp_path, card1, name="run-1")
    import_model(
        course_slug="tst-101",
        run_dir=run1,
        repo_root=repo_root,
        promote=True,
    )

    # Second model
    card2 = _make_card(
        pedagogy_hash=pedagogy_hash,
        model_id="qwen2-5-1-5b-tst-101-bbbb2222",
    )
    run2 = _build_run_dir(tmp_path, card2, name="run-2")
    import_model(
        course_slug="tst-101",
        run_dir=run2,
        repo_root=repo_root,
        promote=True,
    )

    pointers_path = course_dir / "models" / "_pointers.json"
    pointers = json.loads(pointers_path.read_text(encoding="utf-8"))
    assert pointers["current"] == card2["model_id"]
    # Two history entries
    assert len(pointers["history"]) == 2
    # First entry should now have demoted_at populated
    first = pointers["history"][0]
    assert first["model_id"] == card1["model_id"]
    assert first["demoted_at"] is not None
    # Second entry is current
    second = pointers["history"][1]
    assert second["model_id"] == card2["model_id"]
    assert second["demoted_at"] is None


def test_import_model_updates_manifest_slm_processing(tmp_path):
    repo_root, course_dir, pedagogy_hash = _build_repo_with_course(tmp_path)
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card)

    import_model(
        course_slug="tst-101",
        run_dir=run_dir,
        repo_root=repo_root,
        promote=True,
    )

    manifest = json.loads(
        (course_dir / "manifest.json").read_text(encoding="utf-8")
    )
    slm = manifest.get("slm_processing")
    assert slm is not None, "slm_processing block must be populated"
    assert slm["slm_version"] == f"qwen2.5-1.5b/{card['model_id']}"
    assert slm["processing_timestamp"] == card["created_at"]
    # First import → generation 0, parent_version None
    assert slm["generation"] == 0
    assert slm["parent_version"] is None


def test_second_promotion_increments_generation(tmp_path):
    repo_root, course_dir, pedagogy_hash = _build_repo_with_course(tmp_path)

    card1 = _make_card(
        pedagogy_hash=pedagogy_hash,
        model_id="qwen2-5-1-5b-tst-101-aaaa1111",
    )
    run1 = _build_run_dir(tmp_path, card1, name="run-1")
    import_model("tst-101", run1, repo_root=repo_root, promote=True)

    card2 = _make_card(
        pedagogy_hash=pedagogy_hash,
        model_id="qwen2-5-1-5b-tst-101-bbbb2222",
    )
    run2 = _build_run_dir(tmp_path, card2, name="run-2")
    import_model("tst-101", run2, repo_root=repo_root, promote=True)

    manifest = json.loads(
        (course_dir / "manifest.json").read_text(encoding="utf-8")
    )
    slm = manifest.get("slm_processing")
    assert slm is not None
    assert slm["generation"] == 1
    assert slm["parent_version"] == f"qwen2.5-1.5b/{card1['model_id']}"


# ---------------------------------------------------------------------- #
# import_model — failure modes                                             #
# ---------------------------------------------------------------------- #


def test_import_model_fails_loud_on_bad_card(tmp_path):
    """Card missing a required field surfaces as ValidationError."""
    repo_root, course_dir, pedagogy_hash = _build_repo_with_course(tmp_path)
    card = _make_card(pedagogy_hash=pedagogy_hash)
    # Strip a required field
    del card["adapter_format"]
    run_dir = _build_run_dir(tmp_path, card)

    with pytest.raises(ValidationError):
        import_model(
            course_slug="tst-101",
            run_dir=run_dir,
            repo_root=repo_root,
        )


def test_import_model_fails_when_pedagogy_hash_does_not_match(tmp_path):
    """Critical: pedagogy hash mismatch must fail loud."""
    repo_root, course_dir, _ = _build_repo_with_course(tmp_path)
    # Use a wrong hash
    card = _make_card(pedagogy_hash="0" * 64)
    run_dir = _build_run_dir(tmp_path, card)

    with pytest.raises(ValidationError):
        import_model(
            course_slug="tst-101",
            run_dir=run_dir,
            repo_root=repo_root,
        )


def test_import_model_fails_when_run_dir_missing(tmp_path):
    repo_root, _, _ = _build_repo_with_course(tmp_path)
    with pytest.raises(FileNotFoundError):
        import_model(
            course_slug="tst-101",
            run_dir=tmp_path / "no-such-dir",
            repo_root=repo_root,
        )


def test_import_model_fails_when_card_missing(tmp_path):
    repo_root, _, _ = _build_repo_with_course(tmp_path)
    run_dir = tmp_path / "run-bare"
    run_dir.mkdir()
    (run_dir / "adapter.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        import_model(
            course_slug="tst-101",
            run_dir=run_dir,
            repo_root=repo_root,
        )


def test_import_model_fails_when_course_missing(tmp_path):
    repo_root = tmp_path / "libv2-repo"
    (repo_root / "courses").mkdir(parents=True)
    card = _make_card()
    run_dir = _build_run_dir(tmp_path, card)
    with pytest.raises(FileNotFoundError):
        import_model(
            course_slug="ghost-101",
            run_dir=run_dir,
            repo_root=repo_root,
        )


def test_import_model_refuses_overwrite(tmp_path):
    repo_root, course_dir, pedagogy_hash = _build_repo_with_course(tmp_path)
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card)
    import_model(
        course_slug="tst-101",
        run_dir=run_dir,
        repo_root=repo_root,
    )
    with pytest.raises(FileExistsError):
        import_model(
            course_slug="tst-101",
            run_dir=run_dir,
            repo_root=repo_root,
        )


# ---------------------------------------------------------------------- #
# Helpers: list / promote / eval_report                                    #
# ---------------------------------------------------------------------- #


def test_list_course_models_empty(tmp_path):
    repo_root, course_dir, _ = _build_repo_with_course(tmp_path)
    info = list_course_models("tst-101", repo_root)
    assert info == {"current": None, "models": []}


def test_list_course_models_after_import(tmp_path):
    repo_root, _, pedagogy_hash = _build_repo_with_course(tmp_path)
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card)
    import_model(
        course_slug="tst-101",
        run_dir=run_dir,
        repo_root=repo_root,
        promote=True,
    )

    info = list_course_models("tst-101", repo_root)
    assert info["current"] == card["model_id"]
    assert len(info["models"]) == 1
    only = info["models"][0]
    assert only["model_id"] == card["model_id"]
    assert only["is_current"] is True
    assert only["adapter_format"] == "safetensors"
    assert "faithfulness" in only["eval_scores"]


def test_promote_model_demotes_previous(tmp_path):
    repo_root, course_dir, pedagogy_hash = _build_repo_with_course(tmp_path)

    card1 = _make_card(
        pedagogy_hash=pedagogy_hash,
        model_id="qwen2-5-1-5b-tst-101-aaaa1111",
    )
    run1 = _build_run_dir(tmp_path, card1, name="run-1")
    import_model("tst-101", run1, repo_root=repo_root, promote=True)

    # Second model imported but not promoted yet
    card2 = _make_card(
        pedagogy_hash=pedagogy_hash,
        model_id="qwen2-5-1-5b-tst-101-bbbb2222",
    )
    run2 = _build_run_dir(tmp_path, card2, name="run-2")
    import_model("tst-101", run2, repo_root=repo_root, promote=False)

    pointers_path = promote_model(
        course_slug="tst-101",
        model_id=card2["model_id"],
        repo_root=repo_root,
    )
    pointers = json.loads(pointers_path.read_text(encoding="utf-8"))
    assert pointers["current"] == card2["model_id"]
    # Two history entries; first should be demoted
    assert len(pointers["history"]) == 2
    assert pointers["history"][0]["demoted_at"] is not None
    assert pointers["history"][1]["demoted_at"] is None


def test_promote_model_idempotent_when_already_current(tmp_path):
    repo_root, _, pedagogy_hash = _build_repo_with_course(tmp_path)
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card)
    import_model("tst-101", run_dir, repo_root=repo_root, promote=True)

    # Promoting again is a no-op (history not appended twice)
    pointers_path = promote_model("tst-101", card["model_id"], repo_root=repo_root)
    pointers = json.loads(pointers_path.read_text(encoding="utf-8"))
    assert pointers["current"] == card["model_id"]
    assert len(pointers["history"]) == 1


def test_promote_model_unknown_id_raises(tmp_path):
    repo_root, _, _ = _build_repo_with_course(tmp_path)
    with pytest.raises(FileNotFoundError):
        promote_model("tst-101", "no-such-model", repo_root=repo_root)


def test_get_model_eval_report_missing_returns_none(tmp_path):
    repo_root, course_dir, pedagogy_hash = _build_repo_with_course(tmp_path)
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card, write_eval_report=False)
    import_model("tst-101", run_dir, repo_root=repo_root)

    assert get_model_eval_report(
        "tst-101", card["model_id"], repo_root=repo_root,
    ) is None


def test_get_model_eval_report_returns_payload(tmp_path):
    repo_root, _, pedagogy_hash = _build_repo_with_course(tmp_path)
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card, write_eval_report=True)
    import_model("tst-101", run_dir, repo_root=repo_root)

    report = get_model_eval_report(
        "tst-101", card["model_id"], repo_root=repo_root,
    )
    assert report is not None
    assert "faithfulness" in report
    assert report["profile"] == "generic"


# ---------------------------------------------------------------------- #
# _pointers.json schema validation                                         #
# ---------------------------------------------------------------------- #


def test_validate_model_pointers_accepts_minimal_skeleton():
    _validate_model_pointers({"current": None, "history": []})


def test_validate_model_pointers_rejects_missing_history():
    with pytest.raises(ValueError):
        _validate_model_pointers({"current": "m1"})


def test_validate_model_pointers_rejects_history_entry_without_promoted_at():
    with pytest.raises(ValueError):
        _validate_model_pointers({
            "current": "m1",
            "history": [{"model_id": "m1"}],
        })


def test_read_pointers_file_returns_skeleton_when_missing(tmp_path):
    pointers_path = tmp_path / "_pointers.json"
    out = _read_pointers_file(pointers_path)
    assert out == {"current": None, "history": []}
