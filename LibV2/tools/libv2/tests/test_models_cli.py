"""Wave 93 — tests for the ``libv2 models`` + ``libv2 import-model`` CLIs.

Uses Click's :class:`CliRunner` to invoke the CLI in isolation against
synthetic LibV2 courses staged under ``tmp_path``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from click.testing import CliRunner

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from LibV2.tools.libv2.cli import main as libv2_main  # noqa: E402


_HASH64 = "a" * 64


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_card(
    *,
    model_id: str = "qwen2-5-1-5b-tst-101-3a4f8c92",
    course_slug: str = "tst-101",
    pedagogy_hash: str = _HASH64,
    eval_scores: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    card = {
        "model_id": model_id,
        "course_slug": course_slug,
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
            "chunks_hash": _HASH64,
            "pedagogy_graph_hash": pedagogy_hash,
            "instruction_pairs_hash": _HASH64,
            "preference_pairs_hash": _HASH64,
            "concept_graph_hash": _HASH64,
            "vocabulary_ttl_hash": _HASH64,
            "holdout_graph_hash": _HASH64,
        },
        "created_at": "2026-04-26T18:30:00Z",
        "license": "apache-2.0",
    }
    if eval_scores is not None:
        card["eval_scores"] = eval_scores
    return card


@pytest.fixture
def libv2_repo(tmp_path: Path):
    """Build a synthetic LibV2 repo with one course skeleton."""
    repo_root = tmp_path / "libv2-repo"
    slug = "tst-101"
    course_dir = repo_root / "courses" / slug
    course_dir.mkdir(parents=True)
    (course_dir / "graph").mkdir()
    pedagogy_payload = b'{"nodes": [], "edges": []}'
    (course_dir / "graph" / "pedagogy_graph.json").write_bytes(pedagogy_payload)

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

    return repo_root, slug, _sha256_bytes(pedagogy_payload)


def _build_run_dir(
    tmp_path: Path,
    card: Dict[str, Any],
    *,
    name: str = "run-out",
    write_eval_report: bool = True,
) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8"
    )
    (run_dir / "adapter.safetensors").write_bytes(b"safetensors-fake-bytes" * 100)
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
    (run_dir / "training_run.jsonl").write_text(
        '{"event": "run_complete"}\n', encoding="utf-8"
    )
    return run_dir


# ---------------------------------------------------------------------- #
# import-model                                                              #
# ---------------------------------------------------------------------- #


def test_import_model_cli_smoke(libv2_repo, tmp_path):
    """``libv2 import-model`` lands the run dir under courses/<slug>/models/."""
    repo_root, slug, pedagogy_hash = libv2_repo
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card)

    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "import-model", str(run_dir),
        "--course", slug,
    ])
    assert result.exit_code == 0, result.output
    target = repo_root / "courses" / slug / "models" / card["model_id"]
    assert target.exists()
    assert (target / "model_card.json").exists()
    assert (target / "adapter.safetensors").exists()


def test_import_model_cli_with_promote(libv2_repo, tmp_path):
    """``libv2 import-model --promote`` writes _pointers.json."""
    repo_root, slug, pedagogy_hash = libv2_repo
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card)

    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "import-model", str(run_dir),
        "--course", slug,
        "--promote",
        "--promoted-by", "test:user",
    ])
    assert result.exit_code == 0, result.output
    pointers = json.loads(
        (repo_root / "courses" / slug / "models" / "_pointers.json")
        .read_text(encoding="utf-8")
    )
    assert pointers["current"] == card["model_id"]
    assert len(pointers["history"]) == 1


def test_import_model_cli_fails_loud_on_bad_card(libv2_repo, tmp_path):
    """Bad model card surfaces as non-zero exit + helpful message."""
    repo_root, slug, pedagogy_hash = libv2_repo
    card = _make_card(pedagogy_hash=pedagogy_hash)
    del card["adapter_format"]
    run_dir = _build_run_dir(tmp_path, card)

    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "import-model", str(run_dir),
        "--course", slug,
    ])
    assert result.exit_code != 0
    assert "validation failed" in result.output.lower() or "validation" in result.output.lower()


# ---------------------------------------------------------------------- #
# models list                                                              #
# ---------------------------------------------------------------------- #


def test_models_list_empty_course(libv2_repo):
    """``libv2 models list`` on a fresh course reports no models."""
    repo_root, slug, _ = libv2_repo
    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "list", slug,
    ])
    assert result.exit_code == 0, result.output
    assert "No models" in result.output


def test_models_list_after_import(libv2_repo, tmp_path):
    """``libv2 models list`` shows imported models, stars current."""
    repo_root, slug, pedagogy_hash = libv2_repo
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card)

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "import-model", str(run_dir),
        "--course", slug,
        "--promote",
    ])

    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "list", slug,
    ])
    assert result.exit_code == 0, result.output
    assert card["model_id"] in result.output


def test_models_list_json_output(libv2_repo, tmp_path):
    repo_root, slug, pedagogy_hash = libv2_repo
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card)

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "import-model", str(run_dir),
        "--course", slug,
        "--promote",
    ])

    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "list", slug,
        "-o", "json",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["current"] == card["model_id"]
    assert len(payload["models"]) == 1


def test_models_list_unknown_course(libv2_repo):
    repo_root, _, _ = libv2_repo
    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "list", "no-such-course",
    ])
    assert result.exit_code != 0


# ---------------------------------------------------------------------- #
# models promote                                                           #
# ---------------------------------------------------------------------- #


def test_models_promote_cli(libv2_repo, tmp_path):
    repo_root, slug, pedagogy_hash = libv2_repo

    # Import two models without promoting either
    card1 = _make_card(
        pedagogy_hash=pedagogy_hash,
        model_id="qwen2-5-1-5b-tst-101-aaaa1111",
    )
    run1 = _build_run_dir(tmp_path, card1, name="run-1")
    card2 = _make_card(
        pedagogy_hash=pedagogy_hash,
        model_id="qwen2-5-1-5b-tst-101-bbbb2222",
    )
    run2 = _build_run_dir(tmp_path, card2, name="run-2")

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "import-model", str(run1),
        "--course", slug,
        "--promote",
    ])
    runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "import-model", str(run2),
        "--course", slug,
    ])

    # Now promote the second
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "promote", slug, card2["model_id"],
        "--promoted-by", "cli:tester",
    ])
    assert result.exit_code == 0, result.output
    assert card2["model_id"] in result.output

    pointers = json.loads(
        (repo_root / "courses" / slug / "models" / "_pointers.json")
        .read_text(encoding="utf-8")
    )
    assert pointers["current"] == card2["model_id"]
    # Two history entries; first must be demoted
    assert len(pointers["history"]) == 2
    assert pointers["history"][0]["demoted_at"] is not None


def test_models_promote_unknown_model_fails(libv2_repo):
    repo_root, slug, _ = libv2_repo
    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "promote", slug, "no-such-model",
    ])
    assert result.exit_code != 0


# ---------------------------------------------------------------------- #
# models eval                                                              #
# ---------------------------------------------------------------------- #


def test_models_eval_cli_prints_cached_report(libv2_repo, tmp_path):
    repo_root, slug, pedagogy_hash = libv2_repo
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card, write_eval_report=True)

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "import-model", str(run_dir),
        "--course", slug,
        "--promote",
    ])

    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "eval", slug, card["model_id"],
    ])
    assert result.exit_code == 0, result.output
    assert "faithfulness" in result.output


def test_models_eval_cli_reports_missing_report(libv2_repo, tmp_path):
    repo_root, slug, pedagogy_hash = libv2_repo
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card, write_eval_report=False)

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "import-model", str(run_dir),
        "--course", slug,
        "--promote",
    ])

    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "eval", slug, card["model_id"],
    ])
    assert result.exit_code == 0, result.output
    assert "No eval_report.json" in result.output or "has not" in result.output.lower()


def test_models_eval_cli_unknown_model_fails(libv2_repo):
    repo_root, slug, _ = libv2_repo
    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "eval", slug, "no-such-model",
    ])
    assert result.exit_code != 0


def test_models_eval_cli_json_output(libv2_repo, tmp_path):
    repo_root, slug, pedagogy_hash = libv2_repo
    card = _make_card(pedagogy_hash=pedagogy_hash)
    run_dir = _build_run_dir(tmp_path, card, write_eval_report=True)

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "import-model", str(run_dir),
        "--course", slug,
        "--promote",
    ])

    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "eval", slug, card["model_id"],
        "-o", "json",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "faithfulness" in payload


# ---------------------------------------------------------------------- #
# Help surface                                                             #
# ---------------------------------------------------------------------- #


def test_models_group_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(libv2_main, ["models", "--help"])
    assert result.exit_code == 0
    for sub in ("list", "promote", "eval"):
        assert sub in result.output
