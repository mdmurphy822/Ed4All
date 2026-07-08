"""Tests for the LibV2 library-format migration framework (OP4 stage 2).

Two layers:

* **Engine** (``LibV2.tools.libv2.migrate``): version detection, the ordered
  registry / plan chaining, and the backup + validate + rollback apply engine
  driven with an injected validator (hermetic, no schema files needed).
* **CLI** (``libv2 migrate``): dry-run-by-default plan, ``--apply``,
  already-current no-op, and ``--all`` discovery, driven through ``CliRunner``
  over synthetic ``tmp_path`` courses — no live slugs, no fixture corpora.

All courses here are minimal synthetic manifests built in the test.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from LibV2.tools.libv2.cli import main as libv2_main
from LibV2.tools.libv2.migrate import (
    DEFAULT_REGISTRY,
    LEGACY_VERSION,
    MigrationError,
    MigrationRegistry,
    MigrationStep,
    apply_course_migration,
    detect_version,
    discover_courses,
    plan_course_migration,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_SHA = "a" * 64


def _conforming_manifest(slug: str, *, with_version: bool = False) -> dict:
    """A manifest that satisfies the real course_manifest schema + validators.

    Omits ``library_format_version`` unless ``with_version`` — the legacy shape
    the migration stamps.
    """
    m = {
        "libv2_version": "1.2.0",
        "slug": slug,
        "import_timestamp": "2026-07-08T00:00:00Z",
        "sourceforge_manifest": {
            "sourceforge_version": "1.0",
            "export_timestamp": "2026-07-08T00:00:00Z",
            "course_id": slug,
            "course_title": slug,
        },
        "classification": {"division": "STEM", "primary_domain": "physics"},
        "content_profile": {"total_chunks": 1, "total_tokens": 10},
        "dart_chunks_sha256": _SHA,
        "imscc_chunks_sha256": _SHA,
        "concept_graph_sha256": _SHA,
    }
    if with_version:
        m["library_format_version"] = "1.0"
    return m


def _make_course(root: Path, slug: str, *, with_version: bool = False) -> Path:
    """Build a structurally-valid synthetic course under ``root/courses/<slug>``."""
    cdir = root / "courses" / slug
    (cdir / "graph").mkdir(parents=True)
    (cdir / "graph" / "concept_graph.json").write_text("{}", encoding="utf-8")
    (cdir / "imscc_chunks").mkdir(parents=True)
    (cdir / "imscc_chunks" / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    (cdir / "manifest.json").write_text(
        json.dumps(_conforming_manifest(slug, with_version=with_version)),
        encoding="utf-8",
    )
    (root / "catalog").mkdir(parents=True, exist_ok=True)
    return cdir


def _read_manifest(cdir: Path) -> dict:
    return json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Version detection + registry                                                #
# --------------------------------------------------------------------------- #

def test_detect_version_missing_is_legacy():
    assert detect_version({}) == LEGACY_VERSION
    assert detect_version({"library_format_version": ""}) == LEGACY_VERSION


def test_detect_version_present():
    assert detect_version({"library_format_version": "1.0"}) == "1.0"


def test_default_registry_target_is_1_0():
    assert DEFAULT_REGISTRY.target_version() == "1.0"


def test_default_registry_matches_pipeline_constant():
    """Drift guard: the registry terminus must equal the emitter's stamp."""
    try:
        from MCP.tools.pipeline_tools import LIBRARY_FORMAT_VERSION
    except Exception:  # pragma: no cover — import unavailable
        import pytest

        pytest.skip("pipeline_tools import unavailable")
    assert DEFAULT_REGISTRY.target_version() == LIBRARY_FORMAT_VERSION


def test_plan_steps_legacy_chains_to_target():
    steps = DEFAULT_REGISTRY.plan_steps(LEGACY_VERSION)
    assert [s.to_version for s in steps] == ["1.0"]
    assert steps[0].from_version == LEGACY_VERSION


def test_plan_steps_already_current_is_empty():
    assert DEFAULT_REGISTRY.plan_steps("1.0") == []


def test_plan_steps_unknown_version_raises():
    import pytest

    with pytest.raises(MigrationError) as exc:
        DEFAULT_REGISTRY.plan_steps("0.9")
    assert exc.value.code == "UNKNOWN_VERSION"


def test_registry_rejects_duplicate_step():
    import pytest

    reg = MigrationRegistry()
    reg.register(MigrationStep(LEGACY_VERSION, "1.0", "d", lambda m: m))
    with pytest.raises(MigrationError):
        reg.register(MigrationStep(LEGACY_VERSION, "1.1", "d", lambda m: m))


def test_multi_step_chain_orders_correctly():
    reg = MigrationRegistry()

    def to_10(m):
        m = dict(m)
        m["library_format_version"] = "1.0"
        return m

    def to_11(m):
        m = dict(m)
        m["library_format_version"] = "1.1"
        return m

    reg.register(MigrationStep(LEGACY_VERSION, "1.0", "legacy->1.0", to_10))
    reg.register(MigrationStep("1.0", "1.1", "1.0->1.1", to_11))
    assert reg.target_version() == "1.1"
    steps = reg.plan_steps(LEGACY_VERSION)
    assert [s.to_version for s in steps] == ["1.0", "1.1"]


# --------------------------------------------------------------------------- #
# Plan engine                                                                 #
# --------------------------------------------------------------------------- #

def test_plan_course_migration_legacy(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101")
    plan = plan_course_migration(cdir, "demo-101")
    assert plan.from_version == LEGACY_VERSION
    assert plan.to_version == "1.0"
    assert not plan.already_current
    assert len(plan.steps) == 1


def test_plan_course_migration_already_current(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101", with_version=True)
    plan = plan_course_migration(cdir, "demo-101")
    assert plan.already_current
    assert plan.steps == []


def test_plan_missing_manifest_raises(tmp_path):
    import pytest

    cdir = tmp_path / "courses" / "demo-101"
    cdir.mkdir(parents=True)
    with pytest.raises(MigrationError) as exc:
        plan_course_migration(cdir, "demo-101")
    assert exc.value.code == "MANIFEST_NOT_FOUND"


# --------------------------------------------------------------------------- #
# Apply engine — backup + validate + rollback                                 #
# --------------------------------------------------------------------------- #

class _OkValidator:
    valid = True
    errors: list = []


class _FailValidator:
    valid = False
    errors = ["synthetic validation failure"]


def test_apply_stamps_and_backs_up(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101")

    result = apply_course_migration(
        cdir, root, "demo-101", validator=lambda c, r: _OkValidator()
    )

    assert result.applied
    assert not result.rolled_back
    assert result.steps_applied == ["legacy->1.0"]
    # Manifest now carries the stamp.
    assert _read_manifest(cdir)["library_format_version"] == "1.0"
    # A timestamped backup sibling exists and holds the PRE-migration bytes.
    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert "library_format_version" not in json.loads(
        result.backup_path.read_text(encoding="utf-8")
    )


def test_apply_already_current_is_noop(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101", with_version=True)
    before = (cdir / "manifest.json").read_bytes()

    result = apply_course_migration(
        cdir, root, "demo-101", validator=lambda c, r: _FailValidator()
    )

    assert result.already_current
    assert not result.applied
    assert result.backup_path is None
    # Byte-identical: no write, no backup even though validator would fail.
    assert (cdir / "manifest.json").read_bytes() == before
    # No .bak siblings were created.
    assert not list(cdir.glob("manifest.json.*.bak"))


def test_apply_rolls_back_on_validation_failure(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101")
    before = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))

    result = apply_course_migration(
        cdir, root, "demo-101", validator=lambda c, r: _FailValidator()
    )

    assert not result.applied
    assert result.rolled_back
    assert result.validation_errors == ["synthetic validation failure"]
    # Manifest restored to the pre-migration content (no stamp).
    after = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))
    assert after == before
    assert "library_format_version" not in after
    # The backup that enabled the rollback still exists.
    assert result.backup_path is not None and result.backup_path.exists()


def test_apply_real_validator_passes_end_to_end(tmp_path):
    """A schema-conforming legacy course migrates cleanly under the real
    ``validate_course`` pass (structure + schema + taxonomy)."""
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101")

    result = apply_course_migration(cdir, root, "demo-101")  # default validator

    assert result.applied, result.validation_errors
    assert not result.rolled_back
    assert _read_manifest(cdir)["library_format_version"] == "1.0"


def test_discover_courses(tmp_path):
    root = tmp_path / "libv2"
    _make_course(root, "demo-a")
    _make_course(root, "demo-b")
    (root / "courses" / ".hidden").mkdir()
    found = [c.name for c in discover_courses(root)]
    assert found == ["demo-a", "demo-b"]


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _invoke(root: Path, *args, **kwargs):
    runner = CliRunner()
    return runner.invoke(libv2_main, ["--repo", str(root), "migrate", *args], **kwargs)


def test_cli_help_renders():
    runner = CliRunner()
    result = runner.invoke(libv2_main, ["migrate", "--help"])
    assert result.exit_code == 0, result.output
    assert "DRY-RUN BY DEFAULT" in result.output
    assert "--apply" in result.output
    assert "--all" in result.output


def test_cli_dry_run_plan_prints_without_writing(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101")
    before = (cdir / "manifest.json").read_bytes()

    result = _invoke(root, "demo-101")
    assert result.exit_code == 0, result.output
    assert "PLAN legacy -> 1.0" in result.output
    assert "dry-run" in result.output
    # Nothing written.
    assert (cdir / "manifest.json").read_bytes() == before
    assert not list(cdir.glob("manifest.json.*.bak"))


def test_cli_apply_migrates_and_backs_up(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101")

    result = _invoke(root, "demo-101", "--apply")
    assert result.exit_code == 0, result.output
    assert "migrated legacy -> 1.0" in result.output
    assert _read_manifest(cdir)["library_format_version"] == "1.0"
    assert list(cdir.glob("manifest.json.*.bak"))


def test_cli_already_current_noop(tmp_path):
    root = tmp_path / "libv2"
    _make_course(root, "demo-101", with_version=True)

    result = _invoke(root, "demo-101", "--apply")
    assert result.exit_code == 0, result.output
    assert "already current" in result.output


def test_cli_missing_course(tmp_path):
    root = tmp_path / "libv2"
    (root / "courses").mkdir(parents=True)
    (root / "catalog").mkdir(parents=True)
    result = _invoke(root, "nonexistent")
    assert result.exit_code == 1
    assert "Course not found" in result.output


def test_cli_requires_slug_or_all(tmp_path):
    root = tmp_path / "libv2"
    (root / "courses").mkdir(parents=True)
    (root / "catalog").mkdir(parents=True)
    result = _invoke(root)
    assert result.exit_code == 1
    assert "Specify a course SLUG or --all" in result.output


def test_cli_slug_and_all_conflict(tmp_path):
    root = tmp_path / "libv2"
    _make_course(root, "demo-101")
    result = _invoke(root, "demo-101", "--all")
    assert result.exit_code == 1
    assert "not both" in result.output


def test_cli_all_iterates_discovered_courses(tmp_path):
    root = tmp_path / "libv2"
    _make_course(root, "demo-a")
    _make_course(root, "demo-b")
    _make_course(root, "demo-current", with_version=True)

    result = _invoke(root, "--all", "--apply")
    assert result.exit_code == 0, result.output
    # Both legacy courses migrated; the current one is a no-op.
    assert _read_manifest(root / "courses" / "demo-a")["library_format_version"] == "1.0"
    assert _read_manifest(root / "courses" / "demo-b")["library_format_version"] == "1.0"
    assert "demo-current: already current" in result.output


def test_cli_all_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "libv2"
    ca = _make_course(root, "demo-a")
    cb = _make_course(root, "demo-b")
    before_a = ca.joinpath("manifest.json").read_bytes()
    before_b = cb.joinpath("manifest.json").read_bytes()

    result = _invoke(root, "--all")
    assert result.exit_code == 0, result.output
    assert ca.joinpath("manifest.json").read_bytes() == before_a
    assert cb.joinpath("manifest.json").read_bytes() == before_b
