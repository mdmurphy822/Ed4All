"""archive_to_libv2 threads synthesized objectives into the archive.

BLOCKER 2 (libv2_archival objectives plumbing). The strict
``packet_integrity_strict`` gate at the ``libv2_archival`` phase reads
``archive_root / "objectives.json"`` FIRST (with a ``course.json``
fallback) and its ``co_has_parent`` rule requires every component
objective to carry a ``parent_terminal`` resolving to an existing
terminal. The Trainforge-flattened ``course.json`` drops that back-pointer,
so a pipeline-built archive WITHOUT an ``objectives.json`` failed the gate
with one ``ORPHAN_COMPONENT_OBJECTIVE`` per CO.

These tests assert that ``_archive_to_libv2`` now projects the canonical
Worker-A ``objectives.json`` (with ``parent_terminal`` restored from the
synthesized objectives' ``terminal_id``) from the course_planning-emitted
``synthesized_objectives.json``, and that the result satisfies the
packet-integrity ``co_has_parent`` rule.

* The synthetic test is deterministic and always runs.
* The tier-2 test discovers a real Courseforge export dynamically and
  skips when none is present (no hardcoded course slug / path).
"""

from __future__ import annotations

import asyncio
import glob
import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402
from lib.libv2_storage import project_objectives_for_archive  # noqa: E402
from lib.validators.libv2.packet_integrity import (  # noqa: E402
    PacketIntegrityValidator,
)


# --------------------------------------------------------------------------- #
# Unit: projection helper
# --------------------------------------------------------------------------- #


def test_project_objectives_restores_parent_terminal():
    """Every CO with a ``terminal_id`` gets a resolving ``parent_terminal``."""
    synth = {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Understand primes."},
            {"id": "TO-02", "statement": "Apply factoring."},
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {"id": "CO-01", "statement": "Define a prime.",
                     "terminal_id": "TO-01"},
                    {"id": "CO-02", "statement": "Factor 48.",
                     "terminal_id": "TO-02"},
                ],
            },
        ],
    }
    proj = project_objectives_for_archive(synth)
    assert {t["id"] for t in proj["terminal_outcomes"]} == {"TO-01", "TO-02"}
    cos = {c["id"]: c for c in proj["component_objectives"]}
    assert cos["CO-01"]["parent_terminal"] == "to-01"
    assert cos["CO-02"]["parent_terminal"] == "to-02"


def test_project_objectives_anti_fabrication_no_terminal():
    """A CO with no resolvable back-pointer is emitted WITHOUT a
    fabricated parent_terminal (honest gap, not papered over)."""
    synth = {
        "terminal_outcomes": [{"id": "TO-01", "statement": "x"}],
        "component_objectives": [{"id": "CO-99", "statement": "orphan"}],
    }
    proj = project_objectives_for_archive(synth)
    assert proj["component_objectives"][0].get("parent_terminal") in (None, "")


# --------------------------------------------------------------------------- #
# Integration: registry archive emits objectives.json
# --------------------------------------------------------------------------- #


def _make_synthesized(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Understand primes.",
             "bloom_level": "understand"},
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {"id": "CO-01", "statement": "Define a prime number.",
                     "bloom_level": "understand", "terminal_id": "TO-01"},
                    {"id": "CO-02", "statement": "List the first five primes.",
                     "bloom_level": "remember", "terminal_id": "TO-01"},
                ],
            },
        ],
    }), encoding="utf-8")


def test_archive_emits_objectives_json_with_parent_terminal(
    monkeypatch, tmp_path
):
    """The registry _archive_to_libv2 writes objectives.json (Worker-A
    shape) with parent_terminal on every CO."""
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)

    export = tmp_path / "export"
    trainforge_dir = export / "trainforge"
    (trainforge_dir / "corpus").mkdir(parents=True)
    (trainforge_dir / "corpus" / "chunks.jsonl").write_text(
        json.dumps({
            "id": "objplumb_test_chunk_00001",
            "chunk_type": "explanation",
            "text": "A prime number has exactly two divisors.",
            "learning_outcome_refs": ["to-01", "co-01", "co-02"],
        }) + "\n",
        encoding="utf-8",
    )
    _make_synthesized(
        export / "01_learning_objectives" / "synthesized_objectives.json"
    )

    registry = _build_tool_registry()
    tool = registry["archive_to_libv2"]

    course_name = "OBJPLUMB_TEST"
    slug = course_name.lower().replace("_", "-")
    result = json.loads(asyncio.run(tool(
        course_name=course_name,
        project_workspace=str(trainforge_dir),
        pdf_paths="",
        html_paths="",
    )))
    assert "error" not in result, result

    course_dir = tmp_path / "LibV2" / "courses" / slug
    objectives_path = course_dir / "objectives.json"
    assert objectives_path.exists(), (
        f"objectives.json not emitted to {objectives_path}"
    )
    objectives = json.loads(objectives_path.read_text(encoding="utf-8"))
    assert {t["id"] for t in objectives["terminal_outcomes"]} == {"TO-01"}
    cos = objectives["component_objectives"]
    assert {c["id"] for c in cos} == {"CO-01", "CO-02"}
    assert all(c.get("parent_terminal") == "to-01" for c in cos), cos


def test_archive_objectives_pass_co_has_parent_rule(monkeypatch, tmp_path):
    """The emitted objectives.json satisfies the packet-integrity
    co_has_parent rule (the rule that broke without this plumbing)."""
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)

    export = tmp_path / "export"
    trainforge_dir = export / "trainforge"
    (trainforge_dir / "corpus").mkdir(parents=True)
    (trainforge_dir / "corpus" / "chunks.jsonl").write_text(
        json.dumps({
            "id": "objplumb_test2_chunk_00001",
            "chunk_type": "explanation",
            "text": "A prime number has exactly two divisors.",
            "learning_outcome_refs": ["to-01", "co-01", "co-02"],
        }) + "\n",
        encoding="utf-8",
    )
    _make_synthesized(
        export / "01_learning_objectives" / "synthesized_objectives.json"
    )

    registry = _build_tool_registry()
    tool = registry["archive_to_libv2"]
    course_name = "OBJPLUMB_TEST2"
    slug = course_name.lower().replace("_", "-")
    result = json.loads(asyncio.run(tool(
        course_name=course_name,
        project_workspace=str(trainforge_dir),
        pdf_paths="",
        html_paths="",
    )))
    assert "error" not in result, result

    course_dir = tmp_path / "LibV2" / "courses" / slug
    v = PacketIntegrityValidator(strict_coverage=True)
    report = v.validate(course_dir)
    co_orphans = [
        i for i in report.issues
        if i.issue_code == "ORPHAN_COMPONENT_OBJECTIVE"
    ]
    assert not co_orphans, (
        "co_has_parent regressed: objectives.json should carry "
        f"parent_terminal on every CO; got {co_orphans}"
    )
    assert report.summary.get("objectives_source") == "objectives.json", (
        "validator should read the projected objectives.json, not fall "
        f"back to course.json; summary={report.summary}"
    )


# --------------------------------------------------------------------------- #
# W2.3: fail-closed objectives plumbing (no silent no-op)
# --------------------------------------------------------------------------- #


def _make_trainforge(export: Path, chunk_prefix: str) -> Path:
    """Build a minimal export/trainforge dir with a course-matching chunk so
    the chunks-freshness gate passes (mirrors the passing tests above)."""
    trainforge_dir = export / "trainforge"
    (trainforge_dir / "corpus").mkdir(parents=True)
    (trainforge_dir / "corpus" / "chunks.jsonl").write_text(
        json.dumps({
            "id": f"{chunk_prefix}_chunk_00001",
            "chunk_type": "explanation",
            "text": "A prime number has exactly two divisors.",
            "learning_outcome_refs": ["to-01", "co-01"],
        }) + "\n",
        encoding="utf-8",
    )
    return trainforge_dir


def test_missing_source_emits_warning_but_succeeds_by_default(
    monkeypatch, tmp_path
):
    """W2.3: when no synthesized_objectives.json resolves, archival still
    succeeds (default OFF) but surfaces COURSE_JSON_OBJECTIVES_MISSING and
    objectives_status=missing_source (the historical silent no-op is gone)."""
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES", raising=False)

    export = tmp_path / "export"
    trainforge_dir = _make_trainforge(export, "objplumb_miss")
    # NOTE: deliberately NO synthesized_objectives.json.

    registry = _build_tool_registry()
    tool = registry["archive_to_libv2"]
    result = json.loads(asyncio.run(tool(
        course_name="OBJPLUMB_MISS",
        project_workspace=str(trainforge_dir),
        pdf_paths="",
        html_paths="",
    )))
    assert result.get("success") is True, result
    assert result["objectives_status"] == "missing_source", result
    codes = {w["code"] for w in result.get("warnings", [])}
    assert "COURSE_JSON_OBJECTIVES_MISSING" in codes, result
    # No objectives.json written.
    slug = "objplumb-miss"
    assert not (tmp_path / "LibV2" / "courses" / slug / "objectives.json").exists()


def test_empty_projection_does_not_write_empty_objectives(monkeypatch, tmp_path):
    """W2.3: a synthesized file that projects to ZERO objectives is a no-op —
    we must NOT write an empty objectives.json, and status=empty is surfaced."""
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES", raising=False)

    export = tmp_path / "export"
    trainforge_dir = _make_trainforge(export, "objplumb_empty")
    synth_path = export / "01_learning_objectives" / "synthesized_objectives.json"
    synth_path.parent.mkdir(parents=True, exist_ok=True)
    synth_path.write_text(
        json.dumps({"terminal_objectives": [], "chapter_objectives": []}),
        encoding="utf-8",
    )

    registry = _build_tool_registry()
    tool = registry["archive_to_libv2"]
    result = json.loads(asyncio.run(tool(
        course_name="OBJPLUMB_EMPTY",
        project_workspace=str(trainforge_dir),
        pdf_paths="",
        html_paths="",
    )))
    assert result.get("success") is True, result
    assert result["objectives_status"] == "empty", result
    codes = {w["code"] for w in result.get("warnings", [])}
    assert "COURSE_JSON_OBJECTIVES_MISSING" in codes, result
    slug = "objplumb-empty"
    assert not (
        tmp_path / "LibV2" / "courses" / slug / "objectives.json"
    ).exists()


def test_fail_closed_flag_blocks_manifest_on_missing_objectives(
    monkeypatch, tmp_path
):
    """W2.3: with ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES set, a no-op objectives
    projection fails closed — success=False, error_code
    COURSE_JSON_OBJECTIVES_MISSING, and NO manifest is written."""
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES", "1")

    export = tmp_path / "export"
    trainforge_dir = _make_trainforge(export, "objplumb_fc")
    # No synthesized_objectives.json → plumbing no-ops → fail closed.

    registry = _build_tool_registry()
    tool = registry["archive_to_libv2"]
    result = json.loads(asyncio.run(tool(
        course_name="OBJPLUMB_FC",
        project_workspace=str(trainforge_dir),
        pdf_paths="",
        html_paths="",
    )))
    assert result.get("success") is False, result
    assert result.get("error_code") == "COURSE_JSON_OBJECTIVES_MISSING", result
    slug = "objplumb-fc"
    assert not (
        tmp_path / "LibV2" / "courses" / slug / "manifest.json"
    ).exists(), "manifest must not be written when failing closed"


def test_valid_objectives_still_archived_and_no_warning(monkeypatch, tmp_path):
    """W2.3 back-compat: a valid synthesized file still archives objectives
    and emits status=archived with an empty warnings list."""
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES", raising=False)

    export = tmp_path / "export"
    trainforge_dir = _make_trainforge(export, "objplumb_ok")
    _make_synthesized(
        export / "01_learning_objectives" / "synthesized_objectives.json"
    )

    registry = _build_tool_registry()
    tool = registry["archive_to_libv2"]
    result = json.loads(asyncio.run(tool(
        course_name="OBJPLUMB_OK",
        project_workspace=str(trainforge_dir),
        pdf_paths="",
        html_paths="",
    )))
    assert result.get("success") is True, result
    assert result["objectives_status"] == "archived", result
    assert result.get("warnings") == [], result
    slug = "objplumb-ok"
    assert (
        tmp_path / "LibV2" / "courses" / slug / "objectives.json"
    ).exists()


# --------------------------------------------------------------------------- #
# Tier-2: real archive (dynamic discovery, skip if absent)
# --------------------------------------------------------------------------- #


def _discover_real_synthesized() -> Path | None:
    matches = sorted(glob.glob(str(
        PROJECT_ROOT / "Courseforge" / "exports" / "*"
        / "01_learning_objectives" / "synthesized_objectives.json"
    )))
    # Prefer the export with the most COs (closest to a real, full run).
    from lib.ontology.terminal_coverage import flatten_chapter_objectives
    best: Path | None = None
    best_n = -1
    for m in matches:
        try:
            d = json.loads(Path(m).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        n = len(flatten_chapter_objectives(
            d.get("chapter_objectives") or d.get("component_objectives") or []
        ))
        if n > best_n:
            best_n = n
            best = Path(m)
    return best if best_n > 0 else None


def test_real_export_projection_co_has_parent():
    """Project a REAL synthesized_objectives.json and assert every CO
    resolves to a real TO (the strict gate's co_has_parent contract)."""
    src = _discover_real_synthesized()
    if src is None:
        pytest.skip("no Courseforge export with synthesized objectives found")
    synth = json.loads(src.read_text(encoding="utf-8"))
    proj = project_objectives_for_archive(synth)
    cos = proj["component_objectives"]
    if not cos:
        pytest.skip(f"export {src} has no chapter objectives")
    terminal_ids = {(t.get("id") or "").lower() for t in proj["terminal_outcomes"]}
    missing = [c.get("id") for c in cos if not (c.get("parent_terminal") or "")]
    orphan = [
        c.get("id") for c in cos
        if (c.get("parent_terminal") or "").lower()
        and (c.get("parent_terminal") or "").lower() not in terminal_ids
    ]
    assert not missing, (
        f"{len(missing)}/{len(cos)} COs lack parent_terminal after "
        f"projection ({src}): {missing[:5]}"
    )
    assert not orphan, (
        f"{len(orphan)}/{len(cos)} COs point at a non-existent TO "
        f"({src}): {orphan[:5]}"
    )
