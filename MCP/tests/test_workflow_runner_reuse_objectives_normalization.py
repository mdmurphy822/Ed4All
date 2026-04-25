"""Wave 80 Worker A — LibV2 → Courseforge form normalization.

Pins ``_normalize_to_courseforge_form``: when given a Wave 75 LibV2
archive form (``terminal_outcomes`` / ``component_objectives``),
the helper emits the Courseforge synthesized form
(``terminal_objectives`` / ``chapter_objectives``) so downstream
content-generator subagents — which only know how to read the
Courseforge form — can consume reused objectives transparently.

Also pins the disk-write contract: when
``_synthesize_course_planning_reuse_output`` is handed a LibV2-shape
file, the on-disk ``synthesized_objectives.json`` is written in the
Courseforge form (with ``terminal_objectives`` /
``chapter_objectives`` field names).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from MCP.core.workflow_runner import (
    WorkflowRunner,
    _coerce_chapter_groups,
    _normalize_to_courseforge_form,
)


@pytest.fixture
def runner_stub() -> WorkflowRunner:
    return WorkflowRunner(executor=object(), config=object())


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    project = tmp_path / "PROJ-NORM-20260424"
    (project / "01_learning_objectives").mkdir(parents=True)
    return project


# ---------------------------------------------------------------------
# Pure normalization
# ---------------------------------------------------------------------


class TestNormalizeToCourseforgeForm:
    def test_libv2_form_normalizes_to_courseforge_form(self):
        libv2 = {
            "schema_version": "v1",
            "course_code": "test_101",
            "terminal_outcomes": [
                {"id": "to-01", "statement": "T1",
                 "bloom_level": "understand"},
                {"id": "to-02", "statement": "T2"},
            ],
            "component_objectives": [
                {"id": "co-01", "statement": "C1",
                 "parent_terminal": "to-01",
                 "bloom_level": "remember"},
                {"id": "co-02", "statement": "C2",
                 "parent_terminal": "to-02"},
            ],
            "objective_count": {"terminal": 2, "component": 2},
        }
        out = _normalize_to_courseforge_form(libv2)
        assert out is not None

        # Courseforge field names.
        assert "terminal_objectives" in out
        assert "chapter_objectives" in out
        assert "terminal_outcomes" not in out
        assert "component_objectives" not in out

        # Terminal entries preserved with id/statement/bloom_level.
        terminal = out["terminal_objectives"]
        assert len(terminal) == 2
        assert {t["id"] for t in terminal} == {"to-01", "to-02"}
        assert terminal[0]["bloom_level"] == "understand"

        # Chapter groups: one group per CO (libv2 has no chapter
        # grouping). Each group's `objectives` carries the CO with
        # its parent_terminal back-pointer preserved.
        chapter_groups = out["chapter_objectives"]
        assert isinstance(chapter_groups, list)
        assert len(chapter_groups) == 2
        for group in chapter_groups:
            assert "chapter" in group
            assert "objectives" in group
            assert isinstance(group["objectives"], list)
            assert len(group["objectives"]) == 1

        # parent_terminal back-pointer survived the normalization.
        flat_cos = [
            o for g in chapter_groups for o in g["objectives"]
        ]
        assert {c["id"] for c in flat_cos} == {"co-01", "co-02"}
        assert {c.get("parent_terminal") for c in flat_cos} == {
            "to-01", "to-02",
        }

    def test_courseforge_form_passthrough(self):
        cf = {
            "course_name": "X",
            "duration_weeks": 8,
            "terminal_objectives": [{"id": "TO-01", "statement": "T1"}],
            "chapter_objectives": [
                {"chapter": "Week 1", "objectives": [
                    {"id": "CO-01", "statement": "C1"},
                ]},
            ],
        }
        out = _normalize_to_courseforge_form(cf)
        assert out is not None
        assert out["terminal_objectives"] == [
            {"id": "TO-01", "statement": "T1"}
        ]
        assert out["chapter_objectives"][0]["objectives"][0]["id"] == "CO-01"
        assert out["course_name"] == "X"
        assert out["duration_weeks"] == 8

    def test_courseforge_flat_chapters_become_groups(self):
        cf = {
            "terminal_objectives": [{"id": "TO-01", "statement": "T1"}],
            # Flat shape — should be coerced into group shape with
            # one CO per group.
            "chapter_objectives": [
                {"id": "CO-01", "statement": "C1"},
                {"id": "CO-02", "statement": "C2"},
            ],
        }
        out = _normalize_to_courseforge_form(cf)
        assert out is not None
        groups = out["chapter_objectives"]
        assert len(groups) == 2
        for g in groups:
            assert isinstance(g.get("objectives"), list)
            assert len(g["objectives"]) == 1

    def test_unrecognised_shape_returns_none(self):
        out = _normalize_to_courseforge_form({"foo": "bar"})
        assert out is None

    def test_libv2_passes_course_code_into_course_name_slot(self):
        libv2 = {
            "schema_version": "v1",
            "course_code": "phys_101",
            "terminal_outcomes": [{"id": "to-01", "statement": "T1"}],
            "component_objectives": [],
            "objective_count": {"terminal": 1, "component": 0},
        }
        out = _normalize_to_courseforge_form(libv2)
        assert out is not None
        # Courseforge's synthesized JSON uses ``course_name`` as the
        # canonical key — LibV2's ``course_code`` is the closest
        # equivalent.
        assert out["course_name"] == "phys_101"


class TestCoerceChapterGroups:
    def test_group_shape_passthrough(self):
        groups = _coerce_chapter_groups([
            {"chapter": "Week 1", "objectives": [{"id": "CO-01"}]},
        ])
        assert groups == [
            {"chapter": "Week 1", "objectives": [{"id": "CO-01"}]},
        ]

    def test_flat_shape_coerced(self):
        groups = _coerce_chapter_groups([
            {"id": "CO-01"},
            {"id": "CO-02"},
        ])
        assert len(groups) == 2
        for g in groups:
            assert "chapter" in g
            assert g["objectives"][0]["id"].startswith("CO-")


# ---------------------------------------------------------------------
# End-to-end disk-write contract
# ---------------------------------------------------------------------


class TestLibV2ReuseDiskWriteContract:
    """When ``--reuse-objectives`` is handed a LibV2-shape file, the
    on-disk ``synthesized_objectives.json`` must be written in the
    Courseforge form so downstream content-generator subagents can
    consume it without reading Wave 75 archive shapes.
    """

    def test_libv2_input_writes_courseforge_form_to_disk(
        self, runner_stub, project_dir, tmp_path
    ):
        libv2 = tmp_path / "objectives.json"
        libv2.write_text(
            json.dumps({
                "schema_version": "v1",
                "course_code": "test_101",
                "terminal_outcomes": [
                    {"id": "to-01", "statement": "T1",
                     "bloom_level": "understand"},
                ],
                "component_objectives": [
                    {"id": "co-01", "statement": "C1",
                     "parent_terminal": "to-01"},
                ],
                "objective_count": {"terminal": 1, "component": 1},
            }),
            encoding="utf-8",
        )

        params = {
            "reuse_objectives_path": str(libv2),
            "course_name": "TEST_101",
        }
        phase_outputs = {
            "objective_extraction": {
                "project_id": project_dir.name,
                "project_path": str(project_dir),
                "_completed": True,
            },
        }
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, phase_outputs,
        )
        assert out is not None

        # Read what was written.
        written = json.loads(
            Path(out["synthesized_objectives_path"]).read_text(
                encoding="utf-8"
            )
        )

        # Courseforge field names appear; LibV2 ones do not.
        assert "terminal_objectives" in written
        assert "chapter_objectives" in written
        assert "terminal_outcomes" not in written
        assert "component_objectives" not in written

        # Terminal entry preserved.
        assert written["terminal_objectives"][0]["id"] == "to-01"
        assert written["terminal_objectives"][0]["bloom_level"] == "understand"

        # Chapter group carrying the CO with parent_terminal preserved.
        groups = written["chapter_objectives"]
        assert isinstance(groups, list)
        assert len(groups) == 1
        co = groups[0]["objectives"][0]
        assert co["id"] == "co-01"
        assert co["parent_terminal"] == "to-01"

        # mint_method records the reuse path.
        assert written["mint_method"] == "reuse_objectives"
        assert written["generated_from"] == str(libv2)
