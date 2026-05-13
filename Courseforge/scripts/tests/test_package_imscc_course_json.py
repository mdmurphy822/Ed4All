"""Wave2-I3 regression tests for the packaging-shaped ``course.json``
emission from ``MCP.tools.pipeline_tools._package_imscc``.

Source: Finding 3 of
``plans/dispatch-7-execution-inspection-2026-05.md``. Closes the
``PAGE_OBJECTIVES_PATH_MISSING`` blocker by emitting
``<project>/03_content_development/course.json`` from
``<project>/01_learning_objectives/synthesized_objectives.json`` BEFORE
the mature packager + the ``PageObjectivesValidator`` gate fire.

Test surface:
    1. Fresh emit: a project with synthesized_objectives.json and NO
       course.json gets course.json emitted with the required
       ``terminal_objectives`` + ``chapter_objectives`` keys that
       ``PageObjectivesValidator`` -> ``load_canonical_objectives``
       consumes (Courseforge/scripts/generate_course.py:615-646), plus
       the LibV2-shaped ``learning_outcomes[]`` for downstream
       Trainforge parity.
    2. Idempotent skip: a pre-existing course.json at the target path
       is NOT overwritten.
    3. LibV2 archive form: synthesized_objectives carrying
       ``terminal_outcomes`` + ``component_objectives`` (the alternative
       shape documented in CLAUDE.md ``--reuse-objectives``) is
       handled defensively.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Resolve the repo root so we can import MCP.tools.pipeline_tools.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Courseforge/scripts/ also needs to be on sys.path so the lazy import
# of validate_page_objectives + generate_course in the mature packager
# resolves cleanly when these tests run alongside the rest of the suite.
_CF_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_CF_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CF_SCRIPTS))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _project_synthesized_objectives_to_course_json,
)


# ---------------------------------------------------------------------------
# Canonical synthesized_objectives.json shapes
# ---------------------------------------------------------------------------

# 2 TOs + 4 COs spread across 2 chapter groups — the shape
# MCP/tools/pipeline_tools.py::_plan_course_structure writes to disk.
_SYNTHESIZED_COURSEFORGE_FORM = {
    "course_name": "PHYS_101",
    "generated_from": "textbook_structure.json",
    "mint_method": "synthesize_objectives_from_topics",
    "duration_weeks": 8,
    "learning_outcomes": [
        {
            "id": "TO-01",
            "statement": "Apply Newtonian mechanics to projectile motion.",
            "bloom_level": "apply",
            "hierarchy_level": "terminal",
            "cognitive_domain": "procedural",
        },
        {
            "id": "TO-02",
            "statement": "Analyze energy conservation in mechanical systems.",
            "bloom_level": "analyze",
            "hierarchy_level": "terminal",
            "cognitive_domain": "conceptual",
        },
        {
            "id": "CO-01",
            "statement": "Identify the components of a force diagram.",
            "bloom_level": "remember",
            "hierarchy_level": "chapter",
        },
        {
            "id": "CO-02",
            "statement": "Explain Newton's three laws of motion.",
            "bloom_level": "understand",
            "hierarchy_level": "chapter",
        },
        {
            "id": "CO-03",
            "statement": "Compute kinetic and potential energy in a system.",
            "bloom_level": "apply",
            "hierarchy_level": "chapter",
        },
        {
            "id": "CO-04",
            "statement": "Differentiate elastic from inelastic collisions.",
            "bloom_level": "analyze",
            "hierarchy_level": "chapter",
        },
    ],
    "terminal_objectives": [
        {
            "id": "TO-01",
            "statement": "Apply Newtonian mechanics to projectile motion.",
            "bloom_level": "apply",
        },
        {
            "id": "TO-02",
            "statement": "Analyze energy conservation in mechanical systems.",
            "bloom_level": "analyze",
        },
    ],
    "chapter_objectives": [
        {
            "chapter": "Week 1-2: Foundations of Motion",
            "objectives": [
                {
                    "id": "CO-01",
                    "statement": "Identify the components of a force diagram.",
                    "bloom_level": "remember",
                },
                {
                    "id": "CO-02",
                    "statement": "Explain Newton's three laws of motion.",
                    "bloom_level": "understand",
                },
            ],
        },
        {
            "chapter": "Week 3-4: Energy and Collisions",
            "objectives": [
                {
                    "id": "CO-03",
                    "statement": "Compute kinetic and potential energy in a system.",
                    "bloom_level": "apply",
                },
                {
                    "id": "CO-04",
                    "statement": "Differentiate elastic from inelastic collisions.",
                    "bloom_level": "analyze",
                },
            ],
        },
    ],
    "synthesized_at": "2026-05-12T12:00:00",
}


# OpenStax-shaped form (Wave2b — chapter_objectives is a dict of
# chapter-label -> flat list of CO dicts). The Wave 2 smoke verification
# (plans/wave2-smoke-verification-2026-05.md "Surprises") found this
# shape on real OpenStax-derived corpora; before Wave2b both helpers
# silently dropped every CO-NN entry because they only handled the
# list-of-groups + flat-list shapes. Mirrors the OpenStax dict-of-lists
# layout — chapter labels as keys, flat CO dict lists as values.
_SYNTHESIZED_OPENSTAX_FORM = {
    "course_name": "PHYS_101",
    "duration_weeks": 8,
    "terminal_objectives": [
        {
            "id": "TO-01",
            "statement": "Apply Newtonian mechanics to projectile motion.",
            "bloom_level": "apply",
        },
        {
            "id": "TO-02",
            "statement": "Analyze energy conservation in mechanical systems.",
            "bloom_level": "analyze",
        },
    ],
    "chapter_objectives": {
        "1": [
            {
                "id": "CO-01",
                "statement": "Identify the components of a force diagram.",
                "bloom_level": "remember",
            },
            {
                "id": "CO-02",
                "statement": "Explain Newton's three laws of motion.",
                "bloom_level": "understand",
            },
        ],
        "2": [
            {
                "id": "CO-03",
                "statement": "Compute kinetic and potential energy in a system.",
                "bloom_level": "apply",
            },
            {
                "id": "CO-04",
                "statement": "Differentiate elastic from inelastic collisions.",
                "bloom_level": "analyze",
            },
        ],
    },
}


# LibV2 archive form (terminal_outcomes + component_objectives). Per the
# root CLAUDE.md ``--reuse-objectives`` doc the runner normally
# normalizes this to the Courseforge form on disk, but we still must
# handle it defensively in case a hand-edited file slips through.
_SYNTHESIZED_LIBV2_FORM = {
    "course_name": "PHYS_101",
    "duration_weeks": 8,
    "terminal_outcomes": [
        {
            "id": "TO-01",
            "statement": "Apply Newtonian mechanics to projectile motion.",
            "bloom_level": "apply",
        },
        {
            "id": "TO-02",
            "statement": "Analyze energy conservation in mechanical systems.",
            "bloom_level": "analyze",
        },
    ],
    "component_objectives": [
        {
            "chapter": "Week 1-2: Foundations of Motion",
            "objectives": [
                {
                    "id": "CO-01",
                    "statement": "Identify the components of a force diagram.",
                    "bloom_level": "remember",
                },
                {
                    "id": "CO-02",
                    "statement": "Explain Newton's three laws of motion.",
                    "bloom_level": "understand",
                },
            ],
        },
        {
            "chapter": "Week 3-4: Energy and Collisions",
            "objectives": [
                {
                    "id": "CO-03",
                    "statement": "Compute kinetic and potential energy in a system.",
                    "bloom_level": "apply",
                },
                {
                    "id": "CO-04",
                    "statement": "Differentiate elastic from inelastic collisions.",
                    "bloom_level": "analyze",
                },
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_project(
    tmp_path: Path,
    *,
    synthesized: dict | None = None,
    pre_existing_course_json: dict | None = None,
) -> tuple[Path, Path, Path]:
    """Build a Courseforge exports project layout.

    Returns ``(project_path, synthesized_objectives_path,
    content_course_json_path)``.
    """
    project_path = tmp_path / "Courseforge" / "exports" / "PROJ-W2I3-01"
    lo_dir = project_path / "01_learning_objectives"
    content_dir = project_path / "03_content_development"
    lo_dir.mkdir(parents=True, exist_ok=True)
    content_dir.mkdir(parents=True, exist_ok=True)

    synth_path = lo_dir / "synthesized_objectives.json"
    if synthesized is not None:
        synth_path.write_text(
            json.dumps(synthesized, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    course_json_path = content_dir / "course.json"
    if pre_existing_course_json is not None:
        course_json_path.write_text(
            json.dumps(pre_existing_course_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return project_path, synth_path, course_json_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCourseJsonEmissionFromSynthesizedObjectives:
    """Wave2-I3: closes PAGE_OBJECTIVES_PATH_MISSING blocker."""

    def test_fresh_emit_writes_packaging_shaped_course_json(self, tmp_path):
        """Project with synthesized_objectives + no course.json => course.json
        emitted carrying terminal_objectives + chapter_objectives + the
        LibV2-shaped learning_outcomes[] flat list.

        Verifies the PageObjectivesValidator contract: the auto-discovery
        at lib/validators/page_objectives.py:192-195 finds the file and
        load_canonical_objectives reads terminal_objectives +
        chapter_objectives at Courseforge/scripts/generate_course.py:629-639.
        """
        _, synth_path, course_json_path = _build_project(
            tmp_path,
            synthesized=_SYNTHESIZED_COURSEFORGE_FORM,
            pre_existing_course_json=None,
        )
        assert not course_json_path.exists()  # pre-condition

        result = _project_synthesized_objectives_to_course_json(
            synth_path,
            course_json_path,
            course_code="PHYS_101",
            course_title="Physics 101: Mechanics",
        )

        # Returned dict + on-disk file agree.
        assert result is not None
        assert course_json_path.exists()
        on_disk = json.loads(course_json_path.read_text(encoding="utf-8"))
        assert on_disk == result

        # PageObjectivesValidator / load_canonical_objectives contract
        # (Courseforge/scripts/generate_course.py:629 reads
        # terminal_objectives, :632 reads chapter_objectives.get("chapter")
        # + :639 reads chapter.get("objectives")).
        assert "terminal_objectives" in on_disk
        assert isinstance(on_disk["terminal_objectives"], list)
        assert len(on_disk["terminal_objectives"]) == 2
        assert {to["id"] for to in on_disk["terminal_objectives"]} == {
            "TO-01", "TO-02",
        }

        assert "chapter_objectives" in on_disk
        assert isinstance(on_disk["chapter_objectives"], list)
        assert len(on_disk["chapter_objectives"]) == 2
        # Every chapter group carries the "chapter" + "objectives" keys
        # load_canonical_objectives indexes (line :633 reads "chapter",
        # :639 reads "objectives").
        for grp in on_disk["chapter_objectives"]:
            assert "chapter" in grp
            assert "objectives" in grp
            assert isinstance(grp["objectives"], list)
        # 4 COs spread across 2 groups.
        all_co_ids = {
            co["id"]
            for grp in on_disk["chapter_objectives"]
            for co in grp["objectives"]
        }
        assert all_co_ids == {"CO-01", "CO-02", "CO-03", "CO-04"}

        # LibV2 course.json shape (schemas/knowledge/course.schema.json
        # :32-87) — required fields + flat learning_outcomes[].
        assert on_disk["course_code"] == "PHYS_101"
        assert on_disk["title"] == "Physics 101: Mechanics"
        assert isinstance(on_disk["learning_outcomes"], list)
        # The 2 TOs + 4 COs synthesized list.
        assert len(on_disk["learning_outcomes"]) == 6
        lo_ids = {lo["id"] for lo in on_disk["learning_outcomes"]}
        assert lo_ids == {
            "TO-01", "TO-02", "CO-01", "CO-02", "CO-03", "CO-04",
        }
        # Confirm the inspector-report fields are present per LO.
        sample = next(
            lo for lo in on_disk["learning_outcomes"] if lo["id"] == "TO-01"
        )
        assert sample["statement"]
        assert sample["bloom_level"] == "apply"
        assert sample["hierarchy_level"] == "terminal"

        # chunker_version + duration_weeks for downstream Trainforge.
        assert "chunker_version" in on_disk
        assert on_disk["duration_weeks"] == 8

    def test_pre_existing_course_json_is_not_overwritten(self, tmp_path):
        """Idempotent skip: course.json already on disk => no overwrite.

        Mirrors the --reuse-objectives operator path (root CLAUDE.md
        Quick Start) where an external course.json is staged before
        packaging fires.
        """
        sentinel = {
            "course_code": "REUSED_101",
            "title": "Hand-curated course",
            "learning_outcomes": [
                {
                    "id": "TO-99",
                    "statement": "Sentinel objective preserved verbatim.",
                    "hierarchy_level": "terminal",
                },
            ],
            "terminal_objectives": [],
            "chapter_objectives": [],
            "_sentinel_field_only_in_original": True,
        }
        _, synth_path, course_json_path = _build_project(
            tmp_path,
            synthesized=_SYNTHESIZED_COURSEFORGE_FORM,
            pre_existing_course_json=sentinel,
        )
        assert course_json_path.exists()  # pre-condition

        result = _project_synthesized_objectives_to_course_json(
            synth_path,
            course_json_path,
            course_code="PHYS_101",
            course_title="Physics 101: Mechanics",
        )

        # Return value is None on the idempotent-skip path; on-disk
        # content is byte-identical to the sentinel.
        assert result is None
        on_disk = json.loads(course_json_path.read_text(encoding="utf-8"))
        assert on_disk == sentinel
        assert on_disk.get("_sentinel_field_only_in_original") is True
        assert on_disk["course_code"] == "REUSED_101"

    def test_flat_list_chapter_objectives_is_handled(self, tmp_path):
        """Wave2b regression: chapter_objectives as a flat list of CO
        dicts (LibV2 archive's un-grouped variant) is normalized into a
        single synthetic group so every CO-NN id survives.

        The normalizer wraps the flat list under a synthetic
        ``chapter="(ungrouped)"`` group; ``load_canonical_objectives``
        still walks the group's ``objectives`` array exactly as it
        would for the canonical list-of-groups form.
        """
        flat_form = {
            "course_name": "PHYS_101",
            "duration_weeks": 8,
            "terminal_objectives": [
                {"id": "TO-01", "statement": "T1.", "bloom_level": "apply"},
                {"id": "TO-02", "statement": "T2.", "bloom_level": "analyze"},
            ],
            "chapter_objectives": [
                {"id": "CO-01", "statement": "C1.", "bloom_level": "remember"},
                {"id": "CO-02", "statement": "C2.", "bloom_level": "understand"},
                {"id": "CO-03", "statement": "C3.", "bloom_level": "apply"},
                {"id": "CO-04", "statement": "C4.", "bloom_level": "analyze"},
            ],
        }
        _, synth_path, course_json_path = _build_project(
            tmp_path,
            synthesized=flat_form,
        )
        result = _project_synthesized_objectives_to_course_json(
            synth_path,
            course_json_path,
            course_code="PHYS_101",
            course_title="Physics 101 flat-list",
        )
        assert result is not None
        on_disk = json.loads(course_json_path.read_text(encoding="utf-8"))

        # All 4 CO-NN ids preserved through the synthetic-group wrap.
        assert isinstance(on_disk["chapter_objectives"], list)
        assert len(on_disk["chapter_objectives"]) >= 1
        all_co_ids = {
            co["id"]
            for grp in on_disk["chapter_objectives"]
            for co in grp.get("objectives") or []
        }
        assert all_co_ids == {"CO-01", "CO-02", "CO-03", "CO-04"}
        # learning_outcomes[] reconstructed from terminal + chapter.
        assert {lo["id"] for lo in on_disk["learning_outcomes"]} == {
            "TO-01", "TO-02", "CO-01", "CO-02", "CO-03", "CO-04",
        }

    def test_openstax_dict_of_lists_shape_is_handled(self, tmp_path):
        """Wave2b regression: chapter_objectives as a dict-of-lists
        (OpenStax shape) must NOT silently drop every CO-NN entry.

        Pre-Wave2b the projection only handled list-of-groups +
        flat-list shapes; a dict-shaped chapter_objectives was passed
        through as-is, breaking
        ``load_canonical_objectives``' downstream walk (which iterates
        ``chapter_objectives`` expecting list-of-groups) and silently
        emitting an empty CO set. Closes
        ``plans/wave2-smoke-verification-2026-05.md`` "Surprises".
        """
        _, synth_path, course_json_path = _build_project(
            tmp_path,
            synthesized=_SYNTHESIZED_OPENSTAX_FORM,
        )
        result = _project_synthesized_objectives_to_course_json(
            synth_path,
            course_json_path,
            course_code="PHYS_101",
            course_title="Physics 101 OpenStax-form",
        )
        assert result is not None
        on_disk = json.loads(course_json_path.read_text(encoding="utf-8"))

        # Terminal objectives untouched.
        assert len(on_disk["terminal_objectives"]) == 2
        assert {to["id"] for to in on_disk["terminal_objectives"]} == {
            "TO-01", "TO-02",
        }

        # chapter_objectives is normalized to the canonical
        # list-of-groups shape (the projection writes that shape
        # to disk, NOT the dict-shape from the input). 2 groups
        # corresponding to the 2 chapter keys, with sorted-by-key
        # iteration order ("1" before "2").
        assert isinstance(on_disk["chapter_objectives"], list)
        assert len(on_disk["chapter_objectives"]) == 2
        group_chapters = [grp["chapter"] for grp in on_disk["chapter_objectives"]]
        assert group_chapters == ["1", "2"], (
            f"expected deterministic sorted chapter ordering; got {group_chapters!r}"
        )
        # All 4 CO-NN IDs preserved (the silent-drop regression).
        all_co_ids = {
            co["id"]
            for grp in on_disk["chapter_objectives"]
            for co in grp["objectives"]
        }
        assert all_co_ids == {"CO-01", "CO-02", "CO-03", "CO-04"}

        # learning_outcomes[] reconstructed from terminal + chapter
        # since the input doesn't pre-emit it.
        assert isinstance(on_disk["learning_outcomes"], list)
        assert len(on_disk["learning_outcomes"]) == 6
        lo_ids = {lo["id"] for lo in on_disk["learning_outcomes"]}
        assert lo_ids == {
            "TO-01", "TO-02", "CO-01", "CO-02", "CO-03", "CO-04",
        }

    def test_libv2_archive_form_is_handled_defensively(self, tmp_path):
        """terminal_outcomes + component_objectives shape (LibV2 archive
        form) is handled even though the runner normally normalizes it.

        Per root CLAUDE.md --reuse-objectives doc the runner converts to
        the Courseforge form on disk, but the projection accepts both so
        a hand-edited / pre-normalization file still produces a working
        course.json (vs silently emitting an empty one).
        """
        _, synth_path, course_json_path = _build_project(
            tmp_path,
            synthesized=_SYNTHESIZED_LIBV2_FORM,
        )

        result = _project_synthesized_objectives_to_course_json(
            synth_path,
            course_json_path,
            course_code="PHYS_101",
            course_title="Physics 101 LibV2-form",
        )
        assert result is not None
        on_disk = json.loads(course_json_path.read_text(encoding="utf-8"))

        # Terminal objectives carried over from terminal_outcomes.
        assert len(on_disk["terminal_objectives"]) == 2
        assert {to["id"] for to in on_disk["terminal_objectives"]} == {
            "TO-01", "TO-02",
        }
        # Chapter groups carried over from component_objectives.
        assert len(on_disk["chapter_objectives"]) == 2
        all_co_ids = {
            co["id"]
            for grp in on_disk["chapter_objectives"]
            for co in grp["objectives"]
        }
        assert all_co_ids == {"CO-01", "CO-02", "CO-03", "CO-04"}
        # learning_outcomes[] reconstructed when not pre-emitted.
        assert len(on_disk["learning_outcomes"]) == 6
        # Hierarchy levels stamped during reconstruction.
        terminal_los = [
            lo for lo in on_disk["learning_outcomes"]
            if lo.get("hierarchy_level") == "terminal"
        ]
        chapter_los = [
            lo for lo in on_disk["learning_outcomes"]
            if lo.get("hierarchy_level") == "chapter"
        ]
        assert len(terminal_los) == 2
        assert len(chapter_los) == 4


class TestProjectionEdgeCases:
    """Defensive paths that mustn't break the packaging gate."""

    def test_missing_synthesized_objectives_returns_none(self, tmp_path):
        """No upstream synthesized_objectives.json => projection no-ops,
        downstream PageObjectivesValidator fails closed as expected.
        """
        _, synth_path, course_json_path = _build_project(
            tmp_path, synthesized=None,
        )
        assert not synth_path.exists()  # pre-condition

        result = _project_synthesized_objectives_to_course_json(
            synth_path,
            course_json_path,
            course_code="X_101",
            course_title="X",
        )
        assert result is None
        assert not course_json_path.exists()

    def test_malformed_synthesized_objectives_does_not_raise(self, tmp_path):
        """Invalid JSON in the synthesized file => projection no-ops without
        raising so the packaging stage can still surface a structured
        error (the validator gate then fails closed with the canonical
        PAGE_OBJECTIVES_PATH_MISSING code).
        """
        _, synth_path, course_json_path = _build_project(
            tmp_path, synthesized=None,
        )
        synth_path.parent.mkdir(parents=True, exist_ok=True)
        synth_path.write_text("{not valid json", encoding="utf-8")

        result = _project_synthesized_objectives_to_course_json(
            synth_path,
            course_json_path,
            course_code="X_101",
            course_title="X",
        )
        assert result is None
        assert not course_json_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
