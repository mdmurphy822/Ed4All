"""Tests for ``gui.services.course_service`` — objectives read/normalize/write.

Uses the ``courseforge_export`` fixture (temp PROJ-* export with patched
COURSEFORGE_PATH) so the real ``Courseforge/exports/`` is never touched. No
fastapi needed.
"""

from __future__ import annotations

import json

import pytest

from gui.services import course_service


def test_get_objectives_normalizes_id_and_text(courseforge_export):
    doc = course_service.get_objectives(courseforge_export["project_id"])
    terminal = doc["terminal_objectives"]
    chapter = doc["chapter_objectives"]
    assert len(terminal) == 1 and len(chapter) == 2
    # statement mapped to text, id preserved.
    assert terminal[0]["id"] == "TO-01"
    assert terminal[0]["text"] == "Analyze kinematics."
    assert terminal[0]["statement"] == "Analyze kinematics."  # original preserved
    assert {c["id"] for c in chapter} == {"CO-01", "CO-02"}


def test_get_objectives_resolves_by_course_name(courseforge_export):
    doc = course_service.get_objectives(courseforge_export["course_name"])
    assert doc["course_name"] == "PHYS_101"


def test_get_objectives_unknown_course_raises(courseforge_export):
    with pytest.raises(FileNotFoundError):
        course_service.get_objectives("NO_SUCH_COURSE")


def test_save_objectives_round_trip(courseforge_export):
    new_doc = {
        "terminal_objectives": [
            {"id": "TO-01", "text": "Evaluate Newtonian mechanics."},
        ],
        "chapter_objectives": [
            {"id": "CO-01", "text": "State Newton's first law."},
            {"id": "CO-02", "text": "Apply Newton's second law.", "bloom_level": "apply"},
        ],
    }
    saved = course_service.save_objectives(courseforge_export["project_id"], new_doc)
    # mint_method stamped for --reuse-objectives compatibility.
    on_disk = json.loads(courseforge_export["objectives_path"].read_text())
    assert on_disk["mint_method"] == course_service.USER_SUPPLIED_MINT_METHOD
    # text -> statement on write; hierarchy_level stamped.
    to0 = on_disk["terminal_objectives"][0]
    assert to0["statement"] == "Evaluate Newtonian mechanics."
    assert to0["hierarchy_level"] == "terminal"
    assert "text" not in to0
    co1 = on_disk["chapter_objectives"][1]
    assert co1["bloom_level"] == "apply"  # extra field preserved
    assert co1["hierarchy_level"] == "chapter"
    # learning_outcomes mirror = terminal + chapter.
    assert len(on_disk["learning_outcomes"]) == 3
    # Returned doc is normalized back to id+text.
    assert saved["terminal_objectives"][0]["text"] == "Evaluate Newtonian mechanics."

    # Read-back through get_objectives reflects the new content.
    reread = course_service.get_objectives(courseforge_export["project_id"])
    assert reread["terminal_objectives"][0]["id"] == "TO-01"


def test_save_objectives_rejects_bad_lo_id(courseforge_export):
    bad = {
        "terminal_objectives": [{"id": "to_01_lowercase", "text": "bad id"}],
        "chapter_objectives": [],
    }
    with pytest.raises(ValueError) as exc:
        course_service.save_objectives(courseforge_export["project_id"], bad)
    assert "invalid LO id" in str(exc.value)
    # Fail closed: the on-disk doc must NOT have been overwritten by the bad write.
    on_disk = json.loads(courseforge_export["objectives_path"].read_text())
    assert on_disk["mint_method"] == "course_outliner_synthesis"


def test_save_objectives_rejects_bare_string_entry(courseforge_export):
    bad = {"terminal_objectives": ["just a string"], "chapter_objectives": []}
    with pytest.raises(ValueError) as exc:
        course_service.save_objectives(courseforge_export["project_id"], bad)
    assert "bare string" in str(exc.value) or "must be an object" in str(exc.value)


def test_save_objectives_no_export_raises(courseforge_export):
    with pytest.raises(FileNotFoundError):
        course_service.save_objectives("UNKNOWN_NAME", {"terminal_objectives": []})


def test_list_courses_includes_export(courseforge_export):
    courses = course_service.list_courses()
    names = {c["course_name"] for c in courses}
    assert "PHYS_101" in names
    phys = next(c for c in courses if c["course_name"] == "PHYS_101")
    assert phys["project_id"] == courseforge_export["project_id"]
    assert phys["terminal_count"] == 1
    assert phys["chapter_count"] == 2
    assert phys["duration_weeks"] == 10


# The GUI guesses a LibV2 archive dir from a course_name via
# ``course_service._slugify``. It MUST resolve the same directory the importer
# created, so it now delegates to the canonical helper. These pin parity with
# ``LibV2/tools/libv2/importer.py::slugify`` so the two never drift again.
@pytest.mark.parametrize(
    "course_name",
    [
        "PHYS_101",
        "The Great Course",
        "BIO 201: Cell Biology",
        "an apple",
        "Course (Advanced)!",
        "Introduction to Quantum Mechanics for Undergraduate Physics Majors",
    ],
)
def test_gui_slugify_equals_importer(course_name):
    from LibV2.tools.libv2.importer import slugify

    assert course_service._slugify(course_name) == slugify(course_name), (
        f"gui _slugify diverged from the importer on {course_name!r}"
    )
