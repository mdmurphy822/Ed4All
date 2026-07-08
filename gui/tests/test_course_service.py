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


def test_save_objectives_preserves_citations_and_nested_extras(courseforge_export):
    """GET-shaped objects with grounding metadata must round-trip byte-faithfully.

    Mirrors the SPA contract (Q3/I2): the editor keeps the full objects it got
    from GET and PUTs them back with only ``id`` + ``text`` rebound. The
    citation/chunk/grounding/bloom + nested sub-objective fields must survive to
    disk unchanged so an edited objectives doc does not regress the
    objective-entailment / grounding gates on the next build. The only permitted
    deltas are the edited statement, the ``text``→``statement`` remap, the
    stamped ``hierarchy_level``, and the doc-level ``mint_method`` /
    ``synthesized_at`` restamps.
    """
    # As the frontend now sends: the FULL object (extras verbatim) with the
    # editor's `text`, and NO server-derived stale `statement`.
    citations = [
        {"chunk_id": "dart:phys-ch1#b07", "page": 12, "cosine": 0.83},
        {"chunk_id": "dart:phys-ch1#b09", "page": 13, "cosine": 0.71},
    ]
    edited = {
        "terminal_objectives": [
            {
                "id": "TO-01",
                "text": "Analyze kinematics in one dimension.",  # edited text
                "bloom_level": "analyze",
                "cognitive_domain": "cognitive",
                "citations": citations,
                "chunk_ids": ["dart:phys-ch1#b07", "dart:phys-ch1#b09"],
                "sub_objectives": [
                    {"id": "TO-01.1", "statement": "Define displacement.", "chunk_ids": ["dart:phys-ch1#b07"]},
                ],
            },
        ],
        "chapter_objectives": [
            {
                "id": "CO-01",
                "text": "Compute velocity.",  # unchanged text
                "bloom_level": "apply",
                "citations": [{"chunk_id": "dart:phys-ch1#b12", "page": 15}],
                "source_refs": {"module": "M1", "confidence": 0.9},
            },
        ],
    }
    course_service.save_objectives(courseforge_export["project_id"], edited)
    on_disk = json.loads(courseforge_export["objectives_path"].read_text())

    to0 = on_disk["terminal_objectives"][0]
    # Edited text became the statement; text key dropped; hierarchy stamped.
    assert to0["statement"] == "Analyze kinematics in one dimension."
    assert "text" not in to0
    assert to0["hierarchy_level"] == "terminal"
    # Every grounding/citation extra survived byte-faithfully.
    assert to0["bloom_level"] == "analyze"
    assert to0["cognitive_domain"] == "cognitive"
    assert to0["citations"] == citations
    assert to0["chunk_ids"] == ["dart:phys-ch1#b07", "dart:phys-ch1#b09"]
    # Nested sub-objectives preserved verbatim (not flattened / re-shaped).
    assert to0["sub_objectives"] == [
        {"id": "TO-01.1", "statement": "Define displacement.", "chunk_ids": ["dart:phys-ch1#b07"]},
    ]

    co0 = on_disk["chapter_objectives"][0]
    assert co0["statement"] == "Compute velocity."
    assert co0["hierarchy_level"] == "chapter"
    assert co0["citations"] == [{"chunk_id": "dart:phys-ch1#b12", "page": 15}]
    assert co0["source_refs"] == {"module": "M1", "confidence": 0.9}

    # Only the doc-level restamps changed relative to a reuse-compatible doc.
    assert on_disk["mint_method"] == course_service.USER_SUPPLIED_MINT_METHOD
    assert "synthesized_at" in on_disk


def test_save_objectives_edited_text_wins_over_stale_statement(courseforge_export):
    """If a caller sends BOTH `text` and a stale `statement`, `statement` wins.

    This pins the service contract the SPA now relies on: because
    ``_materialize_los`` prefers a present non-empty ``statement`` over ``text``,
    the frontend must drop the server-derived ``statement`` for an edit to take
    effect. This test documents that a client which fails to do so would persist
    the stale statement — the guard rail behind the app.js `statement`-strip.
    """
    payload = {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "OLD statement.", "text": "NEW edited text."},
        ],
        "chapter_objectives": [],
    }
    course_service.save_objectives(courseforge_export["project_id"], payload)
    on_disk = json.loads(courseforge_export["objectives_path"].read_text())
    # statement present → wins; this is exactly why the SPA strips it before PUT.
    assert on_disk["terminal_objectives"][0]["statement"] == "OLD statement."


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
