"""W0.5 integration: DecisionCapture honors ED4ALL_COURSE_IDENTITY_DEDUP."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.decision_capture import DecisionCapture


@pytest.fixture
def libv2_root(tmp_path, monkeypatch):
    root = tmp_path / "libv2"
    (root / "courses").mkdir(parents=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(root))
    monkeypatch.setenv("ED4ALL_TRAINING_CAPTURES_DIR", str(tmp_path / "captures"))
    return root


def _populate(root: Path, slug: str):
    d = root / "courses" / slug / "dart_chunks"
    d.mkdir(parents=True)
    (d / "chunks.jsonl").write_text('{"id": "x_chunk_00000"}\n')


@pytest.mark.integration
def test_flag_off_is_legacy_path(libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_COURSE_IDENTITY_DEDUP", raising=False)
    cap = DecisionCapture("demo-course", "phase-x", tool="trainforge")
    try:
        # Legacy: storage course_id is the verbatim course_code.
        assert cap._storage.course_id == "demo-course"
    finally:
        cap.close()


@pytest.mark.integration
def test_flag_on_consolidates_and_cleans_empty_twin(libv2_root, monkeypatch):
    from lib.decision_capture import normalize_course_code
    from lib.ontology.slugs import libv2_course_slug

    monkeypatch.setenv("ED4ALL_COURSE_IDENTITY_DEDUP", "true")
    # Populated verbatim-name course + an empty hashed twin already on disk.
    # The hashed twin slug is the libv2 slug of normalize_course_code("Ed4All").
    _populate(libv2_root, "ed4all")
    twin_slug = libv2_course_slug(normalize_course_code("Ed4All"))
    twin = libv2_root / "courses" / twin_slug
    (twin / "imscc_chunks").mkdir(parents=True)

    cap = DecisionCapture("Ed4All", "phase-x", tool="trainforge")
    try:
        # Storage consolidated onto the canonical populated slug.
        assert cap._storage.course_slug == "ed4all"
        # The empty hashed twin was cleaned up (strict gate satisfied).
        assert not twin.exists()
        # The populated canonical course is untouched.
        assert (libv2_root / "courses" / "ed4all" / "dart_chunks" / "chunks.jsonl").exists()
    finally:
        cap.close()


@pytest.mark.integration
def test_flag_on_does_not_delete_legit_empty_inprogress(libv2_root, monkeypatch):
    monkeypatch.setenv("ED4ALL_COURSE_IDENTITY_DEDUP", "true")
    # No populated canonical → nothing must be deleted.
    cap = DecisionCapture("Ed4All", "phase-x", tool="trainforge")
    try:
        # Resolver runs without error; capture still usable.
        cap.log_decision(
            decision_type="content_selection",
            decision="noop",
            rationale="exercise the capture path under dedup with no twin.",
        )
    finally:
        cap.close()
