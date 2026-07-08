"""Tests for the I5 human-review attestation block (course_service + router).

Uses the ``libv2_course`` fixture (temp LibV2 manifest under an
``ED4ALL_LIBV2_ROOT`` redirect) so the real ``LibV2/`` tree is never touched.
Covers the service layer (set / server-restamp / invalid-scope rejection /
unknown-course) plus the router surface (PATCH → JSON, 422, 404).
"""

from __future__ import annotations

import json

import pytest

from gui.services import course_service


# --------------------------------------------------------------------------- #
# Service layer
# --------------------------------------------------------------------------- #


def test_save_attestation_sets_block(libv2_course):
    slug = libv2_course["slug"]
    saved = course_service.save_attestation(
        slug, {"reviewed_by": "Dr. Ada", "scope": "full", "note": "spot-checked"}
    )
    assert saved["reviewed_by"] == "Dr. Ada"
    assert saved["scope"] == "full"
    assert saved["note"] == "spot-checked"
    # reviewed_at server-stamped (caller omitted it).
    assert saved["reviewed_at"]

    on_disk = json.loads((libv2_course["course_dir"] / "manifest.json").read_text())
    assert on_disk["attestation"]["reviewed_by"] == "Dr. Ada"
    # The rest of the manifest is preserved (classification untouched).
    assert on_disk["classification"]["primary_domain"] == "physics"

    # Read-back through the getter.
    got = course_service.get_attestation(slug)
    assert got["scope"] == "full"


def test_get_attestation_empty_when_unattested(libv2_course):
    assert course_service.get_attestation(libv2_course["slug"]) == {}


def test_save_attestation_server_restamps_when_omitted(libv2_course):
    """An omitted reviewed_at is server-stamped; a supplied one is honored."""
    slug = libv2_course["slug"]
    supplied = "2020-01-02T03:04:05+00:00"
    first = course_service.save_attestation(
        slug, {"reviewed_by": "R1", "scope": "objectives", "reviewed_at": supplied}
    )
    assert first["reviewed_at"] == supplied

    # A fresh sign-off with no reviewed_at supersedes and gets a new stamp.
    second = course_service.save_attestation(
        slug, {"reviewed_by": "R2", "scope": "content"}
    )
    assert second["reviewed_at"] and second["reviewed_at"] != supplied
    assert second["reviewed_by"] == "R2"
    # Full replace: the prior reviewer is gone (not merged).
    on_disk = json.loads((libv2_course["course_dir"] / "manifest.json").read_text())
    assert on_disk["attestation"]["reviewed_by"] == "R2"


def test_save_attestation_rejects_invalid_scope(libv2_course):
    slug = libv2_course["slug"]
    with pytest.raises(ValueError) as exc:
        course_service.save_attestation(
            slug, {"reviewed_by": "R", "scope": "everything"}
        )
    assert "attestation" in str(exc.value)
    # Fail closed: nothing written.
    on_disk = json.loads((libv2_course["course_dir"] / "manifest.json").read_text())
    assert "attestation" not in on_disk


def test_save_attestation_rejects_empty_reviewer(libv2_course):
    with pytest.raises(ValueError):
        course_service.save_attestation(
            libv2_course["slug"], {"reviewed_by": "", "scope": "full"}
        )


def test_save_attestation_unknown_course_raises(libv2_root):
    with pytest.raises(FileNotFoundError):
        course_service.save_attestation("no-such-course", {"reviewed_by": "R", "scope": "full"})


def test_get_attestation_unknown_course_raises(libv2_root):
    with pytest.raises(FileNotFoundError):
        course_service.get_attestation("no-such-course")


# --------------------------------------------------------------------------- #
# Router surface
# --------------------------------------------------------------------------- #


@pytest.fixture
def client():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from gui.routers.courses import router

    app = fastapi.FastAPI()
    app.include_router(router, prefix="/api/courses")
    return TestClient(app)


def test_router_patch_attestation_round_trip(client, libv2_course):
    slug = libv2_course["slug"]
    resp = client.patch(
        f"/api/courses/{slug}/attestation",
        json={"reviewed_by": "Dr. Ada", "scope": "content"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reviewed_by"] == "Dr. Ada"
    assert body["reviewed_at"]

    got = client.get(f"/api/courses/{slug}/attestation")
    assert got.status_code == 200
    assert got.json()["scope"] == "content"


def test_router_patch_invalid_scope_is_422(client, libv2_course):
    resp = client.patch(
        f"/api/courses/{libv2_course['slug']}/attestation",
        json={"reviewed_by": "R", "scope": "nope"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_attestation"


def test_router_patch_unknown_course_is_404(client, libv2_root):
    resp = client.patch(
        "/api/courses/no-such-course/attestation",
        json={"reviewed_by": "R", "scope": "full"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "attestation_not_found"
