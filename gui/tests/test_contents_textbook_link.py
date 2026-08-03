"""Studio contents → accessible-textbook standing-link backing-API tests.

The Studio viewer (``gui/static/studio/studio.js::attachSourceMaterials``) appends
a standing "Accessible textbook" section to the course contents pane: one link
per archived source doc, pointing at the learner source viewer
(``/api/learn/source/<slug>?item_path=<doc>.html``). There is no JS test runner
in this repo, so these tests lock the **backing API contract** the link depends
on — its presence is a pure function of these two endpoints:

* ``GET /api/courses/<slug>/source-materials`` → ``dart_docs`` (the link list).
* ``GET /api/learn/source/<slug>?item_path=<doc>.html`` → the served accessible
  textbook page (the link target resolves + serves).

The tests use deterministic synthetic tmp-path courses and never inspect an
operator's private ``LibV2/courses`` archive. Skipped on a default install
without the ``gui`` extra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bs4")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402

# Reuse the synthetic source-materials archive builder (no real LibV2 touched).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_source_materials_router import _build_dart_course  # noqa: E402


@pytest.fixture
def client(state_dir, libv2_root):
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# Synthetic course — the contents link list + its target both resolve
# --------------------------------------------------------------------------- #


def test_contents_textbook_link_backed_by_inventory(client, libv2_root):
    slug = _build_dart_course(libv2_root, "alpha-101")

    # 1) The inventory the contents pane lists must expose the source doc.
    inv = client.get(f"/api/courses/{slug}/source-materials").json()
    assert inv["enabled"] is True
    docs = inv["dart_docs"]
    assert len(docs) >= 1
    doc = docs[0]["doc"]

    # 2) The link target the section builds (item_path=<doc>.html) must serve the
    #    accessible textbook through the learner source viewer.
    item_path = f"{doc}.html"
    resp = client.get(
        f"/api/learn/source/{slug}", params={"item_path": item_path}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    # The served page is the accessible source (its heading survived sanitise).
    assert "Real Numbers" in resp.text


def test_contents_link_absent_when_toggle_off(client, libv2_root, monkeypatch):
    # When source materials are disabled, the inventory reports enabled:false +
    # empty dart_docs → the contents section is omitted (no link to build).
    slug = _build_dart_course(libv2_root, "beta-202")
    monkeypatch.setenv("ED4ALL_SOURCE_MATERIALS", "off")
    inv = client.get(f"/api/courses/{slug}/source-materials").json()
    assert inv["enabled"] is False
    assert inv["dart_docs"] == []


def test_contents_link_absent_for_chunks_only_course(client, libv2_root):
    # A course with no archived source HTML yields an empty dart_docs list, so the
    # contents pane appends nothing (additive, never load-bearing).
    course = libv2_root / "courses" / "gamma-303"
    (course / "imscc_chunks").mkdir(parents=True)
    inv = client.get("/api/courses/gamma-303/source-materials").json()
    assert inv["enabled"] is True
    assert inv["dart_docs"] == []
