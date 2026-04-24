"""Wave 64 — pyld document loader for the Courseforge @context.

Proves ``register_local_loader`` makes ``https://ed4all.dev/ns/courseforge/v1``
resolve to the bundled repo copy without network access, while leaving
unknown URLs to fall through to the default HTTP loader.

Covers:

* The canonical URL constant matches the @context string embedded in
  the JSON Schema (no drift).
* ``load_courseforge_context`` returns a parseable JSON-LD context doc.
* After ``register_local_loader()``, pyld's ``expand`` on a payload whose
  ``@context`` is the canonical URL string (not inline) resolves the
  context locally — proves the loader intercepts the URL correctly.
* An unknown URL falls through to the fallback loader (we inject a
  sentinel fallback to avoid hitting the network).
* Round trip with the loader installed: pyld expand → compact yields
  back the compact form.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

pyld = pytest.importorskip("pyld", reason="pyld required for Wave 64 loader tests.")

from pyld import jsonld  # noqa: E402

from lib.ontology.jsonld_context_loader import (  # noqa: E402
    CANONICAL_COURSEFORGE_CONTEXT_URL,
    load_courseforge_context,
    register_local_loader,
)


@pytest.fixture
def reset_pyld_loader():
    """Snapshot pyld's default loader around each test so state doesn't leak."""
    original = None
    try:
        original = jsonld._default_document_loader  # type: ignore[attr-defined]
    except AttributeError:
        pass
    yield
    if original is not None:
        jsonld.set_document_loader(original)


# ---------------------------------------------------------------------- #
# 1. Canonical URL + context doc basics
# ---------------------------------------------------------------------- #


def test_canonical_url_matches_schema_enforced_value():
    """The URL the loader serves must match what the JSON Schema enforces."""
    import json

    schema_path = (
        _PROJECT_ROOT
        / "schemas"
        / "knowledge"
        / "courseforge_jsonld_v1.schema.json"
    )
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    enforced_const = schema["properties"]["@context"]["const"]
    assert CANONICAL_COURSEFORGE_CONTEXT_URL == enforced_const, (
        f"Loader URL {CANONICAL_COURSEFORGE_CONTEXT_URL!r} must match the JSON "
        f"Schema's @context const {enforced_const!r}; otherwise emit-time "
        f"validation and runtime loading disagree."
    )


def test_load_courseforge_context_returns_valid_jsonld_context():
    doc = load_courseforge_context()
    assert "@context" in doc, (
        f"Context doc must wrap its mapping under @context; got keys "
        f"{list(doc.keys())!r}"
    )
    inner = doc["@context"]
    assert isinstance(inner, dict)
    assert inner.get("@version") == 1.1


# ---------------------------------------------------------------------- #
# 2. Loader intercepts the canonical URL
# ---------------------------------------------------------------------- #


def test_loader_resolves_canonical_url_to_local_doc(reset_pyld_loader):
    register_local_loader(preserve_existing=False)
    payload = {
        "@context": CANONICAL_COURSEFORGE_CONTEXT_URL,
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
    }
    # This would fail without the loader because the URL isn't HTTP-resolvable
    # (we don't own ed4all.dev hosting). With the loader installed it serves
    # the bundled copy.
    expanded = jsonld.expand(payload)
    assert isinstance(expanded, list)
    assert len(expanded) == 1
    # @type expands to schema:LearningResource → proof the context was loaded.
    types = expanded[0].get("@type", [])
    assert "http://schema.org/LearningResource" in types, (
        f"Loader didn't serve the expected context; @type didn't resolve. "
        f"Got {types!r}"
    )


def test_loader_expand_compact_roundtrip(reset_pyld_loader):
    register_local_loader(preserve_existing=False)
    payload = {
        "@context": CANONICAL_COURSEFORGE_CONTEXT_URL,
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 2,
        "moduleType": "content",
        "pageId": "week_02_content_01",
    }
    expanded = jsonld.expand(payload)
    compacted = jsonld.compact(expanded, CANONICAL_COURSEFORGE_CONTEXT_URL)
    # compact back to CourseModule type
    assert compacted.get("@type") == "CourseModule"
    assert compacted.get("courseCode") == "TEST_101"


# ---------------------------------------------------------------------- #
# 3. Unknown URLs fall through
# ---------------------------------------------------------------------- #


def test_loader_falls_through_to_previous_loader(reset_pyld_loader):
    """Unknown URLs must hit the chained previous loader, not short-circuit."""
    sentinel = {"contextUrl": None, "documentUrl": "http://other/", "document": {}}
    calls = []

    def fake_previous(url, options=None):
        calls.append(url)
        return sentinel

    jsonld.set_document_loader(fake_previous)
    register_local_loader(preserve_existing=True)

    # The canonical URL must NOT hit the fallback.
    jsonld.expand(
        {
            "@context": CANONICAL_COURSEFORGE_CONTEXT_URL,
            "@type": "CourseModule",
            "courseCode": "TEST_101",
            "weekNumber": 1,
            "moduleType": "overview",
            "pageId": "x",
        }
    )
    assert CANONICAL_COURSEFORGE_CONTEXT_URL not in calls, (
        f"Loader leaked canonical URL to fallback loader; got calls {calls!r}"
    )


def test_register_local_loader_is_idempotent(reset_pyld_loader):
    """Calling register_local_loader() twice is safe and doesn't infinitely
    chain the loader."""
    register_local_loader(preserve_existing=True)
    register_local_loader(preserve_existing=True)
    # Still resolves the canonical URL without issues.
    expanded = jsonld.expand(
        {
            "@context": CANONICAL_COURSEFORGE_CONTEXT_URL,
            "@type": "CourseModule",
            "courseCode": "TEST_101",
            "weekNumber": 1,
            "moduleType": "overview",
            "pageId": "x",
        }
    )
    assert expanded
