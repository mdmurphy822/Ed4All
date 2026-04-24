"""Wave 70 — JSON-LD projection of RetrievalResult.

Covers:

* ``to_jsonld()`` emits ``@context`` and ``@type``, plus the expected
  Schema.org / ed4all: predicates for populated fields.
* None-valued fields are omitted (keeps the emit compact).
* ``pyld.expand`` resolves the @context without network (via the Wave 64
  loader vendored as ``_shacl_validator.register_local_loader``) and
  produces fully-qualified IRI predicates.
* CLI ``--output jsonld`` emits a JSON array of JSON-LD docs suitable
  for a JSON-LD processor.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

pyld = pytest.importorskip(
    "pyld",
    reason="pyld required for JSON-LD tests; install with `pip install pyld`.",
)

from LibV2.tools.libv2._shacl_validator import register_local_loader  # noqa: E402
from LibV2.tools.libv2.cli import main  # noqa: E402
from LibV2.tools.libv2.retriever import RetrievalResult  # noqa: E402


# Make sure the @context URL resolves locally for every test in this module.
@pytest.fixture(autouse=True)
def _local_loader():
    register_local_loader()
    yield


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #


def _full_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-abc-001",
        text="Newton's second law states F = ma",
        score=0.8532,
        course_slug="classical-mechanics",
        domain="physics",
        chunk_type="explanation",
        difficulty="intermediate",
        concept_tags=["newtons-second-law", "force"],
        source={"module_id": "week_03_content", "page_id": "week_03_overview"},
        tokens_estimate=142,
        learning_outcome_refs=["TO-01", "CO-03"],
        bloom_level="apply",
    )


def _minimal_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-xyz",
        text="Short text",
        score=0.5,
        course_slug="demo",
        domain="general",
        chunk_type="summary",
        difficulty=None,
        concept_tags=[],
        source={},
        tokens_estimate=10,
        learning_outcome_refs=[],
        bloom_level=None,
    )


# -------------------------------------------------------------------- #
# to_jsonld() shape
# -------------------------------------------------------------------- #


def test_to_jsonld_has_context_and_type():
    doc = _full_result().to_jsonld()
    assert "@context" in doc, "JSON-LD emit must declare @context"
    assert "@type" in doc
    assert doc["@type"] == "ed4all:RetrievalResult"


def test_to_jsonld_maps_fields_to_predicates():
    """Every populated RetrievalResult field must land on an aligned
    JSON-LD key (per the predicate table in jsonld_emit.py)."""
    doc = _full_result().to_jsonld()
    assert doc["identifier"] == "chunk-abc-001"       # schema:identifier
    assert doc["text"] == "Newton's second law states F = ma"  # schema:text
    assert doc["retrievalScore"] == 0.8532            # ed4all:retrievalScore
    assert doc["bloomLevel"] == "apply"               # ed4all:bloomLevel
    assert doc["keywords"] == ["newtons-second-law", "force"]  # schema:keywords
    # derivedFromObjective values must be minted as stable IRIs.
    derived = doc["derivedFromObjective"]
    assert isinstance(derived, list) and len(derived) == 2
    for iri in derived:
        assert iri.startswith("https://ed4all.dev/ns/courseforge/v1/lo/")
    assert doc["isBasedOn"] == {
        "module_id": "week_03_content",
        "page_id": "week_03_overview",
    }


def test_to_jsonld_omits_none_fields():
    """``difficulty=None`` / ``bloom_level=None`` / empty lists must
    NOT appear as empty literals — that would confuse JSON-LD expand."""
    doc = _minimal_result().to_jsonld()
    assert "difficulty" not in doc
    assert "bloomLevel" not in doc
    assert "keywords" not in doc
    assert "derivedFromObjective" not in doc
    # isBasedOn on an empty source dict is also omitted.
    assert "isBasedOn" not in doc


def test_to_jsonld_accepts_context_override():
    """Callers can point @context at a custom URL (e.g. a proxy or
    versioned copy). The canonical URL is the default."""
    doc = _full_result().to_jsonld(context_url="https://example.org/my-ctx/v1")
    # The @context is augmented with an inline retrieval-specific block
    # so ``doc['@context']`` is a list; the first element is the URL.
    ctx = doc["@context"]
    assert isinstance(ctx, list) and ctx[0] == "https://example.org/my-ctx/v1"


# -------------------------------------------------------------------- #
# pyld.expand round trip
# -------------------------------------------------------------------- #


def test_pyld_expand_resolves_predicates():
    """Feeding the emit to ``pyld.expand`` must produce fully-qualified
    IRI predicates — proves the @context resolves and our predicate
    mappings are valid."""
    from pyld import jsonld as jsonld_mod

    doc = _full_result().to_jsonld()
    expanded = jsonld_mod.expand(doc)
    assert expanded, "pyld.expand returned empty — @context failed to resolve"
    node = expanded[0]

    # @type lifts to ed4all:RetrievalResult as a full IRI.
    assert "@type" in node
    types = node["@type"]
    assert any(
        t.endswith("RetrievalResult")
        for t in (types if isinstance(types, list) else [types])
    )
    # A few key predicates must appear as fully-qualified IRIs.
    assert any(
        k.endswith("courseforge/v1#retrievalScore") or k.endswith("#retrievalScore")
        for k in node
    )
    assert any("schema.org/identifier" in k for k in node)
    assert any("schema.org/keywords" in k for k in node)


def test_pyld_expand_minimal_result_is_clean():
    """A minimal result expands to at least @type + a couple predicates
    without pyld raising over absent fields."""
    from pyld import jsonld as jsonld_mod

    doc = _minimal_result().to_jsonld()
    expanded = jsonld_mod.expand(doc)
    assert expanded
    node = expanded[0]
    assert "@type" in node


# -------------------------------------------------------------------- #
# CLI --output jsonld
# -------------------------------------------------------------------- #


def _make_min_repo(tmp_path: Path) -> Path:
    (tmp_path / "courses").mkdir()
    (tmp_path / "catalog").mkdir()
    return tmp_path


def test_cli_retrieve_jsonld_output(tmp_path):
    """End-to-end: ``libv2 retrieve --output jsonld`` emits valid
    JSON-LD that pyld.expand processes without error."""
    from pyld import jsonld as jsonld_mod

    runner = CliRunner()
    fake_results = [_full_result(), _minimal_result()]

    with mock.patch(
        "LibV2.tools.libv2.retriever.retrieve_chunks",
        return_value=fake_results,
    ):
        repo = _make_min_repo(tmp_path)
        result = runner.invoke(
            main,
            ["--repo", str(repo), "retrieve", "query", "--output", "jsonld"],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list) and len(payload) == 2
    for doc in payload:
        assert "@context" in doc
        assert doc["@type"] == "ed4all:RetrievalResult"
        # At least one expand succeeds per doc.
        expanded = jsonld_mod.expand(doc)
        assert expanded, f"pyld.expand failed for doc: {doc}"


def test_cli_retrieve_jsonld_is_distinct_from_json(tmp_path):
    """``--output jsonld`` and ``--output json`` must produce different
    shapes — jsonld carries @context/@type, json carries flat field names."""
    runner = CliRunner()
    fake_results = [_full_result()]

    with mock.patch(
        "LibV2.tools.libv2.retriever.retrieve_chunks",
        return_value=fake_results,
    ):
        repo = _make_min_repo(tmp_path)
        r_json = runner.invoke(
            main,
            ["--repo", str(repo), "retrieve", "q", "--output", "json"],
        )
        r_jsonld = runner.invoke(
            main,
            ["--repo", str(repo), "retrieve", "q", "--output", "jsonld"],
        )

    assert r_json.exit_code == 0 and r_jsonld.exit_code == 0
    json_payload = json.loads(r_json.output)
    jsonld_payload = json.loads(r_jsonld.output)

    # json shape: flat chunk_id/score/...
    assert "chunk_id" in json_payload[0]
    assert "@context" not in json_payload[0]
    # jsonld shape: @context/identifier/retrievalScore
    assert "@context" in jsonld_payload[0]
    assert "identifier" in jsonld_payload[0]
    assert "chunk_id" not in jsonld_payload[0]
