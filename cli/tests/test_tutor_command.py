"""Tests for ``ed4all tutor`` (Wave 77).

CliRunner-driven smoke tests of the three subcommands (diagnose,
inventory, guardrails) in both ``--format text`` and ``--format json``
modes with count-agnostic assertions. A neutral synthetic archive is built
under ``tmp_path``; tests never discover or read operator course data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.commands.tutor import tutor_group
from MCP.tools.tutoring_tools import load_misconception_index


SLUG = "sample-course"


@pytest.fixture(autouse=True)
def synthetic_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    courses_root = tmp_path / "courses"
    course = courses_root / SLUG
    (course / "corpus").mkdir(parents=True)
    (course / "graph").mkdir()
    misconceptions = [
        (
            "An RDF triple is like a row in a relational table",
            "RDF represents graph statements, not relational rows",
            "rdf-graph",
        ),
        (
            "A shape always changes the source graph",
            "A validation shape reports conformance without rewriting source data",
            "shape-validation",
        ),
        (
            "Every identifier must resolve over the network",
            "Identifiers can name resources without network dereferencing",
            "identifiers",
        ),
        (
            "A missing property and an empty value are identical",
            "Constraint semantics distinguish absence from an explicit empty value",
            "property-constraints",
        ),
    ]
    with (course / "corpus" / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for index, (wrong, correction, concept) in enumerate(misconceptions, 1):
            stream.write(json.dumps({
                "id": f"chunk-{index}",
                "concept_tags": [concept],
                "source": {"source_references": [{"sourceId": f"src-{index}"}]},
                "misconceptions": [{
                    "misconception": wrong,
                    "correction": correction,
                }],
            }) + "\n")
    (course / "graph" / "pedagogy_graph.json").write_text(json.dumps({
        "nodes": [
            {"id": f"mc-{index}", "statement": wrong}
            for index, (wrong, _correction, _concept) in enumerate(misconceptions, 1)
        ],
        "edges": [
            {
                "source": f"mc-{index}",
                "target": f"concept:{concept}",
                "relation_type": "interferes_with",
            }
            for index, (_wrong, _correction, concept) in enumerate(misconceptions, 1)
        ],
    }), encoding="utf-8")

    import MCP.tools.tutoring_tools as tutoring_tools

    monkeypatch.setattr(tutoring_tools, "LIBV2_COURSES", courses_root)
    monkeypatch.setattr(tutoring_tools, "_select_backend", lambda: "jaccard")
    tutoring_tools._INDEX_CACHE.clear()
    yield
    tutoring_tools._INDEX_CACHE.clear()


def _run(*args):
    """Invoke ``tutor`` with ``args`` and return the Result."""
    runner = CliRunner()
    return runner.invoke(tutor_group, list(args))


def _first_concept_with_guardrails():
    """Return any concept slug from this corpus that has at least one
    interferes_with edge, or ``None`` if no concept has guardrails."""
    from MCP.tools.tutoring_tools import preemptive_misconception_guardrails
    index = load_misconception_index(SLUG)
    for c in index.concept_to_mc_keys:
        if preemptive_misconception_guardrails(SLUG, c):
            return c
    return None


# ---------------------------------------------------------------------- #
# diagnose
# ---------------------------------------------------------------------- #


def test_diagnose_text_output_smoke():
    """``ed4all tutor diagnose --slug <slug> --text "..."`` in text
    mode renders top matches with the misconception + correction."""
    res = _run(
        "diagnose",
        "--slug", SLUG,
        "--text", "An RDF triple is like a row in a relational table",
        "--top-k", "3",
        "--format", "text",
    )
    assert res.exit_code == 0, res.output
    out = res.output.lower()
    assert "match" in out
    assert "misconception" in out
    assert "row" in out or "relational" in out


def test_diagnose_json_output_is_valid_json():
    """``--format json`` emits a JSON object with results array."""
    res = _run(
        "diagnose",
        "--slug", SLUG,
        "--text", "An RDF triple is like a row in a relational table",
        "--top-k", "3",
        "--format", "json",
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["slug"] == SLUG
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) >= 1
    assert payload["results"][0]["score"] > 0.5


def test_diagnose_no_matches_is_clean_message():
    """An archive with no misconceptions yields a clean text message."""
    res = _run(
        "diagnose",
        "--slug", "does-not-exist-slug",
        "--text", "anything",
        "--format", "text",
    )
    assert res.exit_code == 0, res.output
    assert "no" in res.output.lower()


# ---------------------------------------------------------------------- #
# inventory
# ---------------------------------------------------------------------- #


def test_inventory_text_output_smoke():
    """Inventory in text mode shows >=1 cluster header per group with
    a Members line."""
    res = _run(
        "inventory",
        "--slug", SLUG,
        "--clusters", "4",
        "--format", "text",
    )
    assert res.exit_code == 0, res.output
    out = res.output
    # At least Cluster 1 always exists when index is populated; higher
    # numbered clusters are corpus-dependent (kmeans caps at unique-
    # statement count).
    assert "Cluster 1" in out
    assert "Members" in out


def test_inventory_json_output_is_valid_json():
    """Inventory ``--format json`` is parseable + carries documented
    shape. Total members must match the loaded misconception index
    size; cluster count is bounded by both --clusters and the unique-
    statement count."""
    res = _run(
        "inventory",
        "--slug", SLUG,
        "--clusters", "4",
        "--format", "json",
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["slug"] == SLUG
    clusters = payload["clusters"]
    assert isinstance(clusters, list)
    expected_total = len(load_misconception_index(SLUG))
    assert 1 <= len(clusters) <= min(4, expected_total)
    total = sum(c["size"] for c in clusters)
    assert total == expected_total


# ---------------------------------------------------------------------- #
# guardrails
# ---------------------------------------------------------------------- #


def test_guardrails_text_output_smoke():
    """Guardrails for some concept in the corpus render in text mode
    with at least one ``Avoid:`` line."""
    concept = _first_concept_with_guardrails()
    if concept is None:
        pytest.fail("synthetic archive must carry at least one guardrail")
    res = _run(
        "guardrails",
        "--slug", SLUG,
        "--concept", concept,
        "--format", "text",
    )
    assert res.exit_code == 0, res.output
    out = res.output
    assert "guardrails" in out.lower() or "avoid" in out.lower()
    assert out.lower().count("avoid:") >= 1


def test_guardrails_json_output_is_valid_json():
    """Guardrails ``--format json`` is parseable + has >=1 entry for
    a concept that's known to carry guardrails in this corpus."""
    concept = _first_concept_with_guardrails()
    if concept is None:
        pytest.fail("synthetic archive must carry at least one guardrail")
    res = _run(
        "guardrails",
        "--slug", SLUG,
        "--concept", concept,
        "--format", "json",
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["concept"] == concept
    assert payload["slug"] == SLUG
    assert isinstance(payload["guardrails"], list)
    assert len(payload["guardrails"]) >= 1


def test_guardrails_unknown_concept_clean_message():
    """Unknown concept yields clean text + empty JSON list."""
    res = _run(
        "guardrails",
        "--slug", SLUG,
        "--concept", "no-such-concept",
        "--format", "text",
    )
    assert res.exit_code == 0, res.output
    assert "no" in res.output.lower()

    res_json = _run(
        "guardrails",
        "--slug", SLUG,
        "--concept", "no-such-concept",
        "--format", "json",
    )
    assert res_json.exit_code == 0, res_json.output
    payload = json.loads(res_json.output)
    assert payload["guardrails"] == []
