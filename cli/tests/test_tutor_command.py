"""Tests for ``ed4all tutor`` (Wave 77).

CliRunner-driven smoke tests of the three subcommands (diagnose,
inventory, guardrails) in both ``--format text`` and ``--format json``
modes. Real archive: ``rdf-shacl-550`` (the canonical Wave 77 fixture
checked into LibV2/courses/).
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from cli.commands.tutor import tutor_group


SLUG = "rdf-shacl-550"


def _run(*args):
    """Invoke ``tutor`` with ``args`` and return the Result."""
    runner = CliRunner()
    return runner.invoke(tutor_group, list(args))


# ---------------------------------------------------------------------- #
# diagnose
# ---------------------------------------------------------------------- #


def test_diagnose_text_output_smoke():
    """``ed4all tutor diagnose --slug rdf-shacl-550 --text "..."``
    in text mode renders top matches with the misconception + correction."""
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
    """Inventory in text mode shows N clusters and at least one
    cluster label per group."""
    res = _run(
        "inventory",
        "--slug", SLUG,
        "--clusters", "4",
        "--format", "text",
    )
    assert res.exit_code == 0, res.output
    out = res.output
    assert "Cluster 1" in out
    assert "Cluster 4" in out
    assert "Members" in out


def test_inventory_json_output_is_valid_json():
    """Inventory ``--format json`` is parseable + carries documented shape."""
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
    assert len(clusters) == 4
    total = sum(c["size"] for c in clusters)
    assert total == 67


# ---------------------------------------------------------------------- #
# guardrails
# ---------------------------------------------------------------------- #


def test_guardrails_text_output_smoke():
    """``rdf-graph`` guardrails render in text mode."""
    res = _run(
        "guardrails",
        "--slug", SLUG,
        "--concept", "rdf-graph",
        "--format", "text",
    )
    assert res.exit_code == 0, res.output
    out = res.output
    assert "guardrails" in out.lower() or "avoid" in out.lower()
    # Lower-bound: at least one Avoid: line
    assert out.lower().count("avoid:") >= 1


def test_guardrails_json_output_is_valid_json():
    """Guardrails ``--format json`` is parseable + has >=1 entry."""
    res = _run(
        "guardrails",
        "--slug", SLUG,
        "--concept", "rdf-graph",
        "--format", "json",
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["concept"] == "rdf-graph"
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
