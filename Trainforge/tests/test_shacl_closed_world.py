"""Wave 88 — TRAINFORGE_SHACL_CLOSED_WORLD flag tests.

Verifies the file-based overlay architecture:

* Flag off (default): cfshapes:ChunkShape and cfshapes:TypedeEdgeShape are
  open. Unminted predicates on chunk / typed-edge nodes do NOT fire
  closure violations. Other validators may still fire (cardinality,
  IRI nodeKind, etc.) but the closed-world component must be silent.
* Flag on: the same unminted predicates trigger
  ``sh:ClosedConstraintComponent`` violations whose ``sh:resultPath``
  points at the offending predicate.
* Flag on with only minted predicates: no closure violations — the
  authoritative predicate set per chunk_v4_v1.jsonld + the reified-edge
  contexts admits the test triples cleanly.
* Perf sanity: closed-world validation of a 1000-node fixture
  completes in < 30 s. The plan flagged pyshacl as 5800x slower than
  Python rules; this test confirms closed-world overhead is bounded
  (sanity gate, not a performance assertion — print the timings so
  the user sees the actual numbers in the punch list).

These tests live alongside the Phase 5 SHACL rule tests
(test_shacl_rules_defined_by.py) and follow the same import-skip
pattern (skip-on-missing pyld/pyshacl/rdflib instead of fail-loud).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Skip the entire module when SHACL extras aren't installed; same
# pattern as lib/validators/tests/test_shacl_runner.py.
pyld = pytest.importorskip(
    "pyld",
    reason="pyld is required for SHACL tests; install with `pip install pyld`.",
)
pyshacl = pytest.importorskip(
    "pyshacl",
    reason="pyshacl is required for SHACL tests; install with `pip install pyshacl`.",
)
rdflib = pytest.importorskip("rdflib", reason="rdflib comes with pyshacl.")

from rdflib import Graph, Literal, Namespace, URIRef  # noqa: E402
from rdflib.namespace import RDF  # noqa: E402

from lib.validators.shacl_runner import (  # noqa: E402
    SHACL_CLOSED_WORLD_OVERLAY,
    ShaclViolation,
    run_shacl,
)

SHAPES_FILE = (
    _REPO_ROOT / "schemas" / "context" / "courseforge_v1.shacl.ttl"
)

SH = Namespace("http://www.w3.org/ns/shacl#")
CF = Namespace("https://ed4all.dev/ns/courseforge/v1#")
SCHEMA = Namespace("http://schema.org/")
EX = Namespace("http://example.org/test/")


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    """Default every test to flag-off; tests that need it set their own."""
    monkeypatch.delenv("TRAINFORGE_SHACL_CLOSED_WORLD", raising=False)
    yield


def _make_chunk_graph_with_unminted_predicate() -> Graph:
    """Build a minimal data graph with one chunk node carrying an
    unminted predicate (ex:randomPredicate) plus the schema:identifier
    that the open ChunkShape requires. Used by the flag-off baseline
    and the flag-on detection tests."""
    g = Graph()
    chunk = EX.chunk_001
    g.add((chunk, RDF.type, CF.Chunk))
    g.add((chunk, SCHEMA.identifier, Literal("chunk_001")))
    # Unminted predicate — should ONLY violate under closed-world.
    g.add((chunk, EX.randomPredicate, Literal("uncategorized-payload")))
    return g


def _make_chunk_graph_with_minted_predicates() -> Graph:
    """Build a chunk node using only Wave 87 minted predicates. Should
    pass closed-world validation (no closure violation)."""
    g = Graph()
    chunk = EX.chunk_002
    g.add((chunk, RDF.type, CF.Chunk))
    g.add((chunk, SCHEMA.identifier, Literal("chunk_002")))
    g.add((chunk, CF.schemaVersion, Literal("v4")))
    g.add((chunk, CF.sectionHeading, Literal("Test Section")))
    g.add((chunk, CF.moduleId, Literal("module:week_01")))
    g.add((chunk, CF.lessonId, Literal("lesson:l1")))
    g.add((chunk, CF.html, Literal("<p>Test body.</p>")))
    g.add((chunk, CF.summary, Literal("A short summary.")))
    g.add((chunk, CF.wordCount, Literal(3)))
    return g


def _closure_violations(violations: List[ShaclViolation]) -> List[ShaclViolation]:
    """Filter to only violations from sh:ClosedConstraintComponent."""
    return [
        v for v in violations
        if v.source_constraint_component
        == "http://www.w3.org/ns/shacl#ClosedConstraintComponent"
    ]


# --------------------------------------------------------------------- #
# Test 1 — Flag-off baseline
# --------------------------------------------------------------------- #


def test_flag_off_unminted_predicate_does_not_fire_closure():
    """With the flag unset, an unminted predicate on a chunk must not
    trigger a closed-world violation. Other validators (e.g. base
    ChunkShape's identifier minCount=1) may still fire — we only
    assert the absence of sh:ClosedConstraintComponent results.
    """
    assert "TRAINFORGE_SHACL_CLOSED_WORLD" not in os.environ
    g = _make_chunk_graph_with_unminted_predicate()
    _conforms, violations = run_shacl(SHAPES_FILE, g)
    closure = _closure_violations(violations)
    assert closure == [], (
        f"Closed-world fired with flag off: {closure!r}. "
        "Default semantics must remain open-world."
    )


# --------------------------------------------------------------------- #
# Test 2 — Flag-on detection
# --------------------------------------------------------------------- #


def test_flag_on_unminted_predicate_triggers_closure(monkeypatch):
    """With the flag set, the same graph must fire at least one
    sh:ClosedConstraintComponent violation whose sh:resultPath points
    at the offending predicate (ex:randomPredicate)."""
    monkeypatch.setenv("TRAINFORGE_SHACL_CLOSED_WORLD", "true")
    assert SHACL_CLOSED_WORLD_OVERLAY.exists(), (
        f"Closed-world overlay missing on disk: {SHACL_CLOSED_WORLD_OVERLAY}"
    )

    g = _make_chunk_graph_with_unminted_predicate()
    conforms, violations = run_shacl(SHAPES_FILE, g)

    closure = _closure_violations(violations)
    assert closure, (
        "Expected at least one ClosedConstraintComponent violation "
        f"under flag-on; got {len(violations)} total violations: "
        f"{[(v.source_constraint_component, v.path) for v in violations]}"
    )

    # At least one closure violation must point at the offending predicate.
    paths = {v.path for v in closure}
    assert str(EX.randomPredicate) in paths, (
        f"Closed-world violation did not target ex:randomPredicate; "
        f"observed paths={paths}"
    )

    assert conforms is False


# --------------------------------------------------------------------- #
# Test 3 — Flag-on legitimate predicates pass
# --------------------------------------------------------------------- #


def test_flag_on_minted_predicates_no_closure_violation(monkeypatch):
    """With the flag on, a chunk built solely from Wave 87 minted
    predicates must NOT fire any closed-world violation. Other
    validators may still report things (and that's fine for this
    test); the assertion is closure-only."""
    monkeypatch.setenv("TRAINFORGE_SHACL_CLOSED_WORLD", "true")

    g = _make_chunk_graph_with_minted_predicates()
    _conforms, violations = run_shacl(SHAPES_FILE, g)
    closure = _closure_violations(violations)
    assert closure == [], (
        f"Closed-world fired against minted predicates: "
        f"{[(v.path, v.message) for v in closure]}. "
        "The Wave 87 minted set must admit cleanly under closure."
    )


# --------------------------------------------------------------------- #
# Test 4 — Perf sanity (1000-node fixture)
# --------------------------------------------------------------------- #


def _build_synthetic_graph(n_chunks: int) -> Graph:
    """Build a synthetic graph with n_chunks chunk nodes, each carrying a
    handful of Wave 87 minted predicates. Used to bound closed-world
    overhead for a corpus-scale fixture."""
    g = Graph()
    for i in range(n_chunks):
        chunk = URIRef(f"http://example.org/test/chunk_{i:05d}")
        g.add((chunk, RDF.type, CF.Chunk))
        g.add((chunk, SCHEMA.identifier, Literal(f"chunk_{i:05d}")))
        g.add((chunk, CF.schemaVersion, Literal("v4")))
        g.add((chunk, CF.sectionHeading, Literal(f"Section {i}")))
        g.add((chunk, CF.moduleId, Literal(f"module:week_{(i % 12) + 1:02d}")))
        g.add((chunk, CF.wordCount, Literal(50 + (i % 200))))
    return g


def test_perf_sanity_closed_world_under_30s(monkeypatch, capsys):
    """Bound closed-world overhead on a 1000-node corpus: must complete
    in < 30 s. The plan tagged pyshacl as 5800x slower than Python
    rules; this is the smoke gate that ensures closed-world overhead
    specifically isn't catastrophic. We print both flag-off and flag-on
    timings so the punch list captures the actual delta."""
    n = 1000
    g_off = _build_synthetic_graph(n)
    g_on = _build_synthetic_graph(n)

    # Flag off baseline.
    monkeypatch.delenv("TRAINFORGE_SHACL_CLOSED_WORLD", raising=False)
    t0 = time.perf_counter()
    _, _ = run_shacl(SHAPES_FILE, g_off)
    off_secs = time.perf_counter() - t0

    # Flag on.
    monkeypatch.setenv("TRAINFORGE_SHACL_CLOSED_WORLD", "true")
    t0 = time.perf_counter()
    _, _ = run_shacl(SHAPES_FILE, g_on)
    on_secs = time.perf_counter() - t0

    # Print into pytest's captured stdout so -s / --capture=no surfaces it.
    print(
        f"\n[Wave 88 closed-world perf, n={n} chunks] "
        f"flag-off={off_secs:.2f}s  flag-on={on_secs:.2f}s  "
        f"delta={on_secs - off_secs:+.2f}s"
    )
    # Force capsys read so the printed line is preserved even if
    # other plugins capture stdout.
    captured = capsys.readouterr()
    sys.stdout.write(captured.out)

    assert on_secs < 30.0, (
        f"Closed-world validation took {on_secs:.2f}s on a {n}-node "
        f"fixture; sanity bound is 30s. If this exceeds 5 minutes, the "
        "plan said STOP rather than ship — escalate to user."
    )
