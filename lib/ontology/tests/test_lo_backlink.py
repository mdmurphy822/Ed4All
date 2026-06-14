"""W2 §4.3 — deterministic CO→TO backlink invariants."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.lo_backlink import backlink_cos_to_tos  # noqa: E402
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402


def test_single_to_shortcut():
    terminals = [{"id": "TO-01", "statement": "Master arithmetic."}]
    chapters = [
        {"id": "CO-01", "statement": "Add numbers."},
        {"id": "CO-02", "statement": "Subtract numbers."},
    ]
    backlink_cos_to_tos(terminals, chapters)
    assert all(co["terminal_id"] == "TO-01" for co in chapters)


def test_nearest_to_by_embedding():
    terminals = [
        {"id": "TO-01", "statement": "Understand algebra equations variables."},
        {"id": "TO-02", "statement": "Understand geometry shapes triangles angles."},
    ]
    chapters = [
        {"id": "CO-01", "statement": "Solve algebra equations with variables."},
        {"id": "CO-02", "statement": "Measure triangle angles in geometry shapes."},
    ]
    backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert chapters[0]["terminal_id"] == "TO-01"
    assert chapters[1]["terminal_id"] == "TO-02"


def test_token_overlap_fallback_never_unset():
    """Embeddings absent → token-overlap fallback; terminal_id always set."""
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear"},
        {"id": "TO-02", "statement": "geometry triangle angle measure"},
    ]
    chapters = [
        {"id": "CO-01", "statement": "solve linear algebra equations"},
    ]
    backlink_cos_to_tos(terminals, chapters, embed=None)
    assert chapters[0]["terminal_id"] == "TO-01"


def test_reconcile_hint_honored():
    terminals = [
        {"id": "TO-01", "statement": "alpha"},
        {"id": "TO-02", "statement": "beta"},
    ]
    chapters = [{"id": "CO-01", "statement": "gamma", "terminal_id": "TO-02"}]
    backlink_cos_to_tos(terminals, chapters, embed=None)
    assert chapters[0]["terminal_id"] == "TO-02"
