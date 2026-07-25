"""Regression net for the assessment item-bank flag + expansive scaling knob.

Two resolvers in ``MCP/tools/pipeline_tools.py``:

* ``_item_bank_enabled`` — gates the ``06_assessments/item_bank.xml``
  ``<objectbank>`` sidecar.
* ``_resolve_items_per_objective`` — multiplies the per-objective item floor
  so a course can emit an EXPANSIVE bank instead of the 1-per-objective
  minimum a fixed weekly exam needs.

Both default to the byte-identical legacy behaviour and parse with fallback,
so a misconfigured value can never shrink coverage below the
one-item-per-objective archival-gate floor.
"""

import pytest

from MCP.tools.pipeline_tools import (
    _item_bank_enabled,
    _resolve_items_per_objective,
)

BANK_ENV = "ED4ALL_ASSESSMENT_ITEM_BANK"
IPO_ENV = "ED4ALL_ASSESSMENT_ITEMS_PER_OBJECTIVE"


# ── item-bank gate ──────────────────────────────────────────────────────────
def test_item_bank_defaults_off(monkeypatch):
    monkeypatch.delenv(BANK_ENV, raising=False)
    assert _item_bank_enabled() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " on "])
def test_item_bank_truthy(monkeypatch, raw):
    monkeypatch.setenv(BANK_ENV, raw)
    assert _item_bank_enabled() is True


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "garbage"])
def test_item_bank_falsey_or_garbage_is_off(monkeypatch, raw):
    monkeypatch.setenv(BANK_ENV, raw)
    assert _item_bank_enabled() is False


# ── expansive scaling knob ──────────────────────────────────────────────────
def test_items_per_objective_defaults_to_one(monkeypatch):
    """Default 1 keeps the per-objective budget byte-identical."""
    monkeypatch.delenv(IPO_ENV, raising=False)
    assert _resolve_items_per_objective() == 1


@pytest.mark.parametrize("raw,expected", [("2", 2), ("3", 3), ("6", 6), (" 4 ", 4)])
def test_items_per_objective_scales(monkeypatch, raw, expected):
    monkeypatch.setenv(IPO_ENV, raw)
    assert _resolve_items_per_objective() == expected


@pytest.mark.parametrize("raw", ["0", "-1", "-99", "garbage", "", "1.5", "None"])
def test_items_per_objective_never_shrinks_below_floor(monkeypatch, raw):
    """Garbage / non-positive must fall back to 1, never 0.

    A 0 multiplier would collapse the per-objective floor and let the strict
    archival gate see uncovered objectives, so the fallback is load-bearing.
    """
    monkeypatch.setenv(IPO_ENV, raw)
    assert _resolve_items_per_objective() == 1


def test_effective_budget_formula(monkeypatch):
    """The budget is max(question_count, n_objectives * multiplier)."""
    monkeypatch.setenv(IPO_ENV, "3")
    mult = _resolve_items_per_objective()
    n_objectives, question_count = 25, 5
    assert max(question_count, n_objectives * mult) == 75
    # The question_count floor still wins when it is the larger of the two.
    assert max(200, n_objectives * mult) == 200


def test_multiplier_of_one_matches_legacy_budget(monkeypatch):
    monkeypatch.delenv(IPO_ENV, raising=False)
    mult = _resolve_items_per_objective()
    n_objectives, question_count = 25, 5
    assert max(question_count, n_objectives * mult) == max(
        question_count, n_objectives
    )
