"""W7.1 + W7.2 — bundled SEMANTIK_DEPLOY_PROFILE + single-column guard.

The deploy/render seam ships figure detection AND column reading-order OFF, so
a deploy-like run drops every image (WCAG 1.1.1 loss) and mis-orders columns.
``SEMANTIK_DEPLOY_PROFILE`` (default OFF) turns BOTH on together. W7.2 also adds
a single-column guard so a genuine single-column doc never gets an invented
gutter. All CPU-only, no model, no PDF.
"""
from __future__ import annotations

import os

import pytest

from dart_semantic.cascade import resolve_detect_figures
from dart_semantic.extract_shared import _detect_figures_enabled
from dart_semantic.reading_order import (
    column_ids_for_x0s,
    resolve_column_order_mode,
    resolve_deploy_profile,
)

_FLAGS = ("SEMANTIK_DEPLOY_PROFILE", "SEMANTIK_COLUMN_ORDER", "SEMANTIK_DETECT_FIGURES")


@pytest.fixture(autouse=True)
def _clear_flags():
    prev = {k: os.environ.get(k) for k in _FLAGS}
    for k in _FLAGS:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Resolver semantics (parse-with-fallback, default OFF).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "On"])
def test_deploy_profile_truthy_on(val):
    os.environ["SEMANTIK_DEPLOY_PROFILE"] = val
    assert resolve_deploy_profile() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "garbage", "2x"])
def test_deploy_profile_falsey_or_garbage_off(val):
    os.environ["SEMANTIK_DEPLOY_PROFILE"] = val
    assert resolve_deploy_profile() is False


def test_default_off_byte_identical_gates():
    # No flags set: figure detection off AND column order off (deploy output
    # byte-identical to current default).
    assert resolve_deploy_profile() is False
    assert resolve_detect_figures() is False
    assert _detect_figures_enabled() is False
    assert resolve_column_order_mode() is False


# ---------------------------------------------------------------------------
# Bundling: the ONE flag turns on BOTH figure detection and column order.
# ---------------------------------------------------------------------------
def test_deploy_profile_bundles_both():
    os.environ["SEMANTIK_DEPLOY_PROFILE"] = "1"
    # W7.1 figure detection (cascade + extract-side mirror both flip).
    assert resolve_detect_figures() is True
    assert _detect_figures_enabled() is True
    # W7.2 column reading-order.
    assert resolve_column_order_mode() is True


def test_individual_flags_still_work_independently():
    os.environ["SEMANTIK_DETECT_FIGURES"] = "1"
    assert resolve_detect_figures() is True
    assert _detect_figures_enabled() is True
    # column order NOT enabled by the figure flag alone.
    assert resolve_column_order_mode() is False

    os.environ.pop("SEMANTIK_DETECT_FIGURES", None)
    os.environ["SEMANTIK_COLUMN_ORDER"] = "1"
    assert resolve_column_order_mode() is True
    assert resolve_detect_figures() is False


# ---------------------------------------------------------------------------
# W7.2 single-column guard.
# ---------------------------------------------------------------------------
def test_guard_default_off_preserves_direct_call():
    # Without the guard kwarg (council BERT-feature path + direct callers),
    # a stray outlier still spawns a column — byte-identical legacy behaviour.
    x0s = [72.0, 72.0, 72.0, 72.0, 300.0]  # gap 228 > 0.06*612
    assert max(column_ids_for_x0s(x0s, 612.0)) == 1


def test_guard_collapses_spurious_single_column():
    # One right-aligned outlier (page number / caption) must NOT invent a
    # column when the reorder path passes single_column_guard=True.
    x0s = [72.0, 72.0, 72.0, 72.0, 300.0]
    assert column_ids_for_x0s(x0s, 612.0, single_column_guard=True) == [0, 0, 0, 0, 0]


def test_guard_keeps_genuine_two_column():
    # Balanced 2-column layout survives the guard (each column well-populated).
    x0s = [72.0, 330.0, 73.0, 332.0, 71.0, 331.0]
    assert column_ids_for_x0s(x0s, 612.0, single_column_guard=True) == [0, 1, 0, 1, 0, 1]


def test_guard_collapses_minority_share_column():
    # A handful of indented lines (minority column) below the 15% share floor
    # collapse; the body stays column 0.
    x0s = [72.0] * 20 + [140.0, 141.0]  # 22 seeds, minority=2 (< max(2, ceil(.15*22)=4))
    out = column_ids_for_x0s(x0s, 612.0, single_column_guard=True)
    assert set(out) == {0}
