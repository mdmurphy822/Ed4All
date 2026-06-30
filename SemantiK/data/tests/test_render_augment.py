"""Tests for the realistic-rendering presentation augmenter (data/render_augment.py).

Locks in the load-bearing contract: default OFF is the identity (byte-identical
render), augmentation is PRESENTATION ONLY (the HTML tag structure is never
altered), and the resolver / profile assignment is deterministic +
parse-with-fallback (mirroring resolve_column_order_mode)."""
from __future__ import annotations

import re
from collections import Counter

import pytest

from data.render_augment import (
    RENDER_PROFILES,
    augment_html,
    resolve_render_augment_mode,
    resolve_render_profile_for,
)

_SAMPLE = (
    "<!DOCTYPE html><html lang=en><head>"
    "<style>body{max-width:7in;margin:1in auto}</style></head>"
    "<body><h1>Title</h1><h2>Sec</h2><p>Para one.</p>"
    "<ul><li>item a</li><li>item b</li></ul>"
    "<blockquote>q</blockquote><table><caption>Cap</caption>"
    "<tr><td>c1</td><td>c2</td></tr></table></body></html>"
)

# Content tags whose count must be invariant under augmentation (the labels
# downstream derive from these — presentation must never add/remove them).
_CONTENT_TAGS = ("h1", "h2", "p", "li", "blockquote", "table", "caption", "td", "tr", "ul")


def _tag_counts(html: str) -> Counter:
    return Counter({t: len(re.findall(rf"<{t}[\s>]", html)) for t in _CONTENT_TAGS})


# ---------------------------------------------------------------------------
# Resolver — default OFF + parse-with-fallback
# ---------------------------------------------------------------------------


def test_default_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_RENDER_AUGMENT", raising=False)
    assert resolve_render_augment_mode() == "off"
    assert resolve_render_profile_for(7) == ("clean", 0)
    assert resolve_render_profile_for(123) == ("clean", 0)


@pytest.mark.parametrize("val", ["1", "true", "YES", "On"])
def test_truthy_is_mixed(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_RENDER_AUGMENT", val)
    assert resolve_render_augment_mode() == "mixed"


@pytest.mark.parametrize(
    "val", ["two_column", "serif_embed", "dense_layout", "two_column+serif_embed", "clean"]
)
def test_pinned_profile_mode(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_RENDER_AUGMENT", val)
    assert resolve_render_augment_mode() == val


@pytest.mark.parametrize("val", ["", "garbage", "2_col", "off"])
def test_garbage_off(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_RENDER_AUGMENT", val)
    assert resolve_render_augment_mode() == "off"


def test_profile_assignment_deterministic():
    for di in (0, 1, 42, 99999):
        assert resolve_render_profile_for(di, "mixed") == resolve_render_profile_for(di, "mixed")


def test_mixed_distribution_two_column_dominant():
    profiles = [resolve_render_profile_for(i, "mixed")[0] for i in range(4000)]
    c = Counter(profiles)
    # two_column is the highest-value profile (the audit's dominant gap).
    top = max(c, key=lambda k: c[k])
    assert top == "two_column", c
    # every drawn profile is renderable.
    for p in c:
        assert all(part in RENDER_PROFILES for part in p.split("+")), p


def test_pinned_profile_for_returns_that_profile():
    prof, seed = resolve_render_profile_for(11, "dense_layout")
    assert prof == "dense_layout" and seed == 11


# ---------------------------------------------------------------------------
# augment_html — clean identity + presentation-only invariant
# ---------------------------------------------------------------------------


def test_clean_is_identity():
    assert augment_html(_SAMPLE, "clean", 0) == _SAMPLE
    assert augment_html(_SAMPLE, "clean", 99) == _SAMPLE
    # a mode that leaked in (off / mixed) is a no-op, not a guess.
    assert augment_html(_SAMPLE, "off", 0) == _SAMPLE
    assert augment_html(_SAMPLE, "mixed", 0) == _SAMPLE
    assert augment_html("", "two_column", 0) == ""


@pytest.mark.parametrize(
    "profile", ["two_column", "serif_embed", "dense_layout", "two_column+serif_embed"]
)
def test_tag_structure_preserved(profile):
    before = _tag_counts(_SAMPLE)
    after = _tag_counts(augment_html(_SAMPLE, profile, 3))
    assert before == after, (profile, before, after)


def test_two_column_injects_multicolumn_css():
    out = augment_html(_SAMPLE, "two_column", 3)
    assert "column-count: 2" in out
    assert 'data-render-augment="two_column"' in out


def test_style_cascades_after_gold_style():
    # The injected block must come AFTER the gold pair's own <style> so it wins
    # on equal specificity / !important.
    out = augment_html(_SAMPLE, "two_column", 3)
    assert out.index("data-render-augment") > out.index("max-width:7in")


def test_serif_embed_uses_license_clean_font():
    out = augment_html(_SAMPLE, "serif_embed", 1)
    assert "DejaVu Serif" in out  # permissively-licensed, on-box


def test_dense_layout_furniture_is_generated_content_not_tags():
    out = augment_html(_SAMPLE, "dense_layout", 2)
    # Running header/footer via ::before/::after generated content — no new tag.
    assert "body::before" in out and "body::after" in out
    assert _tag_counts(out) == _tag_counts(_SAMPLE)


def test_composite_applies_both_segments():
    out = augment_html(_SAMPLE, "two_column+serif_embed", 5)
    assert "column-count: 2" in out and "DejaVu Serif" in out


def test_seed_varies_presentation_but_deterministic():
    a0 = augment_html(_SAMPLE, "two_column", 0)
    a0b = augment_html(_SAMPLE, "two_column", 0)
    a1 = augment_html(_SAMPLE, "two_column", 7)
    assert a0 == a0b  # deterministic per seed
    # different seeds CAN differ (minor margin/gutter/font variation); not
    # required to differ for every pair, so just assert reproducibility above.


def test_injects_without_head():
    no_head = "<html><body><p>x</p></body></html>"
    out = augment_html(no_head, "two_column", 0)
    assert "column-count: 2" in out and out.count("<p>") == 1
    assert out.index("data-render-augment") < out.index("<body")
