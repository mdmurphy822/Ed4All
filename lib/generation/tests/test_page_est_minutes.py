"""W1.6 — per-page estimated learning-time helpers (ED4ALL_PAGE_EST_MINUTES)."""
from __future__ import annotations

import math

import pytest

from lib.generation.content_page_budget import (
    count_interactions,
    count_visible_words,
    estimate_page_minutes,
    page_est_minutes_enabled,
    resolve_page_wpm,
    resolve_per_interaction_minutes,
)


# ---------------------------------------------------------------- enabled ---
def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_PAGE_EST_MINUTES", raising=False)
    assert page_est_minutes_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_enabled_truthy(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_PAGE_EST_MINUTES", val)
    assert page_est_minutes_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "garbage", ""])
def test_enabled_falsey(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_PAGE_EST_MINUTES", val)
    assert page_est_minutes_enabled() is False


# ------------------------------------------------------------------- WPM ----
def test_wpm_default(monkeypatch):
    monkeypatch.delenv("ED4ALL_PAGE_WPM", raising=False)
    assert resolve_page_wpm() == 200


def test_wpm_env_override(monkeypatch):
    monkeypatch.setenv("ED4ALL_PAGE_WPM", "150")
    assert resolve_page_wpm() == 150


def test_wpm_arg_wins(monkeypatch):
    monkeypatch.setenv("ED4ALL_PAGE_WPM", "150")
    assert resolve_page_wpm(300) == 300


@pytest.mark.parametrize("bad", ["0", "-5", "abc", ""])
def test_wpm_garbage_falls_back(monkeypatch, bad):
    monkeypatch.setenv("ED4ALL_PAGE_WPM", bad)
    assert resolve_page_wpm() == 200


# ----------------------------------------------------- per-interaction ------
def test_per_interaction_default(monkeypatch):
    monkeypatch.delenv("ED4ALL_PAGE_INTERACTION_MINUTES", raising=False)
    assert resolve_per_interaction_minutes() == 1.0


def test_per_interaction_zero_honored(monkeypatch):
    monkeypatch.setenv("ED4ALL_PAGE_INTERACTION_MINUTES", "0")
    assert resolve_per_interaction_minutes() == 0.0


@pytest.mark.parametrize("bad", ["-1", "junk"])
def test_per_interaction_garbage_falls_back(monkeypatch, bad):
    monkeypatch.setenv("ED4ALL_PAGE_INTERACTION_MINUTES", bad)
    assert resolve_per_interaction_minutes() == 1.0


# --------------------------------------------------------- word counting ----
def test_count_visible_words_strips_tags():
    html = "<p>one two three</p><h2>four five</h2>"
    assert count_visible_words(html) == 5


def test_count_visible_words_empty():
    assert count_visible_words("") == 0
    assert count_visible_words("<div></div>") == 0


# ----------------------------------------------------- interaction count ----
def test_count_interactions_component_and_class():
    html = (
        '<div data-cf-component="flip-card"></div>'
        '<div data-cf-component="self-check"></div>'
        '<section class="assessment-item"></section>'
        '<section class="guided-practice step"></section>'
    )
    assert count_interactions(html) == 4


def test_count_interactions_none():
    assert count_interactions("<p>plain prose only</p>") == 0


# --------------------------------------------------------- estimate ---------
def test_estimate_empty_page_zero():
    assert estimate_page_minutes("<div></div>") == 0


def test_estimate_reading_only():
    # 400 words at 200 wpm == 2.0 minutes, no interactions.
    html = "<p>" + " ".join(["word"] * 400) + "</p>"
    assert estimate_page_minutes(html) == 2


def test_estimate_adds_interaction_constant():
    # 200 words == 1.0 min reading + 2 interactions * 1.0 == 3.0 min.
    words = "<p>" + " ".join(["word"] * 200) + "</p>"
    html = (
        words
        + '<div data-cf-component="self-check"></div>'
        + '<section class="assessment-item"></section>'
    )
    assert estimate_page_minutes(html) == 3


def test_estimate_rounds_up():
    # 250 words / 200 wpm == 1.25 → ceil == 2.
    html = "<p>" + " ".join(["word"] * 250) + "</p>"
    assert estimate_page_minutes(html) == 2


def test_estimate_wpm_override():
    html = "<p>" + " ".join(["word"] * 300) + "</p>"
    # 300 / 300 wpm == 1.0.
    assert estimate_page_minutes(html, wpm=300) == 1


def test_estimate_interaction_minutes_override():
    words = "<p>" + " ".join(["word"] * 200) + "</p>"
    html = words + '<div data-cf-component="self-check"></div>'
    # 1.0 reading + 1 interaction * 2.5 == 3.5 → ceil 4.
    assert estimate_page_minutes(html, per_interaction_minutes=2.5) == 4
