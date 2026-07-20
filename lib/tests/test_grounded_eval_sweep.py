"""Tests for the E1 flag A/B sweep driver (lib.retrieval.grounded_eval_sweep).

CI-safe: the grounded-answer pipeline is injected via ``answer_fn`` (no real
model / network). A spy answer_fn reads ``os.environ`` to prove each arm's flag
is LIVE during that arm's eval and RESTORED afterwards. The matrix + diff machinery
is checked on the mini-course fixture.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from lib.retrieval.grounded_eval_sweep import (
    BASELINE_ARM,
    SWEEP_FLAGS,
    _MANAGED_ENV,
    render_matrix,
    resolve_arms,
    run_flag_sweep,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "retrieval" / "mini_course"
)


@pytest.fixture
def libv2_course(tmp_path, monkeypatch):
    slug = "mini-retrieval-101"
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / slug
    shutil.copytree(FIXTURE, course_dir)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return tmp_path, slug, course_dir


class _FakeCitation:
    def __init__(self, chunk_id):
        self.chunk_id = chunk_id
        self.anchor_status = "resolved_exact"
        self.page_label = "P"
        self.text_quote = "q"

    def to_dict(self):
        return {"chunk_id": self.chunk_id, "anchor_status": self.anchor_status,
                "page_label": self.page_label, "text_quote": self.text_quote}


class _FakeAnswer:
    def __init__(self, status, citations):
        self.status = status
        self.answer_text = "A." if citations else None
        self.citations = citations
        self.groundedness = None
        self.latency_ms = 1.0
        self.model_id = "fake"
        self.prompt_version = "v"
        self.confidence = {"policy_version": "p"}


_GOLD_MAP = {
    "What does a vector store index?": "mini_alpha_chunk_001",
    "How is retrieval quality commonly measured?": "mini_alpha_chunk_003",
    "Where does the course cover chunking strategies?": "mini_beta_chunk_005",
}


def _plain_answer_fn(repo_root, course_slug, query, **kwargs):
    cid = _GOLD_MAP.get(query)
    if cid is None:
        return _FakeAnswer("refused_low_confidence", [])
    return _FakeAnswer("answered", [_FakeCitation(cid)])


# ===========================================================================
# resolve_arms
# ===========================================================================

def test_resolve_arms_default_is_baseline_plus_seven():
    arms = resolve_arms(None)
    assert arms[0] == BASELINE_ARM
    assert set(arms) == {BASELINE_ARM, *SWEEP_FLAGS.keys()}
    assert len(arms) == 8


def test_resolve_arms_subset_keeps_order_and_baseline():
    arms = resolve_arms(["hyde", "decompose"])
    # baseline always first; the rest in SWEEP_FLAGS declaration order.
    assert arms[0] == BASELINE_ARM
    assert arms == [BASELINE_ARM, "decompose", "hyde"]


def test_resolve_arms_unknown_is_loud():
    with pytest.raises(ValueError) as exc:
        resolve_arms(["not_a_flag"])
    assert "not_a_flag" in str(exc.value)


# ===========================================================================
# Env overlay: live during, restored after
# ===========================================================================

def test_sweep_sets_arm_flag_live_and_restores_env(libv2_course, monkeypatch):
    repo_root, slug, _ = libv2_course
    # Ensure a clean baseline: no managed flag set.
    for env in _MANAGED_ENV:
        monkeypatch.delenv(env, raising=False)
    seen = {}

    def _spy(repo_root, course_slug, query, **kwargs):
        # Record the managed-env snapshot the FIRST time each arm's eval runs a
        # question (keyed by the flags currently on).
        on = tuple(sorted(e for e in _MANAGED_ENV if os.environ.get(e)))
        seen.setdefault(on, 0)
        seen[on] += 1
        return _plain_answer_fn(repo_root, course_slug, query, **kwargs)

    summary = run_flag_sweep(
        repo_root, slug, engine="lexical", answer_fn=_spy,
        arms=[BASELINE_ARM, "hyde", "decompose"],
        with_groundedness=False, write=False,
    )
    # Baseline arm ran with NO managed flag on.
    assert () in seen
    # hyde arm ran with exactly ED4ALL_ANSWER_HYDE on.
    assert ("ED4ALL_ANSWER_HYDE",) in seen
    assert ("ED4ALL_ANSWER_DECOMPOSE",) in seen
    # ENV FULLY RESTORED after the sweep (no managed flag leaked).
    for env in _MANAGED_ENV:
        assert env not in os.environ
    assert summary["baseline_arm"] == BASELINE_ARM


def test_sweep_restores_preexisting_env_value(libv2_course, monkeypatch):
    """A pre-set managed flag is restored to its exact prior value after."""
    repo_root, slug, _ = libv2_course
    monkeypatch.setenv("ED4ALL_ANSWER_HYDE", "preset")
    run_flag_sweep(
        repo_root, slug, engine="lexical", answer_fn=_plain_answer_fn,
        arms=[BASELINE_ARM, "decompose"], with_groundedness=False, write=False,
    )
    # Restored to the exact preset value (the sweep unset it per-arm then put it back).
    assert os.environ["ED4ALL_ANSWER_HYDE"] == "preset"


# ===========================================================================
# Matrix + diff
# ===========================================================================

def test_sweep_matrix_and_diff_shape(libv2_course):
    repo_root, slug, _ = libv2_course
    summary = run_flag_sweep(
        repo_root, slug, engine="lexical", answer_fn=_plain_answer_fn,
        arms=[BASELINE_ARM, "hyde"], with_groundedness=False, write=False,
    )
    # Matrix has a row per pinned metric with a baseline value + per-arm delta.
    assert "answer_rate" in summary["matrix"]
    row = summary["matrix"]["answer_rate"]
    assert row["baseline"] == pytest.approx(1.0)  # all 3 answered
    assert "hyde" in row["arms"]
    # Identical answer_fn → arm == baseline → zero delta, no regression.
    assert row["arms"]["hyde"]["delta"] == pytest.approx(0.0)
    assert summary["regressions"] == []
    # Per-arm diff_vs_baseline present for the flag arm, None for baseline.
    assert summary["arms"][BASELINE_ARM]["diff_vs_baseline"] is None
    assert summary["arms"]["hyde"]["diff_vs_baseline"] is not None
    assert summary["arms"]["hyde"]["diff_vs_baseline"]["exit_code"] == 0


def test_sweep_detects_regression_when_arm_differs(libv2_course):
    """An arm whose answer_fn refuses everything regresses answer_rate."""
    repo_root, slug, _ = libv2_course

    def _arm_aware_fn(repo_root, course_slug, query, **kwargs):
        # The 'hyde' arm turns ED4ALL_ANSWER_HYDE on: make THAT arm refuse all
        # gold questions (a severe, detectable regression). Other arms answer.
        if os.environ.get("ED4ALL_ANSWER_HYDE"):
            return _FakeAnswer("refused_low_confidence", [])
        return _plain_answer_fn(repo_root, course_slug, query, **kwargs)

    summary = run_flag_sweep(
        repo_root, slug, engine="lexical", answer_fn=_arm_aware_fn,
        arms=[BASELINE_ARM, "hyde"], tolerance_pp=5.0,
        with_groundedness=False, write=False,
    )
    # baseline answer_rate 1.0; hyde arm 0.0 → -100pp → regression on answer_rate.
    assert summary["matrix"]["answer_rate"]["arms"]["hyde"]["delta_pp"] == pytest.approx(
        -100.0
    )
    assert {"arm": "hyde", "metric": "answer_rate"} in summary["regressions"]


def test_sweep_flag_config_stamp_records_arm(libv2_course):
    """The eval's own flag_config stamp records the arm's flag as on."""
    repo_root, slug, _ = libv2_course
    summary = run_flag_sweep(
        repo_root, slug, engine="lexical", answer_fn=_plain_answer_fn,
        arms=[BASELINE_ARM, "intent_route"], with_groundedness=False, write=False,
    )
    # We stored per-arm flags_on; intent_route arm names its env var.
    flags_on = summary["arms"]["intent_route"]["flags_on"]
    assert flags_on == {"ED4ALL_ANSWER_INTENT_ROUTE": "1"}
    assert summary["arms"][BASELINE_ARM]["flags_on"] == {}


def test_render_matrix_smoke(libv2_course):
    repo_root, slug, _ = libv2_course
    summary = run_flag_sweep(
        repo_root, slug, engine="lexical", answer_fn=_plain_answer_fn,
        arms=[BASELINE_ARM, "hyde"], with_groundedness=False, write=False,
    )
    text = render_matrix(summary)
    assert "flag sweep" in text
    assert "answer_rate" in text
    assert "no pinned regressions" in text


def test_sweep_writes_per_arm_reports(libv2_course):
    repo_root, slug, course_dir = libv2_course
    out_dir = course_dir / "retrieval_eval"
    summary = run_flag_sweep(
        repo_root, slug, engine="lexical", answer_fn=_plain_answer_fn,
        arms=[BASELINE_ARM, "hyde"], with_groundedness=False, write=True,
        output_dir=out_dir,
    )
    assert (out_dir / "grounded_answer_eval_sweep_baseline.json").exists()
    assert (out_dir / "grounded_answer_eval_sweep_hyde.json").exists()
    assert summary["arms"]["hyde"]["report_path"] is not None


def test_on_value_override_for_rerank(libv2_course):
    """rerank's placeholder on-value can be overridden with a real seat name."""
    repo_root, slug, _ = libv2_course
    seen = {}

    def _spy(repo_root, course_slug, query, **kwargs):
        seen["rerank"] = os.environ.get("ED4ALL_RERANK_PROVIDER")
        return _plain_answer_fn(repo_root, course_slug, query, **kwargs)

    run_flag_sweep(
        repo_root, slug, engine="lexical", answer_fn=_spy,
        arms=[BASELINE_ARM, "rerank"],
        on_value_overrides={"rerank": "local-cross-encoder"},
        with_groundedness=False, write=False,
    )
    assert seen["rerank"] == "local-cross-encoder"
