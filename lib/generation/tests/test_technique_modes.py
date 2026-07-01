"""Tests for the W5 C0..C5 technique-mode resolver (§2)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.generation.technique_modes import (  # noqa: E402
    DEFAULT_MODE,
    MODE_NAMES,
    SELECT_ENTAILMENT_ARGMAX,
    SELECT_GATE_PASS,
    apply_mode_to_env,
    resolve_best_of_n,
    resolve_chunk_scoped,
    resolve_refine_rounds,
    resolve_select_by,
    resolve_self_verify,
    resolve_technique_mode,
)


def test_cumulative_levers():
    expected = {
        "C0": (False, False, False, 0, 1, SELECT_GATE_PASS),
        "C1": (True, False, False, 0, 1, SELECT_GATE_PASS),
        "C2": (True, True, False, 0, 1, SELECT_GATE_PASS),
        "C3": (True, True, True, 0, 1, SELECT_GATE_PASS),
        "C4": (True, True, True, 2, 1, SELECT_GATE_PASS),
        "C5": (True, True, True, 2, 5, SELECT_ENTAILMENT_ARGMAX),
    }
    for name, (cs, ft, sv, rr, bn, sb) in expected.items():
        m = resolve_technique_mode(name)
        assert (m.chunk_scoped, m.free_text_envelope, m.self_verify,
                m.refine_rounds, m.best_of_n, m.select_by) == (cs, ft, sv, rr, bn, sb)


def test_env_resolution(monkeypatch):
    monkeypatch.setenv("ED4ALL_GENERATION_TECHNIQUE", "C2")
    assert resolve_technique_mode().name == "C2"
    monkeypatch.delenv("ED4ALL_GENERATION_TECHNIQUE", raising=False)
    assert resolve_technique_mode().name == DEFAULT_MODE  # C5


def test_garbage_falls_back_to_default():
    assert resolve_technique_mode("nonsense").name == DEFAULT_MODE
    assert resolve_technique_mode("").name == DEFAULT_MODE


def test_best_of_n_override():
    m = resolve_technique_mode("C5", best_of_n=8)
    assert m.best_of_n == 8
    # Override is ignored for non-C5 (best_of_n stays 1).
    assert resolve_technique_mode("C3", best_of_n=8).best_of_n == 1


def test_apply_mode_to_env_projects_knobs():
    env = {}
    apply_mode_to_env(resolve_technique_mode("C5"), env)
    assert env["COURSEFORGE_BEST_OF_N_SELECT_BY"] == SELECT_ENTAILMENT_ARGMAX
    assert env["COURSEFORGE_OUTLINE_N_CANDIDATES"] == "5"
    assert env["COURSEFORGE_BEST_OF_N"] == "5"
    assert env["COURSEFORGE_OUTLINE_GRAMMAR_MODE"] == "none"

    env0 = {}
    apply_mode_to_env(resolve_technique_mode("C0"), env0)
    assert env0["COURSEFORGE_BEST_OF_N_SELECT_BY"] == SELECT_GATE_PASS
    assert env0["COURSEFORGE_OUTLINE_GRAMMAR_MODE"] == "json_object"


def test_per_tier_knob_wins_over_umbrella():
    env = {"COURSEFORGE_REWRITE_N_CANDIDATES": "9"}
    apply_mode_to_env(resolve_technique_mode("C5"), env)
    # The pre-set per-tier rewrite knob is preserved (setdefault).
    assert env["COURSEFORGE_REWRITE_N_CANDIDATES"] == "9"
    assert resolve_best_of_n(env, tier="rewrite") == 9


def test_resolve_select_by_default_gate_pass():
    assert resolve_select_by({}) == SELECT_GATE_PASS
    assert resolve_select_by({"COURSEFORGE_BEST_OF_N_SELECT_BY": "garbage"}) == SELECT_GATE_PASS
    assert resolve_select_by(
        {"COURSEFORGE_BEST_OF_N_SELECT_BY": SELECT_ENTAILMENT_ARGMAX}
    ) == SELECT_ENTAILMENT_ARGMAX


def test_resolve_best_of_n_none_when_unset():
    assert resolve_best_of_n({}) is None
    assert resolve_best_of_n({"COURSEFORGE_BEST_OF_N": "4"}) == 4


def test_mode_names_complete():
    assert MODE_NAMES == ("C0", "C1", "C2", "C3", "C4", "C5")


def test_lever_resolvers_default_off():
    # Unset / blank / false / garbage → OFF (parse-with-fallback).
    for env in ({}, {"COURSEFORGE_CHUNK_SCOPED": ""}, {"COURSEFORGE_CHUNK_SCOPED": "false"},
                {"COURSEFORGE_CHUNK_SCOPED": "nonsense"}):
        assert resolve_chunk_scoped(env) is False
    for env in ({}, {"COURSEFORGE_SELF_VERIFY": "no"}, {"COURSEFORGE_SELF_VERIFY": "x"}):
        assert resolve_self_verify(env) is False
    for env in ({}, {"COURSEFORGE_REFINE_ROUNDS": "0"},
                {"COURSEFORGE_REFINE_ROUNDS": "-4"}, {"COURSEFORGE_REFINE_ROUNDS": "junk"}):
        assert resolve_refine_rounds(env) == 0


def test_lever_resolvers_on():
    assert resolve_chunk_scoped({"COURSEFORGE_CHUNK_SCOPED": "true"}) is True
    assert resolve_self_verify({"COURSEFORGE_SELF_VERIFY": "YES"}) is True
    assert resolve_refine_rounds({"COURSEFORGE_REFINE_ROUNDS": "2"}) == 2


def test_lever_resolvers_consume_projected_env():
    # apply_mode_to_env's projection must round-trip through the resolvers so
    # the C1/C3/C4 levers are live (W8.9 — no longer dead).
    expected = {
        "C0": (False, False, 0),
        "C1": (True, False, 0),
        "C2": (True, False, 0),
        "C3": (True, True, 0),
        "C4": (True, True, 2),
        "C5": (True, True, 2),
    }
    for mode, (cs, sv, rr) in expected.items():
        env = {}
        apply_mode_to_env(resolve_technique_mode(mode), env)
        assert resolve_chunk_scoped(env) is cs, mode
        assert resolve_self_verify(env) is sv, mode
        assert resolve_refine_rounds(env) == rr, mode
