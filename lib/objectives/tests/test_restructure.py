"""Defect F — ``lib/objectives/restructure.py`` unit tests.

Synthetic fixtures ONLY (no course slugs / publisher names / data paths). Locks:
both input shapes (group + flat), CO id-stability, drop-vacuous vs annotate-only,
week-group both modes (TO-membership + ceil-stride), fake-provider refusal, and
the deterministic (no-LLM) TO re-derivation + terminal_id re-point.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from lib.objectives.restructure import (
    RefuseFakeEmbedding,
    RestructureOptions,
    restructure_objectives_doc,
)


class _Resolved:
    kind = "st"


class _StubEmbed:
    """Deterministic one-hot embedder — identical statements → identical vectors.

    Not the poisoning ``fake`` kind (``resolved.kind == "st"``), so the E-merge
    refusal gate lets it through. Vectors are keyed on the statement's leading
    tokens so two near-restatements collide (a merge edge) while distinct skills
    do not.
    """

    resolved = _Resolved()

    def __init__(self, dim: int = 24) -> None:
        self.dim = dim

    def encode_batch(self, texts):
        out = []
        for t in texts:
            key = " ".join(str(t).lower().split()[:4])
            # Stable hash — builtin hash() is PYTHONHASHSEED-randomized, which
            # would make identical-statement collisions (and thus the merge
            # graph) nondeterministic across runs; a real embedder is stable.
            digest = hashlib.sha1(key.encode("utf-8")).digest()
            h = int.from_bytes(digest[:8], "big") % self.dim
            v = np.zeros(self.dim, dtype=float)
            v[h] = 1.0
            out.append(v)
        return np.asarray(out, dtype=float)


class _FakeEmbed:
    class _R:
        kind = "fake"

    resolved = _R()

    def encode_batch(self, texts):  # pragma: no cover - never reached
        return np.zeros((len(texts), 4), dtype=float)


def _chunk(cid, mid, title, pos):
    return {
        "id": cid,
        "source": {"module_id": mid, "module_title": title, "position_in_module": pos},
        "text": f"instructional body for {cid} in {title}",
    }


@pytest.fixture
def modules():
    all_chunks = [
        _chunk("c1", "mod-a", "Chapter A", 0),
        _chunk("c2", "mod-a", "Chapter A", 1),
        _chunk("c3", "mod-b", "Chapter B", 0),
        _chunk("c4", "mod-b", "Chapter B", 1),
    ]
    return all_chunks, {c["id"]: c for c in all_chunks}


def _group_doc():
    return {
        "course_name": "SYN_101",
        "terminal_objectives": [{"id": "TO-99", "statement": "stale"}],
        "chapter_objectives": [
            {"chapter": "Week 1", "objectives": [
                {"id": "CO-01", "statement": "Derive the tangent slope at a point",
                 "bloom_level": "apply", "bloom_verb": "derive", "source_chunk_ids": ["c1"]},
                {"id": "CO-02", "statement": "Derive the tangent slope at a point clearly",
                 "bloom_level": "apply", "bloom_verb": "derive", "source_chunk_ids": ["c2"]},
            ]},
            {"chapter": "Week 2", "objectives": [
                {"id": "CO-03", "statement": "Integrate a rational polynomial expression",
                 "bloom_level": "apply", "bloom_verb": "integrate", "source_chunk_ids": ["c3", "c4"]},
            ]},
        ],
    }


def _flat_doc():
    return {
        "course_name": "SYN_101",
        "chapter_objectives": [
            {"id": "CO-01", "statement": "Derive the tangent slope at a point",
             "bloom_level": "apply", "bloom_verb": "derive", "source_chunk_ids": ["c1"]},
            {"id": "CO-03", "statement": "Integrate a rational polynomial expression",
             "bloom_level": "apply", "bloom_verb": "integrate", "source_chunk_ids": ["c3"]},
        ],
    }


def _opts(**kw):
    kw.setdefault("course_name", "SYN_101")
    kw.setdefault("generated_from", "in.json")
    kw.setdefault("embed", _StubEmbed())
    return RestructureOptions(**kw)


def test_group_shape_produces_book_ordered_terminals(modules):
    all_chunks, cbi = modules
    doc, report = restructure_objectives_doc(_group_doc(), cbi, all_chunks, options=_opts())
    tos = doc["terminal_objectives"]
    assert [t["id"] for t in tos] == ["TO-01", "TO-02"]
    # Book order: mod-a (Chapter A) before mod-b (Chapter B).
    assert "Chapter A" in tos[0]["statement"]
    assert "Chapter B" in tos[1]["statement"]
    # Deterministic mint method / provenance.
    assert doc["mint_method"] == "restructured_objectives"
    assert doc["generated_from"] == "in.json"
    assert doc["objectives_source"] == "operator"
    assert report["a_derivation"]["terminals"] == 2


def test_flat_shape_accepted(modules):
    all_chunks, cbi = modules
    doc, report = restructure_objectives_doc(_flat_doc(), cbi, all_chunks, options=_opts())
    assert report["input_co_count"] == 2
    assert [t["id"] for t in doc["terminal_objectives"]] == ["TO-01", "TO-02"]


def test_co_ids_never_reminted(modules):
    all_chunks, cbi = modules
    doc, _ = restructure_objectives_doc(_group_doc(), cbi, all_chunks, options=_opts())
    surviving_ids = {
        c["id"] for grp in doc["chapter_objectives"] for c in grp["objectives"]
    }
    # Every survivor keeps an ORIGINAL CO id (E merge/drops never mint new ones).
    assert surviving_ids <= {"CO-01", "CO-02", "CO-03"}


def test_e_merge_collapses_near_restatement(modules):
    all_chunks, cbi = modules
    doc, report = restructure_objectives_doc(_group_doc(), cbi, all_chunks, options=_opts())
    # CO-01 / CO-02 are near-restatements → one survives, the other recorded.
    assert report["e_merge"]["cos_dropped"] >= 1
    assert report["e_merge"]["losers"]
    loser = report["e_merge"]["losers"][0]
    assert loser["kept_co_id"] != loser["dropped_co_id"]
    # The survivor absorbs the loser's cited chunk (grounding preserved).
    kept = next(
        c for grp in doc["chapter_objectives"] for c in grp["objectives"]
        if c["id"] == loser["kept_co_id"]
    )
    assert "c1" in kept["source_chunk_ids"] and "c2" in kept["source_chunk_ids"]


def test_terminal_id_repointed_on_every_co(modules):
    all_chunks, cbi = modules
    doc, _ = restructure_objectives_doc(_group_doc(), cbi, all_chunks, options=_opts())
    to_ids = {t["id"] for t in doc["terminal_objectives"]}
    for grp in doc["chapter_objectives"]:
        for co in grp["objectives"]:
            assert co["terminal_id"] in to_ids


def test_drop_vacuous_removes_v1_fails(modules):
    all_chunks, cbi = modules
    doc = _group_doc()
    doc["chapter_objectives"][0]["objectives"].append(
        {"id": "CO-09", "statement": "Apply various techniques appropriately",
         "bloom_level": "apply", "bloom_verb": "apply", "source_chunk_ids": ["c1"]}
    )
    out, report = restructure_objectives_doc(
        doc, cbi, all_chunks, options=_opts(drop_vacuous=True)
    )
    assert report["b_vacuity"]["mode"] == "drop_vacuous"
    ids = {c["id"] for grp in out["chapter_objectives"] for c in grp["objectives"]}
    assert "CO-09" not in ids  # vacuous CO removed


def test_annotate_only_keeps_vacuous_flagged(modules):
    all_chunks, cbi = modules
    doc = _group_doc()
    doc["chapter_objectives"][0]["objectives"].append(
        {"id": "CO-09", "statement": "Apply various techniques appropriately",
         "bloom_level": "apply", "bloom_verb": "apply", "source_chunk_ids": ["c3"]}
    )
    out, report = restructure_objectives_doc(
        doc, cbi, all_chunks, options=_opts(drop_vacuous=False)
    )
    assert report["b_vacuity"]["mode"] == "annotate_only"
    assert not report["b_vacuity"]["dropped_ids"]
    flags = {
        c["id"]: c.get("vacuous")
        for grp in out["chapter_objectives"] for c in grp["objectives"]
    }
    assert flags.get("CO-09") is True


def test_week_groups_ceil_stride_small_book(modules):
    all_chunks, cbi = modules
    doc, report = restructure_objectives_doc(_group_doc(), cbi, all_chunks, options=_opts())
    # 2 TOs → weeks = max(8, 2) = 8 ≠ num_tos → ceil-stride.
    assert report["duration_weeks"] == 8
    assert report["week_mode"] == "ceil_stride"
    assert len(doc["chapter_objectives"]) == 8


def test_week_groups_to_membership_when_weeks_equal_tos(modules):
    all_chunks, cbi = modules
    doc, report = restructure_objectives_doc(
        _group_doc(), cbi, all_chunks, options=_opts(weeks=2)
    )
    assert report["duration_weeks"] == 2
    assert report["week_mode"] == "to_membership"
    # Week N holds exactly TO-N's child COs.
    to_ids = [t["id"] for t in doc["terminal_objectives"]]
    for idx, grp in enumerate(doc["chapter_objectives"]):
        for co in grp["objectives"]:
            assert co["terminal_id"] == to_ids[idx]


def test_refuses_fake_provider(modules):
    all_chunks, cbi = modules
    with pytest.raises(RefuseFakeEmbedding):
        restructure_objectives_doc(
            _group_doc(), cbi, all_chunks, options=_opts(embed=_FakeEmbed())
        )


def test_no_module_signal_degrades_to_single_terminal():
    # Empty chunkset → deterministic single-terminal degrade (never crashes).
    doc = _flat_doc()
    out, report = restructure_objectives_doc(doc, {}, [], options=_opts())
    assert report["a_derivation"].get("degraded_reason") == "no_module_signal"
    assert len(out["terminal_objectives"]) == 1
    assert out["terminal_objectives"][0]["id"] == "TO-01"


def test_stop_sentinel_pauses_restructure(modules, tmp_path, monkeypatch):
    """A pending stop sentinel raises GracefulStopRequested at a per-CO boundary.

    LLM-free → no sidecar; the operator just re-runs from the immutable input.
    """
    from lib.generation import stop_control
    from lib.generation.stop_control import GracefulStopRequested

    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "WF-RESTRUCTURE-TEST")
    stop_control.request_stop(run_id="WF-RESTRUCTURE-TEST", reason="test", source="test")

    all_chunks, cbi = modules
    with pytest.raises(GracefulStopRequested):
        restructure_objectives_doc(
            _group_doc(), cbi, all_chunks,
            options=_opts(run_id="WF-RESTRUCTURE-TEST"),
        )


def test_learning_outcomes_flat_list_shape(modules):
    all_chunks, cbi = modules
    doc, _ = restructure_objectives_doc(_group_doc(), cbi, all_chunks, options=_opts())
    los = doc["learning_outcomes"]
    terminals = [lo for lo in los if lo["hierarchy_level"] == "terminal"]
    chapters = [lo for lo in los if lo["hierarchy_level"] == "chapter"]
    assert len(terminals) == len(doc["terminal_objectives"])
    assert chapters  # at least one CO present
    # Terminals precede chapters in the flat list.
    assert los[0]["hierarchy_level"] == "terminal"
