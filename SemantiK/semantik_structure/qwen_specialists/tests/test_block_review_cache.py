"""Phase 4b — content-hash window-op cache for the full-block reviewer.

The cache memoizes each window's PARSED op-list by a sha256 over the full
window input. It MUST be output-identical whether the cache hits, misses, or
is disabled — pure memoization, never a behavior change. These tests pin:

  - a HIT skips the generate_batch dispatch entirely (and reuses the ops),
  - an input change (model id / flag / record) MISSES and re-dispatches,
  - cache-on vs cache-off produce byte-identical reviewed regions + verdicts,
  - a corrupt entry degrades to a MISS + re-dispatch (never raises),
  - the master flag off => zero cache activity, byte-identical.

The cache dir is redirected to a ``tmp_path`` via ``SEMANTIK_CACHE_DIR`` so
the suite is hermetic and never touches a real cache root.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from semantik_structure.qwen_specialists.reviewer import run_structure_review
from semantik_structure.structure_graph import Region

# Reuse the scripted (no-GPU) fixtures from the reviewer suite.
from semantik_structure.qwen_specialists.tests.test_reviewer import (
    _ScriptedRuntime,
    _code_regions,
    _council_state,
    _fb,
    _low_conf_state,
    _oplist,
)


def _cache_files(tmp_path: Path) -> list[Path]:
    """Every cached op-list file under the redirected cache root."""
    root = tmp_path / "block_review_ops"
    return sorted(root.rglob("*.json")) if root.exists() else []


def _snapshot(regions, verdicts):
    """Byte-comparable (region, verdict) snapshot for the identity contract."""
    return (
        [dataclasses.asdict(r) for r in regions],
        [dataclasses.asdict(v) for v in verdicts],
    )


def _setup_window_env(monkeypatch, tmp_path, *, cache: str | None = None):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "8")  # one window holds both
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))
    if cache is not None:
        monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", cache)
    else:
        monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_CACHE", raising=False)


# ---------------------------------------------------------------------------


def test_cache_hit_skips_dispatch(monkeypatch, tmp_path):
    _setup_window_env(monkeypatch, tmp_path)
    fbs, regions = _code_regions(2)
    state = _low_conf_state(2)
    comp = _oplist([0, 1])  # re-type both code_blocks -> paragraph

    # Run 1 — cold cache: the window MISSES and is dispatched, then stored.
    rt1 = _ScriptedRuntime([comp])
    out1, v1 = run_structure_review(regions, fbs, rt1, council_state=state)
    assert len(rt1.batch_calls) == 1  # miss -> dispatched
    assert [r.kind for r in out1] == ["paragraph", "paragraph"]
    assert len(_cache_files(tmp_path)) == 1  # op-list persisted

    # Run 2 — identical input, FRESH runtime: the window HITS the cache and
    # records ZERO new generate_batch calls; the applied ops are identical.
    rt2 = _ScriptedRuntime([comp])
    out2, v2 = run_structure_review(regions, fbs, rt2, council_state=state)
    assert rt2.batch_calls == []  # HIT -> no dispatch
    assert [r.kind for r in out2] == ["paragraph", "paragraph"]
    assert _snapshot(out1, v1) == _snapshot(out2, v2)


def test_cache_miss_on_input_change(monkeypatch, tmp_path):
    _setup_window_env(monkeypatch, tmp_path)
    fbs, regions = _code_regions(2)
    state = _low_conf_state(2)
    comp = _oplist([0, 1])

    rt1 = _ScriptedRuntime([comp])
    run_structure_review(regions, fbs, rt1, council_state=state)
    assert len(rt1.batch_calls) == 1  # populate the cache

    # (a) A different resolved model id => different key => MISS => dispatch.
    monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW_MODEL", "vendor/other-70b")
    rt2 = _ScriptedRuntime([comp])
    run_structure_review(regions, fbs, rt2, council_state=state)
    assert len(rt2.batch_calls) == 1  # model change -> re-dispatch
    monkeypatch.delenv("SEMANTIK_STRUCTURE_REVIEW_MODEL", raising=False)

    # (b) A prompt-/scope-affecting flag change (edge tokens) => MISS.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", "3")
    rt3 = _ScriptedRuntime([comp])
    run_structure_review(regions, fbs, rt3, council_state=state)
    assert len(rt3.batch_calls) == 1  # flag change -> re-dispatch
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", raising=False)

    # (c) A changed edge record (different council_kind/text) => MISS.
    fbs2 = [_fb("x = compute(y)"), _fb("totally different body content here")]
    regions2 = [
        Region(kind="code_block", feature_block_indices=(0,),
               payload={"text": "x = compute(y)"}),
        Region(kind="code_block", feature_block_indices=(1,),
               payload={"text": "totally different body content here"}),
    ]
    rt4 = _ScriptedRuntime([comp])
    run_structure_review(regions2, fbs2, rt4, council_state=state)
    assert len(rt4.batch_calls) == 1  # different records -> re-dispatch


def test_cache_output_identical_on_vs_off(monkeypatch, tmp_path):
    fbs, regions = _code_regions(2)
    state = _low_conf_state(2)
    comp = _oplist([0, 1])

    # Cache ON (default).
    _setup_window_env(monkeypatch, tmp_path / "on", cache="1")
    rt_on = _ScriptedRuntime([comp])
    out_on, v_on = run_structure_review(regions, fbs, rt_on, council_state=state)

    # Cache OFF — every window dispatches; no reads/writes.
    _setup_window_env(monkeypatch, tmp_path / "off", cache="0")
    rt_off = _ScriptedRuntime([comp])
    out_off, v_off = run_structure_review(regions, fbs, rt_off, council_state=state)

    # Byte-identical reviewed regions + verdicts.
    assert _snapshot(out_on, v_on) == _snapshot(out_off, v_off)
    # Cache OFF wrote nothing.
    assert _cache_files(tmp_path / "off") == []
    # Cache ON persisted the op-list.
    assert len(_cache_files(tmp_path / "on")) == 1


def test_corrupt_cache_entry_redispatches(monkeypatch, tmp_path):
    _setup_window_env(monkeypatch, tmp_path)
    fbs, regions = _code_regions(2)
    state = _low_conf_state(2)
    comp = _oplist([0, 1])

    # Populate the cache, then CORRUPT the stored entry at its computed key.
    rt1 = _ScriptedRuntime([comp])
    out1, v1 = run_structure_review(regions, fbs, rt1, council_state=state)
    files = _cache_files(tmp_path)
    assert len(files) == 1
    files[0].write_text("}{ not json at all <<<")  # garbage entry

    # A corrupt entry degrades to a MISS -> re-dispatch (never raises).
    rt2 = _ScriptedRuntime([comp])
    out2, v2 = run_structure_review(regions, fbs, rt2, council_state=state)
    assert len(rt2.batch_calls) == 1  # corrupt -> re-dispatch
    assert _snapshot(out1, v1) == _snapshot(out2, v2)  # output still correct


def test_flag_off_byte_stable(monkeypatch, tmp_path):
    # Master SEMANTIK_BLOCK_REVIEW OFF -> the windowed path (and thus the whole
    # cache) is never entered: zero reads/writes, byte-identical heading-only
    # output. A would-dispatch low-conf code_block stays unreviewed.
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", "1")  # on, but unreachable
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))

    fbs, regions = _code_regions(2)
    state = _low_conf_state(2)
    rt = _ScriptedRuntime([_oplist([0, 1])])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)

    assert rt.batch_calls == []  # heading-only: content blocks not in scope
    assert out == regions  # byte-stable pass-through
    assert all(v.review_note == "non-heading; not reviewed" for v in verdicts)
    assert _cache_files(tmp_path) == []  # ZERO cache activity
    assert not (tmp_path / "block_review_ops").exists()
