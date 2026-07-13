"""Task-#36 follow-up — assert the cascade threads the REAL Stage-5d review
artifacts into the Stage-9b reasoning-QC pass.

The Stage-9b seam (``cascade.run_full_cascade`` ~L2144) calls
``run_reasoning_qc(..., review_runtime=qc_review_runtime,
review_verdicts=review_verdicts, ...)``. When Stage-5d ran
(``SEMANTIK_STRUCTURE_REVIEW`` on) ``qc_review_runtime`` MUST be the exact
endpoint seat the Stage-5d reviewer built (``make_runtime('endpoint',
model=resolve_structure_review_model())`` ~L1540) and ``review_verdicts`` MUST
be the verdict list ``run_structure_review`` returned — NOT ``None``/defaults,
and NOT a freshly-built fallback seat.

This file drives the REAL ``run_full_cascade`` to completion in ``mock`` runtime
with the heavy upstream (council / arbitrate / structure-graph) mocked out (the
harness ``test_reasoning_qc_cascade`` uses), and additionally mocks:

  * ``qwen_specialists.runtime.make_runtime`` — returns a SENTINEL runtime,
    recording every ``(args, kwargs)`` so we can prove exactly one endpoint
    seat is built when Stage-5d runs (the QC seam reuses it, builds no second).
  * ``qwen_specialists.reviewer.run_structure_review`` — returns the regions
    UNCHANGED (so the C2 cap-safety assert passes and ``review_verdicts`` is
    NOT nulled) plus a sentinel verdicts list; records the runtime it received.
  * ``reasoning_qc.run_reasoning_qc`` — a SPY that records the ``review_runtime``
    / ``review_verdicts`` it was threaded (positional + kwargs) and returns a
    well-formed no-op audit so the cascade's rebind + ``result['reasoning_qc']``
    stay valid. This is the minimal QC-seat mock: no seat resolution, no HTTP,
    no VLM/text-model call is ever reached.

FINDING (2026-07-12, re-read at write time): the threading task #36 tracks is
IMPLEMENTED — expect PASSING tests, not xfail.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Import scaffold — stub axe_playwright_python + playwright for IMPORT ONLY if
# the slim venv lacks them (verbatim from test_reasoning_qc_cascade).
# ---------------------------------------------------------------------------
def _install_axe_stubs() -> None:
    if "axe_playwright_python" not in sys.modules:
        try:
            import axe_playwright_python  # noqa: F401
        except Exception:  # noqa: BLE001 — absent → stub for import only
            pkg = types.ModuleType("axe_playwright_python")
            sub = types.ModuleType("axe_playwright_python.sync_playwright")
            sub.Axe = object
            pkg.sync_playwright = sub
            sys.modules["axe_playwright_python"] = pkg
            sys.modules["axe_playwright_python.sync_playwright"] = sub
    if "playwright" not in sys.modules:
        try:
            import playwright  # noqa: F401
        except Exception:  # noqa: BLE001
            pw = types.ModuleType("playwright")
            pw_sync = types.ModuleType("playwright.sync_api")
            pw_sync.sync_playwright = lambda *a, **k: None
            pw.sync_api = pw_sync
            sys.modules["playwright"] = pw
            sys.modules["playwright.sync_api"] = pw_sync


_install_axe_stubs()

from semantik_structure import cascade  # noqa: E402
from semantik_structure import reasoning_qc  # noqa: E402
from semantik_structure.council import base as council_base  # noqa: E402
from semantik_structure.council.types import CouncilState  # noqa: E402
from semantik_structure.qwen_specialists import reviewer as reviewer_mod  # noqa: E402
from semantik_structure.qwen_specialists import runtime as runtime_mod  # noqa: E402
from semantik_structure.qwen_specialists.reviewer import ReviewVerdict  # noqa: E402
from semantik_structure.qwen_specialists.runtime import (  # noqa: E402
    resolve_structure_review_model,
)
from semantik_structure.structure_graph import Region  # noqa: E402
from semantik_structure.types import FeatureBlock, RawBlock  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_qc_cache(tmp_path, monkeypatch):
    """Isolate the default-ON per-unit reasoning-QC resume cache into a per-test
    tmp dir (verbatim from test_reasoning_qc_cascade)."""
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path / "qc_cache_root"))
    monkeypatch.delenv("SEMANTIK_REASONING_QC_CHECKPOINT", raising=False)
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)


# ---------------------------------------------------------------------------
# Synthetic document (2 FBs on one physical page).
# ---------------------------------------------------------------------------
def _fb(text: str, page: int = 1) -> FeatureBlock:
    return FeatureBlock(
        raw=RawBlock(
            text=text,
            page=page,
            bbox=(0.0, 0.0, 10.0, 10.0),
            page_width=100.0,
            page_height=100.0,
        ),
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
    )


def _regions_and_fbs():
    fbs = [
        _fb("Chapter 1 Whole Numbers", page=1),
        _fb("To add whole numbers, line up the columns and add.", page=1),
    ]
    regions = [
        Region(
            kind="heading",
            feature_block_indices=(0,),
            payload={"text": "Chapter 1 Whole Numbers", "level_hint": 1},
        ),
        Region(
            kind="paragraph",
            feature_block_indices=(1,),
            payload={"text": "To add whole numbers, line up the columns and add."},
        ),
    ]
    return regions, fbs


def _prime_cascade_stage5d_on(monkeypatch):
    """Mock Stages 1-5 (council/arbitrate/structure-graph) + council release,
    force GPU lifecycle OFF, disable the optional upstream passes — EXCEPT flip
    SEMANTIK_STRUCTURE_REVIEW *on* so the Stage-5d branch runs. Stages 6-13 (mock
    authoring, real assembler/gate/theta) then run for real."""
    monkeypatch.setenv("SEMANTIK_STRUCTURE_CLEAN", "0")
    monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW", "on")  # <-- Stage-5d ON
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_DETECT_FIGURES", raising=False)
    monkeypatch.delenv("SEMANTIK_SECOND_PASS", raising=False)
    monkeypatch.delenv("SEMANTIK_OCR_CONFUSABLE_REPAIR", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "0")

    regions, fbs = _regions_and_fbs()
    monkeypatch.setattr(
        cascade, "run_council", lambda _pdf: (CouncilState(outputs={}), [], fbs)
    )
    monkeypatch.setattr(cascade, "arbitrate", lambda _s, _r: object())
    monkeypatch.setattr(
        cascade, "build_structure_graph", lambda *a, **k: list(regions)
    )
    monkeypatch.setattr(council_base, "release_council_gpu", lambda: None)


def _prime_cascade_stage5d_off(monkeypatch):
    """Same harness but Stage-5d OFF (SEMANTIK_STRUCTURE_REVIEW unset)."""
    monkeypatch.setenv("SEMANTIK_STRUCTURE_CLEAN", "0")
    monkeypatch.delenv("SEMANTIK_STRUCTURE_REVIEW", raising=False)  # <-- OFF
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_DETECT_FIGURES", raising=False)
    monkeypatch.delenv("SEMANTIK_SECOND_PASS", raising=False)
    monkeypatch.delenv("SEMANTIK_OCR_CONFUSABLE_REPAIR", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "0")

    regions, fbs = _regions_and_fbs()
    monkeypatch.setattr(
        cascade, "run_council", lambda _pdf: (CouncilState(outputs={}), [], fbs)
    )
    monkeypatch.setattr(cascade, "arbitrate", lambda _s, _r: object())
    monkeypatch.setattr(
        cascade, "build_structure_graph", lambda *a, **k: list(regions)
    )
    monkeypatch.setattr(council_base, "release_council_gpu", lambda: None)


def _mock_make_runtime(monkeypatch):
    """Patch ``qwen_specialists.runtime.make_runtime`` (cascade imports it
    function-level, so module-attr patching binds at call time) → a SENTINEL,
    recording every ``(args, kwargs)``. Returns ``(sentinel_runtime,
    make_runtime_calls)``."""
    sentinel_runtime = object()
    make_runtime_calls: list[tuple] = []

    def _fake_make_runtime(*args, **kwargs):
        make_runtime_calls.append((args, dict(kwargs)))
        return sentinel_runtime

    monkeypatch.setattr(runtime_mod, "make_runtime", _fake_make_runtime)
    return sentinel_runtime, make_runtime_calls


def _mock_run_structure_review(monkeypatch, verdicts):
    """Patch ``qwen_specialists.reviewer.run_structure_review`` → returns regions
    UNCHANGED (cap-safety passes, review_verdicts is NOT nulled) + ``verdicts``;
    records the runtime it received. Returns ``review_seen_runtime`` (a list)."""
    review_seen_runtime: list = []

    def _fake_run_structure_review(regions, fbs, runtime, **kwargs):
        review_seen_runtime.append(runtime)
        # regions unchanged → the FB multiset survives the cap identically →
        # the C2 cap-safety assert passes → review_verdicts stays non-None.
        return list(regions), verdicts

    monkeypatch.setattr(
        reviewer_mod, "run_structure_review", _fake_run_structure_review
    )
    return review_seen_runtime


def _spy_reasoning_qc(monkeypatch):
    """Patch ``reasoning_qc.run_reasoning_qc`` (cascade imports it
    function-level) → SPY recording positional args + the review_runtime /
    review_verdicts kwargs, returning a well-formed no-op audit so the cascade's
    rebind + ``result['reasoning_qc']`` stay valid. No seat resolution / HTTP /
    VLM is ever reached."""
    qc_calls: list[dict] = []

    def _fake_run_reasoning_qc(*args, **kwargs):
        qc_calls.append(
            {
                "args": args,
                "review_runtime": kwargs.get("review_runtime"),
                "review_verdicts": kwargs.get("review_verdicts"),
            }
        )
        capped = args[0]  # positional: the converged region list
        audit = {
            "schema": reasoning_qc.QC_AUDIT_SCHEMA,
            "ran": True,
            "mode": reasoning_qc.resolve_reasoning_qc_mode(),
            "model": "qc:test",
            "retype": {"applied": 0, "proposed": 0},
            "merge_move": {"applied": 0, "proposed": 0},
        }
        # Byte-identical shadow return: capped unchanged, verdicts passed through.
        return capped, kwargs.get("review_verdicts"), audit

    monkeypatch.setattr(reasoning_qc, "run_reasoning_qc", _fake_run_reasoning_qc)
    return qc_calls


def _run(monkeypatch, tmp_path):
    return cascade.run_full_cascade(
        tmp_path / "doc.pdf",
        validator=SimpleNamespace(),
        runtime_mode="mock",
        return_html=True,
    )


# 7-positional-arg ReviewVerdict shape (per test_cascade_stage5d.py):
# (block_id, verdict, kind_before, kind_after, level_before, level_after, note).
def _verdicts():
    return [ReviewVerdict(0, "ok", "heading", "heading", 1, 1, "")]


# ---------------------------------------------------------------------------
# Test 1 — the task-#36 assertion: cascade threads the REAL Stage-5d artifacts.
# ---------------------------------------------------------------------------
def test_cascade_threads_real_stage5d_runtime_and_verdicts(monkeypatch, tmp_path):
    _prime_cascade_stage5d_on(monkeypatch)
    sentinel_runtime, make_runtime_calls = _mock_make_runtime(monkeypatch)
    verdicts = _verdicts()
    review_seen_runtime = _mock_run_structure_review(monkeypatch, verdicts)
    qc_calls = _spy_reasoning_qc(monkeypatch)
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")

    _run(monkeypatch, tmp_path)

    # The QC seam fired exactly once.
    assert len(qc_calls) == 1, f"expected one reasoning-QC call, got {len(qc_calls)}"
    call = qc_calls[0]

    # (1) The REAL Stage-5d endpoint seat is threaded through — identity, not a
    # freshly-built fallback, not None.
    assert call["review_runtime"] is sentinel_runtime, (
        "reasoning-QC did NOT receive the real Stage-5d review_runtime "
        "(task #36 regression)"
    )
    assert call["review_runtime"] is not None

    # (2) The REAL verdicts list run_structure_review returned is threaded through.
    assert call["review_verdicts"] is verdicts, (
        "reasoning-QC did NOT receive the real Stage-5d review_verdicts"
    )

    # (3) The Stage-5d reviewer itself received that SAME single seat.
    assert review_seen_runtime == [sentinel_runtime]

    # (4) Exactly ONE endpoint seat was built (by Stage-5d); the QC seam reused
    # it and did NOT build a second/fallback seat (shadow path).
    endpoint_calls = [
        c for c in make_runtime_calls if c[0] and c[0][0] == "endpoint"
    ]
    assert len(endpoint_calls) == 1, (
        f"expected exactly one endpoint make_runtime call, got {make_runtime_calls}"
    )
    assert endpoint_calls[0][1].get("model") == resolve_structure_review_model()


# ---------------------------------------------------------------------------
# Test 2 — contract pin: shadow WITHOUT Stage-5d threads None (builds no seat).
# ---------------------------------------------------------------------------
def test_shadow_without_stage5d_threads_none(monkeypatch, tmp_path):
    _prime_cascade_stage5d_off(monkeypatch)
    _sentinel, make_runtime_calls = _mock_make_runtime(monkeypatch)
    # run_structure_review is NOT patched — Stage-5d is off, so it never runs.
    qc_calls = _spy_reasoning_qc(monkeypatch)
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")

    _run(monkeypatch, tmp_path)

    assert len(qc_calls) == 1
    call = qc_calls[0]
    # Stage-5d did not run → no reviewer seat, and shadow builds no fallback seat.
    assert call["review_runtime"] is None
    assert call["review_verdicts"] is None
    endpoint_calls = [
        c for c in make_runtime_calls if c[0] and c[0][0] == "endpoint"
    ]
    assert endpoint_calls == [], (
        f"shadow mode without Stage-5d must build no endpoint seat, got {make_runtime_calls}"
    )


# ---------------------------------------------------------------------------
# Test 3 — documented ladder: on-mode WITHOUT Stage-5d builds a fallback seat.
# ---------------------------------------------------------------------------
def test_on_mode_without_stage5d_builds_fallback_endpoint_seat(monkeypatch, tmp_path):
    _prime_cascade_stage5d_off(monkeypatch)
    sentinel_runtime, make_runtime_calls = _mock_make_runtime(monkeypatch)
    qc_calls = _spy_reasoning_qc(monkeypatch)
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")

    _run(monkeypatch, tmp_path)

    assert len(qc_calls) == 1
    call = qc_calls[0]
    # on-mode with no Stage-5d seat → the QC seam builds its OWN fallback
    # endpoint seat (reviewer model pin) so re-typing can apply, but there are
    # no Stage-5d verdicts to thread.
    endpoint_calls = [
        c for c in make_runtime_calls if c[0] and c[0][0] == "endpoint"
    ]
    assert len(endpoint_calls) == 1, (
        f"on-mode without Stage-5d must build exactly one fallback seat, got {make_runtime_calls}"
    )
    assert endpoint_calls[0][1].get("model") == resolve_structure_review_model()
    assert call["review_runtime"] is sentinel_runtime
    assert call["review_verdicts"] is None
