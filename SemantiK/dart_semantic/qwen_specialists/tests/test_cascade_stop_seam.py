"""Graceful-stop SEAMS (a) + (b) in ``run_full_cascade``.

Seam (a) = post-Stage-5e / pre-Stage-6 (the cheapest-loss cooperative
checkpoint: everything upstream is cache-recoverable, Stage 6 is the
~3h/chapter authoring cost). Seam (b) = pre-Stage-13 offline retry.

The cascade pulls axe/playwright at import, so we stub those for IMPORT ONLY
(mirrors ``test_cascade_stage5e`` / ``_stage5d``). The behavioural seam-(a)
test drives the REAL ``run_full_cascade`` with council/arbitrate/structure-
graph mocked and ``run_qwen_specialists`` replaced by a marker-raising recorder
— so a stop at seam (a) short-circuits BEFORE Stage 6 (marker never raised),
and an un-wired run reaches Stage 6 (marker raised), proving the no-op path.
GPU lifecycle is forced OFF so the seam never touches the live ollama/GPU.

Seam (b) runs only after a full fast-lane Stage 6-12 pass, too heavy to drive
on CPU, so it is covered by a source-order guard (the established
``inspect.getsource`` wiring pattern in ``test_verify_refine_loop``).
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest


def _install_axe_stubs() -> None:
    if "axe_playwright_python" not in sys.modules:
        pkg = types.ModuleType("axe_playwright_python")
        sub = types.ModuleType("axe_playwright_python.sync_playwright")
        sub.Axe = object
        pkg.sync_playwright = sub
        sys.modules["axe_playwright_python"] = pkg
        sys.modules["axe_playwright_python.sync_playwright"] = sub
    if "playwright" not in sys.modules:
        pw = types.ModuleType("playwright")
        pw_sync = types.ModuleType("playwright.sync_api")
        pw_sync.sync_playwright = lambda *a, **k: None
        pw.sync_api = pw_sync
        sys.modules["playwright"] = pw
        sys.modules["playwright.sync_api"] = pw_sync


_install_axe_stubs()

from dart_semantic import cascade  # noqa: E402
from dart_semantic.council import base as council_base  # noqa: E402
from dart_semantic.stop_seam import STOP_SENTINEL_ENV, CascadeStopRequested  # noqa: E402
from dart_semantic.structure_graph import Region  # noqa: E402


class _ReachedStage6(Exception):
    """Marker raised by the patched Stage-6 so the test can prove entry."""


def _prime_cascade(monkeypatch):
    """Mock Stages 1-5 + release, force GPU lifecycle OFF, replace Stage 6.

    Returns a one-element ``calls`` list that flips truthy iff Stage 6
    (``run_qwen_specialists``) was entered.
    """
    # Byte-stable, model-free upstream: SEMANTIK_STRUCTURE_CLEAN off (skip the
    # deterministic clean pass), reviewer/resegment/figures off (defaults).
    monkeypatch.setenv("SEMANTIK_STRUCTURE_CLEAN", "0")
    monkeypatch.delenv("SEMANTIK_STRUCTURE_REVIEW", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_DETECT_FIGURES", raising=False)
    # CRITICAL isolation: never let the seam's pre-Stage-6 lease touch the live
    # ollama/GPU. GPU lifecycle OFF -> _gpu_lifecycle_release is a no-op.
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "0")

    regions = [Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "x"})]
    monkeypatch.setattr(cascade, "run_council", lambda _pdf: (object(), [], []))
    monkeypatch.setattr(cascade, "arbitrate", lambda _s, _r: object())
    monkeypatch.setattr(cascade, "build_structure_graph", lambda *a, **k: list(regions))
    # release_council_gpu is imported function-locally from .council.base at BOTH
    # cascade call sites; patch the source module attribute.
    monkeypatch.setattr(council_base, "release_council_gpu", lambda: None)

    calls: list[bool] = []

    def _fake_stage6(*_a, **_k):
        calls.append(True)
        raise _ReachedStage6()

    monkeypatch.setattr(cascade, "run_qwen_specialists", _fake_stage6)
    return calls


def _run(tmp_path):
    return cascade.run_full_cascade(
        tmp_path / "doc.pdf",
        validator=types.SimpleNamespace(),
        runtime_mode="mock",
    )


# --------------------------------------------------------------------------- #
# Seam (a) — behavioural.
# --------------------------------------------------------------------------- #
def test_seam_a_stops_before_stage6(monkeypatch, tmp_path):
    calls = _prime_cascade(monkeypatch)
    sentinel = tmp_path / "STOP_REQUESTED"
    sentinel.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(sentinel))

    with pytest.raises(CascadeStopRequested) as ei:
        _run(tmp_path)
    assert ei.value.site_id == "cascade:post-stage5e-pre-stage6"
    # Stage 6 was NEVER entered.
    assert calls == []


def test_no_sentinel_reaches_stage6(monkeypatch, tmp_path):
    # No wiring -> seam (a) is a byte-identical no-op; Stage 6 runs (marker).
    calls = _prime_cascade(monkeypatch)
    monkeypatch.delenv(STOP_SENTINEL_ENV, raising=False)
    with pytest.raises(_ReachedStage6):
        _run(tmp_path)
    assert calls == [True]


def test_env_set_but_file_absent_reaches_stage6(monkeypatch, tmp_path):
    # Sentinel PATH handed in but the file does not exist -> no-op, Stage 6 runs.
    calls = _prime_cascade(monkeypatch)
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(tmp_path / "STOP_REQUESTED"))
    with pytest.raises(_ReachedStage6):
        _run(tmp_path)
    assert calls == [True]


# --------------------------------------------------------------------------- #
# Source-order guards (both seams) — defend against reordering.
# --------------------------------------------------------------------------- #
def test_seam_a_precedes_stage6_in_source():
    src = inspect.getsource(cascade.run_full_cascade)
    seam = src.index('check_cascade_stop("cascade:post-stage5e-pre-stage6")')
    first_stage6 = src.index("run_qwen_specialists(")
    assert seam < first_stage6, "seam (a) must precede the first Stage-6 call"


def test_seam_b_precedes_offline_retry_in_source():
    src = inspect.getsource(cascade.run_full_cascade)
    seam = src.index('check_cascade_stop("cascade:pre-stage13-offline-retry")')
    offline = src.index("maybe_offline_retry(")
    assert seam < offline, "seam (b) must precede maybe_offline_retry"
