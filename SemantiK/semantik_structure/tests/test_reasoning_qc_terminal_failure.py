"""Terminal-failure LEDGER + honest-page-span regression tests.

Root cause reproduced here: against a persistently-null reasoning-QC endpoint the
fan-out judges EVERY unit's split ladder, records ``_qc_incomplete``, and — before
this fix — NEVER cached that terminal verdict. So a killed run's ``--resume`` (or
any re-drive of the pass) re-POSTed the full expensive ladder for every one of the
(hundreds of) terminally-failed units — an unbounded re-attempt of the same window
family across scheduling rounds. A separate logging defect passed the single
document-``QCWindow``'s representative page (always page 1) into every unit's
split-ladder log, MASKING that distinct sub-slices (later pages) were each failing
once (the "always page 1" 170-failure log).

These tests exercise the fan-out (``_fan_out_page_verifies``) driving the REAL
split ladder (``reasoning_qc_vlm._post_with_retry`` stubbed to always null), and
assert: bounded attempts, one qc_incomplete per unit, pages beyond page 1 all
attempted, linear (never super-linear) attempt growth, a same-run resume that
re-POSTs NOTHING (ledger honoured), a new run that DOES retry (ledger scoped), and
the honest per-unit page span in the split-ladder log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pytest

from semantik_structure import reasoning_qc, reasoning_qc_vlm
from semantik_structure import vlm_extract


# ---------------------------------------------------------------------------
# Shims (mirror test_reasoning_qc_scope.py) — a multi-page, multi-window doc.
# ---------------------------------------------------------------------------
@dataclass
class _Raw:
    text: str
    page: int


@dataclass
class _FB:
    raw: _Raw


@dataclass(frozen=True)
class _Region:
    kind: str
    feature_block_indices: tuple = ()
    payload: dict = field(default_factory=dict)


@dataclass
class _Seat:
    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    model: str = "qc:test"


def _multi_page_windows(n: int):
    """Build the single document QCWindow list. page = 1 + i // 4 (4 blocks/page)."""
    fbs = [_FB(_Raw(f"blk {i}", 1 + i // 4)) for i in range(n)]
    capped = [_Region(kind="paragraph", feature_block_indices=(i,)) for i in range(n)]
    order = list(range(n))
    return reasoning_qc.build_qc_windows(capped, fbs, order)


class _NullPoster:
    """Stub ``_post_with_retry`` — always raises QCNullContentError, records calls.

    Every invocation is exactly ONE endpoint POST (one ladder rung); it records the
    set of block pages the rung saw so a test can assert pages beyond 1 are reached.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.pages_seen: set[int] = set()
        self.per_block_span: list[str] = []

    def __call__(self, **kwargs):
        self.calls += 1
        blocks = kwargs.get("blocks") or []
        pages = [b.get("page") for b in blocks if b.get("page") is not None]
        self.pages_seen.update(pages)
        if pages:
            lo, hi = min(pages), max(pages)
            self.per_block_span.append(str(lo) if lo == hi else f"{lo}-{hi}")
        raise reasoning_qc_vlm.QCNullContentError(
            "simulated null message content", transient=False
        )


@pytest.fixture
def _null_endpoint(monkeypatch):
    poster = _NullPoster()
    monkeypatch.setattr(reasoning_qc_vlm, "_post_with_retry", poster)
    # run_qc_judgment calls _lazy_requests() before _post_with_retry — stub it so no
    # real ``requests`` import / HTTP happens.
    monkeypatch.setattr(vlm_extract, "_lazy_requests", lambda: object())
    return poster


@pytest.fixture(autouse=True)
def _deterministic_partition(tmp_path, monkeypatch):
    # WINDOW=4, SEAM=2 → every unit is exactly _QC_MIN_WINDOW_BLOCKS (4) blocks, so
    # each unit does exactly ONE POST and records qc_incomplete at depth 0 — a clean,
    # deterministic attempt count (== number of units).
    monkeypatch.setenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", "4")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", "2")
    monkeypatch.delenv("SEMANTIK_REASONING_QC_CONCURRENCY", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_MAX_SPLIT_DEPTH", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    monkeypatch.delenv("SEMANTIK_STOP_SENTINEL", raising=False)
    # Isolated cache root; checkpoint default-ON (family + site both unset).
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path / "qc_cache_root"))
    monkeypatch.delenv("SEMANTIK_REASONING_QC_CHECKPOINT", raising=False)
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)


def _fan_out(poster_seat=None):
    windows = _multi_page_windows(12)
    return reasoning_qc._fan_out_page_verifies(
        poster_seat or _Seat(), None, windows, log=lambda _m: None
    )


# 12-block doc, WINDOW=4/SEAM=2 → windows (0,4)(4,8)(8,12) + seams (2,6)(6,10).
_N_UNITS_12 = 5


# ===========================================================================
# (a) bounded attempts + (b) one qc_incomplete per unit + iterator advances.
# ===========================================================================
def test_all_units_attempted_once_and_terminate(_null_endpoint, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-a")
    results = _fan_out()
    # Each of the 5 units POSTed exactly once (≤ rungs; here rungs == 1 at min-size).
    assert _null_endpoint.calls == _N_UNITS_12
    # The document window's stitched verdict records qc_incomplete for EVERY block
    # position, exactly once (a sorted unique list — no unit re-attempted / dropped).
    verdict, _flagged, _div = results[0]
    incomplete = verdict["_qc_incomplete"]
    assert incomplete == sorted(set(incomplete))
    assert incomplete == list(range(12))


# ===========================================================================
# (c) pages beyond page 1 are all attempted (iterator advances past page 1).
# ===========================================================================
def test_pages_beyond_one_are_attempted(_null_endpoint, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-c")
    _fan_out()
    # Blocks span pages 1,2,3 (4 blocks/page). A driver stuck re-attempting "page 1"
    # would only ever see page 1; the advancing iterator reaches 2 and 3.
    assert {1, 2, 3} <= _null_endpoint.pages_seen
    assert _null_endpoint.pages_seen == {1, 2, 3}


# ===========================================================================
# (d) attempt count does not scale super-linearly with the failure count.
# ===========================================================================
def test_attempts_scale_linearly(_null_endpoint, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-d-small")
    reasoning_qc._fan_out_page_verifies(
        _Seat(), None, _multi_page_windows(12), log=lambda _m: None
    )
    small = _null_endpoint.calls  # 5 units
    # Doubling the document (24 blocks → 6 windows + 5 seams = 11 units) roughly
    # doubles the attempts — never squares them.
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-d-big")
    poster_big = _NullPoster()
    monkeypatch.setattr(reasoning_qc_vlm, "_post_with_retry", poster_big)
    reasoning_qc._fan_out_page_verifies(
        _Seat(), None, _multi_page_windows(24), log=lambda _m: None
    )
    big = poster_big.calls  # 11 units
    assert small == 5 and big == 11
    # Linear: ~2× the units → ~2× the POSTs (super-linear would be ~4×+).
    assert big < 3 * small


# ===========================================================================
# THE FIX — a same-run resume re-POSTs NOTHING (terminal-failure ledger honoured).
# ===========================================================================
def test_same_run_resume_does_not_repost_terminal_failures(_null_endpoint, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-resume")
    _fan_out()
    assert _null_endpoint.calls == _N_UNITS_12  # first pass POSTs every unit.
    first = _null_endpoint.calls
    # A resume of the SAME run (same run id, same cache root, same fingerprints):
    # every terminally-failed unit is a run-scoped negative HIT → zero re-POSTs.
    results = _fan_out()
    assert _null_endpoint.calls == first  # NOT one extra POST.
    # And the honoured negative still yields the same honest qc_incomplete verdict.
    verdict, _f, _d = results[0]
    assert verdict["_qc_incomplete"] == list(range(12))


# ===========================================================================
# The ledger is RUN-SCOPED — a genuinely NEW run retries (never permanently
# starved by a stale failure; a later run with a healthy endpoint re-attempts).
# ===========================================================================
def test_new_run_retries_terminal_failures(_null_endpoint, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-one")
    _fan_out()
    assert _null_endpoint.calls == _N_UNITS_12
    # A NEW run id sees the SAME cache path (same fingerprint) but a CROSS-scope
    # negative → treated as a miss → re-attempted.
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-two")
    _fan_out()
    assert _null_endpoint.calls == 2 * _N_UNITS_12


def test_checkpoint_off_reposts_every_pass(monkeypatch):
    # With the resume cache OFF, every pass re-POSTs (no ledger, no reads/writes) —
    # byte-identical to the legacy no-cache path (the escape hatch still works).
    monkeypatch.setenv("SEMANTIK_REASONING_QC_CHECKPOINT", "0")
    poster = _NullPoster()
    monkeypatch.setattr(reasoning_qc_vlm, "_post_with_retry", poster)
    monkeypatch.setattr(vlm_extract, "_lazy_requests", lambda: object())
    _fan_out()
    _fan_out()
    assert poster.calls == 2 * _N_UNITS_12


# ===========================================================================
# Secondary defect — the split-ladder log names the failing unit's REAL page
# span (not the document window's representative "page 1").
# ===========================================================================
def test_split_ladder_log_reports_real_page_span(_null_endpoint, monkeypatch, caplog):
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-log")
    with caplog.at_level(logging.WARNING, logger="semantik_structure.reasoning_qc_vlm"):
        _fan_out()
    exhausted = [
        r.getMessage() for r in caplog.records if "split ladder exhausted" in r.getMessage()
    ]
    assert exhausted, "expected split-ladder-exhausted warnings"
    # The units cover pages 1, 2 and 3 — the log must NOT be uniformly "page 1".
    assert any("page 2" in m for m in exhausted)
    assert any("page 3" in m for m in exhausted)


def test_blocks_page_span_helper():
    span = reasoning_qc_vlm._blocks_page_span
    assert span([{"page": 2}, {"page": 2}], 1) == "2"
    assert span([{"page": 2}, {"page": 3}], 1) == "2-3"
    assert span([{"page": None}], 9) == "9"  # fallback when no block carries a page
    assert span([], 7) == "7"
