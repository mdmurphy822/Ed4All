"""Targeted QC-scope tests (``SEMANTIK_REASONING_QC_SCOPE`` == ``targeted``).

Owner-delegated posture: reasoning-QC is an OUTPUT PROOFREADER. In ``targeted``
scope the unit plan keeps (a) ALL junction seams, (b) windows overlapping
upstream-FLAGGED pages (arranger interventions + structure-review block changes),
(c) a deterministic doc-sha sample of the rest; everything else is skipped with an
honest audit entry. ``full`` (default) stays byte-identical (every window judged).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from semantik_structure import reasoning_qc


@pytest.fixture(autouse=True)
def _isolate_qc_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path / "qc_cache_root"))
    monkeypatch.delenv("SEMANTIK_REASONING_QC_CHECKPOINT", raising=False)
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    monkeypatch.delenv("SEMANTIK_STOP_SENTINEL", raising=False)
    # Deterministic partition: WINDOW=4, SEAM=2 → windows (0,4)(4,8)(8,12),
    # seams (2,6)(6,10) on a 12-block doc; each window is one distinct page.
    monkeypatch.setenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", "4")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", "2")
    monkeypatch.delenv("SEMANTIK_REASONING_QC_CONCURRENCY", raising=False)


# ---------------------------------------------------------------------------
# Shims (mirror test_reasoning_qc.py).
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


class _Assembled:
    def __init__(self, region_provenance):
        self.region_provenance = region_provenance


@dataclass
class _Verdict:
    """Minimal ReviewVerdict-shaped shim for derive_flagged_pages."""

    region_index: int
    kind_before: str
    kind_after: str
    reverted_for_invariant: bool = False
    reverted_for_endpoint_failure: bool = False


def _multi_unit_doc(n=12):
    # page = 1 + i // 4 → blocks 0-3 page 1, 4-7 page 2, 8-11 page 3.
    fbs = [_FB(_Raw(f"blk {i}", 1 + i // 4)) for i in range(n)]
    capped = [_Region(kind="paragraph", feature_block_indices=(i,)) for i in range(n)]
    order = list(range(n))
    return capped, fbs, _Assembled(order), order


@pytest.fixture
def _stub_seat(monkeypatch):
    monkeypatch.setattr(reasoning_qc, "_resolve_qc_seat", lambda: object())
    monkeypatch.setattr(reasoning_qc, "_unload_seat", lambda seat: None)
    monkeypatch.setattr(reasoning_qc, "_build_reasoning_qc_capture", lambda: None)
    monkeypatch.setattr(
        reasoning_qc, "resolve_reasoning_qc_model", lambda default_model=None: "qc:test"
    )


def _record_judged(monkeypatch):
    """Monkeypatch the judgment seam to record which UNITS were judged.

    Returns a list that fills with ``(first_text, last_text)`` per judged unit —
    windows: (blk 0, blk 3)/(blk 4, blk 7)/(blk 8, blk 11); seams: (blk 2, blk 5)/
    (blk 6, blk 9)."""
    judged: list[tuple[str, str]] = []

    def _judge(seat, pdf, page, blocks):
        judged.append((blocks[0]["text"], blocks[-1]["text"]))
        return {}

    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", _judge)
    return judged


_ALL_WINDOWS = {("blk 0", "blk 3"), ("blk 4", "blk 7"), ("blk 8", "blk 11")}
_ALL_SEAMS = {("blk 2", "blk 5"), ("blk 6", "blk 9")}


# ===========================================================================
# Resolvers.
# ===========================================================================
@pytest.mark.parametrize(
    "val,expected",
    [(None, "full"), ("", "full"), ("garbage", "full"), ("full", "full"),
     ("targeted", "targeted"), ("TARGETED", "targeted"), (" Targeted ", "targeted")],
)
def test_resolve_scope(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("SEMANTIK_REASONING_QC_SCOPE", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_REASONING_QC_SCOPE", val)
    assert reasoning_qc.resolve_reasoning_qc_scope() == expected


@pytest.mark.parametrize(
    "val,expected",
    [(None, 15), ("", 15), ("garbage", 15), ("0", 0), ("15", 15), ("100", 100),
     ("-5", 0), ("250", 100)],
)
def test_resolve_sample_pct(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", val)
    assert reasoning_qc.resolve_reasoning_qc_sample_pct() == expected


# ===========================================================================
# full scope — byte-identical (every window + seam judged, no scope_plan).
# ===========================================================================
def test_full_scope_judges_every_unit(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.delenv("SEMANTIK_REASONING_QC_SCOPE", raising=False)  # full
    judged = _record_judged(monkeypatch)

    capped, fbs, assembled, order = _multi_unit_doc(12)
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert new_capped is capped
    assert set(judged) == _ALL_WINDOWS | _ALL_SEAMS  # all 5 units judged
    assert audit["scope"] == "full"
    assert "scope_plan" not in audit  # full carries no skip ledger


# ===========================================================================
# targeted — ALL seams kept even when their windows are skipped.
# ===========================================================================
def test_targeted_keeps_all_seams(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SCOPE", "targeted")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", "0")  # no sample
    judged = _record_judged(monkeypatch)

    capped, fbs, assembled, order = _multi_unit_doc(12)
    # No flagged pages, sample 0 → EVERY window skipped, but ALL seams judged.
    _c, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert set(judged) == _ALL_SEAMS  # exactly the seams, no windows
    assert audit["scope"] == "targeted"
    assert audit["scope_plan"]["skipped_windows"] == 3
    assert audit["scope_plan"]["seams"] == 2
    assert audit["scope_plan"]["kept_windows"] == 0


# ===========================================================================
# targeted — a FLAGGED page's window is kept (arranger audit signal).
# ===========================================================================
def test_targeted_keeps_arranger_flagged_page(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SCOPE", "targeted")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", "0")
    judged = _record_judged(monkeypatch)

    capped, fbs, assembled, order = _multi_unit_doc(12)
    # Arranger struggled on page 2 (heading-sanity intervention) → window (4,8) kept.
    arranger_audit = {
        "page_rows": [
            {"page": 1, "status": "ok", "heading_sanity": 0, "attempts": 1},
            {"page": 2, "status": "ok", "heading_sanity": 3, "attempts": 1},
            {"page": 3, "status": "ok", "heading_sanity": 0, "attempts": 1},
        ]
    }
    _c, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf",
        arranger_audit=arranger_audit,
    )
    # Only page-2 window judged (+ all seams); pages 1/3 windows skipped.
    assert ("blk 4", "blk 7") in judged
    assert ("blk 0", "blk 3") not in judged
    assert ("blk 8", "blk 11") not in judged
    assert _ALL_SEAMS <= set(judged)
    assert audit["scope_plan"]["flagged_pages"] == [2]
    assert audit["scope_plan"]["kept_windows"] == 1


def test_targeted_keeps_failed_and_retry_pages(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SCOPE", "targeted")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", "0")
    judged = _record_judged(monkeypatch)

    capped, fbs, assembled, order = _multi_unit_doc(12)
    arranger_audit = {
        "page_rows": [
            {"page": 1, "status": "arrangement_failed"},   # failed-page fallback
            {"page": 2, "status": "ok", "attempts": 3},    # arrangement retry
            {"page": 3, "status": "ok", "attempts": 1},    # clean → skipped
        ]
    }
    _c, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf",
        arranger_audit=arranger_audit,
    )
    assert ("blk 0", "blk 3") in judged  # page 1 failed
    assert ("blk 4", "blk 7") in judged  # page 2 retried
    assert ("blk 8", "blk 11") not in judged  # page 3 clean
    assert audit["scope_plan"]["flagged_pages"] == [1, 2]


# ===========================================================================
# targeted — a structure-review block CHANGE flags its page.
# ===========================================================================
def test_targeted_keeps_reviewed_block_change_page(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SCOPE", "targeted")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", "0")
    judged = _record_judged(monkeypatch)

    capped, fbs, assembled, order = _multi_unit_doc(12)
    # Region 9 (page 3) had its kind changed; region 1 (page 1) reverted (ignored).
    verdicts = [
        _Verdict(region_index=9, kind_before="paragraph", kind_after="heading"),
        _Verdict(region_index=1, kind_before="paragraph", kind_after="heading",
                 reverted_for_invariant=True),
        _Verdict(region_index=5, kind_before="paragraph", kind_after="paragraph"),  # no change
    ]
    _c, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf",
        review_verdicts=verdicts,
    )
    assert ("blk 8", "blk 11") in judged  # page 3 (region 9 changed)
    assert ("blk 0", "blk 3") not in judged  # page 1 reverted → not flagged
    assert audit["scope_plan"]["flagged_pages"] == [3]


def test_derive_flagged_pages_unit():
    capped, fbs, _a, _o = _multi_unit_doc(12)
    arranger_audit = {"page_rows": [{"page": 2, "coercions": 2, "status": "ok"}]}
    verdicts = [_Verdict(region_index=0, kind_before="p", kind_after="heading")]  # page 1
    pages = reasoning_qc.derive_flagged_pages(
        capped, fbs, arranger_audit=arranger_audit, review_verdicts=verdicts
    )
    assert pages == {1, 2}
    # No signals → empty.
    assert reasoning_qc.derive_flagged_pages(capped, fbs) == set()


# ===========================================================================
# Deterministic sample — identical judged set for the SAME sha + pct.
# ===========================================================================
def test_sample_deterministic_same_sha_pct(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SCOPE", "targeted")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", "50")

    def _run():
        judged = _record_judged(monkeypatch)
        capped, fbs, assembled, order = _multi_unit_doc(12)
        reasoning_qc.run_reasoning_qc(
            capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
        )
        return set(judged)

    first = _run()
    second = _run()
    assert first == second  # doc-sha-seeded sample is stable across runs
    assert _ALL_SEAMS <= first  # seams always in


def test_sample_pct_100_keeps_all_windows(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SCOPE", "targeted")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", "100")
    judged = _record_judged(monkeypatch)

    capped, fbs, assembled, order = _multi_unit_doc(12)
    _c, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert set(judged) == _ALL_WINDOWS | _ALL_SEAMS  # 100% sample → full coverage
    assert audit["scope_plan"]["skipped_windows"] == 0


# ===========================================================================
# Honest audit — skip entries recorded + loud log.
# ===========================================================================
def test_skip_entries_in_audit_and_loud_log(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SCOPE", "targeted")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", "0")
    _record_judged(monkeypatch)
    logs: list[str] = []

    capped, fbs, assembled, order = _multi_unit_doc(12)
    _c, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf",
        log=logs.append,
    )
    plan = audit["scope_plan"]
    assert plan["skipped_windows"] == 3
    assert len(plan["skipped"]) == 3
    assert all(e["skipped"] == "targeted_scope" for e in plan["skipped"])
    assert {tuple(e["range"]) for e in plan["skipped"]} == {(0, 4), (4, 8), (8, 12)}
    # Loud log naming the skipped count.
    assert any("SKIPPED 3 window(s)" in m and "targeted scope" in m for m in logs)


# ===========================================================================
# Unit fingerprint is INDEPENDENT of scope + pct (cached verdicts stay valid).
# ===========================================================================
def test_unit_fingerprint_independent_of_scope(monkeypatch):
    blocks = [{"type": "paragraph", "role": "paragraph", "level": None,
               "page": 1, "text": "hello world"}]
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SCOPE", "full")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", "15")
    a = reasoning_qc._qc_unit_fingerprint(blocks, model="m", kind="window")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SCOPE", "targeted")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SAMPLE_PCT", "80")
    b = reasoning_qc._qc_unit_fingerprint(blocks, model="m", kind="window")
    assert a == b  # scope/pct salt the document audit only, never a unit key
