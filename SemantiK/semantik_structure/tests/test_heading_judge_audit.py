"""Deterministic heading-judge AUDIT tests (pure — NO GPU / no seat / no LLM).

Covers all three arms + report shape + the standalone entry + the byte-identical
guarantees:

* Arm A — residual pending → INCOMPLETE; fully judged → clean.
* Arm B — all-one-level → collapse; 0-applied-while-N-pending → flag; healthy
  dist → clean; a tiny chapter under the min-headings floor → never collapse.
* Arm C — same normalized signature at different levels across 2 chapters →
  REPORTED (report-only), and the input region_provenance is UNCHANGED.
* Report schema shape + version; the flagged_chapters union (Arm A + Arm B, not
  Arm C); the standalone entry writes heading_judge_audit.json + prints a loud
  summary; audit is a pure read (no mutation, no new sidecar).

No campaign / course / book references anywhere.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from semantik_structure.glmocr import heading_judge_audit as audit
from semantik_structure.glmocr import heading_judge_audit_standalone as standalone


# ── Fixtures. ────────────────────────────────────────────────────────────────
def _heading(text, level, *, pending=False, judged=False, rid=0):
    r = {
        "region_kind": "heading",
        "heading_text": text,
        "level": level,
        "first_raw_block_index": rid,
        "source_page": 1,
    }
    if pending:
        r["heading_level_pending"] = True
    if judged:
        r["heading_level_judged"] = {"from": 3, "to": level, "clamped": False}
    return r


def _healthy_chapter(stem="ch01"):
    """A well-judged chapter: L1 opener, L2 sections, L3/L4 judged children."""
    return {
        "stem": stem,
        "region_provenance": [
            _heading("Chapter 1", 1, rid=0),
            _heading("1.1 Introduction", 2, rid=1),
            _heading("Use Place Value", 3, judged=True, rid=2),
            {"region_kind": "paragraph", "raw_text": "body", "first_raw_block_index": 3},
            _heading("1.2 Operations", 2, rid=4),
            _heading("Add and Subtract", 3, judged=True, rid=5),
            _heading("Review", 4, judged=True, rid=6),
        ],
    }


# ── Arm A — completeness. ────────────────────────────────────────────────────
def test_arm_a_residual_pending_flags_incomplete():
    ch = {
        "stem": "ch02",
        "region_provenance": [
            _heading("Chapter 2", 1, rid=0),
            _heading("2.1 Section", 2, rid=1),
            _heading("Judged child", 3, judged=True, rid=2),
            _heading("STILL pending", 3, pending=True, rid=3),
        ],
    }
    row = audit.build_chapter_audit(ch["stem"], ch["region_provenance"])
    codes = {f["code"] for f in row["flags"]}
    assert audit.FLAG_INCOMPLETE in codes
    assert row["n_residual_pending"] == 1


def test_arm_a_fully_judged_is_clean():
    ch = _healthy_chapter()
    row = audit.build_chapter_audit(ch["stem"], ch["region_provenance"])
    assert row["n_residual_pending"] == 0
    assert audit.FLAG_INCOMPLETE not in {f["code"] for f in row["flags"]}


# ── Arm B — bucket-collapse. ─────────────────────────────────────────────────
def test_arm_b_all_one_level_collapses():
    prov = [_heading(f"Title {i}", 3, judged=True, rid=i) for i in range(6)]
    row = audit.build_chapter_audit("ch", prov)
    codes = {f["code"] for f in row["flags"]}
    assert audit.FLAG_COLLAPSE_SINGLE_LEVEL in codes


def test_arm_b_zero_applied_while_pending_flags():
    # 5 pending headings, none judged → the judge changed nothing.
    prov = [_heading(f"Pending {i}", 3, pending=True, rid=i) for i in range(5)]
    row = audit.build_chapter_audit("ch", prov)
    codes = {f["code"] for f in row["flags"]}
    assert audit.FLAG_ZERO_APPLIED in codes
    # ...and it's incomplete too (Arm A), by construction.
    assert audit.FLAG_INCOMPLETE in codes


def test_arm_b_healthy_distribution_is_clean():
    ch = _healthy_chapter()
    row = audit.build_chapter_audit(ch["stem"], ch["region_provenance"])
    codes = {f["code"] for f in row["flags"]}
    assert not (codes & {
        audit.FLAG_COLLAPSE_SINGLE_LEVEL,
        audit.FLAG_COLLAPSE_SHARE,
        audit.FLAG_ZERO_APPLIED,
    })


def test_arm_b_tiny_chapter_never_collapses():
    # 2 headings both at level 3 — under the min-headings floor → NOT collapsed.
    prov = [_heading("A", 3, judged=True, rid=0), _heading("B", 3, judged=True, rid=1)]
    row = audit.build_chapter_audit("ch", prov)
    codes = {f["code"] for f in row["flags"]}
    assert audit.FLAG_COLLAPSE_SINGLE_LEVEL not in codes
    assert audit.FLAG_COLLAPSE_SHARE not in codes


def test_arm_b_share_collapse(monkeypatch):
    # 19 of 20 at level 3 → 0.95 share → collapse_share (distinct levels > 1).
    prov = [_heading(f"T{i}", 3, judged=True, rid=i) for i in range(19)]
    prov.append(_heading("2.1 Section", 2, rid=99))
    row = audit.build_chapter_audit("ch", prov)
    codes = {f["code"] for f in row["flags"]}
    assert audit.FLAG_COLLAPSE_SHARE in codes
    assert audit.FLAG_COLLAPSE_SINGLE_LEVEL not in codes


def test_min_headings_env_override(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_AUDIT_MIN_HEADINGS", "10")
    prov = [_heading(f"T{i}", 3, judged=True, rid=i) for i in range(6)]
    row = audit.build_chapter_audit("ch", prov)
    # 6 headings < floor 10 → never collapsed.
    assert audit.FLAG_COLLAPSE_SINGLE_LEVEL not in {f["code"] for f in row["flags"]}


def test_collapse_share_env_parse_with_fallback(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_AUDIT_COLLAPSE_SHARE", "banana")
    assert audit.resolve_audit_collapse_share() == pytest.approx(0.95)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_AUDIT_COLLAPSE_SHARE", "2.0")
    assert audit.resolve_audit_collapse_share() == pytest.approx(0.95)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_AUDIT_COLLAPSE_SHARE", "0.5")
    assert audit.resolve_audit_collapse_share() == pytest.approx(0.5)


# ── Arm C — cross-chapter consistency (report-only). ─────────────────────────
def test_arm_c_same_signature_different_level_reported_no_mutation():
    ch1 = {
        "stem": "ch01",
        "region_provenance": [
            _heading("1.1 Fractions", 2, rid=0),
            _heading("Summary", 3, judged=True, rid=1),   # in-section summary
        ],
    }
    ch2 = {
        "stem": "ch02",
        "region_provenance": [
            _heading("2.1 Decimals", 2, rid=0),
            _heading("Summary", 4, judged=True, rid=1),   # chapter-end summary
        ],
    }
    before = copy.deepcopy([ch1, ch2])
    inconsistent = audit.cross_chapter_signatures([ch1, ch2])
    sigs = {row["signature"] for row in inconsistent}
    assert "summary" in sigs
    row = next(r for r in inconsistent if r["signature"] == "summary")
    assert row["levels"] == [3, 4]
    # parent CONTEXT annotated (the depth caveat)
    assert all("parent" in occ for occ in row["occurrences"])
    # REPORT-ONLY: the input region_provenance is byte-unchanged.
    assert [ch1, ch2] == before


def test_arm_c_numbering_normalized_but_distinct_titles_not_merged():
    assert audit.normalize_signature("1.1 Introduction") == "introduction"
    assert audit.normalize_signature("2.1 Introduction") == "introduction"
    assert audit.normalize_signature("Summary") == "summary"
    # a bare-number heading keeps its text (never empties)
    assert audit.normalize_signature("1.2") == "1.2"


# ── Report schema + flagged union. ───────────────────────────────────────────
def test_report_schema_shape_and_version():
    ch_flagged = {
        "stem": "bad",
        "region_provenance": [
            _heading(f"T{i}", 3, pending=True, rid=i) for i in range(6)
        ],
    }
    report = audit.build_audit_report([_healthy_chapter("good"), ch_flagged])
    assert report["audit_schema_version"] == audit.AUDIT_SCHEMA_VERSION
    assert set(report) >= {
        "audit_schema_version", "thresholds", "chapters", "book",
        "flagged_chapters",
    }
    assert set(report["book"]) >= {
        "level_distribution", "incomplete_chapters", "collapsed_chapters",
        "inconsistent_signatures",
    }
    for cr in report["chapters"]:
        assert set(cr) >= {
            "stem", "n_headings", "n_residual_pending", "level_distribution",
            "flags",
        }
    # flagged = Arm A + Arm B union; the healthy chapter is NOT flagged.
    assert report["flagged_chapters"] == ["bad"]


def test_arm_c_never_contributes_to_flagged():
    # two clean chapters that only DIFFER in Arm-C signature levels → NOT flagged.
    ch1 = {"stem": "a", "region_provenance": [
        _heading("1.1 X", 2, rid=0), _heading("Notes", 3, judged=True, rid=1)]}
    ch2 = {"stem": "b", "region_provenance": [
        _heading("2.1 Y", 2, rid=0), _heading("Notes", 4, judged=True, rid=1)]}
    report = audit.build_audit_report([ch1, ch2])
    assert report["flagged_chapters"] == []
    assert report["book"]["inconsistent_signatures"]  # but Arm C still reports


# ── Standalone entry (report-only, no mutation, writes the report). ──────────
def _write_corrected(dir_path: Path, stem: str, prov):
    (dir_path / f"{stem}.corrected_layout.json").write_text(
        json.dumps({"region_provenance": prov, "heading_tree": []}),
        encoding="utf-8",
    )


def test_standalone_writes_report_and_does_not_mutate(tmp_path, capsys):
    d = tmp_path / "judged"
    d.mkdir()
    good = _healthy_chapter("good")["region_provenance"]
    bad = [_heading(f"T{i}", 3, pending=True, rid=i) for i in range(6)]
    _write_corrected(d, "good", good)
    _write_corrected(d, "bad", bad)
    before = {p.name: p.read_bytes() for p in d.glob("*.json")}

    out = tmp_path / "out"
    report = standalone.run_audit([d], out_dir=out)
    standalone._print_summary(report)

    report_file = out / "heading_judge_audit.json"
    assert report_file.is_file()
    loaded = json.loads(report_file.read_text(encoding="utf-8"))
    assert loaded["flagged_chapters"] == ["bad"]
    # LOUD summary names the flagged chapter.
    captured = capsys.readouterr().out
    assert "bad" in captured and "FLAGGED" in captured
    # REPORT-ONLY: input sidecars byte-unchanged.
    after = {p.name: p.read_bytes() for p in d.glob("*.json")}
    assert after == before


def test_standalone_main_exit_code(tmp_path):
    d = tmp_path / "judged"
    d.mkdir()
    _write_corrected(d, "good", _healthy_chapter()["region_provenance"])
    rc = standalone.main([str(d), "--out", str(tmp_path / "out")])
    assert rc == 0
    assert (tmp_path / "out" / "heading_judge_audit.json").is_file()


def test_standalone_empty_dir_is_clean(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    report = standalone.run_audit([d], out_dir=tmp_path / "out")
    assert report["flagged_chapters"] == []
    assert report["n_chapters"] == 0
