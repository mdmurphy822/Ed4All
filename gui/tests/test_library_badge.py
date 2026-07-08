"""Tests for the Studio Library certification badge (T1).

Covers ``imscc_service.list_library`` adding the 5-value ``course_status`` enum
to each card from the promotion-chain report (both landing paths), degrading to
an ABSENT field for an ungoverned / malformed course, plus a WCAG gate over the
reconstructed badge + scorecard-dialog markup (the certification state is real
text, never colour-only). Synthetic tmp-path courses only.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List, Tuple

import pytest

from gui.services import imscc_service

# --------------------------------------------------------------------------- #
# Synthetic cartridge + report builders
# --------------------------------------------------------------------------- #

_MANIFEST = """<?xml version='1.0' encoding='utf-8'?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
          identifier="DEMO_manifest">
  <organizations>
    <organization identifier="ORG_1" structure="rooted-hierarchy">
      <item identifier="ROOT">
        <item identifier="ITEM_OV" identifierref="RES_overview">
          <title>Overview</title>
        </item>
      </item>
    </organization>
  </organizations>
  <resources>
    <resource identifier="RES_overview" type="webcontent" href="overview.html">
      <file href="overview.html" />
    </resource>
  </resources>
</manifest>
"""


def _build_cartridge(libv2_root: Path, slug: str) -> Path:
    imscc_dir = libv2_root / "courses" / slug / "source" / "imscc"
    imscc_dir.mkdir(parents=True, exist_ok=True)
    cartridge = imscc_dir / f"{slug.upper()}.imscc"
    with zipfile.ZipFile(cartridge, "w") as zf:
        zf.writestr("imsmanifest.xml", _MANIFEST)
        zf.writestr("overview.html", "<!DOCTYPE html><html><body><h1>Hi</h1></body></html>")
    return cartridge


def _write_chain_report(course_dir: Path, status: str, *, in_trainforge: bool = False) -> None:
    name = "courseforge_promotion_chain_report.json"
    dest = (course_dir / "training_specs" / name) if in_trainforge else (course_dir / name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"schema_version": "1.0", "course_status": status, "arrows": []}), encoding="utf-8")


def _card(libv2_root: Path, slug: str) -> dict:
    cards = imscc_service.list_library(libv2_root)
    return next(c for c in cards if c["slug"] == slug)


# --------------------------------------------------------------------------- #
# list_library course_status
# --------------------------------------------------------------------------- #


def test_ungoverned_course_has_no_status(libv2_root):
    _build_cartridge(libv2_root, "plain-101")
    card = _card(libv2_root, "plain-101")
    assert "course_status" not in card  # absent → no badge


def test_status_from_canonical_path(libv2_root):
    _build_cartridge(libv2_root, "gov-101")
    _write_chain_report(libv2_root / "courses" / "gov-101", "certified_instructional")
    card = _card(libv2_root, "gov-101")
    assert card["course_status"] == "certified_instructional"


def test_status_from_trainforge_fallback_path(libv2_root):
    _build_cartridge(libv2_root, "gov-102")
    _write_chain_report(libv2_root / "courses" / "gov-102", "certified_accessible", in_trainforge=True)
    card = _card(libv2_root, "gov-102")
    assert card["course_status"] == "certified_accessible"


def test_canonical_path_wins_over_fallback(libv2_root):
    _build_cartridge(libv2_root, "gov-103")
    cdir = libv2_root / "courses" / "gov-103"
    _write_chain_report(cdir, "certified_trainable")
    _write_chain_report(cdir, "failed", in_trainforge=True)
    card = _card(libv2_root, "gov-103")
    assert card["course_status"] == "certified_trainable"


def test_failed_status_badged(libv2_root):
    _build_cartridge(libv2_root, "fail-101")
    _write_chain_report(libv2_root / "courses" / "fail-101", "failed")
    card = _card(libv2_root, "fail-101")
    assert card["course_status"] == "failed"


def test_unknown_enum_value_dropped(libv2_root):
    _build_cartridge(libv2_root, "junk-101")
    _write_chain_report(libv2_root / "courses" / "junk-101", "totally_made_up")
    card = _card(libv2_root, "junk-101")
    assert "course_status" not in card  # only the 5 canonical values badge


def test_malformed_report_dropped(libv2_root):
    _build_cartridge(libv2_root, "malf-101")
    cdir = libv2_root / "courses" / "malf-101"
    (cdir / "courseforge_promotion_chain_report.json").write_text("{ not json", encoding="utf-8")
    card = _card(libv2_root, "malf-101")
    assert "course_status" not in card


def test_card_shape_preserved(libv2_root):
    # The base card shape is untouched; course_status is purely additive.
    _build_cartridge(libv2_root, "shape-101")
    card = _card(libv2_root, "shape-101")
    for key in ("slug", "title", "page_count", "has_vector_index", "disk_bytes"):
        assert key in card


# --------------------------------------------------------------------------- #
# WCAG gate over the reconstructed badge + dialog markup
# --------------------------------------------------------------------------- #

# The badge + scorecard dialog exactly as studio.js builds them (the JS is not
# executed here — this reconstructs the DOM the a11y validator must pass, the
# same pattern as test_studio_a11y_gate's view reconstructions).
_LIBRARY_BADGE_FRAGMENT = """
<ul class="cards">
  <li class="card-li">
    <a class="card" href="#/viewer/demo-101">
      <h2>Demo Course</h2>
      <p class="meta">2 pages · 1.0 MB</p>
      <p class="card-badges">
        <span class="badge status-badge status-positive">Certified · Trainable</span>
        <span class="badge">Ask-ready</span>
      </p>
    </a>
    <div class="card-actions">
      <button type="button" class="card-scorecard" aria-label="Quality report for Demo Course">Quality report</button>
      <button type="button" class="card-delete" aria-label="Delete Demo Course">Delete</button>
    </div>
  </li>
</ul>
"""

_SCORECARD_DIALOG_FRAGMENT = """
<div class="modal-dialog scorecard-dialog" role="dialog" aria-modal="true" aria-labelledby="sc-title">
  <h2 id="sc-title">Quality report — Demo Course</h2>
  <div class="scorecard-body" aria-live="polite" aria-busy="false">
    <p class="sc-status"><span class="sc-status-label">Certification: </span>
      <span class="badge status-badge status-positive">Certified · Trainable</span></p>
    <section class="sc-section" aria-label="Retrieval evaluation">
      <h3>Retrieval evaluation</h3>
      <p class="meta">Engine: hybrid-rrf</p>
      <table class="sc-table">
        <thead><tr>
          <th scope="col">Arm</th><th scope="col">Key-point coverage</th>
          <th scope="col">Unsupported-claim rate</th><th scope="col">Latency p50 / p95</th>
        </tr></thead>
        <tbody><tr>
          <th scope="row">Base</th><td>0.40</td><td>0.06</td><td>5222 / 8464 ms</td>
        </tr></tbody>
      </table>
    </section>
    <section class="sc-section" aria-label="Refusal calibration">
      <h3>Refusal calibration</h3>
      <p class="muted sc-not-evaluated">Not yet evaluated.</p>
    </section>
  </div>
  <div class="modal-actions"><button type="button" class="btn">Close</button></div>
</div>
"""


def _wcag_blocking(html: str) -> Tuple[List, List]:
    from lib.validators.wcag import IssueSeverity, WCAGValidator  # noqa: PLC0415

    report = WCAGValidator().validate(html)
    blocking = [
        i for i in report.issues
        if i.severity in {IssueSeverity.CRITICAL, IssueSeverity.HIGH}
    ]
    return blocking, report.issues


def _page(inner: str) -> str:
    # Minimal valid document wrapper with a single h1 landmark so the validator
    # scores the fragment in a well-formed context (mirrors the a11y-gate shell).
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><title>Library</title></head>"
        "<body><main><h1>Course Library</h1>" + inner + "</main></body></html>"
    )


def test_library_badge_markup_zero_aa_findings():
    pytest.importorskip("bs4")
    blocking, _ = _wcag_blocking(_page(_LIBRARY_BADGE_FRAGMENT))
    assert not blocking, [str(i) for i in blocking]


def test_scorecard_dialog_markup_zero_aa_findings():
    pytest.importorskip("bs4")
    blocking, _ = _wcag_blocking(_page(_SCORECARD_DIALOG_FRAGMENT))
    assert not blocking, [str(i) for i in blocking]
