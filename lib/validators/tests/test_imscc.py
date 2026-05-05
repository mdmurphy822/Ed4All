"""Tests for IMSCCValidator's W5 escalation-marker leak check.

Worker W5 (Courseforge validation-wiring fix plan): blocks carrying a
non-null ``escalation_marker`` (consensus failure / outline budget
exhausted / structural unfixable) MUST NOT ship into per-page IMSCC
HTML. The defensive packager-side check inside
``IMSCCValidator._check_escalated_blocks_absent`` walks
``blocks_final.jsonl`` and scans the emitted HTML for any
``data-cf-block-id="{escalated_id}"`` match — emits
``code="ESCALATED_BLOCK_IN_IMSCC"`` (critical) on every leak.

Backward-compat contract: when ``blocks_final_path`` is absent from
the validator inputs (pre-W5 callers / non-two-pass workflows), the
check no-ops silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.imscc import IMSCCValidator  # noqa: E402


def _seed_minimal_imscc_dir(tmp_path: Path) -> Path:
    """Build a minimal extracted-IMSCC dir with a passing manifest so
    ``IMSCCValidator.validate()`` reaches the W5 check instead of
    short-circuiting on FILE_NOT_FOUND / MANIFEST_MISSING."""
    imscc_dir = tmp_path / "imscc_extracted"
    imscc_dir.mkdir()
    manifest = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1">\n'
        '  <organizations>\n'
        '    <organization identifier="org1">\n'
        '      <title>Test</title>\n'
        '    </organization>\n'
        '  </organizations>\n'
        '  <resources>\n'
        '    <resource identifier="r1" type="webcontent" href="page1.html"/>\n'
        '  </resources>\n'
        '</manifest>\n'
    )
    (imscc_dir / "imsmanifest.xml").write_text(manifest, encoding="utf-8")
    return imscc_dir


def _seed_blocks_final(
    tmp_path: Path,
    entries: list,
) -> Path:
    """Persist a ``blocks_final.jsonl`` and return its path."""
    p = tmp_path / "blocks_final.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry))
            fh.write("\n")
    return p


def test_imscc_validator_flags_escalated_block_in_html(tmp_path):
    """Synthetic blocks_final lists one escalated block; the matching
    ``<section data-cf-block-id>`` is present in the content_dir HTML.
    Validator returns passed=False with code=ESCALATED_BLOCK_IN_IMSCC."""
    imscc_dir = _seed_minimal_imscc_dir(tmp_path)
    content_dir = tmp_path / "content"
    content_dir.mkdir()

    escalated_id = "week_01_content_01#objective_to-01_0"
    # The HTML file shipped into the IMSCC carries the escalated id
    # — exactly the leak W5 forbids.
    (content_dir / "page1.html").write_text(
        "<!DOCTYPE html><html><body><main>"
        f'<section data-cf-block-id="{escalated_id}">'
        "<p>leaked content</p></section>"
        "</main></body></html>",
        encoding="utf-8",
    )

    blocks_final = _seed_blocks_final(tmp_path, [
        {
            "block_id": escalated_id,
            "block_type": "objective",
            "escalation_marker": "validator_consensus_fail",
        },
    ])

    validator = IMSCCValidator()
    result = validator.validate({
        "imscc_path": str(imscc_dir),
        "content_dir": str(content_dir),
        "blocks_final_path": str(blocks_final),
    })

    assert result.passed is False, result
    codes = [i.code for i in result.issues]
    assert "ESCALATED_BLOCK_IN_IMSCC" in codes, codes
    # The matching error issue carries the offending block_id.
    leak_issue = next(
        i for i in result.issues
        if i.code == "ESCALATED_BLOCK_IN_IMSCC"
    )
    assert escalated_id in leak_issue.message
    assert leak_issue.severity == "error"


def test_imscc_validator_passes_when_escalated_block_filtered(tmp_path):
    """blocks_final lists an escalated block but the content_dir HTML
    does NOT contain a matching data-cf-block-id (W5 emit-time filter
    worked). Validator does NOT emit ESCALATED_BLOCK_IN_IMSCC."""
    imscc_dir = _seed_minimal_imscc_dir(tmp_path)
    content_dir = tmp_path / "content"
    content_dir.mkdir()

    escalated_id = "week_01_content_01#objective_to-01_0"
    surviving_id = "week_01_content_01#objective_to-02_0"
    (content_dir / "page1.html").write_text(
        "<!DOCTYPE html><html><body><main>"
        f'<section data-cf-block-id="{surviving_id}">'
        "<p>good content</p></section>"
        "</main></body></html>",
        encoding="utf-8",
    )

    blocks_final = _seed_blocks_final(tmp_path, [
        {
            "block_id": escalated_id,
            "block_type": "objective",
            "escalation_marker": "validator_consensus_fail",
        },
        {
            "block_id": surviving_id,
            "block_type": "objective",
            "escalation_marker": None,
        },
    ])

    validator = IMSCCValidator()
    result = validator.validate({
        "imscc_path": str(imscc_dir),
        "content_dir": str(content_dir),
        "blocks_final_path": str(blocks_final),
    })

    codes = [i.code for i in result.issues]
    assert "ESCALATED_BLOCK_IN_IMSCC" not in codes, codes


def test_imscc_validator_no_op_when_blocks_final_missing(tmp_path):
    """Pre-W5 caller (no ``blocks_final_path`` input) → check no-ops
    silently; validator passes on its existing manifest signal."""
    imscc_dir = _seed_minimal_imscc_dir(tmp_path)

    validator = IMSCCValidator()
    result = validator.validate({
        "imscc_path": str(imscc_dir),
    })

    codes = [i.code for i in result.issues]
    assert "ESCALATED_BLOCK_IN_IMSCC" not in codes, codes
    assert "BLOCKS_FINAL_MISSING" not in codes, codes
    # Manifest validates fine in this minimal fixture so the gate
    # passes overall.
    assert result.passed is True, result


def test_imscc_validator_no_escalated_blocks_passes(tmp_path):
    """blocks_final lists ZERO escalated entries → check returns no
    issues regardless of HTML content."""
    imscc_dir = _seed_minimal_imscc_dir(tmp_path)
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "page1.html").write_text(
        '<html><body><section data-cf-block-id="x">ok</section></body></html>',
        encoding="utf-8",
    )

    blocks_final = _seed_blocks_final(tmp_path, [
        {
            "block_id": "x",
            "block_type": "objective",
            "escalation_marker": None,
        },
    ])

    validator = IMSCCValidator()
    result = validator.validate({
        "imscc_path": str(imscc_dir),
        "content_dir": str(content_dir),
        "blocks_final_path": str(blocks_final),
    })

    codes = [i.code for i in result.issues]
    assert "ESCALATED_BLOCK_IN_IMSCC" not in codes
    assert result.passed is True, result


def test_imscc_validator_emits_info_when_blocks_final_path_missing_on_disk(
    tmp_path,
):
    """``blocks_final_path`` provided but file does not exist → info
    issue ``BLOCKS_FINAL_MISSING``; validator does not fail closed."""
    imscc_dir = _seed_minimal_imscc_dir(tmp_path)

    validator = IMSCCValidator()
    result = validator.validate({
        "imscc_path": str(imscc_dir),
        "blocks_final_path": str(tmp_path / "nonexistent.jsonl"),
    })

    codes = [i.code for i in result.issues]
    assert "BLOCKS_FINAL_MISSING" in codes, codes
    # info severity does not flip passed.
    assert result.passed is True, result


def test_imscc_validator_falls_back_to_imscc_path_when_no_content_dir(
    tmp_path,
):
    """When ``content_dir`` is absent, the validator walks the
    extracted ``imscc_path`` for HTML files."""
    imscc_dir = _seed_minimal_imscc_dir(tmp_path)
    escalated_id = "week_01_content_01#objective_to-99_0"
    # Drop the leaking page directly into the imscc_dir (no content_dir).
    (imscc_dir / "page_leak.html").write_text(
        f'<html><body><section data-cf-block-id="{escalated_id}">'
        f"</section></body></html>",
        encoding="utf-8",
    )

    blocks_final = _seed_blocks_final(tmp_path, [
        {
            "block_id": escalated_id,
            "block_type": "objective",
            "escalation_marker": "outline_budget_exhausted",
        },
    ])

    validator = IMSCCValidator()
    result = validator.validate({
        "imscc_path": str(imscc_dir),
        "blocks_final_path": str(blocks_final),
    })

    codes = [i.code for i in result.issues]
    assert "ESCALATED_BLOCK_IN_IMSCC" in codes, codes
