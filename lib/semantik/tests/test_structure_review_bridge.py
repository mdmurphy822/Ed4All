"""Phase 3 acceptance tests — SemantiK 70B structure-reviewer bridge surfacing,
Ed4All DecisionCapture emit, and the deterministic backstop.

Plan: ``plans/finegrain/semantik-70b-structure-reviewer-2026-06-22.md`` §7/§9
(Phase 3 acceptance, verbatim). Built + run with NO models / GPU — every
input is a synthetic cascade-result / bridge-JSON dict.

Covers:
  1. A bridge-JSON round-trip (synthetic cascade result with verdicts ->
     ``run_cascade_json`` dict -> the Ed4All ``_SemantikBridgeResult`` /
     ``build_chapters_ir``) carries the per-region ``review`` key AND the
     doc-level ``structure_review`` verdict list.
  2. A reviewer-OFF / 70B-absent run still strips a phantom-TOC cluster via
     ``drop_toc_and_frontmatter`` (the deterministic backstop) — asserted on a
     bridge JSON with the phantom cluster + ``structure_review=None``.
  3. A ``structure_review`` DecisionCapture JSONL row is emitted with a dynamic
     rationale >=20 chars and the canonical ``decision_type``.
  4. A re-roled-to-paragraph block does NOT leave a lowercase-leading orphan /
     bogus chapter (M2 scope check).

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest lib/semantik/tests/test_structure_review_bridge.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lib.semantik.cascade_ir import build_chapters_ir

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# A tiny duck-typed cascade result with the Stage-5d structure_review audit.
# ---------------------------------------------------------------------------


class _SyntheticCascadeResult:
    """Mimics the in-process ``PipelineV2Result`` shape the bridge builder
    (``run_cascade_json._build_bridge_dict``) reads — the per-region ``review``
    keys ride on ``region_provenance`` and the doc-level audit lives at
    ``cascade["conformance_audit"]["structure_review"]``."""

    def __init__(self, region_provenance, structure_review):
        self.html = "<main></main>"
        self.region_provenance = region_provenance
        self.heading_tree = []
        self.exit_action = "certify"
        self.theta_score = 0.9
        self.wcag_status = "passed"
        self.flags = []
        self.lane_used = "fast"
        self.cascade = {
            "runtime_mode": "real",
            "conformance_audit": {"structure_review": structure_review},
        }


def _reviewed_provenance() -> list[dict]:
    """A document-order provenance list where region 1 was re-roled by the
    Stage-5d reviewer (heading -> metadata_drop) and carries a per-region
    ``review`` key (terse corrected_from/corrected_to/note)."""
    return [
        {
            "region_index": 0,
            "region_kind": "heading",
            "role": "heading",
            "heading_text": "Chapter 1: Whole Numbers",
            "level": 1,
            "first_raw_block_index": 0,
            "pages": [1],
            "raw_text": "Chapter 1: Whole Numbers",
            "wcag_status": "passed",
            "confidence": 0.95,
        },
        {
            "region_index": 1,
            "region_kind": "metadata_drop",  # corrected by the reviewer
            "role": "metadata_drop",
            "heading_text": None,
            "level": None,
            "first_raw_block_index": 1,
            "pages": [1],
            "raw_text": "Answer Key 3.2",
            "wcag_status": "passed",
            "confidence": 0.5,
            "review": {
                "corrected_from": "heading",
                "corrected_to": "metadata_drop",
                "note": "answer-key run mis-promoted to <h2>; reclassified",
            },
        },
        {
            "region_index": 2,
            "region_kind": "paragraph",
            "role": "paragraph",
            "heading_text": None,
            "level": None,
            "first_raw_block_index": 2,
            "pages": [1],
            "raw_text": "Whole numbers are the counting numbers and zero.",
            "wcag_status": "passed",
            "confidence": 0.9,
        },
    ]


def _structure_review_audit() -> list[dict]:
    """The doc-level verdict list (ReviewVerdict-as-dicts)."""
    return [
        {
            "block_id": 1,
            "verdict": "drop_injected_header",
            "kind_before": "heading",
            "kind_after": "metadata_drop",
            "level_before": 2,
            "level_after": None,
            "review_note": "answer-key run; reclassified out of heading role",
            "reverted_for_invariant": False,
        }
    ]


# ---------------------------------------------------------------------------
# 1. Bridge-JSON round-trip carries the review key + doc-level structure_review.
# ---------------------------------------------------------------------------


def _import_bridge_builder():
    """Import ``run_cascade_json`` (a vendored SemantiK script) for its
    bridge-dict builder. It only needs argparse/json/os at module scope (the
    cascade import is lazy inside ``main``), so it imports with no heavy deps.
    """
    scripts_dir = _REPO_ROOT / "SemantiK" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import run_cascade_json  # type: ignore[import-not-found]
    finally:
        # Leave it on sys.path for the test session (harmless); but ensure a
        # repeat import resolves the vendored copy.
        pass
    return run_cascade_json


def test_bridge_json_roundtrip_carries_review_and_structure_review():
    rcj = _import_bridge_builder()
    result = _SyntheticCascadeResult(
        _reviewed_provenance(), _structure_review_audit()
    )

    bridge = rcj._build_bridge_dict(result, pdf="x.pdf")

    # JSON-serializable (the real bridge is written to disk + read back).
    bridge = json.loads(json.dumps(bridge))

    # (a) doc-level structure_review is a top-level bridge key.
    assert "structure_review" in bridge
    assert isinstance(bridge["structure_review"], list)
    assert bridge["structure_review"][0]["block_id"] == 1
    assert bridge["structure_review"][0]["verdict"] == "drop_injected_header"

    # (b) the per-region review key survives verbatim on region 1.
    region1 = next(r for r in bridge["region_provenance"] if r["region_index"] == 1)
    assert "review" in region1
    assert region1["review"]["corrected_to"] == "metadata_drop"

    # (c) the Ed4All-side _SemantikBridgeResult exposes structure_review.
    from MCP.tools.pipeline_tools import _SemantikBridgeResult

    bridge_result = _SemantikBridgeResult(bridge)
    assert bridge_result.structure_review is not None
    assert bridge_result.structure_review[0]["block_id"] == 1

    # (d) the seam resolver reads it back from the bridge result.
    from MCP.tools.pipeline_tools import _semantik_resolve_structure_review

    resolved = _semantik_resolve_structure_review(bridge_result)
    assert resolved is not None and resolved[0]["verdict"] == "drop_injected_header"

    # (e) build_chapters_ir consumes the corrected kinds (region 1 dropped, not
    # a chapter). The IR carries the chapter opened by region 0, with the
    # metadata_drop region NOT opening a chapter.
    chapters = build_chapters_ir(bridge_result)
    assert chapters, "expected at least the chapter from region 0"
    # The answer-key text must not become a chapter title anywhere.
    assert not any("Answer Key" in ch.title for ch in chapters)


# ---------------------------------------------------------------------------
# 2. Reviewer-OFF / 70B-absent run still strips phantom TOCs (backstop).
# ---------------------------------------------------------------------------


def _phantom_toc_bridge() -> dict:
    """A bridge JSON with a contiguous front-matter phantom-TOC run (heading
    regions with TRAILING increasing page numbers) and NO structure_review
    (reviewer off). The deterministic ``drop_toc_and_frontmatter`` backstop in
    ``build_chapters_ir`` must strip the run regardless of the 70B."""
    prov = [
        # Phantom-TOC run (front-matter zone, increasing trailing page #s).
        {
            "region_index": 0,
            "region_kind": "heading",
            "role": "heading",
            "heading_text": "Chapter 1 Whole Numbers 5",
            "level": 1,
            "first_raw_block_index": 0,
            "pages": [2],
            "raw_text": "Chapter 1 Whole Numbers 5",
        },
        {
            "region_index": 1,
            "region_kind": "heading",
            "role": "heading",
            "heading_text": "Chapter 2 Integers 89",
            "level": 1,
            "first_raw_block_index": 1,
            "pages": [2],
            "raw_text": "Chapter 2 Integers 89",
        },
        {
            "region_index": 2,
            "region_kind": "heading",
            "role": "heading",
            "heading_text": "Chapter 3 Fractions 173",
            "level": 1,
            "first_raw_block_index": 2,
            "pages": [2],
            "raw_text": "Chapter 3 Fractions 173",
        },
        {
            # 4th TOC entry so the run reaches _MIN_TOC_RUN (=4).
            "region_index": 3,
            "region_kind": "heading",
            "role": "heading",
            "heading_text": "Chapter 4 Decimals 251",
            "level": 1,
            "first_raw_block_index": 3,
            "pages": [2],
            "raw_text": "Chapter 4 Decimals 251",
        },
        # The REAL content begins here (no trailing page number).
        {
            "region_index": 4,
            "region_kind": "heading",
            "role": "heading",
            "heading_text": "1.1 Introduction to Whole Numbers",
            "level": 2,
            "first_raw_block_index": 4,
            "pages": [5],
            "raw_text": "1.1 Introduction to Whole Numbers",
        },
        {
            "region_index": 5,
            "region_kind": "paragraph",
            "role": "paragraph",
            "heading_text": None,
            "level": None,
            "first_raw_block_index": 5,
            "pages": [5],
            "raw_text": "The whole numbers are the counting numbers and zero.",
        },
    ]
    return {
        "pdf": "x.pdf",
        "html": "<main></main>",
        "region_provenance": prov,
        "heading_tree": [],
        "runtime_mode": "real",
        "structure_review": None,  # reviewer OFF / 70B absent
    }


def test_backstop_strips_phantom_toc_with_reviewer_off(monkeypatch):
    # The deterministic detector is gated ON by default; assert explicitly.
    monkeypatch.setenv("SEMANTIK_DROP_FRONTMATTER_TOC", "1")
    from MCP.tools.pipeline_tools import _SemantikBridgeResult

    bridge_result = _SemantikBridgeResult(_phantom_toc_bridge())
    # Reviewer did NOT run.
    assert bridge_result.structure_review is None

    chapters = build_chapters_ir(bridge_result)

    # The phantom "Chapter 2 Integers 89" / "Chapter 3 Fractions 173" TOC lines
    # must NOT survive as chapters — the backstop strips them.
    titles = " | ".join(ch.title for ch in chapters)
    assert "Integers 89" not in titles
    assert "Fractions 173" not in titles
    assert "Decimals 251" not in titles
    # The real content (the 1.1 section + its paragraph) survives.
    all_text = " ".join(
        b.raw_text or "" for ch in chapters for b in ch.blocks
    )
    assert "counting numbers and zero" in all_text


# ---------------------------------------------------------------------------
# 3. A structure_review DecisionCapture JSONL row is emitted (dynamic >=20).
# ---------------------------------------------------------------------------


def test_decision_capture_row_emitted_dynamic_rationale(monkeypatch):
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")
    # Strict mode: a non-canonical decision_type / tool would RAISE.
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
    monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW_MODEL", "meta/llama-3.3-70b")

    # The autouse conftest fixture redirects captures to a session temp root via
    # ED4ALL_TRAINING_CAPTURES_DIR; resolve THAT (not ED4ALL_HOME).
    from lib.paths import get_training_captures_dir

    captures_root = get_training_captures_dir()

    from MCP.tools.pipeline_tools import _emit_structure_review_capture

    verdicts = _structure_review_audit() + [
        {
            "block_id": 4,
            "verdict": "corrected",
            "kind_before": "paragraph",
            "kind_after": "heading",
            "level_before": None,
            "level_after": 2,
            "review_note": "missed heading; promoted",
            "reverted_for_invariant": True,  # reverted for the invariant
        }
    ]

    _emit_structure_review_capture(
        verdicts, canonical_course_code="SREV_CAP_101", pdf_stem="mybook"
    )

    rows = []
    for jsonl in Path(captures_root).rglob("*.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    sr_rows = [
        r
        for r in rows
        if r.get("decision_type") == "structure_review"
        and r.get("course_id") in {"SREV_CAP_101", "SREV-CAP-101"}
    ]
    assert sr_rows, "expected a structure_review capture row"
    row = sr_rows[0]

    # Canonical decision_type.
    assert row["decision_type"] == "structure_review"
    # Dynamic, replayable rationale >= 20 chars, interpolating the tallies.
    assert len(row["rationale"]) >= 20
    assert "meta/llama-3.3-70b" in row["rationale"]
    assert "reviewed" in row["rationale"].lower()
    # 1 applied correction (block 1, not reverted), 1 reverted (block 4).
    assert "applied=1" in row["decision"]
    assert "reverted=1" in row["decision"]


def test_decision_capture_skips_cleanly_when_reviewer_off():
    from lib.paths import get_training_captures_dir
    from MCP.tools.pipeline_tools import _emit_structure_review_capture

    captures_root = Path(get_training_captures_dir())

    def _count_for(course: str) -> int:
        n = 0
        for jsonl in captures_root.rglob("*.jsonl"):
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("decision_type") == "structure_review" and row.get(
                    "course_id"
                ) in {course, course.replace("_", "-")}:
                    n += 1
        return n

    before = _count_for("SREV_OFF_101")
    # verdicts=None => reviewer did NOT run => NO capture, NO exception.
    _emit_structure_review_capture(
        None, canonical_course_code="SREV_OFF_101", pdf_stem="mybook"
    )
    assert _count_for("SREV_OFF_101") == before, (
        "no structure_review capture should be written when the reviewer is off"
    )


def test_decision_capture_is_best_effort(monkeypatch):
    """A DecisionCapture failure must NOT raise out of the seam (best-effort)."""
    import MCP.tools.pipeline_tools as pt

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("simulated capture init failure")

    monkeypatch.setattr(
        "lib.decision_capture.DecisionCapture", _Boom, raising=True
    )
    # Must swallow the error and return None (no raise).
    assert (
        pt._emit_structure_review_capture(
            _structure_review_audit(),
            canonical_course_code="X",
            pdf_stem="y",
        )
        is None
    )


# ---------------------------------------------------------------------------
# 3b. Phase 5 — per-window full-block-review DecisionCapture (one row / window).
# ---------------------------------------------------------------------------


def _block_review_windowed_audit() -> list[dict]:
    """A two-window full-block-review verdict list (re-type ops), each verdict
    carrying the shared per-window ``block_review_window`` capture metadata the
    Phase-5 windowed reviewer stamps."""
    meta0 = {
        "window_index": 0,
        "members": 2,
        "idx_min": 0,
        "idx_max": 1,
        "page_min": 3,
        "page_max": 4,
        "council_kinds": {"code_block": 2},
        "model": "meta/llama-3.3-70b",
        "max_tokens": 512,
        "confidence_min": 0.31,
        "confidence_mean": 0.34,
        "confidence_max": 0.37,
    }
    meta1 = {
        "window_index": 1,
        "members": 2,
        "idx_min": 2,
        "idx_max": 3,
        "page_min": 5,
        "page_max": 6,
        "council_kinds": {"code_block": 1, "table": 1},
        "model": "meta/llama-3.3-70b",
        "max_tokens": 512,
        "confidence_min": 0.20,
        "confidence_mean": 0.25,
        "confidence_max": 0.30,
    }
    return [
        {
            "block_id": 0,
            "verdict": "corrected",
            "kind_before": "code_block",
            "kind_after": "paragraph",
            "level_before": None,
            "level_after": None,
            "review_note": "TRY IT exercise; re-typed out of code",
            "reverted_for_invariant": False,
            "reverted_for_endpoint_failure": False,
            "role_after": "example",
            "block_review_window": meta0,
        },
        {
            "block_id": 1,
            "verdict": "ok",
            "kind_before": "code_block",
            "kind_after": "code_block",
            "level_before": None,
            "level_after": None,
            "review_note": "kept original",
            "reverted_for_invariant": False,
            "reverted_for_endpoint_failure": False,
            "role_after": None,
            "block_review_window": meta0,
        },
        {
            "block_id": 2,
            "verdict": "corrected",
            "kind_before": "table",
            "kind_after": "paragraph",
            "level_before": None,
            "level_after": None,
            "review_note": "definition row; re-typed",
            "reverted_for_invariant": False,
            "reverted_for_endpoint_failure": False,
            "role_after": "definition",
            "block_review_window": meta1,
        },
        {
            "block_id": 3,
            "verdict": "ok",
            "kind_before": "code_block",
            "kind_after": "code_block",
            "level_before": None,
            "level_after": None,
            "review_note": "kept original",
            "reverted_for_invariant": False,
            "reverted_for_endpoint_failure": False,
            "role_after": None,
            "block_review_window": meta1,
        },
    ]


def _read_capture_rows(captures_root) -> list[dict]:
    rows: list[dict] = []
    for jsonl in Path(captures_root).rglob("*.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def test_block_review_capture_fires_per_window_dynamic_rationale(monkeypatch):
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
    monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW_MODEL", "meta/llama-3.3-70b")

    from lib.paths import get_training_captures_dir
    from MCP.tools.pipeline_tools import _emit_structure_review_capture

    captures_root = get_training_captures_dir()

    _emit_structure_review_capture(
        _block_review_windowed_audit(),
        canonical_course_code="BREV_CAP_101",
        pdf_stem="mybook",
    )

    rows = [
        r
        for r in _read_capture_rows(captures_root)
        if r.get("decision_type") == "structure_review"
        and r.get("course_id") in {"BREV_CAP_101", "BREV-CAP-101"}
    ]
    # ONE capture row PER WINDOW (two windows -> two rows), not a per-doc row.
    assert len(rows) == 2, f"expected one capture per window, got {len(rows)}"
    by_window = {}
    for r in rows:
        assert r["decision_type"] == "structure_review"  # canonical enum reused
        assert len(r["rationale"]) >= 20
        # Dynamic, replayable tokens: model id + window idx-range + ops counts.
        assert "meta/llama-3.3-70b" in r["rationale"]
        assert "full-block reviewer" in r["rationale"]
        if "window=0" in r["decision"]:
            by_window[0] = r
        elif "window=1" in r["decision"]:
            by_window[1] = r
    assert set(by_window) == {0, 1}, "expected window 0 and window 1 rows"
    # Window 0: blocks 0-1, 1 re-type applied (block 0 corrected).
    assert "blocks 0-1" in by_window[0]["rationale"]
    assert "applied=1" in by_window[0]["decision"]
    assert "retype=1" in by_window[0]["decision"]
    # Window 1: blocks 2-3, 1 re-type applied (block 2 corrected).
    assert "blocks 2-3" in by_window[1]["rationale"]
    assert "applied=1" in by_window[1]["decision"]


def test_block_review_capture_best_effort_no_raise(monkeypatch):
    """A DecisionCapture failure on the windowed path must NOT raise."""
    import MCP.tools.pipeline_tools as pt

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("simulated capture init failure")

    monkeypatch.setattr(
        "lib.decision_capture.DecisionCapture", _Boom, raising=True
    )
    assert (
        pt._emit_structure_review_capture(
            _block_review_windowed_audit(),
            canonical_course_code="BREV_BOOM_101",
            pdf_stem="y",
        )
        is None
    )


def test_capture_skips_when_flag_off(monkeypatch):
    """Heading-only verdicts (no block_review_window) emit the per-doc capture
    and NO per-window rows — the flag-off byte-stable path."""
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")

    from lib.paths import get_training_captures_dir
    from MCP.tools.pipeline_tools import _emit_structure_review_capture

    captures_root = get_training_captures_dir()

    _emit_structure_review_capture(
        _structure_review_audit(),
        canonical_course_code="BREV_OFF_101",
        pdf_stem="mybook",
    )

    rows = [
        r
        for r in _read_capture_rows(captures_root)
        if r.get("decision_type") == "structure_review"
        and r.get("course_id") in {"BREV_OFF_101", "BREV-OFF-101"}
    ]
    # The per-doc capture fires (heading-only path) ...
    assert len(rows) == 1, f"expected one per-doc capture row, got {len(rows)}"
    # ... and it is NOT a per-window row (no window grouping signal).
    assert "window=" not in rows[0]["decision"]


# ---------------------------------------------------------------------------
# 3c. A block_resegment DecisionCapture JSONL row is emitted (dynamic >=20),
#     skips cleanly when the re-segmenter is off, and is best-effort.
# ---------------------------------------------------------------------------


def _block_resegment_audit() -> list[dict]:
    """The doc-level block_resegment op list the SEMANTIK_BLOCK_RESEGMENT pass
    surfaces (merge folds N regions into one; split fans one into N)."""
    return [
        {
            "op": "merge",
            "region_indices": [3, 4],
            "conservation_verified": True,
            "note": "continuation fragment folded into preceding paragraph",
        },
        {
            "op": "split",
            "region_indices": [7],
            "conservation_verified": True,
            "note": "two stacked list items split into separate blocks",
        },
    ]


def test_block_resegment_capture_row_emitted_dynamic_rationale(monkeypatch):
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")
    # Strict mode: a non-canonical decision_type / tool would RAISE — this also
    # proves the schema enum carries ``block_resegment``.
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")

    from lib.paths import get_training_captures_dir

    captures_root = get_training_captures_dir()

    from MCP.tools.pipeline_tools import _emit_block_resegment_capture

    _emit_block_resegment_capture(
        _block_resegment_audit(),
        canonical_course_code="BRESEG_CAP_101",
        pdf_stem="mybook",
    )

    rows = []
    for jsonl in Path(captures_root).rglob("*.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    br_rows = [
        r
        for r in rows
        if r.get("decision_type") == "block_resegment"
        and r.get("course_id") in {"BRESEG_CAP_101", "BRESEG-CAP-101"}
    ]
    assert br_rows, "expected a block_resegment capture row"
    row = br_rows[0]

    # Canonical decision_type.
    assert row["decision_type"] == "block_resegment"
    # Dynamic, replayable rationale >= 20 chars interpolating the op tallies.
    assert len(row["rationale"]) >= 20
    assert "1 merge" in row["rationale"]
    assert "1 split" in row["rationale"]
    assert "conservation_verified=True" in row["rationale"]
    # Decision string carries the op counts.
    assert "merges=1" in row["decision"]
    assert "splits=1" in row["decision"]
    assert "ops=2" in row["decision"]


def test_block_resegment_capture_skips_cleanly_when_off():
    from lib.paths import get_training_captures_dir
    from MCP.tools.pipeline_tools import _emit_block_resegment_capture

    captures_root = Path(get_training_captures_dir())

    def _count_for(course: str) -> int:
        n = 0
        for jsonl in captures_root.rglob("*.jsonl"):
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("decision_type") == "block_resegment" and row.get(
                    "course_id"
                ) in {course, course.replace("_", "-")}:
                    n += 1
        return n

    before = _count_for("BRESEG_OFF_101")
    # ops=None => re-segmenter did NOT run => NO capture, NO exception.
    _emit_block_resegment_capture(
        None, canonical_course_code="BRESEG_OFF_101", pdf_stem="mybook"
    )
    assert _count_for("BRESEG_OFF_101") == before, (
        "no block_resegment capture should be written when the re-segmenter is off"
    )


def test_block_resegment_capture_is_best_effort(monkeypatch):
    """A DecisionCapture failure must NOT raise out of the seam (best-effort)."""
    import MCP.tools.pipeline_tools as pt

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("simulated capture init failure")

    monkeypatch.setattr(
        "lib.decision_capture.DecisionCapture", _Boom, raising=True
    )
    assert (
        pt._emit_block_resegment_capture(
            _block_resegment_audit(),
            canonical_course_code="X",
            pdf_stem="y",
        )
        is None
    )


def test_block_resegment_resolver_reads_both_arms():
    """The seam resolver reads block_resegment off the bridge attribute AND the
    in-process conformance_audit, and returns None when neither carries it."""
    from MCP.tools.pipeline_tools import (
        _SemantikBridgeResult,
        _semantik_resolve_block_resegment,
    )

    # Bridge arm — top-level attribute.
    bridge = _SemantikBridgeResult(
        {"pdf": "x.pdf", "block_resegment": _block_resegment_audit()}
    )
    resolved = _semantik_resolve_block_resegment(bridge)
    assert resolved is not None and resolved[0]["op"] == "merge"

    # In-process arm — cascade["conformance_audit"]["block_resegment"].
    inproc = _SyntheticCascadeResult(_reviewed_provenance(), None)
    inproc.block_resegment = None  # in-process result has no bridge attribute
    inproc.cascade["conformance_audit"]["block_resegment"] = (
        _block_resegment_audit()
    )
    resolved2 = _semantik_resolve_block_resegment(inproc)
    assert resolved2 is not None and resolved2[1]["op"] == "split"

    # Neither arm carries it => None (clean skip).
    off = _SyntheticCascadeResult(_reviewed_provenance(), None)
    off.block_resegment = None
    assert _semantik_resolve_block_resegment(off) is None


# ---------------------------------------------------------------------------
# 4. A re-roled-to-paragraph block does NOT leave a lowercase orphan / bogus
#    chapter (M2 scope check).
# ---------------------------------------------------------------------------


def _reroled_to_paragraph_provenance() -> list[dict]:
    """Document order: a real chapter heading, then a region the reviewer
    re-roled heading->paragraph (a lowercase-leading continuation fragment),
    then a normal paragraph. The re-roled block must MERGE as content into the
    surrounding chapter — it must NOT open a new chapter (no heading_text) and
    must NOT be stranded as a content-free orphan chapter."""
    return [
        {
            "region_index": 0,
            "region_kind": "heading",
            "role": "heading",
            "heading_text": "Chapter 1: Whole Numbers",
            "level": 1,
            "first_raw_block_index": 0,
            "pages": [1],
            "raw_text": "Chapter 1: Whole Numbers",
        },
        {
            # Re-roled heading->paragraph: lowercase-leading fragment. As a
            # paragraph it has NO heading_text and CANNOT open a chapter.
            "region_index": 1,
            "region_kind": "paragraph",
            "role": "paragraph",
            "heading_text": None,
            "level": None,
            "first_raw_block_index": 1,
            "pages": [1],
            "raw_text": "and zero, together forming the set of whole numbers.",
            "review": {
                "corrected_from": "heading",
                "corrected_to": "paragraph",
                "note": "lowercase continuation mis-promoted to heading",
            },
        },
        {
            "region_index": 2,
            "region_kind": "paragraph",
            "role": "paragraph",
            "heading_text": None,
            "level": None,
            "first_raw_block_index": 2,
            "pages": [1],
            "raw_text": "We use whole numbers to count discrete objects.",
        },
    ]


def test_reroled_to_paragraph_merges_no_bogus_chapter():
    result = _SyntheticCascadeResult(_reroled_to_paragraph_provenance(), None)
    chapters = build_chapters_ir(result)

    # Exactly ONE chapter — the re-roled paragraph did NOT open a second.
    assert len(chapters) == 1, [ch.title for ch in chapters]
    ch = chapters[0]
    assert ch.title == "Chapter 1: Whole Numbers"

    # No chapter title is a lowercase-leading orphan fragment.
    for c in chapters:
        assert c.title and c.title[0].isupper(), c.title

    # The re-roled fragment merged as a content block under the chapter.
    block_texts = [b.raw_text or "" for b in ch.blocks]
    assert any("set of whole numbers" in t for t in block_texts)
    # And the chapter has real content (not a content-free orphan).
    assert any(
        str(b.region_kind or "") not in {"heading", "metadata_drop", ""}
        for b in ch.blocks
    )


# ---------------------------------------------------------------------------
# 3b. Backstop confirmation — the deterministic detectors STILL run even when
#     the reviewer ran and stamped per-region ``review`` keys (the unknown key
#     does not disable the fail-closed floor; cascade_ir needs no change).
# ---------------------------------------------------------------------------


def _review_stamped_phantom_bridge() -> dict:
    """A phantom-TOC run where EACH region ALSO carries a per-region ``review``
    key (reviewer ran). The deterministic ``drop_toc_and_frontmatter`` +
    ``_is_noncontent_heading`` backstop must STILL strip the run — the unknown
    ``review`` key is ignored by cascade_ir, never short-circuits the floor."""
    bridge = _phantom_toc_bridge()
    bridge["structure_review"] = [
        {
            "block_id": 0,
            "verdict": "ok",
            "kind_before": "heading",
            "kind_after": "heading",
            "level_before": 1,
            "level_after": 1,
            "review_note": "reviewer left this as-is",
            "reverted_for_invariant": False,
        }
    ]
    for region in bridge["region_provenance"]:
        region["review"] = {
            "corrected_from": "heading",
            "corrected_to": "heading",
            "note": "reviewer pass-through",
        }
    # Add an answer-key NON-CONTENT heading carrying a review key — it must be
    # filtered by _is_noncontent_heading regardless of the review metadata.
    bridge["region_provenance"].append(
        {
            "region_index": 6,
            "region_kind": "heading",
            "role": "heading",
            "heading_text": "78 41. 900 42.",  # numeric answer-key row
            "level": 1,
            "first_raw_block_index": 6,
            "pages": [5],
            "raw_text": "78 41. 900 42.",
            "review": {"corrected_from": "heading", "corrected_to": "heading"},
        }
    )
    return bridge


def test_backstop_runs_even_with_review_keys_present(monkeypatch):
    monkeypatch.setenv("SEMANTIK_DROP_FRONTMATTER_TOC", "1")
    from MCP.tools.pipeline_tools import _SemantikBridgeResult

    bridge_result = _SemantikBridgeResult(_review_stamped_phantom_bridge())
    # Reviewer DID run (the per-region review keys + doc-level audit).
    assert bridge_result.structure_review is not None

    chapters = build_chapters_ir(bridge_result)
    titles = " | ".join(ch.title for ch in chapters)

    # The phantom-TOC run is STILL stripped (deterministic detector is the
    # backstop, independent of the reviewer).
    assert "Integers 89" not in titles
    assert "Decimals 251" not in titles
    # The numeric answer-key heading never opens a chapter (_is_noncontent_heading
    # still fires despite the review key on it).
    assert not any("78 41." in (ch.title or "") for ch in chapters)
    # The real content survives.
    all_text = " ".join(b.raw_text or "" for ch in chapters for b in ch.blocks)
    assert "counting numbers and zero" in all_text


def test_cascade_ir_ignores_unknown_review_key_no_crash():
    """cascade_ir consumes corrected region_kind/level/heading_text and IGNORES
    the unknown per-region ``review`` key (no functional change needed)."""
    result = _SyntheticCascadeResult(_reviewed_provenance(), _structure_review_audit())
    # Must not raise on the unknown key; produces a sane IR.
    chapters = build_chapters_ir(result)
    assert chapters
    # The reviewer-corrected metadata_drop region (region 1) is NOT content and
    # never opens a chapter.
    assert not any("Answer Key" in ch.title for ch in chapters)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
