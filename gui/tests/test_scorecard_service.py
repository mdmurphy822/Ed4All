"""Tests for the per-course quality scorecard service + router endpoint (T2).

Composes the scorecard over SYNTHETIC tmp-path course dirs — a builder writes
tiny governance / eval report JSONs under ``<libv2_root>/courses/<slug>/`` (no
real LibV2 course is touched; ``libv2_root`` isolates via the shared fixture).

Asserts: (a) every section is ``not yet evaluated`` when no artifact exists
(never a fabricated number), (b) each section composes from its backing report
when present, (c) the composed course_status rides along when a promotion-chain
report resolves, and (d) the router maps unknown-course / bad-slug to 404 / 422.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402
from gui.services import scorecard_service  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic report builders
# --------------------------------------------------------------------------- #


def _course_dir(libv2_root: Path, slug: str) -> Path:
    d = libv2_root / "courses" / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _eval_scorecard_doc() -> dict:
    return {
        "schema_version": "1.4",
        "engine": "hybrid-rrf",
        "generated_at": "2026-06-14T01:39:37Z",
        "course_slug": "demo",
        "comparison": {
            "base": {
                "key_point_coverage_rate": 0.4,
                "claim_level_unsupported_rate": 0.055,
                "answered": 124,
                "declined": 0,
                "latency_ms": {"p50": 5222.2, "p95": 8464.1},
            },
            "retrieval": {
                "key_point_coverage_rate": 0.928,
                "claim_level_unsupported_rate": "—",
                "latency_ms": {"p50": 300.9, "p95": 409.9},
            },
            "grounded": {
                "key_point_coverage_rate": 0.674,
                "claim_level_unsupported_rate": None,
                "answered": 123,
                "declined": 1,
                "latency_ms": {"p50": 5719.7, "p95": 12114.3},
            },
            "refusal_safety": {"answered_not_refused_rate": 0.12},
        },
    }


def _chain_report_doc(status: str = "certified_instructional") -> dict:
    return {"schema_version": "1.0", "course_status": status, "arrows": []}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(state_dir, libv2_root):
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# Service-level composition
# --------------------------------------------------------------------------- #


def test_bare_course_all_sections_not_evaluated(libv2_root):
    _course_dir(libv2_root, "bare-101")
    sc = scorecard_service.build_scorecard("bare-101", libv2_root)
    assert sc["slug"] == "bare-101"
    assert "course_status" not in sc  # no chain report → no status
    for name in ("retrieval_eval", "refusal_calibration", "assessment_quality", "coverage_map"):
        section = sc["sections"][name]
        assert section["available"] is False
        assert section["status"] == "not yet evaluated"


def test_retrieval_eval_section_composes_latest(libv2_root):
    cdir = _course_dir(libv2_root, "eval-101")
    # Two scorecards; the newer ISO-timestamp name must win.
    _write(cdir / "retrieval_eval" / "eval_scorecard_20260101T000000Z.json", {"comparison": {}})
    _write(cdir / "retrieval_eval" / "eval_scorecard_20260614T013937Z.json", _eval_scorecard_doc())
    sc = scorecard_service.build_scorecard("eval-101", libv2_root)
    section = sc["sections"]["retrieval_eval"]
    assert section["available"] is True
    assert section["source_file"] == "eval_scorecard_20260614T013937Z.json"
    assert section["engine"] == "hybrid-rrf"
    arms = section["arms"]
    assert set(arms) == {"base", "retrieval", "grounded"}
    assert arms["base"]["key_point_coverage_rate"] == 0.4
    assert arms["base"]["latency_ms"] == {"p50": 5222.2, "p95": 8464.1}
    # An unsupported-rate sentinel / null is echoed verbatim, never fabricated.
    assert arms["retrieval"]["claim_level_unsupported_rate"] == "—"
    assert arms["grounded"]["claim_level_unsupported_rate"] is None
    assert section["refusal_safety"] == {"answered_not_refused_rate": 0.12}


def test_refusal_calibration_section(libv2_root):
    cdir = _course_dir(libv2_root, "cal-101")
    _write(
        cdir / "retrieval_eval" / "refusal_calibration.json",
        {
            "schema_version": "1.0",
            "generated_at": "2026-06-12T14:37:31Z",
            "engine": "hybrid-rrf",
            "recommended": {"min_top_score": 0.0296, "refusal_precision": 1.0, "answer_recall": 1.0},
        },
    )
    sc = scorecard_service.build_scorecard("cal-101", libv2_root)
    section = sc["sections"]["refusal_calibration"]
    assert section["available"] is True
    assert section["recommended"]["refusal_precision"] == 1.0


def test_assessment_quality_section(libv2_root):
    cdir = _course_dir(libv2_root, "aq-101")
    _write(
        cdir / "quality" / "trainforge_assessment_quality_report.json",
        {
            "schema_version": "1.0",
            "status": "fail",
            "promotion_decision": {"value": "failed", "rationale": "blocking_failures=2", "contributing_gate_ids": ["x"]},
            "summary": {"total_questions": None, "answerable_rate": None},
        },
    )
    sc = scorecard_service.build_scorecard("aq-101", libv2_root)
    section = sc["sections"]["assessment_quality"]
    assert section["available"] is True
    assert section["status"] == "fail"
    # Only value + rationale are surfaced, not the full nested gate set.
    assert section["promotion_decision"] == {"value": "failed", "rationale": "blocking_failures=2"}
    assert section["summary"]["total_questions"] is None


def test_coverage_map_section_compacts_orphans(libv2_root):
    cdir = _course_dir(libv2_root, "cov-101")
    _write(
        cdir / "coverage_map.json",
        {
            "schema_version": "1.0",
            "summary": {
                "objectives_with_chunks": 0,
                "orphan_chunks": ["c1", "c2", "c3"],
            },
        },
    )
    sc = scorecard_service.build_scorecard("cov-101", libv2_root)
    section = sc["sections"]["coverage_map"]
    assert section["available"] is True
    assert section["summary"]["orphan_chunks_count"] == 3
    assert "orphan_chunks" not in section["summary"]  # the id wall is dropped


def test_course_status_rides_along(libv2_root):
    cdir = _course_dir(libv2_root, "status-101")
    _write(cdir / "courseforge_promotion_chain_report.json", _chain_report_doc("certified_trainable"))
    sc = scorecard_service.build_scorecard("status-101", libv2_root)
    assert sc["course_status"] == "certified_trainable"


def test_malformed_report_degrades_to_not_evaluated(libv2_root):
    cdir = _course_dir(libv2_root, "bad-101")
    (cdir / "coverage_map.json").write_text("{ not json", encoding="utf-8")
    sc = scorecard_service.build_scorecard("bad-101", libv2_root)
    assert sc["sections"]["coverage_map"]["available"] is False


# --------------------------------------------------------------------------- #
# Router endpoint
# --------------------------------------------------------------------------- #


def test_router_scorecard_ok(client, libv2_root):
    cdir = _course_dir(libv2_root, "demo-101")
    _write(cdir / "coverage_map.json", {"summary": {"objectives_with_chunks": 5}})
    resp = client.get("/api/library/demo-101/scorecard")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["slug"] == "demo-101"
    assert body["sections"]["coverage_map"]["available"] is True
    assert body["sections"]["retrieval_eval"]["available"] is False


def test_router_scorecard_unknown_course_404(client, libv2_root):
    resp = client.get("/api/library/nope-999/scorecard")
    assert resp.status_code == 404
    assert resp.json()["error"] == "course_not_found"


def test_router_scorecard_bad_slug_422(client, libv2_root):
    resp = client.get("/api/library/..%2Fevil/scorecard")
    assert resp.status_code in (404, 422)
