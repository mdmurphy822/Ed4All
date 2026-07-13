"""Unit tests for the STANDALONE reasoning-QC runner (task #39).

The standalone runner judges an already-emitted ``*_accessible.html`` with the
Stage-9b reasoning-QC machinery WITHOUT re-running conversion (the omni-vs-Super
QC-seat A/B). These tests pin:

  (1) :func:`parse_accessible_html` → correct ordered block records (index /
      block_id / type / role / level / page / text) from a synthetic fixture;
  (2) :func:`run_standalone_qc` with a STUBBED judgment fn → report JSON shape +
      the input HTML is NOT mutated (report-only);
  (3) :func:`compare_reports` → correct both / only-A / only-B alignment;
  (4) the never-non-thinking + no-image guards hold on the STANDALONE path (the
      composed QC request bodies carry no image part and no thinking-off block),
      reusing the existing transport-level guard assertions.
"""
from __future__ import annotations

import json

import pytest

from semantik_structure import reasoning_qc, reasoning_qc_standalone, vlm_extract
from semantik_structure.extract_shared import VLMSeat


@pytest.fixture(autouse=True)
def _isolate_qc_cache(tmp_path, monkeypatch):
    """Isolate the default-ON per-unit resume cache into a per-test tmp dir so the
    identical inline fixture reused across tests never cross-contaminates via the
    shared on-disk sidecar (the standalone runner reuses the production
    `_fan_out_page_verifies` cache path verbatim)."""
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path / "qc_cache_root"))
    monkeypatch.delenv("SEMANTIK_REASONING_QC_CHECKPOINT", raising=False)
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)


# ---------------------------------------------------------------------------
# Synthetic accessible-HTML fixture (built inline — sections with
# data-dart-block-id, headings, paragraphs, a list + a table).
# ---------------------------------------------------------------------------
def _fixture_html() -> str:
    return """<!doctype html><html><head><title>T</title></head><body><article>
<h1>Doc Title</h1>
<section class="dart-section" data-dart-block-id="s0"
         data-dart-source="synthesized" data-dart-pages="1"
         data-dart-block-role="paragraph" data-dart-wcag="passed">
  <p>Adding integers with the same sign keeps the sign.</p>
</section>
<section class="dart-section" data-dart-block-id="hdr-2"
         data-dart-source="synthesized" data-dart-pages="2"
         data-dart-block-role="try_it" data-dart-wcag="passed">
  <h3>Try It 1.87</h3>
</section>
<section class="dart-section" data-dart-block-id="s1"
         data-dart-source="synthesized" data-dart-pages="2-3"
         data-dart-block-role="list" data-dart-wcag="passed">
  <ul><li>first</li><li>second</li></ul>
</section>
<section class="dart-section" data-dart-block-id="tbl-1"
         data-dart-source="synthesized" data-dart-pages="3"
         data-dart-wcag="passed">
  <table><tr><td>a</td><td>b</td></tr></table>
</section>
</article></body></html>"""


def _write_fixture(tmp_path) -> "any":
    p = tmp_path / "sample-ch01_accessible.html"
    p.write_text(_fixture_html(), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# (1) parse_accessible_html → ordered records.
# ---------------------------------------------------------------------------
def test_parse_accessible_html_ordered_records(tmp_path):
    p = _write_fixture(tmp_path)
    records = reasoning_qc_standalone.parse_accessible_html(p)

    # One record per <section data-dart-block-id> (the bare <h1> is NOT a section).
    assert [r["block_id"] for r in records] == ["s0", "hdr-2", "s1", "tbl-1"]
    assert [r["index"] for r in records] == [0, 1, 2, 3]

    # Type inference (coarse, documented) + role + level + page.
    by_id = {r["block_id"]: r for r in records}
    assert by_id["s0"]["type"] == "paragraph"
    assert by_id["s0"]["role"] == "paragraph"
    assert by_id["s0"]["level"] is None
    assert by_id["s0"]["page"] == 1

    # A section whose first significant descendant is a heading → type=heading + level.
    assert by_id["hdr-2"]["type"] == "heading"
    assert by_id["hdr-2"]["level"] == 3
    assert by_id["hdr-2"]["role"] == "try_it"  # data-dart-block-role wins the role
    assert by_id["hdr-2"]["page"] == 2

    assert by_id["s1"]["type"] == "list"
    assert by_id["s1"]["page"] == 2  # first int of "2-3"
    assert by_id["tbl-1"]["type"] == "table"
    assert "Adding integers" in by_id["s0"]["text"]


# ---------------------------------------------------------------------------
# (1b) REAL adapter shape — the live seam emits data-semantik-* (2026-07 rename),
# NOT data-dart-*. The parser must extract these (regression: it returned 0
# blocks against a real converter output whose sections use data-semantik-*).
# Placeholder text only — no real textbook sentences.
# ---------------------------------------------------------------------------
def _fixture_html_semantik() -> str:
    return """<!doctype html><html><head><title>T</title></head><body><article>
<h1>Doc Title</h1>
<section class="dart-section" id="s0" data-semantik-block-id="s0"
         data-semantik-source="synthesized" data-semantik-pages="1"
         data-semantik-page-kind="physical" data-semantik-block-role="paragraph"
         data-semantik-wcag="passed" data-semantik-demoted-role="definition_list">
  <p>Placeholder body sentence one for the first block.</p>
</section>
<section class="dart-section" id="example-9-9" data-semantik-block-id="example-9-9"
         data-semantik-source="synthesized" data-semantik-pages="2-3"
         data-semantik-block-role="worked_example" data-semantik-wcag="passed">
  <h3>Placeholder Example Heading</h3>
  <p>Placeholder solution walk-through text.</p>
</section>
<section class="dart-section" id="s7" data-semantik-block-id="s7"
         data-semantik-source="synthesized" data-semantik-pages="3"
         data-semantik-block-role="list" data-semantik-wcag="passed">
  <ol><li>alpha</li><li>beta</li></ol>
</section>
<section class="dart-section" id="tbl-2" data-semantik-block-id="tbl-2"
         data-semantik-source="synthesized" data-semantik-pages="4"
         data-semantik-wcag="passed">
  <table><tr><td>x</td><td>y</td></tr></table>
</section>
</article></body></html>"""


def test_parse_accessible_html_semantik_attributes(tmp_path):
    """The live data-semantik-* attribute spelling parses (was 0 blocks pre-fix)."""
    p = tmp_path / "real-ch09_accessible.html"
    p.write_text(_fixture_html_semantik(), encoding="utf-8")
    records = reasoning_qc_standalone.parse_accessible_html(p)

    assert [r["block_id"] for r in records] == ["s0", "example-9-9", "s7", "tbl-2"]
    assert [r["index"] for r in records] == [0, 1, 2, 3]

    by_id = {r["block_id"]: r for r in records}
    assert by_id["s0"]["type"] == "paragraph"
    assert by_id["s0"]["role"] == "paragraph"  # data-semantik-block-role wins
    assert by_id["s0"]["page"] == 1
    # A worked-example section leading with a heading → type=heading + level; the
    # data-semantik-block-role still carries the true role.
    assert by_id["example-9-9"]["type"] == "heading"
    assert by_id["example-9-9"]["level"] == 3
    assert by_id["example-9-9"]["role"] == "worked_example"
    assert by_id["example-9-9"]["page"] == 2  # first int of "2-3"
    assert by_id["s7"]["type"] == "list"
    assert by_id["tbl-2"]["type"] == "table"
    assert "Placeholder body sentence" in by_id["s0"]["text"]


# ---------------------------------------------------------------------------
# (2) run_standalone_qc — stubbed judgment, report shape, NO mutation.
# ---------------------------------------------------------------------------
def _local_seat():
    return VLMSeat(provider="local", base_url="http://localhost:11434/v1", api_key=None, model="stub-model")


def test_run_standalone_qc_report_shape_and_no_mutation(tmp_path, monkeypatch):
    p = _write_fixture(tmp_path)
    original_bytes = p.read_bytes()

    monkeypatch.setattr(reasoning_qc, "_resolve_qc_seat", _local_seat)
    monkeypatch.setattr(reasoning_qc, "_unload_seat", lambda seat: None)

    # STUBBED judgment fn (the module-level seam the shared fan-out calls): flag
    # local block index 1 as a phantom heading with confidence 0.9.
    def _stub_judgment(seat, pdf_path, page_num, blocks):
        return {
            "phantom_headings": [{"index": 1, "reason": "running header, not a section"}],
            "confidence": 0.9,
        }

    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", _stub_judgment)

    out_dir = tmp_path / "reports"
    report = reasoning_qc_standalone.run_standalone_qc(p, out_dir=out_dir, log=lambda m: None)

    # Input HTML is byte-identical — REPORT-ONLY (no reconcile, no rewrite).
    assert p.read_bytes() == original_bytes

    # Report file written at the tidy stem (trailing _accessible dropped).
    out_path = out_dir / "sample-ch01.qc_report.json"
    assert out_path.exists()
    disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert disk == report

    # Shape.
    assert report["schema"] == reasoning_qc_standalone.STANDALONE_QC_SCHEMA
    assert report["seat"]["model"] == "stub-model"
    assert report["seat"]["base_url"] == "http://localhost:11434/v1"
    assert report["units"]["n_blocks"] == 4
    assert report["units"]["windows"] == 1
    assert report["units"]["seams"] == 0
    assert report["units"]["judged"] == 1
    assert report["verdict_confidence"] == 0.9
    assert report["wall_clock_seconds"] >= 0.0
    assert "load_seconds" in report  # None here (lifecycle flag off / lib absent)

    # The finding carries the ABSOLUTE index + resolved block id.
    assert report["n_findings"] == 1
    finding = report["findings"][0]
    assert finding["index"] == 1
    assert finding["block_id"] == "hdr-2"
    assert finding["finding_kind"] == "example_as_heading"  # phantom → reviewer taxonomy


def test_run_standalone_qc_empty_verdict_zero_findings(tmp_path, monkeypatch):
    p = _write_fixture(tmp_path)
    monkeypatch.setattr(reasoning_qc, "_resolve_qc_seat", _local_seat)
    monkeypatch.setattr(reasoning_qc, "_unload_seat", lambda seat: None)
    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", lambda *a, **k: {})

    report = reasoning_qc_standalone.run_standalone_qc(p, out_dir=tmp_path / "r", log=lambda m: None)
    assert report["n_findings"] == 0
    assert report["findings"] == []
    assert report["units"]["qc_incomplete_count"] == 0


# ---------------------------------------------------------------------------
# (3) compare_reports — both / only-A / only-B alignment.
# ---------------------------------------------------------------------------
def _synthetic_report(tmp_path, name, *, findings, model, wall, incomplete=0, n_blocks=10):
    rep = {
        "schema": reasoning_qc_standalone.STANDALONE_QC_SCHEMA,
        "seat": {"provider": "local", "base_url": "http://x/v1", "model": model},
        "units": {"n_blocks": n_blocks, "qc_incomplete_count": incomplete},
        "order_divergence": 0,
        "n_findings": len(findings),
        "findings": findings,
        "wall_clock_seconds": wall,
        "load_seconds": None,
    }
    path = tmp_path / name
    path.write_text(json.dumps(rep), encoding="utf-8")
    return path


def test_compare_reports_alignment(tmp_path):
    a = _synthetic_report(
        tmp_path,
        "a.qc_report.json",
        findings=[
            {"block_id": "b1", "finding_kind": "example_as_heading"},
            {"block_id": "b2", "finding_kind": "mistyped_component"},
        ],
        model="omni",
        wall=10.0,
        incomplete=1,
    )
    b = _synthetic_report(
        tmp_path,
        "b.qc_report.json",
        findings=[
            {"block_id": "b1", "finding_kind": "example_as_heading"},  # shared
            {"block_id": "b3", "finding_kind": "example_misordered_from_body"},  # only-B
        ],
        model="super",
        wall=25.0,
        incomplete=0,
    )

    compare = reasoning_qc_standalone.compare_reports(a, b)
    assert compare["schema"] == reasoning_qc_standalone.QC_COMPARE_SCHEMA
    assert compare["findings"]["both"] == 1
    assert compare["findings"]["only_a"] == 1
    assert compare["findings"]["only_b"] == 1
    assert compare["findings"]["both_keys"] == [
        {"block_id": "b1", "finding_kind": "example_as_heading"}
    ]
    assert compare["seat_a"]["model"] == "omni"
    assert compare["seat_b"]["model"] == "super"
    assert compare["qc_incomplete"]["a_rate"] == 0.1
    assert compare["qc_incomplete"]["b_rate"] == 0.0
    assert compare["wall_clock_seconds"]["delta_b_minus_a"] == 15.0

    # Human-readable summary is renderable + mentions the seat models.
    summary = reasoning_qc_standalone._format_compare_summary(compare)
    assert "omni" in summary and "super" in summary
    assert "both=1" in summary


# ---------------------------------------------------------------------------
# (4) never-non-thinking + no-image guards on the STANDALONE path.
# ---------------------------------------------------------------------------
class _CapturingResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _CapturingRequests:
    """Records EVERY POST body; returns a canned empty-verdict QC completion."""

    def __init__(self):
        self.bodies: list = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.bodies.append(json)
        return _CapturingResp(
            {"choices": [{"message": {"content": '{"reading_order": [], "phantom_headings": []}'}}]}
        )


def _judgment_bodies(req) -> list:
    """Only the QC judgment POSTs (a chat turn with a messages list)."""
    return [b for b in req.bodies if isinstance(b, dict) and "messages" in b]


def test_standalone_path_no_image_no_thinking_off(tmp_path, monkeypatch):
    """Driving the REAL judgment path through the standalone runner emits QC
    requests that carry NO image part and NO thinking-off block (thinking ON)."""
    p = _write_fixture(tmp_path)

    # A loopback reasoning-QC seat so _resolve_qc_seat resolves without a key.
    for env in (
        "SEMANTIK_SPECIALIST_BASE_URL",
        "SEMANTIK_REASONING_QC_MODEL",
        "SEMANTIK_REASONING_QC_DISABLE_THINKING",
        "SEMANTIK_VLM_DISABLE_THINKING",
    ):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_BASE_URL", "http://localhost:11434/v1")

    req = _CapturingRequests()
    # The fan-out → _run_qc_judgment → reasoning_qc_vlm.run_qc_judgment uses this.
    monkeypatch.setattr(vlm_extract, "_lazy_requests", lambda: req)
    monkeypatch.setattr(reasoning_qc, "_unload_seat", lambda seat: None)

    report = reasoning_qc_standalone.run_standalone_qc(p, out_dir=tmp_path / "r", log=lambda m: None)
    assert report["n_findings"] == 0  # empty verdict from the canned completion

    bodies = _judgment_bodies(req)
    assert bodies, "expected at least one composed QC judgment POST"
    for body in bodies:
        # No thinking-off block (thinking stays ON by default on the QC path).
        # chat_template_kwargs may carry ONLY the Nemotron-3 reasoning_budget.
        ctk = body.get("chat_template_kwargs") or {}
        assert "thinking" not in ctk and "enable_thinking" not in ctk, body
        # No image part smuggled into the user turn.
        user = body["messages"][1]["content"]
        for part in user:
            assert part.get("type") != "image_url", body
            assert "data:image" not in str(part.get("text", "")), body
