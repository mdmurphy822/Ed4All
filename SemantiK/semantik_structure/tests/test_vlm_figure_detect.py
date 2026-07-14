"""Task #56 — VLM figure-DETECT lane.

Covers:
  (a) flag resolution + the DETECT_FIGURES dual-gate (flag-off byte-identity)
  (b) the deterministic accept gate — every rejection reason, esp. the two that
      matter: the PAGE-RASTER backstop and the TEXT-COLUMN (WCAG 1.4.5) guard
  (c) the coordinate-space contract for the text-column guard (the OCR-lane
      pixel-vs-point trap that a real bug lived in)
  (d) the DETECT call shape + tolerant parse
  (e) DecisionCapture FIRES on the call path with a dynamic rationale (REQUIRED
      — this is a new LLM call site)
  (f) the injection seam: bboxes land in shared['pages'][i]['images'] in the
      shape the existing ImageCandidate chain already consumes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantik_structure import vlm_figure_detect as vfd  # noqa: E402


class _Seat:
    base_url = "http://localhost:11434/v1"
    api_key = None
    model = "qwen2.5vl:7b"


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _chat(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 42}}


class _Requests:
    """Minimal requests double; records the POST bodies."""

    def __init__(self, content: str):
        self.content = content
        self.posts: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        self.posts.append({"url": url, "body": json})
        return _Resp(_chat(self.content))


# ---------------------------------------------------------------------------
# (a) flag resolution + dual-gate
# ---------------------------------------------------------------------------
def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(vfd.SEMANTIK_VLM_FIGURE_DETECT_ENV, raising=False)
    assert vfd.resolve_vlm_figure_detect_mode() is False


def test_flag_requires_detect_figures_dual_gate(monkeypatch):
    """Flag ON but SEMANTIK_DETECT_FIGURES off -> DISABLED (chapter-fatal guard)."""
    monkeypatch.setenv(vfd.SEMANTIK_VLM_FIGURE_DETECT_ENV, "1")
    monkeypatch.delenv("SEMANTIK_DETECT_FIGURES", raising=False)
    monkeypatch.delenv("SEMANTIK_DEPLOY_PROFILE", raising=False)
    assert vfd.resolve_vlm_figure_detect_mode() is False

    monkeypatch.setenv("SEMANTIK_DETECT_FIGURES", "1")
    assert vfd.resolve_vlm_figure_detect_mode() is True


def test_flag_parse_with_fallback(monkeypatch):
    monkeypatch.setenv("SEMANTIK_DETECT_FIGURES", "1")
    for garbage in ("", "  ", "banana", "0", "false", "off"):
        monkeypatch.setenv(vfd.SEMANTIK_VLM_FIGURE_DETECT_ENV, garbage)
        assert vfd.resolve_vlm_figure_detect_mode() is False
    for truthy in ("1", "true", "YES", "On"):
        monkeypatch.setenv(vfd.SEMANTIK_VLM_FIGURE_DETECT_ENV, truthy)
        assert vfd.resolve_vlm_figure_detect_mode() is True


def test_satellite_flags_parse_with_fallback(monkeypatch):
    for env, resolver, default in (
        (vfd._CONCURRENCY_ENV, vfd.resolve_detect_concurrency, 4),
        (vfd._MAX_PER_PAGE_ENV, vfd.resolve_detect_max_per_page, 8),
    ):
        for garbage in ("", "banana", "-3", "0"):
            monkeypatch.setenv(env, garbage)
            assert resolver() == default
    for garbage in ("", "banana", "-1", "nan", "inf"):
        monkeypatch.setenv(vfd._TIMEOUT_ENV, garbage)
        assert vfd.resolve_detect_timeout() == 600.0


def test_flag_off_injection_is_a_no_op(monkeypatch):
    """Flag OFF -> inject returns None and `shared` is BYTE-IDENTICAL."""
    monkeypatch.delenv(vfd.SEMANTIK_VLM_FIGURE_DETECT_ENV, raising=False)
    shared = {"pages": [{"page_num": 1, "width": 612.0, "height": 792.0}]}
    before = json.dumps(shared, sort_keys=True)

    audit = vfd.inject_figure_candidates(shared, "/nonexistent.pdf")

    assert audit is None
    assert json.dumps(shared, sort_keys=True) == before


# ---------------------------------------------------------------------------
# (b) THE DETERMINISTIC ACCEPT GATE — this is what decides.
# ---------------------------------------------------------------------------
def _accept(props, **kw):
    kw.setdefault("page_w", 612.0)
    kw.setdefault("page_h", 792.0)
    return vfd.accept_figure_boxes(props, **kw)


def test_accepts_a_genuine_sub_page_figure():
    accepted, stats = _accept([{"bbox": [0.2, 0.3, 0.7, 0.55], "kind": "diagram"}])
    assert stats["accepted"] == 1
    box = accepted[0]
    # PDF-POINT space, top-left origin — the ImageCandidate contract.
    assert box["bbox"] == pytest.approx([122.4, 237.6, 428.4, 435.6])
    assert box["px_size"][0] > 0 and box["px_size"][1] > 0
    assert box["vlm_figure"] is True
    assert box["vlm_kind"] == "diagram"


def test_page_raster_backstop_rejects_a_whole_page_return():
    """THE BACKSTOP. A VLM that returns the whole page must still be REJECTED."""
    accepted, stats = _accept([{"bbox": [0.0, 0.0, 1.0, 1.0], "kind": "photo"}])
    assert accepted == []
    assert stats["page_raster"] == 1


def test_page_raster_ceiling_is_the_shared_constant():
    """The accept gate and the downstream mandatory guard cannot drift."""
    from semantik_structure.structure_graph import _PAGE_RASTER_MIN_COVERAGE

    assert vfd._page_raster_min_coverage() == _PAGE_RASTER_MIN_COVERAGE

    # A box just under the ceiling survives the raster check; just over is cut.
    just_over = [{"bbox": [0.0, 0.0, 1.0, 0.95]}]
    assert _accept(just_over)[1]["page_raster"] == 1


def test_text_column_guard_rejects_a_body_text_crop():
    """A crop of body text is a WCAG 1.4.5 images-of-text REGRESSION, not a win."""
    # A left column densely covered by text lines.
    text = [(0.10, 0.10 + i * 0.03, 0.48, 0.12 + i * 0.03) for i in range(25)]
    accepted, stats = _accept(
        [{"bbox": [0.09, 0.09, 0.49, 0.85], "kind": "diagram"}],
        text_boxes_norm=text,
    )
    assert accepted == []
    assert stats["text_column"] == 1


def test_text_column_guard_keeps_a_figure_sitting_in_whitespace():
    """A genuine figure lives in whitespace — the guard must NOT false-reject it."""
    text = [(0.10, 0.05 + i * 0.03, 0.90, 0.07 + i * 0.03) for i in range(5)]  # top prose
    accepted, stats = _accept(
        [{"bbox": [0.25, 0.40, 0.75, 0.70], "kind": "number_line"}],
        text_boxes_norm=text,
    )
    assert stats["accepted"] == 1
    assert stats["text_column"] == 0
    assert accepted[0]["vlm_kind"] == "number_line"


def test_text_density_guard_rejects_a_procedure_table_the_coverage_arm_misses():
    """THE REAL DEFECT, from a live spot-check.

    A pure "Step 1..4" procedure TABLE — an unambiguous images-of-text crop —
    measured area-coverage 0.055, LOWER than a GENUINE counters figure at 0.180,
    because a table's word boxes are thin and separated by cell whitespace. No
    coverage threshold can order those two. Word COUNT separates them cleanly
    (table 45 words vs genuine figures 0/0/0/0/10), so the density arm is
    load-bearing and the coverage arm cannot replace it.
    """
    # A sparse table: few thin text boxes (low coverage) but MANY words.
    table = [
        ((0.12, 0.30 + i * 0.05, 0.45, 0.325 + i * 0.05),
         "Step %d Locate the given place value with an arrow and all digits" % i)
        for i in range(5)
    ]
    cov = vfd._text_coverage((0.10, 0.28, 0.90, 0.60), [bb for bb, _ in table])
    assert cov < vfd._MAX_TEXT_COVERAGE, "precondition: coverage arm does NOT fire"

    accepted, stats = _accept(
        [{"bbox": [0.10, 0.28, 0.90, 0.60], "kind": "diagram"}],
        text_items_norm=table,
    )
    assert accepted == []
    assert stats["text_dense"] == 1
    assert stats["text_column"] == 0  # proof the coverage arm was NOT the one that fired


def test_text_density_guard_keeps_a_genuine_figure_with_a_short_label():
    """A counters figure labelled "5 . 3  add 5, 3 times" must SURVIVE."""
    items = [((0.30, 0.40, 0.55, 0.43), "add 5, 3 times")]
    accepted, stats = _accept(
        [{"bbox": [0.25, 0.38, 0.60, 0.62], "kind": "manipulative"}],
        text_items_norm=items,
    )
    assert stats["accepted"] == 1
    assert stats["text_dense"] == 0


def test_text_density_guard_never_penalises_digits():
    """A number line / chart axis is ALL digits — it must never count as words."""
    axis = [((0.20 + i * 0.08, 0.50, 0.25 + i * 0.08, 0.53), str(i * 10))
            for i in range(9)]
    accepted, stats = _accept(
        [{"bbox": [0.15, 0.45, 0.85, 0.60], "kind": "number_line"}],
        text_items_norm=axis, max_words=3,
    )
    assert stats["accepted"] == 1, "digits are not words"


def test_text_density_guard_can_be_disabled():
    items = [((0.3, 0.4, 0.6, 0.5), " ".join(["word"] * 200))]
    accepted, _ = _accept(
        [{"bbox": [0.2, 0.3, 0.8, 0.7]}], text_items_norm=items, max_words=0
    )
    assert len(accepted) == 1


def test_page_text_items_carry_text_and_boxes(monkeypatch):
    page = {
        "page_num": 1, "width": 612.0, "height": 792.0, "tesseract_width": 1224,
        "merged": {"text_blocks": [{"bbox": [122.4, 158.4, 1101.6, 316.8],
                                    "text": "hello world"}]},
    }
    items = vfd.page_text_items_norm(page)
    assert items[0][1] == "hello world"
    assert items[0][0] == pytest.approx((0.10, 0.10, 0.90, 0.20))
    # the boxes wrapper stays consistent with the items
    assert vfd.page_text_boxes_norm(page) == [items[0][0]]


def test_overlapping_text_blocks_cannot_inflate_coverage_past_one():
    """OCR emits overlapping line+word blocks; area-summing would false-reject."""
    dupes = [(0.30, 0.45, 0.70, 0.50)] * 40  # same band, 40x over
    accepted, _stats = _accept(
        [{"bbox": [0.25, 0.40, 0.75, 0.70], "kind": "chart"}], text_boxes_norm=dupes
    )
    assert len(accepted) == 1  # one thin band != a text column


def test_rejects_degenerate_and_out_of_page_boxes():
    _, stats = _accept(
        [
            {"bbox": [0.5, 0.5, 0.5, 0.5]},           # zero area
            {"bbox": [0.8, 0.8, 0.2, 0.2]},           # inverted
            {"bbox": [0.5, 0.5, 5000.0, 5000.0]},     # beyond every convention -> out of page
            {"bbox": [1, 2, 3]},                       # malformed
            {"bbox": ["a", "b", "c", "d"]},           # malformed
            {"nope": 1},                               # malformed
        ]
    )
    assert stats["accepted"] == 0
    assert stats["degenerate"] >= 2
    assert stats["out_of_page"] == 1
    assert stats["malformed"] == 3


def test_ambiguous_out_of_range_box_is_rejected_somehow_never_accepted():
    """A box in the ambiguous 1..100 band is rescaled as a percentage; whatever
    reason it lands under, it must never be ACCEPTED. (Anything the gate cannot
    confidently read is not a figure.)"""
    accepted, stats = _accept([{"bbox": [0.5, 0.5, 9.9, 9.9]}])
    assert accepted == []
    assert stats["accepted"] == 0


def test_rejects_absurd_aspect_and_tiny_boxes():
    _, stats = _accept(
        [
            {"bbox": [0.0, 0.50, 1.0, 0.505]},   # hairline rule
            {"bbox": [0.40, 0.40, 0.42, 0.42]},  # speck
        ]
    )
    assert stats["accepted"] == 0
    assert stats["too_small"] + stats["bad_aspect"] == 2


def test_dedups_overlapping_proposals_and_caps_per_page():
    dupes = [{"bbox": [0.2, 0.2, 0.6, 0.6]}, {"bbox": [0.21, 0.21, 0.61, 0.61]}]
    accepted, stats = _accept(dupes)
    assert len(accepted) == 1
    assert stats["duplicate"] == 1

    many = [{"bbox": [0.05 + i * 0.09, 0.1, 0.12 + i * 0.09, 0.3]} for i in range(9)]
    accepted, stats = _accept(many, max_per_page=3)
    assert len(accepted) == 3
    assert stats["over_cap"] >= 1


def test_percentage_coordinates_are_rescaled():
    """Some models emit 0-100 instead of fractions."""
    accepted, stats = _accept([{"bbox": [20, 30, 70, 55]}])
    assert stats["accepted"] == 1
    assert accepted[0]["bbox"] == pytest.approx([122.4, 237.6, 428.4, 435.6])


# ---------------------------------------------------------------------------
# (c) coordinate-space contract for the text-column guard
# ---------------------------------------------------------------------------
def test_text_boxes_normalized_by_PIXEL_dims_on_the_ocr_lane():
    """The pixel-vs-point trap: normalizing by the wrong dims defeats the guard.

    On the OCR lane merged text bboxes are IMAGE-PIXEL space (1224x1584 at
    render scale 2.0) while page width/height are PDF POINTS (612x792).
    Normalizing by the points would put a full-width text line at ~2.0 (clamped
    to 1.0) or, for a half-page block, at 25% — silently letting body-text crops
    through. This pins the pixel-dim normalization.
    """
    page = {
        "page_num": 1,
        "width": 612.0,
        "height": 792.0,
        "tesseract_width": 1224,  # render scale 2.0
        "merged": {"text_blocks": [{"bbox": [122.4, 158.4, 1101.6, 316.8]}]},
    }
    boxes = vfd.page_text_boxes_norm(page)
    assert len(boxes) == 1
    # 122.4/1224 = 0.10 ; 158.4/1584 = 0.10 ; 1101.6/1224 = 0.90 ; 316.8/1584 = 0.20
    assert boxes[0] == pytest.approx((0.10, 0.10, 0.90, 0.20))


def test_text_boxes_normalized_by_POINT_dims_on_the_born_digital_lane():
    page = {
        "page_num": 1,
        "width": 612.0,
        "height": 792.0,
        "merged": {"text_blocks": [{"bbox": [61.2, 79.2, 550.8, 158.4]}]},
    }
    boxes = vfd.page_text_boxes_norm(page)
    assert boxes[0] == pytest.approx((0.10, 0.10, 0.90, 0.20))


# ---------------------------------------------------------------------------
# (d) the DETECT call
# ---------------------------------------------------------------------------
def test_detect_call_shape_carries_the_image_and_thinks_by_default(monkeypatch):
    """Thinking is ON by default — localization is a spatial REASONING task.

    MEASURED on the live seat: with `thinking: false` the model returns
    {"figures": []} on EVERY page, including one carrying a full-width photograph
    it describes perfectly in prose. Thinking-on, the same prompt boxes it
    correctly. So the request must NOT carry a thinking-off directive by default.
    """
    monkeypatch.delenv(vfd._DISABLE_THINKING_ENV, raising=False)
    req = _Requests('{"figures": [{"bbox": [0.2,0.3,0.7,0.55], "kind": "diagram"}]}')
    props, _usage, _wall = vfd.detect_page_figures(
        _Seat(), "BASE64BYTES", 7, requests_module=req
    )
    assert props == [{"bbox": [0.2, 0.3, 0.7, 0.55], "kind": "diagram"}]
    body = req.posts[0]["body"]
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    # No thinking-off directive -> the seat reasons.
    assert "chat_template_kwargs" not in body
    content = body["messages"][1]["content"]
    assert any(p.get("type") == "image_url" for p in content)


def test_thinking_can_be_disabled_by_the_operator(monkeypatch):
    monkeypatch.setenv(vfd._DISABLE_THINKING_ENV, "1")
    req = _Requests('{"figures": []}')
    vfd.detect_page_figures(_Seat(), "IMG", 1, requests_module=req)
    kw = req.posts[0]["body"]["chat_template_kwargs"]
    assert kw["thinking"] is False and kw["enable_thinking"] is False


def test_thinking_mode_salts_the_resume_fingerprint(monkeypatch):
    """A cross-mode cache HIT would be a WRONG verdict (thinking-off finds nothing)."""
    monkeypatch.delenv(vfd._DISABLE_THINKING_ENV, raising=False)
    think_on = vfd._page_fingerprint("IMG", model="m", page_label=1)
    monkeypatch.setenv(vfd._DISABLE_THINKING_ENV, "1")
    think_off = vfd._page_fingerprint("IMG", model="m", page_label=1)
    assert think_on != think_off


class _EmptyReq:
    """A seat whose thinker exhausted the window -> content=None, finish=length."""

    posts: list = []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        return _Resp(
            {"choices": [{"message": {"content": None}, "finish_reason": "length"}],
             "usage": {}}
        )


def test_empty_completion_is_loud_and_is_a_NON_JUDGMENT(caplog):
    """A runaway thinker losing a page must WARN — silent loss is the 1.1.1 bug."""
    import logging

    with caplog.at_level(logging.WARNING):
        with pytest.raises(vfd.DetectNonJudgment):
            vfd.detect_page_figures(_Seat(), "IMG", 42, requests_module=_EmptyReq())
    assert any("EMPTY completion" in r.message and "42" in r.message
               for r in caplog.records)


def test_a_non_judgment_is_NEVER_written_to_the_resume_sidecar(monkeypatch, tmp_path):
    """THE BUG THIS PINS: caching a transient blip as "no figures on this page"
    turns it into PERMANENT, SILENT figure loss — every later --resume serves the
    cached [] and never re-POSTs. Only a GENUINE verdict may be persisted
    (page_arranger caches only status=='ok'; reasoning_qc never caches
    _qc_incomplete)."""
    monkeypatch.setattr(vfd, "_cache_root", lambda: tmp_path)
    monkeypatch.delenv(vfd._CHECKPOINT_ENV, raising=False)
    monkeypatch.delenv(vfd._CHECKPOINT_FAMILY_ENV, raising=False)
    assert vfd.resolve_detect_checkpoint() is True

    dp = vfd._DetectPage(page_label=7, image_b64="IMG", page_w=612.0, page_h=792.0,
                         text_items=[], page_ref={})
    out = vfd._fan_out_detect(_Seat(), [dp], log=lambda _m: None,
                              requests_module=_EmptyReq())
    assert out == {7: []}  # the page degrades alone
    assert list(tmp_path.rglob("*.json")) == [], "a NON-judgment must not be cached"

    # ...but a GENUINE empty verdict IS cached (it is a real answer).
    out = vfd._fan_out_detect(_Seat(), [dp], log=lambda _m: None,
                              requests_module=_Requests('{"figures": []}'))
    assert out == {7: []}
    assert len(list(tmp_path.rglob("*.json"))) == 1, "a genuine verdict IS cached"


def test_thousand_scale_coordinates_are_rescaled():
    """The Qwen-VL 0..1000 grid — observed live from this seat under a direct
    grounding ask. Treating it as out-of-page would silently discard EVERY figure."""
    accepted, stats = _accept([{"bbox": [200, 300, 700, 550]}])
    assert stats["accepted"] == 1
    assert accepted[0]["bbox"] == pytest.approx([122.4, 237.6, 428.4, 435.6])


def test_detect_never_posts_blind_without_a_page_image():
    req = _Requests("{}")
    props, usage, wall = vfd.detect_page_figures(_Seat(), "", 1, requests_module=req)
    assert props == [] and usage == {} and wall == 0.0
    assert req.posts == []  # no POST at all


def test_malformed_body_is_a_NON_JUDGMENT_not_an_empty_verdict():
    """A bad body must RAISE DetectNonJudgment, never masquerade as "no figures".

    The distinction is load-bearing for the resume sidecar (see the caching test
    below): an empty verdict is CACHED, a non-judgment must never be.
    """
    for junk in ("not json at all", '{"figures": "nope"}', "```json\n{}\n```"):
        with pytest.raises(vfd.DetectNonJudgment):
            vfd.detect_page_figures(_Seat(), "IMG", 1, requests_module=_Requests(junk))


def test_a_genuine_empty_verdict_is_returned_not_raised():
    props, _u, _w = vfd.detect_page_figures(
        _Seat(), "IMG", 1, requests_module=_Requests('{"figures": []}')
    )
    assert props == []  # "this page really has no figures" — cacheable


def test_detect_strips_markdown_fences():
    req = _Requests('```json\n{"figures": [{"bbox": [0.1,0.1,0.5,0.5]}]}\n```')
    props, _u, _w = vfd.detect_page_figures(_Seat(), "IMG", 1, requests_module=req)
    assert len(props) == 1


# ---------------------------------------------------------------------------
# (e) DecisionCapture MUST fire on the call path (new LLM call site).
# ---------------------------------------------------------------------------
def test_decision_capture_fires_with_a_dynamic_rationale(monkeypatch, tmp_path):
    """REQUIRED regression: the capture fires, and the rationale interpolates
    DYNAMIC signals (page count, proposal/accept counts, per-guard rejection
    tallies, bbox areas, model id) — static boilerplate is forbidden."""
    logged: list[dict] = []

    class _Cap:
        def __init__(self, **kw):
            self.kw = kw

        def log_decision(self, **kw):
            logged.append(kw)

    import types

    fake = types.ModuleType("lib.decision_capture")
    fake.DecisionCapture = _Cap
    lib_mod = types.ModuleType("lib")
    lib_mod.decision_capture = fake
    monkeypatch.setitem(sys.modules, "lib", lib_mod)
    monkeypatch.setitem(sys.modules, "lib.decision_capture", fake)

    monkeypatch.setenv(vfd.SEMANTIK_VLM_FIGURE_DETECT_ENV, "1")
    monkeypatch.setenv("SEMANTIK_DETECT_FIGURES", "1")
    monkeypatch.setenv(vfd._CHECKPOINT_ENV, "0")  # no sidecar in the test

    shared = {
        "pages": [
            {
                "page_num": 3,
                "width": 612.0,
                "height": 792.0,
                "merged": {"text_blocks": []},
            }
        ]
    }
    # One genuine figure + one whole-page raster the backstop must reject.
    req = _Requests(
        '{"figures": ['
        '{"bbox": [0.2,0.3,0.7,0.55], "kind": "diagram", "confidence": 0.9},'
        '{"bbox": [0.0,0.0,1.0,1.0], "kind": "photo"}]}'
    )

    audit = vfd.inject_figure_candidates(
        shared,
        "/fake.pdf",
        seat=_Seat(),
        requests_module=req,
        render_page_image=lambda _p, _n: "IMGB64",
    )

    assert len(logged) == 1, "exactly one structure_detection capture per document"
    row = logged[0]
    assert row["decision_type"] == "structure_detection"
    rationale = row["rationale"]
    assert len(rationale) >= 20

    # DYNAMIC signals — each must actually appear.
    assert "qwen2.5vl:7b" in rationale            # model id
    assert "PROPOSED 2" in rationale               # proposal count
    assert "page_raster=1" in rationale            # the backstop tally
    assert "text_column=0" in rationale            # the 1.4.5 guard tally
    assert "1 page(s)" in rationale                # page count
    assert "0.1563" in rationale or "area" in rationale.lower()  # bbox areas
    assert "(3, 1)" in rationale                   # busiest page = page 3, 1 figure

    # The gate DECIDED: 1 accepted, the page raster rejected.
    assert audit["totals"]["accepted"] == 1
    assert audit["totals"]["page_raster"] == 1


def test_capture_failure_never_breaks_detection(monkeypatch):
    """Best-effort posture — a broken capture must not fail the conversion."""
    import types

    class _Boom:
        def __init__(self, **kw):
            raise RuntimeError("capture backend down")

    fake = types.ModuleType("lib.decision_capture")
    fake.DecisionCapture = _Boom
    monkeypatch.setitem(sys.modules, "lib.decision_capture", fake)

    monkeypatch.setenv(vfd.SEMANTIK_VLM_FIGURE_DETECT_ENV, "1")
    monkeypatch.setenv("SEMANTIK_DETECT_FIGURES", "1")
    monkeypatch.setenv(vfd._CHECKPOINT_ENV, "0")

    shared = {"pages": [{"page_num": 1, "width": 612.0, "height": 792.0}]}
    audit = vfd.inject_figure_candidates(
        shared,
        "/fake.pdf",
        seat=_Seat(),
        requests_module=_Requests('{"figures": [{"bbox":[0.2,0.3,0.7,0.55]}]}'),
        render_page_image=lambda _p, _n: "IMG",
    )
    assert audit["totals"]["accepted"] == 1  # detection completed regardless


# ---------------------------------------------------------------------------
# (f) the injection seam
# ---------------------------------------------------------------------------
def test_injection_appends_in_the_shape_the_existing_chain_consumes(monkeypatch):
    """The injected dicts must be consumable by detect_image_region_candidates."""
    monkeypatch.setenv(vfd.SEMANTIK_VLM_FIGURE_DETECT_ENV, "1")
    monkeypatch.setenv("SEMANTIK_DETECT_FIGURES", "1")
    monkeypatch.setenv(vfd._CHECKPOINT_ENV, "0")

    # The page already carries its full-page raster (as a real scan does).
    raster = {"bbox": [0.0, 0.0, 612.0, 792.0], "px_size": [1224, 1584], "index": 0}
    shared = {
        "pages": [
            {
                "page_num": 1,
                "width": 612.0,
                "height": 792.0,
                "images": [raster],
                "merged": {"text_blocks": []},
            }
        ]
    }
    vfd.inject_figure_candidates(
        shared,
        "/fake.pdf",
        seat=_Seat(),
        requests_module=_Requests('{"figures": [{"bbox":[0.2,0.3,0.7,0.55]}]}'),
        render_page_image=lambda _p, _n: "IMG",
    )

    images = shared["pages"][0]["images"]
    # APPENDED, not replaced — the raster survives for the guard to drop.
    assert len(images) == 2
    assert images[0] is raster
    assert images[1]["index"] == 1  # re-indexed after the existing entries

    # The whole existing chain must consume it: ImageCandidate forms, and the
    # page-raster guard drops the raster while KEEPING the detected figure.
    from semantik_structure.region_detection import detect_image_region_candidates
    from semantik_structure.structure_graph import (
        is_page_raster_candidate,
        page_dims_from_shared,
    )

    cands = detect_image_region_candidates(shared)
    assert len(cands) == 2
    dims = page_dims_from_shared(shared)
    kept = [c for c in cands if not is_page_raster_candidate(c, dims)]
    assert len(kept) == 1, "the page raster is dropped, the VLM figure survives"
    assert kept[0].bbox == pytest.approx((122.4, 237.6, 428.4, 435.6))


def test_graceful_stop_propagates_and_is_never_degraded_to_detect_failed(monkeypatch):
    """A STOP is a control SIGNAL, not an error.

    ``CascadeStopRequested`` subclasses ``RuntimeError``, so the seam's fail-soft
    ``except Exception`` would happily swallow it — logging a benign "detect
    failed (non-fatal)" and then ploughing on into featurize -> arrange -> Stage 6
    (hours of further work) instead of PAUSING, dead-ending the whole per-page
    stop/checkpoint machinery. This pins the re-raise.
    """
    from semantik_structure import page_arranger as pa
    from semantik_structure.stop_seam import CascadeStopRequested

    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER", "1")
    monkeypatch.setenv(vfd.SEMANTIK_VLM_FIGURE_DETECT_ENV, "1")
    monkeypatch.setenv("SEMANTIK_DETECT_FIGURES", "1")

    monkeypatch.setattr(
        pa, "resolve_page_arranger_mode", lambda: True, raising=False
    )
    import semantik_structure.extract_shared as es
    import semantik_structure.features as ft

    monkeypatch.setattr(
        es, "extract_shared_cached", lambda _p: {"pages": []}, raising=False
    )
    monkeypatch.setattr(
        ft, "featurize_with_regions", lambda _s: pytest.fail("must not reach featurize")
    )

    def _boom(*_a, **_kw):
        raise CascadeStopRequested("stop requested mid-detect")

    monkeypatch.setattr(vfd, "inject_figure_candidates", _boom, raising=False)

    with pytest.raises(CascadeStopRequested):
        pa.resolve_page_arranger_route("/fake.pdf")


def test_resume_sidecar_fingerprint_is_content_addressed():
    a = vfd._page_fingerprint("IMGA", model="m1", page_label=1)
    assert a == vfd._page_fingerprint("IMGA", model="m1", page_label=1)  # stable
    assert a != vfd._page_fingerprint("IMGB", model="m1", page_label=1)  # image
    assert a != vfd._page_fingerprint("IMGA", model="m2", page_label=1)  # model
    assert a != vfd._page_fingerprint("IMGA", model="m1", page_label=2)  # page
