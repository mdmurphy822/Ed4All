"""Module tests for the page-arranger I/O half (page_arranger). ALL HTTP mocked.

Covers: flag-off byte-identical route probe (ZERO extraction), the 3-rung ladder
(temps / max_tokens / thinking-off / json response_format), the coverage
invariant, furniture -> metadata_drop listed-but-empty, the resume sidecar
(hit skips POST, a failed page is not cached), stop mid-fan-out
(persist + propagate), the deterministic hints + hints_provider stub, the
schema_version-2 label factory, the DecisionCapture emit, and the FIGURE arm
(task #49 — deterministic figure Regions + the mandatory page-raster guard).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantik_structure import page_arranger as pa
from semantik_structure.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Synthetic builders (NO course/corpus text).
# ---------------------------------------------------------------------------
def _fb(text, *, page=1, y0=0.0, is_image=False, source="tesseract"):
    raw = RawBlock(
        text=text, page=page, bbox=(0.0, y0, 100.0, y0 + 10.0),
        page_width=612.0, page_height=792.0, source=source,
    )
    return FeatureBlock(
        raw=raw, size_bucket="md", gap_above=None, is_top_of_page=False,
        is_centered=False, caps=None, indent_bucket=0, relative_font_ratio=1.0,
        provenance=source, is_image=is_image,
    )


class _Seat:
    def __init__(self, model="arranger-omni", base_url="http://localhost:8000", api_key=None):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = "local"

    @property
    def is_local(self):
        return True

    def require_ready(self):
        return self


class _Resp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}], "usage": {}}


class _FakeRequests:
    """Records every POST body; replays a queue of content strings."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.posts = []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        self.posts.append({"url": url, "body": json, "headers": headers, "timeout": timeout})
        content = self._contents.pop(0) if self._contents else '{"blocks":[]}'
        return _Resp(content)


def _arr(blocks, confidence=0.9):
    return json.dumps({"blocks": blocks, "confidence": confidence})


# ---------------------------------------------------------------------------
# (a) flag-off byte-identical route probe — ZERO extraction.
# ---------------------------------------------------------------------------
def test_route_probe_off_does_zero_extraction(monkeypatch):
    monkeypatch.delenv("SEMANTIK_PAGE_ARRANGER", raising=False)

    def _boom(*_a, **_k):
        raise AssertionError("extraction must NOT run when the flag is off")

    monkeypatch.setattr("semantik_structure.extract_shared.extract_shared_cached", _boom)
    assert pa.resolve_page_arranger_route("/nonexistent.pdf") is None


# ---------------------------------------------------------------------------
# (b) 3-rung ladder — temps / max_tokens / thinking-off / json mode.
# ---------------------------------------------------------------------------
def test_ladder_three_rungs_body_shape(monkeypatch):
    units = [{"id": "p1_b00", "text": "Hello"}, {"id": "p1_b01", "text": "World"}]
    # rung1 drops an id (invalid) -> rung2 still invalid -> rung3 valid.
    bad = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])  # missing p1_b01
    good = _arr([{"ids": ["p1_b00", "p1_b01"], "type": "paragraph"}])
    fake = _FakeRequests([bad, bad, good])
    res = pa.arrange_page(_Seat(), "IMGB64", units, 1, requests_module=fake)
    assert res["status"] == "ok"
    assert res["attempts"] == 3
    assert len(fake.posts) == 3
    temps = [p["body"]["temperature"] for p in fake.posts]
    assert temps == [0.0, 0.0, 0.3]
    for p in fake.posts:
        b = p["body"]
        assert b["max_tokens"] == 6144
        assert b["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}
        assert b["response_format"] == {"type": "json_object"}
    # rung 3 restated the full legal id set
    last_user = fake.posts[-1]["body"]["messages"][-1]["content"]
    text = last_user if isinstance(last_user, str) else last_user[0]["text"]
    assert "p1_b00" in text and "p1_b01" in text


def test_ladder_rung1_success_single_post():
    units = [{"id": "p1_b00", "text": "Hello"}]
    good = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    fake = _FakeRequests([good])
    res = pa.arrange_page(_Seat(), "IMG", units, 1, requests_module=fake)
    assert res["status"] == "ok" and res["attempts"] == 1
    assert len(fake.posts) == 1


# ---------------------------------------------------------------------------
# Helpers to drive arrange_regions with a mocked seat + render + HTTP.
# ---------------------------------------------------------------------------
def _shared_pages(n_pages=1, width=612.0, height=792.0):
    """The extraction's page list — PDF-POINT dims (the ImageCandidate bbox space).

    Load-bearing for the page-raster guard: it reads its page dims from HERE, never
    off a FeatureBlock (on the OCR lane those are IMAGE-PIXEL space). See
    ``structure_graph.is_page_raster_candidate``.
    """
    return {
        "pages": [
            {"page_num": i + 1, "width": width, "height": height}
            for i in range(n_pages)
        ]
    }


def _drive_arrange_regions(
    monkeypatch, feature_blocks, contents, *, seat=None, image_candidates=None,
    shared=None,
):
    seat = seat or _Seat()
    monkeypatch.setattr(pa, "resolve_arranger_seat", lambda: seat)
    monkeypatch.setattr(pa, "_render_page_image_b64", lambda _pdf, _pg: "IMGB64")
    fake = _FakeRequests(contents)
    route = pa.ArrangerRoute(
        pdf_path=Path("/x.pdf"),
        shared=_shared_pages(4) if shared is None else shared,
        feature_set=_FS(feature_blocks, image_candidates=image_candidates),
    )
    regions, audit = route.arrange_regions(feature_blocks, requests_module=fake)
    return regions, audit, fake


class _FS:
    def __init__(self, fbs, image_candidates=None):
        self.feature_blocks = fbs
        # Mirrors features.FeatureSet: `[]` when SEMANTIK_DETECT_FIGURES is off.
        self.image_candidates = image_candidates if image_candidates is not None else []


# ---------------------------------------------------------------------------
# (c) coverage invariant + (d) furniture -> metadata_drop listed.
# ---------------------------------------------------------------------------
def test_coverage_invariant_and_furniture_listed(monkeypatch):
    monkeypatch.delenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", raising=False)
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    fbs = [_fb("Chapter 1 Foundations 5"), _fb("Real body prose here.", y0=20.0)]
    content = _arr([
        {"ids": ["p1_b00"], "type": "furniture"},
        {"ids": ["p1_b01"], "type": "paragraph"},
    ])
    regions, audit, _fake = _drive_arrange_regions(monkeypatch, fbs, [content])
    assert audit["pages_valid"] == 1
    # every non-image FB claimed exactly once (arrange_regions asserts internally)
    claimed = sorted(i for r in regions for i in r.feature_block_indices)
    assert claimed == [0, 1]
    # furniture region is present (listed) but its rendered body is empty; the
    # raw furniture text is preserved in the payload.
    drop = [r for r in regions if r.kind == "metadata_drop"]
    assert len(drop) == 1
    assert drop[0].payload["text"] == ""
    assert "Chapter 1 Foundations" in drop[0].payload["furniture_text"]
    para = [r for r in regions if r.kind == "paragraph"]
    assert para[0].payload["typing_authority"] == "vlm-arranger"


def test_failed_page_emits_per_unit_paragraph_fallback(monkeypatch):
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    fbs = [_fb("A"), _fb("B", y0=20.0)]
    # every rung drops an id -> arrangement_failed
    bad = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    regions, audit, _f = _drive_arrange_regions(monkeypatch, fbs, [bad, bad, bad])
    assert audit["pages_failed"] == 1
    # content preserved: one paragraph region per unit, loud flag on each
    assert len(regions) == 2
    assert all(r.kind == "paragraph" for r in regions)
    assert all(r.payload.get("arrangement_failed") for r in regions)
    assert sorted(i for r in regions for i in r.feature_block_indices) == [0, 1]


# ---------------------------------------------------------------------------
# (e) resume sidecar — hit skips POST; a failed page is NOT cached.
# ---------------------------------------------------------------------------
def test_sidecar_hit_skips_post(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", raising=False)
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    fbs = [_fb("Body one.", y0=20.0)]
    good = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])

    _r1, _a1, fake1 = _drive_arrange_regions(monkeypatch, fbs, [good])
    assert len(fake1.posts) == 1  # cold: one POST
    _r2, _a2, fake2 = _drive_arrange_regions(monkeypatch, fbs, [good])
    assert len(fake2.posts) == 0  # warm: served from sidecar, ZERO POSTs


def test_failed_page_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", raising=False)
    fbs = [_fb("A"), _fb("B", y0=20.0)]
    bad = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])  # perpetually invalid
    _r1, _a1, fake1 = _drive_arrange_regions(monkeypatch, fbs, [bad, bad, bad])
    assert fake1.posts  # ran
    _r2, _a2, fake2 = _drive_arrange_regions(monkeypatch, fbs, [bad, bad, bad])
    assert len(fake2.posts) == 3  # NOT cached — re-ran the whole ladder


# ---------------------------------------------------------------------------
# (f) stop mid-fan-out — completed unit persists + CascadeStopRequested.
# ---------------------------------------------------------------------------
def test_stop_mid_fanout_persists_and_propagates(monkeypatch, tmp_path):
    from semantik_structure.stop_seam import CascadeStopRequested

    sentinel = tmp_path / "STOP"
    monkeypatch.setenv("SEMANTIK_STOP_SENTINEL", str(sentinel))
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", raising=False)
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CONCURRENCY", "1")
    # two pages; the stop fires after the first page completes.
    fbs = [_fb("Page one body.", page=1, y0=20.0), _fb("Page two body.", page=2, y0=20.0)]
    seat = _Seat()
    monkeypatch.setattr(pa, "resolve_arranger_seat", lambda: seat)
    monkeypatch.setattr(pa, "_render_page_image_b64", lambda _pdf, _pg: "IMG")

    good1 = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    good2 = _arr([{"ids": ["p2_b00"], "type": "paragraph"}])
    fake = _FakeRequests([good1, good2])

    calls = {"n": 0}

    def _stop_after_first():
        # allow the first submission; trip the stop before the second. When the
        # stop trips, materialize the REAL sentinel file so the pass-boundary
        # _check_stop (which re-probes the filesystem) raises for real.
        calls["n"] += 1
        if calls["n"] > 1:
            sentinel.write_text("stop")
            return True
        return False

    monkeypatch.setattr(pa, "_stop_requested", _stop_after_first)

    route = pa.ArrangerRoute(pdf_path=Path("/x.pdf"), shared={}, feature_set=_FS(fbs))
    with pytest.raises(CascadeStopRequested):
        route.arrange_regions(fbs, requests_module=fake)
    # page 1 completed + persisted to its sidecar (so a resume serves it).
    cache_root = pa._cache_root()
    persisted = list(cache_root.rglob("*.json"))
    assert persisted, "the completed page-1 unit must be checkpointed before the stop propagates"


# ---------------------------------------------------------------------------
# (g) deterministic hints in prompt + hints_provider stub plumbed.
# ---------------------------------------------------------------------------
def test_deterministic_hints_in_prompt_and_provider_merged():
    units = [
        {"id": "p1_b00", "text": "EXAMPLE 1.2"},
        {"id": "p1_b01", "text": "Some ordinary paragraph text goes here."},
    ]
    hints = pa.build_page_hints(units, running_header_ids=set())
    assert hints.get("p1_b00", "").startswith("pedagogical_label:")

    good = _arr([{"ids": ["p1_b00"], "type": "heading", "level": 2},
                 {"ids": ["p1_b01"], "type": "paragraph"}])
    fake = _FakeRequests([good])
    seen = {"calls": 0}

    def _provider(units_in):
        seen["calls"] += 1
        return {"p1_b01": "bertv2_hint:body"}

    res = pa.arrange_page(_Seat(), "IMG", units, 1, hints=hints,
                          hints_provider=_provider, requests_module=fake)
    assert res["status"] == "ok"
    assert seen["calls"] == 1  # provider called once for the page (rung 1)
    user = fake.posts[0]["body"]["messages"][-1]["content"]
    text = user if isinstance(user, str) else user[0]["text"]
    assert "Deterministic hints" in text
    assert "pedagogical_label" in text
    assert "bertv2_hint:body" in text  # provider label merged into rung 1


def test_running_header_detection_over_pages():
    ubp = {
        1: [{"id": "p1_b00", "text": "Chapter 9 Roots 803"}, {"id": "p1_b01", "text": "body a"}],
        2: [{"id": "p2_b00", "text": "Chapter 9 Roots 815"}, {"id": "p2_b01", "text": "body b"}],
        3: [{"id": "p3_b00", "text": "Chapter 9 Roots 827"}, {"id": "p3_b01", "text": "body c"}],
    }
    hdr = pa._detect_running_header_ids(ubp)
    # the number-masked "chapter # roots #" signature recurs at page-top on all pages
    assert "p1_b00" in hdr and "p2_b00" in hdr and "p3_b00" in hdr
    assert "p1_b01" not in hdr


# ---------------------------------------------------------------------------
# (h) label factory — dir set writes v2 records + schema; unset writes nothing.
# ---------------------------------------------------------------------------
def test_train_records_written_when_dir_set(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    train_dir = tmp_path / "train"
    monkeypatch.setenv("SEMANTIK_ARRANGER_TRAIN_RECORDS_DIR", str(train_dir))
    fbs = [_fb("Body prose.", y0=20.0)]
    good = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    _drive_arrange_regions(monkeypatch, fbs, [good])
    recs = list(train_dir.glob("train_p*.json"))
    assert len(recs) == 1
    doc = json.loads(recs[0].read_text())
    assert doc["schema_version"] == 2
    assert doc["units"][0]["id"] == "p1_b00"
    assert (train_dir / "relations_schema.json").exists()


def test_train_records_unset_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    monkeypatch.delenv("SEMANTIK_ARRANGER_TRAIN_RECORDS_DIR", raising=False)
    fbs = [_fb("Body prose.", y0=20.0)]
    good = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    _drive_arrange_regions(monkeypatch, fbs, [good])
    # no train dir was created anywhere under tmp_path
    assert not list(tmp_path.rglob("train_p*.json"))


# ---------------------------------------------------------------------------
# (i) DecisionCapture fires with dynamic rationale.
# ---------------------------------------------------------------------------
def test_decision_capture_fires_dynamic(monkeypatch):
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    logged = []

    class _Cap:
        def __init__(self, **_k):
            pass

        def log_decision(self, decision_type, decision, rationale, **_k):
            logged.append((decision_type, decision, rationale))

    import lib.decision_capture as dc
    monkeypatch.setattr(dc, "DecisionCapture", _Cap)

    fbs = [_fb("Body prose.", y0=20.0)]
    good = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    _drive_arrange_regions(monkeypatch, fbs, [good])
    assert logged, "one structure_detection decision should fire per document"
    dtype, _dec, rationale = logged[0]
    assert dtype == "structure_detection"
    assert "pages=1" in rationale and "valid=1" in rationale
    assert len(rationale) >= 20


# ---------------------------------------------------------------------------
# (j) heading-sanity post-pass: glued mega-heading -> demote + tail title.
# ---------------------------------------------------------------------------
# The glued-mega-heading pattern the gate-1 live run produced: extraction fused a
# page-top band (running header + two TRY-IT apparatus openers) with the real
# section title trapped at the tail, and the arrange model typed the WHOLE fused
# unit as one heading.
_GLUED_MEGA = (
    "Chapter 1 Foundations 83\n"
    "TRY IT :: 1.93 Simplify the expression by combining like terms.\n"
    "TRY IT :: 1.94 Evaluate the following for the given value.\n"
    "Divide Integers"
)


def test_heading_sanity_glued_unit_demoted_and_tail_extracted():
    units = [{"id": "p1_b00", "text": _GLUED_MEGA}]
    good = _arr([{"ids": ["p1_b00"], "type": "heading", "level": 1}])
    fake = _FakeRequests([good])
    res = pa.arrange_page(_Seat(), "IMG", units, 1, requests_module=fake)
    assert res["status"] == "ok"
    blocks = res["arrangement"]["blocks"]
    # original mega-heading block DEMOTED to paragraph (keeps full text via ids)
    assert blocks[0]["type"] == "paragraph"
    assert blocks[0]["ids"] == ["p1_b00"]
    assert blocks[0]["heading_sanity"]["reason"] == "too_long"
    # a SYNTHETIC tail-title heading is inserted AFTER it
    synth = blocks[1]
    assert synth["synthetic_heading"] is True
    assert synth["type"] == "heading"
    assert synth["ids"] == []
    assert synth["text"] == "Divide Integers"
    assert synth["duplicate_of_tail"] is True
    assert synth["synthesized_from"] == "p1_b00"
    # offsets point at the VERBATIM tail slice of the source unit text
    start, end = synth["tail_offsets"]
    assert _GLUED_MEGA[start:end] == "Divide Integers"
    # recorded in the heading_sanity ledger (label factory learns from it)
    recs = res["heading_sanity"]
    assert len(recs) == 1
    assert recs[0]["op"] == "heading_demote"
    assert recs[0]["tail_title"] == "Divide Integers"
    assert recs[0]["tail_offsets"] == [start, end]


def test_heading_sanity_glued_unit_region_coverage(monkeypatch):
    # end-to-end through arrange_regions: coverage invariant holds and the
    # synthetic heading Region claims ZERO FeatureBlocks (no double-count).
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    fbs = [_fb(_GLUED_MEGA), _fb("Ordinary following paragraph body.", y0=40.0)]
    content = _arr([
        {"ids": ["p1_b00"], "type": "heading", "level": 1},
        {"ids": ["p1_b01"], "type": "paragraph"},
    ])
    regions, audit, _f = _drive_arrange_regions(monkeypatch, fbs, [content])
    assert audit["pages_valid"] == 1
    # coverage invariant asserted internally; both real FBs claimed exactly once
    claimed = sorted(i for r in regions for i in r.feature_block_indices)
    assert claimed == [0, 1]
    # the demoted mega-heading is now a paragraph Region owning FB 0
    demoted = [r for r in regions if r.kind == "paragraph"
               and 0 in r.feature_block_indices]
    assert demoted and demoted[0].payload["arrange_type"] == "paragraph"
    # the synthetic tail-title heading: verbatim title, NO FBs, provenance stamped
    synth = [r for r in regions if r.kind == "heading"
             and r.payload.get("duplicate_of_tail")]
    assert len(synth) == 1
    assert synth[0].feature_block_indices == ()
    assert synth[0].payload["text"] == "Divide Integers"
    assert synth[0].provenance["pass"] == "heading_sanity_tail_title"
    assert synth[0].provenance["synthesized_from"] == "p1_b00"
    # audit tallies the demotion + tail title
    row = audit["page_rows"][0]
    assert row["heading_sanity"] == 1
    assert row["heading_sanity_tail_titles"] == 1


def test_heading_sanity_clean_short_heading_untouched():
    units = [{"id": "p1_b00", "text": "Divide Integers"},
             {"id": "p1_b01", "text": "Body prose that follows the section title."}]
    good = _arr([{"ids": ["p1_b00"], "type": "heading", "level": 2},
                 {"ids": ["p1_b01"], "type": "paragraph"}])
    fake = _FakeRequests([good])
    res = pa.arrange_page(_Seat(), "IMG", units, 1, requests_module=fake)
    assert res["status"] == "ok"
    blocks = res["arrangement"]["blocks"]
    assert len(blocks) == 2  # no synthetic block inserted
    assert blocks[0]["type"] == "heading"  # clean short heading untouched
    assert "heading_sanity" not in blocks[0]
    assert res["heading_sanity"] == []


def test_heading_sanity_furniture_only_glued_heading_demoted_to_furniture():
    # a heading whose ENTIRE text is furniture (running header + page number) ->
    # demoted to furniture, and NO synthetic tail title is emitted.
    units = [{"id": "p1_b00", "text": "Chapter 1 Foundations 83"}]
    good = _arr([{"ids": ["p1_b00"], "type": "heading", "level": 1}])
    fake = _FakeRequests([good])
    res = pa.arrange_page(_Seat(), "IMG", units, 1, requests_module=fake)
    assert res["status"] == "ok"
    blocks = res["arrangement"]["blocks"]
    assert len(blocks) == 1  # no synthetic heading
    assert blocks[0]["type"] == "furniture"
    assert blocks[0]["heading_sanity"]["reason"] == "running_header"
    recs = res["heading_sanity"]
    assert len(recs) == 1 and recs[0]["to"] == "furniture"
    assert "tail_title" not in recs[0]


def test_heading_sanity_flag_off_byte_identical():
    import os
    units = [{"id": "p1_b00", "text": _GLUED_MEGA}]
    good = _arr([{"ids": ["p1_b00"], "type": "heading", "level": 1}])

    def _run():
        fake = _FakeRequests([good])
        return pa.arrange_page(_Seat(), "IMG", units, 1, requests_module=fake)

    # flag ON (default): demotes + inserts synthetic
    on = _run()
    assert len(on["arrangement"]["blocks"]) == 2
    # flag OFF: byte-identical to the raw arrangement (single untouched heading)
    prev = os.environ.get("SEMANTIK_ARRANGER_HEADING_SANITY")
    os.environ["SEMANTIK_ARRANGER_HEADING_SANITY"] = "0"
    try:
        off = _run()
    finally:
        if prev is None:
            os.environ.pop("SEMANTIK_ARRANGER_HEADING_SANITY", None)
        else:
            os.environ["SEMANTIK_ARRANGER_HEADING_SANITY"] = prev
    assert off["heading_sanity"] == []
    assert len(off["arrangement"]["blocks"]) == 1
    assert off["arrangement"]["blocks"][0]["type"] == "heading"
    assert "heading_sanity" not in off["arrangement"]["blocks"][0]


def test_heading_sanity_max_chars_override():
    import os
    # a heading UNDER 120 but OVER a lowered ceiling -> demoted; the tail title is
    # the final sentence-less segment.
    text = "This is a moderately long heading that exceeds a tight ceiling Widget"
    units = [{"id": "p1_b00", "text": text}]
    good = _arr([{"ids": ["p1_b00"], "type": "heading", "level": 2}])
    prev = os.environ.get("SEMANTIK_ARRANGER_HEADING_MAX_CHARS")
    os.environ["SEMANTIK_ARRANGER_HEADING_MAX_CHARS"] = "20"
    try:
        fake = _FakeRequests([good])
        res = pa.arrange_page(_Seat(), "IMG", units, 1, requests_module=fake)
    finally:
        if prev is None:
            os.environ.pop("SEMANTIK_ARRANGER_HEADING_MAX_CHARS", None)
        else:
            os.environ["SEMANTIK_ARRANGER_HEADING_MAX_CHARS"] = prev
    assert res["heading_sanity"][0]["reason"] == "too_long"
    assert res["arrangement"]["blocks"][0]["type"] == "paragraph"


def test_resolve_heading_sanity_defaults(monkeypatch):
    monkeypatch.delenv("SEMANTIK_ARRANGER_HEADING_SANITY", raising=False)
    monkeypatch.delenv("SEMANTIK_ARRANGER_HEADING_MAX_CHARS", raising=False)
    assert pa.resolve_arranger_heading_sanity() is True
    assert pa.resolve_arranger_heading_max_chars() == 120
    monkeypatch.setenv("SEMANTIK_ARRANGER_HEADING_SANITY", "off")
    assert pa.resolve_arranger_heading_sanity() is False
    monkeypatch.setenv("SEMANTIK_ARRANGER_HEADING_SANITY", "garbage")
    assert pa.resolve_arranger_heading_sanity() is True  # parse-with-fallback -> on
    monkeypatch.setenv("SEMANTIK_ARRANGER_HEADING_MAX_CHARS", "-5")
    assert pa.resolve_arranger_heading_max_chars() == 120
    monkeypatch.setenv("SEMANTIK_ARRANGER_HEADING_MAX_CHARS", "40")
    assert pa.resolve_arranger_heading_max_chars() == 40


# ---------------------------------------------------------------------------
# (k) LEADING-TITLE split (outline-anchored) — the second glue variant: the
# section title fused at the HEAD of the following paragraph unit.
# ---------------------------------------------------------------------------
_OUTLINE_UNIT = (
    "Chapter Outline\n"
    "1.4 Multiply Integers\n"
    "1.5 Divide Integers"
)
_GLUED_PARAGRAPH = (
    "Divide Integers What about division? Division is the inverse of "
    "multiplication, so we can use multiplication facts."
)


def _drive_two_pages(monkeypatch, fbs, contents):
    """Drive arrange_regions over a multi-page doc with concurrency 1 (so the
    _FakeRequests reply queue maps to sorted page order deterministically)."""
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CONCURRENCY", "1")
    return _drive_arrange_regions(monkeypatch, fbs, contents)


def test_leading_title_split_outline_anchored(monkeypatch):
    # page 1: the chapter-outline unit (typed paragraph); page 2: the glued
    # paragraph whose HEAD carries the outline title "Divide Integers".
    fbs = [_fb(_OUTLINE_UNIT, page=1), _fb(_GLUED_PARAGRAPH, page=2, y0=20.0)]
    c1 = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    c2 = _arr([{"ids": ["p2_b00"], "type": "paragraph"}])
    regions, audit, _f = _drive_two_pages(monkeypatch, fbs, [c1, c2])
    assert audit["pages_valid"] == 2
    # coverage invariant: both real FBs claimed exactly once
    claimed = sorted(i for r in regions for i in r.feature_block_indices)
    assert claimed == [0, 1]
    # a synthetic LEADING-title heading exists: verbatim head text, zero FBs,
    # provenance pass heading_sanity_leading_title, duplicate_of_head marked
    synth = [r for r in regions if r.kind == "heading"
             and r.payload.get("duplicate_of_head")]
    assert len(synth) == 1
    s = synth[0]
    assert s.feature_block_indices == ()
    assert s.payload["text"] == "Divide Integers"
    assert s.payload["synthesized_from"] == "p2_b00"
    assert s.provenance["pass"] == "heading_sanity_leading_title"
    assert s.provenance["synthesized_from"] == "p2_b00"
    # offsets index the VERBATIM head slice of the page-2 unit text
    start, end = s.payload["head_offsets"]
    assert _GLUED_PARAGRAPH[start:end] == "Divide Integers"
    # the synthetic heading renders BEFORE the paragraph it was carved from
    idx_synth = regions.index(s)
    para2 = [r for r in regions if r.kind == "paragraph"
             and 1 in r.feature_block_indices]
    assert para2 and regions.index(para2[0]) == idx_synth + 1
    # the paragraph keeps its FULL text (token conservation)
    assert para2[0].payload["text"].startswith("Divide Integers What about")
    # audit tallies the split on page 2
    row2 = [r for r in audit["page_rows"] if r["page"] == 2][0]
    assert row2["heading_sanity"] == 1
    assert row2["heading_sanity_leading_titles"] == 1


def test_leading_title_lookalike_not_in_lexicon_untouched(monkeypatch):
    # FALSE-POSITIVE GUARD: first words LOOK title-ish ("Division Basics Are
    # Fun" — capitalized, short, no terminal period) but are NOT in the outline
    # lexicon -> the paragraph is untouched.
    lookalike = ("Division Basics Are Fun The prose continues here with more "
                 "explanation of the concept.")
    fbs = [_fb(_OUTLINE_UNIT, page=1), _fb(lookalike, page=2, y0=20.0)]
    c1 = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    c2 = _arr([{"ids": ["p2_b00"], "type": "paragraph"}])
    regions, audit, _f = _drive_two_pages(monkeypatch, fbs, [c1, c2])
    assert audit["pages_valid"] == 2
    assert not [r for r in regions if r.payload.get("duplicate_of_head")]
    row2 = [r for r in audit["page_rows"] if r["page"] == 2][0]
    assert row2["heading_sanity"] == 0
    assert row2["heading_sanity_leading_titles"] == 0


def test_leading_title_no_outline_heading_typed_units_feed_lexicon(monkeypatch):
    # NO outline unit anywhere: the lexicon is fed ONLY by heading-typed units.
    # Page 1 has a clean heading "Divide Integers"; page 2's paragraph head
    # matches it -> split still fires (source (b)).
    fbs = [
        _fb("Divide Integers", page=1),
        _fb("Body prose on the first page follows here.", page=1, y0=20.0),
        _fb(_GLUED_PARAGRAPH, page=2),
    ]
    c1 = _arr([{"ids": ["p1_b00"], "type": "heading", "level": 2},
               {"ids": ["p1_b01"], "type": "paragraph"}])
    c2 = _arr([{"ids": ["p2_b00"], "type": "paragraph"}])
    regions, audit, _f = _drive_two_pages(monkeypatch, fbs, [c1, c2])
    assert audit["pages_valid"] == 2
    synth = [r for r in regions if r.payload.get("duplicate_of_head")]
    assert len(synth) == 1
    assert synth[0].payload["text"] == "Divide Integers"
    assert synth[0].payload["synthesized_from"] == "p2_b00"


def test_leading_title_flag_off_byte_identical(monkeypatch):
    # flag off -> no lexicon built, no split, no synthetic regions; the page-2
    # paragraph block list is byte-identical to the raw arrangement.
    monkeypatch.setenv("SEMANTIK_ARRANGER_HEADING_SANITY", "0")
    fbs = [_fb(_OUTLINE_UNIT, page=1), _fb(_GLUED_PARAGRAPH, page=2, y0=20.0)]
    c1 = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    c2 = _arr([{"ids": ["p2_b00"], "type": "paragraph"}])
    regions, audit, _f = _drive_two_pages(monkeypatch, fbs, [c1, c2])
    assert audit["pages_valid"] == 2
    assert not [r for r in regions if r.kind == "heading"]  # no synthetics at all
    assert all(r.kind == "paragraph" for r in regions)
    assert all(not r.payload.get("duplicate_of_head") for r in regions)
    assert sorted(i for r in regions for i in r.feature_block_indices) == [0, 1]
    rows = audit["page_rows"]
    assert all(r["heading_sanity"] == 0 for r in rows)


def test_build_title_lexicon_sources():
    ubp = {
        1: [{"id": "p1_b00", "text": _OUTLINE_UNIT, "fb_index": 0}],
        2: [{"id": "p2_b00", "text": "Some prose.", "fb_index": 1}],
    }
    results = {
        2: {"status": "ok", "arrangement": {"blocks": [
            {"ids": ["p2_b00"], "type": "heading", "level": 2},
        ]}},
    }
    lex = pa.build_title_lexicon(ubp, results)
    # outline entries parsed + numbering-stripped + casefolded
    assert "divide integers" in lex
    assert "multiply integers" in lex
    # heading-typed unit text fed in too ("Some prose." carries a terminal
    # period -> NOT plausible -> excluded)
    assert "some prose" not in lex

    # a heading-typed unit WITHOUT terminal period is included
    results2 = {
        2: {"status": "ok", "arrangement": {"blocks": [
            {"ids": ["p2_b00"], "type": "heading", "level": 2},
        ]}},
    }
    ubp2 = {2: [{"id": "p2_b00", "text": "Real Numbers", "fb_index": 1}]}
    lex2 = pa.build_title_lexicon(ubp2, results2)
    assert lex2 == {"real numbers"}


def test_apply_leading_title_split_pure():
    unit_by_id = {"p2_b00": {"id": "p2_b00", "text": _GLUED_PARAGRAPH, "fb_index": 1}}
    arr = {"blocks": [{"ids": ["p2_b00"], "type": "paragraph"}]}
    recs = pa.apply_leading_title_split(
        arr, unit_by_id, {"divide integers"}, enabled=True
    )
    assert len(recs) == 1
    assert recs[0]["op"] == "leading_title_split"
    assert recs[0]["leading_title"] == "Divide Integers"
    start, end = recs[0]["head_offsets"]
    assert _GLUED_PARAGRAPH[start:end] == "Divide Integers"
    blocks = arr["blocks"]
    assert len(blocks) == 2
    assert blocks[0]["synthetic_heading"] is True
    assert blocks[0]["sanity_pass"] == "heading_sanity_leading_title"
    assert blocks[0]["duplicate_of_head"] is True
    assert blocks[0]["ids"] == []
    assert blocks[1]["type"] == "paragraph"  # paragraph untouched, full text kept

    # disabled -> byte-identical no-op
    arr2 = {"blocks": [{"ids": ["p2_b00"], "type": "paragraph"}]}
    assert pa.apply_leading_title_split(arr2, unit_by_id, {"divide integers"}, enabled=False) == []
    assert arr2 == {"blocks": [{"ids": ["p2_b00"], "type": "paragraph"}]}

    # idempotent: re-applying over the already-split arrangement adds nothing
    recs3 = pa.apply_leading_title_split(arr, unit_by_id, {"divide integers"}, enabled=True)
    assert recs3 == []
    assert len(arr["blocks"]) == 2  # no second synthetic for the same block


# ---------------------------------------------------------------------------
# (l) VLM-marker heading PROMOTION — the de-poisoned-lane heading-recall fix:
# the VLM marks a page's section title '##', but the arrange model leaves the
# standalone unit typed paragraph. The promote arm re-types it to heading,
# guarded by the SAME anti-furniture SoT the demotion arm uses.
# ---------------------------------------------------------------------------
def _u(uid, text, *, level=None):
    u = {"id": uid, "text": text, "fb_index": 0}
    if level is not None:
        u["vlm_heading_level"] = level
    return u


def test_md_promote_standalone_section_title_promoted():
    units = [_u("p1_b00", "Divide Integers", level=2)]
    unit_by_id = {u["id"]: u for u in units}
    arr = json.loads(_arr([{"ids": ["p1_b00"], "type": "paragraph"}]))
    recs = pa.apply_md_heading_promote(arr, unit_by_id, max_chars=120, enabled=True)
    blk = arr["blocks"][0]
    assert blk["type"] == "heading"
    assert blk["level"] == 2
    assert blk["ids"] == ["p1_b00"]  # RE-TYPE: keeps its unit ids (coverage-safe)
    assert blk["heading_sanity"]["promoted_from"] == "paragraph"
    assert len(recs) == 1 and recs[0]["op"] == "heading_promote"
    assert recs[0]["reason"] == "vlm_marker"


def test_md_promote_level_clamped_2_to_4():
    for src, want in ((1, 2), (3, 3), (6, 4)):
        units = [_u("p1_b00", "Multiply Integers", level=src)]
        arr = json.loads(_arr([{"ids": ["p1_b00"], "type": "paragraph"}]))
        pa.apply_md_heading_promote(arr, {u["id"]: u for u in units}, max_chars=120, enabled=True)
        assert arr["blocks"][0]["level"] == want


def test_md_promote_rejects_pedagogical_and_apparatus_markers():
    # The VLM also '##'-marks EXAMPLE / apparatus labels; the SoT guard must
    # NOT let those become headings (precision preservation).
    for text in ("EXAMPLE 1.1", "TRY IT :: 1.5", "Solution"):
        units = [_u("p1_b00", text, level=2)]
        arr = json.loads(_arr([{"ids": ["p1_b00"], "type": "paragraph"}]))
        recs = pa.apply_md_heading_promote(arr, {u["id"]: u for u in units}, max_chars=120, enabled=True)
        assert arr["blocks"][0]["type"] == "paragraph", text
        assert recs == []


def test_md_promote_rejects_prose_paragraph():
    # A real prose paragraph (terminal period) is never a section title.
    units = [_u("p1_b00", "What about division? Just as multiplication is repeated addition.", level=2)]
    arr = json.loads(_arr([{"ids": ["p1_b00"], "type": "paragraph"}]))
    recs = pa.apply_md_heading_promote(arr, {u["id"]: u for u in units}, max_chars=120, enabled=True)
    assert arr["blocks"][0]["type"] == "paragraph"
    assert recs == []


def test_md_promote_no_hint_is_noop():
    # No vlm_heading_level on the unit -> nothing to promote (byte-identical).
    units = [_u("p1_b00", "Divide Integers")]  # no level
    arr = json.loads(_arr([{"ids": ["p1_b00"], "type": "paragraph"}]))
    recs = pa.apply_md_heading_promote(arr, {u["id"]: u for u in units}, max_chars=120, enabled=True)
    assert arr["blocks"][0]["type"] == "paragraph"
    assert recs == []


def test_md_promote_flag_off_byte_identical():
    units = [_u("p1_b00", "Divide Integers", level=2)]
    arr = json.loads(_arr([{"ids": ["p1_b00"], "type": "paragraph"}]))
    before = json.dumps(arr, sort_keys=True)
    recs = pa.apply_md_heading_promote(arr, {u["id"]: u for u in units}, max_chars=120, enabled=False)
    assert recs == []
    assert json.dumps(arr, sort_keys=True) == before


def test_md_promote_end_to_end_via_arrange_page():
    # The arrange model types the standalone title paragraph; promotion (default
    # ON within the arranger) re-types it to heading. Runs AFTER heading-sanity.
    units = [_u("p1_b00", "Multiply Integers", level=3)]
    good = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    res = pa.arrange_page(_Seat(), "IMG", units, 1, requests_module=_FakeRequests([good]))
    assert res["status"] == "ok"
    blk = res["arrangement"]["blocks"][0]
    assert blk["type"] == "heading" and blk["level"] == 3
    assert any(r["op"] == "heading_promote" for r in res["heading_sanity"])


def test_mint_units_stamps_vlm_heading_level_whole_block_only():
    fb_head = _fb("Divide Integers")
    fb_head.vlm_hint = {"kind": "heading", "level": 2, "marker": None, "coverage": "whole_block"}
    fb_prefix = _fb("Some prose", y0=20.0)  # prefix heading hint -> NOT a standalone heading
    fb_prefix.vlm_hint = {"kind": "heading", "level": 2, "marker": None, "coverage": "prefix"}
    fb_list = _fb("An item", y0=40.0)
    fb_list.vlm_hint = {"kind": "list_item", "level": None, "marker": "1.", "coverage": "whole_block"}
    fb_plain = _fb("Plain body", y0=60.0)  # no hint
    by_page = pa.mint_units_by_page([fb_head, fb_prefix, fb_list, fb_plain])
    units = by_page[1]
    assert units[0]["vlm_heading_level"] == 2         # whole_block heading -> stamped
    assert "vlm_heading_level" not in units[1]        # prefix coverage -> not stamped
    assert "vlm_heading_level" not in units[2]         # list_item hint -> not stamped
    assert "vlm_heading_level" not in units[3]         # no hint -> absent
    # unit text is VERBATIM (no marker leakage) and the listing is unchanged
    assert units[0]["text"] == "Divide Integers"


def test_resolve_md_heading_promote_defaults(monkeypatch):
    monkeypatch.delenv("SEMANTIK_ARRANGER_MD_HEADING_PROMOTE", raising=False)
    assert pa.resolve_arranger_md_heading_promote() is True
    monkeypatch.setenv("SEMANTIK_ARRANGER_MD_HEADING_PROMOTE", "off")
    assert pa.resolve_arranger_md_heading_promote() is False
    monkeypatch.setenv("SEMANTIK_ARRANGER_MD_HEADING_PROMOTE", "garbage")
    assert pa.resolve_arranger_md_heading_promote() is True  # parse-with-fallback -> on


# ---------------------------------------------------------------------------
# (m) FIGURE arm (task #49) — deterministic figure Regions on the arranger lane,
#     page-raster-guarded, spliced into reading order.
# ---------------------------------------------------------------------------
def _img_fb(*, page=1, bbox=(50.0, 100.0, 250.0, 300.0)):
    """A SYNTHETIC image FeatureBlock (what SEMANTIK_DETECT_FIGURES interleaves)."""
    raw = RawBlock(
        text="", page=page, bbox=bbox,
        page_width=612.0, page_height=792.0, source="pypdfium2:image",
    )
    return FeatureBlock(
        raw=raw, size_bucket="md", gap_above=None, is_top_of_page=False,
        is_centered=False, caps=None, indent_bucket=0, relative_font_ratio=1.0,
        provenance="pypdfium2:image", is_image=True,
    )


def _cand(fb_index, bbox, *, page=1, px_size=(400, 400)):
    from semantik_structure.region_detection import ImageCandidate

    return ImageCandidate(
        kind="figure", bbox=bbox, pages=[page],
        member_block_indices=[fb_index], px_size=px_size,
    )


# A 612x792 page: a sub-page figure covers ~8% (kept); a full-page raster ~100%.
_SUBPAGE_BBOX = (50.0, 100.0, 250.0, 300.0)
_FULLPAGE_BBOX = (0.0, 0.0, 612.0, 792.0)


def test_figure_region_claims_image_fb_exactly_once(monkeypatch):
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    # FB stream: text(0), IMAGE(1), text(2) — the reading-order interleave.
    fbs = [_fb("Intro prose."), _img_fb(bbox=_SUBPAGE_BBOX), _fb("Body prose.", y0=400.0)]
    content = _arr([
        {"ids": ["p1_b00"], "type": "paragraph"},
        {"ids": ["p1_b01"], "type": "paragraph"},
    ])
    regions, audit, _f = _drive_arrange_regions(
        monkeypatch, fbs, [content], image_candidates=[_cand(1, _SUBPAGE_BBOX)]
    )
    figs = [r for r in regions if r.kind == "figure"]
    assert len(figs) == 1
    fig = figs[0]
    assert fig.feature_block_indices == (1,)
    assert fig.payload["typing_authority"] == "vlm-arranger"
    assert fig.provenance["pass"] == "page_arranger_figure"
    assert fig.payload["px_size"] == (400, 400)
    # the arranger stamps its own page geometry
    assert fig.page_bboxes == ((1, _SUBPAGE_BBOX),)
    # COVERAGE: the image FB is claimed exactly once (arrange_regions asserts
    # internally; this pins the expectation explicitly).
    claimed = sorted(i for r in regions for i in r.feature_block_indices)
    assert claimed == [0, 1, 2]
    assert audit["figures"] == 1
    assert audit["page_raster_skipped"] == 0


def test_figure_region_spliced_in_reading_order(monkeypatch):
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    fbs = [_fb("Intro prose."), _img_fb(bbox=_SUBPAGE_BBOX), _fb("Body prose.", y0=400.0)]
    content = _arr([
        {"ids": ["p1_b00"], "type": "paragraph"},
        {"ids": ["p1_b01"], "type": "paragraph"},
    ])
    regions, _a, _f = _drive_arrange_regions(
        monkeypatch, fbs, [content], image_candidates=[_cand(1, _SUBPAGE_BBOX)]
    )
    # The figure lands BETWEEN the two paragraphs (its image FB index is 1).
    assert [r.kind for r in regions] == ["paragraph", "figure", "paragraph"]
    assert [min(r.feature_block_indices) for r in regions] == [0, 1, 2]


def test_figure_splice_preserves_authored_text_region_order(monkeypatch):
    """The arranger's AUTHORED order (which may be non-monotonic in min-FB) is
    preserved EXACTLY — the splice inserts, it never re-sorts."""
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    fbs = [_fb("A"), _fb("B", y0=40.0), _img_fb(bbox=_SUBPAGE_BBOX), _fb("C", y0=500.0)]
    # Arrange model authors the units OUT of min-FB order (b01 before b00).
    content = _arr([
        {"ids": ["p1_b01"], "type": "paragraph"},
        {"ids": ["p1_b00"], "type": "paragraph"},
        {"ids": ["p1_b02"], "type": "paragraph"},
    ])
    regions, _a, _f = _drive_arrange_regions(
        monkeypatch, fbs, [content], image_candidates=[_cand(2, _SUBPAGE_BBOX)]
    )
    text_order = [
        min(r.feature_block_indices) for r in regions if r.kind == "paragraph"
    ]
    assert text_order == [1, 0, 3]  # authored order intact, NOT re-sorted
    # the figure (image FB 2) is inserted before the first text region whose
    # min-FB exceeds 2 (the FB-3 paragraph).
    assert [r.kind for r in regions] == [
        "paragraph", "paragraph", "figure", "paragraph",
    ]


def test_page_raster_candidate_excluded_subpage_figure_kept(monkeypatch):
    """THE PAGE-RASTER GUARD — the regression test for the full-page-image
    catastrophe (a page-raster scan yields one full-page image per page; emitting
    them would ship a photograph of every page as an <img>: WCAG 1.4.5)."""
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    # FB stream: RASTER(0), text(1), FIGURE(2)
    fbs = [
        _img_fb(bbox=_FULLPAGE_BBOX),
        _fb("Body prose.", y0=50.0),
        _img_fb(bbox=_SUBPAGE_BBOX),
    ]
    cands = [_cand(0, _FULLPAGE_BBOX), _cand(2, _SUBPAGE_BBOX)]
    content = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    regions, audit, _f = _drive_arrange_regions(
        monkeypatch, fbs, [content], image_candidates=cands
    )
    figs = [r for r in regions if r.kind == "figure"]
    assert len(figs) == 1                       # ONLY the genuine sub-page figure
    assert figs[0].feature_block_indices == (2,)
    assert audit["figures"] == 1
    assert audit["page_raster_skipped"] == 1
    # The page-raster FB is neither expected nor claimed (coverage holds).
    claimed = sorted(i for r in regions for i in r.feature_block_indices)
    assert claimed == [1, 2]


def test_figure_arm_noop_without_image_candidates(monkeypatch):
    """Flag-off (SEMANTIK_DETECT_FIGURES unset) → no image FBs, no candidates →
    the arm is a natural no-op and the region list is byte-identical."""
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    fbs = [_fb("Alpha."), _fb("Beta.", y0=40.0)]
    content = _arr([
        {"ids": ["p1_b00"], "type": "paragraph"},
        {"ids": ["p1_b01"], "type": "paragraph"},
    ])
    regions, audit, _f = _drive_arrange_regions(monkeypatch, fbs, [content])
    assert not any(r.kind == "figure" for r in regions)
    assert [(r.kind, r.feature_block_indices) for r in regions] == [
        ("paragraph", (0,)), ("paragraph", (1,)),
    ]
    assert not any("px_size" in r.payload for r in regions)
    assert audit["figures"] == 0 and audit["page_raster_skipped"] == 0
    # The splice is the identity no-op with no figures.
    assert pa._splice_figure_regions(regions, []) is regions


def test_figures_and_text_arranged_on_the_same_page(monkeypatch):
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    # heading(0), IMAGE(1), caption text(2), IMAGE(3), body(4)
    fbs = [
        _fb("Section Title"),
        _img_fb(bbox=(50.0, 100.0, 250.0, 300.0)),
        _fb("Figure 1: a widget", y0=320.0),
        _img_fb(bbox=(50.0, 400.0, 250.0, 600.0)),
        _fb("Closing body prose.", y0=650.0),
    ]
    cands = [
        _cand(1, (50.0, 100.0, 250.0, 300.0)),
        _cand(3, (50.0, 400.0, 250.0, 600.0)),
    ]
    content = _arr([
        {"ids": ["p1_b00"], "type": "heading", "level": 2},
        {"ids": ["p1_b01"], "type": "figure_caption"},
        {"ids": ["p1_b02"], "type": "paragraph"},
    ])
    regions, audit, _f = _drive_arrange_regions(
        monkeypatch, fbs, [content], image_candidates=cands
    )
    assert audit["figures"] == 2
    assert [r.kind for r in regions] == [
        "heading", "figure", "paragraph", "figure", "paragraph",
    ]
    # both image FBs claimed exactly once; every text FB still claimed once
    claimed = sorted(i for r in regions for i in r.feature_block_indices)
    assert claimed == [0, 1, 2, 3, 4]
    # the nearest "Figure N" caption below the first image is recorded
    figs = [r for r in regions if r.kind == "figure"]
    assert figs[0].payload["caption_fb_index"] == 2
    assert figs[1].payload["caption_fb_index"] is None


def test_figure_arm_fail_soft_on_bad_candidate(monkeypatch):
    """A malformed candidate is skipped, never aborting the document."""
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    fbs = [_fb("Body."), _img_fb(bbox=_SUBPAGE_BBOX)]

    class _Bad:
        member_block_indices = "not-a-list-of-ints"

    content = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    regions, audit, _f = _drive_arrange_regions(
        monkeypatch, fbs, [content],
        image_candidates=[_Bad(), _cand(1, _SUBPAGE_BBOX)],
    )
    # the good candidate still becomes a figure; the bad one is counted + skipped
    assert audit["figures"] == 1
    assert audit["figure_stats"]["errors"] >= 1
    assert [r.kind for r in regions] == ["paragraph", "figure"]


def test_arrange_model_cannot_type_a_unit_as_a_figure():
    """`figure` is deliberately ABSENT from _TYPE_TO_KIND — a model that types a
    TEXT unit `figure` falls back to paragraph (its text is never dropped)."""
    assert "figure" not in pa._TYPE_TO_KIND
    fbs = [_fb("Real prose the model mislabeled.")]
    units = pa.mint_units_by_page(fbs)[1]
    result = {
        "status": "ok",
        "arrangement": {"blocks": [{"ids": ["p1_b00"], "type": "figure"}]},
        "attempts": 1,
    }
    regions = pa.build_regions_for_page(1, units, result, fbs)
    assert [r.kind for r in regions] == ["paragraph"]
    assert regions[0].payload["text"] == "Real prose the model mislabeled."


def test_figure_capture_rationale_carries_figure_tallies(monkeypatch):
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    logged = []

    class _Cap:
        def __init__(self, **_k):
            pass

        def log_decision(self, **kw):
            logged.append(kw)

    import lib.decision_capture as dc

    monkeypatch.setattr(dc, "DecisionCapture", _Cap)
    fbs = [_img_fb(bbox=_FULLPAGE_BBOX), _fb("Body.", y0=50.0), _img_fb(bbox=_SUBPAGE_BBOX)]
    content = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])
    _drive_arrange_regions(
        monkeypatch, fbs, [content],
        image_candidates=[_cand(0, _FULLPAGE_BBOX), _cand(2, _SUBPAGE_BBOX)],
    )
    assert logged, "the per-doc structure_detection capture must still fire"
    rationale = logged[0]["rationale"]
    assert "figures=1" in rationale
    assert "page_raster_skipped=1" in rationale
    assert logged[0]["decision_type"] == "structure_detection"


# --- the shared structure_graph helpers (page-raster guard + Region builder) ---
_POINT_DIMS = {1: (612.0, 792.0)}


def test_is_page_raster_candidate_predicate():
    from semantik_structure.structure_graph import is_page_raster_candidate

    assert is_page_raster_candidate(_cand(0, _FULLPAGE_BBOX), _POINT_DIMS) is True
    assert is_page_raster_candidate(_cand(1, _SUBPAGE_BBOX), _POINT_DIMS) is False


def test_page_raster_guard_survives_ocr_lane_mixed_coordinate_spaces():
    """REGRESSION (real-data bug, task #49): on the OCR/scan lane — the ONLY lane
    this guard runs on — TEXT FeatureBlocks carry IMAGE-PIXEL-space bboxes and page
    dims (1224x1584 at OCR render scale 2.0) while an ImageCandidate bbox is
    PDF-POINT space (612x792). ``features._interleave_image_feature_blocks`` copies
    the page dims onto the synthetic image FB from a neighbouring TEXT FB, so the
    image FB ends up with PIXEL dims + a POINT bbox.

    An earlier guard read its dims off that FB and measured a FULL-PAGE raster as
    (612*792)/(1224*1584) = 25% coverage -> under threshold -> KEEP. On the real
    198-page scanned chapter all 198/198 page rasters slipped through and would
    have shipped as <img> (a photograph of every page: WCAG 1.4.5 images-of-text).

    The guard must therefore take its dims from the PDF extraction (POINT space,
    the candidate bbox's own space). This test pins that: the image FB carries the
    poisoned pixel-space dims, and the guard STILL fires.
    """
    from semantik_structure.structure_graph import (
        is_page_raster_candidate,
        page_dims_from_shared,
    )

    # The synthetic image FB as features.py really builds it on the OCR lane:
    # PIXEL-space page dims inherited from a text FB, POINT-space bbox.
    poisoned_fb = FeatureBlock(
        raw=RawBlock(text="", page=1, bbox=_FULLPAGE_BBOX,
                     page_width=1224.0, page_height=1584.0,   # <- PIXEL space
                     source="pypdfium2:image"),
        size_bucket="md", gap_above=None, is_top_of_page=False, is_centered=False,
        caps=None, indent_bucket=0, relative_font_ratio=1.0,
        provenance="pypdfium2:image", is_image=True,
    )
    raster = _cand(0, _FULLPAGE_BBOX)                          # <- POINT space
    # Dims from the EXTRACTION (point space) — the contract.
    dims = page_dims_from_shared(_shared_pages(1))
    assert dims == {1: (612.0, 792.0)}
    assert is_page_raster_candidate(raster, dims) is True

    # And end-to-end through the arm: the raster is excluded despite the poisoned FB.
    regions, stats = pa.build_figure_regions([raster], [poisoned_fb], dims)
    assert regions == [] and stats["page_raster_skipped"] == 1 and stats["figures"] == 0


def test_build_figure_regions_without_page_dims_fails_open():
    """No page dims -> the guard cannot measure -> fail-OPEN (keep the candidate).
    Documents why ``arrange_regions`` MUST pass ``page_dims_from_shared``."""
    regions, stats = pa.build_figure_regions(
        [_cand(0, _FULLPAGE_BBOX)], [_img_fb(bbox=_FULLPAGE_BBOX)], {}
    )
    assert stats["figures"] == 1 and stats["page_raster_skipped"] == 0
    assert regions[0].kind == "figure"


def test_is_page_raster_candidate_fails_open_on_bad_geometry():
    """Fail-OPEN: unusable dims -> False (KEEP the candidate; never silently drop
    a real figure)."""
    from semantik_structure.region_detection import ImageCandidate
    from semantik_structure.structure_graph import is_page_raster_candidate

    # Degenerate page dims.
    assert is_page_raster_candidate(_cand(0, _FULLPAGE_BBOX), {1: (0.0, 0.0)}) is False
    # Page not present in the dims map (unknown page) -> keep.
    assert is_page_raster_candidate(_cand(0, _FULLPAGE_BBOX), {7: (612.0, 792.0)}) is False
    # Degenerate candidate bbox -> keep.
    assert is_page_raster_candidate(_cand(0, (5.0, 5.0, 5.0, 5.0)), _POINT_DIMS) is False
    # No page at all -> keep.
    empty = ImageCandidate(kind="figure", bbox=(0, 0, 1, 1), pages=[], member_block_indices=[0])
    assert is_page_raster_candidate(empty, _POINT_DIMS) is False


def test_page_dims_from_shared_tolerates_garbage():
    from semantik_structure.structure_graph import page_dims_from_shared

    assert page_dims_from_shared({}) == {}
    assert page_dims_from_shared({"pages": [{"page_num": 0, "width": 1, "height": 1}]}) == {}
    assert page_dims_from_shared(
        {"pages": [{"page_num": 1, "width": "x", "height": 792.0},
                   {"page_num": 2, "width": 612.0, "height": 792.0}]}
    ) == {2: (612.0, 792.0)}


def test_build_figure_region_from_candidate_shape():
    from semantik_structure.structure_graph import build_figure_region_from_candidate

    fbs = [_fb("intro"), _img_fb(bbox=_SUBPAGE_BBOX), _fb("Figure 1: a widget", y0=400.0)]
    region = build_figure_region_from_candidate(
        _cand(1, _SUBPAGE_BBOX), fbs, source_region_id=7, decision_flags=("f1",),
    )
    assert region.kind == "figure"
    assert region.feature_block_indices == (1,)
    assert region.payload == {
        "alt_hint": None, "caption_fb_index": 2, "px_size": (400, 400),
    }
    assert region.provenance == {
        "pass": "figure_image_candidate", "decision_flags": ("f1",),
    }
    assert region.source_region_id == 7
    # page_bboxes is NOT set here (build_structure_graph re-stamps at its exit)
    assert region.page_bboxes is None
    # no claimable member FB -> None (never a figure with no FB)
    from semantik_structure.region_detection import ImageCandidate

    empty = ImageCandidate(kind="figure", bbox=(0, 0, 1, 1), pages=[1], member_block_indices=[])
    assert build_figure_region_from_candidate(empty, fbs) is None


# ---------------------------------------------------------------------------
# (n) SECTION-TITLE RESCUE (SEMANTIK_ARRANGER_TITLE_RESCUE) — the measured
# scan-lane section-heading recall gap.
#
# ROOT CAUSE: extraction is clean — the running
# header / folio and the section title arrive as SEPARATE units — but the ARRANGE
# MODEL groups them into ONE block, after which every existing mechanism damages
# the title. The four shapes below are the four REAL broken titles, pinned:
#
#   §1.3  "Chapter 1 Foundations 41" + "1.3 Add and Subtract Integers"
#           -> shipped as ONE heading with the running header glued on (WRONG TEXT)
#   §1.4  "64 Chapter 1 Foundations" + "1.4 Multiply and Divide Integers"
#           -> `_is_wholly_furniture` PREFIX-matches the header on the JOINED text
#              and demotes the WHOLE block to furniture: the title is DESTROYED
#   §1.6  "95" + "1.6 Add and Subtract Fractions"
#           -> shipped as the heading "95 1.6 Add and Subtract Fractions" (WRONG TEXT)
#   §1.7  "111" + "1.7 Decimals" + <the whole page body>
#           -> one over-merged block, demoted `too_long`: the title is BURIED
# ---------------------------------------------------------------------------
def _units(*texts, page=1):
    return [
        {"id": f"p{page}_b{i:02d}", "text": t, "fb_index": i, "bbox": None}
        for i, t in enumerate(texts)
    ]


def _by_id(units):
    return {u["id"]: u for u in units}


def _heads(arr, unit_by_id):
    """[(level, joined_text)] for every heading block."""
    out = []
    for b in arr["blocks"]:
        if b.get("type") != "heading":
            continue
        if b.get("synthetic_heading"):
            t = (b.get("text") or "").strip()
        else:
            t = " ".join(
                unit_by_id[i]["text"].replace("\n", " ").strip()
                for i in b["ids"] if i in unit_by_id
            ).strip()
        out.append((b.get("level"), t))
    return out


def _all_ids(arr):
    ids = []
    for b in arr["blocks"]:
        ids.extend(b.get("ids") or [])
    return ids


# --- the PEEL arm -----------------------------------------------------------
def test_peel_rescues_running_header_fused_section_title_1_3():
    """§1.3: header + title fused into ONE heading -> furniture + CLEAN title."""
    units = _units("Chapter 1 Foundations 41", "1.3 Add and Subtract Integers")
    arr = {"blocks": [{"type": "heading", "level": 1, "ids": ["p1_b00", "p1_b01"]}]}
    recs = pa.apply_furniture_unit_peel(arr, _by_id(units), enabled=True)
    assert [r["op"] for r in recs] == ["furniture_unit_peel"]
    assert arr["blocks"][0]["type"] == "furniture"
    assert arr["blocks"][0]["ids"] == ["p1_b00"]
    assert _heads(arr, _by_id(units)) == [(1, "1.3 Add and Subtract Integers")]


def test_peel_saves_title_the_furniture_demote_would_have_destroyed_1_4():
    """§1.4 — the DESTRUCTIVE case.

    Without the peel, `_is_wholly_furniture` prefix-matches the leading-folio
    running header on the JOINED text, so heading-sanity demotes the WHOLE block
    to `furniture` and the section title is DROPPED. Regression-pin BOTH halves.
    """
    units = _units("64 Chapter 1 Foundations", "1.4 Multiply and Divide Integers")
    ub = _by_id(units)

    # (a) today's behaviour WITHOUT the peel: the whole block becomes furniture.
    legacy = {"blocks": [{"type": "heading", "level": 1, "ids": ["p1_b00", "p1_b01"]}]}
    pa.apply_heading_sanity(legacy, ub, max_chars=120, enabled=True)
    assert legacy["blocks"][0]["type"] == "furniture"
    assert _heads(legacy, ub) == []  # the title is GONE

    # (b) with the peel first, the title survives heading-sanity intact.
    fixed = {"blocks": [{"type": "heading", "level": 1, "ids": ["p1_b00", "p1_b01"]}]}
    pa.apply_furniture_unit_peel(fixed, ub, enabled=True)
    pa.apply_heading_sanity(fixed, ub, max_chars=120, enabled=True)
    assert _heads(fixed, ub) == [(1, "1.4 Multiply and Divide Integers")]


def test_peel_strips_bare_folio_from_heading_text_1_6():
    """§1.6: the heading shipped as '95 1.6 Add and Subtract Fractions' (WRONG TEXT)."""
    units = _units("95", "1.6 Add and Subtract Fractions")
    arr = {"blocks": [{"type": "heading", "level": 2, "ids": ["p1_b00", "p1_b01"]}]}
    pa.apply_furniture_unit_peel(arr, _by_id(units), enabled=True)
    assert _heads(arr, _by_id(units)) == [(2, "1.6 Add and Subtract Fractions")]


def test_peel_preserves_the_demote_verdict_never_rescues_prose():
    """ANTI-REGRESSION: peeling must not shrink an over-long PROSE heading back
    under the ceiling, which would cancel heading-sanity's `too_long` demote and
    ship prose as a heading (8 junk headings on the real corpus before this guard).
    """
    prose = "Ndula, an elephant at the San Diego Safari Park, weighs almost 3.2 tons"
    units = _units("Chapter 1 Foundations 167", prose + " " + prose)
    arr = {"blocks": [{"type": "heading", "level": None, "ids": ["p1_b00", "p1_b01"]}]}
    recs = pa.apply_furniture_unit_peel(arr, _by_id(units), enabled=True)
    assert recs == []                                    # refused
    assert arr["blocks"][0]["ids"] == ["p1_b00", "p1_b01"]  # untouched
    pa.apply_heading_sanity(arr, _by_id(units), max_chars=120, enabled=True)
    assert _heads(arr, _by_id(units)) == []              # still demoted, as today


def test_peel_never_peels_an_apparatus_opener():
    """An apparatus opener ('Introduction') is CONTENT, not page furniture — peeling
    it would route real text into a metadata_drop region and DELETE it."""
    units = _units("Introduction", "1.1 Introduction to Whole Numbers")
    arr = {"blocks": [{"type": "heading", "level": 2, "ids": ["p1_b00", "p1_b01"]}]}
    assert pa.apply_furniture_unit_peel(arr, _by_id(units), enabled=True) == []


def test_peel_only_touches_heading_blocks_and_conserves_units():
    units = _units("95", "Some running prose body of the page.")
    arr = {"blocks": [{"type": "paragraph", "level": None, "ids": ["p1_b00", "p1_b01"]}]}
    assert pa.apply_furniture_unit_peel(arr, _by_id(units), enabled=True) == []
    assert _all_ids(arr) == ["p1_b00", "p1_b01"]


def test_peel_flag_off_byte_identical():
    units = _units("Chapter 1 Foundations 41", "1.3 Add and Subtract Integers")
    arr = {"blocks": [{"type": "heading", "level": 1, "ids": ["p1_b00", "p1_b01"]}]}
    snap = json.dumps(arr, sort_keys=True)
    assert pa.apply_furniture_unit_peel(arr, _by_id(units), enabled=False) == []
    assert json.dumps(arr, sort_keys=True) == snap


# --- the OUTLINE harvest ----------------------------------------------------
def test_outline_harvest_reads_per_line_entry_units():
    """THE LEXICON BUG: on a real scanned chapter opener the marker is its OWN unit
    and each entry is a SEPARATE following unit, so the committed same-unit-only
    parse harvested ZERO titles — the lexicon could never supply a title the
    arranger had already missed, i.e. exactly the set that needs rescuing."""
    units = _units(
        "Chapter Outline",
        "1.1 Introduction to Whole Numbers",
        "1.2 Use the Language of Algebra",
        "1.7 Decimals",
        "Introduction",  # ends the run
        "Just like a building needs a firm foundation to support it, ...",
    )
    titles, entry_ids = pa.harvest_outline_titles({1: units})
    assert titles == {
        "introduction to whole numbers",
        "use the language of algebra",
        "decimals",
    }
    assert entry_ids == {"p1_b01", "p1_b02", "p1_b03"}


def test_outline_harvest_still_reads_the_fused_single_unit_shape():
    units = _units("Chapter Outline 1.1 Introduction to Whole Numbers 1.7 Decimals")
    titles, entry_ids = pa.harvest_outline_titles({1: units})
    assert titles == {"introduction to whole numbers", "decimals"}
    assert entry_ids == set()


# --- the CARVE arm ----------------------------------------------------------
def test_carve_recovers_title_buried_in_an_over_merged_block_1_7():
    """§1.7: '111' + '1.7 Decimals' + the whole page body merged into ONE block."""
    units = _units("111", "1.7 Decimals", "Learning Objectives",
                   "Decimals are another way of writing fractions.")
    arr = {"blocks": [{"type": "paragraph", "level": None,
                       "ids": ["p1_b00", "p1_b01", "p1_b02", "p1_b03"]}]}
    recs = pa.apply_outline_title_carve(
        arr, _by_id(units), {"decimals"}, set(), enabled=True
    )
    assert [r["op"] for r in recs] == ["outline_title_carve"]
    assert _heads(arr, _by_id(units)) == [(2, "1.7 Decimals")]
    assert _all_ids(arr) == ["p1_b00", "p1_b01", "p1_b02", "p1_b03"]  # conserved


def test_carve_recovers_title_buried_mid_block_1_1():
    """§1.1: the title sits in the MIDDLE of the chapter-opener block."""
    units = _units("Introduction", "Just like a building needs a firm foundation.",
                   "1.1 Introduction to Whole Numbers", "Learning Objectives")
    arr = {"blocks": [{"type": "paragraph", "level": None,
                       "ids": ["p1_b00", "p1_b01", "p1_b02", "p1_b03"]}]}
    pa.apply_outline_title_carve(
        arr, _by_id(units), {"introduction to whole numbers"}, set(), enabled=True
    )
    assert _heads(arr, _by_id(units)) == [(2, "1.1 Introduction to Whole Numbers")]
    assert [b["ids"] for b in arr["blocks"]] == [
        ["p1_b00", "p1_b01"], ["p1_b02"], ["p1_b03"],
    ]


def test_carve_never_promotes_the_outline_listing_itself():
    """RE-POISONING TRAP: the chapter-outline block's own entries match the lexicon
    exactly — carving them would mint ten spurious section headings on page 1."""
    units = _units("Chapter Outline", "1.1 Introduction to Whole Numbers",
                   "1.7 Decimals")
    titles, entry_ids = pa.harvest_outline_titles({1: units})
    arr = {"blocks": [{"type": "paragraph", "level": None,
                       "ids": ["p1_b00", "p1_b01", "p1_b02"]}]}
    assert pa.apply_outline_title_carve(
        arr, _by_id(units), titles, entry_ids, enabled=True
    ) == []
    assert _heads(arr, _by_id(units)) == []


def test_carve_requires_BOTH_section_shape_and_an_outline_declaration():
    """A bare word that merely normalizes onto an outline title is NOT carved — the
    unit must literally be the declared 'N.M Title' section title."""
    units = _units("Some intro prose.", "Decimals", "More prose.")
    arr = {"blocks": [{"type": "paragraph", "level": None,
                       "ids": ["p1_b00", "p1_b01", "p1_b02"]}]}
    assert pa.apply_outline_title_carve(
        arr, _by_id(units), {"decimals"}, set(), enabled=True
    ) == []


def test_carve_flag_off_byte_identical():
    units = _units("111", "1.7 Decimals", "body")
    arr = {"blocks": [{"type": "paragraph", "level": None,
                       "ids": ["p1_b00", "p1_b01", "p1_b02"]}]}
    snap = json.dumps(arr, sort_keys=True)
    assert pa.apply_outline_title_carve(
        arr, _by_id(units), {"decimals"}, set(), enabled=False
    ) == []
    assert json.dumps(arr, sort_keys=True) == snap


# --- section-title LEVELS ---------------------------------------------------
def test_section_title_levels_nav_for_first_h5_for_summary_repeat():
    """Vendor parity: each section title is <h3> at its section start and <h5> in the
    end-of-chapter summary. Arranger level N renders h(N+1), so 2 then 4."""
    u_start = _units("1.3 Add and Subtract Integers", page=37)
    u_sum = _units("1.3 Add and Subtract Integers", page=181)
    results = {
        37: {"status": "ok", "arrangement": {"blocks": [
            {"type": "heading", "level": 1, "ids": ["p37_b00"]}]}},
        181: {"status": "ok", "arrangement": {"blocks": [
            {"type": "heading", "level": 1, "ids": ["p181_b00"]}]}},
    }
    recs = pa.apply_section_title_levels(
        [37, 181], results, {37: u_start, 181: u_sum},
        {"add and subtract integers"}, enabled=True,
    )
    assert [r["reason"] for r in recs] == ["section_start", "repeat_of_section_title"]
    assert results[37]["arrangement"]["blocks"][0]["level"] == 2   # -> <h3>, nav
    assert results[181]["arrangement"]["blocks"][0]["level"] == 4  # -> <h5>, not nav


def test_section_title_levels_leave_undeclared_section_shaped_headers_alone():
    """'1.3 EXERCISES Practice Makes Perfect' is section-SHAPED but the outline does
    NOT declare it — promoting it into nav would invent a section."""
    units = _units("1.3 EXERCISES Practice Makes Perfect", page=56)
    results = {56: {"status": "ok", "arrangement": {"blocks": [
        {"type": "heading", "level": 1, "ids": ["p56_b00"]}]}}}
    assert pa.apply_section_title_levels(
        [56], results, {56: units}, {"add and subtract integers"}, enabled=True
    ) == []
    assert results[56]["arrangement"]["blocks"][0]["level"] == 1  # untouched


def test_section_title_levels_flag_off_byte_identical():
    units = _units("1.3 Add and Subtract Integers", page=37)
    results = {37: {"status": "ok", "arrangement": {"blocks": [
        {"type": "heading", "level": 1, "ids": ["p37_b00"]}]}}}
    snap = json.dumps(results, sort_keys=True)
    assert pa.apply_section_title_levels(
        [37], results, {37: units}, {"add and subtract integers"}, enabled=False
    ) == []
    assert json.dumps(results, sort_keys=True) == snap


# --- resolver + cache-salt --------------------------------------------------
def test_resolve_title_rescue_defaults(monkeypatch):
    monkeypatch.delenv("SEMANTIK_ARRANGER_TITLE_RESCUE", raising=False)
    assert pa.resolve_arranger_title_rescue() is True       # default ON
    for tok in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("SEMANTIK_ARRANGER_TITLE_RESCUE", tok)
        assert pa.resolve_arranger_title_rescue() is False  # explicit revert lever
    monkeypatch.setenv("SEMANTIK_ARRANGER_TITLE_RESCUE", "garbage")
    assert pa.resolve_arranger_title_rescue() is True       # parse-with-fallback


def test_title_rescue_flag_salts_the_page_sidecar_fingerprint(monkeypatch):
    """The peel is BAKED into the cached arrangement, so a flag flip MUST move the
    key — else a stale pre-peel sidecar is served and the fix silently no-ops."""
    units = _units("Chapter 1 Foundations 41", "1.3 Add and Subtract Integers")
    monkeypatch.setenv("SEMANTIK_ARRANGER_TITLE_RESCUE", "1")
    on = pa._unit_fingerprint(units, "", model="m", include_image=True)
    monkeypatch.setenv("SEMANTIK_ARRANGER_TITLE_RESCUE", "0")
    off = pa._unit_fingerprint(units, "", model="m", include_image=True)
    assert on != off


# ---------------------------------------------------------------------------
# (o) ARRANGER-OVERLAP figure VETO (SEMANTIK_ARRANGER_FIGURE_VETO) — task #58.
#
# The last blocker on flipping SEMANTIK_VLM_FIGURE_DETECT on. The detect lane's
# residual defect is a badly-LOCALIZED proposal that slices text out of a ruled
# table WITHOUT enclosing any of its rules, so it carries NO grid evidence inside
# the box and no crop-local signal can see it. The one signal the crop cannot see
# is the ARRANGER's own typing: if that rectangle is a `table`, a "figure" inside
# it is a crop of the table (images-of-text, WCAG 1.4.5).
#
# THE TRAP these tests pin: a veto that is too aggressive kills NUMBER LINES,
# which are the most valuable figure class in this corpus and sit directly
# above/below tables of values. Killing one is a FAILURE regardless of precision.
# ---------------------------------------------------------------------------
def _region(kind, fb_idxs, page_bboxes, *, arrange_type=None):
    from semantik_structure.structure_graph import Region

    payload = {}
    if arrange_type:
        payload["arrange_type"] = arrange_type
    return Region(
        kind=kind,
        feature_block_indices=tuple(fb_idxs),
        payload=payload,
        provenance={"pass": "test"},
        page_bboxes=tuple(page_bboxes),
    )


# An OCR/scan-lane `shared`: page dims are PDF POINTS (612x792) while the OCR text
# bboxes are IMAGE PIXELS (1224x1584). This mixed-space page is the whole point.
_SCAN_SHARED = {
    "pages": [{
        "page_num": 1, "width": 612.0, "height": 792.0,
        "tesseract_width": 1224.0, "tesseract_height": 1584.0,
    }]
}


def _fig_region(fb_idx, bbox_pt):
    """A figure Region — its bbox is PDF-POINT space (the image FB's own space)."""
    return _region("figure", [fb_idx], [(1, bbox_pt)])


def _table_region(fb_idx, bbox_px):
    """An arranger `table` Region — bbox is IMAGE-PIXEL space (OCR text FBs)."""
    return _region("table", [fb_idx], [(1, bbox_px)], arrange_type="table")


def test_veto_kills_the_strip_sliced_through_a_table():
    """A sloppy strip lying inside a table's rectangle is rejected."""
    # table covers the middle of the page, in PIXEL space (1224x1584)
    table = _table_region(10, (120.0, 400.0, 1100.0, 1000.0))
    # a strip fully inside it, in POINT space (612x792) -> normalized ~same place
    strip = _fig_region(1, (100.0, 250.0, 500.0, 320.0))
    kept, vetoed, stats = pa.apply_figure_overlap_veto(
        [strip], [table], _SCAN_SHARED, enabled=True
    )
    assert kept == [] and len(vetoed) == 1
    assert stats["vetoed"] == 1
    assert stats["rows"][0]["reason"] == "arranger_table_overlap"
    assert stats["rows"][0]["covered_by_table"] == 1.0


def test_veto_SPARES_a_number_line_sitting_directly_ABOVE_a_table():
    """THE TRAP. A number line hugging a table of values must SURVIVE.

    This is the failure mode every previous rejector attempt fell into: scoring
    better on precision by destroying the number lines. Fraction-of-FIGURE-covered
    is what makes this safe -- the number line has ~none of ITSELF inside the
    table, no matter how big that table is.
    """
    table = _table_region(10, (120.0, 800.0, 1100.0, 1500.0))   # lower half, pixels
    numline = _fig_region(1, (60.0, 150.0, 550.0, 220.0))       # upper, points
    kept, vetoed, _ = pa.apply_figure_overlap_veto(
        [numline], [table], _SCAN_SHARED, enabled=True
    )
    assert vetoed == [] and kept == [numline]


def test_veto_SPARES_a_figure_sitting_BESIDE_a_large_table():
    """A big table must not swallow its neighbour. The metric is asymmetric."""
    table = _table_region(10, (20.0, 100.0, 600.0, 1500.0))     # tall left column
    fig = _fig_region(1, (350.0, 300.0, 590.0, 500.0))          # right of it (points)
    kept, vetoed, _ = pa.apply_figure_overlap_veto(
        [fig], [table], _SCAN_SHARED, enabled=True
    )
    assert vetoed == [] and kept == [fig]


def test_veto_uses_fraction_covered_NOT_iou():
    """IoU structurally cannot see this defect -- pin that we do not use it.

    A small strip inside a LARGE table has a tiny IoU (the union is dominated by
    the table) but fraction-of-figure-covered ~1.0. If this veto were IoU-based the
    strip would survive.
    """
    table_px = (0.0, 0.0, 1224.0, 1584.0)      # the whole page, in pixels
    strip_pt = (100.0, 300.0, 500.0, 330.0)    # a thin strip, in points
    table = _table_region(10, table_px)
    strip = _fig_region(1, strip_pt)
    # normalized: strip is ~0.65 x 0.038 of the page; table is the whole page.
    fnorm = (100/612, 300/792, 500/612, 330/792)
    tnorm = (0.0, 0.0, 1.0, 1.0)
    assert pa._fraction_covered(fnorm, tnorm) == pytest.approx(1.0)
    inter = (fnorm[2]-fnorm[0]) * (fnorm[3]-fnorm[1])
    iou = inter / (inter + 1.0 - inter)        # union == the full page
    assert iou < 0.05                          # IoU would NOT reject it
    kept, vetoed, _ = pa.apply_figure_overlap_veto(
        [strip], [table], _SCAN_SHARED, enabled=True
    )
    assert kept == [] and len(vetoed) == 1     # fraction-covered DOES


def test_veto_handles_the_mixed_pixel_vs_point_coordinate_spaces():
    """The load-bearing coordinate contract -- a real bug lived in this exact spot.

    Figure bboxes are POINT space; OCR text/table bboxes are PIXEL space. If the two
    were compared raw (no normalization), a figure at points (100,250)-(500,320)
    would appear in the TOP-LEFT QUADRANT of the table's pixel frame and the overlap
    would be measured against the wrong part of the page entirely.
    """
    # A table occupying the page's BOTTOM half in PIXELS: y 800..1500 of 1584.
    table = _table_region(10, (120.0, 800.0, 1100.0, 1500.0))
    # A figure occupying the page's BOTTOM half in POINTS: y 500..700 of 792.
    # Raw (unnormalized) these DO NOT overlap (fig y<=700 vs table y>=800) --
    # so a naive comparison would MISS a genuine in-table crop.
    fig = _fig_region(1, (200.0, 500.0, 560.0, 700.0))
    kept, vetoed, stats = pa.apply_figure_overlap_veto(
        [fig], [table], _SCAN_SHARED, enabled=True
    )
    # Correctly normalized, both are in the bottom half -> the crop IS vetoed.
    assert kept == [] and len(vetoed) == 1
    assert stats["rows"][0]["covered_by_table"] > 0.9


def test_veto_ignores_exercise_list_and_paragraph_regions():
    """exercise_list is DELIBERATELY not a veto kind -- it is where number lines live.

    Vetoing on it would destroy the figure class the lane exists to recover.
    """
    from semantik_structure.structure_graph import Region

    elist = Region(
        kind="list", feature_block_indices=(10,),
        payload={"arrange_type": "exercise_list"}, provenance={},
        page_bboxes=((1, (0.0, 0.0, 1224.0, 1584.0)),),
    )
    para = _region("paragraph", [11], [(1, (0.0, 0.0, 1224.0, 1584.0))],
                   arrange_type="paragraph")
    numline = _fig_region(1, (60.0, 150.0, 550.0, 220.0))
    kept, vetoed, _ = pa.apply_figure_overlap_veto(
        [numline], [elist, para], _SCAN_SHARED, enabled=True
    )
    assert vetoed == [] and kept == [numline]   # page-covering, but NOT a table


def test_veto_flag_off_is_byte_identical():
    table = _table_region(10, (0.0, 0.0, 1224.0, 1584.0))
    strip = _fig_region(1, (100.0, 300.0, 500.0, 330.0))
    kept, vetoed, stats = pa.apply_figure_overlap_veto(
        [strip], [table], _SCAN_SHARED, enabled=False
    )
    assert kept == [strip] and vetoed == [] and stats["vetoed"] == 0


def test_veto_is_a_natural_noop_with_no_figures():
    """SEMANTIK_DETECT_FIGURES off -> no ImageCandidates -> nothing to veto."""
    table = _table_region(10, (0.0, 0.0, 1224.0, 1584.0))
    kept, vetoed, stats = pa.apply_figure_overlap_veto(
        [], [table], _SCAN_SHARED, enabled=True
    )
    assert kept == [] and vetoed == [] and stats["vetoed"] == 0


def test_veto_born_digital_lane_normalizes_by_point_dims():
    """No tesseract dims -> text bboxes are themselves POINT space."""
    shared = {"pages": [{"page_num": 1, "width": 612.0, "height": 792.0}]}
    table = _table_region(10, (100.0, 240.0, 520.0, 340.0))   # points here
    strip = _fig_region(1, (110.0, 250.0, 500.0, 330.0))      # points
    kept, vetoed, _ = pa.apply_figure_overlap_veto(
        [strip], [table], shared, enabled=True
    )
    assert kept == [] and len(vetoed) == 1


def test_veto_tolerates_missing_and_garbage_geometry():
    """Fail-OPEN: a figure we cannot measure is KEPT (never silently dropped)."""
    table = _table_region(10, (0.0, 0.0, 1224.0, 1584.0))
    bboxless = _region("figure", [1], [])                  # no page_bboxes
    unknown_page = _region("figure", [2], [(99, (0.0, 0.0, 10.0, 10.0))])
    kept, vetoed, _ = pa.apply_figure_overlap_veto(
        [bboxless, unknown_page], [table], _SCAN_SHARED, enabled=True
    )
    assert vetoed == [] and kept == [bboxless, unknown_page]


def test_resolve_figure_veto_defaults(monkeypatch):
    monkeypatch.delenv("SEMANTIK_ARRANGER_FIGURE_VETO", raising=False)
    assert pa.resolve_arranger_figure_veto() is True        # default ON
    for tok in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("SEMANTIK_ARRANGER_FIGURE_VETO", tok)
        assert pa.resolve_arranger_figure_veto() is False   # revert lever
    monkeypatch.setenv("SEMANTIK_ARRANGER_FIGURE_VETO", "garbage")
    assert pa.resolve_arranger_figure_veto() is True        # parse-with-fallback


def test_resolve_figure_veto_min_covered_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("SEMANTIK_ARRANGER_FIGURE_VETO_MIN_COVERED", raising=False)
    assert pa.resolve_arranger_figure_veto_min_covered() == pa._FIGURE_VETO_MIN_COVERED
    monkeypatch.setenv("SEMANTIK_ARRANGER_FIGURE_VETO_MIN_COVERED", "0.75")
    assert pa.resolve_arranger_figure_veto_min_covered() == 0.75
    for bad in ("", "abc", "nan", "inf", "-0.5", "1.5"):
        monkeypatch.setenv("SEMANTIK_ARRANGER_FIGURE_VETO_MIN_COVERED", bad)
        assert pa.resolve_arranger_figure_veto_min_covered() == pa._FIGURE_VETO_MIN_COVERED


# --- WIRING: the veto must actually fire THROUGH arrange_regions -------------
def test_veto_is_wired_into_arrange_regions_and_surfaces_in_the_audit(monkeypatch):
    """End-to-end: a figure inside an arranger-typed `table` is dropped by the lane.

    Guards the WIRING (not just the predicate): the veto runs between
    build_figure_regions and the reading-order splice, the vetoed image FB is then
    neither expected nor claimed by the coverage invariant (same posture as a
    page-raster skip -- if it were still 'expected', _assert_coverage would raise),
    and the tally reaches the audit.
    """
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    monkeypatch.setenv("SEMANTIK_ARRANGER_FIGURE_VETO", "1")
    # A born-digital page (no tesseract dims) so text FBs are POINT space too.
    fbs = [_fb("Row one cell"), _img_fb(bbox=(120.0, 250.0, 300.0, 330.0))]
    # The arranger types the text unit a TABLE spanning the region the figure is in.
    # _fb() gives bbox (0, y0, 100, y0+10) -> widen via a table block over it.
    content = _arr([{"ids": ["p1_b00"], "type": "table"}])
    shared = {"pages": [{"page_num": 1, "width": 612.0, "height": 792.0}]}
    # Make the table region's bbox actually enclose the figure by placing the text
    # FB's bbox over it.
    fbs[0].raw.bbox = (100.0, 240.0, 500.0, 400.0)
    regions, audit, _f = _drive_arrange_regions(
        monkeypatch, fbs, [content],
        image_candidates=[_cand(1, (120.0, 250.0, 300.0, 330.0))],
        shared=shared,
    )
    assert [r for r in regions if r.kind == "figure"] == []   # vetoed
    assert audit["figures"] == 0
    assert audit["arranger_veto"] == 1
    assert audit["figure_veto_rows"][0]["reason"] == "arranger_table_overlap"


def test_veto_off_keeps_the_same_figure_through_arrange_regions(monkeypatch):
    """The revert lever, end-to-end: flag off -> the figure survives the lane."""
    monkeypatch.setenv("SEMANTIK_PAGE_ARRANGER_CHECKPOINT", "0")
    monkeypatch.setenv("SEMANTIK_ARRANGER_FIGURE_VETO", "0")
    fbs = [_fb("Row one cell"), _img_fb(bbox=(120.0, 250.0, 300.0, 330.0))]
    fbs[0].raw.bbox = (100.0, 240.0, 500.0, 400.0)
    content = _arr([{"ids": ["p1_b00"], "type": "table"}])
    shared = {"pages": [{"page_num": 1, "width": 612.0, "height": 792.0}]}
    regions, audit, _f = _drive_arrange_regions(
        monkeypatch, fbs, [content],
        image_candidates=[_cand(1, (120.0, 250.0, 300.0, 330.0))],
        shared=shared,
    )
    assert len([r for r in regions if r.kind == "figure"]) == 1
    assert audit["figures"] == 1
    assert audit["arranger_veto"] == 0


# ---------------------------------------------------------------------------
# (p) DETERMINISTIC id-repair before re-roll (SEMANTIK_ARRANGER_ID_REPAIR) — P2.
# Repairs the three MECHANICAL id failure classes on the current response before
# spending the next ladder rung, so a fixable page finishes with ZERO extra POSTs
# instead of re-POSTing the whole page. Default OFF -> byte-identical ladder.
# ---------------------------------------------------------------------------
_UNITS_2 = [{"id": "p1_b00", "text": "Hello"}, {"id": "p1_b01", "text": "World"}]


def _isolate_side_effects(monkeypatch, tmp_path):
    """Keep the always-on P3 usage tap + the best-effort capture out of the repo."""
    monkeypatch.setenv("SEMANTIK_DATA_DIR", str(tmp_path / "sdata"))
    monkeypatch.setenv("ED4ALL_TRAINING_CAPTURES_DIR", str(tmp_path / "caps"))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)


def test_id_repair_off_is_byte_identical_ladder(monkeypatch, tmp_path):
    """(p.a) Flag OFF -> a missing-id page still spends all three rungs (unchanged)."""
    _isolate_side_effects(monkeypatch, tmp_path)
    monkeypatch.delenv("SEMANTIK_ARRANGER_ID_REPAIR", raising=False)
    bad = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])  # missing p1_b01
    good = _arr([{"ids": ["p1_b00", "p1_b01"], "type": "paragraph"}])
    fake = _FakeRequests([bad, bad, good])
    res = pa.arrange_page(_Seat(), "IMG", list(_UNITS_2), 1, requests_module=fake)
    assert res["status"] == "ok"
    assert res["attempts"] == 3 and len(fake.posts) == 3  # ladder unchanged


def test_id_repair_missing_id_fixed_zero_extra_posts(monkeypatch, tmp_path):
    """(p.c) Flag ON -> a missing id is inserted at its source-order neighbor,
    the page validates, and the ladder STOPS at rung 1 (no re-roll POST)."""
    _isolate_side_effects(monkeypatch, tmp_path)
    monkeypatch.setenv("SEMANTIK_ARRANGER_ID_REPAIR", "1")
    bad = _arr([{"ids": ["p1_b00"], "type": "paragraph"}])  # missing p1_b01
    fake = _FakeRequests([bad])  # only ONE response queued: no re-roll may fire
    res = pa.arrange_page(_Seat(), "IMG", list(_UNITS_2), 1, requests_module=fake)
    assert res["status"] == "ok" and res["attempts"] == 1
    assert len(fake.posts) == 1  # zero extra POSTs
    ids = [uid for blk in res["arrangement"]["blocks"] for uid in blk["ids"]]
    assert sorted(ids) == ["p1_b00", "p1_b01"]  # missing id recovered
    # inserted adjacent to its source-order neighbor (right after p1_b00)
    assert ids.index("p1_b01") == ids.index("p1_b00") + 1
    assert any(r.get("op") == "missing_insert" for r in res["repairs"])


def test_id_repair_dup_only_zero_extra_posts(monkeypatch, tmp_path):
    """(p.b) Flag ON -> a dup-only page needs no re-roll (single POST)."""
    _isolate_side_effects(monkeypatch, tmp_path)
    monkeypatch.setenv("SEMANTIK_ARRANGER_ID_REPAIR", "1")
    dup = _arr([{"ids": ["p1_b00", "p1_b00", "p1_b01"], "type": "paragraph"}])
    fake = _FakeRequests([dup])
    res = pa.arrange_page(_Seat(), "IMG", list(_UNITS_2), 1, requests_module=fake)
    assert res["status"] == "ok" and res["attempts"] == 1
    assert len(fake.posts) == 1
    ids = [uid for blk in res["arrangement"]["blocks"] for uid in blk["ids"]]
    assert sorted(ids) == ["p1_b00", "p1_b01"]


def test_id_repair_hallucinated_id_dropped(monkeypatch, tmp_path):
    """(p.e) Flag ON -> an unknown/hallucinated id is dropped, page validates."""
    _isolate_side_effects(monkeypatch, tmp_path)
    monkeypatch.setenv("SEMANTIK_ARRANGER_ID_REPAIR", "1")
    hall = _arr([{"ids": ["p1_b00", "p1_b01", "zz"], "type": "paragraph"}])
    fake = _FakeRequests([hall])
    res = pa.arrange_page(_Seat(), "IMG", list(_UNITS_2), 1, requests_module=fake)
    assert res["status"] == "ok" and len(fake.posts) == 1
    ids = [uid for blk in res["arrangement"]["blocks"] for uid in blk["ids"]]
    assert "zz" not in ids and sorted(ids) == ["p1_b00", "p1_b01"]
    assert any(r.get("op") == "unknown_drop" for r in res["repairs"])


def test_id_repair_genuine_structural_failure_ladder_unchanged(monkeypatch, tmp_path):
    """(p.d) Flag ON but a NON-mechanical problem (invalid type) is present ->
    repair is NOT attempted and the ladder proceeds exactly as flag-off (3 POSTs)."""
    _isolate_side_effects(monkeypatch, tmp_path)
    monkeypatch.setenv("SEMANTIK_ARRANGER_ID_REPAIR", "1")
    # invalid, non-coercible type AND a missing id -> problems are not all mechanical
    bad = _arr([{"ids": ["p1_b00"], "type": "banana"}])
    good = _arr([{"ids": ["p1_b00", "p1_b01"], "type": "paragraph"}])
    fake = _FakeRequests([bad, bad, good])
    res = pa.arrange_page(_Seat(), "IMG", list(_UNITS_2), 1, requests_module=fake)
    assert res["status"] == "ok"
    assert res["attempts"] == 3 and len(fake.posts) == 3  # ladder unchanged
    assert not any(r.get("op") in ("missing_insert", "unknown_drop") for r in res["repairs"])
