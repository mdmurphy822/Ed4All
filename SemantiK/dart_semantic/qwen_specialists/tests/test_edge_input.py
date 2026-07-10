"""Phase-1 acceptance — edge-input builder + furniture dedup (no model / GPU).

Covers the plan §Phase-1 acceptance list verbatim:
  * a long block -> head/tail are the first/last N tokens; n_tokens correct;
  * a short block (n_tokens <= 2N) -> full text, no truncation, no token loss;
  * content-block role read from CouncilState (not payload); a heading reads
    payload['confidence'];
  * a repeated running header on N pages -> ONE record, dup_count == N, page
    list populated (reuses _running_header_norm / _detect_running_headers);
  * the input region + feature_blocks are unchanged after the call (pure read);
  * the existing reviewer-prompt no-envelope / JSON-only invariant still holds.

These are pure-Python: no LLM, no GPU, no env mutation beyond a monkeypatched
SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS in one test.
"""

from __future__ import annotations

from dart_semantic.qwen_specialists.reviewer_prompt import (
    _SYSTEM_REVIEWER,
    build_edge_input,
    build_reviewer_request,
    dedup_furniture_records,
)
from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Fixtures / builders (mirror test_reviewer / test_block_resegment).
# ---------------------------------------------------------------------------


def _fb(text: str, *, page: int = 1, bbox=(0.0, 0.0, 10.0, 10.0)) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=bbox,
        page_width=100.0,
        page_height=100.0,
    )
    return FeatureBlock(
        raw=raw,
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
    )


def _region(kind, fb_indices, *, text=None, confidence=None, source_region_id=None):
    payload = {}
    if text is not None:
        payload["text"] = text
    if confidence is not None:
        payload["confidence"] = confidence
    return Region(
        kind=kind,
        feature_block_indices=tuple(fb_indices),
        payload=payload,
        source_region_id=source_region_id,
    )


# --- a CouncilState-shaped mock carrying the Structure structural_role head --


class _StructSignal:
    def __init__(self, region_id, label, conf, head_name="structural_role"):
        self.head_name = head_name
        self.region_id = region_id
        self.top_k_labels = [label]
        self.top_k_confidences = [conf]


class _BertOut:
    def __init__(self, signals):
        self.signals = signals


class _State:
    """Minimal CouncilState stand-in carrying Structure structural_role signals.

    ``struct_signals`` maps an FB index -> (label, confidence).
    """

    def __init__(self, struct_signals=None):
        sigs = [
            _StructSignal(idx, label, conf)
            for idx, (label, conf) in (struct_signals or {}).items()
        ]
        self.outputs = {"structure": _BertOut(sigs)}


# ---------------------------------------------------------------------------
# head/tail slicing on a long block.
# ---------------------------------------------------------------------------


def test_edge_record_head_tail_slicing():
    tokens = [f"w{i}" for i in range(100)]
    region = _region("paragraph", (0,))
    fbs = [_fb(" ".join(tokens))]

    rec = build_edge_input(
        region, block_id=7, feature_blocks=fbs, council_state=None, edge_tokens=12
    )

    assert rec["idx"] == 7
    assert rec["council_kind"] == "paragraph"
    assert rec["dup_count"] == 1
    assert rec["n_tokens"] == 100
    # Long block -> head/tail kept, full text omitted.
    assert "text" not in rec
    assert rec["head"].split() == tokens[:12]
    assert rec["tail"].split() == tokens[-12:]


def test_edge_record_edge_tokens_reads_env(monkeypatch):
    """edge_tokens defaults to the _EDGE_TOKENS resolver when not passed."""
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", "3")
    tokens = [f"t{i}" for i in range(40)]
    region = _region("paragraph", (0,))
    fbs = [_fb(" ".join(tokens))]

    rec = build_edge_input(region, block_id=0, feature_blocks=fbs)

    assert rec["head"].split() == tokens[:3]
    assert rec["tail"].split() == tokens[-3:]


# ---------------------------------------------------------------------------
# full text on a short block (no token loss).
# ---------------------------------------------------------------------------


def test_edge_record_full_text_when_short():
    source = "alpha beta gamma"
    region = _region("paragraph", (0,))
    fbs = [_fb(source)]

    rec = build_edge_input(
        region, block_id=0, feature_blocks=fbs, council_state=None, edge_tokens=12
    )

    assert rec["n_tokens"] == 3  # 3 <= 2*12 -> full text
    assert rec["text"] == source
    assert "head" not in rec
    assert "tail" not in rec
    # No token loss: the full text round-trips the source tokens verbatim.
    assert rec["text"].split() == source.split()


def test_edge_record_full_text_at_boundary():
    """Exactly 2N tokens still emits full text (<= boundary, not <)."""
    tokens = [f"b{i}" for i in range(24)]  # 2 * 12
    region = _region("paragraph", (0,))
    fbs = [_fb(" ".join(tokens))]

    rec = build_edge_input(
        region, block_id=0, feature_blocks=fbs, council_state=None, edge_tokens=12
    )

    assert rec["n_tokens"] == 24
    assert rec["text"] == " ".join(tokens)
    assert "head" not in rec


# ---------------------------------------------------------------------------
# council_kind + confidence sourcing.
# ---------------------------------------------------------------------------


def test_edge_record_council_kind_and_confidence():
    # Content block: role comes from CouncilState structural_role, NOT payload
    # (content regions carry no payload['confidence']).
    content_region = _region("code_block", (0,), text="x = 1")
    assert "confidence" not in (content_region.payload or {})
    fbs = [_fb("x = 1"), _fb("Introduction")]
    state = _State({0: ("paragraph", 0.77)})

    rec = build_edge_input(
        content_region, block_id=0, feature_blocks=fbs, council_state=state
    )
    assert rec["council_kind"] == "code_block"
    assert rec["role"] == "paragraph"  # re-derived from the council head
    assert rec["confidence"] == 0.77

    # Heading: confidence comes from payload['confidence']; role is "heading".
    heading_region = _region("heading", (1,), text="Introduction", confidence=0.81)
    rec_h = build_edge_input(
        heading_region, block_id=1, feature_blocks=fbs, council_state=state
    )
    assert rec_h["council_kind"] == "heading"
    assert rec_h["role"] == "heading"
    assert rec_h["confidence"] == 0.81


def test_edge_record_content_role_falls_back_without_signal():
    """No council signal -> role falls back to the council kind, conf None."""
    region = _region("code_block", (0,), text="y = 2")
    fbs = [_fb("y = 2")]

    rec = build_edge_input(region, block_id=0, feature_blocks=fbs, council_state=None)

    assert rec["role"] == "code_block"
    assert rec["confidence"] is None


# ---------------------------------------------------------------------------
# furniture dedup.
# ---------------------------------------------------------------------------


def _running_header_fb(num: int, page: int) -> FeatureBlock:
    # Top margin band (y0 < 0.12 * page_height=100 -> y0 < 12); a 4-word block
    # carrying a page number, so it normalizes to "shades of accessibility #".
    return _fb(f"Shades of Accessibility {num}", page=page, bbox=(0.0, 2.0, 50.0, 8.0))


def test_furniture_dedup_dup_count():
    # Four pages, the same running header on each (page number varies).
    fbs = [_running_header_fb(num=p, page=p) for p in range(1, 5)]
    regions = [
        _region("metadata_drop", (i,), text=fbs[i].raw.text) for i in range(4)
    ]

    records = dedup_furniture_records(regions, feature_blocks=fbs)

    furniture = [r for r in records if r["council_kind"] == "metadata_drop"]
    assert len(furniture) == 1, "repeated running header should collapse to ONE record"
    rec = furniture[0]
    assert rec["dup_count"] == 4
    assert sorted(rec["pages"]) == [1, 2, 3, 4]


def test_furniture_dedup_keeps_content_records():
    # A content region interleaved with furniture passes through untouched.
    fbs = [_running_header_fb(num=p, page=p) for p in range(1, 5)]
    fbs.append(_fb("This is real body content on the page.", page=2))
    regions = [_region("metadata_drop", (i,), text=fbs[i].raw.text) for i in range(4)]
    regions.append(_region("paragraph", (4,)))

    records = dedup_furniture_records(regions, feature_blocks=fbs)

    kinds = [r["council_kind"] for r in records]
    assert kinds.count("metadata_drop") == 1
    assert kinds.count("paragraph") == 1
    para = next(r for r in records if r["council_kind"] == "paragraph")
    assert para["dup_count"] == 1
    assert "pages" not in para  # non-furniture records carry only ``page``


# ---------------------------------------------------------------------------
# pure-read guard — no region / FB mutation.
# ---------------------------------------------------------------------------


def test_no_region_mutation():
    tokens = [f"w{i}" for i in range(50)]
    region = _region("paragraph", (0, 1), text="payload-text")
    fbs = [_fb(" ".join(tokens[:25])), _fb(" ".join(tokens[25:]))]

    fb_texts_before = [fb.raw.text for fb in fbs]
    fb_pages_before = [fb.raw.page for fb in fbs]
    payload_before = dict(region.payload)
    indices_before = region.feature_block_indices

    build_edge_input(region, block_id=0, feature_blocks=fbs, council_state=None)

    # FB token multiset AND order are byte-identical (the pure-read contract).
    assert [fb.raw.text for fb in fbs] == fb_texts_before
    assert [fb.raw.page for fb in fbs] == fb_pages_before
    # The frozen region is untouched.
    assert region.payload == payload_before
    assert region.feature_block_indices == indices_before


# ---------------------------------------------------------------------------
# the existing reviewer-prompt no-envelope / JSON-only invariant still holds.
# ---------------------------------------------------------------------------


def test_no_envelope_json_only():
    # The Stage-6 generation envelope must NOT have crept into the reviewer
    # system directive when build_edge_input landed.
    low = _SYSTEM_REVIEWER.lower()
    assert "convert" not in low or "fragment" not in low
    assert "json only" in low  # the "OUTPUT — JSON ONLY" mandate is intact

    # The prompt builder still emits a SYSTEM/USER reviewer prompt with no
    # "convert … into … fragment" envelope phrasing.
    region = _region("heading", (0,), text="Section One", confidence=0.81)
    prompt = build_reviewer_request(region, (None, None), 0).lower()
    assert "convert the single" not in prompt
    assert "into one accessible html5 fragment" not in prompt


# ---------------------------------------------------------------------------
# ITEM6 — council_top_k edge-record evidence (k=3, probs @2dp). From the live
# council_state, or a fallback to the persisted region.provenance['role_top_k']
# on the post-run re-drive path (state absent). Byte-stable when neither exists.
# ---------------------------------------------------------------------------


class _MultiStructSignal:
    def __init__(self, region_id, labels, confs, head_name="structural_role"):
        self.head_name = head_name
        self.region_id = region_id
        self.top_k_labels = list(labels)
        self.top_k_confidences = list(confs)


class _MultiState:
    def __init__(self, region_id, labels, confs):
        self.outputs = {
            "structure": _BertOut([_MultiStructSignal(region_id, labels, confs)])
        }


def test_edge_record_council_top_k_k3_2dp():
    region = _region("code_block", (0,), text="x = 1")
    fbs = [_fb("x = 1")]
    state = _MultiState(
        0,
        ["code_block", "paragraph", "math", "list_item"],
        [0.512, 0.301, 0.092, 0.05],
    )
    rec = build_edge_input(region, block_id=3, feature_blocks=fbs, council_state=state)
    assert rec["council_top_k"] == [
        ["code_block", 0.51],
        ["paragraph", 0.3],
        ["math", 0.09],
    ]


def test_edge_record_council_top_k_from_provenance_fallback():
    # State absent -> read the persisted provenance stamp; @2dp re-round.
    region = Region(
        kind="paragraph",
        feature_block_indices=(0,),
        payload={"text": "hi"},
        provenance={"role_top_k": [["paragraph", 0.7123], ["blockquote", 0.2011]]},
    )
    fbs = [_fb("hi")]
    rec = build_edge_input(region, block_id=1, feature_blocks=fbs, council_state=None)
    assert rec["council_top_k"] == [["paragraph", 0.71], ["blockquote", 0.2]]


def test_edge_record_no_key_without_signal_or_stamp():
    region = _region("paragraph", (0,), text="plain")
    fbs = [_fb("plain")]
    rec = build_edge_input(region, block_id=2, feature_blocks=fbs, council_state=None)
    assert "council_top_k" not in rec
