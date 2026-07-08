"""Mayer CTML — MayerCtmlValidator regression suite.

Fixtures are built directly as ``Block`` dataclasses / dicts in-test; NO course
slug / path / run is discovered (the B04/B06/B05 types only emit under
``ED4ALL_NEW_BLOCK_TYPES`` on a real run, so a course run would yield zero blocks
to audit — see the validator module docstring).
"""
from __future__ import annotations

import yaml

from Courseforge.scripts.blocks import Block
from lib.validators.mayer_ctml import MayerCtmlValidator, resolve_mayer_ctml


def _block(content, *, block_type="diagram", block_id="b1") -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="week_01_content",
        sequence=1,
        content=content,
    )


def _validate(blocks, **extra):
    inputs = {"blocks": blocks, "mayer_ctml_enabled": True}
    inputs.update(extra)
    return MayerCtmlValidator().validate(inputs)


# --------------------------------------------------------------------------
# flag gating
# --------------------------------------------------------------------------


def test_disabled_is_noop_pass():
    block = _block('<figure><img src="x"><figcaption>A</figcaption></figure>')
    res = MayerCtmlValidator().validate(
        {"blocks": [block], "mayer_ctml_enabled": False}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"MAYER_CTML_DISABLED"}
    assert all(i.severity == "info" for i in res.issues)
    assert res.action is None


def test_garbage_env_parse_with_fallback(monkeypatch):
    monkeypatch.setenv("ED4ALL_MAYER_CTML", "banana")
    assert resolve_mayer_ctml() is False
    block = _block('<figure><img src="x"></figure>')
    # No explicit enable override → resolver reads env → off.
    res = MayerCtmlValidator().validate({"blocks": [block]})
    assert {i.code for i in res.issues} == {"MAYER_CTML_DISABLED"}


def test_truthy_env_enables(monkeypatch):
    monkeypatch.setenv("ED4ALL_MAYER_CTML", "ON")
    assert resolve_mayer_ctml() is True


# --------------------------------------------------------------------------
# SIGNALING
# --------------------------------------------------------------------------


def test_signaling_missing_cue_flagged():
    html = (
        '<figure class="diagram"><img src="x">'
        + ("A diagram body with substantive explanatory prose. " * 3)
        + "</figure>"
    )
    res = _validate([_block(html)])
    codes = {i.code for i in res.issues}
    assert "CTML_NO_SIGNALING" in codes
    assert res.action == "regenerate"
    assert res.metadata["mayer_ctml"]["principle_counts"]["CTML_NO_SIGNALING"] == 1


def test_signaling_with_figcaption_clean():
    html = (
        '<figure class="diagram"><img src="x">'
        + ("A diagram body with substantive explanatory prose. " * 3)
        + "<figcaption>Cycle of water phases</figcaption></figure>"
    )
    res = _validate([_block(html)])
    assert not any(i.code == "CTML_NO_SIGNALING" for i in res.issues)


# --------------------------------------------------------------------------
# SPATIAL CONTIGUITY
# --------------------------------------------------------------------------


def test_contiguity_caption_outside_figure_flagged():
    # <video> inside the figure, but the caption is rendered OUTSIDE it.
    html = (
        '<figure class="multimedia"><video controls></video></figure>'
        "<p>A caption rendered outside the figure.</p>"
    )
    res = _validate([_block(html, block_type="multimedia")])
    assert any(i.code == "CTML_CAPTION_NOT_ADJACENT" for i in res.issues)


def test_contiguity_caption_inside_figure_clean():
    html = (
        '<figure class="multimedia"><video controls></video>'
        "<figcaption>Demo clip</figcaption></figure>"
    )
    res = _validate([_block(html, block_type="multimedia")])
    assert not any(i.code == "CTML_CAPTION_NOT_ADJACENT" for i in res.issues)


# --------------------------------------------------------------------------
# REDUNDANCY
# --------------------------------------------------------------------------


def test_redundancy_transcript_equals_audio_desc_flagged():
    same = "The water cycle moves water between sea sky and land."
    html = (
        '<figure class="multimedia"><video controls></video>'
        f"<details data-cf-transcript><p>{same}</p></details>"
        f'<p class="audio-description">Audio description: {same}</p>'
        "<figcaption>Water cycle clip</figcaption></figure>"
    )
    res = _validate([_block(html, block_type="multimedia")])
    assert any(i.code == "CTML_REDUNDANT_NARRATION" for i in res.issues)


def test_redundancy_distinct_transcript_vs_ad_clean():
    html = (
        '<figure class="multimedia"><video controls></video>'
        "<details data-cf-transcript><p>Narrator explains the full cycle "
        "in spoken sentences.</p></details>"
        '<p class="audio-description">Audio description: an animated blue '
        "arrow loops between cloud, sea, and ground.</p>"
        "<figcaption>Water cycle clip</figcaption></figure>"
    )
    res = _validate([_block(html, block_type="multimedia")])
    assert not any(i.code == "CTML_REDUNDANT_NARRATION" for i in res.issues)


# --------------------------------------------------------------------------
# SEGMENTING
# --------------------------------------------------------------------------


def test_segmenting_long_unsegmented_body_flagged():
    long_body = "Continuous worked-example prose with no breaks. " * 40  # > ceiling
    html = (
        '<figure class="multimedia"><video controls></video>'
        f"<figcaption>Cue</figcaption><p>{long_body}</p></figure>"
    )
    res = _validate([_block(html, block_type="worked_example")])
    assert any(i.code == "CTML_NOT_SEGMENTED" for i in res.issues)


def test_segmenting_with_details_clean():
    long_body = "Continuous worked-example prose with no breaks. " * 40
    html = (
        '<figure class="multimedia"><video controls></video>'
        f"<figcaption>Cue</figcaption><details><p>{long_body}</p></details></figure>"
    )
    res = _validate([_block(html, block_type="worked_example")])
    assert not any(i.code == "CTML_NOT_SEGMENTED" for i in res.issues)


# --------------------------------------------------------------------------
# dual dict path
# --------------------------------------------------------------------------


def test_dict_signaling_flagged():
    block = {
        "block_id": "d1",
        "block_type": "diagram",
        "content": {"body": "A substantive diagram description with prose " * 2},
    }
    res = _validate([block])
    assert any(i.code == "CTML_NO_SIGNALING" for i in res.issues)


def test_dict_signaling_with_caption_clean():
    block = {
        "block_id": "d1",
        "block_type": "diagram",
        "content": {
            "body": "A substantive diagram description with prose " * 2,
            "caption": "Water cycle diagram",
        },
    }
    res = _validate([block])
    assert not any(i.code == "CTML_NO_SIGNALING" for i in res.issues)


def test_dict_segmenting_flagged_and_str_only_skipped():
    long_body = "Continuous multimedia narration with no segment breaks. " * 40
    block = {
        "block_id": "d2",
        "block_type": "worked_example",
        "content": {"caption": "Cue", "body": long_body},
    }
    res = _validate([block])
    codes = {i.code for i in res.issues}
    assert "CTML_NOT_SEGMENTED" in codes
    # str-only principles never fire on the dict path:
    assert "CTML_CAPTION_NOT_ADJACENT" not in codes
    assert "CTML_REDUNDANT_NARRATION" not in codes


def test_dict_segmenting_with_step_list_clean():
    long_body = "Continuous multimedia narration with no segment breaks. " * 40
    block = {
        "block_id": "d3",
        "block_type": "worked_example",
        "content": {
            "caption": "Cue",
            "body": long_body,
            "steps": ["step one", "step two", "step three"],
        },
    }
    res = _validate([block])
    assert not any(i.code == "CTML_NOT_SEGMENTED" for i in res.issues)


# --------------------------------------------------------------------------
# scope + graceful handling
# --------------------------------------------------------------------------


def test_non_visual_block_ignored():
    block = _block("<p>Just an idea, no figure or media.</p>", block_type="concept")
    res = _validate([block])
    assert res.metadata["mayer_ctml"]["audited"] == 0
    assert not res.issues


def test_empty_input_graceful_pass():
    res = MayerCtmlValidator().validate({"mayer_ctml_enabled": True})
    assert res.passed is True
    assert {i.code for i in res.issues} == {"MISSING_BLOCKS_INPUT"}


def test_issue_list_cap():
    # >50 visual blocks each missing signaling → issue list capped at 50.
    bad = '<figure class="diagram"><img src="x">' + ("body prose here. " * 4) + "</figure>"
    blocks = [_block(bad, block_id=f"b{i}") for i in range(80)]
    res = _validate(blocks)
    assert len(res.issues) <= 50
    # but the flagged COUNT in metadata reflects all of them.
    assert res.metadata["mayer_ctml"]["flagged"] >= 80


# --------------------------------------------------------------------------
# decision capture
# --------------------------------------------------------------------------


class _SpyCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, *, decision_type, decision, rationale, **kwargs):
        self.events.append(
            {"decision_type": decision_type, "decision": decision, "rationale": rationale}
        )


def test_decision_capture_fires():
    spy = _SpyCapture()
    html = '<figure class="diagram"><img src="x">' + ("body prose " * 5) + "</figure>"
    _validate([_block(html)], decision_capture=spy)
    assert len(spy.events) == 1
    ev = spy.events[0]
    assert ev["decision_type"] == "mayer_ctml_check"
    assert len(ev["rationale"]) >= 20
    assert "CTML_NO_SIGNALING" in ev["rationale"]
    assert "1" in ev["rationale"]  # audited count interpolated


# --------------------------------------------------------------------------
# config wiring
# --------------------------------------------------------------------------


def test_gate_wired_in_both_workflows_post_rewrite():
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    with open(os.path.join(root, "config", "workflows.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    for wf_name in ("course_generation", "textbook_to_course"):
        wf = cfg["workflows"][wf_name]
        phases = wf["phases"]
        post = next(p for p in phases if p["name"] == "post_rewrite_validation")
        gate_ids = {g["gate_id"] for g in post["validation_gates"]}
        assert "mayer_ctml" in gate_ids, wf_name
        gate = next(g for g in post["validation_gates"] if g["gate_id"] == "mayer_ctml")
        # Flipped critical at flip-wave-2 (calibration-validated) — see the
        # gate comment in config/workflows.yaml.
        assert gate["severity"] == "critical"
        assert gate["behavior"]["on_fail"] == "block"
        assert gate["validator"] == "lib.validators.mayer_ctml.MayerCtmlValidator"


def test_decision_event_enum_has_member():
    import json
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    with open(
        os.path.join(root, "schemas", "events", "decision_event.schema.json")
    ) as fh:
        schema = json.load(fh)
    enum = schema["properties"]["decision_type"]["enum"]
    assert "mayer_ctml_check" in enum
