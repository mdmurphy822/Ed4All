"""NearDupExampleValidator unit tests (Gap #11 — within-module anchor-example
de-duplication).

Pins the deterministic within-module recurring-anchor-example detector: the same
worked example (its normalized number-sequence signature) recurring across
``>= min_repeat`` blocks in one module is flagged; distinct examples pass; blocks
below the signature-length / distinctness floors never form a match. Numbers are
normalized (``3.990`` == ``3.99``, thousands separators stripped, ``$3.99`` ==
``3.99``) so display variance cannot hide a reuse. No course slugs / publisher
vocabulary anywhere — fixtures are synthetic HTML block dict rows.
"""
from __future__ import annotations

from typing import Any, Dict, List

from lib.validators.near_dup_example import (
    NearDupExampleValidator,
    _CODE_NEAR_DUP,
    _DECISION_TYPE,
    module_key,
    number_signature,
)


# --------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------- #
def _block(*, block_id: str, page_id: str, content: Any, block_type: str = "example") -> Dict[str, Any]:
    return {
        "block_id": block_id,
        "page_id": page_id,
        "block_type": block_type,
        "content": content,
    }


class _RecordingCapture:
    """Minimal DecisionCapture double — records log_decision calls."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(
        self, *, decision_type: str, decision: str, rationale: str, **_kw: Any
    ) -> None:
        self.calls.append(
            {"decision_type": decision_type, "decision": decision, "rationale": rationale}
        )


def _validate(blocks: List[Dict[str, Any]], **extra: Any):
    v = NearDupExampleValidator()
    inputs: Dict[str, Any] = {"blocks": blocks}
    inputs.update(extra)
    return v.validate(inputs)


# An anchor example reused verbatim (>=3 distinct numbers) across blocks.
_ANCHOR = "<p>A pack costs 3.99 dollars for 24 items, so 3.99 / 24 = 0.166 each.</p>"


# --------------------------------------------------------------------- #
# number_signature / module_key primitives
# --------------------------------------------------------------------- #
def test_number_signature_orders_and_normalizes():
    sig = number_signature("<p>Buy 1,000 at $3.990 each over 24 weeks.</p>")
    assert sig == ("1000", "3.99", "24")


def test_number_signature_too_few_numbers_is_none():
    assert number_signature("<p>Only 5 and 6 here.</p>") is None  # 2 numbers < 3


def test_number_signature_too_few_distinct_is_none():
    assert number_signature("<p>7 and 7 and 7 again.</p>") is None  # 1 distinct


def test_number_signature_empty_is_none():
    assert number_signature("") is None
    assert number_signature("<p>No numbers at all.</p>") is None


def test_module_key_from_page_id():
    assert module_key("week_01_content_02") == "week_01"
    assert module_key("week_10_overview") == "week_10"
    assert module_key("module3_content_01") == "module3"
    assert module_key("unit-2_content_00") == "unit-2"


def test_module_key_no_prefix_falls_back():
    assert module_key("standalone") == "standalone"
    assert module_key("") == "<no-page>"


# --------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------- #
def test_recurring_anchor_flagged_at_threshold():
    blocks = [
        _block(block_id=f"week_01_content_{i:02d}#example", page_id=f"week_01_content_{i:02d}", content=_ANCHOR)
        for i in range(3)
    ]
    res = _validate(blocks)
    assert res.passed is True  # warning-severity, never blocks
    codes = [i.code for i in res.issues]
    assert codes.count(_CODE_NEAR_DUP) == 1
    issue = next(i for i in res.issues if i.code == _CODE_NEAR_DUP)
    assert issue.severity == "warning"
    assert issue.location == "week_01"
    assert res.metadata["flagged_modules"] == 1
    assert res.metadata["max_repeat"] == 3


def test_below_threshold_not_flagged():
    # Only 2 blocks share the anchor (default threshold 3).
    blocks = [
        _block(block_id=f"week_01_content_{i:02d}", page_id=f"week_01_content_{i:02d}", content=_ANCHOR)
        for i in range(2)
    ]
    res = _validate(blocks)
    assert res.issues == []
    assert res.metadata["flagged_modules"] == 0


def test_distinct_examples_pass():
    blocks = [
        _block(block_id="week_01_content_00", page_id="week_01_content_00",
               content="<p>2.50 for 10 gives 0.25 each.</p>"),
        _block(block_id="week_01_content_01", page_id="week_01_content_01",
               content="<p>9.00 over 3 weeks is 3.00 per week.</p>"),
        _block(block_id="week_01_content_02", page_id="week_01_content_02",
               content="<p>Area 12 by 5 equals 60 square units.</p>"),
    ]
    res = _validate(blocks)
    assert res.issues == []
    assert res.score == 1.0


def test_same_anchor_in_different_modules_not_flagged():
    # The anchor recurs but SPLIT across three different weeks — no single
    # module reaches the threshold.
    blocks = [
        _block(block_id=f"week_{w:02d}_content_00", page_id=f"week_{w:02d}_content_00", content=_ANCHOR)
        for w in (1, 2, 3)
    ]
    res = _validate(blocks)
    assert res.issues == []


def test_min_repeat_override_from_config():
    blocks = [
        _block(block_id=f"week_01_content_{i:02d}", page_id=f"week_01_content_{i:02d}", content=_ANCHOR)
        for i in range(2)
    ]
    # Lower the threshold to 2 → the 2-block reuse now flags.
    res = _validate(blocks, min_repeat=2)
    assert any(i.code == _CODE_NEAR_DUP for i in res.issues)


def test_min_repeat_garbage_falls_back_to_default():
    blocks = [
        _block(block_id=f"week_01_content_{i:02d}", page_id=f"week_01_content_{i:02d}", content=_ANCHOR)
        for i in range(2)
    ]
    # Garbage / <2 → default 3, so 2 blocks do NOT flag.
    for bad in ("garbage", 1, 0, -5, None):
        res = _validate(blocks, min_repeat=bad)
        assert not any(i.code == _CODE_NEAR_DUP for i in res.issues)


def test_normalization_matches_display_variants():
    # Same example written with $, thousands-commas, and trailing zeros must
    # collide into one signature.
    variants = [
        "<p>$3.99 for 24 → 0.166.</p>",
        "<p>3.990 dollars, 24 units, 0.1660 each.</p>",
        "<p>Price 3.99 over 24 gives 0.166.</p>",
    ]
    blocks = [
        _block(block_id=f"week_02_content_{i:02d}", page_id=f"week_02_content_{i:02d}", content=c)
        for i, c in enumerate(variants)
    ]
    res = _validate(blocks)
    assert any(i.code == _CODE_NEAR_DUP for i in res.issues)


# --------------------------------------------------------------------- #
# Decision capture
# --------------------------------------------------------------------- #
def test_decision_capture_fires_per_module():
    blocks = [
        _block(block_id=f"week_01_content_{i:02d}", page_id=f"week_01_content_{i:02d}", content=_ANCHOR)
        for i in range(3)
    ] + [
        _block(block_id="week_02_content_00", page_id="week_02_content_00",
               content="<p>10 by 4 by 2 is 80.</p>"),
    ]
    cap = _RecordingCapture()
    _validate(blocks, decision_capture=cap)
    # One event per module that carried a signature-bearing block.
    modules_logged = {c["decision"] for c in cap.calls}
    assert cap.calls, "capture must fire"
    assert all(c["decision_type"] == _DECISION_TYPE for c in cap.calls)
    # week_01 flagged, week_02 passed → both decisions present.
    assert any(c["decision"].startswith("flagged:") for c in cap.calls)
    assert any(c["decision"] == "passed" for c in cap.calls)
    for c in cap.calls:
        assert len(c["rationale"]) >= 20
        assert "min_repeat=" in c["rationale"]


def test_capture_optional_no_crash():
    res = _validate([_block(block_id="week_01_content_00", page_id="week_01_content_00", content=_ANCHOR)])
    assert res.passed is True


# --------------------------------------------------------------------- #
# Input coercion / edge cases
# --------------------------------------------------------------------- #
def test_missing_blocks_input_is_critical_skip():
    res = NearDupExampleValidator().validate({})
    assert res.passed is False
    assert any(i.code == "MISSING_BLOCKS_INPUT" for i in res.issues)


def test_non_list_blocks_is_critical():
    res = NearDupExampleValidator().validate({"blocks": "not-a-list"})
    assert res.passed is False
    assert any(i.code == "INVALID_BLOCKS_INPUT" for i in res.issues)


def test_empty_blocks_pass_vacuously():
    res = _validate([])
    assert res.passed is True
    assert res.score == 1.0
    assert res.issues == []


def test_blocks_without_content_skipped():
    blocks = [
        _block(block_id="week_01_content_00", page_id="week_01_content_00", content=""),
        _block(block_id="week_01_content_01", page_id="week_01_content_01", content=None),
    ]
    res = _validate(blocks)
    assert res.issues == []


def test_hydrates_from_jsonl_path(tmp_path):
    import json

    p = tmp_path / "blocks_outline.jsonl"
    rows = [
        _block(block_id=f"week_01_content_{i:02d}", page_id=f"week_01_content_{i:02d}", content=_ANCHOR)
        for i in range(3)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    res = NearDupExampleValidator().validate({"blocks_outline_path": str(p)})
    assert any(i.code == _CODE_NEAR_DUP for i in res.issues)


# --------------------------------------------------------------------- #
# Wiring — gate_input_routing registration + config/workflows.yaml
# --------------------------------------------------------------------- #
_VALIDATOR_PATH = "lib.validators.near_dup_example.NearDupExampleValidator"


def test_registered_in_default_router():
    from MCP.hardening.gate_input_routing import default_router

    r = default_router()
    assert _VALIDATOR_PATH in r.builders


def test_wired_at_inter_tier_for_both_workflows():
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    doc = yaml.safe_load((root / "config" / "workflows.yaml").read_text(encoding="utf-8"))
    workflows = doc["workflows"] if "workflows" in doc else doc
    found = {}
    for wf_name in ("course_generation", "textbook_to_course"):
        wf = workflows[wf_name]
        phases = wf["phases"] if isinstance(wf, dict) and "phases" in wf else wf
        hit = False
        for phase in phases:
            if phase.get("name") != "inter_tier_validation":
                continue
            for gate in phase.get("validation_gates", []) or []:
                if gate.get("gate_id") == "near_dup_example":
                    assert gate.get("validator") == _VALIDATOR_PATH
                    assert gate.get("severity") == "warning"
                    hit = True
        found[wf_name] = hit
    assert found == {"course_generation": True, "textbook_to_course": True}

