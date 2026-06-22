"""IB4.5 — UdlCoverageValidator unit tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.udl_coverage import UdlCoverageValidator  # noqa: E402


def _block(content, *, block_type="concept", page_id="week_01_overview",
           block_id=None, **kw):
    if block_id is None:
        block_id = f"{page_id}#{block_type}_demo_0"
    return Block(
        block_id=block_id, block_type=block_type, page_id=page_id,
        sequence=0, content=content, **kw,
    )


def _validate(blocks):
    return UdlCoverageValidator().validate({"blocks": blocks})


def _codes(result):
    return [i.code for i in result.issues]


def test_single_representation_block_warns() -> None:
    block = _block("<p>only prose</p>")
    result = _validate([block])
    assert "UDL_SINGLE_REPRESENTATION" in _codes(result)
    assert result.passed is True  # warning-day-1


def test_two_representations_clears_floor() -> None:
    # prose + table => 2 representation modes (derived on read; scenario phrase
    # in the prose clears the autonomy floor too).
    block = _block(
        "<p>Imagine you run a shop.</p><table><caption>t</caption></table>"
    )
    result = _validate([block])
    assert "UDL_SINGLE_REPRESENTATION" not in _codes(result)
    assert "UDL_NO_AUTONOMY_AFFORDANCE" not in _codes(result)


def test_week_with_zero_autonomy_flags() -> None:
    # Two concept blocks, both 2+ reps but no response/engagement affordance.
    b1 = _block("<p>a</p><table></table>", block_id="week_01#concept_a_0")
    b2 = _block("<p>b</p><ul><li>x</li></ul>", block_id="week_01#concept_b_0")
    result = _validate([b1, b2])
    assert "UDL_NO_AUTONOMY_AFFORDANCE" in _codes(result)
    assert result.passed is True


def test_engagement_affordance_clears_week_autonomy_floor() -> None:
    b1 = _block("<p>a</p><table></table>", block_id="week_01#concept_a_0",
                engagement_affordance="real_world")
    b2 = _block("<p>b</p><ul><li>x</li></ul>", block_id="week_01#concept_b_0")
    result = _validate([b1, b2])
    assert "UDL_NO_AUTONOMY_AFFORDANCE" not in _codes(result)


def test_response_format_clears_week_autonomy_floor() -> None:
    # self_check_question derives response_formats=("select",) on read.
    sc = _block(
        "<p>q</p><details><summary>a</summary></details>",
        block_type="self_check_question", block_id="week_01#scq_0",
    )
    result = _validate([sc])
    assert "UDL_NO_AUTONOMY_AFFORDANCE" not in _codes(result)


def test_chrome_and_objective_skipped() -> None:
    chrome = _block("<div></div>", block_type="chrome", block_id="week_01#chrome_0")
    obj = _block("Understand X.", block_type="objective", block_id="week_01#objective_0",
                 objective_ids=("CO-01",))
    result = _validate([chrome, obj])
    # Neither should produce a single-representation warning.
    assert "UDL_SINGLE_REPRESENTATION" not in _codes(result)


def test_derives_on_read_when_fields_empty() -> None:
    # Block carries no UDL fields; validator derives 3 reps from content.
    block = _block("<p>prose</p><table></table><img alt='x'>")
    result = _validate([block])
    assert "UDL_SINGLE_REPRESENTATION" not in _codes(result)


def test_missing_blocks_input_passes_clean() -> None:
    result = UdlCoverageValidator().validate({})
    assert result.passed is True
    assert _codes(result) == ["MISSING_BLOCKS_INPUT"]


def test_decision_type_accepted_by_schema() -> None:
    schema_path = _REPO_ROOT / "schemas" / "events" / "decision_event.schema.json"
    schema = json.loads(schema_path.read_text())
    enum = schema["properties"]["decision_type"]["enum"]
    assert "udl_coverage_check" in enum
