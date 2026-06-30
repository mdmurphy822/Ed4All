"""misconception_rich — MisconceptionProductiveFailureValidator tests."""

from __future__ import annotations

from lib.validators.misconception_productive_failure import (
    MisconceptionProductiveFailureValidator,
)


def _block(**kw):
    base = {"block_type": "misconception", "block_id": "p#misc_0"}
    base.update(kw)
    return base


def test_off_no_ops_with_info(monkeypatch):
    monkeypatch.delenv("ED4ALL_MISCONCEPTION_RICH", raising=False)
    res = MisconceptionProductiveFailureValidator().validate({"blocks": [_block()]})
    assert res.passed is True
    assert any(i.code == "MISCONCEPTION_RICH_DISABLED" for i in res.issues)


def test_on_pass_named_with_three_phases():
    html = (
        '<div class="misconception-card">'
        '<p class="mc-named-concept">like terms</p>'
        '<p class="misconception-predict">Predict...</p>'
        '<p class="misconception-claim">Wrong idea</p>'
        '<p class="mc-reconcile">Why it fails</p></div>'
    )
    res = MisconceptionProductiveFailureValidator().validate(
        {
            "blocks": [_block(content=html, mc_named_concept="like_terms")],
            "misconception_rich_enabled": True,
        }
    )
    assert res.passed is True
    assert not [i for i in res.issues if i.severity == "warning"]


def test_on_warns_generic_claim_correction_card():
    html = (
        '<div class="misconception-card">'
        '<p class="misconception-claim">Wrong</p>'
        '<p class="misconception-correction">Right</p></div>'
    )
    res = MisconceptionProductiveFailureValidator().validate(
        {
            "blocks": [_block(content=html)],
            "misconception_rich_enabled": True,
        }
    )
    codes = {i.code for i in res.issues}
    assert "MISCONCEPTION_NO_NAMED_CONCEPT" in codes
    assert "MISCONCEPTION_NO_PRODUCTIVE_FAILURE" in codes
    assert res.passed is True  # warning-day-1, never blocks
    assert res.action == "regenerate"


def test_on_outline_dict_requires_named_concept():
    # named present -> pass
    res_ok = MisconceptionProductiveFailureValidator().validate(
        {
            "blocks": [_block(content={"misconception": "x"}, mc_named_concept="foo")],
            "misconception_rich_enabled": True,
        }
    )
    assert not [i for i in res_ok.issues if i.severity == "warning"]
    # named absent -> warning
    res_bad = MisconceptionProductiveFailureValidator().validate(
        {
            "blocks": [_block(content={"misconception": "x"})],
            "misconception_rich_enabled": True,
        }
    )
    assert "MISCONCEPTION_NO_NAMED_CONCEPT" in {i.code for i in res_bad.issues}


def test_non_misconception_blocks_ignored():
    res = MisconceptionProductiveFailureValidator().validate(
        {
            "blocks": [{"block_type": "concept", "block_id": "c", "content": "x"}],
            "misconception_rich_enabled": True,
        }
    )
    assert res.metadata["misconception_productive_failure"]["misconception_blocks"] == 0
