"""recall_self_check — RecallSelfCheckFormatValidator tests."""

from __future__ import annotations

from lib.validators.recall_self_check import RecallSelfCheckFormatValidator


def _block(**kw):
    base = {"block_type": "self_check_question", "block_id": "p#sc_0"}
    base.update(kw)
    return base


def test_off_no_ops_with_info(monkeypatch):
    monkeypatch.delenv("ED4ALL_RECALL_SELF_CHECK", raising=False)
    res = RecallSelfCheckFormatValidator().validate({"blocks": [_block()]})
    assert res.passed is True
    assert any(i.code == "RECALL_SELF_CHECK_DISABLED" for i in res.issues)


def test_on_pass_for_produce_answer_with_reveal():
    html = (
        '<div class="self-check"><p class="sc-recall-prompt">Name the law.</p>'
        '<input type="text"><details><summary>Show answer</summary>'
        '<div>Distributive</div></details></div>'
    )
    res = RecallSelfCheckFormatValidator().validate(
        {
            "blocks": [_block(recall_format="free_recall", content=html)],
            "recall_self_check_enabled": True,
        }
    )
    assert res.passed is True
    assert not [i for i in res.issues if i.severity == "warning"]


def test_on_warns_options_visible_radio():
    html = (
        '<div class="self-check"><p>Q</p>'
        '<input type="radio" name="q1">A</label>'
        '<details><summary>Show answer</summary>A</details></div>'
    )
    res = RecallSelfCheckFormatValidator().validate(
        {
            "blocks": [_block(recall_format="cloze", content=html)],
            "recall_self_check_enabled": True,
        }
    )
    codes = {i.code for i in res.issues}
    assert "RECALL_SELF_CHECK_OPTIONS_VISIBLE" in codes
    assert res.passed is True  # warning-day-1, never blocks
    assert res.action == "regenerate"


def test_on_warns_answer_inline_no_details():
    html = '<div class="self-check"><p class="sc-recall-prompt">Q</p><input type="text"></div>'
    res = RecallSelfCheckFormatValidator().validate(
        {
            "blocks": [_block(recall_format="free_recall", content=html)],
            "recall_self_check_enabled": True,
        }
    )
    assert "RECALL_SELF_CHECK_ANSWER_INLINE" in {i.code for i in res.issues}


def test_on_outline_dict_options_visible():
    res = RecallSelfCheckFormatValidator().validate(
        {
            "blocks": [
                _block(
                    recall_format="cloze",
                    content={"question": "Q", "options": [{"text": "A"}]},
                )
            ],
            "recall_self_check_enabled": True,
        }
    )
    assert "RECALL_SELF_CHECK_OPTIONS_VISIBLE" in {i.code for i in res.issues}


def test_non_recall_self_check_ignored():
    html = '<div class="self-check"><input type="radio" name="q1"></div>'
    res = RecallSelfCheckFormatValidator().validate(
        {
            # no recall_format -> recognition self-check, ignored
            "blocks": [_block(content=html)],
            "recall_self_check_enabled": True,
        }
    )
    assert not [i for i in res.issues if i.severity == "warning"]
    assert res.metadata["recall_self_check_format"]["recall_blocks"] == 0
