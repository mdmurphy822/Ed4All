"""Unit tests for :mod:`lib.utils.html_balance`.

Coverage contract (repair pass for the ``rewrite_html_shape`` gate):

1. Stray close tags (matching nothing open) are DROPPED.
2. Missing close tags are APPENDED at fragment end in LIFO order.
3. Mid-stack closes get missing intermediate closes INSERTED (LIFO).
4. Void elements are untouched (including the tolerated ``</br>`` form).
5. The trailing hidden ``data-cf-curie`` span tail is untouched and stays a
   top-level sibling (appended closes land BEFORE it).
6. Idempotency: ``repair(repair(x)) == repair(x)`` — always.
7. A fixture reproducing the real rewrite-tier block pattern (section >
   h3/p content with one stray close).
8. CRITICAL: repaired output PASSES the actual RewriteHtmlShapeValidator
   balance check (imported and called).
9. VOID_TAGS stays in lockstep with the gate's ``_VOID_TAGS``.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.utils.html_balance import (
    VOID_TAGS,
    repair_html_balance,
    repair_jsonl,
    resolve_rewrite_html_repair,
    summarize_ops,
)
from lib.validators.rewrite_html_shape import (
    _VOID_TAGS as _GATE_VOID_TAGS,
    _ShapeParser,
    RewriteHtmlShapeValidator,
)


def _gate_balanced(fragment: str) -> bool:
    """True iff the ACTUAL gate parser considers ``fragment`` balanced."""
    parser = _ShapeParser()
    parser.feed(fragment.lstrip())
    parser.close()
    parser.finalize()
    return not parser.unbalanced


def _assert_idempotent(fragment: str) -> None:
    once, _ = repair_html_balance(fragment)
    twice, ops2 = repair_html_balance(once)
    assert twice == once
    assert not any(
        o["op"] in ("drop_stray_close", "insert_close", "append_close")
        for o in ops2
    )


# ---------------------------------------------------------------------------
# Core repair behaviors
# ---------------------------------------------------------------------------

def test_balanced_fragment_untouched() -> None:
    src = '<section data-cf-block-id="b"><h3>T</h3><p>x <strong>y</strong></p></section>'
    out, ops = repair_html_balance(src)
    assert out == src
    assert ops == []


def test_stray_close_dropped() -> None:
    src = "<section><h3>t</h3><p>x</p></h3></section>"
    out, ops = repair_html_balance(src)
    assert out == "<section><h3>t</h3><p>x</p></section>"
    assert [o["op"] for o in ops] == ["drop_stray_close"]
    assert ops[0]["tag"] == "h3"
    assert ops[0]["raw"] == "</h3>"
    assert _gate_balanced(out)
    _assert_idempotent(src)


def test_stray_close_at_top_level_dropped() -> None:
    # Close against a completely empty stack.
    src = "</p><p>x</p>"
    out, ops = repair_html_balance(src)
    assert out == "<p>x</p>"
    assert [o["op"] for o in ops] == ["drop_stray_close"]
    assert _gate_balanced(out)
    _assert_idempotent(src)


def test_missing_close_appended() -> None:
    src = "<section><p>x"
    out, ops = repair_html_balance(src)
    assert out == "<section><p>x</p></section>"
    assert [(o["op"], o["tag"]) for o in ops] == [
        ("append_close", "p"),
        ("append_close", "section"),
    ]
    assert _gate_balanced(out)
    _assert_idempotent(src)


def test_nested_lifo_append_order() -> None:
    src = "<div><span><code>x"
    out, _ops = repair_html_balance(src)
    assert out == "<div><span><code>x</code></span></div>"
    assert _gate_balanced(out)
    _assert_idempotent(src)


def test_mid_stack_close_inserts_intermediates() -> None:
    # The gate marks pop-over-intermediates unbalanced even though the final
    # stack empties — the repair must insert the missing closes, not ignore.
    src = "<div><span>x</div>"
    assert not _gate_balanced(src)
    out, ops = repair_html_balance(src)
    assert out == "<div><span>x</span></div>"
    assert [(o["op"], o["tag"]) for o in ops] == [("insert_close", "span")]
    assert _gate_balanced(out)
    _assert_idempotent(src)


def test_mid_stack_multiple_intermediates_lifo() -> None:
    src = "<section><div><p>x</section>"
    out, _ops = repair_html_balance(src)
    assert out == "<section><div><p>x</p></div></section>"
    assert _gate_balanced(out)
    _assert_idempotent(src)


def test_void_elements_untouched() -> None:
    src = '<p>a<br>b<img src="x" alt="d">.<hr></p>'
    out, ops = repair_html_balance(src)
    assert out == src
    assert ops == []


def test_void_spurious_close_tolerated_and_untouched() -> None:
    # The gate ignores ``</br>`` (browser-tolerated); the repair must NOT
    # drop it.
    src = "<p>a<br></br>b</p>"
    out, ops = repair_html_balance(src)
    assert out == src
    assert ops == []
    assert _gate_balanced(out)


def test_self_closing_form_untouched() -> None:
    # ``<tag />`` is never pushed by the gate; the repair mirrors it.
    src = "<p>x<span /><em>y</em></p>"
    out, ops = repair_html_balance(src)
    assert out == src
    assert ops == []
    assert _gate_balanced(out)


def test_mathml_balanced_untouched_and_stray_dropped() -> None:
    ok = "<p><math><mrow><mn>1</mn><mo>+</mo><mn>2</mn></mrow></math></p>"
    out, ops = repair_html_balance(ok)
    assert out == ok and ops == []
    bad = "<p><math><mrow><mn>1</mn></mrow></mn></math></p>"
    out, ops = repair_html_balance(bad)
    assert out == ok.replace("<mo>+</mo><mn>2</mn>", "")
    assert [(o["op"], o["tag"]) for o in ops] == [("drop_stray_close", "mn")]
    assert _gate_balanced(out)
    _assert_idempotent(bad)


def test_text_and_attributes_never_edited() -> None:
    src = (
        '<section data-cf-block-id="B" class="x  y">'
        "<p>keep &amp; exact   spacing</p></h2>"
    )
    out, ops = repair_html_balance(src)
    # Only the stray </h2> is gone and </section> appended; everything else
    # byte-identical.
    assert out == (
        '<section data-cf-block-id="B" class="x  y">'
        "<p>keep &amp; exact   spacing</p></section>"
    )
    assert {o["op"] for o in ops} == {"drop_stray_close", "append_close"}


# ---------------------------------------------------------------------------
# Curie tail
# ---------------------------------------------------------------------------

def test_curie_tail_untouched_and_stays_top_level() -> None:
    tail = (
        '<span hidden data-cf-curie="algglm01:foo">algglm01:foo</span>'
        '<span hidden data-cf-curie="algglm01:bar" data-cf-curie-forced="true">'
        "algglm01:bar</span>"
    )
    src = f'<section data-cf-block-id="b"><p>x{tail}'
    out, ops = repair_html_balance(src)
    # Appended closes land BEFORE the tail — the curie spans stay top-level
    # siblings, byte-identical.
    assert out == f'<section data-cf-block-id="b"><p>x</p></section>{tail}'
    assert [(o["op"], o["tag"]) for o in ops] == [
        ("append_close", "p"),
        ("append_close", "section"),
    ]
    assert _gate_balanced(out)
    _assert_idempotent(src)


def test_balanced_fragment_with_curie_tail_untouched() -> None:
    src = (
        '<section data-cf-block-id="b"><p>x</p></section>'
        '<span hidden data-cf-curie="a:b">a:b</span>'
    )
    out, ops = repair_html_balance(src)
    assert out == src
    assert ops == []


def test_unclosed_curie_span_is_a_defect_not_a_tail() -> None:
    # An UNCLOSED trailing curie span is itself a missing-close defect;
    # closes append at the very end.
    src = '<p>x</p><span hidden data-cf-curie="a:b">a:b'
    out, _ops = repair_html_balance(src)
    assert out == '<p>x</p><span hidden data-cf-curie="a:b">a:b</span>'
    assert _gate_balanced(out)
    _assert_idempotent(src)


# ---------------------------------------------------------------------------
# Real rewrite-tier block pattern
# ---------------------------------------------------------------------------

REAL_PATTERN_FRAGMENT = (
    '<section data-cf-block-id="week_01_content_01#concept_place-value_0" '
    'data-cf-content-type="concept_explanation" data-cf-key-terms="place value">'
    "<h3>Place Value</h3>"
    "<p>Each digit in a whole number has a place value determined by its "
    "position.</p>"
    "<p>For example, in 352 the digit 3 occupies the hundreds place.</p></h3>"
    "</section>"
    '<span hidden data-cf-curie="algglm01:place_value">algglm01:place_value</span>'
)


def test_real_block_pattern_one_stray_close() -> None:
    assert not _gate_balanced(REAL_PATTERN_FRAGMENT)
    out, ops = repair_html_balance(REAL_PATTERN_FRAGMENT)
    assert [(o["op"], o["tag"]) for o in ops] == [("drop_stray_close", "h3")]
    assert out == REAL_PATTERN_FRAGMENT.replace("</p></h3>", "</p>")
    assert _gate_balanced(out)
    _assert_idempotent(REAL_PATTERN_FRAGMENT)


def test_malformed_attribute_quote_converges_or_original() -> None:
    # An unterminated attribute quote makes the stdlib parser swallow later
    # markup into the attr value; the repair must still reach a deterministic
    # fixed point (or return the original untouched — never a partial edit).
    src = '<section><div class="key-rule><strong>Key</strong> idea</div><p>x</p>'
    out, _ops = repair_html_balance(src)
    again, ops2 = repair_html_balance(out)
    assert again == out
    assert not any(
        o["op"] in ("drop_stray_close", "insert_close", "append_close")
        for o in ops2
    )


def test_idempotency_matrix() -> None:
    cases = [
        "",
        "   ",
        "plain text no tags",
        "<p>x</p>",
        "<div><span>x</div>",
        "</h3>",
        "<section><p>a</p>",
        REAL_PATTERN_FRAGMENT,
        '<p>a<br></br>b</p>',
        '<ul><li>a<li>b</ul>',
    ]
    for src in cases:
        _assert_idempotent(src)


# ---------------------------------------------------------------------------
# CRITICAL: repaired output passes the ACTUAL rewrite_html_shape validator
# ---------------------------------------------------------------------------

def _block(content: str, block_type: str = "concept", block_id: str = "p1#b1"):
    return SimpleNamespace(
        block_id=block_id,
        block_type=block_type,
        page_id="p1",
        sequence=0,
        content=content,
        escalation_marker=None,
    )


def test_repaired_output_passes_rewrite_html_shape_validator() -> None:
    broken = [
        # stray close (real pattern)
        REAL_PATTERN_FRAGMENT,
        # missing closes
        '<section data-cf-block-id="b2" data-cf-content-type="c" '
        'data-cf-key-terms="k"><h3>T</h3><p>x',
        # mid-stack close
        '<div data-cf-block-id="b3" data-cf-content-type="c" '
        'data-cf-key-terms="k"><p><span>x</p></div>',
    ]
    validator = RewriteHtmlShapeValidator()

    # Sanity: the broken set FAILS the gate on tag balance.
    before = validator.validate(
        {"blocks": [_block(c, block_id=f"p1#b{i}") for i, c in enumerate(broken)]}
    )
    assert not before.passed
    assert any(i.code == "REWRITE_HTML_PARSE_FAIL" for i in before.issues)

    repaired_blocks = []
    for i, content in enumerate(broken):
        fixed, ops = repair_html_balance(content)
        assert ops, f"case {i} expected repair ops"
        repaired_blocks.append(_block(fixed, block_id=f"p1#b{i}"))

    after = validator.validate({"blocks": repaired_blocks})
    parse_fails = [
        i for i in after.issues if i.code == "REWRITE_HTML_PARSE_FAIL"
    ]
    assert parse_fails == []
    assert after.passed, [
        (i.code, i.message) for i in after.issues if i.severity == "critical"
    ]
    assert after.score == 1.0


# ---------------------------------------------------------------------------
# Lockstep + helpers + runner
# ---------------------------------------------------------------------------

def test_void_tags_lockstep_with_gate() -> None:
    assert VOID_TAGS == _GATE_VOID_TAGS


def test_summarize_ops_deterministic() -> None:
    ops = [
        {"op": "append_close", "tag": "p"},
        {"op": "drop_stray_close", "tag": "h3"},
        {"op": "drop_stray_close", "tag": "p"},
    ]
    assert summarize_ops(ops) == "append_close=1;drop_stray_close=2"
    assert summarize_ops([]) == ""


def test_resolve_flag_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COURSEFORGE_REWRITE_HTML_REPAIR", raising=False)
    assert resolve_rewrite_html_repair() is False
    monkeypatch.setenv("COURSEFORGE_REWRITE_HTML_REPAIR", "0")
    assert resolve_rewrite_html_repair() is False
    monkeypatch.setenv("COURSEFORGE_REWRITE_HTML_REPAIR", "garbage")
    assert resolve_rewrite_html_repair() is False
    for token in ("1", "true", "YES", " on "):
        monkeypatch.setenv("COURSEFORGE_REWRITE_HTML_REPAIR", token)
        assert resolve_rewrite_html_repair() is True


def test_repair_jsonl_runner(tmp_path: Path) -> None:
    rows = [
        {
            "block_id": "p1#concept_x_0",
            "block_type": "concept",
            "page_id": "p1",
            "sequence": 0,
            "content": REAL_PATTERN_FRAGMENT,
            "touched_by": [],
        },
        {
            "block_id": "p1#callout_y_1",
            "block_type": "callout",
            "page_id": "p1",
            "sequence": 1,
            "content": '<div data-cf-block-id="p1#callout_y_1"><p>fine</p></div>',
        },
    ]
    src = tmp_path / "blocks_final.jsonl"
    src.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    out_dir = tmp_path / "out"
    report = repair_jsonl(src, out_dir)

    # Input untouched.
    assert src.read_text(encoding="utf-8") == "".join(
        json.dumps(r) + "\n" for r in rows
    )
    assert report["total_blocks"] == 2
    assert report["repaired"] == 1
    assert report["untouched"] == 1
    assert report["op_counts"] == {"drop_stray_close": 1}
    assert report["blocks"][0]["block_id"] == "p1#concept_x_0"

    repaired_rows = [
        json.loads(line)
        for line in (out_dir / "blocks_final.repaired.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    # Same schema, only content changed.
    assert repaired_rows[0].keys() == rows[0].keys()
    assert repaired_rows[1] == rows[1]
    assert repaired_rows[0]["content"] == REAL_PATTERN_FRAGMENT.replace(
        "</p></h3>", "</p>"
    )
    # Report sidecar exists + gate probe ran.
    report_on_disk = json.loads(
        (out_dir / "repair_report.json").read_text(encoding="utf-8")
    )
    assert report_on_disk["repaired"] == 1
    assert report_on_disk["gate"]["after"]["score"] == 1.0


# ---------------------------------------------------------------------------
# Pipeline wire-in (COURSEFORGE_REWRITE_HTML_REPAIR)
# ---------------------------------------------------------------------------

def test_pipeline_wirein_flag_off_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from MCP.tools.pipeline_tools import _maybe_repair_block_html_balance

    monkeypatch.delenv("COURSEFORGE_REWRITE_HTML_REPAIR", raising=False)
    blocks = [_block(REAL_PATTERN_FRAGMENT)]
    out, n, ops = _maybe_repair_block_html_balance(blocks)
    assert out is blocks
    assert n == 0 and ops == {}
    assert out[0].content == REAL_PATTERN_FRAGMENT


def test_pipeline_wirein_repairs_and_stamps_touch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Courseforge.scripts.blocks import Block
    from MCP.tools.pipeline_tools import _maybe_repair_block_html_balance

    monkeypatch.setenv("COURSEFORGE_REWRITE_HTML_REPAIR", "1")
    broken = Block(
        block_id="p1#concept_x_0",
        block_type="concept",
        page_id="p1",
        sequence=0,
        content=REAL_PATTERN_FRAGMENT,
    )
    healthy = Block(
        block_id="p1#callout_y_1",
        block_type="callout",
        page_id="p1",
        sequence=1,
        content='<div data-cf-block-id="p1#callout_y_1"><p>fine</p></div>',
    )
    out, n, op_counts = _maybe_repair_block_html_balance([broken, healthy])
    assert n == 1
    assert op_counts == {"drop_stray_close": 1}
    assert out[0].content == REAL_PATTERN_FRAGMENT.replace("</p></h3>", "</p>")
    # Repaired block records its ops on the touched_by chain.
    touch = out[0].touched_by[-1]
    assert touch.model == "html-balance-repair"
    assert touch.provider == "deterministic"
    assert touch.tier == "rewrite_val"
    assert touch.purpose == "html_balance_repair"
    assert "drop_stray_close=1" in touch.decision_capture_id
    assert "p1#concept_x_0" in touch.decision_capture_id
    # Healthy block untouched (same object, no touch appended).
    assert out[1] is healthy
