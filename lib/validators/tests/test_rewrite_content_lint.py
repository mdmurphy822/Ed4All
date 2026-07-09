"""RewriteContentLintValidator unit tests.

Pins the deterministic rewrite-tier content lint against the three leak
categories that grounding / NLI / numeric / symbolic-math gates are blind to
(hand-found in 7B-authored vendor-HTML builds):

* pseudo-markup — escaped ``&lt;solution&gt;`` / namespaced ``<course:concept>``
  / custom-element ``<associative-property>`` angle-tags + ``\\text{}`` slug
  leftovers;
* publisher apparatus — generic numbered cross-refs (``Try It 3.2`` /
  ``Example 3.2`` / ``(see Figure 3`` / ``see Section 3.2``);
* slug-glue — a doubled term (``the discriminant, discriminant``) and a
  definition sentence starting with a bare lowercase slug (``decimal provides
  …``).

Every true-positive category MUST flag, clean prose MUST pass (including the
legitimate ``Example 3.`` worked-example label and an escaped ``&lt;p&gt;`` in a
code sample), and every issue MUST carry the block id in ``location`` so
``courseforge-rewrite --block-ids`` can consume the failure list.

No course slugs anywhere — the fixtures are synthetic HTML blocks built as plain
dict rows (the validator hydrates from Block instances OR JSONL dicts).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from lib.validators.rewrite_content_lint import (
    RewriteContentLintValidator,
    _CODE_APPARATUS,
    _CODE_PSEUDO_MARKUP,
    _CODE_SLUG_GLUE,
    _DECISION_TYPE,
)


# --------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------- #


def _block(*, block_id: str, block_type: str, content: Any) -> Dict[str, Any]:
    """A rewrite-tier block dict row (validator accepts dict OR Block)."""
    return {"block_id": block_id, "block_type": block_type, "content": content}


class _RecordingCapture:
    """Minimal DecisionCapture double — records log_decision calls."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(
        self, *, decision_type: str, decision: str, rationale: str, **_kw: Any
    ) -> None:
        self.calls.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
            }
        )


def _run(
    blocks: List[Dict[str, Any]],
    *,
    capture: Optional[_RecordingCapture] = None,
    shadow: bool = False,
) -> Any:
    inputs: Dict[str, Any] = {"blocks": blocks, "shadow": shadow}
    if capture is not None:
        inputs["decision_capture"] = capture
    return RewriteContentLintValidator().validate(inputs)


def _codes(result: Any) -> List[str]:
    return [i.code for i in result.issues]


# --------------------------------------------------------------------- #
# Pattern (1) — pseudo-markup leaks
# --------------------------------------------------------------------- #


def test_escaped_single_word_pseudo_tag_flagged():
    block = _block(
        block_id="week_01_content_01#concept_intro_00",
        block_type="concept",
        content="<p>The &lt;solution&gt; step yields the answer.</p>",
    )
    result = _run([block])
    assert _CODE_PSEUDO_MARKUP in _codes(result)


def test_namespaced_curie_pseudo_tag_flagged():
    block = _block(
        block_id="b1",
        block_type="concept",
        content="<p>Apply the <course:associative_property> here.</p>",
    )
    result = _run([block])
    assert _CODE_PSEUDO_MARKUP in _codes(result)


def test_custom_element_slug_tag_flagged():
    block = _block(
        block_id="b1",
        block_type="explanation",
        content="<p>The <associative-property> governs regrouping.</p>",
    )
    result = _run([block])
    assert _CODE_PSEUDO_MARKUP in _codes(result)


def test_text_brace_underscore_slug_leftover_flagged():
    block = _block(
        block_id="b1",
        block_type="concept",
        content=r"<p>Recall \(\text{associative_property}\) rules.</p>",
    )
    result = _run([block])
    assert _CODE_PSEUDO_MARKUP in _codes(result)


def test_escaped_real_html_tag_not_flagged():
    # A programming course legitimately shows an escaped <p> in a code sample;
    # the universal HTML-tag denylist keeps this out of the pseudo-markup arm.
    block = _block(
        block_id="b1",
        block_type="concept",
        content="<pre><code>&lt;p&gt;Hello&lt;/p&gt;</code></pre>",
    )
    result = _run([block])
    assert _CODE_PSEUDO_MARKUP not in _codes(result)


def test_text_brace_plain_word_not_flagged():
    # \text{} carrying a plain / hyphenated word is not a slug signal.
    block = _block(
        block_id="b1",
        block_type="concept",
        content=r"<p>Here \(x = 3\text{ well-known}\) holds.</p>",
    )
    result = _run([block])
    assert _CODE_PSEUDO_MARKUP not in _codes(result)


# --------------------------------------------------------------------- #
# Pattern (2) — publisher apparatus leaks
# --------------------------------------------------------------------- #


def test_try_it_apparatus_flagged():
    block = _block(
        block_id="b1",
        block_type="explanation",
        content="<p>Now attempt Try It 3.2 to practice.</p>",
    )
    result = _run([block])
    assert _CODE_APPARATUS in _codes(result)


def test_dotted_example_cross_ref_flagged():
    block = _block(
        block_id="b1",
        block_type="explanation",
        content="<p>As shown in Example 3.2, the rule applies.</p>",
    )
    result = _run([block])
    assert _CODE_APPARATUS in _codes(result)


def test_see_figure_paren_flagged():
    block = _block(
        block_id="b1",
        block_type="concept",
        content="<p>The curve rises (see Figure 3 for detail).</p>",
    )
    result = _run([block])
    assert _CODE_APPARATUS in _codes(result)


def test_see_section_cross_ref_flagged():
    block = _block(
        block_id="b1",
        block_type="explanation",
        content="<p>Review the prerequisite in see Section 3.2 first.</p>",
    )
    result = _run([block])
    assert _CODE_APPARATUS in _codes(result)


def test_legitimate_example_label_not_flagged():
    # The contract emits "Example N." (single number) as a worked-example label;
    # only the DOTTED "Example N.M" cross-ref shape is apparatus.
    block = _block(
        block_id="b1",
        block_type="example",
        content=(
            '<div class="example-box"><strong>Example 3.</strong> '
            "Solve for x.</div>"
        ),
    )
    result = _run([block])
    assert _CODE_APPARATUS not in _codes(result)


# --------------------------------------------------------------------- #
# Pattern (3) — slug-glue leaks
# --------------------------------------------------------------------- #


def test_doubled_term_flagged():
    block = _block(
        block_id="b1",
        block_type="concept",
        content="<p>We compute the discriminant, discriminant of the equation.</p>",
    )
    result = _run([block])
    assert _CODE_SLUG_GLUE in _codes(result)


def test_bare_slug_definition_sentence_flagged():
    block = _block(
        block_id="b1",
        block_type="concept",
        content="<p>decimal provides a way to represent fractional values.</p>",
    )
    result = _run([block])
    assert _CODE_SLUG_GLUE in _codes(result)


def test_capitalized_definition_sentence_not_flagged():
    # A properly-authored definition begins capitalized with an article.
    block = _block(
        block_id="b1",
        block_type="concept",
        content="<p>A decimal provides a way to represent fractional values.</p>",
    )
    result = _run([block])
    assert _CODE_SLUG_GLUE not in _codes(result)


# --------------------------------------------------------------------- #
# Clean prose + severity + block-id threading + capture
# --------------------------------------------------------------------- #


def test_clean_prose_passes_no_issues():
    block = _block(
        block_id="b1",
        block_type="concept",
        content=(
            "<section><h2>Adding Fractions</h2>"
            "<p>To add fractions, find a common denominator, then add the "
            "numerators. The result is a single fraction.</p></section>"
        ),
    )
    result = _run([block])
    assert result.issues == []
    assert result.passed is True
    assert result.score == 1.0


def test_issues_are_warning_severity_and_gate_passes_day1():
    block = _block(
        block_id="b1",
        block_type="concept",
        content="<p>The &lt;decimal&gt; value is shown.</p>",
    )
    result = _run([block])
    assert result.issues, "expected at least one flagged issue"
    assert all(i.severity == "warning" for i in result.issues)
    # Warning day-1: no critical issues, so the gate passes (computes+captures).
    assert result.passed is True


def test_issue_location_carries_block_id_for_block_ids_filter():
    block_id = "week_04_content_02#concept_discriminant_03"
    block = _block(
        block_id=block_id,
        block_type="concept",
        content="<p>Try It 4.1 next.</p>",
    )
    result = _run([block])
    assert result.issues
    assert all(i.location == block_id for i in result.issues)
    # The suggestion threads the block id into a runnable rewrite command.
    assert any(block_id in (i.suggestion or "") for i in result.issues)


def test_dict_content_outline_tier_skipped_silently():
    # Outline-tier blocks carry dict content; the post-rewrite lint skips them.
    block = _block(
        block_id="b1",
        block_type="concept",
        content={"curies": [], "key_claims": []},
    )
    result = _run([block])
    assert result.issues == []
    assert result.metadata["scored_blocks"] == 0


def test_decision_capture_fires_per_scored_block():
    capture = _RecordingCapture()
    blocks = [
        _block(block_id="b1", block_type="concept",
               content="<p>Clean prose about fractions.</p>"),
        _block(block_id="b2", block_type="concept",
               content="<p>The &lt;solution&gt; leaks.</p>"),
    ]
    result = _run(blocks, capture=capture)
    assert len(capture.calls) == 2
    assert all(c["decision_type"] == _DECISION_TYPE for c in capture.calls)
    decisions = {c["decision"] for c in capture.calls}
    assert "passed" in decisions
    assert any(d.startswith("failed") for d in decisions)
    assert all(len(c["rationale"]) >= 20 for c in capture.calls)
    # b2 leaked; it is countable for the --block-ids re-roll.
    assert result.metadata["flagged_blocks"] == 1


def test_blocks_final_path_jsonl_hydration(tmp_path):
    import json

    jsonl = tmp_path / "blocks_final.jsonl"
    rows = [
        {"block_id": "b1", "block_type": "concept",
         "content": "<p>Clean prose here.</p>"},
        {"block_id": "b2", "block_type": "concept",
         "content": "<p>Leaked <course:concept> token.</p>"},
    ]
    with jsonl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    result = RewriteContentLintValidator().validate(
        {"blocks_final_path": str(jsonl)}
    )
    assert _CODE_PSEUDO_MARKUP in _codes(result)
    assert all(i.location == "b2" for i in result.issues)


def test_missing_blocks_input_is_critical():
    result = RewriteContentLintValidator().validate({})
    assert result.passed is False
    assert any(i.code == "MISSING_BLOCKS_INPUT" for i in result.issues)


def test_validator_loadable_via_gate_manager():
    from MCP.hardening.validation_gates import ValidationGateManager

    mgr = ValidationGateManager()
    validator = mgr.load_validator(
        "lib.validators.rewrite_content_lint.RewriteContentLintValidator"
    )
    result = validator.validate(
        {"blocks": [_block(block_id="b1", block_type="concept",
                           content="<p>Clean.</p>")]}
    )
    assert result.passed is True
