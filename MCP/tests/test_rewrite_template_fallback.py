"""7B/NVIDIA runaway-block recovery: deterministic styled-template fallback.

Unit tests for the standalone, LLM-free fallback renderer in
``MCP/tools/pipeline_tools.py``:

  * ``_render_block_fallback_html`` — maps a Block's ``block_type`` to the
    SAME styled component classes the successful rewrite path emits, drawing
    its content from the outline block's already-concise ``content.key_claims``.
  * ``_fallback_claim_texts`` — extracts the concise claim strings.

The fallback fires (in ``_run_content_generation_rewrite``'s page-emit loop)
when a block carries a non-null ``escalation_marker`` AND has no usable
rewritten HTML string content — i.e. the rewrite tier exhausted its
parse-retry budget because the model ran away (repetition truncated at
max_tokens). Rather than DROP the block, we render a concise styled grounded
block so the page stays styled + complete.

No LLM dispatch, no pipeline run.
"""

import re

from Courseforge.scripts.blocks import Block
import MCP.tools.pipeline_tools as pt


def _approx_token_count(html: str) -> int:
    """Rough whitespace/punct token count (math-safe, no model load)."""
    return len(re.findall(r"\S+", html))


def _make_block(block_type, key_claims, **kw):
    """Build an outline-shape Block with dict content carrying key_claims."""
    content = {"key_claims": [{"claim": c, "source_chunk_ids": ["ch_1"]}
                              for c in key_claims]}
    return Block(
        block_id=kw.pop("block_id", f"week_01_content_01#{block_type}_x_0"),
        block_type=block_type,
        page_id=kw.pop("page_id", "week_01_content_01"),
        sequence=kw.pop("sequence", 0),
        content=content,
        escalation_marker="validator_consensus_fail",
        **kw,
    )


# ---------------------------------------------------------------------------
# Test (1) — example block: worked-example / step-row / solution-line + markers
# ---------------------------------------------------------------------------

def test_example_fallback_renders_worked_example_with_steps():
    blk = _make_block(
        "example",
        [
            "Start by writing the fraction 6/8.",
            "Divide numerator and denominator by the GCF 2.",
            "The simplified fraction is 3/4.",
        ],
        source_ids=("dart:alg#blk_3",),
    )
    out = pt._render_block_fallback_html(
        blk,
        objective_statements={},
        minted_curie_map={},
        source_ids=["dart:alg#blk_3"],
    )
    assert 'class="example-box"' in out
    assert 'class="worked-example"' in out
    # 3 claims -> 2 step-rows + 1 solution-line (last claim is the solution).
    assert out.count('class="step-row"') == 2
    assert 'class="solution-line"' in out
    assert "3/4" in out  # solution body present
    # Marker + teaching role + audit fallback marker present.
    assert 'data-cf-teaching-role=' in out
    assert 'data-cf-fallback="template"' in out
    # SHORT by construction — well under 1200 tokens.
    assert _approx_token_count(out) < 1200


# ---------------------------------------------------------------------------
# Test (2) — explanation: definition claim -> definition-box; rule -> key-rule
# ---------------------------------------------------------------------------

def test_explanation_fallback_definition_box_and_key_rule():
    blk = _make_block(
        "explanation",
        [
            "A prime number is defined as a number divisible only by 1 and "
            "itself.",
            "The distributive property states a(b + c) equals ab + ac.",
            "Practice helps reinforce these ideas.",
        ],
    )
    out = pt._render_block_fallback_html(blk, objective_statements={})
    assert 'class="definition-box"' in out  # definition claim
    assert 'class="key-rule"' in out        # rule/property claim
    # The plain claim is a bare <p>.
    assert "<p>Practice helps reinforce these ideas.</p>" in out
    assert 'data-cf-content-type=' in out


# ---------------------------------------------------------------------------
# Test (3) — objective: <li> shows the STATEMENT, never the bare TO-NN
# ---------------------------------------------------------------------------

def test_objective_fallback_renders_statement_not_bare_id():
    blk = Block(
        block_id="week_01_objectives#objective_x_0",
        block_type="objective",
        page_id="week_01_objectives",
        sequence=0,
        content={"key_claims": []},
        objective_ids=("TO-01", "CO-03"),
        escalation_marker="validator_consensus_fail",
    )
    stmts = {
        "TO-01": "Simplify rational expressions to lowest terms.",
        "CO-03": "Identify the greatest common factor of two integers.",
    }
    out = pt._render_block_fallback_html(blk, objective_statements=stmts)
    assert "Simplify rational expressions to lowest terms." in out
    assert "Identify the greatest common factor of two integers." in out
    # The id stays only in the attribute, never as bare visible <li> body.
    assert 'data-cf-objective-id="TO-01"' in out
    assert not re.search(r">\s*TO-01\s*<", out)
    assert not re.search(r">\s*CO-03\s*<", out)


# ---------------------------------------------------------------------------
# Test (4) — no visible minted-CURIE / bare TO-NN tokens leak in any output
# ---------------------------------------------------------------------------

def test_no_visible_curie_or_lo_id_tokens_leak():
    minted = {
        "samplecoursea:integers": {"surface_forms": ["integers"]},
    }
    blk = _make_block(
        "concept",
        [
            "The set of samplecoursea:integers includes negatives.",
            "An integer is defined as a whole number or its negation.",
        ],
    )
    out = pt._render_block_fallback_html(
        blk, objective_statements={}, minted_curie_map=minted,
    )
    # The minted CURIE token must NOT appear in visible prose.
    assert "samplecoursea:integers" not in out
    assert "integers includes negatives" in out  # surface term substituted
    # No stray bare TO-NN / CO-NN token in visible body.
    assert not re.search(r"\bTO-\d{2}\b", out)
    assert not re.search(r"\bCO-\d{2}\b", out)


def test_self_check_fallback_question_card():
    blk = _make_block(
        "self_check_question",
        [
            "What is the simplest form of 6/8?",
            "3/4, found by dividing both by the GCF 2.",
        ],
    )
    out = pt._render_block_fallback_html(blk, objective_statements={})
    assert "self-check" in out and "self-check-item" in out
    assert "What is the simplest form of 6/8?" in out
    # The answer is behind a reveal, not inline next to the question text.
    assert "<details>" in out
    assert 'data-cf-fallback="template"' in out


# ---------------------------------------------------------------------------
# Test (5) — the fallback FIRES in the emit loop on a truncation-exhausted
# block and REPLACES the degraded content (integration of the wiring point).
# ---------------------------------------------------------------------------

def test_fallback_fires_on_escalated_block_in_emit_loop(tmp_path, monkeypatch):
    """Drive the real ``_run_content_generation_rewrite`` handler with one
    truncation-exhausted block and assert the emitted page HTML carries the
    styled fallback (``data-cf-fallback``) instead of dropping the block.

    Exercises the actual wiring point: the rewrite router is monkeypatched to
    return the input block with a non-null ``escalation_marker`` and its
    outline dict content untouched (no usable rewritten HTML) — the exact
    shape a model that "ran away" to max_tokens leaves behind.
    """
    import asyncio
    import dataclasses as _dc
    import json as _json
    from pathlib import Path as _P

    # The handler resolves the project root from courseforge_exports_dir().
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: exports_root)

    project_id = "PROJ-EX"
    project_root = exports_root / project_id
    (project_root / "01_learning_objectives").mkdir(parents=True)
    (project_root / "project_config.json").write_text(
        _json.dumps({"course_name": "EX_101"}), encoding="utf-8",
    )
    (project_root / "01_learning_objectives"
     / "synthesized_objectives.json").write_text(
        _json.dumps({
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Simplify fractions."},
            ],
            "chapter_objectives": [
                {"objectives": [
                    {"id": "CO-01", "statement": "Find the GCF."},
                ]},
            ],
        }),
        encoding="utf-8",
    )

    # The validated outline-tier blocks the rewrite phase consumes — one
    # example block with 3 concise key_claims (grounded content the fallback
    # renders from).
    blocks_validated = project_root / "blocks_validated.jsonl"
    blk_entry = {
        "block_id": "week_01_content_01#example_simplify_0",
        "block_type": "example",
        "page_id": "week_01_content_01",
        "sequence": 0,
        "content": {"key_claims": [
            {"claim": "Write the fraction 6/8.", "source_chunk_ids": ["c1"]},
            {"claim": "Divide both by the GCF 2.", "source_chunk_ids": ["c1"]},
            {"claim": "The answer is 3/4.", "source_chunk_ids": ["c1"]},
        ]},
        "objective_ids": ["CO-01"],
    }
    blocks_validated.write_text(
        _json.dumps(blk_entry) + "\n", encoding="utf-8",
    )

    # Monkeypatch the rewrite router so the block "runs away": the dispatch
    # returns the SAME block with an escalation_marker + its dict content
    # (no usable rewritten HTML string) — the truncation-exhausted shape the
    # emit loop must rescue rather than drop.
    def _fake_route_rewrite(self, blk, **kwargs):  # noqa: ANN001
        return _dc.replace(blk, escalation_marker="validator_consensus_fail")

    monkeypatch.setattr(
        "Courseforge.router.router.CourseforgeRouter."
        "route_rewrite_with_remediation",
        _fake_route_rewrite,
        raising=False,
    )

    result_json = asyncio.run(pt._run_content_generation_rewrite(
        project_id=project_id,
        blocks_validated_path=str(blocks_validated),
        libv2_root=str(tmp_path / "libv2"),
    ))
    result = _json.loads(result_json)
    assert result.get("success") is True, result
    page_paths = result.get("page_paths") or []
    assert page_paths, "expected at least one emitted page"

    html_text = _P(page_paths[0]).read_text(encoding="utf-8")
    # The degraded block was NOT dropped — the styled fallback shipped.
    assert 'data-cf-fallback="template"' in html_text
    assert 'class="worked-example"' in html_text
    assert "3/4" in html_text

    # And the on-disk blocks_final.jsonl preserved the escalation marker
    # (audit trail) — the fallback is an EMIT-time rescue, not a marker wipe.
    blocks_final = result.get("blocks_final_path")
    assert blocks_final and _P(blocks_final).exists()
    final_text = _P(blocks_final).read_text(encoding="utf-8")
    assert "validator_consensus_fail" in final_text
