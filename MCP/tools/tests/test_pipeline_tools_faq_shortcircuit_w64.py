"""W6.4 — FAQ deterministic grounded pass-through short-circuit.

Regression for the rewrite-tier short-circuit landed in
``MCP/tools/pipeline_tools.py``: a Block tagged
``template_type == lib.generation.faq_page.FAQ_TEMPLATE_TYPE`` carries a
pre-authored, grounded FAQ card, so the rewrite loop must PASS IT STRAIGHT
THROUGH ``_apply_str_backstops`` (no LLM dispatch) — exactly the way a
high-quality ``key_terms`` vocab card short-circuits — rather than re-author
the grounded answer through the LLM.

Pure-deterministic — no model, no GPU. Forces the per-block rewrite path
(``COURSEFORGE_REWRITE_BATCH=0``) so ``_process_one_block`` runs; the batched
cloud path mirrors the exact same predicate in its Phase-A prep.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import Block  # noqa: E402
from MCP.tools import pipeline_tools as _pt  # noqa: E402
from lib.generation.faq_page import FAQ_TEMPLATE_TYPE  # noqa: E402


# A unique token embedded in the grounded FAQ answer so we can prove the
# ORIGINAL content survived (was not overwritten by a would-be LLM rewrite).
_FAQ_ANSWER_TOKEN = "GROUNDED_FAQ_ANSWER_SURVIVES"
_FAQ_CONTENT = (
    '<div class="callout faq-card" data-cf-content-type="faq">'
    "<p><strong>Is it true that you can divide by zero?</strong></p>"
    f'<p>{_FAQ_ANSWER_TOKEN}: dividing by zero is undefined.</p>'
    "</div>"
)
# The sentinel a would-be LLM rewrite writes; if this shows up on the FAQ
# block, the short-circuit failed.
_LLM_SENTINEL = "LLM_REWROTE_THIS_BLOCK"


def _seed_project(tmp_path: Path, project_id: str) -> Path:
    exports_root = tmp_path / "Courseforge" / "exports"
    project_path = exports_root / project_id
    project_path.mkdir(parents=True, exist_ok=True)
    lo_dir = project_path / "01_learning_objectives"
    lo_dir.mkdir(exist_ok=True)
    (lo_dir / "synthesized_objectives.json").write_text(
        json.dumps({
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Describe core concept A in detail."}
            ],
            "chapter_objectives": [],
        }),
        encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({"course_name": project_id, "duration_weeks": 1}),
        encoding="utf-8",
    )
    return project_path


def _seed_outline_blocks(project_path: Path, blocks: List[Block]) -> Path:
    out_dir = project_path / "01_outline"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = out_dir / "blocks_outline.jsonl"
    with blocks_path.open("w", encoding="utf-8") as fh:
        for blk in blocks:
            fh.write(json.dumps(
                _pt._block_to_snake_case_entry(blk), ensure_ascii=False,
            ))
            fh.write("\n")
    return blocks_path


def _seed_outline_sidecars(
    project_path: Path, chunks_lookup: Dict[str, Any], objectives: List[Any],
) -> Dict[str, Path]:
    out_dir = project_path / "01_outline"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "outline_chunks.json"
    objectives_path = out_dir / "outline_objectives.json"
    chunks_path.write_text(json.dumps(chunks_lookup), encoding="utf-8")
    objectives_path.write_text(json.dumps(objectives), encoding="utf-8")
    return {"chunks": chunks_path, "objectives": objectives_path}


def _faq_block() -> Block:
    return Block(
        block_id="week_01_faq#vocab_card_faq_0",
        block_type="vocab_card",
        page_id="week_01_faq",
        sequence=0,
        content=_FAQ_CONTENT,
        template_type=FAQ_TEMPLATE_TYPE,
        objective_ids=("CO-01",),
        source_ids=("dart:test#c1",),
        target_bloom="understand",
    )


def _normal_block() -> Block:
    return Block(
        block_id="week_01_content_01#concept_a_0",
        block_type="concept",
        page_id="week_01_content_01",
        sequence=0,
        content={"key_claims": [{"claim": "Concept A is foundational."}]},
        objective_ids=("CO-01",),
    )


def test_faq_template_block_short_circuits_llm_and_survives(tmp_path, monkeypatch):
    """A FAQ-template block is NOT dispatched to the rewrite LLM and its
    grounded content survives verbatim; a sibling normal block IS dispatched."""
    project_id = "TEST_W64_FAQ_SHORTCIRCUIT"
    project_path = _seed_project(tmp_path, project_id)
    monkeypatch.setattr(_pt, "PROJECT_ROOT", tmp_path)
    # Force the deterministic per-block path (byte-stable predicate mirror in
    # the batched path).
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH", "0")

    faq = _faq_block()
    normal = _normal_block()
    blocks_path = _seed_outline_blocks(project_path, [faq, normal])
    sidecar_paths = _seed_outline_sidecars(
        project_path,
        {faq.block_id: [], normal.block_id: []},
        [{"id": "CO-01", "statement": "Explain concept A."}],
    )

    invoked: List[str] = []
    from Courseforge.router import router as _router_mod

    def fake_remediation(self, block, **kw):
        invoked.append(block.block_id)
        # A real LLM rewrite would overwrite the grounded answer with the
        # sentinel; the FAQ block must never reach here.
        return dataclasses.replace(
            block, content=f"<p>{_LLM_SENTINEL}</p>",
        )

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter,
        "route_rewrite_with_remediation",
        fake_remediation,
    )

    result = asyncio.run(_pt._run_content_generation_rewrite(
        project_id=project_id,
        blocks_validated_path=str(blocks_path),
        workflow_type="textbook_to_course",
        outline_chunks_path=str(sidecar_paths["chunks"]),
        outline_objectives_path=str(sidecar_paths["objectives"]),
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    # (1) The FAQ block was SHORT-CIRCUITED — the LLM rewrite entry point was
    #     never invoked for it; the normal block WAS routed through it.
    assert faq.block_id not in invoked, (
        f"FAQ-template block was sent to the rewrite LLM: invoked={invoked!r}"
    )
    assert normal.block_id in invoked, (
        f"Normal block should still route through the rewrite LLM: "
        f"invoked={invoked!r}"
    )

    # (2) The FAQ block's grounded content survived (str-backstop pass-through,
    #     no LLM sentinel); the normal block carries the LLM sentinel.
    blocks_final = Path(payload["blocks_final_path"])
    parsed = {
        json.loads(ln)["block_id"]: json.loads(ln)
        for ln in blocks_final.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    faq_entry = parsed[faq.block_id]
    assert _FAQ_ANSWER_TOKEN in faq_entry["content"], faq_entry
    assert _LLM_SENTINEL not in faq_entry["content"], faq_entry
    assert _LLM_SENTINEL in parsed[normal.block_id]["content"]
