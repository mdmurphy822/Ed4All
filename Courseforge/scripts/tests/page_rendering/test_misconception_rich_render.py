"""misconception_rich — render scaffold + JSON-LD emit + hash-exclusion tests."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import Block  # noqa: E402
from generate_course import (  # noqa: E402
    _build_misconception_blocks,
    _render_misconception_productive_failure,
)


def _rich_block():
    return Block(
        block_id="p#misc_0",
        block_type="misconception",
        page_id="p",
        sequence=0,
        content={"misconception": "Wrong", "correction": "Right"},
        mc_named_concept="like_terms",
        mc_predict_prompt="Predict what happens.",
        mc_reconcile="Why it fails.",
    )


def test_render_off_returns_empty(monkeypatch):
    monkeypatch.delenv("ED4ALL_MISCONCEPTION_RICH", raising=False)
    assert _render_misconception_productive_failure(_rich_block()) == ""


def test_render_on_emits_predict_reconcile_scaffold(monkeypatch):
    monkeypatch.setenv("ED4ALL_MISCONCEPTION_RICH", "1")
    html = _render_misconception_productive_failure(_rich_block())
    assert "misconception-productive-failure" in html
    assert "mc-named-concept" in html
    assert "mc-predict" in html
    assert "mc-reconcile" in html
    assert "like terms" in html


def test_jsonld_off_byte_identical(monkeypatch):
    monkeypatch.delenv("ED4ALL_MISCONCEPTION_RICH", raising=False)
    entry = _rich_block().to_jsonld_entry()
    assert "conceptId" not in entry
    assert "productiveFailure" not in entry


def test_jsonld_on_emits_concept_and_productive_failure(monkeypatch):
    monkeypatch.setenv("ED4ALL_MISCONCEPTION_RICH", "1")
    entry = _rich_block().to_jsonld_entry()
    assert entry["conceptId"] == "like_terms"
    assert entry["productiveFailure"]["predict"]
    assert entry["productiveFailure"]["reconcile"]


def test_build_misconception_blocks_threads_fields():
    blocks = _build_misconception_blocks(
        [
            {
                "misconception": "Wrong",
                "correction": "Right",
                "mc_named_concept": "like_terms",
                "mc_predict_prompt": "Predict.",
                "mc_reconcile": "Reconcile.",
            }
        ],
        page_id="p",
    )
    assert blocks[0].mc_named_concept == "like_terms"
    assert blocks[0].mc_predict_prompt == "Predict."
    assert blocks[0].mc_reconcile == "Reconcile."


def test_block_mc_fields_excluded_from_content_hash():
    base = Block(
        block_id="p#misc_0",
        block_type="misconception",
        page_id="p",
        sequence=0,
        content={"misconception": "Wrong", "correction": "Right"},
    )
    assert base.compute_content_hash() == _rich_block().compute_content_hash()
