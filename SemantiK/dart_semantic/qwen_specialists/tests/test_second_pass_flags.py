"""Phase-0 acceptance — SEMANTIK_SECOND_PASS + SEMANTIK_SECOND_PASS_ROUNDS.

The two resolvers are DEAD-BUT-IMPORTABLE scaffolding for the Pass-2
verify-refine loop: nothing in the cascade calls them yet, so the flag-off path
is byte-identical. These tests pin the parse-with-fallback contract (copied
verbatim from ``resolve_structure_review_mode`` / ``resolve_block_review_window``)
and prove ``run_structure_review`` + ``assemble_document`` are byte-identical
with both flags absent vs explicitly off.
"""

from __future__ import annotations

import json

from dart_semantic.assembler.api import assemble_document, AssemblerConfig
from dart_semantic.qwen_specialists.reviewer import (
    resolve_second_pass_mode,
    resolve_second_pass_rounds,
    run_structure_review,
)
from dart_semantic.soft_reranker.types import RankedCandidate
from dart_semantic.qwen_specialists.types import Candidate
from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Master gate — SEMANTIK_SECOND_PASS (default OFF).
# ---------------------------------------------------------------------------


def test_second_pass_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SECOND_PASS", raising=False)
    assert resolve_second_pass_mode() is False
    for v in ("", "  ", "0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("SEMANTIK_SECOND_PASS", v)
        assert resolve_second_pass_mode() is False


def test_second_pass_mode_truthy_tokens(monkeypatch):
    for v in ("1", "true", "YES", "on", "  On  "):
        monkeypatch.setenv("SEMANTIK_SECOND_PASS", v)
        assert resolve_second_pass_mode() is True


# ---------------------------------------------------------------------------
# Round bound — SEMANTIK_SECOND_PASS_ROUNDS (int parse-with-fallback, def 2).
# ---------------------------------------------------------------------------


def test_second_pass_rounds_default_and_fallback(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SECOND_PASS_ROUNDS", raising=False)
    assert resolve_second_pass_rounds() == 2
    for bad in ("", "  ", "garbage", "0", "-1", "1.5"):
        monkeypatch.setenv("SEMANTIK_SECOND_PASS_ROUNDS", bad)
        assert resolve_second_pass_rounds() == 2
    monkeypatch.setenv("SEMANTIK_SECOND_PASS_ROUNDS", "3")
    assert resolve_second_pass_rounds() == 3


# ---------------------------------------------------------------------------
# Flag-off byte-stability — the resolvers are unconsumed, so run_structure_review
# + assemble_document are byte-identical with both flags absent vs off.
# ---------------------------------------------------------------------------


def _fb(text: str) -> FeatureBlock:
    raw = RawBlock(text=text, page=1, bbox=(0.0, 0.0, 10.0, 10.0),
                   page_width=100.0, page_height=100.0)
    return FeatureBlock(raw=raw, size_bucket="md", gap_above=None,
                        is_top_of_page=False, is_centered=False, caps=None,
                        indent_bucket=0, relative_font_ratio=1.0)


class _StubRuntime:
    """Minimal runtime — no heading regions in the fixture, so generate_batch
    is never called; returns an empty list if it ever is."""

    def generate_batch(self, prompts, **_kw):
        return [None for _ in prompts]


def _stage6(text: str) -> RankedCandidate:
    return RankedCandidate(candidate=Candidate(adapter="prose", request_id="r", text=text), score=1.0)


def _build():
    fbs = [_fb("Body para one."), _fb("Body para two.")]
    regions = [
        Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "Body para one."}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "Body para two."}),
    ]
    top = {0: _stage6("<p>Body para one.</p>"), 1: _stage6("<p>Body para two.</p>")}
    return fbs, regions, top


def _run_once():
    fbs, regions, top = _build()
    corrected, verdicts = run_structure_review(regions, fbs, _StubRuntime())
    doc = assemble_document(top, corrected, fbs, config=AssemblerConfig(skip_gap_fill=True))
    verdict_repr = [
        (v.block_id, v.verdict, v.kind_before, v.kind_after) for v in verdicts
    ]
    return [r.kind for r in corrected], verdict_repr, doc.html


def test_flag_off_byte_stable(monkeypatch):
    # Flags ABSENT.
    monkeypatch.delenv("SEMANTIK_SECOND_PASS", raising=False)
    monkeypatch.delenv("SEMANTIK_SECOND_PASS_ROUNDS", raising=False)
    kinds_a, verdicts_a, html_a = _run_once()

    # Flags explicitly OFF + a garbage round value.
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "off")
    monkeypatch.setenv("SEMANTIK_SECOND_PASS_ROUNDS", "9")
    kinds_b, verdicts_b, html_b = _run_once()

    assert kinds_a == kinds_b
    assert verdicts_a == verdicts_b
    assert html_a == html_b
    # And the resolvers do not crash on the round value (parse-with-fallback).
    assert json.dumps(kinds_a) == json.dumps(kinds_b)
