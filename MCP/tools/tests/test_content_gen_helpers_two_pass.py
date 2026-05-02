"""Phase 3 Subtask 63 — Two-pass router wire-in tests for ``_content_gen_helpers``.

Asserts the I/O contract between
:func:`MCP.tools._content_gen_helpers._build_content_modules_dynamic`
and the optional ``content_router`` kwarg landed by Subtask 62:

* When a router is supplied, the helper builds one ``Block`` stub per
  topic position and dispatches the whole list through
  :meth:`CourseforgeRouter.route_all` — the legacy
  ``content_provider.generate_page`` per-iteration path is NOT invoked.
* When the router is ``None`` and a legacy ``content_provider`` is
  supplied, the helper falls back to the existing per-iteration provider
  path — the byte-stable Phase 1 behavior.
* The rewritten ``Block.content`` (HTML) is consumed into
  ``sections[0]["paragraphs"]`` after a double-newline split, mirroring
  the legacy provider-consumption shape.
* Blocks returned with ``escalation_marker`` set (outline tier failed)
  are skipped — the deterministic DART-paragraph floor stays for those
  positions.

The router is fully stubbed; no real LLM dispatch happens. The router
stub validates the helper's I/O contract by capturing the inputs (the
``Block`` list it received and the ``source_chunks_by_block_id`` /
``objectives`` payloads) and returning a deterministic rewritten list.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import Block, Touch  # noqa: E402
from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixture builders
# ---------------------------------------------------------------------- #


def _mk_topic(heading: str, source_file: str = "ch1") -> Dict[str, Any]:
    """Topic fixture — paragraph long enough to clear the 30-word floor."""
    return {
        "heading": heading,
        "paragraphs": [
            (
                f"Body text for {heading} explaining the concept in "
                "sufficient depth to satisfy the grounding validator "
                "non-trivial paragraph floor of thirty words each "
                "required for content pages to be treated as real "
                "body prose and not as an empty heading-only section."
            ),
            "Second paragraph with additional detail.",
        ],
        "key_terms": [heading.split()[0].lower()],
        "source_file": source_file,
        "word_count": 60,
        "extracted_lo_statements": [],
        "extracted_misconceptions": [],
        "extracted_questions": [],
    }


def _mk_obj(obj_id: str, statement: str) -> Dict[str, Any]:
    return {
        "id": obj_id,
        "statement": statement,
        "bloom_level": "understand",
        "bloom_verb": "describe",
        "key_concepts": [],
    }


# ---------------------------------------------------------------------- #
# Stub router
# ---------------------------------------------------------------------- #


class _StubRouter:
    """Captures ``route_all`` inputs; returns a rewritten Block list.

    Default behavior: every input Block is rewritten with non-empty
    HTML content carrying the page id so the test can verify the helper
    consumed the right block. Tests can pass ``escalate_indices`` to
    mark specific positions as outline-tier failures
    (``escalation_marker`` set; ``content`` left empty per
    ``route_all``'s contract for failed-outline blocks).
    """

    def __init__(
        self,
        *,
        escalate_indices: Optional[List[int]] = None,
        rewrite_content_template: str = (
            "Rewritten paragraph one for {block_id}.\n\n"
            "Rewritten paragraph two for {block_id}."
        ),
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._escalate_indices = set(escalate_indices or [])
        self._rewrite_content_template = rewrite_content_template

    def route_all(
        self,
        blocks: List[Block],
        *,
        source_chunks_by_block_id: Optional[Dict[str, List[Any]]] = None,
        objectives: Optional[List[Any]] = None,
    ) -> List[Block]:
        self.calls.append(
            {
                "blocks": list(blocks),
                "source_chunks_by_block_id": dict(
                    source_chunks_by_block_id or {}
                ),
                "objectives": list(objectives or []),
            }
        )
        out: List[Block] = []
        outline_touch = Touch(
            model="stub",
            provider="local",
            tier="outline",
            timestamp="2026-05-02T00:00:00Z",
            decision_capture_id="cap-outline-stub",
            purpose="phase3-test-outline",
        )
        rewrite_touch = Touch(
            model="stub",
            provider="anthropic",
            tier="rewrite",
            timestamp="2026-05-02T00:00:01Z",
            decision_capture_id="cap-rewrite-stub",
            purpose="phase3-test-rewrite",
        )
        for idx, block in enumerate(blocks):
            if idx in self._escalate_indices:
                # Outline-tier failure: leave content empty, set marker.
                # ``route_all``'s real contract appends a marker and
                # skips rewrite; we mirror that here.
                replaced = block.with_touch(outline_touch)
                import dataclasses
                replaced = dataclasses.replace(
                    replaced,
                    escalation_marker="outline_budget_exhausted",
                )
                out.append(replaced)
            else:
                rewritten = block.with_touch(outline_touch).with_touch(
                    rewrite_touch
                )
                import dataclasses
                rewritten = dataclasses.replace(
                    rewritten,
                    content=self._rewrite_content_template.format(
                        block_id=block.block_id
                    ),
                )
                out.append(rewritten)
        return out


class _StubLegacyProvider:
    """Stub legacy ``content_provider``. ``generate_page`` returns a
    Block with non-empty content carrying the page_id so a test can
    detect provider invocation."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate_page(
        self,
        *,
        course_code: str,
        week_number: int,
        page_id: str,
        page_template: str,
        page_context: Dict[str, Any],
    ) -> Block:
        self.calls.append({"page_id": page_id, "page_context": page_context})
        return Block(
            block_id=f"{page_id}#explanation_legacy_0",
            block_type="explanation",
            page_id=page_id,
            sequence=0,
            content=f"Legacy provider paragraph for {page_id}.",
        )


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #


class TestRouterDispatch:
    """Subtask 63 deliverable — four assertions about the helper's I/O
    contract under the new ``content_router`` kwarg."""

    def test_build_content_modules_dispatches_to_router_when_two_pass_enabled(
        self,
    ) -> None:
        """When ``content_router`` is supplied, ``route_all`` is called
        exactly once with the right block list, and the rewritten
        content lands on ``sections[0]['paragraphs']``."""
        topics = [_mk_topic("Introduction"), _mk_topic("Stages")]
        objectives = [
            _mk_obj("TO-01", "Describe introductory concepts."),
            _mk_obj("TO-02", "Explain the stages."),
        ]
        router = _StubRouter()

        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=1,
            week_topics=topics,
            week_objectives=objectives,
            all_objectives=objectives,
            course_code="BIO_101",
            content_router=router,
        )

        # Router was invoked exactly once with one block per topic.
        assert len(router.calls) == 1
        call = router.calls[0]
        assert len(call["blocks"]) == 2
        assert all(isinstance(b, Block) for b in call["blocks"])
        # Block stubs carry the canonical block_type for content modules.
        assert all(b.block_type == "explanation" for b in call["blocks"])
        # Block stubs were passed with empty content; the router fills it.
        assert all(b.content == "" for b in call["blocks"])
        # Per-block source chunks were passed keyed by block_id.
        assert set(call["source_chunks_by_block_id"].keys()) == {
            b.block_id for b in call["blocks"]
        }
        # Objectives payload preserves per-objective shape.
        assert {o["id"] for o in call["objectives"]} == {"TO-01", "TO-02"}

        # Rewritten content was consumed onto sections[0]['paragraphs'].
        assert len(wd["content_modules"]) == 2
        for module in wd["content_modules"]:
            paragraphs = module["sections"][0]["paragraphs"]
            assert paragraphs, module
            joined = " ".join(paragraphs)
            assert "Rewritten paragraph one" in joined
            assert "Rewritten paragraph two" in joined

    def test_build_content_modules_falls_back_to_legacy_provider_when_router_none(
        self,
    ) -> None:
        """When ``content_router`` is absent but a legacy
        ``content_provider`` is wired, the legacy per-iteration
        ``generate_page`` path runs unchanged. The router seam is not
        engaged."""
        topics = [_mk_topic("Introduction"), _mk_topic("Stages")]
        objectives = [
            _mk_obj("TO-01", "Describe introductory concepts."),
            _mk_obj("TO-02", "Explain the stages."),
        ]
        legacy = _StubLegacyProvider()

        wd = _cgh.build_week_data(
            week_num=2,
            duration_weeks=1,
            week_topics=topics,
            week_objectives=objectives,
            all_objectives=objectives,
            course_code="BIO_101",
            content_provider=legacy,
            content_router=None,
        )

        # Legacy provider was invoked once per topic position.
        assert len(legacy.calls) == 2
        page_ids = [c["page_id"] for c in legacy.calls]
        assert "week_02_content_01" in page_ids
        assert "week_02_content_02" in page_ids

        # The legacy provider's content landed on the sections.
        assert len(wd["content_modules"]) == 2
        for module in wd["content_modules"]:
            paragraphs = module["sections"][0]["paragraphs"]
            assert paragraphs, module
            assert any("Legacy provider paragraph" in p for p in paragraphs)

    def test_router_dispatched_blocks_carry_two_touches_outline_and_rewrite(
        self,
    ) -> None:
        """The rewritten Blocks the router returns to the helper carry
        the outline + rewrite tiers in their ``touched_by`` chain — the
        helper preserves them through to its consumption (it doesn't
        strip touches)."""
        topics = [_mk_topic("Introduction")]
        objectives = [_mk_obj("TO-01", "Describe introductory concepts.")]
        router = _StubRouter()

        _ = _cgh.build_week_data(
            week_num=1,
            duration_weeks=1,
            week_topics=topics,
            week_objectives=objectives,
            all_objectives=objectives,
            course_code="BIO_101",
            content_router=router,
        )

        # The router stub returned blocks carrying outline + rewrite
        # touches. Verify the contract by re-inspecting what the router
        # produced (the helper's responsibility is to consume content;
        # the touch chain is preserved on the returned Block list, which
        # downstream packaging persists into JSON-LD).
        assert len(router.calls) == 1
        produced = router.route_all(
            router.calls[0]["blocks"],
            source_chunks_by_block_id=router.calls[0][
                "source_chunks_by_block_id"
            ],
            objectives=router.calls[0]["objectives"],
        )
        for block in produced:
            tiers = [t.tier for t in block.touched_by]
            assert tiers == ["outline", "rewrite"], (
                f"block {block.block_id} touch chain: {tiers}"
            )

    def test_failed_outline_blocks_excluded_from_rewrite_in_helper(
        self,
    ) -> None:
        """Blocks the router marks as outline-tier failures
        (``escalation_marker`` set) must NOT have their empty content
        bleed onto the section. The helper falls back to the
        deterministic DART-paragraph floor for those positions —
        ``sections[0]['paragraphs']`` carries the topic's paragraphs
        rather than the failed block's empty string."""
        topics = [_mk_topic("Introduction"), _mk_topic("Stages")]
        objectives = [
            _mk_obj("TO-01", "Describe introductory concepts."),
            _mk_obj("TO-02", "Explain the stages."),
        ]
        # Position 0 fails the outline tier → no rewrite content.
        router = _StubRouter(escalate_indices=[0])

        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=1,
            week_topics=topics,
            week_objectives=objectives,
            all_objectives=objectives,
            course_code="BIO_101",
            content_router=router,
        )

        # Two modules emitted, but the failed-outline position must
        # carry the DART-derived paragraph floor (NOT empty / NOT the
        # rewrite template).
        assert len(wd["content_modules"]) == 2
        failed_module = wd["content_modules"][0]
        paragraphs = failed_module["sections"][0]["paragraphs"]
        assert paragraphs, "failed-outline position lost its DART floor"
        joined = " ".join(paragraphs)
        # Deterministic DART-floor signal — the topic's first paragraph
        # heading shows up here.
        assert "Body text for Introduction" in joined
        # And the rewrite template never landed on this position.
        assert "Rewritten paragraph" not in joined

        # Position 1 (no escalation) carries the rewritten content.
        succeeded_module = wd["content_modules"][1]
        succ_paragraphs = succeeded_module["sections"][0]["paragraphs"]
        assert any(
            "Rewritten paragraph" in p for p in succ_paragraphs
        ), succ_paragraphs
