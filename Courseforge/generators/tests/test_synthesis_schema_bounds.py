"""Anti-truncation output-schema bounds for the four synthesis surfaces.

Pins the correctness fix that bounds the four previously-UNBOUNDED
``TextbookSynthesisProvider`` dispatch methods — ``synthesize_concepts`` /
``synthesize_outline`` / ``synthesize_chapter_objectives`` /
``reconcile_terminal_objectives`` — so a dense chapter can no longer emit an
unbounded array that hits the ``max_tokens`` cap (``finish_reason='length'``)
and truncates. Each surface now rides the same constrained-decoding schema +
parse-side ``jsonschema`` validation as the already-fixed
``synthesize_window_objectives`` path.

Covered per method, hermetically (a recorder replaces ``_dispatch_call`` — NO
network, NO GPU):

* when the grammar mode resolves to a schema-supporting mode, ``_dispatch_call``
  receives an ``extra_payload`` carrying the bounded schema (``maxItems`` /
  ``maxLength`` present on the right fields);
* an over-budget (too many items) / over-long (over-``maxLength`` text) response
  is rejected by the parse-side validation and triggers the EXISTING retry, and
  a within-bounds response passes;
* the OFF path (grammar mode ``none``) dispatches byte-identically — no
  ``extra_payload`` schema on the wire.

Neutral placeholder content only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators.outline._textbook_synthesis_provider import (  # noqa: E402
    TextbookSynthesisProvider,
    TextbookSynthesisProviderError,
    _CHAPTER_OBJECTIVES_SCHEMA,
    _CONCEPTS_SCHEMA,
    _OUTLINE_SCHEMA,
    _RECONCILE_SCHEMA,
)


# ---------------------------------------------------------------------------
# Recorder — replaces the instance ``_dispatch_call`` (no network / GPU).
# Returns successive responses (last repeats) and records the ``extra_payload``
# the surface passed on every attempt.
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[Optional[Dict[str, Any]]] = []
        self.prompts: List[str] = []

    def __call__(
        self,
        user_prompt: str,
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        idx = min(len(self.calls), len(self._responses) - 1)
        self.calls.append(extra_payload)
        self.prompts.append(user_prompt)
        return (self._responses[idx] if self._responses else ""), 0


def _provider(
    responses: List[str], *, grammar_mode: Optional[str]
) -> Tuple[TextbookSynthesisProvider, _Recorder]:
    """A provider whose ``_dispatch_call`` is a recorder. Provider ``anthropic``
    keeps construction network-free; an EXPLICIT ``grammar_mode`` wins in
    ``_build_synthesis_grammar_payload`` regardless of provider, so the ON path
    is exercised deterministically."""
    # A dummy anthropic_client only satisfies the constructor's no-API-key
    # guard; the recorder replaces ``_dispatch_call`` so it is never used.
    p = TextbookSynthesisProvider(
        provider="anthropic",
        grammar_mode=grammar_mode,
        anthropic_client=object(),
    )
    rec = _Recorder(responses)
    p._dispatch_call = rec  # type: ignore[assignment,method-assign]
    return p, rec


# ---------------------------------------------------------------------------
# Neutral fixture inputs + valid / invalid payloads
# ---------------------------------------------------------------------------

_CHAPTER = {"id": "ch1", "chapter_text": "Neutral placeholder chapter prose."}
_STRUCTURE = {
    "chapters": [
        {"id": "ch1", "title": "Chapter One", "sections": [], "chapter_text": "x"}
    ]
}
_COURSE = "TST_914"


def _concepts_valid(n: int = 1) -> str:
    return json.dumps(
        {
            "concepts": [
                {
                    "canonical": f"term number {i}",
                    "aliases": ["alias a"],
                    "definition_hint": "a short neutral hint",
                    "chapter_ids": ["ch1"],
                }
                for i in range(n)
            ]
        }
    )


def _objective(i: int, *, sub: bool = False) -> Dict[str, Any]:
    obj: Dict[str, Any] = {
        "statement": f"Explain neutral idea number {i}.",
        "bloom_level": "understand",
        "bloom_verb": "explain",
        "source_refs": [],
    }
    if sub:
        obj["sub_objectives"] = ["a sub step"]
    return obj


def _outline_valid(n_tos: int = 1) -> str:
    return json.dumps(
        {
            "course_summary": "A neutral placeholder course summary.",
            "themes": [
                {
                    "title": "Theme One",
                    "summary": "neutral",
                    "chapter_ids": ["ch1"],
                    "prerequisite_theme_titles": [],
                }
            ],
            "draft_terminal_objectives": [_objective(i) for i in range(n_tos)],
        }
    )


def _chapter_valid(n: int = 1) -> str:
    return json.dumps(
        {"chapter_objectives": [_objective(i, sub=True) for i in range(n)]}
    )


def _reconcile_valid(n: int = 1) -> str:
    return json.dumps(
        {"terminal_objectives": [_objective(i) for i in range(n)]}
    )


def _call(p: TextbookSynthesisProvider, method: str) -> Any:
    if method == "concepts":
        return p.synthesize_concepts(_CHAPTER, course_name=_COURSE)
    if method == "outline":
        return p.synthesize_outline(_STRUCTURE, course_name=_COURSE)
    if method == "chapter":
        return p.synthesize_chapter_objectives(_CHAPTER, course_name=_COURSE)
    if method == "reconcile":
        return p.reconcile_terminal_objectives(
            [_objective(0)], [_objective(0)], course_name=_COURSE
        )
    raise AssertionError(method)


# Per-method: (method key, valid payload, over-COUNT payload, the top-level
# array key, its schema maxItems, the module schema object).
_CASES = [
    ("concepts", _concepts_valid(1), _concepts_valid(25), "concepts", 24, _CONCEPTS_SCHEMA),
    ("outline", _outline_valid(1), _outline_valid(25), "draft_terminal_objectives", 24, _OUTLINE_SCHEMA),
    ("chapter", _chapter_valid(1), _chapter_valid(17), "chapter_objectives", 16, _CHAPTER_OBJECTIVES_SCHEMA),
    ("reconcile", _reconcile_valid(1), _reconcile_valid(25), "terminal_objectives", 24, _RECONCILE_SCHEMA),
]


# ===========================================================================
# The module schemas are actually size-bounded (maxItems + maxLength present)
# ===========================================================================


@pytest.mark.parametrize("method,_v,_o,array_key,max_items,schema", _CASES)
def test_schema_declares_maxitems_on_primary_array(
    method, _v, _o, array_key, max_items, schema
):
    arr = schema["properties"][array_key]
    assert arr["maxItems"] == max_items


def test_free_text_fields_are_maxlength_bounded():
    # concepts definition_hint + objective statement are the dominant free-text.
    c_item = _CONCEPTS_SCHEMA["properties"]["concepts"]["items"]["properties"]
    assert c_item["definition_hint"]["maxLength"] > 0
    assert c_item["canonical"]["maxLength"] > 0
    o_item = _CHAPTER_OBJECTIVES_SCHEMA["properties"]["chapter_objectives"][
        "items"
    ]["properties"]
    assert o_item["statement"]["maxLength"] > 0
    assert o_item["sub_objectives"]["maxItems"] > 0
    # bloom_level is a bounded STRING, never an enum (normaliser lowercases).
    assert o_item["bloom_level"]["type"] == "string"
    assert "enum" not in o_item["bloom_level"]


# ===========================================================================
# ON path — the schema rides the wire as extra_payload
# ===========================================================================


@pytest.mark.parametrize("method,valid,_o,array_key,max_items,_s", _CASES)
def test_schema_supporting_mode_passes_bounded_schema(
    method, valid, _o, array_key, max_items, _s
):
    p, rec = _provider([valid], grammar_mode="json_schema")
    _call(p, method)
    assert rec.calls, "dispatch never fired"
    ep = rec.calls[0]
    assert ep is not None, "ON path must carry an extra_payload"
    # json_schema mode → {"format": <schema>}.
    schema = ep["format"]
    assert schema["properties"][array_key]["maxItems"] == max_items


def test_response_format_mode_carries_maxitems():
    p, rec = _provider(
        [_concepts_valid(1)], grammar_mode="response_format"
    )
    _call(p, "concepts")
    ep = rec.calls[0]
    schema = ep["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["concepts"]["maxItems"] == 24
    # And a nested free-text bound survived into the wire schema.
    item = schema["properties"]["concepts"]["items"]["properties"]
    assert item["definition_hint"]["maxLength"] > 0


# ===========================================================================
# Parse-side validation rejects an over-budget response and RETRIES
# ===========================================================================


@pytest.mark.parametrize("method,valid,over,_k,_m,_s", _CASES)
def test_over_count_rejected_then_retry_succeeds(
    method, valid, over, _k, _m, _s
):
    # attempt 0 → over-count (schema-rejected) → attempt 1 → valid (accepted).
    p, rec = _provider([over, valid], grammar_mode="json_schema")
    _call(p, method)
    assert len(rec.calls) == 2, "the parse-side rejection must trigger a retry"
    # The retry prompt carries the schema-hint remediation.
    assert "failed JSON Schema validation" in rec.prompts[1]


@pytest.mark.parametrize("method,valid,over,_k,_m,_s", _CASES)
def test_over_count_every_attempt_exhausts(method, valid, over, _k, _m, _s):
    p, rec = _provider([over], grammar_mode="json_schema")
    with pytest.raises(TextbookSynthesisProviderError):
        _call(p, method)
    # Exhausted the existing 3-attempt budget (never a new retry ladder).
    assert len(rec.calls) == 3


def test_over_length_text_rejected_then_retry_succeeds():
    # An over-``maxLength`` definition_hint (200 char cap) is rejected even
    # though the array count is within bounds — the length cap is the real
    # anti-truncation lever.
    long_hint = "x" * 400
    over = json.dumps(
        {"concepts": [{"canonical": "term one", "definition_hint": long_hint}]}
    )
    p, rec = _provider([over, _concepts_valid(1)], grammar_mode="json_schema")
    out = p.synthesize_concepts(_CHAPTER, course_name=_COURSE)
    assert len(rec.calls) == 2
    assert out["concepts"], "the within-bounds retry must succeed"


def test_over_length_statement_rejected():
    over = json.dumps(
        {
            "chapter_objectives": [
                {"statement": "y" * 800, "bloom_level": "understand"}
            ]
        }
    )
    p, rec = _provider([over], grammar_mode="json_schema")
    with pytest.raises(TextbookSynthesisProviderError):
        p.synthesize_chapter_objectives(_CHAPTER, course_name=_COURSE)
    assert len(rec.calls) == 3


# ===========================================================================
# OFF path — grammar mode ``none`` → byte-identical (no schema on the wire)
# ===========================================================================


@pytest.mark.parametrize("method,valid,_o,_k,_m,_s", _CASES)
def test_off_path_passes_no_extra_payload(method, valid, _o, _k, _m, _s):
    p, rec = _provider([valid], grammar_mode="none")
    _call(p, method)
    assert rec.calls, "dispatch never fired"
    # grammar mode ``none`` → empty payload → ``extra_payload or None`` == None
    # → byte-identical to the legacy bare dispatch.
    assert all(ep is None for ep in rec.calls)


def test_off_path_still_validates_and_bounds_parse_side():
    # Even OFF (schema-ignoring backend), the post-parse validation still
    # protects: an over-count response is rejected and retried (the backstop),
    # while the WIRE dispatch stayed byte-identical (no extra_payload).
    p, rec = _provider(
        [_concepts_valid(25), _concepts_valid(1)], grammar_mode="none"
    )
    p.synthesize_concepts(_CHAPTER, course_name=_COURSE)
    assert len(rec.calls) == 2
    assert all(ep is None for ep in rec.calls)


# ===========================================================================
# A within-bounds response still round-trips to the normalised shape
# ===========================================================================


def test_within_bounds_concepts_round_trip():
    p, _rec = _provider([_concepts_valid(3)], grammar_mode="json_schema")
    out = p.synthesize_concepts(_CHAPTER, course_name=_COURSE)
    assert len(out["concepts"]) == 3
    assert out["concepts"][0]["canonical"].startswith("term number")


def test_within_bounds_reconcile_round_trip():
    p, _rec = _provider([_reconcile_valid(2)], grammar_mode="json_schema")
    out = p.reconcile_terminal_objectives(
        [_objective(0)], [_objective(0)], course_name=_COURSE
    )
    assert len(out["terminal_objectives"]) == 2
    # IDs are re-minted TO-NN by the normaliser.
    assert out["terminal_objectives"][0]["id"].startswith("TO-")
