"""Tests for ``TextbookSynthesisProvider`` (three-stage textbook synthesis).

Plan reference: ``plans/textbook-llm-synthesis-3stage-2026-05.md`` § 13
test-plan item 1.

Pins the in-process three-stage textbook-synthesis provider:

- All four public methods (``synthesize_outline`` / ``synthesize_concepts``
  / ``synthesize_chapter_objectives`` / ``reconcile_terminal_objectives``)
  return their canonical normalised shape against a mocked
  ``anthropic_client``.
- Lenient JSON parse recovers a markdown-fenced response.
- Registry-dynamic provider resolution (monkeypatch
  ``_OPENAI_COMPATIBLE_PROVIDERS``).
- ``TextbookSynthesisProviderError`` fires with the correct ``code`` per
  stage on exhausted parse.
- The Stage-1 chaptered split fires past ``_SKELETON_CHAR_BUDGET``.
- The per-chapter batching helper dispatches in groups of ≤10.
- All four decision-capture events fire with the runtime ``provider``.

Output SHAPE is asserted directly via mocks — these tests do NOT
validate against ``schemas/knowledge/*`` files (the other Wave-A worker
owns those schemas).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators import _textbook_synthesis_provider as tsp  # noqa: E402
from Courseforge.generators._textbook_synthesis_provider import (  # noqa: E402
    DEFAULT_PROVIDER,
    ENV_PROVIDER,
    ENV_TIMEOUT,
    TextbookSynthesisProvider,
    TextbookSynthesisProviderError,
    _build_supported_providers,
    _CHAPTER_BATCH_SIZE,
    _DEFAULT_TIMEOUT_SECONDS,
    _DRAFT_TO_BAND_COLLAPSE,
    _DRAFT_TO_BAND_DEFAULT,
    _LOCAL_NUM_CTX,
    _SKELETON_CHAR_BUDGET,
    _SKELETON_MAX_SECTION_TITLES,
)
from Trainforge.generators._openai_compatible_client import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS,
    SynthesisProviderError,
)


# ---------------------------------------------------------------------------
# Mock anthropic client — per-call response queue
# ---------------------------------------------------------------------------


class _FakeMessages:
    """Anthropic ``client.messages`` shim.

    Returns successive entries from ``responses`` per ``create()`` call,
    repeating the last entry once the queue drains. Records every call.
    """

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        text = self._responses[idx] if self._responses else ""
        return {"content": [{"type": "text", "text": text}]}


class _FakeAnthropicClient:
    """Anthropic SDK client shim wrapping a :class:`_FakeMessages`."""

    def __init__(self, responses: List[str]) -> None:
        self.messages = _FakeMessages(responses)


class _FakeCapture:
    """Minimal DecisionCapture shim — records every ``log_decision``."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
    ) -> None:
        self.events.append({
            "decision_type": decision_type,
            "decision": decision,
            "rationale": rationale,
        })


# ---------------------------------------------------------------------------
# Payload fixtures
# ---------------------------------------------------------------------------


def _outline_payload() -> dict:
    return {
        "course_summary": "A first course in elementary algebra.",
        "themes": [
            {
                "title": "Linear Equations",
                "chapter_ids": ["ch1", "ch2"],
                "summary": "Solving and graphing linear equations.",
                "prerequisite_theme_titles": [],
            },
            {
                "title": "Polynomials",
                "chapter_ids": ["ch3"],
                "summary": "Operations on polynomials.",
                "prerequisite_theme_titles": ["Linear Equations"],
            },
        ],
        "draft_terminal_objectives": [
            {
                "statement": "Students will analyze linear systems.",
                "bloom_level": "analyze",
                "bloom_verb": "analyze",
                "abcd": {
                    "audience": "Students",
                    "behavior": {"verb": "analyze", "action_object": "systems"},
                    "condition": "given a word problem",
                    "degree": "with 80% accuracy",
                },
                "source_refs": [{"ref": "ch1", "chunk_ids": []}],
            },
            {
                "statement": "Students will evaluate polynomial expressions.",
                "bloom_level": "evaluate",
                "bloom_verb": "evaluate",
                "source_refs": [],
            },
        ],
    }


def _concepts_payload() -> dict:
    return {
        "concepts": [
            {
                "canonical": "slope-intercept form",
                "aliases": ["slope intercept", "y = mx + b"],
                "definition_hint": "the form y = mx + b of a linear equation",
                "chapter_ids": ["ch4"],
            },
            {
                "canonical": "prime factorization",
                "aliases": [],
                "definition_hint": "expressing an integer as a product of primes",
            },
        ]
    }


def _chapter_objectives_payload() -> dict:
    return {
        "chapter_objectives": [
            {
                "statement": "Students will solve two-step equations.",
                "bloom_level": "apply",
                "bloom_verb": "solve",
                "abcd": {
                    "audience": "Students",
                    "behavior": {"verb": "solve", "action_object": "equations"},
                    "condition": "given an equation",
                    "degree": "correctly",
                },
                "source_refs": [{"ref": "ch1", "chunk_ids": []}],
                "sub_objectives": [
                    "Isolate the variable term.",
                    "Divide by the coefficient.",
                ],
            }
        ]
    }


def _reconcile_payload() -> dict:
    return {
        "terminal_objectives": [
            {
                "statement": "Students will solve and graph linear equations.",
                "bloom_level": "apply",
                "bloom_verb": "solve",
                "abcd": {
                    "audience": "Students",
                    "behavior": {"verb": "solve", "action_object": "equations"},
                    "condition": "given a problem",
                    "degree": "accurately",
                },
                "source_refs": [],
            }
        ]
    }


def _stub_chapter(cid: str = "ch1", text: str = "Some chapter prose.") -> dict:
    return {
        "id": cid,
        "title": f"Chapter {cid}",
        "chapter_text": text,
        "sections": [{"title": "Section A"}],
    }


def _provider(responses: List[str], capture: Any = None) -> TextbookSynthesisProvider:
    """Build an anthropic-backed provider with a mocked client."""
    return TextbookSynthesisProvider(
        provider="anthropic",
        anthropic_client=_FakeAnthropicClient(responses),
        capture=capture,
    )


# ===========================================================================
# Construction
# ===========================================================================


def test_default_provider_is_anthropic_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    assert DEFAULT_PROVIDER == "anthropic"
    p = TextbookSynthesisProvider(anthropic_client=object())
    assert p._provider == "anthropic"


def test_env_var_selects_provider(monkeypatch):
    monkeypatch.setenv(ENV_PROVIDER, "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    p = TextbookSynthesisProvider()
    assert p._provider == "together"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    with pytest.raises(ValueError, match="unknown provider"):
        TextbookSynthesisProvider(provider="not-a-provider")


# ===========================================================================
# Registry-dynamic resolution (W-D12 contract)
# ===========================================================================


def test_registry_dynamic_provider_resolution(monkeypatch):
    """A monkeypatched registry entry resolves without a code edit."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    fake_registry = dict(tsp._OPENAI_COMPATIBLE_PROVIDERS)
    fake_registry["acme"] = {"base_url_default": "http://acme/v1"}
    monkeypatch.setattr(tsp, "_OPENAI_COMPATIBLE_PROVIDERS", fake_registry)

    # The supported tuple now includes the patched-in provider.
    assert "acme" in _build_supported_providers()

    # Construction resolves it: base-init wires under "local", and the
    # runtime provider label is restored to the registry name.
    p = TextbookSynthesisProvider(provider="acme")
    assert p._provider == "acme"
    assert p._registry_provider == "acme"


def test_registry_provider_routes_through_wd12_backend(monkeypatch):
    """A registry provider dispatches via the injected W-D12 backend."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    fake_registry = dict(tsp._OPENAI_COMPATIBLE_PROVIDERS)
    fake_registry["acme"] = {"base_url_default": "http://acme/v1"}
    monkeypatch.setattr(tsp, "_OPENAI_COMPATIBLE_PROVIDERS", fake_registry)

    class _FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def complete_sync(self, *args: Any, **kwargs: Any) -> str:
            self.calls += 1
            return json.dumps(_outline_payload())

    backend = _FakeBackend()
    p = TextbookSynthesisProvider(
        provider="acme",
        openai_compatible_backend=backend,
    )
    out = p.synthesize_outline(
        {"chapters": [_stub_chapter()]}, course_name="ALG_101"
    )
    assert backend.calls == 1
    assert out["structure_enrichment"]["provider"] == "acme"


def test_registry_provider_forwards_grammar_extra_payload(monkeypatch):
    """The W-D12 backend branch forwards ``extra_payload`` verbatim.

    Regression (introalgebra-bc-02 attempt 4): the registry-provider branch of
    ``_dispatch_call`` dropped ``extra_payload``, so the grammar /
    ``response_format`` schema never reached the seat — window synthesis ran
    UNCONSTRAINED and dense windows truncated (ch7#w1) or emitted unparseable
    JSON (ch9#w0), each a §5.4 chapter-content loss.
    """
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    fake_registry = dict(tsp._OPENAI_COMPATIBLE_PROVIDERS)
    fake_registry["acme"] = {"base_url_default": "http://acme/v1"}
    monkeypatch.setattr(tsp, "_OPENAI_COMPATIBLE_PROVIDERS", fake_registry)

    class _RecordingBackend:
        def __init__(self) -> None:
            self.kwargs: Dict[str, Any] = {}

        def complete_sync(self, *args: Any, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "{}"

    backend = _RecordingBackend()
    p = TextbookSynthesisProvider(
        provider="acme",
        openai_compatible_backend=backend,
    )
    grammar = {"response_format": {"type": "json_schema"}}
    p._dispatch_call("hello", extra_payload=grammar)
    assert backend.kwargs.get("extra_payload") == grammar

    # No grammar payload → the kwarg is passed as None (byte-stable body).
    p._dispatch_call("hello", extra_payload=None)
    assert backend.kwargs.get("extra_payload") is None


# ===========================================================================
# Stage 1 — synthesize_outline
# ===========================================================================


def test_synthesize_outline_returns_normalised_shape():
    p = _provider([json.dumps(_outline_payload())])
    out = p.synthesize_outline(
        {"chapters": [_stub_chapter("ch1"), _stub_chapter("ch2")]},
        course_name="ALG_101",
    )
    assert set(out) == {
        "semantic_outline",
        "draft_terminal_objectives",
        "structure_enrichment",
    }
    assert out["semantic_outline"]["course_summary"]
    assert len(out["semantic_outline"]["themes"]) == 2
    drafts = out["draft_terminal_objectives"]
    assert len(drafts) == 2
    # IDs minted sequentially; draft flag stamped.
    assert [d["id"] for d in drafts] == ["TO-01", "TO-02"]
    assert all(d["draft"] is True for d in drafts)
    enr = out["structure_enrichment"]
    assert enr["enriched"] is True
    assert enr["call_mode"] == "single"
    assert enr["calls"] == 1
    assert enr["provider"] == "anthropic"


def test_synthesize_outline_lenient_fenced_json():
    fenced = "```json\n" + json.dumps(_outline_payload()) + "\n```"
    p = _provider([fenced])
    out = p.synthesize_outline(
        {"chapters": [_stub_chapter()]}, course_name="ALG_101"
    )
    assert len(out["draft_terminal_objectives"]) == 2


def test_synthesize_outline_empty_course_name_raises():
    p = _provider([json.dumps(_outline_payload())])
    with pytest.raises(ValueError, match="course_name required"):
        p.synthesize_outline({"chapters": []}, course_name="  ")


def test_synthesize_outline_exhausted_raises_outline_code():
    p = _provider(["not json at all"])
    with pytest.raises(TextbookSynthesisProviderError) as exc:
        p.synthesize_outline(
            {"chapters": [_stub_chapter()]}, course_name="ALG_101"
        )
    assert exc.value.code == "outline_exhausted"


def test_synthesize_outline_chaptered_split_past_skeleton_budget():
    """A textbook whose rendered skeleton exceeds the budget splits
    into multiple chaptered calls, each emitting its own decision."""
    # Each chapter's chapter_text contributes a ~200-char sample to the
    # skeleton; pad it well past the budget across many chapters.
    big_text = "x" * 5_000
    n_chapters = (_SKELETON_CHAR_BUDGET // 250) + 5
    chapters = [
        _stub_chapter(f"ch{i}", text=big_text)
        for i in range(n_chapters)
    ]
    capture = _FakeCapture()
    # Provide one payload per expected call (last repeats anyway).
    p = _provider([json.dumps(_outline_payload())] * 50, capture=capture)
    out = p.synthesize_outline({"chapters": chapters}, course_name="ALG_101")

    assert out["structure_enrichment"]["call_mode"] == "chaptered"
    assert out["structure_enrichment"]["calls"] >= 2
    # One decision event per chaptered call.
    outline_events = [
        e for e in capture.events
        if e["decision_type"] == "textbook_outline_call"
    ]
    assert len(outline_events) == out["structure_enrichment"]["calls"]
    # Merged themes accumulate across calls.
    assert len(out["semantic_outline"]["themes"]) >= 2
    # Draft TO IDs are re-minted sequentially across the merged set.
    drafts = out["draft_terminal_objectives"]
    assert [d["id"] for d in drafts] == [
        f"TO-{i:02d}" for i in range(1, len(drafts) + 1)
    ]


# ===========================================================================
# Stage 3 — synthesize_concepts
# ===========================================================================


def test_synthesize_concepts_returns_normalised_shape():
    p = _provider([json.dumps(_concepts_payload())])
    out = p.synthesize_concepts(
        _stub_chapter("ch4"), course_name="ALG_101"
    )
    assert out["chapter_id"] == "ch4"
    concepts = out["concepts"]
    assert len(concepts) == 2
    first = concepts[0]
    assert first["canonical"] == "slope-intercept form"
    assert first["aliases"] == ["slope intercept", "y = mx + b"]
    assert first["definition_hint"]
    # Concept with no chapter_ids inherits the chapter's id.
    assert concepts[1]["chapter_ids"] == ["ch4"]


def test_synthesize_concepts_exhausted_raises_concepts_code():
    p = _provider(["garbage"])
    with pytest.raises(TextbookSynthesisProviderError) as exc:
        p.synthesize_concepts(_stub_chapter("ch4"), course_name="ALG_101")
    assert exc.value.code == "concepts_exhausted"


def test_synthesize_concepts_truncation_flag_in_decision():
    """A chapter past _CHAPTER_TEXT_BUDGET flags chapter_text_truncated."""
    capture = _FakeCapture()
    huge = "z" * (tsp._CHAPTER_TEXT_BUDGET + 5_000)
    p = _provider([json.dumps(_concepts_payload())], capture=capture)
    p.synthesize_concepts(
        _stub_chapter("ch9", text=huge), course_name="ALG_101"
    )
    ev = [
        e for e in capture.events
        if e["decision_type"] == "textbook_concept_call"
    ]
    assert len(ev) == 1
    assert "chapter_text_truncated=True" in ev[0]["rationale"]


# ===========================================================================
# Stage 2 — synthesize_chapter_objectives
# ===========================================================================


def test_synthesize_chapter_objectives_returns_normalised_shape():
    p = _provider([json.dumps(_chapter_objectives_payload())])
    out = p.synthesize_chapter_objectives(
        _stub_chapter("ch1"),
        course_name="ALG_101",
        draft_terminal_objectives=[
            {"statement": "Students will analyze linear systems."}
        ],
    )
    assert out["chapter_id"] == "ch1"
    objs = out["chapter_objectives"]
    assert len(objs) == 1
    obj = objs[0]
    # CO IDs are NOT minted by the provider (plan §5.2).
    assert "id" not in obj
    assert obj["statement"]
    assert obj["bloom_level"] == "apply"
    assert obj["sub_objectives"] == [
        "Isolate the variable term.",
        "Divide by the coefficient.",
    ]
    assert obj["chapter_id"] == "ch1"


def test_synthesize_chapter_objectives_exhausted_raises_code():
    p = _provider(["::not json::"])
    with pytest.raises(TextbookSynthesisProviderError) as exc:
        p.synthesize_chapter_objectives(
            _stub_chapter("ch1"), course_name="ALG_101"
        )
    assert exc.value.code == "chapter_objectives_exhausted"


# ===========================================================================
# Reconciliation — reconcile_terminal_objectives
# ===========================================================================


def test_reconcile_terminal_objectives_returns_normalised_shape():
    p = _provider([json.dumps(_reconcile_payload())])
    out = p.reconcile_terminal_objectives(
        draft_terminal_objectives=[
            {"statement": "Students will analyze linear systems.",
             "draft": True},
        ],
        chapter_objectives=[
            {"statement": "Students will solve two-step equations."},
        ],
        course_name="ALG_101",
    )
    terminals = out["terminal_objectives"]
    assert len(terminals) == 1
    # TO IDs re-minted, draft flag dropped.
    assert terminals[0]["id"] == "TO-01"
    assert "draft" not in terminals[0]
    assert terminals[0]["statement"]


def test_reconcile_terminal_objectives_exhausted_raises_code():
    p = _provider(["nope"])
    with pytest.raises(TextbookSynthesisProviderError) as exc:
        p.reconcile_terminal_objectives(
            draft_terminal_objectives=[{"statement": "x"}],
            chapter_objectives=[{"statement": "y"}],
            course_name="ALG_101",
        )
    assert exc.value.code == "reconcile_exhausted"


# ===========================================================================
# WS1 — author_terminal_for_cluster (bottom-up TO derivation)
# ===========================================================================


def _author_terminal_payload() -> dict:
    return {
        "terminal_objective": {
            "statement": "Students will solve and graph linear equations.",
            "bloom_level": "apply",
            "bloom_verb": "solve",
            "abcd": {
                "audience": "Students",
                "behavior": {"verb": "solve", "action_object": "equations"},
                "condition": "given a problem",
                "degree": "accurately",
            },
            "source_refs": [],
        }
    }


def _cluster_cos() -> List[Dict[str, Any]]:
    return [
        {"statement": "Students will solve one-step linear equations."},
        {"statement": "Students will graph a line from slope-intercept form."},
    ]


def test_author_terminal_for_cluster_returns_single_to():
    p = _provider([json.dumps(_author_terminal_payload())])
    to = p.author_terminal_for_cluster(
        _cluster_cos(), course_name="ALG_101", cluster_index=1
    )
    assert to is not None
    assert to["statement"]
    assert to["bloom_level"] == "apply"
    # Caller mints the id — the provider must NOT.
    assert "id" not in to


def test_author_terminal_lenient_list_payload():
    """A model that emits a single-element terminal_objectives LIST is
    leniently accepted (still ONE TO, no id)."""
    payload = {
        "terminal_objectives": [
            {
                "statement": "Students will analyze linear systems.",
                "bloom_level": "analyze",
                "source_refs": [],
            }
        ]
    }
    p = _provider([json.dumps(payload)])
    to = p.author_terminal_for_cluster(
        _cluster_cos(), course_name="ALG_101", cluster_index=2
    )
    assert to is not None
    assert "id" not in to
    assert to["bloom_level"] == "analyze"


def test_author_terminal_parse_exhaustion_returns_none():
    """Fail-SOFT: parse exhaustion → None (not a raise); capture success=false.

    Three garbage replies — one per attempt of the parse-retry budget added
    with the constrained-dispatch fix (pre-fix this surface was single-shot).
    """
    capture = _FakeCapture()
    p = _provider(["not json at all"] * 3, capture=capture)
    to = p.author_terminal_for_cluster(
        _cluster_cos(), course_name="ALG_101", cluster_index=3
    )
    assert to is None
    ev = next(
        e for e in capture.events
        if e["decision_type"] == "terminal_objective_authoring"
    )
    assert "success=False" in ev["rationale"]


def test_author_terminal_retries_past_bad_reply():
    """A garbage first reply no longer demotes the TO to the template —
    the retry loop recovers on the second attempt."""
    p = _provider(
        ["not json at all", json.dumps(_author_terminal_payload())]
    )
    to = p.author_terminal_for_cluster(
        _cluster_cos(), course_name="ALG_101", cluster_index=4
    )
    assert to is not None
    assert to["statement"]


def test_author_terminal_forwards_grammar_payload(monkeypatch):
    """The cluster-author call dispatches schema-constrained (regression:
    it was the last synthesis surface running unconstrained)."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    fake_registry = dict(tsp._OPENAI_COMPATIBLE_PROVIDERS)
    fake_registry["acme"] = {"base_url_default": "http://acme/v1"}
    monkeypatch.setattr(tsp, "_OPENAI_COMPATIBLE_PROVIDERS", fake_registry)

    class _RecordingBackend:
        def __init__(self) -> None:
            self.kwargs: Dict[str, Any] = {}

        def complete_sync(self, *args: Any, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return json.dumps(_author_terminal_payload())

    backend = _RecordingBackend()
    p = TextbookSynthesisProvider(
        provider="acme",
        openai_compatible_backend=backend,
        grammar_mode="response_format",
    )
    to = p.author_terminal_for_cluster(
        _cluster_cos(), course_name="ALG_101", cluster_index=1
    )
    assert to is not None
    rf = (backend.kwargs.get("extra_payload") or {}).get("response_format")
    assert rf and rf["type"] == "json_schema"
    assert (
        rf["json_schema"]["schema"]["required"] == ["terminal_objective"]
    )


def test_author_terminal_decision_capture_fires():
    """MANDATORY — new LLM call site emits exactly one
    terminal_objective_authoring event with a dynamic-signal rationale."""
    capture = _FakeCapture()
    p = _provider([json.dumps(_author_terminal_payload())], capture=capture)
    p.author_terminal_for_cluster(
        _cluster_cos(), course_name="ALG_101", cluster_index=7
    )
    events = [
        e for e in capture.events
        if e["decision_type"] == "terminal_objective_authoring"
    ]
    assert len(events) == 1
    rationale = events[0]["rationale"]
    assert len(rationale) >= 20
    # ≥4 dynamic signals: provider, model, cluster_index, co_count_input.
    assert "provider=anthropic" in rationale
    assert "model=" in rationale
    assert "cluster_index=7" in rationale
    assert "co_count_input=2" in rationale
    assert "success=True" in rationale


def test_author_terminal_prompt_keeps_anti_invent_guard():
    p = _provider([json.dumps(_author_terminal_payload())])
    prompt = p._render_author_terminal_prompt(
        cluster_cos=_cluster_cos(), course_name="ALG_101"
    )
    assert "Do NOT invent" in prompt
    assert "EXACTLY ONE terminal" in prompt
    assert "terminal_objective" in prompt


# ===========================================================================
# Per-chapter batching helper (plan §5.3)
# ===========================================================================


def test_batch_chapters_groups_at_most_ten():
    chapters = list(range(23))
    batches = TextbookSynthesisProvider.batch_chapters(chapters)
    assert _CHAPTER_BATCH_SIZE == 10
    assert [len(b) for b in batches] == [10, 10, 3]
    # Order preserved, no chapter lost or duplicated.
    assert [c for b in batches for c in b] == chapters


def test_batch_chapters_custom_size():
    batches = TextbookSynthesisProvider.batch_chapters(
        list(range(7)), batch_size=3
    )
    assert [len(b) for b in batches] == [3, 3, 1]


def test_batch_chapters_empty():
    assert TextbookSynthesisProvider.batch_chapters([]) == []


def test_batch_chapters_dispatch_in_groups():
    """Simulate a handler-style batched dispatch: every batch is ≤10
    and every chapter produces exactly one concept call."""
    chapters = [_stub_chapter(f"ch{i}") for i in range(25)]
    capture = _FakeCapture()
    p = _provider([json.dumps(_concepts_payload())] * 30, capture=capture)

    batches = TextbookSynthesisProvider.batch_chapters(chapters)
    assert all(len(b) <= _CHAPTER_BATCH_SIZE for b in batches)
    results = []
    for batch in batches:
        for chapter in batch:
            results.append(
                p.synthesize_concepts(chapter, course_name="ALG_101")
            )
    assert len(results) == 25
    concept_events = [
        e for e in capture.events
        if e["decision_type"] == "textbook_concept_call"
    ]
    assert len(concept_events) == 25


# ===========================================================================
# Decision capture — all four events fire with runtime provider
# ===========================================================================


def test_all_four_decision_events_fire_with_runtime_provider():
    capture = _FakeCapture()
    p = _provider(
        [
            json.dumps(_outline_payload()),
            json.dumps(_concepts_payload()),
            json.dumps(_chapter_objectives_payload()),
            json.dumps(_reconcile_payload()),
        ],
        capture=capture,
    )
    p.synthesize_outline(
        {"chapters": [_stub_chapter()]}, course_name="ALG_101"
    )
    p.synthesize_concepts(_stub_chapter("ch4"), course_name="ALG_101")
    p.synthesize_chapter_objectives(
        _stub_chapter("ch1"), course_name="ALG_101"
    )
    p.reconcile_terminal_objectives(
        draft_terminal_objectives=[{"statement": "x"}],
        chapter_objectives=[{"statement": "y"}],
        course_name="ALG_101",
    )
    seen_types = {e["decision_type"] for e in capture.events}
    assert seen_types == {
        "textbook_outline_call",
        "textbook_concept_call",
        "chapter_objective_call",
        "terminal_objective_reconciliation",
    }
    # Every rationale interpolates the RUNTIME provider, not a static
    # label, and is ≥20 chars.
    for ev in capture.events:
        assert "provider=anthropic" in ev["rationale"]
        assert len(ev["rationale"]) >= 20


def test_decision_events_carry_registry_provider_name(monkeypatch):
    """When the runtime provider is a registry entry, the decision
    rationale records that registry name — not the base alias."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    fake_registry = dict(tsp._OPENAI_COMPATIBLE_PROVIDERS)
    fake_registry["acme"] = {"base_url_default": "http://acme/v1"}
    monkeypatch.setattr(tsp, "_OPENAI_COMPATIBLE_PROVIDERS", fake_registry)

    class _FakeBackend:
        def complete_sync(self, *args: Any, **kwargs: Any) -> str:
            return json.dumps(_concepts_payload())

    capture = _FakeCapture()
    p = TextbookSynthesisProvider(
        provider="acme",
        openai_compatible_backend=_FakeBackend(),
        capture=capture,
    )
    p.synthesize_concepts(_stub_chapter("ch4"), course_name="ALG_101")
    assert capture.events
    assert "provider=acme" in capture.events[0]["rationale"]


def test_decision_event_fires_on_failure_with_success_false():
    """A failed (exhausted) call still emits a decision event with
    success=False before raising."""
    capture = _FakeCapture()
    p = _provider(["garbage"], capture=capture)
    with pytest.raises(TextbookSynthesisProviderError):
        p.synthesize_concepts(_stub_chapter("ch4"), course_name="ALG_101")
    assert len(capture.events) == 1
    assert "success=False" in capture.events[0]["rationale"]
    assert "last_error=" in capture.events[0]["rationale"]


# ===========================================================================
# H2 — HTTP timeout threading (kwarg / env / default → OpenAICompatibleClient)
# ===========================================================================


def test_timeout_default_is_300s_threaded_into_local_client(monkeypatch):
    """Unset timeout resolves to 300s and lands on the embedded
    OpenAICompatibleClient for the local backend."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv(ENV_TIMEOUT, raising=False)
    p = TextbookSynthesisProvider(provider="local")
    assert _DEFAULT_TIMEOUT_SECONDS == 300.0
    assert p._timeout == 300.0
    # The constructor-built OpenAICompatibleClient carries the timeout.
    assert p._oa_client is not None
    assert p._oa_client._timeout == 300.0


def test_timeout_kwarg_overrides_default(monkeypatch):
    """An explicit timeout kwarg wins over the 300s default."""
    monkeypatch.delenv(ENV_TIMEOUT, raising=False)
    p = TextbookSynthesisProvider(provider="local", timeout=123.0)
    assert p._timeout == 123.0
    assert p._oa_client._timeout == 123.0


def test_timeout_env_var_threaded_into_client(monkeypatch):
    """TEXTBOOK_SYNTHESIS_TIMEOUT_SECONDS is honoured when no kwarg."""
    monkeypatch.setenv(ENV_TIMEOUT, "240")
    p = TextbookSynthesisProvider(provider="local")
    assert p._timeout == 240.0
    assert p._oa_client._timeout == 240.0


def test_timeout_kwarg_beats_env_var(monkeypatch):
    """Resolution chain: kwarg > env > default."""
    monkeypatch.setenv(ENV_TIMEOUT, "240")
    p = TextbookSynthesisProvider(provider="local", timeout=99.0)
    assert p._timeout == 99.0
    assert p._oa_client._timeout == 99.0


def test_timeout_threaded_into_together_client(monkeypatch):
    """The together backend's OpenAICompatibleClient also gets the
    resolved timeout."""
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    monkeypatch.delenv(ENV_TIMEOUT, raising=False)
    p = TextbookSynthesisProvider(provider="together", timeout=150.0)
    assert p._oa_client._timeout == 150.0


def test_invalid_timeout_env_falls_back_to_default(monkeypatch):
    """A non-numeric env value is ignored — default 300s wins."""
    monkeypatch.setenv(ENV_TIMEOUT, "not-a-number")
    p = TextbookSynthesisProvider(provider="local")
    assert p._timeout == _DEFAULT_TIMEOUT_SECONDS


def test_base_provider_default_timeout_when_unset():
    """Sanity: with no timeout passed the base falls through to the
    OpenAICompatibleClient's own DEFAULT_TIMEOUT_SECONDS."""
    from Courseforge.generators._base import _BaseLLMProvider

    class _MiniProvider(_BaseLLMProvider):
        def _render_user_prompt(self, *a, **k):  # pragma: no cover
            return ""

        def _emit_per_call_decision(self, **k):  # pragma: no cover
            return None

    p = _MiniProvider(provider="local")
    assert p._oa_client._timeout == DEFAULT_TIMEOUT_SECONDS


# ===========================================================================
# M1 — lenient JSON parser robustness (trailing commas, double fences)
# ===========================================================================


def test_extract_json_lenient_strips_trailing_commas():
    """Trailing commas (,} / ,]) — a Qwen-14B-q4 drift mode — parse."""
    blob = (
        '{"concepts": ['
        '{"canonical": "slope", "aliases": ["a", "b",],},'
        '],}'
    )
    parsed = TextbookSynthesisProvider._extract_json_lenient(blob)
    assert parsed is not None
    assert parsed["concepts"][0]["canonical"] == "slope"
    assert parsed["concepts"][0]["aliases"] == ["a", "b"]


def test_extract_json_lenient_trailing_comma_inside_fence():
    """Trailing-comma repair also fires inside a markdown fence."""
    inner = '{"course_summary": "x", "themes": [],}'
    blob = "```json\n" + inner + "\n```"
    parsed = TextbookSynthesisProvider._extract_json_lenient(blob)
    assert parsed is not None
    assert parsed["course_summary"] == "x"


def test_extract_json_lenient_first_of_two_fences():
    """Two fenced blocks: the regex anchors to the FIRST block and
    does not span across both."""
    blob = (
        "Here is the answer:\n"
        '```json\n{"course_summary": "real"}\n```\n'
        "And some scratch work:\n"
        "```json\n{\"junk\": true}\n```\n"
    )
    parsed = TextbookSynthesisProvider._extract_json_lenient(blob)
    assert parsed is not None
    assert parsed.get("course_summary") == "real"


def test_extract_json_lenient_clean_still_parses():
    """Regression guard: clean JSON still parses on the happy path."""
    parsed = TextbookSynthesisProvider._extract_json_lenient(
        json.dumps(_outline_payload())
    )
    assert parsed is not None
    assert parsed["course_summary"]


# ===========================================================================
# M2 — SynthesisProviderError re-raised as TextbookSynthesisProviderError
# ===========================================================================


class _RaisingBackend:
    """W-D12 backend stub whose complete_sync raises
    SynthesisProviderError (mirrors the OpenAICompatibleClient
    non-retryable-4xx / max-retries / model-not-found path)."""

    def __init__(self, code: str = "404") -> None:
        self._code = code

    def complete_sync(self, *args: Any, **kwargs: Any) -> str:
        raise SynthesisProviderError(
            "backend exploded", code=self._code
        )


def test_synthesis_provider_error_resurfaces_as_textbook_error(monkeypatch):
    """A SynthesisProviderError from the backend surfaces to the
    caller as TextbookSynthesisProviderError, not the raw type."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    fake_registry = dict(tsp._OPENAI_COMPATIBLE_PROVIDERS)
    fake_registry["acme"] = {"base_url_default": "http://acme/v1"}
    monkeypatch.setattr(tsp, "_OPENAI_COMPATIBLE_PROVIDERS", fake_registry)

    p = TextbookSynthesisProvider(
        provider="acme",
        openai_compatible_backend=_RaisingBackend(code="404"),
    )
    with pytest.raises(TextbookSynthesisProviderError) as exc:
        p.synthesize_concepts(_stub_chapter("ch4"), course_name="ALG_101")
    # The original SynthesisProviderError must NOT escape uncaught.
    assert not isinstance(exc.value, SynthesisProviderError)
    assert exc.value.code == "dispatch_404"
    assert isinstance(exc.value.__cause__, SynthesisProviderError)


def test_synthesis_provider_error_on_local_backend_resurfaces(monkeypatch):
    """The local / together path (super()._dispatch_call) is also
    wrapped — a SynthesisProviderError from _post_with_retry surfaces
    as TextbookSynthesisProviderError."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv(ENV_TIMEOUT, raising=False)
    p = TextbookSynthesisProvider(provider="local")

    def _boom(*args: Any, **kwargs: Any):
        raise SynthesisProviderError("HTTP 400", code="400")

    monkeypatch.setattr(p._oa_client, "_post_with_retry", _boom)
    with pytest.raises(TextbookSynthesisProviderError) as exc:
        p.synthesize_outline(
            {"chapters": [_stub_chapter()]}, course_name="ALG_101"
        )
    assert exc.value.code == "dispatch_400"


# ===========================================================================
# Fix 5 — local backend _dispatch_call pins options.num_ctx
# ===========================================================================


def test_local_dispatch_includes_num_ctx_in_payload(monkeypatch):
    """The local backend's wire payload carries options.num_ctx."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv(ENV_TIMEOUT, raising=False)
    p = TextbookSynthesisProvider(provider="local")

    seen: Dict[str, Any] = {}

    def _capture_payload(payload: Dict[str, Any]):
        seen.update(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(_outline_payload())
                    },
                    "finish_reason": "stop",
                }
            ]
        }, 0

    monkeypatch.setattr(p._oa_client, "_post_with_retry", _capture_payload)
    p.synthesize_outline(
        {"chapters": [_stub_chapter()]}, course_name="ALG_101"
    )
    assert "options" in seen
    assert seen["options"]["num_ctx"] == _LOCAL_NUM_CTX


def test_merge_local_num_ctx_preserves_caller_extra_payload():
    """A caller-supplied extra_payload (and options sub-keys) is
    preserved; num_ctx is only setdefault-ed."""
    p = TextbookSynthesisProvider(provider="local")
    merged = p._merge_local_num_ctx(
        {"grammar": "g", "options": {"temperature": 0.1}}
    )
    assert merged["grammar"] == "g"
    assert merged["options"]["temperature"] == 0.1
    assert merged["options"]["num_ctx"] == _LOCAL_NUM_CTX


def test_merge_local_num_ctx_respects_caller_num_ctx():
    """A caller-supplied num_ctx wins over the default pin."""
    p = TextbookSynthesisProvider(provider="local")
    merged = p._merge_local_num_ctx({"options": {"num_ctx": 8192}})
    assert merged["options"]["num_ctx"] == 8192


def test_merge_local_num_ctx_noop_for_together(monkeypatch):
    """The together / anthropic backends are NOT given options.num_ctx."""
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    p = TextbookSynthesisProvider(provider="together")
    assert p._merge_local_num_ctx({"x": 1}) == {"x": 1}
    assert p._merge_local_num_ctx(None) is None


def test_finish_reason_length_surfaces_output_truncated():
    """OpenAICompatibleClient._extract_text raises a distinct
    output_truncated code when finish_reason == 'length'."""
    from Trainforge.generators._openai_compatible_client import (
        OpenAICompatibleClient,
    )

    body = {
        "choices": [
            {
                "message": {"content": '{"partial": '},
                "finish_reason": "length",
            }
        ]
    }
    with pytest.raises(SynthesisProviderError) as exc:
        OpenAICompatibleClient._extract_text(body)
    assert exc.value.code == "output_truncated"


# ===========================================================================
# W2 — synthesize_window_objectives: ⊆ enforcement + grammar + parse
# ===========================================================================


def _window(chunk_ids, *, chapter_id="ch1", window_index=0):
    return {
        "chapter_id": chapter_id,
        "window_index": window_index,
        "chunk_ids": list(chunk_ids),
        "chunks": [{"id": c, "text": f"body for {c}"} for c in chunk_ids],
        "join_method": "sourceid",
        "estimated_prompt_tokens": 100,
    }


def _window_payload(source_chunk_ids):
    return {
        "candidate_objectives": [
            {
                "statement": "Explain the core idea of this window's chunks.",
                "bloom_level": "understand",
                "bloom_verb": "explain",
                "source_chunk_ids": list(source_chunk_ids),
                "sub_objectives": ["sub one"],
            }
        ]
    }


def test_window_objectives_strips_out_of_set_citation():
    """A cited id NOT in the window's allowed set is stripped; counted."""
    capture = _FakeCapture()
    # Model cites one in-set id (c1) + one out-of-set id (c9).
    p = _provider([json.dumps(_window_payload(["c1", "c9"]))], capture=capture)
    out = p.synthesize_window_objectives(
        _window(["c1", "c2"]), course_name="ALG_101",
    )
    objs = out["candidate_objectives"]
    assert len(objs) == 1
    assert objs[0]["source_chunk_ids"] == ["c1"]  # c9 stripped (⊆ enforce)
    assert objs[0]["grounded_citation"] is True
    # source_refs reconstructed for on-disk back-compat.
    assert objs[0]["source_refs"] == [{"ref": "ch1", "chunk_ids": ["c1"]}]
    # The decision event surfaces the measured out-of-set drop count.
    ev = [e for e in capture.events if e["decision_type"] == "chapter_objective_call"]
    assert ev
    assert "citation_out_of_set_dropped=1" in ev[0]["rationale"]
    assert "allowed_chunk_id_count=2" in ev[0]["rationale"]


def test_window_objectives_empty_citation_not_dropped_at_normalize():
    """All cited ids out of set → empty surviving set, grounded_citation False,
    objective KEPT (Pass C owns the drop)."""
    capture = _FakeCapture()
    p = _provider([json.dumps(_window_payload(["zzz"]))], capture=capture)
    out = p.synthesize_window_objectives(
        _window(["c1", "c2"]), course_name="ALG_101",
    )
    objs = out["candidate_objectives"]
    assert len(objs) == 1  # NOT dropped at normalize
    assert objs[0]["source_chunk_ids"] == []
    assert objs[0]["grounded_citation"] is False
    ev = [e for e in capture.events if e["decision_type"] == "chapter_objective_call"]
    assert "empty_citation_objectives=1" in ev[0]["rationale"]


def test_window_objectives_lenient_fenced_json():
    """Markdown-fenced + trailing-comma variant recovers via lenient parse."""
    fenced = (
        "```json\n"
        + json.dumps(_window_payload(["c1"])).rstrip("}")
        + ",}\n```"  # trailing comma drift
    )
    p = _provider([fenced])
    out = p.synthesize_window_objectives(_window(["c1"]), course_name="ALG_101")
    assert len(out["candidate_objectives"]) == 1
    assert out["candidate_objectives"][0]["source_chunk_ids"] == ["c1"]


def test_window_objectives_schema_valid_parses_one_attempt():
    """A clean schema-valid JSON parses on the first attempt."""
    p = _provider([json.dumps(_window_payload(["c1", "c2"]))])
    out = p.synthesize_window_objectives(
        _window(["c1", "c2"]), course_name="ALG_101",
    )
    assert out["chapter_id"] == "ch1"
    assert out["window_index"] == 0
    assert out["candidate_objectives"][0]["source_chunk_ids"] == ["c1", "c2"]


def test_window_objectives_exhausted_raises_chapter_code():
    """Unparseable output exhausts the retry budget → chapter_objectives_exhausted."""
    p = _provider(["not json at all"])
    with pytest.raises(TextbookSynthesisProviderError) as exc:
        p.synthesize_window_objectives(_window(["c1"]), course_name="ALG_101")
    assert exc.value.code == "chapter_objectives_exhausted"


def test_window_objectives_empty_minitems_rejected_then_retried():
    """An empty source_chunk_ids array fails the schema (minItems 1) and the
    valid retry succeeds."""
    bad = json.dumps(_window_payload([]))          # source_chunk_ids: [] → invalid
    good = json.dumps(_window_payload(["c1"]))
    p = _provider([bad, good])
    out = p.synthesize_window_objectives(_window(["c1"]), course_name="ALG_101")
    assert out["candidate_objectives"][0]["source_chunk_ids"] == ["c1"]


def test_synthesis_grammar_payload_local_uses_format(monkeypatch):
    """Autodetect for a local Ollama provider → {format: <window schema>}."""
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")
    p = TextbookSynthesisProvider(provider="local")
    payload = p._build_synthesis_grammar_payload()
    assert "format" in payload
    assert payload["format"]["required"] == ["candidate_objectives"]


def test_synthesis_grammar_mode_kwarg_wins(monkeypatch):
    """grammar_mode='none' → empty payload (no constrained decoding)."""
    p = TextbookSynthesisProvider(provider="anthropic",
                                  anthropic_client=object(),
                                  grammar_mode="none")
    assert p._build_synthesis_grammar_payload() == {}


# ===========================================================================
# WS4 §1 — Stage-1 skeleton de-blinding (section-title + chapter_text sampling)
# ===========================================================================


def _stub_chapter_with_sections(cid, section_titles, text=""):
    return {
        "id": cid,
        "title": f"Chapter {cid}",
        "chapter_text": text,
        "sections": [{"title": t} for t in section_titles],
    }


def test_skeleton_section_titles_span_full_range():
    """141 section titles → the rendered skeleton contains the LAST title and
    a back-half title (not just the first 12)."""
    titles = [f"Section {i:03d}" for i in range(141)]
    chapter = _stub_chapter_with_sections("ch1", titles)
    rendered = TextbookSynthesisProvider._render_chapter_skeleton(chapter)
    # last title always present (forced inclusion of index n-1)
    assert "Section 140" in rendered
    # The legacy [:12] head-slice stopped at index 011; assert a meaningful
    # share of back-half titles (index >= 70) survive the even-stride sample.
    sampled = TextbookSynthesisProvider._sample_section_titles(titles)
    back_half = [t for t in sampled if int(t.split()[1]) >= 70]
    assert len(back_half) >= 10, f"too few back-half titles: {sampled}"
    # And those back-half titles are actually rendered into the skeleton.
    assert back_half[0] in rendered


def test_skeleton_section_titles_capped_at_max():
    """Sampled section-title count never exceeds _SKELETON_MAX_SECTION_TITLES,
    and the sample always includes index 0 + n-1."""
    titles = [f"S{i:03d}" for i in range(300)]
    sampled = TextbookSynthesisProvider._sample_section_titles(titles)
    assert len(sampled) <= _SKELETON_MAX_SECTION_TITLES + 1  # +1 for forced last
    # First and last always present.
    assert sampled[0] == "S000"
    assert sampled[-1] == "S299"
    # All sampled titles are members of the input (no fabrication).
    assert set(sampled).issubset(set(titles))


def test_skeleton_section_titles_under_cap_unchanged():
    """A short section list (<= cap) is returned verbatim."""
    titles = [f"S{i}" for i in range(5)]
    assert TextbookSynthesisProvider._sample_section_titles(titles) == titles


def test_skeleton_chapter_text_multi_offset():
    """Long chapter_text → head/mid/tail samples present and distinct."""
    # Build text where each third is identifiable.
    text = ("AAA" * 200) + ("MMM" * 200) + ("ZZZ" * 200)
    chapter = _stub_chapter_with_sections("ch1", ["S1"], text=text)
    rendered = TextbookSynthesisProvider._render_chapter_skeleton(chapter)
    assert "sample[head]:" in rendered
    assert "sample[mid]:" in rendered
    assert "sample[tail]:" in rendered
    samples = TextbookSynthesisProvider._sample_chapter_text(text)
    labels = [lbl for lbl, _ in samples]
    assert labels == ["head", "mid", "tail"]
    head, mid, tail = (snip for _, snip in samples)
    assert head.startswith("A")
    assert tail.endswith("Z")
    # The three windows draw from different regions.
    assert head != mid != tail


def test_skeleton_short_chapter_text_single_window():
    """Short chapter_text (< 2 windows) → only a head sample."""
    short = "x" * 100
    samples = TextbookSynthesisProvider._sample_chapter_text(short)
    assert samples == [("head", short)]
    chapter = _stub_chapter_with_sections("ch1", ["S1"], text=short)
    rendered = TextbookSynthesisProvider._render_chapter_skeleton(chapter)
    assert "sample[head]:" in rendered
    assert "sample[mid]:" not in rendered
    assert "sample[tail]:" not in rendered


def test_skeleton_empty_chapter_text_no_sample():
    """Empty chapter_text → no sample lines at all."""
    assert TextbookSynthesisProvider._sample_chapter_text("") == []
    chapter = _stub_chapter_with_sections("ch1", ["S1"], text="")
    rendered = TextbookSynthesisProvider._render_chapter_skeleton(chapter)
    assert "sample[" not in rendered


# ===========================================================================
# WS4 §2 — scope-aware DRAFT-TO band
# ===========================================================================


def test_draft_to_band_collapse_signature():
    """1 chapter with >40 sections → collapse band; normal multi-chapter →
    default band."""
    mega = _stub_chapter_with_sections(
        "ch1", [f"S{i}" for i in range(141)]
    )
    assert TextbookSynthesisProvider._draft_to_band([mega]) == _DRAFT_TO_BAND_COLLAPSE
    assert _DRAFT_TO_BAND_COLLAPSE == (6, 12)

    normal = [
        _stub_chapter_with_sections("ch1", ["S1", "S2"]),
        _stub_chapter_with_sections("ch2", ["S1", "S2"]),
        _stub_chapter_with_sections("ch3", ["S1"]),
    ]
    lo, hi = TextbookSynthesisProvider._draft_to_band(normal)
    assert (lo, hi)[0] == _DRAFT_TO_BAND_DEFAULT[0]
    # high is max(default_high, min(12, chapter_count)) == 6 for 3 chapters.
    assert hi == 6


def test_draft_to_band_single_chapter_few_sections_is_default():
    """A single chapter with FEW sections stays on the default band (only the
    >40-section collapse signature bumps it)."""
    small = _stub_chapter_with_sections("ch1", ["S1", "S2", "S3"])
    assert TextbookSynthesisProvider._draft_to_band([small]) == _DRAFT_TO_BAND_DEFAULT


def test_draft_to_band_boundary_strict_gt():
    """Exactly 40 sections → default band; 41 → collapse band (strict >)."""
    at = _stub_chapter_with_sections("ch1", [f"S{i}" for i in range(40)])
    over = _stub_chapter_with_sections("ch1", [f"S{i}" for i in range(41)])
    assert TextbookSynthesisProvider._draft_to_band([at]) == _DRAFT_TO_BAND_DEFAULT
    assert TextbookSynthesisProvider._draft_to_band([over]) == _DRAFT_TO_BAND_COLLAPSE


def test_outline_prompt_interpolates_band_and_span_clause():
    """The collapse band (6-12) and the span clause appear in the prompt."""
    p = TextbookSynthesisProvider(provider="anthropic", anthropic_client=object())
    mega = _stub_chapter_with_sections(
        "ch1", [f"S{i}" for i in range(141)], text="prose"
    )
    prompt = p._render_outline_prompt(chapters=[mega], course_name="ALG_101")
    assert "6-12 DRAFT" in prompt
    assert "span the whole" in prompt
    assert "section range" in prompt


def test_outline_prompt_default_band_normal_input():
    """Normal multi-chapter input interpolates the default 3-6 band."""
    p = TextbookSynthesisProvider(provider="anthropic", anthropic_client=object())
    chapters = [
        _stub_chapter_with_sections("ch1", ["S1"]),
        _stub_chapter_with_sections("ch2", ["S1"]),
    ]
    prompt = p._render_outline_prompt(chapters=chapters, course_name="ALG_101")
    assert "3-6 DRAFT" in prompt


def test_single_megachapter_stays_call_mode_single():
    """A single 141-section megachapter renders under _SKELETON_CHAR_BUDGET →
    one outline call (§3 guard)."""
    titles = [f"Section number {i:03d} about a topic" for i in range(141)]
    mega = _stub_chapter_with_sections(
        "ch1", titles, text="y" * 3_000
    )
    rendered = TextbookSynthesisProvider._render_chapter_skeleton(mega)
    assert len(rendered) < _SKELETON_CHAR_BUDGET
    capture = _FakeCapture()
    p = _provider([json.dumps(_outline_payload())] * 4, capture=capture)
    out = p.synthesize_outline({"chapters": [mega]}, course_name="ALG_101")
    assert out["structure_enrichment"]["call_mode"] == "single"
    assert out["structure_enrichment"]["calls"] == 1


def test_outline_decision_carries_band_and_section_count():
    """The textbook_outline_call decision rationale carries draft_to_band +
    section_count_input."""
    mega = _stub_chapter_with_sections(
        "ch1", [f"S{i}" for i in range(141)], text="prose"
    )
    capture = _FakeCapture()
    p = _provider([json.dumps(_outline_payload())], capture=capture)
    p.synthesize_outline({"chapters": [mega]}, course_name="ALG_101")
    events = [
        e for e in capture.events
        if e["decision_type"] == "textbook_outline_call"
    ]
    assert len(events) == 1
    rationale = events[0]["rationale"]
    assert "section_count_input=141" in rationale
    assert "draft_to_band=(6, 12)" in rationale


# ---------------------------------------------------------------------------
# bloom_verb backfill: a model-omitted verb is recovered from the objective's
# OWN statement (never fabricated, never copied from abcd). Regression for the
# real calibration course_planning review (2 of 31 COs had bloom_verb=None).
# ---------------------------------------------------------------------------


def test_normalise_backfills_missing_bloom_verb_from_statement():
    # No bloom_verb emitted -> recovered from the statement's own verb.
    e = TextbookSynthesisProvider._normalise_one_objective(
        {"statement": "Apply the rounding process to whole numbers.",
         "bloom_level": "apply"}
    )
    assert e["bloom_verb"] == "apply"
    # A concrete action verb is preferred over the bare level name.
    e2 = TextbookSynthesisProvider._normalise_one_objective(
        {"statement": "Apply the order of operations to simplify expressions.",
         "bloom_level": "apply"}
    )
    assert e2["bloom_verb"] == "simplify"


def test_normalise_does_not_fabricate_bloom_verb_when_statement_has_none():
    # Statement carries no Bloom verb -> field stays absent (the gate flags it),
    # never invented.
    e = TextbookSynthesisProvider._normalise_one_objective(
        {"statement": "Prime numbers exist in the natural numbers.",
         "bloom_level": "remember"}
    )
    assert "bloom_verb" not in e


def test_normalise_preserves_explicit_bloom_verb():
    e = TextbookSynthesisProvider._normalise_one_objective(
        {"statement": "Anything.", "bloom_level": "apply", "bloom_verb": "solve"}
    )
    assert e["bloom_verb"] == "solve"


# ===========================================================================
# Reasoning "detailed thinking off" on the LOCAL dispatch path +
# TEXTBOOK_SYNTHESIS_MAX_TOKENS knob (nano-omni authoring-seat blocker fix)
# ===========================================================================

import httpx  # noqa: E402

from Courseforge.generators._textbook_synthesis_provider import (  # noqa: E402
    ENV_MAX_TOKENS,
    _DEFAULT_MAX_TOKENS,
    _resolve_synthesis_max_tokens,
)


def _local_body_capture(
    monkeypatch: Any,
    *,
    thinking_env: str | None,
    max_tokens_env: str | None = None,
    max_tokens_kwarg: int | None = None,
) -> Dict[str, Any]:
    """Dispatch one local call through a MockTransport + return the request JSON.

    Builds a ``TextbookSynthesisProvider(provider="local")`` whose embedded
    OpenAICompatibleClient is wired to an injected httpx MockTransport, so the
    outgoing ``/chat/completions`` body can be asserted without a live server.
    """
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    if thinking_env is None:
        monkeypatch.delenv("ED4ALL_REASONING_THINKING_OFF", raising=False)
    else:
        monkeypatch.setenv("ED4ALL_REASONING_THINKING_OFF", thinking_env)
    if max_tokens_env is None:
        monkeypatch.delenv(ENV_MAX_TOKENS, raising=False)
    else:
        monkeypatch.setenv(ENV_MAX_TOKENS, max_tokens_env)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"themes": []}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs: Dict[str, Any] = {"provider": "local", "client": client}
    if max_tokens_kwarg is not None:
        kwargs["max_tokens"] = max_tokens_kwarg
    p = TextbookSynthesisProvider(**kwargs)
    p._dispatch_call("author the objectives")
    assert seen, "no request captured"
    return json.loads(seen[0].content.decode("utf-8")), p


def test_local_dispatch_applies_thinking_off_when_env_on(monkeypatch):
    """ED4ALL_REASONING_THINKING_OFF=1 → the composed LOCAL synthesis request
    carries chat_template_kwargs.enable_thinking=false AND the system directive.

    Regression for the objective_extraction blocker: this dispatch path
    (_base.py::_dispatch_call_with_usage) builds the payload directly and
    bypasses OpenAICompatibleClient.chat_completion, so before the fix the
    thinking-off injection never reached a reasoning model here.
    """
    body, _p = _local_body_capture(monkeypatch, thinking_env="1")
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    system_msgs = [m for m in body["messages"] if m.get("role") == "system"]
    assert system_msgs, "expected a system message"
    assert "detailed thinking off" in system_msgs[0]["content"]


def test_local_dispatch_thinking_off_unset_is_byte_identical(monkeypatch):
    """Flag unset → NO chat_template_kwargs key on the LOCAL synthesis body
    (byte-compat for a non-reasoning Qwen deployment)."""
    body, _p = _local_body_capture(monkeypatch, thinking_env=None)
    assert "chat_template_kwargs" not in body


def test_local_dispatch_max_tokens_env_knob_lands_on_wire(monkeypatch):
    """TEXTBOOK_SYNTHESIS_MAX_TOKENS is honoured and rides the request body."""
    body, p = _local_body_capture(
        monkeypatch, thinking_env=None, max_tokens_env="6000"
    )
    assert p._max_tokens == 6000
    assert body["max_tokens"] == 6000


def test_max_tokens_default_when_env_unset(monkeypatch):
    """Unset env → the generous provider default (not an 800-style floor)."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv(ENV_MAX_TOKENS, raising=False)
    p = TextbookSynthesisProvider(provider="local")
    assert p._max_tokens == _DEFAULT_MAX_TOKENS == 4096


def test_max_tokens_kwarg_beats_env(monkeypatch):
    """Resolution chain: kwarg > env > default."""
    monkeypatch.setenv(ENV_MAX_TOKENS, "6000")
    p = TextbookSynthesisProvider(provider="local", max_tokens=1234)
    assert p._max_tokens == 1234


def test_resolve_synthesis_max_tokens_parse_with_fallback(monkeypatch):
    """Garbage / non-positive env → the default; explicit kwarg always wins."""
    monkeypatch.delenv(ENV_MAX_TOKENS, raising=False)
    assert _resolve_synthesis_max_tokens(None) == _DEFAULT_MAX_TOKENS
    for bad in ("not-an-int", "0", "-5", "  "):
        monkeypatch.setenv(ENV_MAX_TOKENS, bad)
        assert _resolve_synthesis_max_tokens(None) == _DEFAULT_MAX_TOKENS
    monkeypatch.setenv(ENV_MAX_TOKENS, "9000")
    assert _resolve_synthesis_max_tokens(None) == 9000
    # Explicit kwarg wins over env.
    assert _resolve_synthesis_max_tokens(2048) == 2048


# ===========================================================================
# Structure-aware synthesis — ED4ALL_SYNTHESIS_SKELETON per-window skeleton
# ===========================================================================

ENV_SYNTHESIS_SKELETON = tsp.ENV_SYNTHESIS_SKELETON


def _structure_with_headings():
    """A textbook_structure with a nested heading tree for chapter ``ch1``."""
    return {
        "chapters": [
            {
                "id": "ch1",
                "headingLevel": 1,
                "headingText": "Foundations",
                "sections": [
                    {
                        "headingLevel": 2,
                        "headingText": "Number Systems",
                        "subsections": [
                            {"headingLevel": 3, "headingText": "Integers"},
                            {"headingLevel": 3, "headingText": "Rationals"},
                        ],
                    },
                    {
                        "headingLevel": 2,
                        "headingText": "Order of Operations",
                        "subsections": [],
                    },
                ],
            },
            {
                "id": "ch2",
                "headingLevel": 1,
                "headingText": "Linear Equations",
                "sections": [{"headingLevel": 2, "headingText": "Slope"}],
            },
        ]
    }


def test_skeleton_flag_off_prompt_byte_identical(monkeypatch):
    """Flag off → the window prompt is byte-identical with and without the
    threaded textbook_structure (the CONTEXT-ONLY skeleton never appears)."""
    monkeypatch.delenv(ENV_SYNTHESIS_SKELETON, raising=False)
    p_no_structure = _provider([json.dumps(_window_payload(["c1"]))])
    p_with_structure = TextbookSynthesisProvider(
        provider="anthropic",
        anthropic_client=_FakeAnthropicClient([]),
        textbook_structure=_structure_with_headings(),
    )
    window = _window(["c1"], chapter_id="ch1")
    base = p_no_structure._render_window_objectives_prompt(
        chapter_id="ch1",
        window_index=0,
        window_chunks=window["chunks"],
        course_name="ALG_101",
        draft_terminal_objectives=[],
        heading_skeleton=p_no_structure._build_window_heading_skeleton("ch1"),
    )
    with_struct = p_with_structure._render_window_objectives_prompt(
        chapter_id="ch1",
        window_index=0,
        window_chunks=window["chunks"],
        course_name="ALG_101",
        draft_terminal_objectives=[],
        heading_skeleton=p_with_structure._build_window_heading_skeleton("ch1"),
    )
    assert base == with_struct
    assert "heading structure" not in base.lower()


def test_skeleton_flag_off_build_returns_empty(monkeypatch):
    monkeypatch.delenv(ENV_SYNTHESIS_SKELETON, raising=False)
    p = TextbookSynthesisProvider(
        provider="anthropic",
        anthropic_client=_FakeAnthropicClient([]),
        textbook_structure=_structure_with_headings(),
    )
    assert p._build_window_heading_skeleton("ch1") == ""


def test_skeleton_flag_on_section_present(monkeypatch):
    """Flag on → the prompt gains a CONTEXT-ONLY heading skeleton carrying the
    chapter + section + subsection headings, and the citation contract is
    unchanged (still 'cite ONLY these ids')."""
    monkeypatch.setenv(ENV_SYNTHESIS_SKELETON, "1")
    p = TextbookSynthesisProvider(
        provider="anthropic",
        anthropic_client=_FakeAnthropicClient([]),
        textbook_structure=_structure_with_headings(),
    )
    skeleton = p._build_window_heading_skeleton("ch1")
    assert skeleton  # non-empty
    prompt = p._render_window_objectives_prompt(
        chapter_id="ch1",
        window_index=0,
        window_chunks=[{"id": "c1", "text": "body"}],
        course_name="ALG_101",
        draft_terminal_objectives=[],
        heading_skeleton=skeleton,
    )
    assert "Document heading structure" in prompt
    assert "Foundations" in prompt
    assert "Number Systems" in prompt
    assert "Integers" in prompt
    # CONTEXT-ONLY: the citation-extraction contract is preserved verbatim.
    assert "cite ONLY these ids in source_chunk_ids" in prompt
    assert "NEVER emit source_chunk_ids: []" in prompt
    # The skeleton precedes the source-chunk section.
    assert prompt.index("Document heading structure") < prompt.index("Source chunks")


def test_skeleton_flag_on_unknown_chapter_empty(monkeypatch):
    """A window whose chapter_id is not in the structure → empty skeleton."""
    monkeypatch.setenv(ENV_SYNTHESIS_SKELETON, "1")
    p = TextbookSynthesisProvider(
        provider="anthropic",
        anthropic_client=_FakeAnthropicClient([]),
        textbook_structure=_structure_with_headings(),
    )
    assert p._build_window_heading_skeleton("ch99") == ""


def test_skeleton_deepest_level_dropped_first_when_over_budget(monkeypatch):
    """Over-budget skeleton drops the DEEPEST heading level first; the chapter
    + section spine survives."""
    monkeypatch.setenv(ENV_SYNTHESIS_SKELETON, "1")
    # Build a chapter whose deepest (level-3) leaves blow the token budget but
    # whose level-1/2 spine fits.
    deep_subs = [
        {"headingLevel": 3, "headingText": "Leaf " + ("x" * 200) + f" {i}"}
        for i in range(40)
    ]
    structure = {
        "chapters": [
            {
                "id": "ch1",
                "headingLevel": 1,
                "headingText": "Chapter One Spine",
                "sections": [
                    {
                        "headingLevel": 2,
                        "headingText": "Section A Spine",
                        "subsections": deep_subs,
                    }
                ],
            }
        ]
    }
    p = TextbookSynthesisProvider(
        provider="anthropic",
        anthropic_client=_FakeAnthropicClient([]),
        textbook_structure=structure,
    )
    skeleton = p._build_window_heading_skeleton("ch1")
    budget_chars = int(
        tsp._SKELETON_HEADING_TOKEN_BUDGET
        * tsp._SKELETON_HEADING_CHARS_PER_TOKEN
    )
    assert len(skeleton) <= budget_chars
    # The coarse spine survives; the deep leaves are shed.
    assert "Chapter One Spine" in skeleton
    assert "Section A Spine" in skeleton
    assert "Leaf" not in skeleton


def test_skeleton_chapter_heading_never_dropped(monkeypatch):
    """Even a pathologically huge single chapter heading is never dropped (a
    single surviving level breaks the truncation loop)."""
    monkeypatch.setenv(ENV_SYNTHESIS_SKELETON, "1")
    structure = {
        "chapters": [
            {"id": "ch1", "headingLevel": 1, "headingText": "H" * 9000}
        ]
    }
    p = TextbookSynthesisProvider(
        provider="anthropic",
        anthropic_client=_FakeAnthropicClient([]),
        textbook_structure=structure,
    )
    skeleton = p._build_window_heading_skeleton("ch1")
    assert "H" * 9000 in skeleton


def test_skeleton_flag_on_synthesize_window_still_cites_chunks(monkeypatch):
    """End-to-end: flag on, the window synthesis still resolves source_chunk_ids
    from the allowed set (citation path unchanged by the skeleton)."""
    monkeypatch.setenv(ENV_SYNTHESIS_SKELETON, "1")
    p = TextbookSynthesisProvider(
        provider="anthropic",
        anthropic_client=_FakeAnthropicClient(
            [json.dumps(_window_payload(["c1", "c9"]))]
        ),
        textbook_structure=_structure_with_headings(),
    )
    out = p.synthesize_window_objectives(
        _window(["c1", "c2"], chapter_id="ch1"), course_name="ALG_101",
    )
    objs = out["candidate_objectives"]
    assert len(objs) == 1
    # c9 (out of set) stripped; c1 kept — the ⊆ enforcement is untouched.
    assert objs[0]["source_chunk_ids"] == ["c1"]
    assert objs[0]["grounded_citation"] is True


def test_resolve_synthesis_skeleton_parse_with_fallback(monkeypatch):
    for truthy in ("1", "true", "TRUE", "Yes", "on"):
        monkeypatch.setenv(ENV_SYNTHESIS_SKELETON, truthy)
        assert tsp._resolve_synthesis_skeleton() is True
    for falsey in ("0", "false", "no", "off", "", "garbage"):
        monkeypatch.setenv(ENV_SYNTHESIS_SKELETON, falsey)
        assert tsp._resolve_synthesis_skeleton() is False
    monkeypatch.delenv(ENV_SYNTHESIS_SKELETON, raising=False)
    assert tsp._resolve_synthesis_skeleton() is False
