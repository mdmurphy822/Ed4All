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
    _LOCAL_NUM_CTX,
    _SKELETON_CHAR_BUDGET,
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
