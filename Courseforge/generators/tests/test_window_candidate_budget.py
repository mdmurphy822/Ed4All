"""Per-window candidate budget (ED4ALL_OBJECTIVE_WINDOW_MAX_CANDIDATES).

Pins the vendor-depth Stage-2 candidate-budget surface:

- ``resolve_window_max_candidates`` parse-with-fallback (env int > per-section
  default 12 > ``None`` legacy);
- ``_window_objectives_schema`` identity-when-None / budget-adjusted maxItems;
- the resolved budget reaches BOTH the synthesis prompt ("Synthesize 1-N")
  and the parse-side schema cap (an over-budget payload is rejected);
- the flag-off prompt stays byte-identical ("Synthesize 1-3", no Section line);
- a per-section window's ``section_label`` reaches the prompt header.
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

from Courseforge.generators.outline._textbook_synthesis_provider import (  # noqa: E402
    ENV_WINDOW_MAX_CANDIDATES,
    TextbookSynthesisProvider,
    TextbookSynthesisProviderError,
    _PER_SECTION_DEFAULT_MAX_CANDIDATES,
    _WINDOW_OBJECTIVES_SCHEMA,
    _window_objectives_schema,
    resolve_window_max_candidates,
)

_PER_SECTION_ENV = "ED4ALL_OBJECTIVE_WINDOW_PER_SECTION"


# ---------------------------------------------------------------------------
# Mock anthropic client (mirrors test_textbook_synthesis_provider.py)
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        text = self._responses[idx] if self._responses else ""
        return {"content": [{"type": "text", "text": text}]}


class _FakeAnthropicClient:
    def __init__(self, responses: List[str]) -> None:
        self.messages = _FakeMessages(responses)


def _provider(responses: List[str]) -> TextbookSynthesisProvider:
    return TextbookSynthesisProvider(
        provider="anthropic",
        anthropic_client=_FakeAnthropicClient(responses),
    )


def _window(chunk_ids, *, section_label: str = "") -> dict:
    wd = {
        "chapter_id": "ch1",
        "window_index": 0,
        "chunk_ids": list(chunk_ids),
        "chunks": [{"id": c, "text": f"body for {c}"} for c in chunk_ids],
        "join_method": "sourceid",
        "estimated_prompt_tokens": 100,
    }
    if section_label:
        wd["section_label"] = section_label
    return wd


def _payload(n_candidates: int, chunk_id: str = "c1") -> dict:
    return {
        "candidate_objectives": [
            {
                "statement": f"Explain distinct idea number {i} of the window.",
                "bloom_level": "understand",
                "bloom_verb": "explain",
                "source_chunk_ids": [chunk_id],
                "sub_objectives": [],
            }
            for i in range(n_candidates)
        ]
    }


def _sent_prompts(p: TextbookSynthesisProvider) -> str:
    return json.dumps(p._anthropic_client.messages.calls)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_WINDOW_MAX_CANDIDATES, raising=False)
    monkeypatch.delenv(_PER_SECTION_ENV, raising=False)


# ===========================================================================
# resolve_window_max_candidates — parse-with-fallback
# ===========================================================================


def test_env_int_wins(monkeypatch):
    monkeypatch.setenv(ENV_WINDOW_MAX_CANDIDATES, "20")
    assert resolve_window_max_candidates() == 20


def test_env_int_wins_even_with_per_section_on(monkeypatch):
    monkeypatch.setenv(ENV_WINDOW_MAX_CANDIDATES, "7")
    monkeypatch.setenv(_PER_SECTION_ENV, "1")
    assert resolve_window_max_candidates() == 7


def test_unset_and_per_section_off_is_legacy_none():
    assert resolve_window_max_candidates() is None


def test_unset_and_per_section_on_defaults_12(monkeypatch):
    monkeypatch.setenv(_PER_SECTION_ENV, "true")
    assert resolve_window_max_candidates() == _PER_SECTION_DEFAULT_MAX_CANDIDATES
    assert resolve_window_max_candidates() == 12


def test_garbage_env_falls_through(monkeypatch):
    for tok in ("banana", "0", "-3", "1.5"):
        monkeypatch.setenv(ENV_WINDOW_MAX_CANDIDATES, tok)
        assert resolve_window_max_candidates() is None, tok
        monkeypatch.setenv(_PER_SECTION_ENV, "1")
        assert resolve_window_max_candidates() == 12, tok
        monkeypatch.delenv(_PER_SECTION_ENV, raising=False)


# ===========================================================================
# _window_objectives_schema
# ===========================================================================


def test_schema_none_is_module_identity():
    assert _window_objectives_schema(None) is _WINDOW_OBJECTIVES_SCHEMA


def test_schema_budget_sets_max_items_without_mutating_module():
    s = _window_objectives_schema(7)
    assert s["properties"]["candidate_objectives"]["maxItems"] == 7
    # The module-level legacy schema is untouched (deep copy).
    assert (
        _WINDOW_OBJECTIVES_SCHEMA["properties"]["candidate_objectives"][
            "maxItems"
        ]
        == 12
    )


# ===========================================================================
# Budget reaches the prompt AND the parse-side cap
# ===========================================================================


def test_budget_reaches_prompt(monkeypatch):
    monkeypatch.setenv(ENV_WINDOW_MAX_CANDIDATES, "12")
    p = _provider([json.dumps(_payload(1))])
    p.synthesize_window_objectives(_window(["c1"]), course_name="TST_901")
    sent = _sent_prompts(p)
    assert "Synthesize 1-12 candidate learning" in sent
    assert "Synthesize 1-3 candidate" not in sent


def test_legacy_prompt_byte_identical_when_unset():
    p = _provider([json.dumps(_payload(1))])
    p.synthesize_window_objectives(_window(["c1"]), course_name="TST_901")
    sent = _sent_prompts(p)
    assert (
        "Synthesize 1-3 candidate learning objectives this window's chunks "
        "teach." in json.loads(sent)[0]["messages"][0]["content"]
    )
    assert "Section:" not in sent
    assert "do not stop at the first few" not in sent


def test_over_budget_payload_rejected_by_schema(monkeypatch):
    """Budget 2 + a 3-candidate payload → schema maxItems rejects → the
    parse-retry budget exhausts (prompt and parse cap AGREE)."""
    monkeypatch.setenv(ENV_WINDOW_MAX_CANDIDATES, "2")
    p = _provider([json.dumps(_payload(3))])
    with pytest.raises(TextbookSynthesisProviderError) as exc:
        p.synthesize_window_objectives(_window(["c1"]), course_name="TST_901")
    assert exc.value.code == "chapter_objectives_exhausted"


def test_budget_12_accepts_12_candidates(monkeypatch):
    monkeypatch.setenv(ENV_WINDOW_MAX_CANDIDATES, "12")
    p = _provider([json.dumps(_payload(12))])
    out = p.synthesize_window_objectives(_window(["c1"]), course_name="TST_901")
    assert len(out["candidate_objectives"]) == 12


def test_per_section_env_defaults_budget_into_prompt(monkeypatch):
    """Per-section mode alone (no explicit budget env) drives the 12 default
    into the prompt — the run-env only needs the one flag."""
    monkeypatch.setenv(_PER_SECTION_ENV, "1")
    p = _provider([json.dumps(_payload(1))])
    p.synthesize_window_objectives(_window(["c1"]), course_name="TST_901")
    assert "Synthesize 1-12 candidate learning" in _sent_prompts(p)


# ===========================================================================
# Section label reaches the prompt header
# ===========================================================================


def test_section_label_reaches_prompt(monkeypatch):
    monkeypatch.setenv(_PER_SECTION_ENV, "1")
    p = _provider([json.dumps(_payload(1))])
    p.synthesize_window_objectives(
        _window(["c1"], section_label="1.2 Use the Language of Algebra"),
        course_name="TST_901",
    )
    sent = json.loads(_sent_prompts(p))[0]["messages"][0]["content"]
    assert "Section: 1.2 Use the Language of Algebra\n" in sent


def test_no_section_line_without_label(monkeypatch):
    monkeypatch.setenv(_PER_SECTION_ENV, "1")
    p = _provider([json.dumps(_payload(1))])
    p.synthesize_window_objectives(_window(["c1"]), course_name="TST_901")
    sent = json.loads(_sent_prompts(p))[0]["messages"][0]["content"]
    assert "Section:" not in sent
