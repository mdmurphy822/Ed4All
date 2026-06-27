"""Phase 3: rollback-flagged cutover to the LLM-agnostic SynthesisProvider.

The instruction/preference factory dispatch arms (and run_synthesis, exercised
by ``test_synthesize_training.py``) construct the LLM-agnostic
:class:`SynthesisProvider` for the OpenAI-compatible seats (``local`` /
``together``) when ``TRAINFORGE_AGNOSTIC_SYNTHESIS`` is ON (the default), and
fall back to the per-vendor leaves (``LocalSynthesisProvider`` /
``TogetherSynthesisProvider``) when it is OFF (the rollback path).

These tests assert WHICH class the factory constructs (the cutover seam) +
the ``terse_prompts`` mapping (local terse, hosted verbose) — they do not
re-prove paraphrase byte-parity (``test_synthesis_provider.py`` owns that).
The ``anthropic`` arm always constructs ``AnthropicSynthesisProvider``
regardless of the flag (P4 will handle it); ``claude_session`` is never
lazily constructed by the factory at all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators._synthesis_provider import (  # noqa: E402
    ENV_AGNOSTIC_SYNTHESIS,
    agnostic_synthesis_enabled,
)
from Trainforge.generators.instruction_factory import (  # noqa: E402
    synthesize_instruction_pair,
)
from Trainforge.generators.preference_factory import (  # noqa: E402
    synthesize_preference_pair,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "mini_course_training"


# ---------------------------------------------------------------------------
# Resolver — default ON; only explicit falsey tokens disable.
# ---------------------------------------------------------------------------


def test_resolver_default_on(monkeypatch):
    monkeypatch.delenv(ENV_AGNOSTIC_SYNTHESIS, raising=False)
    assert agnostic_synthesis_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "FALSE", "Off", " no "])
def test_resolver_falsey_tokens_disable(monkeypatch, val):
    monkeypatch.setenv(ENV_AGNOSTIC_SYNTHESIS, val)
    assert agnostic_synthesis_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "garbage", ""])
def test_resolver_truthy_or_garbage_enables(monkeypatch, val):
    monkeypatch.setenv(ENV_AGNOSTIC_SYNTHESIS, val)
    assert agnostic_synthesis_enabled() is True


# ---------------------------------------------------------------------------
# Construction spies — record (tag, kwargs) per provider class built.
# ---------------------------------------------------------------------------


def _install_spies(monkeypatch) -> List[Tuple[str, Dict[str, Any]]]:
    """Patch every construction symbol the factory dispatch reaches with a
    pass-through recorder. Returns the shared record list.

    The agnostic seat now routes through the shared
    ``build_synthesis_provider`` builder (the #5 dedup), so the spy records
    the BUILDER call (``("agnostic", {"provider": ...})``) rather than the
    direct ``SynthesisProvider(...)`` construction. The per-leaf
    ``terse_prompts`` / ``preserve_tokens_enabled`` / 60s-timeout mapping is
    owned + proved by the builder unit tests in
    ``test_synthesis_provider.py``."""
    record: List[Tuple[str, Dict[str, Any]]] = []

    class _Rec:
        def __init__(self, *args, **kwargs):
            pass

        def paraphrase_instruction(self, draft, chunk, *, preserve_tokens=None):
            return draft

        def paraphrase_preference(self, draft, chunk, *, preserve_tokens=None):
            return draft

    def _leaf_recorder(tag: str):
        class _Leaf(_Rec):
            def __init__(self, **kwargs):
                record.append((tag, dict(kwargs)))
                super().__init__()

        return _Leaf

    def _builder_recorder(provider, **kwargs):
        record.append(("agnostic", {"provider": provider, **kwargs}))
        return _Rec()

    from Trainforge.generators import _synthesis_provider as sp_mod
    from Trainforge.generators import _local_provider as lp_mod
    from Trainforge.generators import _together_provider as tp_mod
    from Trainforge.generators import _anthropic_provider as ap_mod

    monkeypatch.setattr(sp_mod, "build_synthesis_provider", _builder_recorder)
    monkeypatch.setattr(lp_mod, "LocalSynthesisProvider", _leaf_recorder("local_leaf"))
    monkeypatch.setattr(tp_mod, "TogetherSynthesisProvider", _leaf_recorder("together_leaf"))
    monkeypatch.setattr(ap_mod, "AnthropicSynthesisProvider", _leaf_recorder("anthropic_leaf"))
    return record


def _instruction_chunk() -> Dict[str, Any]:
    """Minimal chunk that reaches the factory dispatch (mirrors the
    test_instruction_factory ``_make_chunk`` shape)."""
    return {
        "id": "chunk_cutover_instr",
        "text": "Some pedagogical material about a focused subject area.",
        "summary": (
            "A short paraphrased summary of the chunk that does not echo the "
            "chunk text verbatim and stays well clear of any verbatim span."
        ),
        "learning_outcome_refs": ["TO-01"],
        "concept_tags": ["focused-subject"],
        "bloom_level": "understand",
        "content_type_label": "explanation",
    }


def _preference_chunk() -> Dict[str, Any]:
    """A real misconception-bearing fixture chunk so the preference factory
    produces a pair and reaches its dispatch arm."""
    import json

    chunks = [
        json.loads(line)
        for line in (FIXTURE_ROOT / "corpus" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for c in chunks:
        if c.get("id") == "chunk_mc_02":
            return c
    raise AssertionError("fixture chunk_mc_02 not found")


# ---------------------------------------------------------------------------
# Instruction factory dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["local", "together"])
def test_instruction_flag_on_constructs_agnostic(monkeypatch, provider):
    monkeypatch.delenv(ENV_AGNOSTIC_SYNTHESIS, raising=False)  # default ON
    record = _install_spies(monkeypatch)

    result = synthesize_instruction_pair(
        _instruction_chunk(), seed=11, provider=provider,
    )
    assert result.pair is not None
    # The factory routes the agnostic seat through the shared builder.
    assert ("agnostic", {"provider": provider}) in record
    # The leaf was NOT constructed.
    assert not any(tag.endswith("_leaf") for tag, _ in record)


@pytest.mark.parametrize(
    "provider,expected_tag",
    [("local", "local_leaf"), ("together", "together_leaf")],
)
def test_instruction_flag_off_constructs_leaf(monkeypatch, provider, expected_tag):
    monkeypatch.setenv(ENV_AGNOSTIC_SYNTHESIS, "0")  # rollback
    record = _install_spies(monkeypatch)

    result = synthesize_instruction_pair(
        _instruction_chunk(), seed=11, provider=provider,
    )
    assert result.pair is not None
    assert (expected_tag, {}) in record
    assert not any(tag == "agnostic" for tag, _ in record)


@pytest.mark.parametrize("flag", [None, "0"])
def test_instruction_anthropic_always_constructs_leaf(monkeypatch, flag):
    """The anthropic arm is untouched by the cutover — constructs
    AnthropicSynthesisProvider regardless of the flag (P4 owns it)."""
    if flag is None:
        monkeypatch.delenv(ENV_AGNOSTIC_SYNTHESIS, raising=False)
    else:
        monkeypatch.setenv(ENV_AGNOSTIC_SYNTHESIS, flag)
    record = _install_spies(monkeypatch)

    result = synthesize_instruction_pair(
        _instruction_chunk(), seed=11, provider="anthropic",
    )
    assert result.pair is not None
    assert ("anthropic_leaf", {}) in record
    assert not any(tag == "agnostic" for tag, _ in record)


# ---------------------------------------------------------------------------
# Preference factory dispatch (same dispatch code path)
# ---------------------------------------------------------------------------


def test_preference_flag_on_constructs_agnostic_local(monkeypatch):
    monkeypatch.delenv(ENV_AGNOSTIC_SYNTHESIS, raising=False)  # default ON
    record = _install_spies(monkeypatch)

    result = synthesize_preference_pair(_preference_chunk(), seed=7, provider="local")
    assert result.pair is not None
    assert ("agnostic", {"provider": "local"}) in record


def test_preference_flag_off_constructs_leaf_together(monkeypatch):
    monkeypatch.setenv(ENV_AGNOSTIC_SYNTHESIS, "off")  # rollback
    record = _install_spies(monkeypatch)

    result = synthesize_preference_pair(_preference_chunk(), seed=7, provider="together")
    assert result.pair is not None
    assert ("together_leaf", {}) in record
    assert not any(tag == "agnostic" for tag, _ in record)
