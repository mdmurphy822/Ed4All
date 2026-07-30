"""Bloom-ladder initiative WI-14 — trivote voter-2 fine-tuned-heads backend.

Covers the ``ED4ALL_BLOOM_TRIVOTE_HEADS`` swap end to end:

- :mod:`lib.classifiers.bloom_zero_shot` — the new ``resolve_trivote_
  heads_mode`` flag resolver (parse-with-fallback) and the new
  ``heads_vote`` backend function (empty-text / unavailable-heads /
  classify-raises abstention, floor=0.0 pass-through so the module's own
  higher internal floor never double-abstains).
- :class:`lib.validators.bloom.classifier_disagreement.
  BloomClassifierDisagreementValidator._validate_trivote` — the backend
  swap at exactly the ``zs_level``/``zs_conf`` call site: flag+live-heads
  ON uses the heads backend and NEVER touches the injected NLI stub;
  flag ON but no live heads degrades silently back to zero-shot; flag OFF
  never even resolves a heads instance (byte-identical to before this
  work item). The existing 0.4 confidence-floor abstention region and the
  decision-capture wiring are exercised unchanged, regardless of which
  backend produced the vote.

No real model weights are loaded anywhere in this suite — every
``BloomDebertaHeads`` surface is a stub or explicitly patched to return
``None`` (the real loader already degrades to ``None`` on a fresh
checkout with no shipped ``models/bloom_classifiers`` weights, but the
tests pin that explicitly rather than depending on the ambient
filesystem state).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Repo root on path for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Block lives at Courseforge/scripts/blocks.py — import bridge mirror.
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.classifiers.bloom_deberta_heads import BloomDebertaHeads  # noqa: E402
from lib.classifiers.bloom_zero_shot import (  # noqa: E402
    _BLOOM_HYPOTHESIS_TEMPLATES,
    heads_vote,
    resolve_trivote_heads_mode,
)
from lib.validators.bloom.classifier_disagreement import (  # noqa: E402
    BloomClassifierDisagreementValidator,
)


# --------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------- #


class _StubHeads:
    """Drop-in test stub for ``BloomDebertaHeads``.

    Records every ``classify`` invocation (text + floor kwarg) so tests
    can assert the ``floor=0.0`` pass-through contract, and returns a
    per-text scripted response (falls back to ``default``).
    """

    def __init__(
        self,
        response_map: Optional[Dict[str, Optional[Tuple[str, float]]]] = None,
        default: Optional[Tuple[str, float]] = None,
    ) -> None:
        self.response_map = response_map or {}
        self.default = default
        self.calls: List[Tuple[str, Optional[float]]] = []

    def classify(
        self, text: str, *, floor: Optional[float] = None
    ) -> Optional[Tuple[str, float]]:
        self.calls.append((text, floor))
        if text in self.response_map:
            return self.response_map[text]
        return self.default


class _FakeNliScore:
    def __init__(self, entailment: float) -> None:
        self.entailment = entailment
        self.neutral = 0.0
        self.contradiction = 0.0


class _TargetNli:
    """High entailment for ``target_level``'s hypothesis; low otherwise."""

    def __init__(self, target_level: str, high: float = 0.9) -> None:
        self.target_level = target_level
        self.high = high
        self.calls: List[List[Tuple[str, str]]] = []

    def score_batch(self, *, pairs, batch_size=None):  # noqa: ANN001
        self.calls.append(list(pairs))
        target_hyp = _BLOOM_HYPOTHESIS_TEMPLATES[self.target_level]
        return [
            _FakeNliScore(self.high if hyp == target_hyp else 0.1)
            for _p, hyp in pairs
        ]


_NOVERB = "The weather is nice today with clouds overhead."


def _make_assessment_block(
    *,
    block_id: str = "page_01#assessment_item_quiz_0",
    statement: str = "Identify the main themes of the passage.",
    bloom_level: Optional[str] = "remember",
) -> Block:
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id="page_01",
        sequence=0,
        content={"statement": statement},
        bloom_level=bloom_level,
    )


@pytest.fixture(autouse=True)
def _reset_heads_singleton() -> None:
    """Isolate from any real ``BloomDebertaHeads`` singleton state.

    Tests in this module either inject a stub heads instance directly
    (bypassing ``get_or_load`` entirely) or explicitly patch
    ``get_or_load``, so no test should ever depend on cross-module
    singleton caching from ``test_bloom_deberta_heads.py``.
    """
    BloomDebertaHeads._reset_for_tests()
    yield
    BloomDebertaHeads._reset_for_tests()


@pytest.fixture
def _trivote_heads_on(monkeypatch: pytest.MonkeyPatch):
    """Both master + heads-backend flags on; strict mode off."""
    monkeypatch.setenv("ED4ALL_BLOOM_TRIVOTE", "1")
    monkeypatch.setenv("ED4ALL_BLOOM_TRIVOTE_HEADS", "1")
    monkeypatch.delenv("TRAINFORGE_REQUIRE_BERT_ENSEMBLE", raising=False)
    yield


# --------------------------------------------------------------------- #
# resolve_trivote_heads_mode — parse-with-fallback
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True),
        ("on", True), ("On", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("garbage", False), ("", False),
    ],
)
def test_resolve_trivote_heads_mode(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    monkeypatch.setenv("ED4ALL_BLOOM_TRIVOTE_HEADS", value)
    assert resolve_trivote_heads_mode() is expected


def test_resolve_trivote_heads_mode_default_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ED4ALL_BLOOM_TRIVOTE_HEADS", raising=False)
    assert resolve_trivote_heads_mode() is False


# --------------------------------------------------------------------- #
# heads_vote — the voter-2 backend function
# --------------------------------------------------------------------- #


def test_heads_vote_empty_text_abstains() -> None:
    heads = _StubHeads(default=("apply", 0.9))
    assert heads_vote("", heads=heads) is None
    assert heads_vote("   ", heads=heads) is None
    assert heads.calls == []


def test_heads_vote_no_heads_injected_and_singleton_unavailable_abstains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``heads=`` injected + ``get_or_load()`` returns None -> abstain."""
    monkeypatch.setattr(
        BloomDebertaHeads, "get_or_load", classmethod(lambda cls: None)
    )
    assert heads_vote("Solve the equation.", heads=None) is None


def test_heads_vote_get_or_load_raises_abstains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(cls: Any) -> Any:
        raise RuntimeError("simulated loader failure")

    monkeypatch.setattr(BloomDebertaHeads, "get_or_load", classmethod(_boom))
    assert heads_vote("Solve the equation.", heads=None) is None


def test_heads_vote_pass_through_level_and_confidence() -> None:
    """A live heads winner passes straight through as (level, confidence)."""
    text = "Design a completely new experimental protocol."
    heads = _StubHeads(response_map={text: ("create", 0.8123456)})
    result = heads_vote(text, heads=heads)
    assert result == ("create", 0.8123)  # rounded 4dp, mirrors zero_shot_bloom


def test_heads_vote_calls_classify_with_floor_zero() -> None:
    """floor=0.0 is passed through — abstention is the CALLER's job, not
    the heads module's own (different, 0.5) internal floor."""
    text = "Recall the definition of single sign-on."
    heads = _StubHeads(response_map={text: ("remember", 0.42)})
    heads_vote(text, heads=heads)
    assert heads.calls == [(text, 0.0)]


def test_heads_vote_none_result_abstains() -> None:
    """The heads backend's own None (its 0.0-floor call still can return
    None if the underlying classify implementation chooses to) propagates
    as an abstention."""
    text = "Ambiguous text."
    heads = _StubHeads(response_map={text: None})
    assert heads_vote(text, heads=heads) is None


def test_heads_vote_classify_raises_abstains() -> None:
    class _BoomHeads:
        def classify(self, text: str, *, floor: Optional[float] = None) -> Any:
            raise RuntimeError("simulated head failure")

    assert heads_vote("Analyze the passage.", heads=_BoomHeads()) is None


# --------------------------------------------------------------------- #
# Validator wiring — ED4ALL_BLOOM_TRIVOTE_HEADS backend swap
# --------------------------------------------------------------------- #


def test_trivote_heads_backend_used_when_flag_on_and_heads_live(
    _trivote_heads_on,
) -> None:
    """Flag ON + live heads -> voter 2 uses the heads backend; the
    injected NLI stub (which would flip the verdict if consulted) is
    NEVER called."""
    text = "Some task text for the student."
    heads = _StubHeads(response_map={text: ("apply", 0.9)})
    # Would score "create" high if consulted — proves the NLI path is
    # bypassed when the heads backend is live.
    nli = _TargetNli("create")
    validator = BloomClassifierDisagreementValidator(nli=nli, heads=heads)
    block = _make_assessment_block(statement=text, bloom_level="apply")
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert result.issues == []
    assert heads.calls == [(text, 0.0)]
    assert nli.calls == []


def test_trivote_heads_backend_disagreement_emits_regenerate(
    _trivote_heads_on,
) -> None:
    """Heads backend winner disagrees with the declared level (above the
    0.4 confidence floor) -> BERT_ENSEMBLE_DISAGREEMENT, same as the
    zero-shot backend's contract."""
    heads = _StubHeads(default=("create", 0.85))
    nli = _TargetNli("understand")  # would disagree differently if used
    validator = BloomClassifierDisagreementValidator(nli=nli, heads=heads)
    block = _make_assessment_block(statement=_NOVERB, bloom_level="remember")
    result = validator.validate({"blocks": [block]})

    assert result.action == "regenerate"
    codes = {i.code for i in result.issues}
    assert "BERT_ENSEMBLE_DISAGREEMENT" in codes
    assert nli.calls == []


def test_trivote_heads_backend_below_confidence_floor_abstains(
    _trivote_heads_on,
) -> None:
    """A heads-backend winner below the 0.4 confidence floor abstains —
    the SAME abstention region the zero-shot backend uses (:89 in
    classifier_disagreement.py), verbatim, regardless of backend."""
    heads = _StubHeads(default=("create", 0.39))  # just under the 0.4 floor
    validator = BloomClassifierDisagreementValidator(heads=heads)
    # verb also abstains (_NOVERB) -> only asserted remains -> structured skip.
    block = _make_assessment_block(statement=_NOVERB, bloom_level="remember")
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert not any(
        i.code == "BERT_ENSEMBLE_DISAGREEMENT" for i in result.issues
    )
    assert any(
        i.code == "BLOOM_TRIVOTE_INSUFFICIENT_VOTERS" for i in result.issues
    )


def test_trivote_heads_flag_on_but_unavailable_falls_back_to_zero_shot(
    _trivote_heads_on,
) -> None:
    """Flag ON but no live heads instance (missing/unloadable weights) ->
    silently falls back to the zero-shot NLI backend; a build never fails
    on absent weights."""
    monkeypatch_target = BloomDebertaHeads
    original = monkeypatch_target.get_or_load
    try:
        monkeypatch_target.get_or_load = classmethod(lambda cls: None)
        nli = _TargetNli("apply")
        # No heads= injected -> _resolve_heads reaches get_or_load(),
        # which is patched to return None (degrade).
        validator = BloomClassifierDisagreementValidator(nli=nli)
        block = _make_assessment_block(statement=_NOVERB, bloom_level="apply")
        result = validator.validate({"blocks": [block]})
    finally:
        monkeypatch_target.get_or_load = original

    assert result.passed
    assert result.action is None
    # The zero-shot voter was actually consulted (6 hypothesis pairs).
    assert len(nli.calls) == 1
    assert len(nli.calls[0]) == 6


def test_trivote_heads_flag_off_never_resolves_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ED4ALL_BLOOM_TRIVOTE on but ED4ALL_BLOOM_TRIVOTE_HEADS off (or
    unset) -> voter 2 stays on zero-shot; an injected heads stub (which
    would flip the verdict if consulted) is NEVER called. Byte-identical
    to the pre-WI-14 trivote behavior."""
    monkeypatch.setenv("ED4ALL_BLOOM_TRIVOTE", "1")
    monkeypatch.delenv("ED4ALL_BLOOM_TRIVOTE_HEADS", raising=False)
    monkeypatch.delenv("TRAINFORGE_REQUIRE_BERT_ENSEMBLE", raising=False)

    text = "Some task text for the student."
    heads = _StubHeads(response_map={text: ("create", 0.95)})  # would disagree
    nli = _TargetNli("apply")
    validator = BloomClassifierDisagreementValidator(nli=nli, heads=heads)
    block = _make_assessment_block(statement=text, bloom_level="apply")
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert result.issues == []
    assert heads.calls == []
    assert len(nli.calls) == 1


def test_trivote_master_flag_off_never_resolves_heads_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both flags unset entirely -> the legacy BERT-ensemble path runs;
    neither the injected NLI nor the injected heads voter is ever
    consulted. This is the ROOT byte-identical-when-unset guarantee."""
    monkeypatch.delenv("ED4ALL_BLOOM_TRIVOTE", raising=False)
    monkeypatch.delenv("ED4ALL_BLOOM_TRIVOTE_HEADS", raising=False)

    class _StubEnsemble:
        def __init__(self) -> None:
            self.classify_calls: List[str] = []

        def _load_members(self) -> List[Any]:
            return [object()]

        def classify(self, text: str) -> Dict[str, Any]:
            self.classify_calls.append(text)
            return {
                "winner_level": "remember",
                "winner_score": 0.9,
                "dispersion": 0.1,
                "per_member": [("remember", 0.9)],
            }

    ensemble = _StubEnsemble()
    heads = _StubHeads(response_map={"x": ("create", 0.95)})
    nli = _TargetNli("create")
    text = "Recall the definition of single sign-on."
    validator = BloomClassifierDisagreementValidator(
        ensemble=ensemble, nli=nli, heads=heads
    )
    block = _make_assessment_block(statement=text, bloom_level="remember")
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert ensemble.classify_calls == [text]
    assert nli.calls == []
    assert heads.calls == []


def test_trivote_strict_mode_heads_live_avoids_critical_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode + NLI unavailable BUT a live heads backend -> voter 2
    is satisfied by the heads backend, so the gate does NOT fail closed
    (the legacy strict-raise only fires when NEITHER backend is up)."""
    monkeypatch.setenv("ED4ALL_BLOOM_TRIVOTE", "1")
    monkeypatch.setenv("ED4ALL_BLOOM_TRIVOTE_HEADS", "1")
    monkeypatch.setenv("TRAINFORGE_REQUIRE_BERT_ENSEMBLE", "true")

    heads = _StubHeads(default=("apply", 0.9))
    validator = BloomClassifierDisagreementValidator(nli=None, heads=heads)
    block = _make_assessment_block(statement=_NOVERB, bloom_level="apply")
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert heads.calls


def test_trivote_decision_capture_fires_with_heads_backend(
    _trivote_heads_on,
) -> None:
    """One capture per evaluated block, rationale citing the heads-backend
    vote under the same ``zero_shot=`` field label the existing capture
    contract uses (backend-agnostic rationale, unchanged capture path)."""

    class _StubCapture:
        def __init__(self) -> None:
            self.calls: List[Tuple[str, str, str]] = []

        def log_decision(self, decision_type, decision, rationale, **kwargs):
            self.calls.append((decision_type, decision, rationale))

    text = "Some task text for the student."
    heads = _StubHeads(response_map={text: ("apply", 0.9)})
    capture = _StubCapture()
    validator = BloomClassifierDisagreementValidator(heads=heads)
    block = _make_assessment_block(statement=text, bloom_level="apply")
    validator.validate({"blocks": [block], "decision_capture": capture})

    assert len(capture.calls) == 1
    decision_type, decision, rationale = capture.calls[0]
    assert decision_type == "bloom_classifier_disagreement_check"
    assert decision == "passed"
    assert len(rationale) >= 20
    assert "asserted=apply" in rationale
    assert "zero_shot=apply" in rationale
