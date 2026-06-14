"""W4 — ``BlockProseEntailmentValidator`` tests.

Covers the per-block full-prose NLI entailment gate. Injects a deterministic
fake NLI via the ``nli=...`` test seam (threaded into
``score_groundedness(nli=...)``) so CI fresh-checkout needs no DeBERTa load.

Test cases (per ``plans/finegrain/w4-nli-grounding-gate.md`` §5.1):

1. Keystone: valid copied source-ref on FABRICATED on-topic prose that ALSO
   passes cosine → NLI gate FAILS it (BLOCK_PROSE_UNGROUNDED, regenerate).
2. Faithful paraphrase PASSES.
3. Contradicted block → BLOCK_PROSE_CONTRADICTED standalone.
4. Graceful-degrade (NLI absent → warning passed=True; strict → raise).
5. Shadow path computes without gating.
6. Decision-capture fires with dynamic signals.
7. Skip set (assessment_item/objective skipped; example/self_check scored).
8. No-grounding-source → warning, passed=True.
9. Empty/computational-only block → no-op pass.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Validator wraps the [embedding] NLI deps (DeBERTa-v3 via the singleton);
# tests stub the scorer today but the file is slow-marked so the nightly
# extras-installed run picks them up via -m "slow".
pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.classifiers.nli_classifier import NliScore  # noqa: E402
from lib.validators.block_prose_entailment import (  # noqa: E402
    BlockProseEntailmentValidator,
    _CODE_BLOCK_PROSE_CONTRADICTED,
    _CODE_BLOCK_PROSE_NO_GROUNDING_SOURCE,
    _CODE_BLOCK_PROSE_UNGROUNDED,
    _CODE_NLI_DEPS_MISSING,
    _DECISION_TYPE,
)


# --------------------------------------------------------------------- #
# Fake NLI — deterministic per-sentence verdict via embedded markers
# --------------------------------------------------------------------- #


class _FakeNli:
    """Deterministic NLI stub.

    ``score_batch(pairs)`` returns one NliScore per (premise, hypothesis):

    * hypothesis containing ``[ENTAIL]`` → high entailment.
    * hypothesis containing ``[CONTRA]`` → high contradiction.
    * otherwise (``[FAB]`` or unmarked) → low entailment + low contradiction
      (unsupported — the fabricated-on-topic case NLI catches).

    Exposes ``_revision`` + ``device`` so ``score_groundedness`` can record
    provenance pins.
    """

    _revision = "fake-nli-rev-1"
    device = "cpu"

    def __init__(self) -> None:
        self.batch_calls: List[List[Tuple[str, str]]] = []

    def score_batch(self, *, pairs: List[Tuple[str, str]]) -> List[NliScore]:
        self.batch_calls.append(list(pairs))
        out: List[NliScore] = []
        for _premise, hypothesis in pairs:
            if "[ENTAIL]" in hypothesis:
                out.append(NliScore(entailment=0.95, neutral=0.03, contradiction=0.02))
            elif "[CONTRA]" in hypothesis:
                out.append(NliScore(entailment=0.05, neutral=0.10, contradiction=0.85))
            else:
                out.append(NliScore(entailment=0.10, neutral=0.85, contradiction=0.05))
        return out


def _make_block(
    *,
    block_id: str = "page_01#concept_intro_0",
    block_type: str = "concept",
    content: Any = None,
    source_ids: Tuple[str, ...] = ("dart:slug#blk_0",),
    source_references: Optional[Tuple[Dict[str, Any], ...]] = None,
) -> Block:
    refs = source_references if source_references is not None else tuple(
        {"sourceId": sid} for sid in source_ids
    )
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content=content if content is not None else "<p>placeholder.</p>",
        source_ids=source_ids,
        source_references=refs,
    )


_CHUNK = (
    "The mitochondrion is the powerhouse of the cell and produces ATP "
    "through oxidative phosphorylation across the inner membrane."
)


# --------------------------------------------------------------------- #
# Test 1: Keystone — fabricated-on-topic prose with valid source-ref FAILS
# --------------------------------------------------------------------- #


def test_keystone_fabricated_on_topic_prose_fails_nli() -> None:
    """A block whose prose is on-topic + carries a VALID copied source-ref
    that resolves, but is NOT entailed by the cited chunk → NLI fails it
    (the exact case cosine + the resolution gate both MISS)."""
    nli = _FakeNli()
    validator = BlockProseEntailmentValidator(nli=nli)

    # Two on-topic-but-unentailed sentences (no [ENTAIL] marker) — long enough
    # to clear the content-token floor in score_groundedness.split_claims.
    prose = (
        "<p>The mitochondrion secretly manufactures cellular gold deep within "
        "its folded cristae structures every passing moment. It also quietly "
        "encodes ancient memories into the surrounding cytoplasm of the cell "
        "for safekeeping.</p>"
    )
    block = _make_block(content=prose, source_ids=("dart:slug#blk_0",))
    source_chunks = {"dart:slug#blk_0": _CHUNK}

    result = validator.validate(
        {"blocks": [block], "source_chunks": source_chunks}
    )

    ungrounded = [
        i for i in result.issues if i.code == _CODE_BLOCK_PROSE_UNGROUNDED
    ]
    assert len(ungrounded) == 1
    # Default (non-shadow) construction → critical severity, passed flips.
    assert ungrounded[0].severity == "critical"
    assert result.passed is False
    assert result.action == "regenerate"


# --------------------------------------------------------------------- #
# Test 2: Faithful paraphrase passes
# --------------------------------------------------------------------- #


def test_faithful_paraphrase_passes() -> None:
    nli = _FakeNli()
    validator = BlockProseEntailmentValidator(nli=nli)

    prose = (
        "<p>The mitochondrion is the cell's powerhouse and generates ATP "
        "through oxidative phosphorylation across its inner membrane "
        "[ENTAIL]. This energy production sustains the metabolic activity of "
        "the entire cell continuously [ENTAIL].</p>"
    )
    block = _make_block(content=prose, source_ids=("dart:slug#blk_0",))
    source_chunks = {"dart:slug#blk_0": _CHUNK}

    result = validator.validate(
        {"blocks": [block], "source_chunks": source_chunks}
    )

    assert result.passed is True
    assert result.action is None
    assert not any(
        i.code in (_CODE_BLOCK_PROSE_UNGROUNDED, _CODE_BLOCK_PROSE_CONTRADICTED)
        for i in result.issues
    )


# --------------------------------------------------------------------- #
# Test 3: Contradicted block fires standalone
# --------------------------------------------------------------------- #


def test_contradicted_block_fires_standalone() -> None:
    """Even with some entailed sentences, a contradiction crosses the 5%
    ceiling and fires BLOCK_PROSE_CONTRADICTED."""
    nli = _FakeNli()
    validator = BlockProseEntailmentValidator(nli=nli)

    prose = (
        "<p>The mitochondrion is the cell's powerhouse producing ATP through "
        "oxidative phosphorylation across its inner membrane [ENTAIL]. The "
        "mitochondrion actually destroys all cellular ATP and halts every "
        "metabolic process within the cell [CONTRA].</p>"
    )
    block = _make_block(content=prose, source_ids=("dart:slug#blk_0",))
    source_chunks = {"dart:slug#blk_0": _CHUNK}

    result = validator.validate(
        {"blocks": [block], "source_chunks": source_chunks}
    )

    contradicted = [
        i for i in result.issues if i.code == _CODE_BLOCK_PROSE_CONTRADICTED
    ]
    assert len(contradicted) == 1
    assert result.action == "regenerate"
    assert result.passed is False


# --------------------------------------------------------------------- #
# Test 4: Graceful-degrade
# --------------------------------------------------------------------- #


def test_nli_deps_missing_emits_warning_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_or_load returns None → NLI_DEPS_MISSING warning, passed=True."""
    from lib.classifiers import nli_classifier as mod

    monkeypatch.setattr(
        mod.NliClassifier, "get_or_load", classmethod(lambda cls: None),
    )
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)

    validator = BlockProseEntailmentValidator()  # nli=None → real loader
    block = _make_block(
        content="<p>Some on-topic prose about the cell here please.</p>",
        source_ids=("dart:slug#blk_0",),
    )
    result = validator.validate(
        {"blocks": [block], "source_chunks": {"dart:slug#blk_0": _CHUNK}}
    )

    deps = [i for i in result.issues if i.code == _CODE_NLI_DEPS_MISSING]
    assert len(deps) == 1
    assert deps[0].severity == "warning"
    assert result.passed is True
    assert result.action is None


def test_strict_mode_with_missing_deps_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.classifiers import nli_classifier as mod

    monkeypatch.setattr(
        mod.NliClassifier, "get_or_load", classmethod(lambda cls: None),
    )
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")

    validator = BlockProseEntailmentValidator()
    block = _make_block(
        content="<p>Some on-topic prose about the cell here please.</p>",
        source_ids=("dart:slug#blk_0",),
    )
    with pytest.raises(RuntimeError) as excinfo:
        validator.validate(
            {"blocks": [block], "source_chunks": {"dart:slug#blk_0": _CHUNK}}
        )
    assert "TRAINFORGE_REQUIRE_EMBEDDINGS" in str(excinfo.value)


# --------------------------------------------------------------------- #
# Test 5: Shadow path computes without gating
# --------------------------------------------------------------------- #


class _CaptureSpy:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def test_shadow_computes_without_gating() -> None:
    """Same fabricated-prose block as the keystone, but shadow=True: the
    capture fires with the low rate recorded (compute happened) but passed
    stays True and action is not 'regenerate'."""
    nli = _FakeNli()
    validator = BlockProseEntailmentValidator(nli=nli)
    capture = _CaptureSpy()

    prose = (
        "<p>The mitochondrion secretly manufactures cellular gold deep within "
        "its folded cristae structures every passing moment. It also quietly "
        "encodes ancient memories into the surrounding cytoplasm of the cell "
        "for safekeeping.</p>"
    )
    block = _make_block(content=prose, source_ids=("dart:slug#blk_0",))

    result = validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"dart:slug#blk_0": _CHUNK},
            "shadow": True,
            "decision_capture": capture,
        }
    )

    # The UNGROUNDED issue still fires (compute happened) but at warning
    # severity, so passed stays True and there's no regenerate action.
    ungrounded = [
        i for i in result.issues if i.code == _CODE_BLOCK_PROSE_UNGROUNDED
    ]
    assert len(ungrounded) == 1
    assert ungrounded[0].severity == "warning"
    assert result.passed is True
    assert result.action != "regenerate"
    # The capture recorded the computed rate.
    assert len(capture.events) == 1
    rationale = capture.events[0]["rationale"]
    assert "block_entailment_rate=" in rationale
    assert "shadow=True" in rationale


# --------------------------------------------------------------------- #
# Test 6: Decision-capture fires with dynamic signals
# --------------------------------------------------------------------- #


def test_decision_capture_dynamic_signals() -> None:
    nli = _FakeNli()
    validator = BlockProseEntailmentValidator(nli=nli)
    capture = _CaptureSpy()

    prose = (
        "<p>The mitochondrion is the cell's powerhouse and generates ATP "
        "through oxidative phosphorylation across its inner membrane "
        "[ENTAIL].</p>"
    )
    block = _make_block(content=prose, source_ids=("dart:slug#blk_0",))

    validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"dart:slug#blk_0": _CHUNK},
            "decision_capture": capture,
        }
    )

    assert len(capture.events) == 1
    evt = capture.events[0]
    assert evt["decision_type"] == _DECISION_TYPE
    rationale = evt["rationale"]
    assert len(rationale) >= 20
    for signal in ("block_entailment_rate=", "scored_count=", "nli_revision="):
        assert signal in rationale
    assert "fake-nli-rev-1" in rationale


# --------------------------------------------------------------------- #
# Test 7: Skip set asymmetry
# --------------------------------------------------------------------- #


def test_skip_set_skips_assessment_and_objective_scores_example() -> None:
    nli = _FakeNli()
    validator = BlockProseEntailmentValidator(nli=nli)
    capture = _CaptureSpy()

    # assessment_item + objective are skipped silently (no event).
    prose = "<p>On-topic prose about the cell and its inner membrane [ENTAIL].</p>"
    skipped = [
        _make_block(block_id="p#ai", block_type="assessment_item", content=prose),
        _make_block(block_id="p#obj", block_type="objective", content=prose),
    ]
    # example IS scored (deliberate asymmetry vs rewrite_source_grounding).
    example = _make_block(block_id="p#ex", block_type="example", content=prose)

    result = validator.validate(
        {
            "blocks": skipped + [example],
            "source_chunks": {"dart:slug#blk_0": _CHUNK},
            "decision_capture": capture,
        }
    )

    # Only the example block emitted a capture.
    assert len(capture.events) == 1
    assert "p#ex" in capture.events[0]["rationale"]
    assert result.passed is True


# --------------------------------------------------------------------- #
# Test 8: No-grounding-source
# --------------------------------------------------------------------- #


def test_no_grounding_source_warns_passes() -> None:
    nli = _FakeNli()
    validator = BlockProseEntailmentValidator(nli=nli)

    prose = "<p>On-topic prose about the cell and its inner membrane here.</p>"
    # source_ids resolve to nothing in source_chunks → no cited passages.
    block = _make_block(content=prose, source_ids=("dart:slug#missing",))

    result = validator.validate(
        {"blocks": [block], "source_chunks": {"dart:slug#blk_0": _CHUNK}}
    )

    no_src = [
        i for i in result.issues
        if i.code == _CODE_BLOCK_PROSE_NO_GROUNDING_SOURCE
    ]
    assert len(no_src) == 1
    assert no_src[0].severity == "warning"
    assert result.passed is True
    assert result.action is None


# --------------------------------------------------------------------- #
# Test 9: Computational-only block → no-op pass
# --------------------------------------------------------------------- #


def test_computational_only_block_no_op_pass() -> None:
    nli = _FakeNli()
    validator = BlockProseEntailmentValidator(nli=nli)

    # All sentences are derived-numeric (computational exemption); scored_count
    # is 0 so no failure is fabricated.
    prose = "<p>Therefore 3 + 10 = 13. The answer is 13. Also 25 * 4 = 100.</p>"
    block = _make_block(content=prose, source_ids=("dart:slug#blk_0",))

    result = validator.validate(
        {"blocks": [block], "source_chunks": {"dart:slug#blk_0": _CHUNK}}
    )

    assert result.passed is True
    assert result.action is None
    assert not any(
        i.code in (_CODE_BLOCK_PROSE_UNGROUNDED, _CODE_BLOCK_PROSE_CONTRADICTED)
        for i in result.issues
    )


# --------------------------------------------------------------------- #
# §5.4 — Calibration harness: shadow measures the distribution, gates nothing
# --------------------------------------------------------------------- #


def test_calibration_shadow_measures_without_mutating() -> None:
    """Run a small block batch with shadow=True; the captures carry the
    per-block entailment rates (compute happened) while NO block was gated —
    the in-code proof that the §4 calibration harness measures without
    mutating (the whole posture-inversion safety contract)."""
    nli = _FakeNli()
    validator = BlockProseEntailmentValidator(nli=nli)
    capture = _CaptureSpy()

    entailed_prose = (
        "<p>The mitochondrion is the cell's powerhouse and generates ATP "
        "through oxidative phosphorylation across its inner membrane "
        "[ENTAIL].</p>"
    )
    fabricated_prose = (
        "<p>The mitochondrion secretly manufactures cellular gold deep within "
        "its folded cristae structures every passing moment continuously.</p>"
    )
    blocks = [
        _make_block(block_id="p#a", content=entailed_prose),
        _make_block(block_id="p#b", content=fabricated_prose),
    ]

    result = validator.validate(
        {
            "blocks": blocks,
            "source_chunks": {"dart:slug#blk_0": _CHUNK},
            "shadow": True,
            "decision_capture": capture,
        }
    )

    # NOTHING was gated.
    assert result.passed is True
    assert result.action != "regenerate"
    assert all(i.severity == "warning" for i in result.issues)

    # The distribution WAS computed — one capture per scored block, each with a
    # numeric rate recorded.
    assert len(capture.events) == 2
    rates: List[float] = []
    for evt in capture.events:
        rationale = evt["rationale"]
        assert "block_entailment_rate=" in rationale
        assert "shadow=True" in rationale
        token = rationale.split("block_entailment_rate=")[1]
        rates.append(float(token.split(",")[0]))
    # The fabricated block read below the floor; the entailed block at/above —
    # the calibration signal is present and separable.
    assert min(rates) < 0.60
    assert max(rates) >= 0.60
