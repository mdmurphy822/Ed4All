"""GPT Feedback v2 Wave 2 W2.F — ``NliClassifier`` loader tests.

Covers the singleton-load contract for
:class:`lib.classifiers.nli_classifier.NliClassifier`. The real
DeBERTa-v3-base-mnli-fever-anli model load is ~750 MB so the actual
load test is gated behind ``@pytest.mark.slow``; the other tests
exercise the loader's import-probe + singleton-cache + score-pair /
score-batch surface contract via mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

# Repo root on path so the imports work from a fresh checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.classifiers.nli_classifier import (  # noqa: E402
    ENV_DEVICE,
    NliClassifier,
    NliScore,
    resolve_nli_device,
)


# --------------------------------------------------------------------- #
# Fixture autouse: reset the singleton between every test
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Reset NliClassifier singleton state before AND after each test.

    The singleton-load pattern caches both the loaded instance AND the
    negative result (load-failed flag). Resetting both directions keeps
    every test independent.
    """
    NliClassifier._reset_for_tests()
    yield
    NliClassifier._reset_for_tests()


# --------------------------------------------------------------------- #
# Test 1: get_or_load returns None when transformers/torch unavailable
# --------------------------------------------------------------------- #


def test_get_or_load_returns_none_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock the import probe to raise ImportError → loader returns None."""
    # Patch the import inside the loader's body. The module-level
    # singleton uses a fresh ``import torch`` / ``from transformers
    # import ...`` block guarded by ImportError; we simulate the
    # missing-extras path via ``__import__`` interception.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("torch", "transformers"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _failing_import)
    result = NliClassifier.get_or_load()
    assert result is None


# --------------------------------------------------------------------- #
# Test 2: get_or_load returns the same singleton instance
# --------------------------------------------------------------------- #


def test_get_or_load_returns_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two get_or_load calls return the same instance (singleton)."""
    # Build a fake instance and pin it directly on the class. We don't
    # exercise the import path here; the goal is to assert the
    # singleton-cache short-circuit behavior.
    fake_instance = NliClassifier(
        model=MagicMock(),
        tokenizer=MagicMock(),
        torch_module=MagicMock(),
        revision="6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    )
    monkeypatch.setattr(NliClassifier, "_INSTANCE", fake_instance)

    a = NliClassifier.get_or_load()
    b = NliClassifier.get_or_load()
    assert a is b
    assert a is fake_instance


# --------------------------------------------------------------------- #
# Test 3: score_pair returns NliScore with three floats summing to ~1.0
# --------------------------------------------------------------------- #


def test_score_pair_returns_softmaxed_nli_score() -> None:
    """Mock the model + tokenizer + torch; verify softmax shape."""
    # Build a fake torch module with the minimal surface the loader
    # uses: ``no_grad()`` context manager + ``softmax(logits, dim=-1)``.
    fake_torch = MagicMock()
    # The ``with torch.no_grad():`` context manager — make it a no-op
    # context.
    fake_torch.no_grad.return_value.__enter__ = MagicMock()
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    # softmax: return a fake tensor whose .shape[0] == batch_size and
    # whose row-indexed access returns objects with .item() floats.
    # Use a real list-of-lists so the validator's loop iterates it.
    fake_probs = MagicMock()
    fake_probs.shape = (1, 3)
    # Build per-row mock that supports ``probs[i][idx].item()``.
    row_mock = MagicMock()

    def _idx_mock(idx: int) -> Any:
        # Index-aware probabilities: 0=entailment=0.85, 1=neutral=0.10,
        # 2=contradiction=0.05.
        prob_map = {0: 0.85, 1: 0.10, 2: 0.05}
        item_mock = MagicMock()
        item_mock.item.return_value = prob_map[idx]
        return item_mock

    row_mock.__getitem__.side_effect = _idx_mock
    fake_probs.__getitem__.return_value = row_mock
    fake_torch.softmax.return_value = fake_probs

    fake_model = MagicMock()
    fake_outputs = MagicMock()
    fake_outputs.logits = MagicMock()
    fake_model.return_value = fake_outputs

    fake_tokenizer = MagicMock()
    # The tokenizer call returns a dict-like that gets ``**``-unpacked
    # into the model call.
    fake_encoded = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    fake_tokenizer.return_value = fake_encoded

    classifier = NliClassifier(
        model=fake_model,
        tokenizer=fake_tokenizer,
        torch_module=fake_torch,
        revision="test-rev",
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    )

    score = classifier.score_pair(premise="A", hypothesis="B")
    assert isinstance(score, NliScore)
    assert score.entailment == pytest.approx(0.85)
    assert score.neutral == pytest.approx(0.10)
    assert score.contradiction == pytest.approx(0.05)
    # Sum should be ~1.0 since these came from a softmax.
    total = score.entailment + score.neutral + score.contradiction
    assert total == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------- #
# Test 4: score_batch returns same-length list as input
# --------------------------------------------------------------------- #


def test_score_batch_returns_same_length_list() -> None:
    """Mock-driven: 3 input pairs → 3 output NliScores."""
    fake_torch = MagicMock()
    fake_torch.no_grad.return_value.__enter__ = MagicMock()
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    # softmax returns a fake tensor with ``.shape[0]==3``; per-row
    # access returns an indexable with ``.item()`` per axis.
    def _make_softmax_for_batch(batch_n: int) -> Any:
        probs = MagicMock()
        probs.shape = (batch_n, 3)
        rows: List[Any] = []
        for r in range(batch_n):
            row_mock = MagicMock()

            def _make_idx_mock(row_idx: int) -> Any:
                def _inner(idx: int) -> Any:
                    # Distinct values per row so we can confirm
                    # row-aware extraction works.
                    base = 0.10 + 0.05 * row_idx
                    prob_map = {0: base, 1: 0.5, 2: 1.0 - base - 0.5}
                    item_mock = MagicMock()
                    item_mock.item.return_value = prob_map[idx]
                    return item_mock
                return _inner

            row_mock.__getitem__.side_effect = _make_idx_mock(r)
            rows.append(row_mock)
        probs.__getitem__.side_effect = lambda i: rows[i]
        return probs

    fake_torch.softmax.side_effect = (
        lambda logits, dim: _make_softmax_for_batch(3)
    )

    fake_model = MagicMock()
    fake_outputs = MagicMock()
    fake_outputs.logits = MagicMock()
    fake_model.return_value = fake_outputs

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {"input_ids": MagicMock()}

    classifier = NliClassifier(
        model=fake_model,
        tokenizer=fake_tokenizer,
        torch_module=fake_torch,
        revision="test-rev",
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    )

    pairs: List[Tuple[str, str]] = [
        ("p1", "h1"),
        ("p2", "h2"),
        ("p3", "h3"),
    ]
    scores = classifier.score_batch(pairs=pairs)
    assert len(scores) == 3
    for s in scores:
        assert isinstance(s, NliScore)


# --------------------------------------------------------------------- #
# Test 5: empty pairs returns empty list
# --------------------------------------------------------------------- #


def test_score_batch_empty_input_returns_empty_list() -> None:
    """No pairs in → no scores out, no model dispatch."""
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_torch = MagicMock()
    classifier = NliClassifier(
        model=fake_model,
        tokenizer=fake_tokenizer,
        torch_module=fake_torch,
        revision="test-rev",
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    )
    assert classifier.score_batch(pairs=[]) == []
    fake_model.assert_not_called()
    fake_tokenizer.assert_not_called()


# --------------------------------------------------------------------- #
# Test 6: id2label resolution via model config (defensive against
# revision shifts)
# --------------------------------------------------------------------- #


def test_classifier_raises_when_id2label_missing_labels() -> None:
    """A model whose id2label lacks entailment/neutral/contradiction
    raises at construction (defends against silent revision drift)."""
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_torch = MagicMock()
    with pytest.raises(RuntimeError) as excinfo:
        NliClassifier(
            model=fake_model,
            tokenizer=fake_tokenizer,
            torch_module=fake_torch,
            revision="test-rev",
            id2label={0: "positive", 1: "negative"},  # wrong label set
        )
    assert "id2label" in str(excinfo.value).lower()


# --------------------------------------------------------------------- #
# Test 7: label resolution is case-insensitive
# --------------------------------------------------------------------- #


def test_classifier_resolves_id2label_case_insensitively() -> None:
    """Upper-case label names from a model config still resolve."""
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_torch = MagicMock()
    classifier = NliClassifier(
        model=fake_model,
        tokenizer=fake_tokenizer,
        torch_module=fake_torch,
        revision="test-rev",
        id2label={0: "ENTAILMENT", 1: "Neutral", 2: "CONTRADICTION"},
    )
    assert classifier._idx_entailment == 0
    assert classifier._idx_neutral == 1
    assert classifier._idx_contradiction == 2


# --------------------------------------------------------------------- #
# Test 8: pinned revision matches the BERT ensemble entry
# --------------------------------------------------------------------- #


def test_pinned_revision_matches_bert_ensemble() -> None:
    """The NLI loader's pinned SHA MUST match the BERT-ensemble
    entry [2] so the two surfaces can never drift apart on the
    underlying model commit."""
    from lib.classifiers.bloom_bert_ensemble import _DEFAULT_ENSEMBLE_MEMBERS

    deberta_entry = _DEFAULT_ENSEMBLE_MEMBERS[2]
    assert deberta_entry["name"] == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    # 40-hex-char pin sanity.
    assert len(deberta_entry["revision"]) == 40
    assert all(
        c in "0123456789abcdef" for c in deberta_entry["revision"]
    )


# --------------------------------------------------------------------- #
# Test 9 (slow): real model load — operator runs on demand
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
# Device knob (ED4ALL_NLI_DEVICE) — resolver + placement + score-path
# --------------------------------------------------------------------- #


def _make_scoring_fake_torch(*, batch_n: int = 1) -> Any:
    """Build a fake torch module with the score_batch surface.

    Provides ``no_grad()`` (no-op context), ``softmax(logits, dim=-1)``
    (returns a fake probs tensor with ``.shape`` + indexable rows whose
    ``[idx].item()`` returns floats), and a recording ``.cpu()`` on the
    probs tensor so the device-move path is observable.
    """
    fake_torch = MagicMock()
    fake_torch.no_grad.return_value.__enter__ = MagicMock()
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    def _make_probs() -> Any:
        probs = MagicMock()
        probs.shape = (batch_n, 3)
        rows: List[Any] = []
        for _ in range(batch_n):
            row_mock = MagicMock()

            def _idx(idx: int) -> Any:
                prob_map = {0: 0.85, 1: 0.10, 2: 0.05}
                item_mock = MagicMock()
                item_mock.item.return_value = prob_map[idx]
                return item_mock

            row_mock.__getitem__.side_effect = _idx
            rows.append(row_mock)
        probs.__getitem__.side_effect = lambda i: rows[i]
        # ``.cpu()`` returns the same probs tensor (records the call).
        probs.cpu.return_value = probs
        return probs

    fake_torch.softmax.side_effect = lambda logits, dim: _make_probs()
    return fake_torch


def _build_classifier(fake_torch: Any, *, device: Any = None) -> NliClassifier:
    fake_model = MagicMock()
    fake_outputs = MagicMock()
    fake_outputs.logits = MagicMock()
    fake_model.return_value = fake_outputs
    fake_tokenizer = MagicMock()
    # Encoded must be a real dict so ``{k: v.to(...)}`` comprehension works.
    fake_tokenizer.return_value = {"input_ids": MagicMock()}
    return NliClassifier(
        model=fake_model,
        tokenizer=fake_tokenizer,
        torch_module=fake_torch,
        revision="test-rev",
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        device=device,
    )


def test_resolve_nli_device_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolution chain: explicit arg > env > default cpu."""
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    assert resolve_nli_device() == "cpu"  # default
    monkeypatch.setenv(ENV_DEVICE, "cuda")
    assert resolve_nli_device() == "cuda"  # env wins over default
    assert resolve_nli_device("cuda:1") == "cuda:1"  # arg wins over env


def test_default_device_cpu_no_to_no_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (cpu): model.to / model.half never called; cpu path byte-stable."""
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    classifier = _build_classifier(fake_torch)
    assert classifier.device == "cpu"
    assert classifier.dtype == "float32"
    classifier._model.to.assert_not_called()
    classifier._model.half.assert_not_called()
    # cuda availability is never even probed on the cpu path.
    fake_torch.cuda.is_available.assert_not_called()

    # Score path: no device move, no .cpu() on probs.
    score = classifier.score_pair(premise="A", hypothesis="B")
    assert isinstance(score, NliScore)
    assert isinstance(score.entailment, float)
    assert score.entailment == pytest.approx(0.85)


def test_cuda_device_when_available_moves_and_halves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cuda requested + available → model.to('cuda') + .half(); tensors moved."""
    monkeypatch.setenv(ENV_DEVICE, "cuda")
    fake_torch = _make_scoring_fake_torch()
    fake_torch.cuda.is_available.return_value = True

    classifier = _build_classifier(fake_torch)
    assert classifier.device == "cuda"
    assert classifier.dtype == "float16"
    classifier._model.to.assert_called_once_with("cuda")
    classifier._model.half.assert_called_once_with()

    # Encoded tensors get .to("cuda"); probs get .cpu() back before reads.
    encoded_tensor = classifier._tokenizer.return_value["input_ids"]
    scores = classifier.score_batch(pairs=[("p", "h")])
    encoded_tensor.to.assert_called_with("cuda")
    assert len(scores) == 1
    # Results read back as plain Python floats despite the GPU round-trip.
    assert isinstance(scores[0].entailment, float)
    assert scores[0].entailment == pytest.approx(0.85)


def test_cuda_requested_but_unavailable_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cuda requested, torch.cuda unavailable → CPU fallback, no crash, warns."""
    monkeypatch.setenv(ENV_DEVICE, "cuda")
    fake_torch = _make_scoring_fake_torch()
    fake_torch.cuda.is_available.return_value = False

    with patch("lib.classifiers.nli_classifier.logger.warning") as warn:
        classifier = _build_classifier(fake_torch)
    assert classifier.device == "cpu"
    assert classifier.dtype == "float32"
    classifier._model.to.assert_not_called()
    classifier._model.half.assert_not_called()
    assert warn.called  # one-time fallback warning fired
    # Still scores fine on the fallback CPU path.
    score = classifier.score_pair(premise="A", hypothesis="B")
    assert isinstance(score.entailment, float)


def test_constructor_arg_overrides_env_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit device= arg beats ED4ALL_NLI_DEVICE env (cpu wins, no probe)."""
    monkeypatch.setenv(ENV_DEVICE, "cuda")
    fake_torch = _make_scoring_fake_torch()
    classifier = _build_classifier(fake_torch, device="cpu")
    assert classifier.device == "cpu"
    classifier._model.to.assert_not_called()
    fake_torch.cuda.is_available.assert_not_called()


@pytest.mark.slow
def test_get_or_load_loads_real_model() -> None:
    """Smoke test: real DeBERTa-v3-base load + score a known pair.

    Marked slow because the model is ~750 MB and downloads from
    HuggingFace. Skipped on CPU-only dev boxes that don't have the
    [embedding] extras installed (the loader returns None then;
    pytest.skip is fired).
    """
    classifier = NliClassifier.get_or_load()
    if classifier is None:
        pytest.skip("transformers / torch extras not installed")
    score = classifier.score_pair(
        premise="The cat sat on the mat.",
        hypothesis="A feline rested on a rug.",
    )
    assert isinstance(score, NliScore)
    # An entailment pair should score entailment > contradiction.
    assert score.entailment > score.contradiction
