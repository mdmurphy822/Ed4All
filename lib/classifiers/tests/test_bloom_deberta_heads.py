"""Bloom-ladder initiative WI-05 — ``BloomDebertaHeads`` loader tests.

Covers the singleton-load contract for
:class:`lib.classifiers.bloom_deberta_heads.BloomDebertaHeads`: the
directory-driven all-or-nothing degrade, the HF-offline
``local_files_only`` construction, the device policy shared with
``NliClassifier`` (``ED4ALL_NLI_DEVICE``), and the
``classify_batch`` argmax + abstention-floor contract. No real model
weights are loaded anywhere in this suite — every torch/transformers
surface is mocked, and the directory-driven tests use empty
``tmp_path`` trees (a ``config.json`` sentinel file is enough to
satisfy the loadable-dir check without a real checkpoint).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

# Repo root on path so the imports work from a fresh checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.classifiers.bloom_deberta_heads import (  # noqa: E402
    BLOOM_LEVELS,
    BloomDebertaHeads,
    ENV_HEADS_DIR,
    default_registry,
    resolve_bloom_heads_dir,
    resolve_multiclass_head_dir,
)
from lib.classifiers.nli_classifier import ENV_DEVICE  # noqa: E402


# --------------------------------------------------------------------- #
# Fixture autouse: reset the singleton + clear env between every test
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the singleton + clear the two governing env vars per test.

    Mirrors ``test_nli_classifier.py``'s autouse reset. Clearing
    ``ED4ALL_NLI_DEVICE`` matters here too since this module reuses that
    exact knob (no sibling device flag was introduced).
    """
    monkeypatch.delenv(ENV_HEADS_DIR, raising=False)
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    BloomDebertaHeads._reset_for_tests()
    yield
    BloomDebertaHeads._reset_for_tests()


def _make_checkpoint_dirs(base: Path, levels: List[str]) -> None:
    """Create ``<base>/<level>/final/config.json`` for each of ``levels``."""
    for level in levels:
        d = base / level / "final"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")


def _make_multiclass_checkpoint_dir(base: Path) -> None:
    """Create ``<base>/multiclass/final/config.json``."""
    d = base / "multiclass" / "final"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")


# --------------------------------------------------------------------- #
# resolve_bloom_heads_dir — parse-with-fallback resolution chain
# --------------------------------------------------------------------- #


def test_resolve_bloom_heads_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_HEADS_DIR, raising=False)
    assert resolve_bloom_heads_dir() == "models/bloom_classifiers"


def test_resolve_bloom_heads_dir_env_wins_over_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_HEADS_DIR, "/custom/heads")
    assert resolve_bloom_heads_dir() == "/custom/heads"


def test_resolve_bloom_heads_dir_arg_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_HEADS_DIR, "/env/heads")
    assert resolve_bloom_heads_dir("/arg/heads") == "/arg/heads"


def test_resolve_bloom_heads_dir_blank_env_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_HEADS_DIR, "   ")
    assert resolve_bloom_heads_dir() == "models/bloom_classifiers"


def test_resolve_bloom_heads_dir_blank_arg_falls_through_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_HEADS_DIR, "/env/heads")
    assert resolve_bloom_heads_dir("   ") == "/env/heads"


# --------------------------------------------------------------------- #
# default_registry
# --------------------------------------------------------------------- #


def test_default_registry_covers_all_six_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_HEADS_DIR, "/base")
    registry = default_registry()
    assert set(registry.keys()) == set(BLOOM_LEVELS)
    for level in BLOOM_LEVELS:
        assert registry[level] == Path("/base") / level / "final"


# --------------------------------------------------------------------- #
# resolve_multiclass_head_dir
# --------------------------------------------------------------------- #


def test_resolve_multiclass_head_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_HEADS_DIR, raising=False)
    assert resolve_multiclass_head_dir() == (
        Path("models/bloom_classifiers") / "multiclass" / "final"
    )


def test_resolve_multiclass_head_dir_follows_heads_dir_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_HEADS_DIR, "/base")
    assert resolve_multiclass_head_dir() == Path("/base") / "multiclass" / "final"


# --------------------------------------------------------------------- #
# get_or_load — directory-driven all-or-nothing degrade
# --------------------------------------------------------------------- #


def test_get_or_load_returns_none_when_heads_dir_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Fresh checkout (no shipped weights) -> None, no import attempted."""
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path / "does_not_exist"))
    # Sabotage the heavy import so a test failure here would prove the
    # loader tried to import despite the missing directories.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("torch", "transformers"):
            raise AssertionError(
                f"BloomDebertaHeads imported {name!r} despite missing "
                f"checkpoint directories — the cheap fs check should "
                f"short-circuit before any heavy import."
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _failing_import)
    assert BloomDebertaHeads.get_or_load() is None


def test_get_or_load_returns_none_on_partial_ladder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Only 5 of 6 level dirs present -> None (all-or-nothing)."""
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path))
    _make_checkpoint_dirs(tmp_path, list(BLOOM_LEVELS[:-1]))  # drop "create"
    assert BloomDebertaHeads.get_or_load() is None


def test_get_or_load_returns_none_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """All six dirs present, but torch/transformers unavailable -> None."""
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path))
    _make_checkpoint_dirs(tmp_path, list(BLOOM_LEVELS))

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("torch", "transformers"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _failing_import)
    assert BloomDebertaHeads.get_or_load() is None


def test_get_or_load_returns_none_when_a_head_load_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """One head's from_pretrained raises -> whole loader degrades to None."""
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path))
    _make_checkpoint_dirs(tmp_path, list(BLOOM_LEVELS))

    fake_transformers = MagicMock()
    fake_tokenizer_cls = MagicMock()
    fake_tokenizer_cls.from_pretrained.return_value = MagicMock()
    fake_model_cls = MagicMock()
    fake_model_cls.from_pretrained.side_effect = RuntimeError("corrupt checkpoint")
    fake_transformers.AutoTokenizer = fake_tokenizer_cls
    fake_transformers.AutoModelForSequenceClassification = fake_model_cls

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    assert BloomDebertaHeads.get_or_load() is None


def test_get_or_load_loads_local_dirs_only_with_local_files_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Every from_pretrained call uses the LOCAL checkpoint path + local_files_only=True.

    HF-offline-by-construction contract: assert the exact local
    directory string is passed (never a bare hub repo id), and that
    ``local_files_only=True`` is set on every one of the twelve calls
    (6 tokenizers + 6 models).
    """
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path))
    _make_checkpoint_dirs(tmp_path, list(BLOOM_LEVELS))

    fake_torch = MagicMock()
    fake_transformers = MagicMock()
    fake_tokenizer_cls = MagicMock()
    fake_tokenizer_cls.from_pretrained.return_value = MagicMock()
    fake_model_cls = MagicMock()

    def _model_from_pretrained(path: str, **kwargs: Any) -> Any:
        m = MagicMock()
        return m

    fake_model_cls.from_pretrained.side_effect = _model_from_pretrained
    fake_transformers.AutoTokenizer = fake_tokenizer_cls
    fake_transformers.AutoModelForSequenceClassification = fake_model_cls

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    instance = BloomDebertaHeads.get_or_load()
    assert instance is not None
    assert isinstance(instance, BloomDebertaHeads)

    registry = default_registry()
    expected_paths = {str(registry[level]) for level in BLOOM_LEVELS}

    tok_calls = fake_tokenizer_cls.from_pretrained.call_args_list
    model_calls = fake_model_cls.from_pretrained.call_args_list
    assert len(tok_calls) == 6
    assert len(model_calls) == 6
    for call in tok_calls + model_calls:
        args, kwargs = call
        assert args[0] in expected_paths
        assert kwargs.get("local_files_only") is True
        # Never a bare hub-style repo id (no "/"-free org/name shape
        # without a leading path component, no huggingface.co reference).
        assert str(tmp_path) in args[0]


def test_get_or_load_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two get_or_load calls return the same instance once loaded."""
    fake_instance = BloomDebertaHeads(
        heads={level: (MagicMock(), MagicMock()) for level in BLOOM_LEVELS},
        torch_module=MagicMock(),
    )
    monkeypatch.setattr(BloomDebertaHeads, "_INSTANCE", fake_instance)

    a = BloomDebertaHeads.get_or_load()
    b = BloomDebertaHeads.get_or_load()
    assert a is b
    assert a is fake_instance


def test_get_or_load_caches_negative_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A failed load latches _LOAD_FAILED so a second call is O(1)/no re-probe."""
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path / "missing"))
    assert BloomDebertaHeads.get_or_load() is None
    assert BloomDebertaHeads._LOAD_FAILED is True
    # Second call short-circuits on the cached flag without re-resolving
    # the (now-different) env value.
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path / "still-missing"))
    assert BloomDebertaHeads.get_or_load() is None


# --------------------------------------------------------------------- #
# get_or_load — multiclass artifact: auto-detect + preference order
# --------------------------------------------------------------------- #


def _fake_multiclass_transformers_module() -> Any:
    fake_transformers = MagicMock()
    fake_tokenizer_cls = MagicMock()
    fake_tokenizer_cls.from_pretrained.return_value = MagicMock()
    fake_model_cls = MagicMock()

    def _model_from_pretrained(path: str, **kwargs: Any) -> Any:
        return MagicMock()

    fake_model_cls.from_pretrained.side_effect = _model_from_pretrained
    fake_transformers.AutoTokenizer = fake_tokenizer_cls
    fake_transformers.AutoModelForSequenceClassification = fake_model_cls
    return fake_transformers


def test_get_or_load_loads_multiclass_head_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path))
    _make_multiclass_checkpoint_dir(tmp_path)

    fake_transformers = _fake_multiclass_transformers_module()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())

    instance = BloomDebertaHeads.get_or_load()
    assert instance is not None
    assert instance.head_mode == "multiclass"
    # Exactly one tokenizer/model load -- never the six-way ladder.
    assert fake_transformers.AutoTokenizer.from_pretrained.call_count == 1
    assert fake_transformers.AutoModelForSequenceClassification.from_pretrained.call_count == 1
    call_args = fake_transformers.AutoTokenizer.from_pretrained.call_args
    assert call_args.args[0] == str(resolve_multiclass_head_dir())
    assert call_args.kwargs.get("local_files_only") is True


def test_get_or_load_prefers_multiclass_over_one_vs_rest_ladder_when_both_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Both artifacts on disk -> multiclass wins; the six-way ladder is
    never even probed for loadability beyond the multiclass short-circuit."""
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path))
    _make_multiclass_checkpoint_dir(tmp_path)
    _make_checkpoint_dirs(tmp_path, list(BLOOM_LEVELS))

    fake_transformers = _fake_multiclass_transformers_module()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())

    instance = BloomDebertaHeads.get_or_load()
    assert instance is not None
    assert instance.head_mode == "multiclass"
    # One tokenizer/model call each -- not six (the ladder was skipped).
    assert fake_transformers.AutoTokenizer.from_pretrained.call_count == 1
    assert fake_transformers.AutoModelForSequenceClassification.from_pretrained.call_count == 1


def test_get_or_load_falls_back_to_one_vs_rest_when_multiclass_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path))
    _make_checkpoint_dirs(tmp_path, list(BLOOM_LEVELS))
    # No multiclass dir created.

    fake_transformers = _fake_multiclass_transformers_module()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())

    instance = BloomDebertaHeads.get_or_load()
    assert instance is not None
    assert instance.head_mode == "one-vs-rest"
    assert fake_transformers.AutoTokenizer.from_pretrained.call_count == 6
    assert fake_transformers.AutoModelForSequenceClassification.from_pretrained.call_count == 6


def test_get_or_load_returns_none_when_multiclass_head_load_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv(ENV_HEADS_DIR, str(tmp_path))
    _make_multiclass_checkpoint_dir(tmp_path)

    fake_transformers = MagicMock()
    fake_tokenizer_cls = MagicMock()
    fake_tokenizer_cls.from_pretrained.return_value = MagicMock()
    fake_model_cls = MagicMock()
    fake_model_cls.from_pretrained.side_effect = RuntimeError("corrupt checkpoint")
    fake_transformers.AutoTokenizer = fake_tokenizer_cls
    fake_transformers.AutoModelForSequenceClassification = fake_model_cls

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    assert BloomDebertaHeads.get_or_load() is None


# --------------------------------------------------------------------- #
# Constructor validation — all-or-nothing head set / multiclass exclusivity
# --------------------------------------------------------------------- #


def test_constructor_raises_on_partial_heads_dict() -> None:
    partial = {level: (MagicMock(), MagicMock()) for level in BLOOM_LEVELS[:-1]}
    with pytest.raises(RuntimeError, match="missing"):
        BloomDebertaHeads(heads=partial, torch_module=MagicMock())


def test_constructor_raises_on_extra_keys() -> None:
    heads = {level: (MagicMock(), MagicMock()) for level in BLOOM_LEVELS}
    heads["bogus_level"] = (MagicMock(), MagicMock())
    with pytest.raises(RuntimeError, match="extra"):
        BloomDebertaHeads(heads=heads, torch_module=MagicMock())


def test_constructor_raises_when_neither_heads_nor_multiclass_given() -> None:
    with pytest.raises(RuntimeError, match="requires exactly one"):
        BloomDebertaHeads(torch_module=MagicMock())


def test_constructor_raises_when_both_heads_and_multiclass_given() -> None:
    heads = {level: (MagicMock(), MagicMock()) for level in BLOOM_LEVELS}
    with pytest.raises(RuntimeError, match="never both"):
        BloomDebertaHeads(
            heads=heads,
            multiclass_head=(MagicMock(), MagicMock()),
            torch_module=MagicMock(),
        )


def test_constructor_multiclass_sets_head_mode() -> None:
    instance = BloomDebertaHeads(
        multiclass_head=(MagicMock(), MagicMock()), torch_module=MagicMock(),
    )
    assert instance.head_mode == "multiclass"


def test_constructor_one_vs_rest_sets_head_mode() -> None:
    heads = {level: (MagicMock(), MagicMock()) for level in BLOOM_LEVELS}
    instance = BloomDebertaHeads(heads=heads, torch_module=MagicMock())
    assert instance.head_mode == "one-vs-rest"


# --------------------------------------------------------------------- #
# Device policy (reuses ED4ALL_NLI_DEVICE) — mirrors test_nli_classifier.py
# --------------------------------------------------------------------- #


def _make_scoring_fake_torch() -> Any:
    """Fake torch supporting no_grad() + sigmoid(logits) -> scriptable probs."""
    fake_torch = MagicMock()
    fake_torch.no_grad.return_value.__enter__ = MagicMock()
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    return fake_torch


def _build_heads_dict() -> Any:
    heads = {}
    for level in BLOOM_LEVELS:
        model = MagicMock()
        outputs = MagicMock()
        outputs.logits = MagicMock()
        model.return_value = outputs
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": MagicMock()}
        heads[level] = (model, tokenizer)
    return heads


def test_default_device_cpu_no_to_no_half(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    heads = _build_heads_dict()
    instance = BloomDebertaHeads(heads=heads, torch_module=fake_torch)
    assert instance.device == "cpu"
    assert instance.dtype == "float32"
    for model, _tok in heads.values():
        model.to.assert_not_called()
        model.half.assert_not_called()
    fake_torch.cuda.is_available.assert_not_called()


def test_cuda_device_when_available_moves_and_halves_every_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_DEVICE, "cuda")
    fake_torch = _make_scoring_fake_torch()
    fake_torch.cuda.is_available.return_value = True
    heads = _build_heads_dict()

    instance = BloomDebertaHeads(heads=heads, torch_module=fake_torch)
    assert instance.device == "cuda"
    assert instance.dtype == "float16"
    for model, _tok in heads.values():
        model.to.assert_called_once_with("cuda")
        model.half.assert_called_once_with()


def test_cuda_requested_but_unavailable_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_DEVICE, "cuda")
    fake_torch = _make_scoring_fake_torch()
    fake_torch.cuda.is_available.return_value = False
    heads = _build_heads_dict()

    with patch("lib.classifiers.bloom_deberta_heads.logger.warning") as warn:
        instance = BloomDebertaHeads(heads=heads, torch_module=fake_torch)
    assert instance.device == "cpu"
    assert instance.dtype == "float32"
    for model, _tok in heads.values():
        model.to.assert_not_called()
    assert warn.called


def test_constructor_device_arg_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_DEVICE, "cuda")
    fake_torch = _make_scoring_fake_torch()
    heads = _build_heads_dict()
    instance = BloomDebertaHeads(heads=heads, torch_module=fake_torch, device="cpu")
    assert instance.device == "cpu"
    fake_torch.cuda.is_available.assert_not_called()


# --------------------------------------------------------------------- #
# classify_batch — argmax over six one-vs-rest sigmoid scores + abstention
# --------------------------------------------------------------------- #


def _scripted_scores_classifier(
    fake_torch: Any, score_by_level: dict,
) -> BloomDebertaHeads:
    """Build a classifier whose per-level POSITIVE-class score (single text,
    batch size 1) is exactly ``score_by_level[level]`` for every level.

    Models the real ``train_bloom_deberta`` one-vs-rest artifact: each head
    is ``num_labels=2`` (logits shape ``(batch, 2)``; index 0 = negative
    class, 1 = positive), so the loader must softmax and read column 1 —
    NOT sigmoid column 0, which would score inverted on trained heads.
    """
    heads = {}
    for level in BLOOM_LEVELS:
        model = MagicMock()
        outputs = MagicMock()
        outputs.logits = MagicMock()
        outputs.logits.shape = (1, 2)
        model.return_value = outputs
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": MagicMock()}
        heads[level] = (model, tokenizer)

    def _softmax(logits: Any, *args: Any, **kwargs: Any) -> Any:
        # Identify which level's model produced these logits by walking
        # the mock identity map built above.
        for level, (model, _tok) in heads.items():
            if logits is model.return_value.logits:
                score = score_by_level[level]
                probs = MagicMock()  # softmax(...)[..., 1]
                item_mock = MagicMock()
                item_mock.item.return_value = score
                probs.__getitem__.return_value = item_mock  # probs[i]
                probs.cpu.return_value = probs
                sm = MagicMock()  # softmax(...) before the [..., 1] slice
                sm.__getitem__.return_value = probs
                return sm
        raise AssertionError("unrecognized logits object in softmax()")

    fake_torch.softmax.side_effect = _softmax
    fake_torch.sigmoid.side_effect = AssertionError(
        "sigmoid() must not be called for num_labels=2 heads"
    )
    return BloomDebertaHeads(heads=heads, torch_module=fake_torch)


def test_single_logit_heads_use_sigmoid_positive_prob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A head exporting a single logit (shape ``(batch, 1)``) scores via
    ``sigmoid(logits[..., 0])`` — the sigmoid IS the positive probability."""
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    heads = {}
    marker_to_level = {}
    for level in BLOOM_LEVELS:
        model = MagicMock()
        outputs = MagicMock()
        outputs.logits = MagicMock()
        outputs.logits.shape = (1, 1)
        marker = MagicMock()
        outputs.logits.__getitem__.return_value = marker  # logits[..., 0]
        marker_to_level[id(marker)] = level
        model.return_value = outputs
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": MagicMock()}
        heads[level] = (model, tokenizer)
    scores = {
        "remember": 0.20,
        "understand": 0.95,
        "apply": 0.10,
        "analyze": 0.10,
        "evaluate": 0.10,
        "create": 0.10,
    }

    def _sigmoid(sliced: Any, *args: Any, **kwargs: Any) -> Any:
        level = marker_to_level[id(sliced)]
        probs = MagicMock()
        item_mock = MagicMock()
        item_mock.item.return_value = scores[level]
        probs.__getitem__.return_value = item_mock
        probs.cpu.return_value = probs
        return probs

    fake_torch.sigmoid.side_effect = _sigmoid
    fake_torch.softmax.side_effect = AssertionError(
        "softmax() must not be called for single-logit heads"
    )
    classifier = BloomDebertaHeads(heads=heads, torch_module=fake_torch)
    assert classifier.classify_batch(["some text"]) == [("understand", 0.95)]


def test_classify_batch_argmax_picks_highest_scoring_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    scores = {
        "remember": 0.10,
        "understand": 0.20,
        "apply": 0.91,
        "analyze": 0.30,
        "evaluate": 0.15,
        "create": 0.05,
    }
    classifier = _scripted_scores_classifier(fake_torch, scores)

    results = classifier.classify_batch(["some text"])
    assert results == [("apply", 0.91)]


def test_classify_batch_tie_break_is_lexicographic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two levels tie at the max score -> the lexicographically-first wins."""
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    scores = {
        "remember": 0.10,
        "understand": 0.80,
        "apply": 0.80,  # ties with understand; "apply" < "understand" lexically
        "analyze": 0.30,
        "evaluate": 0.15,
        "create": 0.05,
    }
    classifier = _scripted_scores_classifier(fake_torch, scores)

    results = classifier.classify_batch(["some text"])
    assert results == [("apply", 0.80)]


def test_classify_batch_abstains_below_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Winner below the abstention floor -> None (no vote cast)."""
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    scores = {
        "remember": 0.10,
        "understand": 0.20,
        "apply": 0.45,  # winner, but below the default 0.5 floor
        "analyze": 0.30,
        "evaluate": 0.15,
        "create": 0.05,
    }
    classifier = _scripted_scores_classifier(fake_torch, scores)

    results = classifier.classify_batch(["some text"])
    assert results == [None]


def test_classify_batch_custom_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-supplied floor overrides the default 0.5."""
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    scores = {
        "remember": 0.10,
        "understand": 0.20,
        "apply": 0.45,
        "analyze": 0.30,
        "evaluate": 0.15,
        "create": 0.05,
    }
    classifier = _scripted_scores_classifier(fake_torch, scores)

    # Lower floor admits the 0.45 winner.
    results = classifier.classify_batch(["some text"], floor=0.4)
    assert results == [("apply", 0.45)]
    # Higher floor rejects it even harder.
    results2 = classifier.classify_batch(["some text"], floor=0.9)
    assert results2 == [None]


def test_classify_batch_empty_input_returns_empty_list() -> None:
    fake_torch = _make_scoring_fake_torch()
    heads = _build_heads_dict()
    classifier = BloomDebertaHeads(heads=heads, torch_module=fake_torch)
    results = classifier.classify_batch([])
    assert results == []
    for model, tok in heads.values():
        model.assert_not_called()
        tok.assert_not_called()


def test_classify_single_text_delegates_to_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    scores = {
        "remember": 0.10,
        "understand": 0.20,
        "apply": 0.91,
        "analyze": 0.30,
        "evaluate": 0.15,
        "create": 0.05,
    }
    classifier = _scripted_scores_classifier(fake_torch, scores)
    result = classifier.classify("some text")
    assert result == ("apply", 0.91)


# --------------------------------------------------------------------- #
# classify_batch — multiclass: one softmax head, argmax + abstention
# --------------------------------------------------------------------- #


def _scripted_multiclass_classifier(
    fake_torch: Any, rows_scores: List[dict],
) -> BloomDebertaHeads:
    """Build a multiclass classifier whose per-text six-way softmax row is
    exactly ``rows_scores[i]`` (dict keyed by level) for ``texts[i]`` --
    all texts scored in a single forward pass (batch <= default 8)."""
    model = MagicMock()
    outputs = MagicMock()
    outputs.logits = MagicMock()
    model.return_value = outputs
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": MagicMock()}

    probs_list = [[row[level] for level in BLOOM_LEVELS] for row in rows_scores]
    probs_mock = MagicMock()
    probs_mock.cpu.return_value = probs_mock
    probs_mock.tolist.return_value = probs_list

    def _softmax(logits: Any, dim: int = -1) -> Any:
        assert logits is outputs.logits
        return probs_mock

    fake_torch.softmax.side_effect = _softmax

    return BloomDebertaHeads(
        multiclass_head=(model, tokenizer), torch_module=fake_torch,
    )


def test_classify_batch_multiclass_argmax_picks_highest_scoring_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    scores = {
        "remember": 0.05,
        "understand": 0.10,
        "apply": 0.70,
        "analyze": 0.10,
        "evaluate": 0.03,
        "create": 0.02,
    }
    classifier = _scripted_multiclass_classifier(fake_torch, [scores])
    results = classifier.classify_batch(["some text"])
    assert results == [("apply", 0.70)]


def test_classify_batch_multiclass_tie_break_is_lexicographic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A softmax distribution can't have two classes both exceed 0.5 at
    # once, so the tie sits exactly ON the default floor (0.5 each) --
    # still admitted (the floor rejects strictly-below, not equal-to).
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    scores = {
        "remember": 0.0,
        "understand": 0.50,
        "apply": 0.50,  # ties with understand; "apply" < "understand" lexically
        "analyze": 0.0,
        "evaluate": 0.0,
        "create": 0.0,
    }
    classifier = _scripted_multiclass_classifier(fake_torch, [scores])
    results = classifier.classify_batch(["some text"])
    assert results == [("apply", 0.50)]


def test_classify_batch_multiclass_abstains_below_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Winner below the abstention floor -> None (no vote cast)."""
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    scores = {
        "remember": 0.15,
        "understand": 0.20,
        "apply": 0.30,  # winner, but below the default 0.5 floor
        "analyze": 0.20,
        "evaluate": 0.10,
        "create": 0.05,
    }
    classifier = _scripted_multiclass_classifier(fake_torch, [scores])
    results = classifier.classify_batch(["some text"])
    assert results == [None]


def test_classify_batch_multiclass_custom_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    scores = {
        "remember": 0.15,
        "understand": 0.20,
        "apply": 0.30,
        "analyze": 0.20,
        "evaluate": 0.10,
        "create": 0.05,
    }
    classifier = _scripted_multiclass_classifier(fake_torch, [scores])

    # Lower floor admits the 0.30 winner.
    results = classifier.classify_batch(["some text"], floor=0.25)
    assert results == [("apply", 0.30)]
    # Higher floor rejects it even harder.
    results2 = classifier.classify_batch(["some text"], floor=0.9)
    assert results2 == [None]


def test_classify_batch_multiclass_multiple_texts_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    row_a = {
        "remember": 0.80, "understand": 0.05, "apply": 0.05,
        "analyze": 0.05, "evaluate": 0.03, "create": 0.02,
    }
    row_b = {
        "remember": 0.02, "understand": 0.03, "apply": 0.05,
        "analyze": 0.05, "evaluate": 0.05, "create": 0.80,
    }
    classifier = _scripted_multiclass_classifier(fake_torch, [row_a, row_b])
    results = classifier.classify_batch(["text a", "text b"])
    assert results == [("remember", 0.80), ("create", 0.80)]


def test_classify_batch_multiclass_empty_input_returns_empty_list() -> None:
    fake_torch = _make_scoring_fake_torch()
    instance = BloomDebertaHeads(
        multiclass_head=(MagicMock(), MagicMock()), torch_module=fake_torch,
    )
    results = instance.classify_batch([])
    assert results == []


def test_classify_single_text_multiclass_delegates_to_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    fake_torch = _make_scoring_fake_torch()
    scores = {
        "remember": 0.05,
        "understand": 0.10,
        "apply": 0.70,
        "analyze": 0.10,
        "evaluate": 0.03,
        "create": 0.02,
    }
    classifier = _scripted_multiclass_classifier(fake_torch, [scores])
    result = classifier.classify("some text")
    assert result == ("apply", 0.70)
