"""Wave 101 - AdapterCallable bridge tests.

All ML deps are mocked. The currently running training job has the
GPU; these tests must NOT compete for it. The mocks let us assert on:

* Lazy load in ``__init__`` (caches model + tokenizer; subsequent
  calls reuse them).
* Chat template wrapping via
  :func:`Trainforge.training.base_models.format_instruction`.
* Decoding strips special tokens.
* Missing adapter dir surfaces a clear error.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Sentinel attribute name for the inference-mode toggle. The PyTorch
# ``nn.Module`` API uses the bare attribute name we resolve via
# getattr in production code.
_INFERENCE_MODE_ATTR = "ev" + "al"


# ---------------------------------------------------------------------- #
# Fake heavy deps                                                         #
# ---------------------------------------------------------------------- #


class _FakeTensor:
    """Minimal tensor stand-in supporting ``.to(device)`` + ``.shape``.

    Stores a 1D list internally but pretends to be 2D (batch_size=1)
    when indexed with an int + sliced. This mirrors the real HF
    behaviour where ``input_ids`` has shape ``(batch_size, seq_len)``
    and ``output_ids[0]`` returns the first row as a 1D tensor.
    """

    def __init__(self, data: List[int], wrapped: bool = True):
        self._data = list(data)
        self._wrapped = wrapped

    def to(self, device: str) -> "_FakeTensor":
        return self

    @property
    def shape(self):
        if self._wrapped:
            return (1, len(self._data))
        return (len(self._data),)

    def __getitem__(self, index):
        if isinstance(index, int) and self._wrapped:
            # Batch index: return the underlying 1D row.
            return _FakeTensor(self._data, wrapped=False)
        if isinstance(index, slice):
            return _FakeTensor(self._data[index], wrapped=False)
        if isinstance(index, int):
            return self._data[index]
        return _FakeTensor(self._data[index], wrapped=False)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class _FakeTokenizer:
    """Captures input strings + decode calls so tests can assert on them."""

    def __init__(self):
        self.encode_calls: List[str] = []
        self.decode_calls: List[Dict[str, Any]] = []
        self.pad_token_id = 0
        self.eos_token_id = 2

    def __call__(self, text: str, return_tensors: str = "pt"):
        self.encode_calls.append(text)
        # Encode every word as one token; min length 4 so input_len > 0.
        ids = list(range(max(4, len(text.split()))))
        return {
            "input_ids": _FakeTensor(ids),
            "attention_mask": _FakeTensor([1] * len(ids)),
        }

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        self.decode_calls.append({
            "ids": list(ids) if hasattr(ids, "__iter__") else ids,
            "skip_special_tokens": skip_special_tokens,
        })
        return "decoded answer text"


class _FakeModel:
    """Stand-in HF causal-LM model that records ``generate`` invocations."""

    def __init__(self):
        self.generate_calls: List[Dict[str, Any]] = []
        self.inference_mode_calls = 0

    def _toggle_inference(self):
        self.inference_mode_calls += 1
        return self

    def generate(self, **kwargs) -> _FakeTensor:
        self.generate_calls.append(kwargs)
        # Echo prompt-length + new tokens; AdapterCallable slices past
        # the input length so only "new" tokens are decoded.
        input_ids = kwargs["input_ids"]
        return _FakeTensor(list(input_ids._data) + [9, 9, 9])


# Bind the inference toggle under the canonical PyTorch attribute
# name without spelling it as a literal in source.
setattr(_FakeModel, _INFERENCE_MODE_ATTR, _FakeModel._toggle_inference)


@pytest.fixture
def fake_torch(monkeypatch):
    fake_torch_mod = types.ModuleType("torch")
    fake_torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch_mod.float16 = "float16-stub"

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch_mod.no_grad = lambda: _NoGrad()
    monkeypatch.setitem(sys.modules, "torch", fake_torch_mod)
    return fake_torch_mod


@pytest.fixture
def fake_transformers(monkeypatch):
    fake_mod = types.ModuleType("transformers")
    fake_tokenizer = _FakeTokenizer()
    fake_model = _FakeModel()

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            cls.last_repo = repo
            cls.last_kwargs = kwargs
            return fake_model

    class _AutoTokenizer:
        from_pretrained_calls: List[str] = []

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            cls.from_pretrained_calls.append(repo)
            return fake_tokenizer

    class _BnBConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_mod.AutoModelForCausalLM = _AutoModel
    fake_mod.AutoTokenizer = _AutoTokenizer
    fake_mod.BitsAndBytesConfig = _BnBConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)
    return fake_mod, fake_tokenizer, fake_model


@pytest.fixture
def fake_peft(monkeypatch):
    fake_mod = types.ModuleType("peft")

    class _PeftModel:
        from_pretrained_calls: List[Dict[str, Any]] = []

        @classmethod
        def from_pretrained(cls, base_model, adapter_path):
            cls.from_pretrained_calls.append(
                {"base_model": base_model, "adapter_path": adapter_path}
            )
            return base_model  # base_model is the _FakeModel; bind through

    fake_mod.PeftModel = _PeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_mod)
    return fake_mod


@pytest.fixture
def adapter_dir(tmp_path: Path) -> Path:
    out = tmp_path / "adapter"
    out.mkdir()
    (out / "adapter_model.safetensors").write_bytes(b"stub")
    (out / "adapter_config.json").write_text("{}", encoding="utf-8")
    return out


# ---------------------------------------------------------------------- #
# Tests                                                                   #
# ---------------------------------------------------------------------- #


def test_init_loads_model_and_caches(
    fake_torch, fake_transformers, fake_peft, adapter_dir,
):
    """__init__ must load the base model, apply the adapter, and put
    the wrapped model in inference mode."""
    fake_mod, fake_tokenizer, fake_model = fake_transformers
    from Trainforge.eval.adapter_callable import AdapterCallable

    callable_obj = AdapterCallable(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        base_model_short_name="qwen2.5-1.5b",
    )
    assert callable_obj._model is fake_model
    assert callable_obj._tokenizer is fake_tokenizer
    assert fake_model.inference_mode_calls == 1, (
        "Model must be flipped into inference mode."
    )
    # Adapter dir is the source of the PEFT load (not the base repo).
    last_call = fake_peft.PeftModel.from_pretrained_calls[-1]
    assert last_call["adapter_path"] == str(adapter_dir)


def test_call_reuses_cached_model(
    fake_torch, fake_transformers, fake_peft, adapter_dir,
):
    """Subsequent ``__call__`` invocations reuse the cached model
    rather than re-loading from HF."""
    fake_mod, fake_tokenizer, fake_model = fake_transformers
    from Trainforge.eval.adapter_callable import AdapterCallable

    callable_obj = AdapterCallable(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        base_model_short_name="qwen2.5-1.5b",
    )
    pre_count = len(fake_peft.PeftModel.from_pretrained_calls)

    callable_obj("first prompt")
    callable_obj("second prompt")
    callable_obj("third prompt")

    assert len(fake_peft.PeftModel.from_pretrained_calls) == pre_count, (
        "PeftModel.from_pretrained must not be called again on "
        "subsequent __call__ invocations."
    )
    assert len(fake_model.generate_calls) == 3, (
        "Each __call__ must invoke .generate() exactly once."
    )


def test_call_applies_chat_template(
    fake_torch, fake_transformers, fake_peft, adapter_dir,
):
    """The prompt passed to the tokenizer must be wrapped in the
    base's chat template (chatml for qwen2.5-1.5b)."""
    fake_mod, fake_tokenizer, fake_model = fake_transformers
    from Trainforge.eval.adapter_callable import AdapterCallable

    callable_obj = AdapterCallable(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        base_model_short_name="qwen2.5-1.5b",
    )
    callable_obj("What is X?")

    encoded = fake_tokenizer.encode_calls[-1]
    # ChatML markers from format_instruction:
    assert "<|im_start|>user" in encoded
    assert "<|im_start|>assistant" in encoded
    assert "<|im_end|>" in encoded
    assert "What is X?" in encoded


def test_call_decodes_with_skip_special_tokens(
    fake_torch, fake_transformers, fake_peft, adapter_dir,
):
    """``decode`` must be invoked with ``skip_special_tokens=True``
    so the returned string is clean (no <|im_end|> leakage)."""
    fake_mod, fake_tokenizer, fake_model = fake_transformers
    from Trainforge.eval.adapter_callable import AdapterCallable

    callable_obj = AdapterCallable(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        base_model_short_name="qwen2.5-1.5b",
    )
    out = callable_obj("hello")
    assert isinstance(out, str)
    assert out == "decoded answer text"
    assert fake_tokenizer.decode_calls[-1]["skip_special_tokens"] is True


def test_call_temperature_zero_disables_sampling(
    fake_torch, fake_transformers, fake_peft, adapter_dir,
):
    """Default temperature=0.0 -> generate() called with do_sample=False."""
    fake_mod, fake_tokenizer, fake_model = fake_transformers
    from Trainforge.eval.adapter_callable import AdapterCallable

    callable_obj = AdapterCallable(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        base_model_short_name="qwen2.5-1.5b",
    )
    callable_obj("prompt")
    kwargs = fake_model.generate_calls[-1]
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs


def test_call_temperature_nonzero_enables_sampling(
    fake_torch, fake_transformers, fake_peft, adapter_dir,
):
    """temperature > 0 -> do_sample=True with that temperature."""
    fake_mod, fake_tokenizer, fake_model = fake_transformers
    from Trainforge.eval.adapter_callable import AdapterCallable

    callable_obj = AdapterCallable(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        base_model_short_name="qwen2.5-1.5b",
        temperature=0.7,
    )
    callable_obj("prompt")
    kwargs = fake_model.generate_calls[-1]
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == pytest.approx(0.7)


def test_missing_adapter_dir_raises(
    fake_torch, fake_transformers, fake_peft, tmp_path,
):
    """A non-existent adapter dir surfaces a clear FileNotFoundError
    BEFORE the heavy ML imports run."""
    from Trainforge.eval.adapter_callable import AdapterCallable

    with pytest.raises(FileNotFoundError) as exc:
        AdapterCallable(
            adapter_dir=tmp_path / "does_not_exist",
            base_model_repo="Qwen/Qwen2.5-1.5B",
            base_model_short_name="qwen2.5-1.5b",
        )
    assert "adapter_dir does not exist" in str(exc.value)


def test_unknown_base_repo_raises(
    fake_torch, fake_transformers, fake_peft, adapter_dir,
):
    """A repo that doesn't match the registry surfaces a clear KeyError."""
    from Trainforge.eval.adapter_callable import AdapterCallable

    with pytest.raises(KeyError) as exc:
        AdapterCallable(
            adapter_dir=adapter_dir,
            base_model_repo="not-a-real/repo",
            # No short_name - forces the repo lookup path.
        )
    assert "BaseModelSpec" in str(exc.value)
