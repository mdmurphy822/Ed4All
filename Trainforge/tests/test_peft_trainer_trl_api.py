"""SFT-D S9 — trl 1.x API tolerance + completion-only loss-mask verification.

CPU-only, no trl / torch required (the fit_sft path is exercised with fully
faked ML modules, mirroring test_training_runner.test_fit_sft_adapter_filename).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.training.peft_trainer import (  # noqa: E402
    PEFTTrainer,
    _accepted_kwarg,
    _assert_completion_only_masked,
    _filter_accepted,
)


# ---------------------------------------------------------------------- #
# _accepted_kwarg                                                         #
# ---------------------------------------------------------------------- #


def test_accepted_kwarg_prefers_first_match_old_api():
    class Old:
        def __init__(self, tokenizer=None):
            pass

    assert _accepted_kwarg(
        Old, ["processing_class", "tokenizer"], default="processing_class",
        surface="x",
    ) == "tokenizer"


def test_accepted_kwarg_new_api():
    class New:
        def __init__(self, processing_class=None):
            pass

    assert _accepted_kwarg(
        New, ["processing_class", "tokenizer"], default="processing_class",
        surface="x",
    ) == "processing_class"


def test_accepted_kwarg_var_kwargs_returns_default():
    class Fake:
        def __init__(self, *a, **k):
            pass

    assert _accepted_kwarg(
        Fake, ["processing_class", "tokenizer"], default="processing_class",
        surface="x",
    ) == "processing_class"


def test_accepted_kwarg_no_match_no_varkw_raises():
    class Unsupported:
        def __init__(self, something_else=None):
            pass

    with pytest.raises(RuntimeError) as exc:
        _accepted_kwarg(
            Unsupported, ["processing_class", "tokenizer"],
            default="processing_class", surface="trl.SFTTrainer",
        )
    assert "trl.SFTTrainer" in str(exc.value)


def test_filter_accepted_drops_unknown_on_real_sig():
    class Cfg:
        def __init__(self, a=None, b=None):
            pass

    out = _filter_accepted(Cfg, {"a": 1, "b": 2, "completion_only_loss": True})
    assert out == {"a": 1, "b": 2}


def test_filter_accepted_keeps_all_on_varkw():
    class Cfg:
        def __init__(self, **k):
            pass

    payload = {"a": 1, "completion_only_loss": True}
    assert _filter_accepted(Cfg, payload) == payload


# ---------------------------------------------------------------------- #
# _assert_completion_only_masked                                          #
# ---------------------------------------------------------------------- #


def test_loss_mask_ok_when_prompt_masked():
    # row: prompt masked (-100) + live completion tokens.
    rows = [[-100, -100, 5, 6, 7], [-100, 9, 10]]
    _assert_completion_only_masked(rows)  # no raise


def test_loss_mask_raises_when_nothing_masked():
    rows = [[1, 2, 3, 4], [5, 6, 7]]
    with pytest.raises(RuntimeError) as exc:
        _assert_completion_only_masked(rows)
    assert "prompt" in str(exc.value).lower()


def test_loss_mask_raises_on_empty_batch():
    with pytest.raises(RuntimeError):
        _assert_completion_only_masked([])


def test_loss_mask_raises_when_row_fully_masked():
    rows = [[-100, -100, -100]]
    with pytest.raises(RuntimeError) as exc:
        _assert_completion_only_masked(rows)
    assert "entirely masked" in str(exc.value).lower()


# ---------------------------------------------------------------------- #
# fit_sft with faked trl (new processing_class API + loss-mask verify)   #
# ---------------------------------------------------------------------- #


def _install_fake_ml(monkeypatch, *, sftconfig_accepts_completion_only, labels):
    """Install fake torch/peft/transformers/trl/datasets modules.

    ``labels`` is what the fake trainer's dataloader yields as batch["labels"].
    ``sftconfig_accepts_completion_only`` toggles whether the fake SFTConfig's
    signature exposes completion_only_loss (drives the completion-only branch).
    """
    captured = {}

    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def add_callback(self, cb):
            pass

        def get_train_dataloader(self):
            return [{"labels": labels}]

        def train(self, *a, **k):
            pass

        def save_model(self, path):
            (Path(path) / "adapter_model.safetensors").write_bytes(b"x")

    class _FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    class _FakeBnB:
        def __init__(self, *a, **k):
            pass

    class _FakeDataset:
        @classmethod
        def from_dict(cls, mapping):
            captured["dataset_cols"] = sorted(mapping.keys())
            return cls()

    class _FakeLoraConfig:
        def __init__(self, *a, **k):
            pass

    if sftconfig_accepts_completion_only:
        class _FakeSFTConfig:
            def __init__(self, output_dir=None, num_train_epochs=None,
                         per_device_train_batch_size=None,
                         gradient_accumulation_steps=None, learning_rate=None,
                         warmup_ratio=None, weight_decay=None, optim=None,
                         bf16=None, fp16=None, seed=None, max_length=None,
                         save_strategy=None, logging_steps=None,
                         save_total_limit=None, completion_only_loss=None):
                captured["sft_config_kwargs"] = {
                    "completion_only_loss": completion_only_loss,
                    "max_length": max_length,
                }
    else:
        class _FakeSFTConfig:
            def __init__(self, output_dir=None, num_train_epochs=None,
                         per_device_train_batch_size=None,
                         gradient_accumulation_steps=None, learning_rate=None,
                         warmup_ratio=None, weight_decay=None, optim=None,
                         bf16=None, fp16=None, seed=None, max_seq_length=None,
                         save_strategy=None, logging_steps=None,
                         save_total_limit=None):
                captured["sft_config_kwargs"] = {
                    "max_seq_length": max_seq_length,
                }

    monkeypatch.setattr(
        "Trainforge.training.peft_trainer._require_training_deps_impl",
        lambda *, require_bnb: None,
    )

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
    )
    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = _FakeLoraConfig
    fake_peft.prepare_model_for_kbit_training = lambda m: m
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _FakeModel
    fake_transformers.AutoTokenizer = _FakeTokenizer
    fake_transformers.BitsAndBytesConfig = _FakeBnB
    fake_trl = types.ModuleType("trl")
    fake_trl.SFTTrainer = _FakeTrainer
    fake_trl.SFTConfig = _FakeSFTConfig
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _FakeDataset

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    return captured


def test_fit_sft_uses_processing_class_and_completion_only(tmp_path, monkeypatch):
    captured = _install_fake_ml(
        monkeypatch,
        sftconfig_accepts_completion_only=True,
        labels=[[-100, -100, 5, 6]],  # masked prompt + live completion
    )
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    out.mkdir()
    returned = trainer.fit_sft(
        [{"prompt": "Q?", "completion": "A."}], out,
    )
    assert returned == out / "adapter_model.safetensors"
    assert returned.exists()
    # New API: processing_class chosen (fake trainer takes **kwargs -> default).
    assert "processing_class" in captured["trainer_kwargs"]
    # Completion-only branch: dataset carried prompt+completion columns.
    assert captured["dataset_cols"] == ["completion", "prompt"]
    assert captured["sft_config_kwargs"]["completion_only_loss"] is True
    # max_length (new SFTConfig field) resolved, not max_seq_length.
    assert captured["sft_config_kwargs"]["max_length"] == 2048


def test_fit_sft_falls_back_to_text_when_no_completion_only(tmp_path, monkeypatch):
    captured = _install_fake_ml(
        monkeypatch,
        sftconfig_accepts_completion_only=False,
        labels=[[1, 2, 3]],  # unmasked — but verify is skipped in text mode
    )
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    out.mkdir()
    trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    # Old SFTConfig lacks completion_only_loss -> text column path, and the
    # loss-mask verify does NOT fire (so unmasked labels don't raise).
    assert captured["dataset_cols"] == ["text"]
    assert "max_seq_length" in captured["sft_config_kwargs"]


def test_fit_sft_loss_mask_verify_fails_loud(tmp_path, monkeypatch):
    _install_fake_ml(
        monkeypatch,
        sftconfig_accepts_completion_only=True,
        labels=[[1, 2, 3, 4]],  # NOTHING masked -> verification must raise
    )
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(RuntimeError) as exc:
        trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    assert "loss-mask verification" in str(exc.value).lower()
