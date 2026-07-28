"""Training-setup audit 2026-07-21 — nemotron3-nano-30b enablement fixes.

Covers, CPU-only (all heavy ML deps faked, mirroring
``test_peft_trainer_trl_api.py``):

* the new ``configs/nemotron3-nano-30b.yaml`` (load_config no longer
  FileNotFoundError; target-module list has NO gate_proj / bare
  up_proj / down_proj),
* ``trust_remote_code`` threading (spec-driven, True only for nemotron),
* the ``fit_dpo`` NameError regression (``bf16_ok``/``cuda_ok`` were
  referenced in ``dpo_config_kwargs`` before definition),
* the Mamba fast-path loud assertion (nemotron_h + missing kernels →
  RuntimeError; other bases untouched),
* the ``gradient_checkpointing`` knob + train-time thinking-off
  (``chat_template_kwargs={"enable_thinking": False}``) plumbing through
  the accepted-kwargs shim,
* a snapshot-render check of the real nemotron chat_template.jinja via
  jinja2 (skipped when the snapshot / jinja2 is unavailable).
"""
from __future__ import annotations

import glob
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.training.base_models import BaseModelRegistry  # noqa: E402
from Trainforge.training.configs import load_config  # noqa: E402
from Trainforge.training.peft_trainer import (  # noqa: E402
    PEFTTrainer,
    _missing_mamba_kernel_packages,
)


# ---------------------------------------------------------------------- #
# Fake-ML harness (SFT + DPO)                                             #
# ---------------------------------------------------------------------- #


def _install_fake_ml(
    monkeypatch,
    *,
    labels=None,
    sft_config_extra_fields=(),
    auto_config_model_type="qwen2",
):
    """Install fake torch/peft/transformers/trl/datasets modules.

    ``sft_config_extra_fields`` lets a test opt the fake SFTConfig
    signature into newer knobs (``gradient_checkpointing``,
    ``chat_template_kwargs`` …) so the accepted-kwargs shim keeps them.
    """
    captured = {}
    labels = labels if labels is not None else [[-100, -100, 5, 6]]

    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def add_callback(self, cb):
            pass

        def get_train_dataloader(self):
            return [{"labels": labels}]

        def train(self, *a, **k):
            captured["trained"] = True
            captured.setdefault("train_calls", []).append(
                {"args": a, "kwargs": k}
            )

        def save_model(self, path):
            (Path(path) / "adapter_model.safetensors").write_bytes(b"x")

    class _FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

        @classmethod
        def from_pretrained(cls, *a, **k):
            captured.setdefault("tokenizer_from_pretrained", []).append(
                {"args": a, "kwargs": k}
            )
            return cls()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            captured.setdefault("model_from_pretrained", []).append(
                {"args": a, "kwargs": k}
            )
            return cls()

    class _FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, *a, **k):
            captured.setdefault("auto_config_calls", []).append(
                {"args": a, "kwargs": k}
            )
            return types.SimpleNamespace(model_type=auto_config_model_type)

    class _FakePeftModel:
        @classmethod
        def from_pretrained(cls, base, path, **k):
            captured["peft_model_from_pretrained"] = {"path": path, "kwargs": k}
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
            captured["lora_kwargs"] = k

    # Base modern-trl SFTConfig surface (max_length + completion_only_loss),
    # extendable per-test with newer knobs.
    base_fields = [
        "output_dir", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "learning_rate", "warmup_ratio",
        "weight_decay", "optim", "bf16", "fp16", "seed", "max_length",
        "save_strategy", "logging_steps", "save_total_limit",
        "completion_only_loss",
    ] + list(sft_config_extra_fields)

    ns = {}
    args_sig = ", ".join(f"{f}=None" for f in base_fields)
    exec(  # noqa: S102 - test-only dynamic signature builder
        f"def __init__(self, {args_sig}):\n"
        f"    captured['sft_config_kwargs'] = {{\n"
        + "".join(f"        {f!r}: {f},\n" for f in base_fields)
        + "    }\n",
        {"captured": captured},
        ns,
    )
    _FakeSFTConfig = type("_FakeSFTConfig", (), {"__init__": ns["__init__"]})

    class _FakeDPOConfig:
        def __init__(self, output_dir=None, num_train_epochs=None,
                     per_device_train_batch_size=None,
                     gradient_accumulation_steps=None, learning_rate=None,
                     warmup_ratio=None, weight_decay=None, seed=None,
                     bf16=None, fp16=None, optim=None, max_length=None,
                     gradient_checkpointing=None,
                     gradient_checkpointing_kwargs=None,
                     chat_template_kwargs=None, save_strategy=None):
            captured["dpo_config_kwargs"] = {
                "bf16": bf16, "fp16": fp16, "optim": optim,
                "output_dir": output_dir,
                "max_length": max_length,
                "gradient_checkpointing": gradient_checkpointing,
                "gradient_checkpointing_kwargs":
                    gradient_checkpointing_kwargs,
                "chat_template_kwargs": chat_template_kwargs,
            }

    monkeypatch.setattr(
        "Trainforge.training.peft_trainer._require_training_deps_impl",
        lambda *, require_bnb: None,
    )
    monkeypatch.setattr(
        "Trainforge.training.peft_trainer._assert_supported_runtime",
        lambda: {},
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
    fake_peft.PeftModel = _FakePeftModel
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _FakeModel
    fake_transformers.AutoTokenizer = _FakeTokenizer
    fake_transformers.AutoConfig = _FakeAutoConfig
    fake_transformers.BitsAndBytesConfig = _FakeBnB
    fake_trl = types.ModuleType("trl")
    fake_trl.SFTTrainer = _FakeTrainer
    fake_trl.SFTConfig = _FakeSFTConfig
    fake_trl.DPOTrainer = _FakeTrainer
    fake_trl.DPOConfig = _FakeDPOConfig
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _FakeDataset

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    return captured


# ---------------------------------------------------------------------- #
# Fix 1 — per-base YAML config                                            #
# ---------------------------------------------------------------------- #


def test_nemotron_config_loads_and_pins_recipe():
    cfg = load_config("nemotron3-nano-30b")
    assert cfg.base_model == "nemotron3-nano-30b"
    assert cfg.learning_rate == pytest.approx(1e-4)
    assert cfg.epochs == 3
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.max_seq_length == 4096
    assert cfg.batch_size == 1
    assert cfg.gradient_accumulation_steps == 16
    # bf16 LoRA only — QLoRA unsupported on the hybrid arch.
    assert cfg.use_4bit is False
    # Fix 4: activation checkpointing on for the 30B fit ONLY.
    assert cfg.gradient_checkpointing is True
    # Sibling-parity orchestration knobs.
    assert cfg.save_total_limit == 3
    assert cfg.completion_only_loss is True
    assert cfg.verify_loss_mask is True


def test_nemotron_target_modules_exclude_moe_and_gate():
    cfg = load_config("nemotron3-nano-30b")
    assert cfg.target_modules == [
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj",
    ]
    # gate_proj doesn't exist on nemotron_h; bare up_proj/down_proj would
    # wrap ~5,900 per-expert Linears in the MoE blocks.
    for forbidden in ("gate_proj", "up_proj", "down_proj"):
        assert forbidden not in cfg.target_modules


def test_sibling_configs_unaffected_by_gc_knob():
    cfg = load_config("qwen2.5-1.5b")
    assert cfg.gradient_checkpointing is False


# ---------------------------------------------------------------------- #
# Fix 2 — trust_remote_code threading                                     #
# ---------------------------------------------------------------------- #


def test_spec_trust_remote_code_only_on_nemotron():
    assert BaseModelRegistry.resolve("nemotron3-nano-30b").trust_remote_code is True
    for name in ("qwen2.5-1.5b", "llama-3.2-1b", "llama-3.2-3b",
                 "smollm2-1.7b", "phi-3.5-mini"):
        assert BaseModelRegistry.resolve(name).trust_remote_code is False


def test_fit_sft_threads_trust_remote_code_for_nemotron(tmp_path, monkeypatch):
    captured = _install_fake_ml(
        monkeypatch, auto_config_model_type="nemotron_h",
    )
    # Kernels "present" so the preflight passes.
    monkeypatch.setattr(
        "Trainforge.training.peft_trainer._missing_mamba_kernel_packages",
        lambda: [],
    )
    trainer = PEFTTrainer(
        base_model="nemotron3-nano-30b",
        training_config={
            "epochs": 1, "batch_size": 1, "dpo_learning_rate": 1e-6,
        },
    )
    out = tmp_path / "run"
    out.mkdir()
    trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    assert captured["tokenizer_from_pretrained"][0]["kwargs"][
        "trust_remote_code"
    ] is True
    assert captured["model_from_pretrained"][0]["kwargs"][
        "trust_remote_code"
    ] is True


def test_fit_sft_trust_remote_code_false_for_qwen(tmp_path, monkeypatch):
    captured = _install_fake_ml(monkeypatch)
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    out.mkdir()
    trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    assert captured["tokenizer_from_pretrained"][0]["kwargs"][
        "trust_remote_code"
    ] is False
    assert captured["model_from_pretrained"][0]["kwargs"][
        "trust_remote_code"
    ] is False


# ---------------------------------------------------------------------- #
# Fix 3 — fit_dpo NameError regression                                    #
# ---------------------------------------------------------------------- #


def test_fit_dpo_config_names_resolve(tmp_path, monkeypatch):
    """bf16_ok / cuda_ok must be defined BEFORE dpo_config_kwargs.

    Pre-fix this raised ``NameError: name 'bf16_ok' is not defined`` on
    every real fit_dpo invocation.
    """
    captured = _install_fake_ml(monkeypatch)
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    out.mkdir()
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    returned = trainer.fit_dpo(
        [{"prompt": "Q?", "chosen": "good", "rejected": "bad"}],
        sft_dir,
        out,
    )
    assert returned == out / "adapter_model.safetensors"
    assert returned.exists()
    # The hoisted flags landed in the DPOConfig (CPU fake → both False).
    assert captured["dpo_config_kwargs"]["bf16"] is False
    assert captured["dpo_config_kwargs"]["fp16"] is False
    assert captured["dpo_config_kwargs"]["optim"] == "adamw_torch"


def test_fit_dpo_threads_trust_remote_code_on_base_model(tmp_path, monkeypatch):
    captured = _install_fake_ml(
        monkeypatch, auto_config_model_type="nemotron_h",
    )
    monkeypatch.setattr(
        "Trainforge.training.peft_trainer._missing_mamba_kernel_packages",
        lambda: [],
    )
    trainer = PEFTTrainer(
        base_model="nemotron3-nano-30b",
        training_config={
            "epochs": 1, "batch_size": 1, "dpo_learning_rate": 1e-6,
        },
    )
    out = tmp_path / "run"
    out.mkdir()
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    trainer.fit_dpo(
        [{"prompt": "Q?", "chosen": "good", "rejected": "bad"}],
        sft_dir,
        out,
    )
    # Both tokenizer loads (adapter-dir first) + the base-model load carry it.
    for call in captured["tokenizer_from_pretrained"]:
        assert call["kwargs"]["trust_remote_code"] is True
    assert captured["model_from_pretrained"][0]["kwargs"][
        "trust_remote_code"
    ] is True


# ---------------------------------------------------------------------- #
# Fix 5 — Mamba fast-path loud assertion                                  #
# ---------------------------------------------------------------------- #


def test_mamba_fast_path_missing_raises_for_nemotron(monkeypatch):
    _install_fake_ml(monkeypatch, auto_config_model_type="nemotron_h")
    monkeypatch.setattr(
        "Trainforge.training.peft_trainer._missing_mamba_kernel_packages",
        lambda: ["mamba_ssm", "causal_conv1d"],
    )
    trainer = PEFTTrainer(
        base_model="nemotron3-nano-30b",
        training_config={},
    )
    with pytest.raises(RuntimeError) as exc:
        trainer._assert_mamba_fast_path()
    msg = str(exc.value)
    assert "mamba_ssm" in msg
    assert "causal_conv1d" in msg
    assert "aarch64" in msg


def test_mamba_autoconfig_permission_error_fails_loud(monkeypatch):
    _install_fake_ml(monkeypatch, auto_config_model_type="nemotron_h")
    from transformers import AutoConfig

    def _permission_denied(*args, **kwargs):
        raise PermissionError("generated module cache is read-only")

    monkeypatch.setattr(AutoConfig, "from_pretrained", _permission_denied)
    monkeypatch.setenv("HF_MODULES_CACHE", "/portable/cache/modules")
    trainer = PEFTTrainer(
        base_model="nemotron3-nano-30b",
        training_config={},
    )
    with pytest.raises(RuntimeError) as exc:
        trainer._assert_mamba_fast_path()
    msg = str(exc.value)
    assert "AutoConfig preflight failed" in msg
    assert "PermissionError" in msg
    assert "generated module cache is read-only" in msg
    assert "HF_MODULES_CACHE" in msg
    assert "cbd3fa9f933d55ef16a84236559f4ee2a0526848" in msg


def test_mamba_autoconfig_identity_mismatch_fails_loud(monkeypatch):
    _install_fake_ml(monkeypatch, auto_config_model_type="unexpected_arch")
    trainer = PEFTTrainer(
        base_model="nemotron3-nano-30b",
        training_config={},
    )
    with pytest.raises(RuntimeError, match="identity mismatch") as exc:
        trainer._assert_mamba_fast_path()
    assert "expected model_type='nemotron_h'" in str(exc.value)
    assert "unexpected_arch" in str(exc.value)


def test_mamba_fast_path_does_not_fire_for_qwen(monkeypatch):
    _install_fake_ml(monkeypatch)
    monkeypatch.setattr(
        "Trainforge.training.peft_trainer._missing_mamba_kernel_packages",
        lambda: ["mamba_ssm", "causal_conv1d"],
    )
    trainer = PEFTTrainer(base_model="qwen2.5-1.5b", training_config={})
    trainer._assert_mamba_fast_path()  # no raise — spec opts out


def test_fit_sft_aborts_before_load_when_kernels_missing(tmp_path, monkeypatch):
    captured = _install_fake_ml(
        monkeypatch, auto_config_model_type="nemotron_h",
    )
    monkeypatch.setattr(
        "Trainforge.training.peft_trainer._missing_mamba_kernel_packages",
        lambda: ["mamba_ssm"],
    )
    trainer = PEFTTrainer(
        base_model="nemotron3-nano-30b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(RuntimeError, match="mamba_ssm"):
        trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    # Aborted BEFORE any weight/tokenizer load and before training.
    assert "model_from_pretrained" not in captured
    assert "trained" not in captured


def test_missing_mamba_probe_real_import_shape():
    # The probe validates the exact fused symbols, including binary/ABI import
    # failures, and must always attribute a failure to one of the two packages.
    missing = _missing_mamba_kernel_packages()
    assert all(
        item.startswith(("mamba_ssm", "causal_conv1d"))
        for item in missing
    )


# ---------------------------------------------------------------------- #
# Stage-isolated native checkpoints                                      #
# ---------------------------------------------------------------------- #


def test_sft_and_dpo_use_disjoint_checkpoint_namespaces(tmp_path, monkeypatch):
    captured = _install_fake_ml(monkeypatch)
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    out.mkdir()

    # An existing SFT checkpoint must not make a fresh DPO trainer resume.
    sft_checkpoint = out / ".trainer_checkpoints" / "sft" / "checkpoint-9"
    sft_checkpoint.mkdir(parents=True)
    sft_dir = tmp_path / "sft-adapter"
    sft_dir.mkdir()
    trainer.fit_dpo(
        [{"prompt": "Q?", "chosen": "good", "rejected": "bad"}],
        sft_dir,
        out,
    )
    assert captured["dpo_config_kwargs"]["output_dir"] == str(
        out / ".trainer_checkpoints" / "dpo"
    )
    assert captured["train_calls"][-1]["kwargs"] == {}


def test_dpo_resumes_only_its_own_checkpoint(tmp_path, monkeypatch):
    captured = _install_fake_ml(monkeypatch)
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    (out / ".trainer_checkpoints" / "dpo" / "checkpoint-4").mkdir(
        parents=True
    )
    sft_dir = tmp_path / "sft-adapter"
    sft_dir.mkdir()
    trainer.fit_dpo(
        [{"prompt": "Q?", "chosen": "good", "rejected": "bad"}],
        sft_dir,
        out,
    )
    assert captured["train_calls"][-1]["kwargs"] == {
        "resume_from_checkpoint": True
    }


def test_sft_ignores_dpo_checkpoint_and_uses_sft_namespace(
    tmp_path, monkeypatch
):
    captured = _install_fake_ml(monkeypatch)
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    (out / ".trainer_checkpoints" / "dpo" / "checkpoint-12").mkdir(
        parents=True
    )
    trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    assert captured["sft_config_kwargs"]["output_dir"] == str(
        out / ".trainer_checkpoints" / "sft"
    )
    assert captured["train_calls"][-1]["kwargs"] == {}


def test_sft_resumes_only_its_own_checkpoint(tmp_path, monkeypatch):
    captured = _install_fake_ml(monkeypatch)
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    (out / ".trainer_checkpoints" / "sft" / "checkpoint-3").mkdir(
        parents=True
    )
    trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    assert captured["train_calls"][-1]["kwargs"] == {
        "resume_from_checkpoint": True
    }


def test_legacy_shared_checkpoint_fails_loud_without_moving_it(
    tmp_path, monkeypatch
):
    captured = _install_fake_ml(monkeypatch)
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    legacy = out / "checkpoint-7"
    legacy.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="ambiguous legacy trainer checkpoints"):
        trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    assert legacy.is_dir()
    assert "trained" not in captured


# ---------------------------------------------------------------------- #
# Fix 4 + 6 — gradient checkpointing + thinking-off plumbing              #
# ---------------------------------------------------------------------- #


def test_gc_and_thinking_off_pass_through_when_supported(tmp_path, monkeypatch):
    captured = _install_fake_ml(
        monkeypatch,
        sft_config_extra_fields=(
            "gradient_checkpointing",
            "gradient_checkpointing_kwargs",
            "chat_template_kwargs",
        ),
    )
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={
            "epochs": 1, "batch_size": 1, "gradient_checkpointing": True,
        },
    )
    out = tmp_path / "run"
    out.mkdir()
    trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    kwargs = captured["sft_config_kwargs"]
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    # Train-time thinking-off rides the completion-only branch.
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert kwargs["completion_only_loss"] is True


def test_gc_and_thinking_off_dropped_on_old_band(tmp_path, monkeypatch):
    # Fake SFTConfig WITHOUT the new fields (the currently-installed trl
    # 0.12 band): the shim must drop the keys, not crash.
    captured = _install_fake_ml(monkeypatch)
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={
            "epochs": 1, "batch_size": 1, "gradient_checkpointing": True,
        },
    )
    out = tmp_path / "run"
    out.mkdir()
    trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    kwargs = captured["sft_config_kwargs"]
    assert "gradient_checkpointing" not in kwargs
    assert "chat_template_kwargs" not in kwargs


def test_gc_off_by_default(tmp_path, monkeypatch):
    captured = _install_fake_ml(
        monkeypatch,
        sft_config_extra_fields=(
            "gradient_checkpointing", "gradient_checkpointing_kwargs",
        ),
    )
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    out = tmp_path / "run"
    out.mkdir()
    trainer.fit_sft([{"prompt": "Q?", "completion": "A."}], out)
    # Default off: the key is never SET (fake captures the None default).
    assert captured["sft_config_kwargs"]["gradient_checkpointing"] is None


# ---------------------------------------------------------------------- #
# Fix 6 — snapshot chat-template render (thinking-off semantics)          #
# ---------------------------------------------------------------------- #

_SNAPSHOT_TEMPLATE_GLOB = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    / "snapshots/*/chat_template.jinja"
)


def _load_snapshot_template():
    jinja2 = pytest.importorskip("jinja2")
    matches = glob.glob(str(_SNAPSHOT_TEMPLATE_GLOB))
    if not matches:
        pytest.skip("nemotron snapshot chat_template.jinja not present")
    source = Path(matches[0]).read_text(encoding="utf-8")
    env = jinja2.Environment(  # noqa: S701 - plain-text chat template
        trim_blocks=True, lstrip_blocks=True,
    )
    try:
        return env.from_string(source)
    except jinja2.TemplateError as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"installed jinja2 cannot compile the template: {exc}")


def test_snapshot_template_thinking_off_renders_closed_think():
    """enable_thinking=False must never open a dangling ``<think>`` block.

    NB: literal absence of the ``<think>`` token is impossible with this
    template — the DISABLED sentinel is the closed empty pair
    ``<think></think>`` (the template injects it on every assistant turn
    and on the generation prompt). The training-relevant property is that
    no think block is left OPEN (an open ``<think>\\n`` invites reasoning
    tokens into the completion), so that is what we assert.
    """
    template = _load_snapshot_template()
    rendered = template.render(
        messages=[
            {"role": "user", "content": "What is a derivative?"},
            {"role": "assistant", "content": "The instantaneous rate of change."},
            {"role": "user", "content": "And an integral?"},
        ],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Generation prompt carries the CLOSED sentinel, not an open block.
    assert rendered.endswith("<think></think>")
    assert "<think>\n" not in rendered
    # Every <think> in the rendered example is immediately closed.
    assert rendered.count("<think>") == rendered.count("<think></think>")
    # No reasoning content leaked between think tags anywhere.
    assert "</think>\n" not in rendered.rsplit("<|im_start|>assistant", 1)[-1]


def test_snapshot_template_thinking_on_control():
    # Control arm: default/enabled thinking DOES open a dangling block —
    # proving the enable_thinking kwarg is load-bearing, not a no-op.
    template = _load_snapshot_template()
    rendered = template.render(
        messages=[{"role": "user", "content": "What is a derivative?"}],
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert rendered.endswith("<think>\n")
