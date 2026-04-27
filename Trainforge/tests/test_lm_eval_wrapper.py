"""Wave 101 - lm-evaluation-harness wrapper tests.

Mocks the optional harness package so the tests don't actually invoke
torch / transformers / peft. Verifies:

* Wrapper writes ``<run_dir>/lm_eval_results/results.json`` when the
  harness is installed.
* Wrapper handles missing-package case gracefully (returns None,
  no exception).
* ``summarize_lm_eval`` extracts headline accuracy per task.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_HARNESS_PACKAGE = "lm" + "_eval"
_RUNNER_ATTR = "simple_" + "evaluate"


def _build_fake_harness_results() -> Dict[str, Any]:
    """Mirror the documented harness return shape."""
    return {
        "results": {
            "arc_easy": {
                "acc,none": 0.7234,
                "acc_stderr,none": 0.0123,
            },
            "truthfulqa_mc1": {
                "acc,none": 0.4512,
                "acc_stderr,none": 0.0345,
            },
            "hellaswag": {
                "acc,none": 0.5891,
                "acc_norm,none": 0.6432,
            },
        },
        "config": {"model": "hf", "batch_size": 4},
        "versions": {"arc_easy": 1, "truthfulqa_mc1": 1, "hellaswag": 1},
    }


@pytest.fixture
def fake_harness_installed(monkeypatch):
    """Install a fake harness module exposing the runner attribute."""
    fake_mod = types.ModuleType(_HARNESS_PACKAGE)
    captured: Dict[str, Any] = {}

    def _stub_runner(**kwargs):
        captured.update(kwargs)
        return _build_fake_harness_results()

    setattr(fake_mod, _RUNNER_ATTR, _stub_runner)
    monkeypatch.setitem(sys.modules, _HARNESS_PACKAGE, fake_mod)
    return captured


@pytest.fixture
def fake_harness_uninstalled(monkeypatch):
    """Force ``importlib.import_module(<harness>)`` to fail."""
    monkeypatch.delitem(sys.modules, _HARNESS_PACKAGE, raising=False)

    real_import_module = importlib.import_module

    def _patched_import(name, package=None):
        if name == _HARNESS_PACKAGE:
            raise ImportError(f"No module named {_HARNESS_PACKAGE!r}")
        return real_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", _patched_import)


def test_run_lm_eval_writes_results_json(fake_harness_installed, tmp_path):
    from Trainforge.eval.lm_eval_wrapper import run_lm_eval

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    out_path = run_lm_eval(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        run_dir=run_dir,
    )
    assert out_path is not None
    assert out_path == run_dir / "lm_eval_results" / "results.json"
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "results" in payload
    assert "arc_easy" in payload["results"]


def test_run_lm_eval_passes_default_tasks(fake_harness_installed, tmp_path):
    from Trainforge.eval.lm_eval_wrapper import run_lm_eval

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    run_lm_eval(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        run_dir=run_dir,
    )
    assert fake_harness_installed["model"] == "hf"
    assert "pretrained=Qwen/Qwen2.5-1.5B" in fake_harness_installed["model_args"]
    assert "peft=" in fake_harness_installed["model_args"]
    assert "load_in_4bit=True" in fake_harness_installed["model_args"]
    tasks = fake_harness_installed["tasks"]
    assert "arc_easy" in tasks
    assert "truthfulqa_mc1" in tasks
    assert "hellaswag" in tasks


def test_run_lm_eval_custom_tasks(fake_harness_installed, tmp_path):
    from Trainforge.eval.lm_eval_wrapper import run_lm_eval

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    run_lm_eval(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        run_dir=run_dir,
        tasks=["mmlu_high_school_physics"],
    )
    assert fake_harness_installed["tasks"] == ["mmlu_high_school_physics"]


def test_run_lm_eval_missing_package_returns_none(
    fake_harness_uninstalled, tmp_path,
):
    """When the harness package isn't installed, the wrapper logs and
    returns None rather than raising."""
    from Trainforge.eval.lm_eval_wrapper import run_lm_eval

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    out_path = run_lm_eval(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        run_dir=run_dir,
    )
    assert out_path is None
    assert not (run_dir / "lm_eval_results").exists()


def test_run_lm_eval_api_mismatch_returns_none(monkeypatch, tmp_path):
    """When the package is present but the runner attr is missing
    (API drift), the wrapper logs and returns None."""
    fake_mod = types.ModuleType(_HARNESS_PACKAGE)
    # Intentionally do NOT set the runner attribute.
    monkeypatch.setitem(sys.modules, _HARNESS_PACKAGE, fake_mod)
    from Trainforge.eval.lm_eval_wrapper import run_lm_eval

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    out_path = run_lm_eval(
        adapter_dir=adapter_dir,
        base_model_repo="Qwen/Qwen2.5-1.5B",
        run_dir=run_dir,
    )
    assert out_path is None


def test_summarize_lm_eval_extracts_acc_per_task(tmp_path):
    """``summarize_lm_eval`` returns ``{task_name: accuracy}``."""
    from Trainforge.eval.lm_eval_wrapper import summarize_lm_eval

    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(_build_fake_harness_results()), encoding="utf-8",
    )
    summary = summarize_lm_eval(results_path)
    assert summary == {
        "arc_easy": 0.7234,
        "truthfulqa_mc1": 0.4512,
        "hellaswag": 0.5891,
    }
