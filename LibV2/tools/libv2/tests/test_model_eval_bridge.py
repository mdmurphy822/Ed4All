"""Tests for the LibV2 fresh-eval bridge (Wave 92 deferral closed).

All CPU-only: the heavy :class:`AdapterCallable` model load is never
exercised — tests inject a fake ``model_callable`` (so the
:class:`SLMEvalHarness` wiring runs without GPU) and monkeypatch the
harness + adapter-callable classes. Course + model fixtures are built
under ``tmp_path`` with a SYNTHETIC slug (never a real LibV2 course).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from click.testing import CliRunner

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from LibV2.tools.libv2.evaluation import model_bridge as model_eval_bridge  # noqa: E402
from LibV2.tools.libv2.cli import main as libv2_main  # noqa: E402


_SLUG = "syn-eval-101"
_MODEL_ID = "qwen2-5-1-5b-syn-eval-101-deadbeef"


def _make_card(*, model_id: str = _MODEL_ID) -> Dict[str, Any]:
    return {
        "model_id": model_id,
        "course_slug": _SLUG,
        "base_model": {
            "name": "qwen2.5-1.5b",
            "revision": "abc123def",
            "huggingface_repo": "Qwen/Qwen2.5-1.5B",
        },
        "adapter_format": "safetensors",
        "created_at": "2026-07-07T00:00:00Z",
    }


def _build_course(tmp_path: Path, *, with_card: bool = True,
                  with_eval_config: bool = False,
                  eval_config_overrides: Dict[str, Any] | None = None) -> Path:
    """Stage a synthetic LibV2 repo with one course + one model dir."""
    repo_root = tmp_path / "libv2-repo"
    course_dir = repo_root / "courses" / _SLUG
    model_dir = course_dir / "models" / _MODEL_ID
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_model.safetensors").write_bytes(b"fake" * 32)
    if with_card:
        (model_dir / "model_card.json").write_text(
            json.dumps(_make_card(), indent=2), encoding="utf-8"
        )
    if with_eval_config:
        eval_dir = course_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "benchmark": "ed4all-bench",
            "benchmark_version": "1.0",
            "top_k": 5,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 321,
            "seed": 7,
            "prompt_template_file": "prompt_template.txt",
            "rubric_file": "rubric.md",
        }
        if eval_config_overrides:
            cfg.update(eval_config_overrides)
        import yaml as _yaml
        (eval_dir / "eval_config.yaml").write_text(
            _yaml.safe_dump(cfg), encoding="utf-8"
        )
        (eval_dir / "prompt_template.txt").write_text(
            "{context_section}\n{question}", encoding="utf-8"
        )
    return repo_root


class _FakeHarness:
    """Stand-in for SLMEvalHarness — records args, writes a fake report."""

    last_instance: "_FakeHarness | None" = None

    def __init__(self, course_path, model_callable, smoke_mode=False, **kwargs):
        self.course_path = Path(course_path)
        self.model_callable = model_callable
        self.smoke_mode = smoke_mode
        self.profile_name = "generic"
        self.run_all_output = None
        _FakeHarness.last_instance = self

    def run_all(self, output_path=None):
        self.run_all_output = Path(output_path)
        Path(output_path).write_text(
            json.dumps({
                "faithfulness": 0.66,
                "coverage": 0.55,
                "profile": self.profile_name,
                "smoke_mode": self.smoke_mode,
            }),
            encoding="utf-8",
        )
        return Path(output_path)


@pytest.fixture
def patch_harness(monkeypatch):
    """Replace SLMEvalHarness with the fake in the module run_fresh_eval imports."""
    import Trainforge.eval.runners.slm_eval_harness as harness_mod
    monkeypatch.setattr(harness_mod, "SLMEvalHarness", _FakeHarness)
    _FakeHarness.last_instance = None
    return _FakeHarness


def _fake_callable(prompt: str) -> str:
    return "answer"


# --------------------------------------------------------------------- #
# run_fresh_eval — output-path + injection behavior                       #
# --------------------------------------------------------------------- #


def test_fresh_eval_default_is_nondestructive(tmp_path, patch_harness):
    repo_root = _build_course(tmp_path)
    model_dir = repo_root / "courses" / _SLUG / "models" / _MODEL_ID
    # Pre-existing canonical report must survive.
    canonical = model_dir / "eval_report.json"
    canonical.write_text(json.dumps({"faithfulness": 0.11}), encoding="utf-8")

    report_path = model_eval_bridge.run_fresh_eval(
        _SLUG, _MODEL_ID, repo_root, model_callable=_fake_callable,
    )

    assert report_path.name.startswith("eval_report.fresh-")
    assert report_path.exists()
    # Canonical untouched.
    assert json.loads(canonical.read_text())["faithfulness"] == 0.11
    # Injected callable + smoke flag threaded into the harness.
    assert patch_harness.last_instance.model_callable is _fake_callable
    assert patch_harness.last_instance.smoke_mode is False


def test_fresh_eval_smoke_flag_threads_through(tmp_path, patch_harness):
    repo_root = _build_course(tmp_path)
    model_eval_bridge.run_fresh_eval(
        _SLUG, _MODEL_ID, repo_root, model_callable=_fake_callable, smoke=True,
    )
    assert patch_harness.last_instance.smoke_mode is True


def test_fresh_eval_replace_backs_up_canonical(tmp_path, patch_harness):
    repo_root = _build_course(tmp_path)
    model_dir = repo_root / "courses" / _SLUG / "models" / _MODEL_ID
    canonical = model_dir / "eval_report.json"
    canonical.write_text(json.dumps({"faithfulness": 0.11}), encoding="utf-8")

    report_path = model_eval_bridge.run_fresh_eval(
        _SLUG, _MODEL_ID, repo_root, model_callable=_fake_callable, replace=True,
    )

    assert report_path == canonical
    # Fresh scores written to the canonical path.
    assert json.loads(canonical.read_text())["faithfulness"] == 0.66
    # Prior canonical preserved as .bak.
    backup = model_dir / "eval_report.json.bak"
    assert backup.exists()
    assert json.loads(backup.read_text())["faithfulness"] == 0.11


def test_fresh_eval_replace_no_prior_no_backup(tmp_path, patch_harness):
    repo_root = _build_course(tmp_path)
    model_dir = repo_root / "courses" / _SLUG / "models" / _MODEL_ID

    report_path = model_eval_bridge.run_fresh_eval(
        _SLUG, _MODEL_ID, repo_root, model_callable=_fake_callable, replace=True,
    )
    assert report_path == model_dir / "eval_report.json"
    assert not (model_dir / "eval_report.json.bak").exists()


def test_fresh_eval_explicit_output_path(tmp_path, patch_harness):
    repo_root = _build_course(tmp_path)
    out = tmp_path / "custom_report.json"
    report_path = model_eval_bridge.run_fresh_eval(
        _SLUG, _MODEL_ID, repo_root, model_callable=_fake_callable, output_path=out,
    )
    assert report_path == out
    assert out.exists()


def test_fresh_eval_missing_course_raises(tmp_path, patch_harness):
    repo_root = _build_course(tmp_path)
    with pytest.raises(model_eval_bridge.FreshEvalError):
        model_eval_bridge.run_fresh_eval(
            "no-such-course", _MODEL_ID, repo_root, model_callable=_fake_callable,
        )


def test_fresh_eval_missing_model_raises(tmp_path, patch_harness):
    repo_root = _build_course(tmp_path)
    with pytest.raises(model_eval_bridge.FreshEvalError):
        model_eval_bridge.run_fresh_eval(
            _SLUG, "no-such-model", repo_root, model_callable=_fake_callable,
        )


# --------------------------------------------------------------------- #
# Decision capture fires                                                  #
# --------------------------------------------------------------------- #


def test_fresh_eval_capture_fires(tmp_path, patch_harness, monkeypatch):
    monkeypatch.setenv("ED4ALL_TRAINING_CAPTURES_DIR", str(tmp_path / "captures"))
    # Fail closed if fresh_eval_invocation ever drops out of the canonical enum.
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
    from lib.decision_capture import DecisionCapture

    repo_root = _build_course(tmp_path)
    capture = DecisionCapture(course_code=_SLUG, phase="libv2-indexing", tool="libv2")
    n_before = len(capture.decisions)

    model_eval_bridge.run_fresh_eval(
        _SLUG, _MODEL_ID, repo_root, model_callable=_fake_callable, capture=capture,
    )

    events = [d for d in capture.decisions
              if d.get("decision_type") == "fresh_eval_invocation"]
    assert len(capture.decisions) > n_before
    assert len(events) == 1
    rationale = events[0]["rationale"]
    # Dynamic rationale interpolates real per-call signals.
    assert _MODEL_ID in rationale
    assert "Qwen/Qwen2.5-1.5B" in rationale
    assert len(rationale) >= 20


def test_fresh_eval_capture_failure_never_fails_eval(tmp_path, patch_harness):
    repo_root = _build_course(tmp_path)

    class _BoomCapture:
        def log_decision(self, *a, **k):
            raise RuntimeError("capture boom")

    # A capture that raises must not abort the eval (best-effort contract).
    report_path = model_eval_bridge.run_fresh_eval(
        _SLUG, _MODEL_ID, repo_root, model_callable=_fake_callable,
        capture=_BoomCapture(),
    )
    assert report_path.exists()


# --------------------------------------------------------------------- #
# build_adapter_callable — card -> AdapterCallable kwarg mapping          #
# --------------------------------------------------------------------- #


class _RecorderCallable:
    last_kwargs: Dict[str, Any] | None = None

    def __init__(self, **kwargs):
        _RecorderCallable.last_kwargs = kwargs

    def __call__(self, prompt: str) -> str:  # pragma: no cover — never called
        return ""


def test_build_adapter_callable_maps_card_and_gen_knobs(tmp_path, monkeypatch):
    import Trainforge.eval.retrieval.adapter_callable as ac_mod
    from Trainforge.eval.eval_config import load_eval_config
    monkeypatch.setattr(ac_mod, "AdapterCallable", _RecorderCallable)
    _RecorderCallable.last_kwargs = None

    repo_root = _build_course(tmp_path, with_eval_config=True)
    course_dir = repo_root / "courses" / _SLUG
    model_dir = course_dir / "models" / _MODEL_ID
    loaded = load_eval_config(course_dir)

    model_eval_bridge.build_adapter_callable(model_dir, loaded)

    kw = _RecorderCallable.last_kwargs
    assert kw is not None
    assert kw["base_model_repo"] == "Qwen/Qwen2.5-1.5B"
    assert kw["base_model_short_name"] == "qwen2.5-1.5b"
    assert kw["revision"] == "abc123def"
    assert Path(kw["adapter_dir"]) == model_dir
    # Generation knobs pulled from the per-course eval_config.yaml.
    assert kw["max_new_tokens"] == 321
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 0.9
    assert kw["seed"] == 7


def test_build_adapter_callable_missing_card_raises(tmp_path, monkeypatch):
    import Trainforge.eval.retrieval.adapter_callable as ac_mod
    from Trainforge.eval.eval_config import load_eval_config
    monkeypatch.setattr(ac_mod, "AdapterCallable", _RecorderCallable)

    repo_root = _build_course(tmp_path, with_card=False)
    course_dir = repo_root / "courses" / _SLUG
    model_dir = course_dir / "models" / _MODEL_ID
    loaded = load_eval_config(course_dir)

    with pytest.raises(model_eval_bridge.FreshEvalError):
        model_eval_bridge.build_adapter_callable(model_dir, loaded)


# --------------------------------------------------------------------- #
# CLI surface — --fresh / judge=none / ImportError guidance               #
# --------------------------------------------------------------------- #


def test_cli_models_eval_fresh_routes_through_bridge(tmp_path, monkeypatch):
    repo_root = _build_course(tmp_path)
    model_dir = repo_root / "courses" / _SLUG / "models" / _MODEL_ID

    def _fake_run(course_slug, model_id, repo_root, **kwargs):
        out = model_dir / "eval_report.fresh-test.json"
        out.write_text(json.dumps({"faithfulness": 0.42}), encoding="utf-8")
        assert kwargs["smoke"] is True
        assert kwargs["replace"] is False
        return out

    monkeypatch.setattr(model_eval_bridge, "run_fresh_eval", _fake_run)

    result = CliRunner().invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "eval", _SLUG, _MODEL_ID, "--fresh", "--smoke",
    ])
    assert result.exit_code == 0, result.output
    assert "0.42" in result.output


def test_cli_models_eval_fresh_importerror_guidance(tmp_path, monkeypatch):
    repo_root = _build_course(tmp_path)

    def _boom(*a, **k):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(model_eval_bridge, "run_fresh_eval", _boom)

    result = CliRunner().invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "eval", _SLUG, _MODEL_ID, "--fresh",
    ])
    assert result.exit_code == 1
    assert "[training]" in result.output
    assert "gpu_guard" in result.output


def test_cli_models_eval_default_still_cached(tmp_path, monkeypatch):
    """Without --fresh, the command prints the cached report (no bridge call)."""
    repo_root = _build_course(tmp_path)
    model_dir = repo_root / "courses" / _SLUG / "models" / _MODEL_ID
    (model_dir / "eval_report.json").write_text(
        json.dumps({"faithfulness": 0.9}), encoding="utf-8"
    )

    called = {"n": 0}

    def _should_not_call(*a, **k):
        called["n"] += 1
        raise AssertionError("bridge must not run without --fresh")

    monkeypatch.setattr(model_eval_bridge, "run_fresh_eval", _should_not_call)

    result = CliRunner().invoke(libv2_main, [
        "--repo", str(repo_root),
        "models", "eval", _SLUG, _MODEL_ID,
    ])
    assert result.exit_code == 0, result.output
    assert called["n"] == 0
    assert "faithfulness" in result.output


def test_cli_eval_run_judge_none_routes_through_bridge(tmp_path, monkeypatch):
    repo_root = _build_course(tmp_path)
    model_dir = repo_root / "courses" / _SLUG / "models" / _MODEL_ID

    calls = {}

    def _fake_run(course_slug, model_id, repo_root, **kwargs):
        calls["slug"] = course_slug
        calls["model_id"] = model_id
        out = model_dir / "eval_report.fresh-run.json"
        out.write_text(json.dumps({"faithfulness": 0.5}), encoding="utf-8")
        return out

    monkeypatch.setattr(model_eval_bridge, "run_fresh_eval", _fake_run)

    result = CliRunner().invoke(libv2_main, [
        "--repo", str(repo_root),
        "eval", "run", _SLUG, _MODEL_ID,  # judge defaults to "none"
    ])
    assert result.exit_code == 0, result.output
    assert calls["slug"] == _SLUG
    assert calls["model_id"] == _MODEL_ID
