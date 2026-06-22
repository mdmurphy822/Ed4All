"""R7 VRAM-OOM mitigation — theta device-resolution unit tests.

Exercises ``dart_semantic.theta.semantic_preservation``'s
``SEMANTIK_THETA_DEVICE`` resolution + the ``_place_theta_on_device``
graceful-CUDA-fallback helper WITHOUT any GPU, real torch, or the theta
model artifacts. The vendored SemantiK ``dart_semantic.theta`` package
``__init__`` eagerly imports the cascade's heavy deps (axe / llama-cpp /
torch), which are absent from Ed4All's slim venv, so we load the
``semantic_preservation`` SUBMODULE under its real qualified name behind
empty namespace-package shims — exactly enough for the dataclass +
resolver + device-placement helper to import.

Run CPU-pinned (the project default):
  ED4ALL_NLI_DEVICE=cpu python -m pytest \
    lib/semantik/tests/test_theta_device.py -q
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Load the vendored semantic_preservation submodule behind namespace shims.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SEMANTIK_THETA = _REPO_ROOT / "SemantiK" / "dart_semantic" / "theta"
_SP_PATH = _SEMANTIK_THETA / "semantic_preservation.py"


def _load_sp():
    """Import dart_semantic.theta.semantic_preservation in isolation.

    Registers minimal empty namespace packages for ``dart_semantic`` and
    ``dart_semantic.theta`` (pointing at the real vendored dirs) so the
    submodule imports under its real ``__name__`` — required because the
    module defines a frozen dataclass whose field annotations are resolved
    against ``sys.modules[__module__]``. We do NOT run the package
    ``__init__`` (which pulls the heavy cascade deps).
    """
    if "dart_semantic.theta.semantic_preservation" in sys.modules:
        return sys.modules["dart_semantic.theta.semantic_preservation"]

    for name, path in (
        ("dart_semantic", _REPO_ROOT / "SemantiK" / "dart_semantic"),
        ("dart_semantic.theta", _SEMANTIK_THETA),
    ):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
            pkg.__package__ = name
            sys.modules[name] = pkg

    spec = importlib.util.spec_from_file_location(
        "dart_semantic.theta.semantic_preservation", str(_SP_PATH)
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sp = _load_sp()


# ---------------------------------------------------------------------------
# Fakes — a minimal torch + a model/head recording .to()/.half() calls.
# ---------------------------------------------------------------------------


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(self, *, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)


class _RecordingModule:
    """Stands in for the theta model / head; records device + dtype moves."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.device = "cpu"
        self.halved = False

    def to(self, device):  # noqa: ANN001
        self.calls.append(f"to:{device}")
        self.device = device
        return self

    def half(self):
        self.calls.append("half")
        self.halved = True
        return self


# ---------------------------------------------------------------------------
# resolve_theta_device — precedence chain + default.
# ---------------------------------------------------------------------------


def test_resolve_default_is_cpu(monkeypatch):
    monkeypatch.delenv(sp.ENV_THETA_DEVICE, raising=False)
    monkeypatch.delenv("DART_THETA_DEVICE", raising=False)
    assert sp.resolve_theta_device() == "cpu"


def test_resolve_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv(sp.ENV_THETA_DEVICE, "cuda")
    assert sp.resolve_theta_device("cuda:1") == "cuda:1"


def test_resolve_reads_semantik_env(monkeypatch):
    monkeypatch.delenv("DART_THETA_DEVICE", raising=False)
    monkeypatch.setenv(sp.ENV_THETA_DEVICE, "cuda")
    assert sp.resolve_theta_device() == "cuda"


def test_resolve_legacy_env_lower_precedence(monkeypatch):
    """DART_THETA_DEVICE is honored only when SEMANTIK_THETA_DEVICE is unset."""
    monkeypatch.delenv(sp.ENV_THETA_DEVICE, raising=False)
    monkeypatch.setenv("DART_THETA_DEVICE", "cuda:2")
    assert sp.resolve_theta_device() == "cuda:2"
    # SEMANTIK_THETA_DEVICE wins when both are present.
    monkeypatch.setenv(sp.ENV_THETA_DEVICE, "cpu")
    assert sp.resolve_theta_device() == "cpu"


def test_resolve_garbage_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("DART_THETA_DEVICE", raising=False)
    monkeypatch.setenv(sp.ENV_THETA_DEVICE, "   ")
    assert sp.resolve_theta_device() == "cpu"


# ---------------------------------------------------------------------------
# _place_theta_on_device — default CPU, CUDA half-cast, graceful fallback.
# ---------------------------------------------------------------------------


def test_place_cpu_no_half():
    model, head = _RecordingModule(), _RecordingModule()
    torch = _FakeTorch(cuda_available=True)
    used = sp._place_theta_on_device(model, head, torch, "cpu")
    assert used == "cpu"
    # CPU path: model/head moved to cpu, NEVER half-cast.
    assert model.halved is False and head.halved is False
    assert "half" not in model.calls and "half" not in head.calls


def test_place_cuda_available_moves_and_halves():
    model, head = _RecordingModule(), _RecordingModule()
    torch = _FakeTorch(cuda_available=True)
    used = sp._place_theta_on_device(model, head, torch, "cuda")
    assert used == "cuda"
    assert model.device == "cuda" and head.device == "cuda"
    assert model.halved is True and head.halved is True


def test_place_cuda_unavailable_falls_back_to_cpu():
    """cuda requested but torch.cuda.is_available() False → graceful CPU,
    no crash, no half-cast."""
    model, head = _RecordingModule(), _RecordingModule()
    torch = _FakeTorch(cuda_available=False)
    used = sp._place_theta_on_device(model, head, torch, "cuda")
    assert used == "cpu"
    assert model.device == "cpu" and head.device == "cpu"
    assert model.halved is False and head.halved is False


def test_place_cuda_move_error_falls_back_to_cpu():
    """An OOM/driver error during the .to()/.half() degrades to CPU."""

    class _ExplodingModule(_RecordingModule):
        def half(self):
            raise RuntimeError("CUDA out of memory (simulated)")

    model, head = _ExplodingModule(), _ExplodingModule()
    torch = _FakeTorch(cuda_available=True)
    used = sp._place_theta_on_device(model, head, torch, "cuda")
    assert used == "cpu"
    # After the failed GPU attempt, the helper re-pins to CPU.
    assert model.device == "cpu" and head.device == "cpu"


# ---------------------------------------------------------------------------
# Provenance — the resolved device is recorded on the score() breakdown.
# ---------------------------------------------------------------------------


def test_score_breakdown_records_theta_device():
    """The empty-doc fast path stamps theta_device from the model bundle so a
    CPU- vs CUDA-scored run is distinguishable (mirrors nli_device)."""

    class _FakeDoc:
        region_provenance: list = []
        sub_task_log: dict = {}
        html = ""

    class _FakeModel:
        version = "v8/test"
        device = "cuda:0"

    score_val, breakdown = sp.score(
        _FakeDoc(), feature_blocks=[], model=_FakeModel()
    )
    assert breakdown["theta_device"] == "cuda:0"
    assert breakdown["method"] == sp.METHOD_MODEL
