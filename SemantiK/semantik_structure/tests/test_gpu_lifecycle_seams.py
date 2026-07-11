"""SemantiK cross-venv GPU-lifecycle twin + cascade-seam tests.

Contract under test:

  * The SemantiK twin (``semantik_structure.gpu_lifecycle``) has NO Ed4All import
    (it must be importable in the SemantiK venv, which lacks ``lib/``).
  * The twin's ollama sweep enumerates a mocked ``/api/ps`` and POSTs
    ``keep_alive:0`` per resident model; fail-soft; returns evicted names.
  * ``resolve_gpu_lifecycle_mode`` default-ON / falsey-off / garbage-ON.
  * ``stage_lease`` releases on clean exit + exception; no-op when off.
  * The cascade's ``_gpu_lifecycle_release`` seam helper gates on the flag
    (flag-off → ZERO release calls) and dispatches the requested arms.
  * The cascade wires the release seams in the correct RELATIVE order — NOT
    between Stage-5d and Stage-5e (same-lease), and covering the
    second-pass+ocr_repair window as one lease — asserted against the module
    source so no live model / GPU is needed.

No live model / no live ollama — httpx + torch are mocked.
"""

from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import semantik_structure.gpu_lifecycle as twin


# --------------------------------------------------------------------------
# import isolation — the twin must not pull in Ed4All's lib/
# --------------------------------------------------------------------------
def test_twin_has_no_ed4all_import() -> None:
    src = Path(twin.__file__).read_text()
    tree = ast.parse(src)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
    # No module-level import may reference Ed4All's lib package or the Ed4All
    # gpu_lifecycle original (the whole point of the twin).
    for name in imported:
        assert not name.startswith("lib"), f"twin imports Ed4All lib: {name}"
        assert "vram_reclaim" not in name
        assert not name.startswith("MCP"), f"twin imports MCP: {name}"


# --------------------------------------------------------------------------
# fake httpx
# --------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload=None, *, raise_on_status=False):
        self._payload = payload
        self._raise = raise_on_status

    def raise_for_status(self):
        if self._raise:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, ps_models, *, get_raises=False):
        self._ps_models = ps_models
        self._get_raises = get_raises
        self.unloaded = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url):
        if self._get_raises:
            raise RuntimeError("refused")
        return _FakeResp({"models": self._ps_models})

    def post(self, url, json=None):
        assert json.get("keep_alive") == 0
        self.unloaded.append(json.get("model"))
        return _FakeResp({})


def _install_fake_httpx(monkeypatch, client):
    fake = types.ModuleType("httpx")
    fake.Client = lambda *a, **k: client
    monkeypatch.setitem(sys.modules, "httpx", fake)


# --------------------------------------------------------------------------
# twin parse semantics + arms
# --------------------------------------------------------------------------
def test_mode_default_on(monkeypatch):
    monkeypatch.delenv("ED4ALL_GPU_LIFECYCLE", raising=False)
    assert twin.resolve_gpu_lifecycle_mode() is True


@pytest.mark.parametrize("val", ["0", "false", "off", "no"])
def test_mode_falsey_off(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", val)
    assert twin.resolve_gpu_lifecycle_mode() is False


@pytest.mark.parametrize("val", ["1", "on", "garbage", ""])
def test_mode_truthy_garbage_on(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", val)
    assert twin.resolve_gpu_lifecycle_mode() is True


def test_twin_root_resolution_prefers_specialist_base_url(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BASE_URL", "http://box:1234/v1")
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    # /v1 stripped to the native root.
    assert twin.resolve_ollama_root() == "http://box:1234"


def test_twin_release_ollama_unloads_each(monkeypatch):
    client = _FakeClient([{"name": "a"}, {"name": "b"}])
    _install_fake_httpx(monkeypatch, client)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BASE_URL", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    assert twin.release_ollama_models() == ["a", "b"]
    assert client.unloaded == ["a", "b"]


def test_twin_release_ollama_failsoft(monkeypatch):
    client = _FakeClient([{"name": "a"}], get_raises=True)
    _install_fake_httpx(monkeypatch, client)
    assert twin.release_ollama_models() == []


def test_twin_release_torch_failsoft_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    twin.release_torch()  # no raise


def test_twin_stage_lease_releases_and_noop(monkeypatch):
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "on")
    ollama = MagicMock(return_value=[])
    torch = MagicMock()
    monkeypatch.setattr(twin, "release_ollama_models", ollama)
    monkeypatch.setattr(twin, "release_torch", torch)

    with twin.stage_lease("s"):
        pass
    ollama.assert_called_once()
    torch.assert_called_once()

    ollama.reset_mock()
    torch.reset_mock()
    # exception path still releases
    with pytest.raises(ValueError):
        with twin.stage_lease("s"):
            raise ValueError("x")
    ollama.assert_called_once()
    torch.assert_called_once()

    # off → no-op
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "off")
    ollama.reset_mock()
    torch.reset_mock()
    with twin.stage_lease("s"):
        pass
    ollama.assert_not_called()
    torch.assert_not_called()


# --------------------------------------------------------------------------
# cascade seam helper — gating
# --------------------------------------------------------------------------
def test_cascade_release_helper_gates_on_flag(monkeypatch):
    from semantik_structure import cascade

    ollama = MagicMock(return_value=[])
    torch = MagicMock()
    monkeypatch.setattr(twin, "release_ollama_models", ollama)
    monkeypatch.setattr(twin, "release_torch", torch)

    # flag ON → the requested arms fire
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "on")
    cascade._gpu_lifecycle_release(ollama=True, torch=True, stage="seam")
    ollama.assert_called_once()
    torch.assert_called_once()

    # flag OFF → zero release calls
    ollama.reset_mock()
    torch.reset_mock()
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "off")
    cascade._gpu_lifecycle_release(ollama=True, torch=True, stage="seam")
    ollama.assert_not_called()
    torch.assert_not_called()


def test_cascade_release_helper_failsoft(monkeypatch):
    from semantik_structure import cascade

    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "on")
    monkeypatch.setattr(
        twin, "release_ollama_models", MagicMock(side_effect=RuntimeError("x"))
    )
    # A seam release failure must never crash the cascade.
    cascade._gpu_lifecycle_release(ollama=True, stage="seam")


def test_cascade_release_helper_arms_selectable(monkeypatch):
    from semantik_structure import cascade

    ollama = MagicMock(return_value=[])
    torch = MagicMock()
    monkeypatch.setattr(twin, "release_ollama_models", ollama)
    monkeypatch.setattr(twin, "release_torch", torch)
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "on")

    cascade._gpu_lifecycle_release(torch=True, stage="captioner")
    ollama.assert_not_called()
    torch.assert_called_once()


# --------------------------------------------------------------------------
# seam placement (source-level; no live cascade run)
# --------------------------------------------------------------------------
def _cascade_source() -> str:
    from semantik_structure import cascade

    return Path(cascade.__file__).read_text()


def test_ollama_seam_after_stage5e_before_stage6_gguf():
    src = _cascade_source()
    # The post-Stage-5e ollama release sits AFTER release_council_gpu() and
    # BEFORE the first _run_inner("fast") (the Stage-6 GGUF entry).
    i_release_council = src.index("release_council_gpu()")
    i_ollama_seam = src.index('stage="post-Stage-5e/pre-Stage-6"')
    i_run_inner_fast = src.index('_run_inner("fast")')
    assert i_release_council < i_ollama_seam < i_run_inner_fast


def test_no_ollama_release_between_5d_and_5e():
    # Same-lease assertion: there is NO ollama release seam textually between
    # the Stage-5d reviewer dispatch and the Stage-5e resegment section — the
    # 5d+5e run is one ollama-consumer window.
    src = _cascade_source()
    # The Stage-5d reviewer log line and the Stage-5e enter log line bracket
    # the 5d+5e window.
    i_5d = src.index("running Stage 5d (70B structure review")
    i_5e = src.index("Stage 5e — block JOIN / SPLIT")
    window = src[i_5d:i_5e]
    assert "_gpu_lifecycle_release" not in window


def test_ollama_seam_covers_second_pass_and_ocr_repair_before_stage13():
    src = _cascade_source()
    # OCR-repair block, then the ollama lease #2, then Stage 13.
    i_ocr = src.index("running OCR-confusable micro-repair pass")
    i_seam = src.index('stage="post-second-pass+ocr_repair"')
    i_stage13 = src.index("Stage 13 — orchestrate the offline retry")
    assert i_ocr < i_seam < i_stage13


def test_theta_torch_seam_after_stage12():
    src = _cascade_source()
    i_stage12 = src.index('stages[f"stage12{suffix}"]')
    i_theta_seam = src.index("post-Stage-12-theta[")
    i_lane_outputs = src.index("lane_outputs[lane] = {")
    assert i_stage12 < i_theta_seam < i_lane_outputs
