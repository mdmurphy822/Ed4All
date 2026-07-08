"""Unit tests for :mod:`lib.llm.vram_doctor`.

Hermetic + GPU-free: every external (torch, the NVML/torch VRAM probe, the
ollama ``/api/ps`` HTTP surface, the env resolvers, the runs dir) is
monkeypatched, so the suite needs no GPU and no ollama server. Covers the
flag idiom, the snapshot degrade paths, every fit-check outcome branch, the
never-raising trajectory writer, and the report renderer.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from lib.llm import vram_doctor
from lib.llm.vram_doctor import (
    ENV_EMBEDDING_DEVICE,
    ENV_VRAM_DOCTOR,
    FitVerdict,
    VramSnapshot,
    fit_check,
    format_doctor_report,
    snapshot_vram,
    vram_doctor_enabled,
    write_trajectory_row,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        ENV_VRAM_DOCTOR,
        ENV_EMBEDDING_DEVICE,
        "ED4ALL_NLI_DEVICE",
        "ED4ALL_NLI_MIN_FREE_VRAM_MIB",
        "ED4ALL_NLI_EVICT_FOR_CUDA",
        "LOCAL_SYNTHESIS_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------- #
# vram_doctor_enabled — parse-with-fallback flag idiom
# --------------------------------------------------------------------- #


def test_enabled_off_by_default() -> None:
    assert vram_doctor_enabled() is False


@pytest.mark.parametrize("token", ["1", "true", "TRUE", "yes", "On", "  on  "])
def test_enabled_truthy_tokens(monkeypatch: pytest.MonkeyPatch, token: str) -> None:
    monkeypatch.setenv(ENV_VRAM_DOCTOR, token)
    assert vram_doctor_enabled() is True


@pytest.mark.parametrize("token", ["0", "false", "no", "off", "", "garbage", "2"])
def test_enabled_falsey_and_garbage(
    monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setenv(ENV_VRAM_DOCTOR, token)
    assert vram_doctor_enabled() is False


def test_enabled_explicit_value_wins() -> None:
    assert vram_doctor_enabled("on") is True
    assert vram_doctor_enabled("nonsense") is False


# --------------------------------------------------------------------- #
# Fake ollama /api/ps HTTP surface
# --------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.get_calls: List[str] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get(self, url: str) -> _FakeResponse:
        self.get_calls.append(url)
        return _FakeResponse(self._payload)


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch, client: Any) -> None:
    fake_httpx = MagicMock()
    fake_httpx.Client.return_value = client
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)


class _FakeTorch:
    """Minimal torch stand-in exposing only ``cuda.is_available`` / mem_get_info."""

    def __init__(self, available: bool = True, total_bytes: int = 8 * 1024**3) -> None:
        self._available = available
        self._total_bytes = total_bytes

        class _Cuda:
            def __init__(self, outer: "_FakeTorch") -> None:
                self._outer = outer

            def is_available(self) -> bool:
                return self._outer._available

            def mem_get_info(self, _dev: Any = None) -> tuple[int, int]:
                # (free, total) bytes — only total is read by the doctor.
                return (0, self._outer._total_bytes)

        self.cuda = _Cuda(self)


# --------------------------------------------------------------------- #
# snapshot_vram
# --------------------------------------------------------------------- #


def test_snapshot_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch present + CUDA + probe + /api/ps → fully populated snapshot."""
    fake_torch = _FakeTorch(available=True, total_bytes=8 * 1024**3)
    monkeypatch.setattr(vram_doctor, "_import_torch", lambda: fake_torch)
    monkeypatch.setattr(
        vram_doctor, "probe_free_vram_mib_with_source", lambda *a, **k: (200, "nvml")
    )
    client = _FakeClient(
        {"models": [{"name": "qwen2.5:7b", "size_vram": 5_300 * 1024 * 1024}]}
    )
    _install_fake_httpx(monkeypatch, client)

    snap = snapshot_vram()
    assert snap.cuda_available is True
    assert snap.free_mib == 200
    assert snap.total_mib == 8 * 1024  # 8 GiB in MiB
    assert snap.probe_source == "nvml"
    assert snap.error is None
    assert snap.resident_models == [{"name": "qwen2.5:7b", "vram_mib": 5_300}]
    assert client.get_calls == ["http://localhost:11434/api/ps"]


def test_snapshot_torch_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch missing → cuda_available False, no crash, ollama still probed."""
    monkeypatch.setattr(vram_doctor, "_import_torch", lambda: None)
    client = _FakeClient({"models": []})
    _install_fake_httpx(monkeypatch, client)

    snap = snapshot_vram()
    assert snap.cuda_available is False
    assert snap.free_mib is None
    assert snap.total_mib is None
    assert snap.probe_source == "unavailable"
    assert snap.resident_models == []


def test_snapshot_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch present but no CUDA device → degraded snapshot, no probe."""
    monkeypatch.setattr(
        vram_doctor, "_import_torch", lambda: _FakeTorch(available=False)
    )
    _install_fake_httpx(monkeypatch, _FakeClient({"models": []}))

    snap = snapshot_vram()
    assert snap.cuda_available is False
    assert snap.free_mib is None
    assert snap.probe_source == "unavailable"


def test_snapshot_ollama_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """/api/ps raises → resident_models [] + error set, never raises."""
    monkeypatch.setattr(vram_doctor, "_import_torch", lambda: None)

    class _DownClient(_FakeClient):
        def get(self, url: str) -> _FakeResponse:
            raise ConnectionError("connection refused")

    _install_fake_httpx(monkeypatch, _DownClient({"models": []}))

    snap = snapshot_vram()
    assert snap.resident_models == []
    assert snap.error is not None
    assert "unreachable" in snap.error


def test_snapshot_httpx_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx import failure → [] + error, no crash."""
    monkeypatch.setattr(vram_doctor, "_import_torch", lambda: None)
    real_import = __import__

    def _no_httpx(name: str, *a: Any, **k: Any) -> Any:
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", _no_httpx)
    monkeypatch.delitem(sys.modules, "httpx", raising=False)

    snap = snapshot_vram()
    assert snap.resident_models == []
    assert snap.error is not None and "httpx" in snap.error


# --------------------------------------------------------------------- #
# Finding #4: probe_source reflects the REAL backend, not a re-derive
# --------------------------------------------------------------------- #


def _install_succeeding_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import pynvml; nvmlInit()`` succeed so a NAIVE re-derive of the
    probe source would (wrongly) stamp ``"nvml"`` regardless of the real
    backend. The new design must IGNORE this and trust the primitive's tag.
    """
    import types

    fake_pynvml = types.SimpleNamespace()
    fake_pynvml.nvmlInit = lambda: None
    fake_pynvml.nvmlShutdown = lambda: None
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)


def test_snapshot_probe_source_is_torch_when_value_is_torch_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primitive says the value came from the torch fallback → the
    snapshot stamps ``probe_source="torch"`` EVEN THOUGH an NVML init would
    succeed (the old _classify_probe_source re-derive bug)."""
    fake_torch = _FakeTorch(available=True, total_bytes=8 * 1024**3)
    monkeypatch.setattr(vram_doctor, "_import_torch", lambda: fake_torch)
    _install_succeeding_nvml(monkeypatch)
    monkeypatch.setattr(
        vram_doctor,
        "probe_free_vram_mib_with_source",
        lambda *a, **k: (5419, "torch"),
    )
    _install_fake_httpx(monkeypatch, _FakeClient({"models": []}))

    snap = snapshot_vram()
    assert snap.free_mib == 5419
    assert snap.probe_source == "torch"  # NOT "nvml" despite the NVML init


def test_snapshot_probe_source_unavailable_when_value_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primitive returns ``(None, "unavailable")`` → probe_source unavailable."""
    fake_torch = _FakeTorch(available=True, total_bytes=8 * 1024**3)
    monkeypatch.setattr(vram_doctor, "_import_torch", lambda: fake_torch)
    monkeypatch.setattr(
        vram_doctor,
        "probe_free_vram_mib_with_source",
        lambda *a, **k: (None, "unavailable"),
    )
    _install_fake_httpx(monkeypatch, _FakeClient({"models": []}))

    snap = snapshot_vram()
    assert snap.free_mib is None
    assert snap.probe_source == "unavailable"


def test_snapshot_probe_source_forced_unavailable_when_value_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant guard: value None coerces probe_source to 'unavailable' even
    if a (hypothetical) primitive returned a non-unavailable tag with None."""
    fake_torch = _FakeTorch(available=True, total_bytes=8 * 1024**3)
    monkeypatch.setattr(vram_doctor, "_import_torch", lambda: fake_torch)
    monkeypatch.setattr(
        vram_doctor,
        "probe_free_vram_mib_with_source",
        lambda *a, **k: (None, "nvml"),
    )
    _install_fake_httpx(monkeypatch, _FakeClient({"models": []}))

    snap = snapshot_vram()
    assert snap.free_mib is None
    assert snap.probe_source == "unavailable"


# --------------------------------------------------------------------- #
# Finding #8: the doctor + the evictor share the /api/ps parser; the
# doctor passes a SHORT timeout (latency-sensitive).
# --------------------------------------------------------------------- #


def test_doctor_and_evictor_share_ps_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both the doctor's resident-model query and the evictor's name list go
    through ``vram_reclaim._fetch_resident_models`` on the SAME /api/ps body."""
    from lib.llm import vram_reclaim
    from lib.llm.vram_reclaim import _fetch_resident_models, _list_running_models

    # The doctor imports the shared parser from vram_reclaim — same object.
    assert vram_doctor._fetch_resident_models is _fetch_resident_models

    payload = {
        "models": [{"name": "qwen2.5:7b", "size_vram": 5_300 * 1024 * 1024}]
    }
    client = _FakeClient(payload)
    entries = _fetch_resident_models(client, "http://localhost:11434")
    # Shared shape carries name + raw size_vram bytes.
    assert entries == [
        {"name": "qwen2.5:7b", "size_vram": 5_300 * 1024 * 1024}
    ]
    # Evictor projects to names only.
    assert _list_running_models(_FakeClient(payload), "http://localhost:11434") == [
        "qwen2.5:7b"
    ]
    # Doctor projects to name + vram_mib.
    monkeypatch.setattr(vram_doctor, "_import_torch", lambda: None)
    _install_fake_httpx(monkeypatch, _FakeClient(payload))
    snap = snapshot_vram()
    assert snap.resident_models == [{"name": "qwen2.5:7b", "vram_mib": 5_300}]
    assert vram_reclaim is not None  # imported for the identity assertion above


def test_doctor_uses_short_ps_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The doctor must NOT inherit the evictor's 30s timeout — it passes a
    short one (inspected via the fake httpx.Client call args)."""
    monkeypatch.setattr(vram_doctor, "_import_torch", lambda: None)
    fake_httpx = MagicMock()
    fake_httpx.Client.return_value = _FakeClient({"models": []})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    snapshot_vram()

    fake_httpx.Client.assert_called_once()
    _args, kwargs = fake_httpx.Client.call_args
    assert kwargs.get("timeout") == vram_doctor._DOCTOR_PS_TIMEOUT_SECONDS
    assert vram_doctor._DOCTOR_PS_TIMEOUT_SECONDS <= 5.0  # short, not the 30s reclaim


# --------------------------------------------------------------------- #
# fit_check — every outcome branch
# --------------------------------------------------------------------- #


def _snap(
    *,
    free: Optional[int] = 200,
    cuda: bool = True,
    models: Optional[List[Dict[str, Any]]] = None,
) -> VramSnapshot:
    return VramSnapshot(
        free_mib=free,
        total_mib=8192,
        cuda_available=cuda,
        probe_source="nvml" if cuda else "unavailable",
        resident_models=models if models is not None else [],
    )


def _nli_verdict(verdicts: List[FitVerdict]) -> FitVerdict:
    return next(v for v in verdicts if v.consumer == "nli")


def _emb_verdict(verdicts: List[FitVerdict]) -> FitVerdict:
    return next(v for v in verdicts if v.consumer == "embedding")


def test_fit_nli_cpu_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ED4ALL_NLI_DEVICE", "cpu")
    v = _nli_verdict(fit_check(_snap()))
    assert v.outcome == "cpu_requested"


def test_fit_nli_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ED4ALL_NLI_DEVICE", "cuda")
    v = _nli_verdict(fit_check(_snap(free=4000)))
    assert v.outcome == "fits"


def test_fit_nli_need_uses_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """need_mib = max(resolve_min_free_vram_mib(), 900); default floor 1024."""
    monkeypatch.setenv("ED4ALL_NLI_DEVICE", "cuda")
    v = _nli_verdict(fit_check(_snap(free=4000)))
    assert v.need_mib == 1024
    # A configured floor below the spike floor is raised to 900.
    monkeypatch.setenv("ED4ALL_NLI_MIN_FREE_VRAM_MIB", "100")
    v2 = _nli_verdict(fit_check(_snap(free=4000)))
    assert v2.need_mib == 900


def test_fit_nli_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ED4ALL_NLI_DEVICE", "cuda")
    v = _nli_verdict(fit_check(_snap(cuda=False, free=None)))
    assert v.outcome == "would_fallback_cpu"


def test_fit_nli_unknown_on_probe_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ED4ALL_NLI_DEVICE", "cuda")
    v = _nli_verdict(fit_check(_snap(free=None)))
    assert v.outcome == "unknown"


def test_fit_nli_would_evict_then_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ED4ALL_NLI_DEVICE", "cuda")
    monkeypatch.setenv("ED4ALL_NLI_EVICT_FOR_CUDA", "true")
    snap = _snap(free=200, models=[{"name": "qwen2.5:7b", "vram_mib": 5300}])
    v = _nli_verdict(fit_check(snap))
    assert v.outcome == "would_evict_then_fit"


def test_fit_nli_evict_cant_free_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ED4ALL_NLI_DEVICE", "cuda")
    monkeypatch.setenv("ED4ALL_NLI_EVICT_FOR_CUDA", "true")
    # tiny resident model can't free enough headroom.
    snap = _snap(free=100, models=[{"name": "embed", "vram_mib": 50}])
    v = _nli_verdict(fit_check(snap))
    assert v.outcome == "would_fallback_cpu"


def test_fit_nli_would_oom_when_evict_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ED4ALL_NLI_DEVICE", "cuda")
    monkeypatch.setenv("ED4ALL_NLI_EVICT_FOR_CUDA", "off")
    snap = _snap(free=100, models=[{"name": "qwen2.5:7b", "vram_mib": 5300}])
    v = _nli_verdict(fit_check(snap))
    assert v.outcome == "would_oom"


def test_fit_embedding_cpu_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding device defaults to cpu → cpu_requested."""
    v = _emb_verdict(fit_check(_snap(free=100)))
    assert v.outcome == "cpu_requested"


def test_fit_embedding_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_EMBEDDING_DEVICE, "cuda")
    v = _emb_verdict(fit_check(_snap(free=4000)))
    assert v.outcome == "fits"


def test_fit_embedding_would_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_EMBEDDING_DEVICE, "cuda")
    v = _emb_verdict(fit_check(_snap(free=100, models=[{"name": "x", "vram_mib": 9000}])))
    assert v.outcome == "would_oom"  # no eviction path for embeddings


def test_fit_embedding_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_EMBEDDING_DEVICE, "cuda")
    v = _emb_verdict(fit_check(_snap(cuda=False, free=None)))
    assert v.outcome == "would_fallback_cpu"


def test_fit_embedding_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_EMBEDDING_DEVICE, "cuda")
    v = _emb_verdict(fit_check(_snap(free=None)))
    assert v.outcome == "unknown"


def test_fit_check_returns_both_consumers() -> None:
    verdicts = fit_check(_snap())
    assert {v.consumer for v in verdicts} == {"nli", "embedding"}


# --------------------------------------------------------------------- #
# Finding #9: _build_verdict shared helper drives both consumers
# --------------------------------------------------------------------- #


def test_build_verdict_embedding_no_evict_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """allow_evict=False → near-full card reports would_oom (no evict tail),
    matching the public embedding fit-check outcome + detail."""
    from lib.llm.vram_doctor import _build_verdict

    snap = _snap(free=100, models=[{"name": "x", "vram_mib": 9000}])
    v = _build_verdict("embedding", "cuda", 500, snap, allow_evict=False)
    assert v.outcome == "would_oom"
    assert "no eviction path" in v.detail
    # Identical to the public path (which resolves the device from env).
    monkeypatch.setenv(ENV_EMBEDDING_DEVICE, "cuda")
    public = _emb_verdict(
        fit_check(_snap(free=100, models=[{"name": "x", "vram_mib": 9000}]))
    )
    assert public.outcome == "would_oom"
    assert public.detail == v.detail


def test_build_verdict_nli_evict_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """allow_evict=True with a big resident model → would_evict_then_fit."""
    from lib.llm.vram_doctor import _build_verdict

    monkeypatch.setenv("ED4ALL_NLI_EVICT_FOR_CUDA", "true")
    snap = _snap(free=200, models=[{"name": "qwen2.5:7b", "vram_mib": 5300}])
    v = _build_verdict("nli", "cuda", 1024, snap, allow_evict=True)
    assert v.outcome == "would_evict_then_fit"
    assert v.detail.endswith("(fp16).") is False  # this is the evict detail, not fits


# --------------------------------------------------------------------- #
# write_trajectory_row
# --------------------------------------------------------------------- #


def test_write_trajectory_row_writes_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path))
    snap = _snap(free=200, models=[{"name": "qwen2.5:7b", "vram_mib": 5300}])

    write_trajectory_row(
        "WF-TEST-123",
        "post_rewrite_validation",
        "2026-06-26T00:00:00Z",
        snap,
        event="phase_boundary",
        extra={"note": "hello"},
    )

    target = tmp_path / "WF-TEST-123" / "vram_trajectory.jsonl"
    assert target.exists()
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["run_id"] == "WF-TEST-123"
    assert row["phase"] == "post_rewrite_validation"
    assert row["event"] == "phase_boundary"
    assert row["free_mib"] == 200
    assert row["total_mib"] == 8192
    assert row["probe_source"] == "nvml"
    assert row["cuda_available"] is True
    assert row["resident_models"] == [{"name": "qwen2.5:7b", "vram_mib": 5300}]
    assert row["note"] == "hello"
    # ``when`` is the boundary LABEL; ``ts`` is a real wall-clock UTC instant so
    # residency-hours can be joined across rows later.
    assert "ts" in row
    parsed = datetime.fromisoformat(row["ts"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_write_trajectory_row_appends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path))
    snap = _snap()
    write_trajectory_row("R1", "p1", "t1", snap)
    write_trajectory_row("R1", "p2", "t2", snap)
    target = tmp_path / "R1" / "vram_trajectory.jsonl"
    assert len(target.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_write_trajectory_row_bad_run_id_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A failing/unwritable path must NOT raise (best-effort)."""
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path))
    # Pre-create a FILE where the run dir should be so mkdir fails.
    clash = tmp_path / "R-CLASH"
    clash.write_text("not a dir", encoding="utf-8")
    # Must not raise.
    write_trajectory_row("R-CLASH", "p", "t", _snap())
    # The clash file is untouched (write degraded silently).
    assert clash.read_text(encoding="utf-8") == "not a dir"


# --------------------------------------------------------------------- #
# format_doctor_report
# --------------------------------------------------------------------- #


def test_format_report_contains_free_number_and_consumers() -> None:
    snap = _snap(free=200, models=[{"name": "qwen2.5:7b", "vram_mib": 5300}])
    verdicts = [
        FitVerdict("nli", "cuda", 1024, 200, "would_evict_then_fit", "evict ok"),
        FitVerdict("embedding", "cpu", 500, 200, "cpu_requested", "cpu only"),
    ]
    report = format_doctor_report(snap, verdicts)
    assert isinstance(report, str) and report
    assert "200 MiB" in report
    assert "nli" in report
    assert "embedding" in report
    assert "qwen2.5:7b" in report
    # markers rendered
    assert "✓" in report


def test_format_report_handles_unknown_free() -> None:
    snap = _snap(free=None, cuda=False)
    report = format_doctor_report(snap, fit_check(snap))
    assert "unknown" in report
    assert report  # non-empty
