"""Tests for ``ed4all doctor`` (GPU/VRAM-contention preflight).

GPU-free + deterministic: ``snapshot_vram`` / ``fit_check`` are monkeypatched
to return constructed dataclass instances so the tests never touch a real GPU,
ollama, or torch.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from cli.commands import doctor as doctor_mod
from cli.commands.doctor import doctor_command
from lib.llm.vram_doctor import FitVerdict, VramSnapshot


# ---------------------------------------------------------------------- #
# Builders
# ---------------------------------------------------------------------- #


def _snapshot(**overrides) -> VramSnapshot:
    base = dict(
        free_mib=6000,
        total_mib=8192,
        cuda_available=True,
        probe_source="nvml",
        resident_models=[{"name": "qwen2.5:7b", "vram_mib": 5300}],
        error=None,
    )
    base.update(overrides)
    return VramSnapshot(**base)


def _verdict(consumer: str, outcome: str, device: str = "cuda", **overrides) -> FitVerdict:
    base = dict(
        consumer=consumer,
        device_requested=device,
        need_mib=900,
        free_mib=6000,
        outcome=outcome,
        detail=f"{consumer} {outcome} detail.",
    )
    base.update(overrides)
    return FitVerdict(**base)


def _patch(monkeypatch, snapshot: VramSnapshot, verdicts: list[FitVerdict]) -> None:
    monkeypatch.setattr(doctor_mod, "snapshot_vram", lambda base_url=None: snapshot)
    monkeypatch.setattr(doctor_mod, "fit_check", lambda snap: verdicts)


# ---------------------------------------------------------------------- #
# Report rendering
# ---------------------------------------------------------------------- #


def test_doctor_prints_report_with_free_line_and_both_consumers(monkeypatch):
    verdicts = [
        _verdict("nli", "fits"),
        _verdict("embedding", "cpu_requested", device="cpu"),
    ]
    _patch(monkeypatch, _snapshot(), verdicts)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exit_code == 0, result.output
    # Free-VRAM header line.
    assert "free 6000 MiB" in result.output
    # Both consumers appear in the fit-check section.
    assert "nli" in result.output
    assert "embedding" in result.output
    assert "OK" in result.output


def test_doctor_handles_unavailable_snapshot_without_raising(monkeypatch):
    """A degraded/unavailable (no-GPU) snapshot must not raise."""
    snap = _snapshot(
        free_mib=None,
        total_mib=None,
        cuda_available=False,
        probe_source="unavailable",
        resident_models=[],
        error="ollama /api/ps unreachable: connection refused",
    )
    verdicts = [
        _verdict("nli", "cpu_requested", device="cpu", free_mib=None),
        _verdict("embedding", "cpu_requested", device="cpu", free_mib=None),
    ]
    _patch(monkeypatch, snap, verdicts)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    assert result.exit_code == 0, result.output
    assert "unknown" in result.output  # free/total render as "unknown"


# ---------------------------------------------------------------------- #
# JSON output
# ---------------------------------------------------------------------- #


def test_doctor_json_emits_parseable_snapshot_and_verdicts(monkeypatch):
    verdicts = [
        _verdict("nli", "fits"),
        _verdict("embedding", "fits"),
    ]
    _patch(monkeypatch, _snapshot(), verdicts)

    result = CliRunner().invoke(doctor_command, ["--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["snapshot"]["free_mib"] == 6000
    assert payload["snapshot"]["probe_source"] == "nvml"
    assert len(payload["verdicts"]) == 2
    assert {v["consumer"] for v in payload["verdicts"]} == {"nli", "embedding"}
    assert payload["exit_code"] == 0
    assert payload["summary"] == "OK"


# ---------------------------------------------------------------------- #
# Exit-code gating
# ---------------------------------------------------------------------- #


def test_doctor_exit_2_on_would_oom(monkeypatch):
    verdicts = [
        _verdict("nli", "would_oom"),
        _verdict("embedding", "fits"),
    ]
    _patch(monkeypatch, _snapshot(free_mib=200), verdicts)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exit_code == 2, result.output
    assert "DANGER" in result.output


def test_doctor_exit_1_on_cuda_fallback_cpu(monkeypatch):
    verdicts = [
        _verdict("nli", "would_fallback_cpu", device="cuda"),
        _verdict("embedding", "fits"),
    ]
    _patch(monkeypatch, _snapshot(free_mib=300), verdicts)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exit_code == 1, result.output
    assert "DEGRADED" in result.output


def test_doctor_exit_0_on_fits_and_cpu_requested(monkeypatch):
    verdicts = [
        _verdict("nli", "fits"),
        _verdict("embedding", "cpu_requested", device="cpu"),
    ]
    _patch(monkeypatch, _snapshot(), verdicts)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exit_code == 0, result.output


def test_doctor_cpu_fallback_on_cpu_device_is_not_degraded(monkeypatch):
    """A would_fallback_cpu whose device was already CPU is not exit-1.

    (Defensive: the foundation only emits would_fallback_cpu on a cuda
    request, but the gating must key on device_requested != cpu.)
    """
    verdicts = [
        _verdict("nli", "would_fallback_cpu", device="cpu"),
        _verdict("embedding", "fits"),
    ]
    _patch(monkeypatch, _snapshot(), verdicts)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exit_code == 0, result.output


def test_doctor_json_reflects_danger_exit_code(monkeypatch):
    verdicts = [_verdict("embedding", "would_oom")]
    _patch(monkeypatch, _snapshot(free_mib=100), verdicts)

    result = CliRunner().invoke(doctor_command, ["--json"])

    assert result.exit_code == 2, result.output
    payload = json.loads(result.output)
    assert payload["exit_code"] == 2
    assert "DANGER" in payload["summary"]
