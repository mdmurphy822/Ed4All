"""Unit tests for :mod:`lib.diagnostics.gpu_profile` — big-memory profile.

Hermetic + GPU-free: the total-VRAM probe (:func:`snapshot_vram`) is
monkeypatched to return a constructed :class:`VramSnapshot`, so the suite
never touches a GPU, NVML, or an ollama server. The per-flag states are
driven through their real resolvers via ``monkeypatch.setenv`` so the tests
exercise the same resolution the runtime uses. Asserts: big-memory + all
flags default fires the lifecycle + NLI warnings (and the rate-limit /
gpu_guard warnings when those envs are set); big-memory + flags correctly
off is clean; a small GPU is not-applicable (silent); an unprobeable GPU
(no torch/CUDA) skips gracefully; and the threshold env override is honored.
"""

from __future__ import annotations

import pytest

from lib.diagnostics import gpu_profile
from lib.diagnostics.core import CheckContext, Severity, resolve_exit_code
from lib.llm.vram_doctor import VramSnapshot


@pytest.fixture(autouse=True)
def _clean_gpu_env(monkeypatch) -> None:
    """Start every test from a known env: all four knobs UNSET.

    With everything unset, ``ED4ALL_GPU_LIFECYCLE`` + ``ED4ALL_NLI_EVICT_FOR_CUDA``
    resolve to their default-ON state, and the rate-limit / gpu_guard knobs
    are off — the canonical "dev-box defaults" a big-memory box inherits.
    """
    for env in (
        "ED4ALL_GPU_LIFECYCLE",
        "ED4ALL_NLI_EVICT_FOR_CUDA",
        "ED4ALL_CLOUD_RATE_LIMIT",
        "ED4ALL_GPU_MAX_USED_MB",
        "ED4ALL_BIG_MEMORY_MIN_MIB",
    ):
        monkeypatch.delenv(env, raising=False)


def _patch_total(monkeypatch, total_mib) -> None:
    """Monkeypatch snapshot_vram to report ``total_mib`` (None → unprobeable)."""

    def fake_snapshot(base_url=None):
        return VramSnapshot(
            free_mib=None if total_mib is None else total_mib,
            total_mib=total_mib,
            cuda_available=total_mib is not None,
            probe_source="nvml" if total_mib is not None else "unavailable",
            resident_models=[],
            error=None,
        )

    monkeypatch.setattr("lib.llm.vram_doctor.snapshot_vram", fake_snapshot)


def _names(results):
    return [r.name for r in results]


def _warns(results):
    return [r for r in results if r.severity is Severity.WARN]


# --------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------- #


def test_register_no_import_side_effect() -> None:
    from lib.diagnostics.core import clear_registry, registered_checks

    clear_registry()
    assert registered_checks() == []
    gpu_profile.register_gpu_profile_checks()
    pairs = registered_checks()
    assert [g for g, _ in pairs] == ["gpu_profile"]
    assert pairs[0][1] is gpu_profile.gpu_profile_checks
    clear_registry()


# --------------------------------------------------------------------- #
# Big-memory detection + the default-flags warning path
# --------------------------------------------------------------------- #


def test_big_memory_default_flags_warns_lifecycle_and_nli(monkeypatch) -> None:
    # 128 GiB unified-memory box, all knobs at their dev-box defaults.
    _patch_total(monkeypatch, 131072)
    results = gpu_profile.gpu_profile_checks(CheckContext())

    # OK summary present, exit code stays 1 (DEGRADED) not 2 (never FAIL).
    assert results[0].name == "gpu_profile_detected"
    assert results[0].severity is Severity.OK
    assert resolve_exit_code(results) == 1
    assert not any(r.severity is Severity.FAIL for r in results)

    warn_names = _names(_warns(results))
    assert "gpu_profile_lifecycle_sweep" in warn_names
    assert "gpu_profile_nli_evict" in warn_names
    # Rate-limit + gpu_guard NOT set → not warned.
    assert "gpu_profile_cloud_rate_limit" not in warn_names
    assert "gpu_profile_gpu_guard" not in warn_names

    # Each warning names its flag + carries a remediation.
    for w in _warns(results):
        assert w.remediation
        assert w.data.get("flag")


def test_big_memory_all_four_warnings(monkeypatch) -> None:
    _patch_total(monkeypatch, 131072)
    monkeypatch.setenv("ED4ALL_CLOUD_RATE_LIMIT", "true")
    monkeypatch.setenv("ED4ALL_GPU_MAX_USED_MB", "1500")  # << half of 131072
    results = gpu_profile.gpu_profile_checks(CheckContext())
    warn_names = _names(_warns(results))
    assert warn_names == [
        "gpu_profile_lifecycle_sweep",
        "gpu_profile_nli_evict",
        "gpu_profile_cloud_rate_limit",
        "gpu_profile_gpu_guard",
    ]


def test_big_memory_flags_correctly_off_is_clean(monkeypatch) -> None:
    _patch_total(monkeypatch, 131072)
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "0")
    monkeypatch.setenv("ED4ALL_NLI_EVICT_FOR_CUDA", "0")
    # rate-limit + gpu_guard already unset by the autouse fixture.
    results = gpu_profile.gpu_profile_checks(CheckContext())
    assert _warns(results) == []
    # Only the OK detection summary; exit code clean.
    assert _names(results) == ["gpu_profile_detected"]
    assert resolve_exit_code(results) == 0


# --------------------------------------------------------------------- #
# gpu_guard ceiling: only warn when set BELOW half the total VRAM
# --------------------------------------------------------------------- #


def test_gpu_guard_not_warned_when_ceiling_above_half(monkeypatch) -> None:
    _patch_total(monkeypatch, 131072)
    # A ceiling above total VRAM is exactly the recommended remediation.
    monkeypatch.setenv("ED4ALL_GPU_MAX_USED_MB", "200000")
    results = gpu_profile.gpu_profile_checks(CheckContext())
    assert "gpu_profile_gpu_guard" not in _names(_warns(results))


def test_gpu_guard_garbage_env_not_warned(monkeypatch) -> None:
    _patch_total(monkeypatch, 131072)
    monkeypatch.setenv("ED4ALL_GPU_MAX_USED_MB", "not-a-number")
    results = gpu_profile.gpu_profile_checks(CheckContext())
    assert "gpu_profile_gpu_guard" not in _names(_warns(results))


# --------------------------------------------------------------------- #
# Not-applicable: small GPU / no GPU / graceful skip
# --------------------------------------------------------------------- #


def test_small_gpu_is_silent_no_op(monkeypatch) -> None:
    # 8GB dev box — below the 48 GiB threshold → zero results, exit 0.
    _patch_total(monkeypatch, 8192)
    results = gpu_profile.gpu_profile_checks(CheckContext())
    assert results == []
    assert resolve_exit_code(results) == 0


def test_unprobeable_gpu_skips_gracefully(monkeypatch) -> None:
    # No torch / no CUDA / nvidia-smi absent → total_mib None → not-applicable.
    _patch_total(monkeypatch, None)
    results = gpu_profile.gpu_profile_checks(CheckContext())
    assert results == []


def test_snapshot_raising_degrades_to_not_applicable(monkeypatch) -> None:
    def boom(base_url=None):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr("lib.llm.vram_doctor.snapshot_vram", boom)
    # Must NEVER raise and must NOT surface a WARN — a probe failure is
    # "not applicable", not a problem report.
    results = gpu_profile.gpu_profile_checks(CheckContext())
    assert results == []


# --------------------------------------------------------------------- #
# Threshold override
# --------------------------------------------------------------------- #


def test_threshold_env_override_respected(monkeypatch) -> None:
    # Lower the threshold so an otherwise-small 16 GiB card counts as big.
    _patch_total(monkeypatch, 16384)
    monkeypatch.setenv("ED4ALL_BIG_MEMORY_MIN_MIB", "12000")
    results = gpu_profile.gpu_profile_checks(CheckContext())
    assert results and results[0].name == "gpu_profile_detected"
    assert "gpu_profile_lifecycle_sweep" in _names(_warns(results))


def test_threshold_override_garbage_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_BIG_MEMORY_MIN_MIB", "garbage")
    assert gpu_profile.resolve_big_memory_min_mib() == 49152
    # 16 GiB with the default 48 GiB threshold → not big-memory.
    _patch_total(monkeypatch, 16384)
    assert gpu_profile.gpu_profile_checks(CheckContext()) == []


def test_threshold_override_non_positive_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_BIG_MEMORY_MIN_MIB", "0")
    assert gpu_profile.resolve_big_memory_min_mib() == 49152
    monkeypatch.setenv("ED4ALL_BIG_MEMORY_MIN_MIB", "-5")
    assert gpu_profile.resolve_big_memory_min_mib() == 49152
