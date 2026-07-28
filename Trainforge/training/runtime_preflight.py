"""Fail-loud dependency contract for real Trainforge GPU training.

The main Ed4All environment intentionally remains usable without the heavy
training stack.  A real adapter fit, however, must run inside the qualified
band declared by ``pyproject.toml`` and the training-window runbook.  Checking
before weight loading prevents a 59-GiB Nano load from becoming an accidental
integration test of an unsupported Transformers/TRL combination.
"""
from __future__ import annotations

from importlib import metadata
import platform
from typing import Mapping, Optional

from packaging.version import Version


_SUPPORTED = {
    "torch": ("2.9.0", "2.10.0"),
    "transformers": ("4.57.0", "5.0.0"),
    "trl": ("0.25.0", "0.27.0"),
    "peft": ("0.17.0", "1.0.0"),
    "accelerate": ("1.0.0", "2.0.0"),
    "datasets": ("2.18.0", None),
}

_GB10_CU130_EXACT = {
    "torch": "2.13.0",
    "transformers": "4.57.6",
    "trl": "0.26.2",
    "peft": "0.19.1",
    "accelerate": "1.12.0",
    "datasets": "4.4.1",
    "mamba-ssm": "2.3.2.post1",
    "causal-conv1d": "1.6.2.post1",
}


def training_runtime_versions() -> dict[str, str]:
    """Return installed versions for every required training distribution."""
    out: dict[str, str] = {}
    missing: list[str] = []
    for distribution in _SUPPORTED:
        try:
            out[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        raise RuntimeError(
            "Trainforge training runtime is incomplete; missing "
            f"{missing}. Build the repo-managed environment with "
            "scripts/bootstrap-training-env.sh."
        )
    return out


def assert_supported_training_runtime(
    versions: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Return versions or raise when any distribution is outside its band."""
    if versions is None and platform.machine() == "aarch64":
        try:
            installed_torch = metadata.version("torch")
        except metadata.PackageNotFoundError:
            installed_torch = ""
        if installed_torch == _GB10_CU130_EXACT["torch"]:
            return assert_gb10_cu130_training_runtime()
    resolved = dict(versions or training_runtime_versions())
    incompatible: list[str] = []
    for distribution, (minimum, maximum) in _SUPPORTED.items():
        raw = resolved.get(distribution)
        if raw is None:
            incompatible.append(f"{distribution}=missing")
            continue
        parsed = Version(raw)
        if parsed < Version(minimum) or (
            maximum is not None and parsed >= Version(maximum)
        ):
            ceiling = f",<{maximum}" if maximum is not None else ""
            incompatible.append(
                f"{distribution}={raw} (requires >={minimum}{ceiling})"
            )
    if incompatible:
        raise RuntimeError(
            "unsupported Trainforge training runtime: "
            + "; ".join(incompatible)
            + ". Do not modify system Python; run through "
            "scripts/ed4all-training using the repo-managed .venv-training."
        )
    return resolved


def assert_gb10_cu130_training_runtime(
    versions: Optional[Mapping[str, str]] = None,
    *,
    machine: Optional[str] = None,
    torch_module: object | None = None,
    verify_extensions: bool = True,
) -> dict[str, str]:
    """Fail before weight loading unless the exact GB10 CUDA 13 profile is live."""
    resolved: dict[str, str] = {}
    missing: list[str] = []
    supplied = dict(versions) if versions is not None else None
    for distribution in _GB10_CU130_EXACT:
        if supplied is not None:
            raw = supplied.get(distribution)
            if raw is None:
                missing.append(distribution)
            else:
                resolved[distribution] = raw
            continue
        try:
            resolved[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        raise RuntimeError(
            "GB10 CUDA 13 training runtime is incomplete; missing "
            f"{missing}. Run scripts/bootstrap-training-env.sh."
        )

    incompatible = [
        f"{name}={resolved[name]} (requires =={expected})"
        for name, expected in _GB10_CU130_EXACT.items()
        if resolved[name] != expected
    ]
    actual_machine = machine or platform.machine()
    if actual_machine != "aarch64":
        incompatible.append(f"machine={actual_machine} (requires aarch64)")
    if incompatible:
        raise RuntimeError(
            "unsupported GB10 CUDA 13 training runtime: "
            + "; ".join(incompatible)
        )

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    torch_build = str(getattr(torch_module, "__version__", "missing"))
    torch_cuda = str(getattr(getattr(torch_module, "version", None), "cuda", None))
    cuda = getattr(torch_module, "cuda", None)
    if torch_build != "2.13.0+cu130" or torch_cuda != "13.0":
        raise RuntimeError(
            "GB10 training requires torch 2.13.0+cu130 with CUDA 13.0; "
            f"found torch={torch_build}, torch.version.cuda={torch_cuda}"
        )
    if cuda is None or not cuda.is_available() or cuda.device_count() < 1:
        raise RuntimeError(
            "GB10 CUDA 13 training runtime has no visible CUDA device; "
            "CPU-only Torch is not a supported training path"
        )
    props = cuda.get_device_properties(0)
    capability = tuple(cuda.get_device_capability(0))
    if props.name != "NVIDIA GB10" or capability != (12, 1):
        raise RuntimeError(
            "GB10 CUDA 13 profile requires NVIDIA GB10 SM 12.1; "
            f"found device={props.name!r}, capability={capability}"
        )

    if verify_extensions:
        try:
            from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # noqa: F401
            from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn  # noqa: F401
            from mamba_ssm.ops.triton.selective_state_update import (  # noqa: F401
                selective_state_update,
            )
            from mamba_ssm.ops.triton.ssd_combined import (  # noqa: F401
                mamba_chunk_scan_combined,
                mamba_split_conv1d_scan_combined,
            )
        except Exception as exc:  # noqa: BLE001 - ABI errors are not uniform
            raise RuntimeError(
                "GB10 SSM extension ABI/import validation failed before weight "
                f"loading: {type(exc).__name__}: {exc}"
            ) from exc
    return resolved


__all__ = [
    "assert_gb10_cu130_training_runtime",
    "assert_supported_training_runtime",
    "training_runtime_versions",
]
