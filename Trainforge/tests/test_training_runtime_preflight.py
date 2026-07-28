from __future__ import annotations

import pytest

from Trainforge.training.runtime_preflight import (
    assert_gb10_cu130_training_runtime,
    assert_supported_training_runtime,
)


QUALIFIED = {
    "torch": "2.9.0",
    "transformers": "4.57.6",
    "trl": "0.26.2",
    "peft": "0.18.0",
    "accelerate": "1.12.0",
    "datasets": "4.4.1",
}


def test_qualified_training_band_passes() -> None:
    assert assert_supported_training_runtime(QUALIFIED) == QUALIFIED


@pytest.mark.parametrize(
    ("name", "version"),
    [("transformers", "5.0.0"), ("trl", "1.0.0"), ("peft", "1.0.0")],
)
def test_unsupported_major_fails_before_weight_load(
    name: str, version: str,
) -> None:
    versions = dict(QUALIFIED)
    versions[name] = version
    with pytest.raises(RuntimeError, match="repo-managed .venv-training"):
        assert_supported_training_runtime(versions)


def test_missing_distribution_fails_loud() -> None:
    versions = dict(QUALIFIED)
    versions.pop("trl")
    with pytest.raises(RuntimeError, match="trl=missing"):
        assert_supported_training_runtime(versions)


GB10_QUALIFIED = {
    **QUALIFIED,
    "torch": "2.13.0",
    "peft": "0.19.1",
    "mamba-ssm": "2.3.2.post1",
    "causal-conv1d": "1.6.2.post1",
}


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_properties(_: int):
        return type("Props", (), {"name": "NVIDIA GB10"})()

    @staticmethod
    def get_device_capability(_: int) -> tuple[int, int]:
        return (12, 1)


class _QualifiedTorch:
    __version__ = "2.13.0+cu130"
    version = type("Version", (), {"cuda": "13.0"})()
    cuda = _Cuda()


def test_exact_gb10_profile_accepts_without_loading_weights() -> None:
    assert (
        assert_gb10_cu130_training_runtime(
            GB10_QUALIFIED,
            machine="aarch64",
            torch_module=_QualifiedTorch(),
            verify_extensions=False,
        )
        == GB10_QUALIFIED
    )


def test_gb10_profile_rejects_cpu_only_pypi_torch() -> None:
    cpu_torch = type(
        "CpuTorch",
        (),
        {
            "__version__": "2.9.0",
            "version": type("Version", (), {"cuda": None})(),
            "cuda": type(
                "Cuda",
                (),
                {
                    "is_available": staticmethod(lambda: False),
                    "device_count": staticmethod(lambda: 0),
                },
            )(),
        },
    )()
    versions = dict(GB10_QUALIFIED)
    versions["torch"] = "2.9.0"
    with pytest.raises(RuntimeError, match="requires ==2.13.0"):
        assert_gb10_cu130_training_runtime(
            versions,
            machine="aarch64",
            torch_module=cpu_torch,
            verify_extensions=False,
        )
