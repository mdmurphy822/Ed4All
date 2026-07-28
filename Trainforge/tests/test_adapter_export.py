from __future__ import annotations

import pytest

from Trainforge.training.adapter_export import serving_artifact_mode


def test_trtllm_falls_back_to_merged_checkpoint_not_feature_disable() -> None:
    assert serving_artifact_mode(
        "trtllm", dynamic_lora_supported=False,
    ) == "merged_checkpoint"


def test_exact_supported_backend_can_use_dynamic_adapter() -> None:
    assert serving_artifact_mode(
        "vllm", dynamic_lora_supported=True,
    ) == "dynamic_adapter"


def test_unknown_backend_fails_loud() -> None:
    with pytest.raises(ValueError, match="unsupported serving backend"):
        serving_artifact_mode("unknown", dynamic_lora_supported=False)  # type: ignore[arg-type]
