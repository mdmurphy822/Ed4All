"""Trainforge training submodule (Wave 90 — slm-training-2026-04-26).

Public API:

    from Trainforge.training import (
        TrainingRunner,
        TrainingRunResult,
        BaseModelRegistry,
        BaseModelSpec,
        TrainingConfig,
        load_config,
        ComputeBackend,
        LocalBackend,
        RunPodBackend,
    )

The runner consumes an already-imported LibV2 course (its
``training_specs/`` + ``corpus/`` + ``graph/`` /  ``pedagogy/``
artifacts) and writes ``models/<model_id>/`` back into the same slug.
``ComputeBackend`` is the swap point — Wave 90 ships ``LocalBackend``
fully and ``RunPodBackend`` as a stub for the follow-up wave.
"""
from Trainforge.training.base_models import (  # noqa: F401
    BaseModelRegistry,
    BaseModelSpec,
    format_instruction,
)
from Trainforge.training.compute_backend import (  # noqa: F401
    ComputeBackend,
    LocalBackend,
    RunPodBackend,
    TrainingJobResult,
    TrainingJobSpec,
)
from Trainforge.training.configs import TrainingConfig, load_config  # noqa: F401
from Trainforge.training.runner import (  # noqa: F401
    TrainingRunner,
    TrainingRunResult,
)


__all__ = [
    "BaseModelRegistry",
    "BaseModelSpec",
    "ComputeBackend",
    "LocalBackend",
    "RunPodBackend",
    "TrainingConfig",
    "TrainingJobResult",
    "TrainingJobSpec",
    "TrainingRunResult",
    "TrainingRunner",
    "format_instruction",
    "load_config",
]
