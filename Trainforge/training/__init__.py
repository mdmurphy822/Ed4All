"""Trainforge adapter-training API.

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
    )

The runner consumes an already-imported LibV2 course (its
``training_specs/`` + ``corpus/`` + ``graph/`` /  ``pedagogy/``
artifacts) and writes ``models/<model_id>/`` back into the same slug.
``ComputeBackend`` is the injection point for training execution;
``LocalBackend`` is the supported implementation.
"""
from Trainforge.training.base_models import (  # noqa: F401
    BaseModelRegistry,
    BaseModelSpec,
    format_instruction,
)
from Trainforge.training.compute_backend import (  # noqa: F401
    ComputeBackend,
    LocalBackend,
    TrainingJobResult,
    TrainingJobSpec,
)
from Trainforge.training.configs import (  # noqa: F401
    ConfigOverrideError,
    TrainingConfig,
    coerce_config_overrides,
    load_config,
    parse_config_overrides,
)
from Trainforge.training.runner import (  # noqa: F401
    TrainingRunner,
    TrainingRunResult,
)

__all__ = [
    "BaseModelRegistry",
    "BaseModelSpec",
    "ComputeBackend",
    "ConfigOverrideError",
    "LocalBackend",
    "TrainingConfig",
    "TrainingJobResult",
    "TrainingJobSpec",
    "TrainingRunResult",
    "TrainingRunner",
    "coerce_config_overrides",
    "format_instruction",
    "load_config",
    "parse_config_overrides",
]
