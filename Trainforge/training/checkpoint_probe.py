"""Compatibility entry point for checkpoint probe imports and CLI usage."""

from __future__ import annotations

from Trainforge.training.probes.checkpoint import (
    CourseCheckpointProbe,
    main,
    preflight_checkpoint_probe,
)

__all__ = ["CourseCheckpointProbe", "preflight_checkpoint_probe", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
