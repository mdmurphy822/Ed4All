"""v2 pipeline orchestrator.

Lives alongside `semantik_structure.pipeline` (v1) so v1 keeps producing
byte-identical output throughout the rollout. The mode parameter selects:

    mode="v1"  ->  delegates to pipeline.run_pipeline (byte-for-byte v1)
    mode="v2"  ->  runs the full Stage 1..13 cascade via
                   `semantik_structure.cascade.run_pipeline_v2` and returns a
                   `PipelineV2Result`.

This module imports `pipeline` and `cascade` lazily inside `run` so
importing `semantik_structure.pipeline_v2` does NOT trigger v1-side or
cascade-side imports until the first call. That keeps
`pytest --collect-only` cheap and lets the type-only Phase-0 tests run
without paying the heavy import cost.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .v2_config import DEFAULT_V2_CONFIG, V2Config


def run(pdf_path: Path,
        *,
        mode: str = "v1",
        config: V2Config = DEFAULT_V2_CONFIG,
        **v1_kwargs: Any) -> Any:
    """Run the pipeline.

    Args:
        pdf_path: PDF to process.
        mode: "v1" (default) delegates to pipeline.run_pipeline unchanged;
              "v2" runs the full Stage 1..13 cascade via
              `semantik_structure.cascade.run_pipeline_v2`.
        config: v2 component config. Ignored when mode="v1"; forwarded to
                run_pipeline_v2 when mode="v2".
        **v1_kwargs: forwarded verbatim to pipeline.run_pipeline when
                     mode="v1" (e.g. classifier=, reasoner=, glm_ocr=,
                     language=). When mode="v2", forwarded to
                     run_pipeline_v2 (valid keys: k, max_regions_per_kind,
                     runtime_mode, enable_glm_ocr_stage); an unknown key
                     raises TypeError.

    Returns:
        For mode="v1": a `pipeline.PipelineResult`.
        For mode="v2": a `cascade.PipelineV2Result` wrapping the assembled
                       HTML and the Stage 13 exit decision.
    """
    if mode == "v1":
        # Lazy import so Phase-0 tests on type-only modules don't pay the
        # v1 import cost (transformers, peft, playwright …).
        from . import pipeline as _v1_pipeline
        return _v1_pipeline.run_pipeline(pdf_path, **v1_kwargs)

    if mode == "v2":
        # Lazy import — keep cascade-side imports off the type-only path.
        from . import cascade
        return cascade.run_pipeline_v2(pdf_path, config=config, **v1_kwargs)

    raise ValueError(f"unknown pipeline mode: {mode!r}; expected 'v1' or 'v2'")
