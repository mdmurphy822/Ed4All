"""R7 VRAM-OOM mitigation — run_cascade_json.py CUDA-allocator guard tests.

The bridge runner ``SemantiK/scripts/run_cascade_json.py`` may set
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` at module top, BEFORE
any torch/transformers/llama-cpp import (those are pulled lazily by the
``dart_semantic.cascade`` import inside ``main()``) — but ONLY when the
opt-in ``SEMANTIK_EXPANDABLE_SEGMENTS`` is truthy (the conf backfired with a
CUDACachingAllocator INTERNAL ASSERT, so it is OFF by default; theta->CPU is
the primary OOM fix). These tests prove:

  1. The gated setdefault is sourced ABOVE the lazy cascade import (the only
     torch entry point) — a source-order assertion (no torch needed).
  2. Importing the runner module in a clean subprocess does NOT set the env by
     default; sets it only under SEMANTIK_EXPANDABLE_SEGMENTS=1; and a pre-set
     operator value is NOT clobbered (setdefault semantics).

No GPU / torch / real cascade. CPU-pinned by construction.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER = _REPO_ROOT / "SemantiK" / "scripts" / "run_cascade_json.py"


def test_runner_sets_alloc_conf_before_torch_in_source():
    """The gated ``_maybe_set_alloc_conf()`` call precedes the only line that
    can trigger a torch import (the lazy ``from dart_semantic.cascade import ...``
    inside main())."""
    src = _RUNNER.read_text(encoding="utf-8")
    alloc_idx = src.index("_maybe_set_alloc_conf()")
    cascade_idx = src.index("from dart_semantic.cascade import run_pipeline_v2")
    assert alloc_idx < cascade_idx, (
        "_maybe_set_alloc_conf() must be called before the cascade import (the "
        "torch entry point)."
    )
    # And it sits at module top — above main()'s def.
    main_idx = src.index("def main(")
    assert alloc_idx < main_idx
    # The setdefault itself is gated by the opt-in flag.
    gate_idx = src.index('_truthy("SEMANTIK_EXPANDABLE_SEGMENTS")')
    setdefault_idx = src.index('os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF"')
    assert gate_idx < setdefault_idx, (
        "The PYTORCH_CUDA_ALLOC_CONF setdefault must be guarded by the "
        "SEMANTIK_EXPANDABLE_SEGMENTS opt-in."
    )


def _import_runner_and_print_env(
    preset: str | None, *, opt_in: bool = False
) -> str:
    """Import the runner module in a clean subprocess; print the resolved
    PYTORCH_CUDA_ALLOC_CONF. ``preset`` (if not None) is exported first to
    prove setdefault does not clobber an operator value. ``opt_in`` exports
    SEMANTIK_EXPANDABLE_SEGMENTS=1 to enable the gated setdefault."""
    code = (
        "import os, importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('rcj', r'{_RUNNER}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "print(os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>'))\n"
    )
    env = {"PATH": "/usr/bin:/bin"}
    if preset is not None:
        env["PYTORCH_CUDA_ALLOC_CONF"] = preset
    if opt_in:
        env["SEMANTIK_EXPANDABLE_SEGMENTS"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_runner_import_default_does_not_set_alloc_conf():
    """A clean import (no opt-in) leaves PYTORCH_CUDA_ALLOC_CONF unset —
    expandable_segments is opt-in (allocator INTERNAL-ASSERT risk)."""
    assert _import_runner_and_print_env(None) == "<unset>"


def test_runner_import_opt_in_sets_alloc_conf():
    """SEMANTIK_EXPANDABLE_SEGMENTS=1 → the expandable_segments conf is set."""
    assert (
        _import_runner_and_print_env(None, opt_in=True)
        == "expandable_segments:True"
    )


def test_runner_import_does_not_clobber_operator_value():
    """A pre-set PYTORCH_CUDA_ALLOC_CONF survives even when opted in (setdefault)."""
    assert (
        _import_runner_and_print_env("max_split_size_mb:128", opt_in=True)
        == "max_split_size_mb:128"
    )
