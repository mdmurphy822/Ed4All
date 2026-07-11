"""Bug 2 — run_cascade_json expandable-segments allocator gate is OPT-IN.

``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` backfired (it can trigger
a CUDACachingAllocator INTERNAL ASSERT in a council BERT), so it is now gated
behind ``SEMANTIK_EXPANDABLE_SEGMENTS`` and OFF by default. ``run_cascade_json``
factors the gate into ``_maybe_set_alloc_conf()`` so it can be unit-tested
directly after monkeypatching env — without importing the heavy cascade (the
module-top imports are all stdlib).

Run CPU-pinned:
  ED4ALL_NLI_DEVICE=cpu python -m pytest \
    lib/semantik/tests/test_run_cascade_json_alloc_conf.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER_PATH = _REPO_ROOT / "SemantiK" / "scripts" / "run_cascade_json.py"


def _load_runner():
    """Load run_cascade_json.py in isolation (stdlib-only module top).

    The module-top ``_maybe_set_alloc_conf()`` call runs at import time; the
    heavy ``semantik_structure`` cascade import is lazy inside ``main()`` and is NOT
    triggered here. We re-exec fresh so the loaded module object exposes the
    helper for direct calls.
    """
    spec = importlib.util.spec_from_file_location(
        "run_cascade_json_under_test", str(_RUNNER_PATH)
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_unset_does_not_set_alloc_conf(monkeypatch):
    """SEMANTIK_EXPANDABLE_SEGMENTS unset → PYTORCH_CUDA_ALLOC_CONF untouched."""
    monkeypatch.delenv("SEMANTIK_EXPANDABLE_SEGMENTS", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    runner = _load_runner()
    runner._maybe_set_alloc_conf()
    assert "PYTORCH_CUDA_ALLOC_CONF" not in __import__("os").environ


def test_opt_in_sets_expandable_segments(monkeypatch):
    """SEMANTIK_EXPANDABLE_SEGMENTS=1 → allocator conf is set."""
    monkeypatch.setenv("SEMANTIK_EXPANDABLE_SEGMENTS", "1")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    runner = _load_runner()
    runner._maybe_set_alloc_conf()
    assert (
        __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:True"
    )


def test_operator_value_never_clobbered_even_when_opted_in(monkeypatch):
    """Operator-set PYTORCH_CUDA_ALLOC_CONF survives even with opt-in (setdefault)."""
    monkeypatch.setenv("SEMANTIK_EXPANDABLE_SEGMENTS", "1")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    runner = _load_runner()
    runner._maybe_set_alloc_conf()
    assert (
        __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"]
        == "max_split_size_mb:64"
    )


def test_truthy_helper_semantics(monkeypatch):
    """_truthy parses 1/true/yes/on (case-insensitive); else falsey."""
    runner = _load_runner()
    for val in ("1", "true", "YES", "On", "  true  "):
        monkeypatch.setenv("SEMANTIK_EXPANDABLE_SEGMENTS", val)
        assert runner._truthy("SEMANTIK_EXPANDABLE_SEGMENTS") is True
    for val in ("0", "false", "no", "off", "", "garbage"):
        monkeypatch.setenv("SEMANTIK_EXPANDABLE_SEGMENTS", val)
        assert runner._truthy("SEMANTIK_EXPANDABLE_SEGMENTS") is False
    monkeypatch.delenv("SEMANTIK_EXPANDABLE_SEGMENTS", raising=False)
    assert runner._truthy("SEMANTIK_EXPANDABLE_SEGMENTS") is False
