"""Wave 22 F1 regression tests — executor hardening imports are wired.

Pre-Wave-22, ``MCP/core/executor.py`` tried to import the Phase 0
hardening modules from ``.error_classifier`` / ``.checkpoint`` /
``.validation_gates`` / ``.lockfile`` — relative paths that pointed at
``MCP/core/`` where those modules do not live. Every import silently
hit the ``except ImportError`` arm, the four ``HARDENING_*`` flags
flipped to ``False``, and the entire Phase 0 stack (retry
classification, poison-pill detection, phase checkpoints, executor-
layer validation gates, cross-process locking) was a no-op at runtime.

These tests assert that:

1. ``TaskExecutor()`` wires every hardening component on ``__init__``.
2. Each ``HARDENING_*`` leaf flag imports cleanly.
3. ``HARDENING_PHASE_0`` — the aggregate gate — is ``True``.
4. Importing ``MCP.core.executor`` does not emit the silent-ImportError
   debug line under normal conditions (monkeypatch the core logger so
   a future regression is noisy).
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def fresh_executor_module():
    """Return a fresh import of ``MCP.core.executor``.

    Re-imports the module so monkeypatched loggers observe any
    import-time debug lines emitted by the ``except ImportError`` arms.
    """
    for mod_name in list(sys.modules):
        if mod_name.startswith("MCP.core.executor"):
            del sys.modules[mod_name]
    return importlib.import_module("MCP.core.executor")


@pytest.mark.unit
def test_task_executor_error_classifier_wired(fresh_executor_module):
    """``TaskExecutor().error_classifier`` must be non-None."""
    executor = fresh_executor_module.TaskExecutor()
    assert executor.error_classifier is not None, (
        "ErrorClassifier import regressed — check the F1 fix on "
        "MCP/core/executor.py: imports must be from ..hardening.*"
    )


@pytest.mark.unit
def test_task_executor_checkpoint_manager_wired(fresh_executor_module):
    """``TaskExecutor().checkpoint_manager`` must be non-None."""
    executor = fresh_executor_module.TaskExecutor()
    assert executor.checkpoint_manager is not None, (
        "CheckpointManager import regressed — phase checkpointing "
        "silently no-ops when this flips to None."
    )


@pytest.mark.unit
def test_task_executor_gate_manager_wired(fresh_executor_module):
    """``TaskExecutor().gate_manager`` must be non-None."""
    executor = fresh_executor_module.TaskExecutor()
    assert executor.gate_manager is not None, (
        "ValidationGateManager import regressed — executor-layer "
        "gate enforcement silently no-ops when this flips to None."
    )


@pytest.mark.unit
def test_task_executor_lock_manager_wired(fresh_executor_module):
    """``TaskExecutor().lock_manager`` must be non-None.

    LockfileManager was imported but never instantiated pre-Wave-22.
    The Wave 22 F1 fix threads it through ``_init_hardening``.
    """
    executor = fresh_executor_module.TaskExecutor()
    assert executor.lock_manager is not None, (
        "LockfileManager was never instantiated — Wave 22 F1 fix "
        "adds ``self.lock_manager = LockfileManager(self.run_path)``."
    )


@pytest.mark.unit
def test_hardening_phase_0_aggregate_is_true(fresh_executor_module):
    """``HARDENING_PHASE_0`` (aggregate) must be True after a clean import."""
    assert fresh_executor_module.HARDENING_PHASE_0 is True, (
        "HARDENING_PHASE_0 is the single-source-of-truth gate for the "
        "Phase 0 hardening stack — regression means one of the four "
        "leaf imports is silently failing."
    )


@pytest.mark.unit
def test_executor_import_does_not_log_silent_import_error(monkeypatch, caplog):
    """A clean import must not emit any ``Hardening import failed`` debug line.

    The Wave 22 F1 fix adds a one-line debug log inside every
    ``except ImportError`` arm precisely so a future silent regression
    (e.g. a rename that puts the imports back onto non-existent core
    submodules) becomes observable in logs.
    """
    # Force-reload at DEBUG so we can see the debug lines the guard arms emit.
    for mod_name in list(sys.modules):
        if mod_name.startswith("MCP.core.executor"):
            del sys.modules[mod_name]

    with caplog.at_level(logging.DEBUG, logger="MCP.core.executor"):
        importlib.import_module("MCP.core.executor")

    for record in caplog.records:
        if "Hardening import failed" in record.getMessage():
            pytest.fail(
                f"Executor import logged a silent-ImportError debug line "
                f"(this means one of the hardening imports is failing): "
                f"{record.getMessage()}"
            )
