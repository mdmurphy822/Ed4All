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


def _save_executor_module_state():
    """Snapshot ``sys.modules`` entries + the ``MCP.core`` package's
    ``executor`` attribute that point at the live ``MCP.core.executor``
    module, so ``_restore_executor_module_state`` can put them back.

    Why both? ``sys.modules['MCP.core.executor']`` and the
    ``MCP.core.executor`` attribute on the parent ``MCP.core`` package
    are independently bound to the module object. When CPython
    resolves ``import MCP.core.executor as e`` it returns the
    **parent package's attribute**, NOT ``sys.modules``. A fresh
    ``importlib.import_module("MCP.core.executor")`` updates BOTH —
    so a polluter that re-imports without restoration leaves the
    parent attribute pointing at a stale fresh module that subsequent
    ``monkeypatch.setattr("MCP.core.executor.STATE_PATH", ...)``
    calls cannot reach (the monkeypatch lands on the original
    sys.modules entry, while the W9 test's ``import MCP.core.executor``
    silently picks up the orphaned fresh module via the parent
    package, sees the un-patched module-level ``STATE_PATH``, and
    looks up tasks under the stale repo-root state dir — explaining
    the silent "0 tasks executed" failure mode against
    ``test_workflow_runner_courseforge_two_pass_gates_e2e.py``.
    """
    saved_modules = {
        mod_name: sys.modules[mod_name]
        for mod_name in list(sys.modules)
        if mod_name.startswith("MCP.core.executor")
    }
    parent = sys.modules.get("MCP.core")
    saved_parent_attr = (
        getattr(parent, "executor", None) if parent is not None else None
    )
    return saved_modules, saved_parent_attr


def _restore_executor_module_state(saved_modules, saved_parent_attr):
    for mod_name in list(sys.modules):
        if mod_name.startswith("MCP.core.executor"):
            del sys.modules[mod_name]
    for mod_name, mod_obj in saved_modules.items():
        sys.modules[mod_name] = mod_obj
    parent = sys.modules.get("MCP.core")
    if parent is not None and saved_parent_attr is not None:
        parent.executor = saved_parent_attr


@pytest.fixture
def fresh_executor_module():
    """Return a fresh import of ``MCP.core.executor``.

    Re-imports the module so monkeypatched loggers observe any
    import-time debug lines emitted by the ``except ImportError`` arms.

    Saves + restores both the ``sys.modules`` entries AND the parent
    package's attribute binding so subsequent tests in the same session
    see the original module object via every retrieval path
    (``import MCP.core.executor``, ``from MCP.core import executor``,
    ``sys.modules['MCP.core.executor']``). See
    ``_save_executor_module_state`` docstring for the failure mode
    that motivated the parent-attribute restoration.
    """
    saved_modules, saved_parent_attr = _save_executor_module_state()
    for mod_name in list(saved_modules):
        del sys.modules[mod_name]
    try:
        yield importlib.import_module("MCP.core.executor")
    finally:
        _restore_executor_module_state(saved_modules, saved_parent_attr)


@pytest.mark.unit
def test_task_executor_error_classifier_wired(fresh_executor_module, state_runs_isolated):
    """``TaskExecutor().error_classifier`` must be non-None.

    Wave 74: opted into ``state_runs_isolated`` so the timestamp-fallback
    ``run_path`` lands in tmp_path.
    """
    executor = fresh_executor_module.TaskExecutor()
    assert executor.error_classifier is not None, (
        "ErrorClassifier import regressed — check the F1 fix on "
        "MCP/core/executor.py: imports must be from ..hardening.*"
    )


@pytest.mark.unit
def test_task_executor_checkpoint_manager_wired(fresh_executor_module, state_runs_isolated):
    """``TaskExecutor().checkpoint_manager`` must be non-None."""
    executor = fresh_executor_module.TaskExecutor()
    assert executor.checkpoint_manager is not None, (
        "CheckpointManager import regressed — phase checkpointing "
        "silently no-ops when this flips to None."
    )


@pytest.mark.unit
def test_task_executor_gate_manager_wired(fresh_executor_module, state_runs_isolated):
    """``TaskExecutor().gate_manager`` must be non-None."""
    executor = fresh_executor_module.TaskExecutor()
    assert executor.gate_manager is not None, (
        "ValidationGateManager import regressed — executor-layer "
        "gate enforcement silently no-ops when this flips to None."
    )


@pytest.mark.unit
def test_task_executor_lock_manager_wired(fresh_executor_module, state_runs_isolated):
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
    # Save + restore both the ``sys.modules`` entries AND the parent
    # package's attribute binding (see ``_save_executor_module_state``
    # docstring for why both are needed).
    saved_modules, saved_parent_attr = _save_executor_module_state()
    for mod_name in list(saved_modules):
        del sys.modules[mod_name]

    try:
        with caplog.at_level(logging.DEBUG, logger="MCP.core.executor"):
            importlib.import_module("MCP.core.executor")

        for record in caplog.records:
            if "Hardening import failed" in record.getMessage():
                pytest.fail(
                    f"Executor import logged a silent-ImportError debug line "
                    f"(this means one of the hardening imports is failing): "
                    f"{record.getMessage()}"
                )
    finally:
        _restore_executor_module_state(saved_modules, saved_parent_attr)
