"""Ed4All root test configuration"""
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    """Returns the Ed4All project root directory"""
    return PROJECT_ROOT


@pytest.fixture
def temp_project_dir(tmp_path):
    """Creates a temporary project directory structure for testing"""
    dirs = [
        "DART",
        "Courseforge",
        "Trainforge",
        "LibV2/courses",
        "LibV2/catalog",
        "MCP/tools",
        "orchestrator",
        "lib",
        "state",
    ]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def state_runs_isolated(tmp_path, monkeypatch):
    """Redirect any test that writes to ``state/runs/`` into ``tmp_path``.

    Tests opt in by requesting this fixture. Setting
    ``ED4ALL_STATE_RUNS_DIR`` is honored by:

    - ``MCP/core/executor.py`` (TaskExecutor.run_path fallback)
    - ``MCP/orchestrator/task_mailbox.py`` (TaskMailbox.base_dir fallback)
    - ``MCP/orchestrator/local_dispatcher.py``
      (LocalDispatcher.mailbox_base_dir fallback)
    - ``lib/paths.py::get_state_runs_dir``

    Also snapshots + restores ``ED4ALL_RUN_ID`` so tests that drive
    ``PipelineOrchestrator._get_executor`` (which directly mutates
    ``os.environ["ED4ALL_RUN_ID"] = run_id`` at
    ``MCP/orchestrator/pipeline_orchestrator.py:250`` so downstream
    tools can auto-resolve a MailboxBrokeredBackend) don't leak the
    run_id past teardown. Without this snapshot, an unrelated later
    test that calls ``build_backend()`` (e.g.
    ``lib/tests/test_llm_backend.py::test_local_mode_default``) reads
    the stale run_id, picks the ``MailboxBrokeredBackend`` branch,
    and trips its ``isinstance(..., LocalBackend)`` assertion. We
    can't use ``monkeypatch.delenv`` for this — monkeypatch only
    restores values that were SET via monkeypatch, but the value is
    set by production code mid-test, so we manually capture +
    restore in a finalizer.

    Returns the ``state/runs`` dir under ``tmp_path``. Not auto-applied;
    each test must request the fixture explicitly.
    """
    import os

    state_dir = tmp_path / "state" / "runs"
    state_dir.mkdir(parents=True)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(state_dir))

    # Snapshot ED4ALL_RUN_ID + restore in teardown so production-code
    # mutations during the test don't leak to subsequent tests.
    _original_run_id = os.environ.get("ED4ALL_RUN_ID")

    def _restore_run_id():
        if _original_run_id is None:
            os.environ.pop("ED4ALL_RUN_ID", None)
        else:
            os.environ["ED4ALL_RUN_ID"] = _original_run_id

    # Register the restore to run after the test (and after monkeypatch
    # teardown — request_finalizer order is LIFO, so this fires first
    # but on env vars not touched by monkeypatch).
    import atexit  # noqa: F401  # documents intent; we use addfinalizer below

    yield state_dir
    _restore_run_id()
