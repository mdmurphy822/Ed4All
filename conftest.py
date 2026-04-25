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

    Also wraps ``ED4ALL_RUN_ID`` in monkeypatch so tests that drive
    ``PipelineOrchestrator._get_executor`` (which publishes
    ``params.run_id`` into ``ED4ALL_RUN_ID`` so downstream tools can
    auto-resolve a MailboxBrokeredBackend) don't leak the run_id past
    teardown — otherwise an unrelated later test that calls
    ``build_backend()`` reads the stale run_id and recreates the
    matching ``state/runs/<old-run-id>/mailbox/`` tree.

    Returns the ``state/runs`` dir under ``tmp_path``. Not auto-applied;
    each test must request the fixture explicitly.
    """
    state_dir = tmp_path / "state" / "runs"
    state_dir.mkdir(parents=True)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(state_dir))
    return state_dir
