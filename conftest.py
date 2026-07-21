"""Ed4All root test configuration"""
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.testing.reachability import make_local_synthesis_skip_hook  # noqa: E402

# Repo-wide skip hook: a ``*_SYNTHESIS_PROVIDER=local`` env override (the
# marketable-v1 license-clean default present on some CI runners) re-routes a
# nominally ``provider="mock"`` synthesis test to a live OpenAI-compatible
# server on ``localhost:11434``. When no such server is listening the run fails
# with a connection-refused error (raised, or surfaced inside a phase
# result-dict ``error`` field the test asserts on). This hook converts EXACTLY
# that environment-shaped failure into a skip, repo-wide so every synthesis
# test family is covered from one place. Strict no-op when the backend is up or
# the failure isn't a refused connect to it — when a local server IS up, every
# test runs against the live backend exactly as before. See
# lib/testing/reachability.py for the precision contract.
pytest_runtest_call = make_local_synthesis_skip_hook()


def pytest_configure(config):
    """Register the ``real_libv2_archive`` opt-out marker.

    Tests decorated with ``@pytest.mark.real_libv2_archive`` are NOT
    redirected by the autouse ``isolate_libv2_and_captures`` fixture —
    they read/write the real in-tree ``LibV2/`` archive (e.g. RDF-export
    round-trips, full-archive emit tests). Everything else is isolated
    into ``tmp_path`` so a pytest run never litters ``LibV2/courses/``
    with empty skeleton dirs or grows ``training-captures/`` by thousands
    of files.
    """
    config.addinivalue_line(
        "markers",
        "real_libv2_archive: opt out of the autouse LibV2/training-captures "
        "tmp redirect; test reads/writes the real in-tree LibV2 archive.",
    )


@pytest.fixture(scope="session", autouse=True)
def _ed4all_default_roots_isolated(tmp_path_factory):
    """Session-scoped: repoint the lib-layer DEFAULT roots into tmp.

    Autouse so EVERY test is hermetic by default. Without this, any test
    that exercises an LLM call path constructs
    ``LibV2Storage(course_code, auto_create=True)`` →
    ``ensure_directories()`` which mkdir's ``LibV2/courses/<slug>/...`` +
    ``LibV2/catalog/<COURSE>/...`` in the REAL tree, and
    ``DecisionCapture`` mirrors decision JSONL into the real
    ``training-captures/<tool>/<COURSE>/`` tree. An audit found this
    mechanism had littered ``LibV2/courses/`` with 87 empty skeleton dirs
    and grew ``training-captures/`` by thousands of files per pytest run.

    Design notes (why setattr on the DEFAULT constants, not env vars):

    * ``ED4ALL_LIBV2_ROOT`` is a leg of the documented resolution chain
      (explicit kwarg > env > default). Setting it autouse would flip
      every test that deliberately exercises the DEFAULT leg (e.g.
      ``lib/validators/tests/test_libv2_manifest_dual_chunkset.py``
      patches ``pipeline_tools._PROJECT_ROOT`` and expects the default
      fallthrough) onto the env leg. Patching ``lib.paths.LIBV2_PATH`` /
      ``lib.paths.TRAINING_DIR`` instead redirects only what the default
      leg RESOLVES TO, leaving chain semantics untouched.
    * Session scope (not function scope) so module-/session-scoped
      fixtures — which instantiate BEFORE any function-scoped autouse
      fixture — are covered too (e.g. the module-scoped CourseProcessor
      fixture in ``Trainforge/tests/test_summary_factory.py``).
    * Both ``lib/paths.py::libv2_path`` and ``::get_training_captures_dir``
      read the module globals at call time, so the setattr is honored by
      every ``LibV2Storage`` / ``DecisionCapture`` / ``StreamingCapture``
      constructed during the session.
    * ``ED4ALL_STATE_RUNS_DIR`` IS set as an env var (no test asserts a
      default-leg destination against the real ``state/runs/``), covering
      tests that construct ``TaskExecutor()`` / ``CheckpointManager``
      without the explicit ``state_runs_isolated`` fixture (whose
      construction-time ``checkpoints/`` mkdir leaked
      ``state/runs/run_<date>_<time>/`` dirs). The explicit
      ``state_runs_isolated`` fixture still overrides per-test.

    Per-test opt-out: ``@pytest.mark.real_libv2_archive`` (see the
    function-scoped fixture below).
    """
    import lib.paths as _paths

    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("ed4all_session_isolated")
    libv2_root = base / "LibV2"
    captures_root = base / "training-captures"
    state_runs_root = base / "state" / "runs"
    libv2_root.mkdir(parents=True, exist_ok=True)
    captures_root.mkdir(parents=True, exist_ok=True)
    state_runs_root.mkdir(parents=True, exist_ok=True)

    mp.setattr(_paths, "LIBV2_PATH", libv2_root)
    mp.setattr(_paths, "TRAINING_DIR", captures_root)
    # ED4ALL_TRAINING_CAPTURES_DIR is also set as an env var (unlike
    # ED4ALL_LIBV2_ROOT) because (a) it's a new variable no
    # resolution-chain test depends on being unset, and (b) env vars —
    # unlike setattr — propagate to subprocess-spawned capture writers.
    mp.setenv("ED4ALL_TRAINING_CAPTURES_DIR", str(captures_root))
    mp.setenv("ED4ALL_STATE_RUNS_DIR", str(state_runs_root))
    yield {
        "libv2_root": libv2_root,
        "captures_root": captures_root,
        "state_runs_root": state_runs_root,
    }
    mp.undo()


@pytest.fixture(autouse=True)
def isolate_libv2_and_captures(request, monkeypatch):
    """Per-test opt-out hook for the session-level default-root isolation.

    Tests marked ``@pytest.mark.real_libv2_archive`` get the REAL in-tree
    defaults restored for their duration (most such tests resolve via
    ``PROJECT_ROOT`` literals anyway and don't need this, but the marker
    documents intent and guarantees correctness if they later switch to
    the resolvers).
    """
    if request.node.get_closest_marker("real_libv2_archive"):
        import lib.paths as _paths

        monkeypatch.setattr(_paths, "LIBV2_PATH", _paths.PROJECT_ROOT / "LibV2")
        monkeypatch.setattr(
            _paths, "TRAINING_DIR", _paths.PROJECT_ROOT / "training-captures"
        )
        monkeypatch.delenv("ED4ALL_TRAINING_CAPTURES_DIR", raising=False)
    yield


@pytest.fixture
def project_root():
    """Returns the Ed4All project root directory"""
    return PROJECT_ROOT


@pytest.fixture
def temp_project_dir(tmp_path):
    """Creates a temporary project directory structure for testing"""
    dirs = [
        "SemantiK",
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
