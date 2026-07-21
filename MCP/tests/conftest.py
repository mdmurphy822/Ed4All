"""
Ed4All orchestrator/ test configuration and shared fixtures.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# ENVIRONMENT HERMETICITY (cross-test pollution guard)
# =============================================================================
#
# Several production code paths exercised by these tests write routing /
# provider env vars DIRECTLY into ``os.environ`` rather than through
# ``monkeypatch`` — most notably ``WorkflowRunner.run_workflow`` →
# ``_apply_corpus_generalization_defaults`` / ``_apply_authoring_route_env``,
# which fill ``COURSEFORGE_PROVIDER``, ``COURSEPLANNER_PROVIDER``, the
# ``TRAINFORGE_*`` corpus-generalization quartet, and the synthesis-provider
# seats. Because ``monkeypatch`` only tears down the changes IT made, those
# direct writes survive teardown and leak into every downstream test that keys
# behavior off the same envs (dispatch-routing short-circuits, textbook
# synthesis/structure, plan_course_structure, …). Individually the polluting
# files pass; the FULL suite in one process fails.
#
# This autouse fixture snapshots ``os.environ`` before each test and restores
# it verbatim afterward, so a test can never observe env state leaked by an
# earlier test. It is additive to ``monkeypatch`` (whose own teardown still
# runs first) and makes the MCP suite hermetic regardless of test order.
# Function-scoped and cheap (a dict copy); higher-scope fixtures instantiate
# before it, so their env is captured in the snapshot and preserved.


@pytest.fixture(autouse=True)
def _restore_os_environ():
    _snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(_snapshot)


# =============================================================================
# EMBEDDING-CLIENT IMPORT-FREEZE GUARD (cross-test pollution guard)
# =============================================================================
#
# ``lib.objectives.objective_dedup`` binds ``build_embedding_client`` at module
# import via ``from lib.embedding.providers import build_embedding_client``.
# Several tests (test_textbook_synthesis_grounded, test_stage2_chapter_anchor,
# test_textbook_synthesis_stage2, …) monkeypatch
# ``lib.embedding.providers.build_embedding_client`` to inject a fake / None
# embedder. If objective_dedup's FIRST import happens while such a patch is
# active (it is a lazy import inside the Stage-2 synthesis path), the patched
# callable freezes into objective_dedup's own module namespace. ``monkeypatch``
# then restores the providers module attribute — but NOT objective_dedup's
# independent binding — so the orphaned fake leaks into every later test whose
# dedup pass builds its own client, cosine-collapsing distinct objectives
# (e.g. 4 → 2) and failing assertions that pass in isolation.
#
# Fix the anti-pattern once, at collection time: import objective_dedup now
# (before any test patch is active) and replace its module-level
# ``build_embedding_client`` with a thin forwarder that resolves the LIVE
# ``lib.embedding.providers`` attribute on every call. The name can no longer
# freeze — a per-test monkeypatch of the providers module is honored while
# active and cleanly reverts on teardown, making the dedup embedding path
# order-independent.
try:  # pragma: no cover - defensive; slim slices may lack the module
    import lib.embedding.providers as _emb_providers_mod
    import lib.objectives.objective_dedup as _objective_dedup_mod

    def _live_build_embedding_client(*args, **kwargs):
        return _emb_providers_mod.build_embedding_client(*args, **kwargs)

    _objective_dedup_mod.build_embedding_client = _live_build_embedding_client
except Exception:  # noqa: BLE001 - never block collection on this guard
    pass

# NOTE: the "absent local synthesis backend → skip" hookwrapper lives in the
# repo-root conftest.py so it covers MCP synthesis-phase tests too. Not
# duplicated here.


# =============================================================================
# MCP CLIENT FIXTURES
# =============================================================================

@pytest.fixture
def mock_mcp_client():
    """AsyncMock MCP client for executor tests."""
    client = AsyncMock()
    client.call_tool = AsyncMock(return_value={"status": "success", "data": {}})
    return client


@pytest.fixture
def mock_mcp_client_with_errors():
    """MCP client that raises various errors."""
    client = AsyncMock()

    # Will raise different errors based on call count
    call_count = [0]

    async def call_tool_with_errors(tool_name, params):
        call_count[0] += 1
        if call_count[0] == 1:
            raise TimeoutError("Connection timeout")
        elif call_count[0] == 2:
            raise ConnectionError("Connection reset")
        else:
            return {"status": "success"}

    client.call_tool = AsyncMock(side_effect=call_tool_with_errors)
    return client


# =============================================================================
# CONFIG FIXTURES
# =============================================================================

@pytest.fixture
def sample_workflows_yaml():
    """Sample workflows.yaml content."""
    return """
workflows:
  course_generation:
    description: "Generate new course content"
    max_concurrent: 10
    phases:
      - name: planning
        agents: [course-outliner]
        timeout_minutes: 30
      - name: content_generation
        agents: [content-generator]
        timeout_minutes: 120
      - name: packaging
        agents: [brightspace-packager]
        timeout_minutes: 30

  rag_training:
    description: "Assessment-based RAG training"
    max_concurrent: 4
    phases:
      - name: extraction
        agents: [assessment-extractor]
        timeout_minutes: 60
"""


@pytest.fixture
def sample_agents_yaml():
    """Sample agents.yaml content."""
    return """
agents:
  course-outliner:
    description: "Creates course structure and learning objectives"
    tool: create_course_project
    capabilities:
      - course_planning
      - objective_mapping

  content-generator:
    description: "Generates educational content for modules"
    tool: generate_course_content
    capabilities:
      - content_generation
      - wcag_compliance

  brightspace-packager:
    description: "Packages course as IMSCC"
    tool: package_imscc
    capabilities:
      - imscc_packaging

  semantik-converter:
    description: "Converts PDF to accessible HTML"
    tool: extract_and_convert_pdf
    capabilities:
      - pdf_conversion
      - accessibility
"""


@pytest.fixture
def sample_config_dir(tmp_path, sample_workflows_yaml, sample_agents_yaml):
    """Create config directory with sample YAML files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "workflows.yaml").write_text(sample_workflows_yaml)
    (config_dir / "agents.yaml").write_text(sample_agents_yaml)

    return config_dir


# =============================================================================
# WORKFLOW STATE FIXTURES
# =============================================================================

@pytest.fixture
def sample_workflow_state():
    """Sample workflow state dictionary."""
    return {
        "workflow_id": "W001",
        "workflow_type": "course_generation",
        "status": "running",
        "created_at": "2025-01-01T00:00:00",
        "phases": {
            "planning": {"status": "completed"},
            "content_generation": {"status": "running"},
            "packaging": {"status": "pending"},
        },
        "tasks": {
            "T001": {
                "task_id": "T001",
                "agent_type": "course-outliner",
                "status": "complete",
                "params": {"course_name": "TEST_101"},
            },
            "T002": {
                "task_id": "T002",
                "agent_type": "content-generator",
                "status": "running",
                "params": {"week": 1, "module": 1},
            },
            "T003": {
                "task_id": "T003",
                "agent_type": "content-generator",
                "status": "pending",
                "params": {"week": 1, "module": 2},
            },
        },
    }


@pytest.fixture
def workflow_state_file(tmp_path, sample_workflow_state):
    """Create workflow state JSON file."""
    workflows_dir = tmp_path / "state" / "workflows"
    workflows_dir.mkdir(parents=True)

    state_path = workflows_dir / "W001.json"
    state_path.write_text(json.dumps(sample_workflow_state, indent=2))

    return state_path


# =============================================================================
# CHECKPOINT FIXTURES
# =============================================================================

@pytest.fixture
def checkpoint_manager(tmp_path):
    """CheckpointManager with temp state directory."""
    try:
        from MCP.hardening.checkpoint import CheckpointManager
        run_path = tmp_path / "runs" / "RUN_001"
        run_path.mkdir(parents=True)
        return CheckpointManager(run_path)
    except ImportError:
        pytest.skip("checkpoint module not available")


@pytest.fixture
def sample_checkpoint_data():
    """Sample checkpoint data dictionary."""
    return {
        "run_id": "RUN_001",
        "workflow_id": "W001",
        "phase_name": "content_generation",
        "phase_index": 1,
        "status": "started",
        "started_at": "2025-01-01T10:00:00",
        "tasks_completed": ["T001", "T002"],
        "tasks_failed": [],
        "tasks_pending": ["T003", "T004", "T005"],
        "last_event_seq": 42,
    }


# =============================================================================
# ERROR CLASSIFIER FIXTURES
# =============================================================================

@pytest.fixture
def error_classifier():
    """ErrorClassifier instance."""
    try:
        from MCP.hardening.error_classifier import ErrorClassifier
        return ErrorClassifier()
    except ImportError:
        pytest.skip("error_classifier module not available")


@pytest.fixture
def poison_pill_detector():
    """PoisonPillDetector instance."""
    try:
        from MCP.hardening.error_classifier import PoisonPillDetector
        return PoisonPillDetector(threshold=3, time_window_seconds=60)
    except ImportError:
        pytest.skip("error_classifier module not available")


@pytest.fixture
def sample_errors():
    """Collection of sample errors for classification testing."""
    return {
        "timeout": TimeoutError("Connection timeout"),
        "connection": ConnectionError("Connection reset by peer"),
        "rate_limit": Exception("Rate limit exceeded, retry after 60s"),
        "file_not_found": FileNotFoundError("No such file: config.yaml"),
        "permission": PermissionError("Access denied"),
        "validation": ValueError("Invalid input: expected string"),
        "memory": MemoryError("Out of memory"),
        "auth": Exception("Authentication failed: invalid API key"),
        "generic": Exception("Something went wrong"),
    }


# =============================================================================
# EXECUTOR FIXTURES
# =============================================================================

@pytest.fixture
def mock_tool_registry():
    """Mock tool registry for executor tests."""
    try:
        from lib.tool_registry import SandboxLevel, ToolCapability, ToolRegistry

        registry = ToolRegistry()

        # Register test tools
        registry.register(ToolCapability(
            tool_name="create_course_project",
            version="1.0.0",
            description="Create course project",
            sandbox_level=SandboxLevel.RESTRICTED,
        ))
        registry.register(ToolCapability(
            tool_name="generate_course_content",
            version="1.0.0",
            description="Generate content",
            sandbox_level=SandboxLevel.RESTRICTED,
        ))
        registry.register(ToolCapability(
            tool_name="package_imscc",
            version="1.0.0",
            description="Package IMSCC",
            sandbox_level=SandboxLevel.RESTRICTED,
        ))

        return registry

    except ImportError:
        # Return mock if not available
        registry = Mock()
        registry.get.return_value = Mock(tool_name="test_tool")
        registry.list_tools.return_value = ["create_course_project", "generate_course_content"]
        return registry


# =============================================================================
# TASK FIXTURES
# =============================================================================

@pytest.fixture
def sample_task():
    """Sample task dictionary."""
    return {
        "task_id": "T001",
        "agent_type": "content-generator",
        "status": "pending",
        "params": {
            "course_name": "TEST_101",
            "week": 1,
            "module": 1,
            "content_type": "introduction",
        },
        "dependencies": [],
    }


@pytest.fixture
def batch_tasks():
    """Batch of tasks for parallel execution testing."""
    return [
        {
            "task_id": f"T{i:03d}",
            "agent_type": "content-generator",
            "status": "pending",
            "params": {"week": (i // 3) + 1, "module": (i % 3) + 1},
            "dependencies": [],
        }
        for i in range(10)
    ]


# =============================================================================
# MODEL-HERMETICITY FIXTURES (Family-A test-cluster fix, Subtask-25 follow-up)
# =============================================================================
#
# Some MCP synthesis-phase tests (e.g. test_synthesis_training_phase.py) drive
# the real pair-promotion + claim-support validators, which lazy-load ~700MB
# BERT/DeBERTa models when the optional ``[bert]`` / ``[embedding]`` extras are
# installed (they ARE in this venv since 2026-05-08). Unit tests must NOT load
# real weights: it's slow AND real DeBERTa NLI legitimately rejects the toy
# fixtures these tests feed it. We monkeypatch the two model-loader seams to
# return None, routing both validators through their designed deps-missing
# pass arms. Integration tests opt out with ``@pytest.mark.real_models``.
#
# This is additive to the existing fixtures above and mirrors the autouse
# stub in Trainforge/tests/conftest.py.


def pytest_configure(config):
    """Register the ``real_models`` opt-out marker (``--strict-markers``)."""
    config.addinivalue_line(
        "markers",
        "real_models: opt out of the autouse model-loader stub; the test "
        "intentionally loads the real BERT/DeBERTa models (integration).",
    )


@pytest.fixture(autouse=True)
def _stub_real_models(request, monkeypatch):
    """Force the BERT-ensemble + NLI model loaders to return ``None``.

    Routes ``TrainingPairPromotionValidator`` (criterion 6 bloom) and the
    claim-support NLI path through their designed deps-missing pass arms so
    unit tests never load real weights. Opt out with
    ``@pytest.mark.real_models``.
    """
    if request.node.get_closest_marker("real_models"):
        yield
        return

    try:
        from lib.validators.pair.promotion import (
            TrainingPairPromotionValidator,
        )

        monkeypatch.setattr(
            TrainingPairPromotionValidator,
            "_resolve_bloom_classifier",
            lambda self: None,
        )
    except Exception:  # noqa: BLE001 — module may be absent in a slim slice
        pass

    try:
        from lib.classifiers.nli_classifier import NliClassifier

        monkeypatch.setattr(
            NliClassifier,
            "get_or_load",
            classmethod(lambda cls: None),
        )
    except Exception:  # noqa: BLE001
        pass

    # Seam 3: sentence-embedder resolver on the promotion validator.
    # Returning (None, False) routes criteria 2/3/4 through the designed
    # deps-missing fallback arms (answer-support floor -1.0, jaccard
    # distinctness, prompt-chunk floor deactivated) so unit tests never
    # load sentence-transformers weights. Mirrors seams 1-2.
    try:
        from lib.validators.pair.promotion import TrainingPairPromotionValidator
        monkeypatch.setattr(
            TrainingPairPromotionValidator,
            "_resolve_embedder",
            lambda self: (None, False),
        )
    except Exception:  # noqa: BLE001
        pass

    yield
