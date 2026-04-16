"""
Tests for orchestrator/core/config.py - Orchestrator configuration management.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from MCP.core.config import (
        AgentConfig,
        OrchestratorConfig,
        WorkflowConfig,
        WorkflowPhase,
    )
except ImportError:
    pytest.skip("config not available", allow_module_level=True)


# =============================================================================
# WORKFLOW PHASE TESTS
# =============================================================================

class TestWorkflowPhase:
    """Test WorkflowPhase dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """Should have sensible defaults."""
        phase = WorkflowPhase(
            name="planning",
            agents=["course-outliner"],
        )

        assert phase.name == "planning"
        assert phase.parallel is True
        assert phase.max_concurrent == 10
        assert phase.batch_by is None

    @pytest.mark.unit
    def test_custom_values(self):
        """Should accept custom values."""
        phase = WorkflowPhase(
            name="content_generation",
            agents=["content-generator"],
            parallel=False,
            max_concurrent=5,
            batch_by="week",
        )

        assert phase.parallel is False
        assert phase.max_concurrent == 5
        assert phase.batch_by == "week"


# =============================================================================
# WORKFLOW CONFIG TESTS
# =============================================================================

class TestWorkflowConfig:
    """Test WorkflowConfig dataclass."""

    @pytest.mark.unit
    def test_create_workflow_config(self):
        """Should create workflow config with phases."""
        phases = [
            WorkflowPhase(name="planning", agents=["course-outliner"]),
            WorkflowPhase(name="generation", agents=["content-generator"]),
        ]

        config = WorkflowConfig(
            description="Test workflow",
            phases=phases,
        )

        assert config.description == "Test workflow"
        assert len(config.phases) == 2
        assert config.phases[0].name == "planning"


# =============================================================================
# AGENT CONFIG TESTS
# =============================================================================

class TestAgentConfig:
    """Test AgentConfig dataclass."""

    @pytest.mark.unit
    def test_default_max_instances(self):
        """Should default to 1 max instance."""
        agent = AgentConfig(
            source="courseforge/agents/outliner.md",
            type="planning",
            capabilities=["course_structure"],
        )

        assert agent.max_instances == 1

    @pytest.mark.unit
    def test_custom_max_instances(self):
        """Should accept custom max instances."""
        agent = AgentConfig(
            source="courseforge/agents/generator.md",
            type="generation",
            capabilities=["content_generation"],
            max_instances=10,
        )

        assert agent.max_instances == 10


# =============================================================================
# ORCHESTRATOR CONFIG LOADING TESTS
# =============================================================================

class TestOrchestratorConfigLoading:
    """Test OrchestratorConfig loading from YAML."""

    @pytest.fixture
    def valid_config_dir(self, tmp_path):
        """Create config directory with valid YAML files."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        workflows = """
workflows:
  course_generation:
    description: "Generate course content"
    phases:
      - name: planning
        agents:
          - course-outliner
        parallel: false
      - name: content_generation
        agents:
          - content-generator
        max_concurrent: 10
"""
        (config_dir / "workflows.yaml").write_text(workflows)

        agents = """
agents:
  course-outliner:
    source: "courseforge/agents/outliner.md"
    type: "planning"
    capabilities:
      - course_structure
      - objective_mapping

  content-generator:
    source: "courseforge/agents/generator.md"
    type: "generation"
    capabilities:
      - content_generation
    max_instances: 10
"""
        (config_dir / "agents.yaml").write_text(agents)

        return config_dir

    @pytest.mark.unit
    def test_load_valid_config(self, valid_config_dir):
        """Should load valid configuration."""
        config = OrchestratorConfig.load(valid_config_dir)

        assert "course_generation" in config.workflows
        assert "course-outliner" in config.agents

    @pytest.mark.unit
    def test_load_workflow_phases(self, valid_config_dir):
        """Should load workflow phases correctly."""
        config = OrchestratorConfig.load(valid_config_dir)

        workflow = config.workflows["course_generation"]
        assert len(workflow.phases) == 2
        assert workflow.phases[0].name == "planning"
        assert workflow.phases[0].parallel is False
        assert workflow.phases[1].max_concurrent == 10

    @pytest.mark.unit
    def test_load_agent_capabilities(self, valid_config_dir):
        """Should load agent capabilities."""
        config = OrchestratorConfig.load(valid_config_dir)

        agent = config.agents["course-outliner"]
        assert "course_structure" in agent.capabilities
        assert "objective_mapping" in agent.capabilities

    @pytest.mark.unit
    def test_load_missing_dir_returns_empty(self, tmp_path):
        """Should return empty config for missing directory."""
        config = OrchestratorConfig.load(tmp_path / "nonexistent")

        assert config.workflows == {}
        assert config.agents == {}

    @pytest.mark.unit
    def test_load_invalid_yaml_raises(self, tmp_path):
        """Should raise ValueError for invalid YAML."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "workflows.yaml").write_text("invalid: yaml: syntax:")

        with pytest.raises(ValueError, match="Invalid YAML"):
            OrchestratorConfig.load(config_dir)

    @pytest.mark.unit
    def test_load_empty_file_raises(self, tmp_path):
        """Should raise ValueError for empty config file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "workflows.yaml").write_text("")

        with pytest.raises(ValueError, match="Empty config file"):
            OrchestratorConfig.load(config_dir)

    @pytest.mark.unit
    def test_load_non_dict_raises(self, tmp_path):
        """Should raise ValueError for non-dict config."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "workflows.yaml").write_text("- just\n- a\n- list")

        with pytest.raises(ValueError, match="must be a dictionary"):
            OrchestratorConfig.load(config_dir)


# =============================================================================
# CONFIG ACCESSOR TESTS
# =============================================================================

class TestConfigAccessors:
    """Test config accessor methods."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create config with test data."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        workflows = """
workflows:
  test_workflow:
    description: "Test"
    phases:
      - name: test_phase
        agents: [test-agent]
"""
        (config_dir / "workflows.yaml").write_text(workflows)

        agents = """
agents:
  test-agent:
    source: "test.md"
    type: "test"
    capabilities: [testing]
"""
        (config_dir / "agents.yaml").write_text(agents)

        return OrchestratorConfig.load(config_dir)

    @pytest.mark.unit
    def test_get_workflow(self, config):
        """Should get workflow by name."""
        workflow = config.get_workflow("test_workflow")

        assert workflow is not None
        assert workflow.description == "Test"

    @pytest.mark.unit
    def test_get_workflow_missing(self, config):
        """Should return None for missing workflow."""
        workflow = config.get_workflow("nonexistent")

        assert workflow is None

    @pytest.mark.unit
    def test_get_agent(self, config):
        """Should get agent by name."""
        agent = config.get_agent("test-agent")

        assert agent is not None
        assert agent.type == "test"

    @pytest.mark.unit
    def test_get_agent_missing(self, config):
        """Should return None for missing agent."""
        agent = config.get_agent("nonexistent")

        assert agent is None

    @pytest.mark.unit
    def test_get_all_workflow_agents(self, config):
        """Should get all unique agents from workflows."""
        agents = config.get_all_workflow_agents()

        assert "test-agent" in agents

    @pytest.mark.unit
    def test_check_agents_exist(self, config):
        """Should identify missing agents."""
        missing = config.check_agents_exist(["test-agent", "missing-agent"])

        assert "missing-agent" in missing
        assert "test-agent" not in missing


# =============================================================================
# CONFIG VALIDATION TESTS
# =============================================================================

class TestConfigValidation:
    """Test config validation."""

    @pytest.fixture
    def valid_config_dir(self, tmp_path):
        """Create config directory with valid YAML files."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        workflows = """
workflows:
  test_workflow:
    description: "Test workflow"
    phases:
      - name: planning
        agents:
          - existing-agent
"""
        (config_dir / "workflows.yaml").write_text(workflows)

        agents = """
agents:
  existing-agent:
    source: ""
    type: "test"
    capabilities: [testing]
"""
        (config_dir / "agents.yaml").write_text(agents)

        return config_dir

    @pytest.fixture
    def invalid_config_dir(self, tmp_path):
        """Create config directory with missing agent reference."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        workflows = """
workflows:
  test_workflow:
    description: "Test workflow"
    phases:
      - name: planning
        agents:
          - missing-agent
"""
        (config_dir / "workflows.yaml").write_text(workflows)

        agents = """
agents:
  other-agent:
    source: ""
    type: "test"
    capabilities: [testing]
"""
        (config_dir / "agents.yaml").write_text(agents)

        return config_dir

    @pytest.mark.unit
    def test_validate_valid_config(self, valid_config_dir):
        """Should pass validation for valid config."""
        config = OrchestratorConfig.load(valid_config_dir)

        issues = config.validate(fail_fast=False)

        assert len(issues["missing_agents"]) == 0

    @pytest.mark.unit
    def test_validate_missing_agents(self, invalid_config_dir):
        """Should detect missing agent references."""
        config = OrchestratorConfig.load(invalid_config_dir)

        issues = config.validate(fail_fast=False)

        assert len(issues["missing_agents"]) == 1
        assert "missing-agent" in issues["missing_agents"][0]

    @pytest.mark.unit
    def test_validate_fail_fast_raises(self, invalid_config_dir):
        """Should raise on missing agents when fail_fast=True."""
        config = OrchestratorConfig.load(invalid_config_dir)

        with pytest.raises(ValueError, match="missing-agent"):
            config.validate(fail_fast=True)

    @pytest.mark.unit
    def test_validate_and_raise(self, invalid_config_dir):
        """validate_and_raise should raise on issues."""
        config = OrchestratorConfig.load(invalid_config_dir)

        with pytest.raises(ValueError):
            config.validate_and_raise()


# =============================================================================
# DEFAULT VALUES TESTS
# =============================================================================

class TestDefaultValues:
    """Test OrchestratorConfig default values."""

    @pytest.mark.unit
    def test_default_orchestrator_settings(self):
        """Should have sensible default values."""
        config = OrchestratorConfig()

        assert config.max_parallel_agents == 12
        assert config.task_timeout_minutes == 60
        assert config.retry_attempts == 2
        assert config.retry_delay_seconds == 30

    @pytest.mark.unit
    def test_default_batch_settings(self):
        """Should have default batch settings."""
        config = OrchestratorConfig()

        assert config.max_batch_size == 10
        assert config.lock_ttl_seconds == 3600

    @pytest.mark.unit
    def test_default_paths_set(self):
        """Should have default paths from lib.paths."""
        config = OrchestratorConfig()

        assert config.project_dir is not None
        assert config.state_dir is not None
        assert config.config_dir is not None
