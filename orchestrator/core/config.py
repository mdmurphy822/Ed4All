"""
Orchestrator Configuration

Centralized configuration management for the Ed4All orchestrator.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Add project path for lib imports
_CORE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CORE_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import (  # noqa: E402
    CONFIG_PATH,
    COURSEFORGE_PATH,
    DART_PATH,
    PROJECT_ROOT,
    STATE_PATH,
    TRAINFORGE_PATH,
)

logger = logging.getLogger(__name__)


# Use centralized paths
PROJECT_DIR = PROJECT_ROOT


@dataclass
class WorkflowPhase:
    """Configuration for a workflow phase."""
    name: str
    agents: List[str]
    parallel: bool = True
    max_concurrent: int = 10
    batch_by: Optional[str] = None


@dataclass
class WorkflowConfig:
    """Configuration for a workflow type."""
    description: str
    phases: List[WorkflowPhase]


@dataclass
class AgentConfig:
    """Configuration for an agent type."""
    source: str
    type: str
    capabilities: List[str]
    max_instances: int = 1


@dataclass
class OrchestratorConfig:
    """Main orchestrator configuration."""

    # Paths - use centralized paths from lib.paths
    project_dir: Path = field(default_factory=lambda: PROJECT_ROOT)
    state_dir: Path = field(default_factory=lambda: STATE_PATH)
    config_dir: Path = field(default_factory=lambda: CONFIG_PATH)
    dart_dir: Path = field(default_factory=lambda: DART_PATH)
    courseforge_dir: Path = field(default_factory=lambda: COURSEFORGE_PATH)
    trainforge_dir: Path = field(default_factory=lambda: TRAINFORGE_PATH)

    # Orchestrator settings
    max_parallel_agents: int = 12
    task_timeout_minutes: int = 60
    retry_attempts: int = 2
    retry_delay_seconds: int = 30

    # Batch settings
    max_batch_size: int = 10
    lock_ttl_seconds: int = 3600

    # Workflows
    workflows: Dict[str, WorkflowConfig] = field(default_factory=dict)

    # Agents
    agents: Dict[str, AgentConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, config_dir: Optional[Path] = None) -> "OrchestratorConfig":
        """
        Load configuration from YAML files.

        Raises:
            ValueError: If config files are invalid or malformed
            FileNotFoundError: If required config files don't exist
        """
        config_dir = config_dir or PROJECT_DIR / "config"

        instance = cls()

        # Load workflows
        workflows_path = config_dir / "workflows.yaml"
        if workflows_path.exists():
            try:
                with open(workflows_path, 'r') as f:
                    data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {workflows_path}: {e}") from e
            except FileNotFoundError:
                raise ValueError(f"Config file not found: {workflows_path}") from None

            # Validate data structure
            if data is None:
                raise ValueError(f"Empty config file: {workflows_path}")

            if not isinstance(data, dict):
                raise ValueError(f"Config must be a dictionary, got {type(data).__name__}")

            try:
                workflows = data.get("workflows", {})
                if not isinstance(workflows, dict):
                    raise TypeError("workflows must be a dictionary")

                for name, wf_data in workflows.items():
                    if not isinstance(wf_data, dict):
                        raise TypeError(f"Workflow '{name}' config must be a dictionary")

                    phases_list = wf_data.get("phases", [])
                    if not isinstance(phases_list, list):
                        raise TypeError(f"Workflow '{name}' phases must be a list")

                    phases = []
                    for p in phases_list:
                        if not isinstance(p, dict):
                            raise TypeError(f"Phase config in workflow '{name}' must be a dictionary")

                        if "name" not in p:
                            raise KeyError(f"Phase in workflow '{name}' missing required 'name' field")

                        phases.append(WorkflowPhase(
                            name=p["name"],
                            agents=p.get("agents", []),
                            parallel=p.get("parallel", True),
                            max_concurrent=p.get("max_concurrent", 10),
                            batch_by=p.get("batch_by")
                        ))

                    instance.workflows[name] = WorkflowConfig(
                        description=wf_data.get("description", ""),
                        phases=phases
                    )
            except (KeyError, TypeError) as e:
                raise ValueError(f"Malformed workflows config structure: {e}") from e

        # Load agents
        agents_path = config_dir / "agents.yaml"
        if agents_path.exists():
            try:
                with open(agents_path, 'r') as f:
                    data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {agents_path}: {e}") from e
            except FileNotFoundError:
                raise ValueError(f"Config file not found: {agents_path}") from None

            # Validate data structure
            if data is None:
                raise ValueError(f"Empty config file: {agents_path}")

            if not isinstance(data, dict):
                raise ValueError(f"Config must be a dictionary, got {type(data).__name__}")

            try:
                agents = data.get("agents", {})
                if not isinstance(agents, dict):
                    raise TypeError("agents must be a dictionary")

                for name, agent_data in agents.items():
                    if not isinstance(agent_data, dict):
                        raise TypeError(f"Agent '{name}' config must be a dictionary")

                    instance.agents[name] = AgentConfig(
                        source=agent_data.get("source", ""),
                        type=agent_data.get("type", ""),
                        capabilities=agent_data.get("capabilities", []),
                        max_instances=agent_data.get("max_instances", 1)
                    )
            except (KeyError, TypeError) as e:
                raise ValueError(f"Malformed agents config structure: {e}") from e

        return instance

    def get_workflow(self, name: str) -> Optional[WorkflowConfig]:
        """Get workflow configuration by name."""
        return self.workflows.get(name)

    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name."""
        return self.agents.get(name)

    def validate(self, fail_fast: bool = True) -> Dict[str, List[str]]:
        """
        Validate configuration consistency and integrity.

        Checks:
        1. All agents referenced in workflows exist in agents config
        2. All agent source files exist on disk
        3. Phase dependencies reference valid phases (if defined)

        Args:
            fail_fast: If True, raise ValueError on first error.
                      If False, collect and return all issues.

        Returns:
            Dict with categorized issues: 'missing_agents', 'missing_sources',
            'invalid_dependencies'

        Raises:
            ValueError: If fail_fast=True and validation fails.
        """
        issues = {
            "missing_agents": [],
            "missing_sources": [],
            "invalid_dependencies": [],
            "invalid_validators": [],
        }

        # Check 1: All workflow agents exist in agents config
        for workflow_name, workflow in self.workflows.items():
            for phase in workflow.phases:
                for agent_name in phase.agents:
                    if agent_name not in self.agents:
                        issue = f"Workflow '{workflow_name}' phase '{phase.name}' references unknown agent: '{agent_name}'"
                        issues["missing_agents"].append(issue)
                        if fail_fast:
                            raise ValueError(issue)

        # Check 2: All agent source files exist
        for agent_name, agent_config in self.agents.items():
            if agent_config.source:
                source_path = self.project_dir / agent_config.source
                if not source_path.exists():
                    issue = f"Agent '{agent_name}' source not found: {agent_config.source}"
                    issues["missing_sources"].append(issue)
                    if fail_fast:
                        raise ValueError(issue)

        # Check 3: Validate validator paths from workflows.yaml can be imported
        workflows_path = self.config_dir / "workflows.yaml"
        if workflows_path.exists():
            try:
                with open(workflows_path, 'r') as f:
                    raw_config = yaml.safe_load(f) or {}
                for wf_name, wf_data in raw_config.get("workflows", {}).items():
                    for phase in wf_data.get("phases", []):
                        for gate in phase.get("validation_gates", []):
                            validator_path = gate.get("validator", "")
                            if not validator_path:
                                continue
                            if '.' not in validator_path:
                                issue = f"Invalid validator path (no module separator): '{validator_path}' in workflow '{wf_name}'"
                                issues["invalid_validators"].append(issue)
                                if fail_fast:
                                    raise ValueError(issue)
                                continue
                            module_path, class_name = validator_path.rsplit('.', 1)
                            try:
                                import importlib
                                mod = importlib.import_module(module_path)
                                if not hasattr(mod, class_name):
                                    issue = f"Validator class '{class_name}' not found in module '{module_path}' (workflow '{wf_name}')"
                                    issues["invalid_validators"].append(issue)
                                    if fail_fast:
                                        raise ValueError(issue)
                            except ImportError as e:
                                issue = f"Cannot import validator module '{module_path}': {e} (workflow '{wf_name}')"
                                issues["invalid_validators"].append(issue)
                                if fail_fast:
                                    raise ValueError(issue)
            except yaml.YAMLError:
                pass  # Already validated during load()

        # Log summary if not fail_fast
        if not fail_fast:
            total_issues = sum(len(v) for v in issues.values())
            if total_issues > 0:
                logger.warning(f"Config validation found {total_issues} issues")
                for category, category_issues in issues.items():
                    for issue in category_issues:
                        logger.warning(f"  [{category}] {issue}")
            else:
                logger.info("Config validation passed")

        return issues

    def validate_and_raise(self) -> None:
        """
        Validate configuration and raise on any issues.

        This is a convenience method for fail-fast validation at startup.

        Raises:
            ValueError: If any validation check fails.
        """
        self.validate(fail_fast=True)

    def get_all_workflow_agents(self) -> List[str]:
        """Get all unique agent names referenced in workflows."""
        agents = set()
        for workflow in self.workflows.values():
            for phase in workflow.phases:
                agents.update(phase.agents)
        return sorted(agents)

    def check_agents_exist(self, agent_names: List[str]) -> List[str]:
        """
        Check which agents from the list don't exist in config.

        Args:
            agent_names: List of agent names to check

        Returns:
            List of missing agent names
        """
        return [name for name in agent_names if name not in self.agents]
