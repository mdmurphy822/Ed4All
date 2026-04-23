"""
Worker contracts for the pipeline orchestrator.

Every phase worker, whether dispatched as a Claude Code subagent (local mode)
or run as a Python coroutine (api mode), produces and consumes the same
shape defined here. These dataclasses are JSON-serializable so they can
round-trip through subagent boundaries, state checkpoints, and inter-process
handoff.

See: plans/pipeline-orchestration/design.md sections "Phase worker contracts"
and "LLM backend abstraction".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional


def _path_to_str(value: Any) -> Any:
    """Recursively convert Path objects to strings for JSON serialization."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _path_to_str(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_path_to_str(v) for v in value]
    return value


@dataclass
class GateResult:
    """Result of running a single validation gate against a phase output.

    Mirrors the hardening layer's validation gate result shape so that
    orchestrator callers can treat gate results uniformly across phases.
    """

    gate_id: str
    severity: Literal["critical", "warning"]
    passed: bool
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _path_to_str(asdict(self))


@dataclass
class PhaseInput:
    """Input packet handed to a phase worker.

    ``llm_factory`` is a zero-arg callable that yields a fresh ``LLMBackend``
    each time. Callers that need a backend should use the factory rather than
    hold onto a single instance — this keeps the contract JSON-friendly when
    the dataclass is serialized (factory is excluded via dict_filter below).
    """

    run_id: str
    workflow_type: str
    phase_name: str
    phase_config: Dict[str, Any]
    params: Dict[str, Any]
    mode: Literal["local", "api"]
    llm_factory: Optional[Callable[[], Any]] = None
    project_root: Optional[Path] = None
    state_dir: Optional[Path] = None
    captures_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (drops ``llm_factory``)."""
        data = {
            "run_id": self.run_id,
            "workflow_type": self.workflow_type,
            "phase_name": self.phase_name,
            "phase_config": self.phase_config,
            "params": self.params,
            "mode": self.mode,
            "project_root": str(self.project_root) if self.project_root else None,
            "state_dir": str(self.state_dir) if self.state_dir else None,
            "captures_dir": str(self.captures_dir) if self.captures_dir else None,
        }
        return _path_to_str(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class PhaseOutput:
    """Output packet produced by a phase worker.

    ``status`` is one of ``ok`` | ``warn`` | ``fail``:
    - ``ok``: phase succeeded and all gates passed
    - ``warn``: phase succeeded but at least one non-critical gate flagged issues
    - ``fail``: phase or critical gate failed; orchestrator should halt (or retry)
    """

    run_id: str
    phase_name: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Path] = field(default_factory=list)
    gate_results: Dict[str, GateResult] = field(default_factory=dict)
    decision_captures_path: Optional[Path] = None
    status: Literal["ok", "warn", "fail"] = "ok"
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "phase_name": self.phase_name,
            "outputs": self.outputs,
            "artifacts": [str(p) for p in self.artifacts],
            "gate_results": {
                gid: gr.to_dict() if isinstance(gr, GateResult) else gr
                for gid, gr in self.gate_results.items()
            },
            "decision_captures_path": (
                str(self.decision_captures_path)
                if self.decision_captures_path
                else None
            ),
            "status": self.status,
            "error": self.error,
            "metrics": self.metrics,
        }
        return _path_to_str(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhaseOutput:
        """Rebuild a PhaseOutput from its serialized dict form."""
        gate_results = {}
        for gid, gr in (data.get("gate_results") or {}).items():
            if isinstance(gr, GateResult):
                gate_results[gid] = gr
            else:
                gate_results[gid] = GateResult(
                    gate_id=gr.get("gate_id", gid),
                    severity=gr.get("severity", "warning"),
                    passed=bool(gr.get("passed", False)),
                    issues=gr.get("issues", []) or [],
                    details=gr.get("details", {}) or {},
                )
        artifacts = [Path(p) for p in (data.get("artifacts") or [])]
        captures = data.get("decision_captures_path")
        return cls(
            run_id=data.get("run_id", ""),
            phase_name=data.get("phase_name", ""),
            outputs=data.get("outputs", {}) or {},
            artifacts=artifacts,
            gate_results=gate_results,
            decision_captures_path=Path(captures) if captures else None,
            status=data.get("status", "ok"),
            error=data.get("error"),
            metrics=data.get("metrics", {}) or {},
        )
