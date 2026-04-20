"""
Ed4All Pipeline Orchestrator

High-level orchestration layer that sits on top of the WorkflowRunner
engine and exposes a mode-agnostic entry point for running workflows.

Wave 7 introduces:
- PipelineOrchestrator: front controller that dispatches phases through
  mode-specific dispatchers (local Claude Code subagent vs. API backend)
- LLMBackend protocol + implementations (LocalBackend, AnthropicBackend,
  OpenAIBackend stub, MockBackend)
- Worker contracts (PhaseInput, PhaseOutput, GateResult) shared between
  dispatchers so every worker speaks the same JSON-serializable language

This package is additive: existing callers that go through WorkflowRunner
directly keep working. The orchestrator is the new primary surface.
"""

from .llm_backend import (
    AnthropicBackend,
    LLMBackend,
    LocalBackend,
    MockBackend,
    OpenAIBackend,
    build_backend,
)
from .pipeline_orchestrator import PipelineOrchestrator
from .worker_contracts import GateResult, PhaseInput, PhaseOutput

__all__ = [
    "AnthropicBackend",
    "GateResult",
    "LLMBackend",
    "LocalBackend",
    "MockBackend",
    "OpenAIBackend",
    "PhaseInput",
    "PhaseOutput",
    "PipelineOrchestrator",
    "build_backend",
]
