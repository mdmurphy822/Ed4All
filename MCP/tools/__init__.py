"""
Ed4All MCP Tools

Tool modules for DART, Courseforge, Trainforge, and Orchestration operations.
"""

from .courseforge_tools import register_courseforge_tools
from .dart_tools import register_dart_tools
from .orchestrator_tools import register_orchestrator_tools
from .trainforge_tools import register_trainforge_tools

__all__ = [
    'register_dart_tools',
    'register_courseforge_tools',
    'register_orchestrator_tools',
    'register_trainforge_tools',
]
