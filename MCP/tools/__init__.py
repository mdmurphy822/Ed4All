"""
Ed4All MCP Tools

Tool modules for Courseforge, Trainforge, and Orchestration operations.
"""

from .courseforge_tools import register_courseforge_tools
from .orchestrator_tools import register_orchestrator_tools
from .trainforge_tools import register_trainforge_tools

__all__ = [
    'register_courseforge_tools',
    'register_orchestrator_tools',
    'register_trainforge_tools',
]
