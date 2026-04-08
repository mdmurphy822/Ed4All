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
