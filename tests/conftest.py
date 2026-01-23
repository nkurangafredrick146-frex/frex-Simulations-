"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_config():
    """Provide a sample configuration for tests."""
    return {
        "resolution": (1920, 1080),
        "fps": 60,
        "quality": "high",
        "enable_physics": True,
        "enable_rendering": True,
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path / "outputs"


@pytest.fixture(scope="session")
def project_root_path():
    """Provide the project root path."""
    return Path(__file__).parent.parent
