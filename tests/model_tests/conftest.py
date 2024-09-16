"""Configuration file for pytest."""

import pytest
import os
import sys

# Add parent directory (one level above) to sys.path so 'model' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(
    __file__), '../../')))

# Import setup_env and load_data after modifying sys.path
from model import setup_env, load_data  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Fixture to set up the environment before tests."""

    # Set the working directory (adjust according to your project structure)
    wd = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Call the setup_env function to set up the environment and load the config
    config = setup_env(wd)

    # Print the current working directory for verification
    print(f"Working directory set to: {wd}")

    return config


@pytest.fixture
def data():
    """Fixture to load the data using the load_data function."""
    return load_data()
