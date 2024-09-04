import pytest
import os
import sys
import yaml
import logging

# Add the directory containing model.py to sys.path
sys.path.append('/mnt/d/GitHub/nd0821-c3-starter-code/starter')

from model import setup_env, load_data  # Import setup_env and load_data

@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Fixture to set up the environment before tests."""
    # Define the working directory for the tests
    working_directory = '/mnt/d/GitHub/nd0821-c3-starter-code/starter'
    
    # Call the setup_env function to set up the environment and load the config
    config = setup_env(working_directory)
    
    # Print the current working directory for verification
    print(f"Working directory set to: {os.getcwd()}")

@pytest.fixture
def data():
    """Fixture to load the data using the load_data function."""
    return load_data()
