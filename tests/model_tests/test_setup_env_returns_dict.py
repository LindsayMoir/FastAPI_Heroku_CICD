import os
from model import setup_env


def test_setup_env_returns_dict():
    """
    Test that the setup_env function returns a dictionary.
    """
    # Use the working directory for the test, adjust the path accordingly
    working_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../'))

    # Call the setup_env function
    config = setup_env(working_directory)

    # Ensure that the returned value is a dictionary
    assert isinstance(config, dict), "setup_env should return a dictionary"

    # Ensure that the 'files' section and 'log_file' key exist in the config
    assert 'files' in config, "'files' section is missing in the config"
    assert 'log_file' in config['files'], "'log_file' missing config['files']"

    # Optional: Test that the correct working directory has been set
    assert os.getcwd() == working_directory, "Wrong working directory"
