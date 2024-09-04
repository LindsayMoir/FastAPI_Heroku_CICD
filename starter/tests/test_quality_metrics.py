import os
from model import setup_env


def test_quality_metrics(y_test = None, predictions = None):
    """Test to check if the quality_metrics function generates the expected files."""
    # Get config
    config = setup_env(os.getcwd())
    
    # Define the expected file paths
    conf_matrix_path = config['models']['confusion_matrix']
    roc_curve_path = config['models']['roc_curve']
    
    # Check if the files exist
    assert os.path.exists(conf_matrix_path), f"File {conf_matrix_path} was not created."
    assert os.path.exists(roc_curve_path), f"File {roc_curve_path} was not created."
