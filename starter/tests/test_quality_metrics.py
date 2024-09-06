import os
from model import setup_env
import pandas as pd


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

    # Check if the files have been created within an appropriate time frame
    df = pd.read_csv(config['files']['run_log'])

    # Get the start and end times of the last run
    df = pd.read_csv(config['files']['run_log'])
    start_time = pd.to_datetime(df.loc[df.index[-1], 'start_time'])
    end_time = pd.to_datetime(df.loc[df.index[-1], 'end_time'])

    # Get the time the files were last modified
    conf_matrix_mtime = pd.to_datetime(os.path.getmtime(conf_matrix_path), unit='s')
    roc_curve_mtime = pd.to_datetime(os.path.getmtime(roc_curve_path), unit='s')

    # This is unix time which is utc. Just subtract 7 hours (pst) to get the correct time.
    conf_matrix_mtime = conf_matrix_mtime - pd.Timedelta(hours=7)
    roc_curve_mtime = roc_curve_mtime - pd.Timedelta(hours=7)

    # Check if the files were created within the appropriate time frame
    assert start_time <= conf_matrix_mtime <= end_time, f"\
        File {conf_matrix_path} was not created within the appropriate time frame."
    assert start_time <= roc_curve_mtime <= end_time, f"\
        File {roc_curve_path} was not created within the appropriate time frame."
