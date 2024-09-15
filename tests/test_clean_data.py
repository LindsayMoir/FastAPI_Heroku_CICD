import numpy as np
import os
import pandas as pd
from model import setup_env, clean_data  # Import the function and the config


def test_clean_data(data):
    """Test to assert that the columns are correct after cleaning the data."""
    # Get config
    config = setup_env(os.getcwd())

    # Apply the clean_data function
    cleaned_data = clean_data(data)

    # Get the expected columns from the config
    expected_columns = config['columns']['expected']

    # Assert the columns match the expected columns from the config
    assert cleaned_data.columns.tolist() == expected_columns

    # Get the resampled data
    resampled_data = pd.read_csv(config['data']['resampled_data'])

    # Create a describe dataframe for the resampled_data
    resampled_describe_df = resampled_data.describe()

    # remove the count row
    resampled_describe_df = resampled_describe_df.iloc[
        1:, :].reset_index(drop=True)

    # Smote skewed these columns a lot
    resampled_describe_df.drop(columns=['capital_gain', 'capital_loss'],
                               inplace=True)

    # Get the describe_df of the cleaned data
    describe_df = pd.read_csv(config['data']['describe_data_path'],
                              index_col=0)
    describe_df = describe_df.iloc[1:, :]  # remove the count row
    # Smote skewed these columns a lot. So, you need to drop them.
    describe_df.drop(columns=['capital_gain', 'capital_loss'],
                     inplace=True)

    # Calculate the absolute percentage difference
    diff = ((resampled_describe_df - describe_df) / describe_df).abs()

    # Check if any value in the diff dataframe exceeds 20%
    offside_values = diff > 0.2
    if offside_values.any().any():  # Check if any True values are present
        # Find the locations and values of the offside cells
        offside_cells = np.where(offside_values)
        messages = []
        for row, col in zip(offside_cells[0], offside_cells[1]):
            messages.append(
                f"Value offside at row '{resampled_describe_df.index[row]}', \
                column '{resampled_describe_df.columns[col]}': "
                f"resampled = {resampled_describe_df.iloc[row, col]}, \
                original = {describe_df.iloc[row, col]}, "
                f"diff = {diff.iloc[row, col]:.2%}"
            )
        # Fail the test with detailed messages
        raise AssertionError("\n".join(messages))
