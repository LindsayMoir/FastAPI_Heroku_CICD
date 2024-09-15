def test_load_data_shape(data):
    """Test to assert that the data shape is correct."""
    # Assert the number of rows is between 10,000 and 1,000,000
    assert 10000 < len(data) < 1000000, "The number of rows should be \
        between 10,000 and 1,000,000."

    # Assert the number of columns is 15
    assert data.shape[1] == 15, "The number of columns should be equal to 15."