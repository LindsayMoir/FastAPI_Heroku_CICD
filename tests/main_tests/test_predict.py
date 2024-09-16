import os
import sys
from fastapi.testclient import TestClient

# Add parent directory (2 levels above) to sys.path so 'model' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(
    __file__), '../../')))

from main import app  # noqa: E402

client = TestClient(app)


def test_post_inference_low():
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40
    }

    # Get the response from the API
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    # Check the response
    prediction = response.json().get("prediction")
    assert prediction == "<=50K", f"Unexpected prediction value: {prediction}"


def test_post_inference_high():
    payload = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "capital_gain": 15000,
        "capital_loss": 0,
        "hours_per_week": 45
    }

    # Get the response from the API
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    # Check the response
    prediction = response.json().get("prediction")
    assert prediction == ">50K", f"Unexpected prediction value: {prediction}"
