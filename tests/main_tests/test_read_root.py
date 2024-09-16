from fastapi.testclient import TestClient
from main import app  # Import the FastAPI app

# Create a TestClient instance with the FastAPI app
client = TestClient(app)


def test_read_root():
    # Send a GET request to the root endpoint
    response = client.get("/")

    # Assert that the response status code is 200 (OK)
    assert response.status_code == 200

    # Assert that the response JSON matches the expected message
    assert response.json() == {
        "message": "Deploying a ML Model with GBC, FastAPI, Heroku, and DVC"}
