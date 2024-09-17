import json
import os
import requests

# Get the base URL from the environment variable or default to localhost
url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict")

# Data to send in the POST request
data = {
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

# Set the headers for the request, indicating the data is in JSON format
headers = {
    "Content-Type": "application/json"
}

try:
    # Make the POST request to the FastAPI app with the data
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response from the FastAPI app
        result = response.json()
        prediction = result.get("prediction", "No prediction found")
        print(f"Response from FastAPI: {result}")
    else:
        print(f"Failed to get a successful response. Status code: \
        {response.status_code}")
        print(f"Response content: {response.text}")

except requests.exceptions.RequestException as e:
    # Handle any exceptions that occur during the request
    print(f"An error occurred: {e}")
