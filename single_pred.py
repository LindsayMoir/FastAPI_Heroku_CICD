import json
import os
import requests
import yaml

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract the log file path from the configuration
log_file_path = config['files']['log_file']

# Define the URLs
heroku_url = config['urls']['heroku_url']
local_url = config['urls']['local_url']

# Get the base URL from the environment variable or default to Heroku URL
url = os.getenv("FASTAPI_URL", heroku_url)

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


def send_request(url, data, headers):
    try:
        # Make the POST request to the FastAPI app with the data
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response from the FastAPI app
            result = response.json()
            return result, None
        elif response.status_code == 503:
            # Handle specific case where the Heroku app is unavailable
            return None, "Heroku is currently unavailable."
        else:
            # Handle other response codes
            return None, f"Failed to get a successful response. Status code: \
            {response.status_code}"
    except requests.exceptions.RequestException:
        # Handle any exceptions that occurs during the request
        return None, "An error occurred while sending the request."


# Try sending the request to Heroku first
print('Trying Heroku URL:', heroku_url)
result, error = send_request(heroku_url, data, headers)

# If there was an error (including 503), fallback to local URL
if error:
    print("Error:", error)
    print("Falling back to local server:", local_url)
    result, error = send_request(local_url, data, headers)
else:
    print("This prediction is from Heroku:", heroku_url)

# Print the result
if result:
    print(f"Response using FastAPI: {result}")
else:
    print("Failed to get a response from both Heroku and local servers.")
