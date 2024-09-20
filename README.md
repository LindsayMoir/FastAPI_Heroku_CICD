# FastAPI Heroku CICD

## Overview

This repository provides an example of deploying a FastAPI machine learning model using Heroku and continuous integration with GitHub Actions. A command-line environment is recommended for ease of use with Git, DVC, and other tools. For Windows users, WSL1 or WSL2 is recommended. The repository does the following:
- Ingests data, trains a decision tree on that data and makes predictions.
- It does this in batch mode and also in real time via Heroku.
- Continuous Integration and Continuous Deployment (CICD) is enabled via workflows in GitHub and hooks into Heroku from GitHub.
- You can find the repository [here](https://github.com/LindsayMoir/FastAPI_Heroku_CICD).

## Environment Setup

### Install Conda

If you don’t already have it, download and install Conda.

### Set Up the Environment

1. Use the `requirements.txt` file to create a new environment, or

2. Create a new environment manually:
   
   ```bash
   conda create -n [envname] "python=3.11" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge```

### Install Git

   ```bash
   conda install git
   ```

## Files

```bash
./
├── CODEOWNERS
├── EDA.ipynb - Initial EDA on data
├── LICENSE.txt
├── Procfile
├── README.md
├── __pycache__/
│   ├── main.cpython-311.pyc
│   ├── model.cpython-311.pyc
│   └── test_fastapi_inference.cpython-311-pytest-7.4.4.pyc
├── config/
│   └── config.yaml - All constants
├── coverage.xml
├── data/
│   ├── census.csv - Original data
│   ├── cleaned_data.csv - Cleaned data
│   └── describe_cleaned_data.csv - Dataframe used for testing for outliers
├── directory_structure.txt - File that creates this listing of files
├── dvc_on_heroku_instructions.md
├── htmlcov/
│   ├── index.html - Open this with a browser to see statistics on test coverage
│   └── style_cb_8e611ae1.css
├── logs/
│   ├── app.log - Log from the running (all) of main.py (FastAPI)
│   └── ml.log - Log from the last run of model.py
├── main.py
├── ml/
│   ├── best_model.pkl - Best estimator and INCLUDES pipeline components
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── model.py - Python code that ingests data, cleans data, feature engineers,
               does inference (batch and single predictions), produces visualizations, and statistics
├── model_card.md
├── requirements.txt
├── runtime.txt
├── sanitycheck.py
├── screenshots/
│   ├── continuous_deployment.png
│   ├── continuous_integration.png
│   ├── example.png
│   ├──fast_api_post.png
│   ├── Heroku Login Apps Scale Status Pred Heroku Pred Local.png
│   ├── Heroku Login.png
│   ├── Heroku Overview of App Deployment.png
│   ├── Heroku Settings.png
│   ├── Heroku Web App Deployed single_pred.py
│   ├── Heroku_continuous_deployment.png
│   ├── live_get.png
│   ├── Push GitHub Auto Deploy Heroku.png
│   └── sanitycheck_run.png
├── setup.py
├── single_pred.py - Python code that does inference for one prediction
└── tests/
    ├── main_tests/
    └── model_tests/
```

13 directories, 84 files

## GitHub Actions

GitHub Actions have been implemented. `flake8` and `pytest` are automatically run every time there is a git push to the `main` branch.

## Model

The `model.py` script reads, cleans data, trains the model, and performs inference (batch and single prediction).

### Model.py Features

- Collects run statistics for every run.
- Uses a config file for all constants.
- Utilizes `best_estimator` (best_model) from the pipeline class for inference. Since this is part of the pipeline, feature engineering is handled automatically. The assumption is that any prediction requests will be cleaned by the web service.
- Calculates and visualizes feature importance.
- Produces visualizations (confusion matrix, ROC curve).
- Evaluates model slices by analyzing each categorical feature and its unique values, calculating all metrics to assess the impact.

### To run `model.py` locally:

1. Clone the repository.
2. Navigate to the root directory.
3. Provide a working directory as an argument:

```bash
python model.py [your working directory]
```

## FastAPI

### Local Web Server

1. Navigate to the directory where `main.py` is located (under the root).
2. Start the web server:

```bash
uvicorn main:app --reload
```

#### Run Inference (Post) Using Curl
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 39,
    "workclass": "State-gov",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40
  }'
```

##### Expected Prediction
```bash
{"prediction":"<=50K"}
```

#### Run Inference (Post) Using requests
Ensure the web server is still running. Then, run the following command from the root directory:
```bash
python single_pred.py
```

##### Expected Prediction
```bash
{"prediction":">50K"}
```

### Heroku Web Server
This application is deployed using Heroku's automatic deployment feature. Heroku updates the app automatically every time a new commit is pushed to GitHub.

##### Login
```bash
heroku login
```

##### Lists all apps that you have on Heroku
```bash
heroku apps
```

##### Supply resources to the app
```bash
heroku ps:scale web=1 --app income-projection
```

##### Get the status of the app
```bash
heroku ps --app income-projection
```

##### Open the app in a browser window
```bash
heroku open --app income-projection
```

##### Make sure you are in root directory
This will prove that the app is working and running on Heroku. Run single_pred.py
```bash
python single_pred.py
```

##### Scale the app down so it is using 0 resources
```bash
heroku ps:scale web=0 --app income-projection
```
