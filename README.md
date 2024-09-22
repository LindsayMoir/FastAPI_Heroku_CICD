# FastAPI Heroku CICD

![Confusion Matrix](/ml/confusion_matrix.png)

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
  - [Install Conda](#install-conda)
  - [Set Up the Environment](#set-up-the-environment)
  - [Install Git](#install-git)
- [Files](#files)
- [GitHub Actions](#github-actions)
- [Model](#model)
  - [Model.py Features](#modelpy-features)
  - [Run Model.py Locally](#to-run-modelpy-locally)
- [FastAPI](#fastapi)
  - [Local Web Server](#local-web-server)
  - [Run Inference Using Curl](#run-inference-post-using-curl)
  - [Run Inference Using requests](#run-inference-post-using-requests)
  - [Heroku Web Server](#heroku-web-server)
- [Screenshots](#screenshots)
- [Summary](#summary)

## Overview

This repository provides an example of deploying a FastAPI machine learning model using Heroku and continuous integration with GitHub Actions. A command-line environment is recommended for ease of use with Git, Heroku, and other tools. For Windows users, WSL1 or WSL2 is recommended. The repository does the following:
- Ingests data, trains a decision tree on that data, and makes predictions.
- It does this in batch mode and also in real time via FastAPI and Heroku.
- Continuous Integration and Continuous Deployment (CI/CD) is enabled via workflows in GitHub and hooks into Heroku from GitHub.
- You can find the repository [here](https://github.com/LindsayMoir/FastAPI_Heroku_CICD).

## Environment Setup

### Install Conda
If you don’t already have it, download and install [Conda here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

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
│   ├── feature_importance.png
│   ├── label_encoder.pkl
│   ├── roc_curve.png
│   └── slice_output.txt

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
│   ├── fast_api_post.png
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
└── static/
    ├── favicon.ico
└── tests/
    ├── main_tests/
    └── model_tests/
```

13 directories, 84 files

## GitHub Actions

GitHub Actions have been implemented. `flake8` and `pytest` are automatically run every time there is a git push to the `main` branch.

### Pytest

There are over 10 tests 
```bash
pytest -v
```
that are ran automatically every time there is a:
```bash
git push origin main
```

### Flake8

```bash
flake 8
```
is also ran every time there is a 
```bash
git push origin main
```

## Model

The `model.py` script reads, cleans data, trains the model, and performs inference (batch and single prediction). This is the classifiation report. We are using f1 to choose our best_mdodel.

```bash

                precision    recall  f1-score   support

           0       0.93      0.90      0.91      4940
           1       0.70      0.78      0.74      1568

    accuracy                           0.87      6508
   macro avg       0.82      0.84      0.83      6508
weighted avg       0.87      0.87      0.87      6508
```

### Model.py Features

- Collects run statistics for every run.
- Uses a config file for all constants.
- Uses Gradient Boosted Classifier (GBC) ensemble decision tree from sklearn.
- Utilizes `best_estimator` (best_model) from the pipeline class for inference. Since this is part of the pipeline, feature engineering is handled automatically. The assumption is that any prediction requests will be cleaned by the web service.
- Calculates and visualizes
![Feature Importance](/ml/feature_importance.png)
- Produces visualizations (confusion matrix, ROC curve).
![AUC ROC Curve](/ml/roc_curve.png)
- Evaluates model slices by analyzing each categorical feature and its unique values, calculating all metrics to assess the impact.

### To run `model.py` locally:

1. Clone the repository.
2. Navigate to the root directory.
3. Provide a working directory as an argument:

```bash
python model.py [your working directory]
```
The working directory that you want to use is the path to where model.py is executing on the target machine. This is not necessarily in the root directory where the folder FastAPI Heroku CICD lives. Depending on how your machine is setup and especially in the GitHub case, it will be executing a in a different directory than where you are positioned in the folder hierarchy with terminal.

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

#### Deploying to Heroku
This application is deployed using Heroku's automatic deployment feature. Heroku updates the app automatically every time a new commit is pushed to GitHub.

##### Login to Heroku
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

## Screenshots

- Continuous Deployment:  
  ![Continuous Deployment](/screenshots/continuous_deployment.png)

- Continuous Integration:  
  ![Continuous Integration](/screenshots/continuous_integration.png)

- Example FastAPI POST Request:  
  ![FastAPI POST Example](/screenshots/fast_api_post.png)

There are also many more!

## Summary

This repository demonstrates how to deploy a machine learning model using FastAPI and Heroku, with continuous integration and deployment (CI/CD) via GitHub Actions. It includes detailed setup instructions, visualizations of model performance, and scripts for training, inference, and web service interaction. Key features include automated data ingestion, real-time predictions, and evaluation of model performance on specific data slices. This README provides a comprehensive guide to setting up the environment, running the model, and deploying the web app on Heroku.