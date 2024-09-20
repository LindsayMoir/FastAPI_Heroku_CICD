# FastAPI Heroku CICD

## Overview

This repository provides an example of deploying a FastAPI machine learning model using Heroku and continuous integration with GitHub Actions. A command-line environment is recommended for ease of use with Git, DVC, and other tools. For Windows users, WSL1 or WSL2 is recommended.

You can find the repository [here](https://github.com/LindsayMoir/FastAPI_Heroku_CICD).

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
   conda install git```

## Files

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
│   └── config.yaml
├── coverage.xml
├── data/
│   ├── census.csv
│   ├── cleaned_data.csv
│   └── describe_cleaned_data.csv
├── directory_structure.txt
├── dvc_on_heroku_instructions.md
├── htmlcov/
│   ├── index.html
│   └── style_cb_8e611ae1.css
├── logs/
│   ├── app.log
│   └── ml.log
├── main.py
├── ml/
│   ├── best_model.pkl
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── model.py
├── model_card.md
├── requirements.txt
├── runtime.txt
├── sanitycheck.py
├── screenshots/
│   ├── Heroku Income Projection Metrics.png
│   └── Heroku Overview of App Deployment.png
├── setup.py
├── single_pred.py
└── tests/
    ├── main_tests/
    └── model_tests/

13 directories, 84 files

## GitHub Actions
Workflow has been implemented on this GitHub repository. flake8 and pytest are ran 
automatically every time there is git push origin main.

## Data
Is on DVC TBD

## Model
This is the python code that reads the data, cleans the data, trains the model, and does 
inference (both batch and single prediction).
- To run model.py on your local environment, clone the above respository.
- Get in the root directory
You will need to provide a working directory to this python command. This program accepts
an argument which is the working directory. 
- in terminal: python model.py [your working directory where root is]

## FastAPI

### Local Web Server

#### Start Web Server
Get in the directory where main.py is. It is under the root directory (FastAPI_Heroku_CICD). Then run this command
uvicorn main:app --reload
This starts the web server

#### Run Inference (Post) Using Curl
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
##### Expected Prediction
{"prediction":"<=50K"}

#### Run Inference (Post) Using requests
With the web server still running as above (Start Web Server)
Get in the root directory FastAPI_Heroku_CICD/
Run this command:
python single_pred.py

##### Expected Prediction
{"prediction":">50K"}

### Heroku Web Server
We have chosen automatic deployment on Heroku for our app. We are connected to GitHub
to do that. This means every time we push to GitHub, Heroku updates the app that is
running. Do all of this from terminal

##### Login
heroku login

##### Lists all apps that you have on Heroku
heroku apps

##### Supply resources to the app
heroku ps:scale web=1 --app income-projection

##### Get the status of the app
heroku ps --app income-projection

##### Open the app in a browser window
heroku open --app income-projection

##### Make sure you are in root directory
This will prove that the app is working and running on Heroku. Run single_pred.py
python single_pred.py

##### Scale the app down so it is using 0 resources
heroku ps:scale web=0 --app income-projection
