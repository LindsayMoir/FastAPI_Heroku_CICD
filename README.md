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


## Files
./
(0tqq (BCODEOWNERS*
(0tqq (BEDA.ipynb*
(0tqq (BLICENSE.txt*
(0tqq (BProcfile*
(0tqq (BREADME.md*
(0tqq (B__pycache__/
(0x   tqq (Bmain.cpython-311.pyc*
(0x   tqq (Bmodel.cpython-311.pyc*
(0x   mqq (Btest_fastapi_inference.cpython-311-pytest-7.4.4.pyc*
(0tqq (Bconfig/
(0x   mqq (Bconfig.yaml*
(0tqq (Bcoverage.xml*
(0tqq (Bdata/
(0x   tqq (Bcensus.csv*
(0x   tqq (Bcleaned_data.csv*
(0x   tqq (Bdescribe_cleaned_data.csv*
(0x   mqq (Bresampled_data.csv*
(0tqq (Bdirectory_structure.txt*
(0tqq (Bdvc_on_heroku_instructions.md*
(0tqq (Bhtmlcov/
(0x   tqq (Bclass_index.html*
(0x   tqq (Bcoverage_html_cb_6fb7b396.js*
(0x   tqq (Bfavicon_32_cb_58284776.png*
(0x   tqq (Bfunction_index.html*
(0x   tqq (Bindex.html*
(0x   tqq (Bkeybd_closed_cb_ce680311.png*
(0x   tqq (Bmain_py.html*
(0x   tqq (Bmodel_py.html*
(0x   tqq (Bstatus.json*
(0x   tqq (Bstyle_cb_8e611ae1.css*
(0x   tqq (Bz_52e6229cf3831f7b_conftest_py.html*
(0x   tqq (Bz_52e6229cf3831f7b_test_clean_data_py.html*
(0x   tqq (Bz_52e6229cf3831f7b_test_load_data_py.html*
(0x   tqq (Bz_52e6229cf3831f7b_test_quality_metrics_py.html*
(0x   tqq (Bz_f2852576403184d2_test_favicon_py.html*
(0x   mqq (Bz_f2852576403184d2_test_main_py.html*
(0tqq (Blogs/
(0x   tqq (Bapp.log*
(0x   tqq (Bml.log*
(0x   mqq (Brun_statistics.csv*
(0tqq (Bmain.py*
(0tqq (Bml/
(0x   tqq (Bbest_model.pkl*
(0x   tqq (Bconfusion_matrix.png*
(0x   tqq (Bfeature_importance.png*
(0x   tqq (Blabel_encoder.pkl*
(0x   tqq (Broc_curve.png*
(0x   mqq (Bslice_output.txt*
(0tqq (Bmodel.py*
(0tqq (Bmodel_card.md*
(0tqq (Brequirements.txt*
(0tqq (Bruntime.txt*
(0tqq (Bsanitycheck.py*
(0tqq (Bscreenshots/
(0x   tqq (BHeroku Income Projection Metrics.png*
(0x   tqq (BHeroku Login Apps Scale Status Pred Heroku Pred Local.png*
(0x   tqq (BHeroku Login.png*
(0x   tqq (BHeroku Logs.png*
(0x   tqq (BHeroku Overview of App Deployment.png*
(0x   tqq (BHeroku Settings.png*
(0x   tqq (BHeroku Web App Deployed single_pred.py Run Successfully Against https income-projection-61635563fc60.herokuapp com.png*
(0x   tqq (BHeroku_continuous_deployment.png*
(0x   tqq (BPush GitHub Auto Deploy Heroku.png*
(0x   tqq (Bcontinuous_deployment.png*
(0x   tqq (Bcontinuous_integration.png*
(0x   tqq (Bexample.png*
(0x   tqq (Bfast_api_post.png*
(0x   tqq (Blive_get.png*
(0x   mqq (Bsanitycheck_run.png*
(0tqq (Bsetup.py*
(0tqq (Bsingle_pred.py*
(0tqq (Bstatic/
(0x   mqq (Bfavicon.ico*
(0mqq (Btests/
(0    tqq (Bmain_tests/
(0    x   tqq (B__pycache__/
(0    x   x   tqq (Btest_favicon.cpython-311-pytest-7.4.4.pyc*
(0    x   x   tqq (Btest_main.cpython-311-pytest-7.4.4.pyc*
(0    x   x   tqq (Btest_main.cpython-311.pyc*
(0    x   x   tqq (Btest_post_inference.cpython-311-pytest-7.4.4.pyc*
(0    x   x   tqq (Btest_predict.cpython-311-pytest-7.4.4.pyc*
(0    x   x   mqq (Btest_read_root.cpython-311-pytest-7.4.4.pyc*
(0    x   tqq (Btest_favicon.py*
(0    x   mqq (Btest_main.py*
(0    mqq (Bmodel_tests/
(0        tqq (B__pycache__/
(0        x   tqq (Bconftest.cpython-311-pytest-7.4.4.pyc*
(0        x   tqq (Bconftest.cpython-311-pytest-8.2.2.pyc*
(0        x   tqq (Btest_clean_data.cpython-311-pytest-7.4.4.pyc*
(0        x   tqq (Btest_load_data.cpython-311-pytest-7.4.4.pyc*
(0        x   tqq (Btest_quality_metrics.cpython-311-pytest-7.4.4.pyc*
(0        x   mqq (Btest_setup_env_returns_dict.cpython-311-pytest-7.4.4.pyc*
(0        tqq (Bconftest.py*
(0        tqq (Btest_clean_data.py*
(0        tqq (Btest_load_data.py*
(0        tqq (Btest_quality_metrics.py*
(0        mqq (Btest_setup_env_returns_dict.py*

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
