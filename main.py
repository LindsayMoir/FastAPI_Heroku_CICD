import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from typing import Dict
import uvicorn
import model  # Assuming 'model.py' contains your inference logic

# Create the 'logs' directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler that logs info messages to 'logs/app.log'
file_handler = RotatingFileHandler('logs/app.log',
                                   maxBytes=10**6, backupCount=5)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))

# Add the file handler to the logger
logger.addHandler(file_handler)

app = FastAPI()

# Serve the favicon.ico file from a 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")


class Features(BaseModel):
    age: int
    workclass: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int


@app.get("/")
def read_root() -> dict:
    message = "Deploying a ML Model with GBC, FastAPI, Heroku, and DVC"
    logger.info(f"Root accessed. Message: {message}")
    return {"message": message}


# Provide the favicon route
@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> RedirectResponse:
    return RedirectResponse(url="/static/favicon.ico")


@app.post("/predict")
def predict(features: Features) -> Dict[str, str]:
    """
    Predict the income category based on the provided features.

    Args:
        features (Features): The input data model containing the values.

    Returns:
        Dict[str, str]: A dictionary with the prediction result.
    """
    try:
        # Perform inference using the model.inference function
        prediction = model.inference(features.model_dump())
        logger.info(f"Features used for prediction: {features}")
        logger.info(f"Prediction made successfully: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500,
                            detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
