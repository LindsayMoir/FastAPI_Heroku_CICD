# Put the code for your API here.
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
import uvicorn

app = FastAPI()

# Serve the favicon.ico file from a 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    message = "Deploying a ML Model with GBC, FastAPI, Heroku, and DVC"
    return {"message": message}

# Provide the favicon route


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
