from fastapi import FastAPI
# from fastapi.responses import HtmlResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import views.query_views as query_views
import views.workspace as workspace
import views.feedback_view as feedback
import routes.auth as auth
import views.download as download 
from dotenv.main import load_dotenv
import os

from routes import load_routes

app = FastAPI(
    title=f"Wildfire and Landscape Resilience Data Hub Staging API",
    description="The staging API for Wildfire and Landscape Resilience Data Hub.",
    docs_url='/docs',
    openapi_url=f"/v1/openapi.json",
    root_path="/staging-api")

load_dotenv('./fastApi/.env')

base_url = os.environ['backend_full_url']

origins = [
    "*",
    "http://localhost:3000",
    "localhost:3000",
    "twsa.ucsd.edu",
    "sparcal.sdsc.edu",
    "hubbub.sdsc.edu",	
    "rrkmetric.sdsc.edu",
    base_url
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(download.router)
app.include_router(query_views.router)
app.include_router(workspace.router)
app.include_router(feedback.router)
app.include_router(auth.router)

app.include_router(load_routes())


@app.get('/', tags=["root"])
def home():
    return {
        'message': 'the home page'
    }

if __name__ == '__main__':
    url = os.environ['root_url']
    port=int(os.environ['backend_port'])
    uvicorn.run(app, host=url, port=port)