import logging

from decouple import config
from fastapi import FastAPI
from fastapi_utils.timing import add_timing_middleware
from starlette.middleware.cors import CORSMiddleware

from routes import load_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=f"Wildfire and Landscape Resilience Data API ({config('WFR_VERSION')})",
    description="The REST API for Wildfire and Landscape Resilience Data Hub.",
    version=f"{config('WFR_VERSION')}",
    docs_url=config('WFR_DOCS_URL'),
    openapi_url=f"{config('WFR_BASE_PATH')}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

add_timing_middleware(app, record=logger.info, prefix="app", exclude="untimed")

app.include_router(load_routes())

