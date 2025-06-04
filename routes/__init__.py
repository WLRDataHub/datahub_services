
from decouple import config
from fastapi import APIRouter
from tqdm import tqdm

from . import (
    auth,
    dataset,
    dataset_collection,
    taxonomy,
    color_map,
    user,
    wps,
    dictionary_section,
    ckan,
    wcs_clip,
    shapfile_to_geojson,
    user_activity_log,
    wms_thumbnail,
    wcs_clip,
    its_report,
    its_report_for_region, 
    its_report_for_region_2,
    its_report_for_region_3
)

# List of all routable FastAPI routers
routables = [
    r.router
    for r in [
        dataset,
        dataset_collection,
        taxonomy,
        color_map,
        user,
        wps,
        dictionary_section,
        ckan,
        wcs_clip,
        shapfile_to_geojson,
        user_activity_log,
        wms_thumbnail,
        wcs_clip,
        its_report,
        its_report_for_region, 
        its_report_for_region_2,
        its_report_for_region_3
    ]
]

# Create the root router with a configurable prefix
root_router = APIRouter(prefix=config('WFR_BASE_PATH'))

def load_routes():
    for r in tqdm(
        routables,
        unit="route",
        leave=False,
        colour="blue",
        desc="Loading routes...",
    ):
        root_router.include_router(r)
    return root_router


