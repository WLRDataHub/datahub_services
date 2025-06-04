from fastapi import APIRouter

from controller.endpoint_collections import EndpointCollection, EditableSubfield
import models.wfr_database as wfr_database

import models.wfr_pydantic as wfr_pydantic


router = APIRouter(tags=["Legend"], prefix='/ColorMap')

EndpointCollection(
    name="ColorMap",
    id_key="color_map_id",
    database_model=wfr_database.ColorMap,
    pydantic_model=wfr_pydantic.ColorMap,
    pydantic_model_update=wfr_pydantic.ColorMapUpdate,
    pydantic_model_create=wfr_pydantic.ColorMapBase,
).with_subfields(
    [
        EditableSubfield(
            name="ColorMapEntry",
            id_key="color_map_entry_id",
            database_model=wfr_database.ColorMapEntry,
            pydantic_model=wfr_pydantic.ColorMapEntry,
            pydantic_model_update=wfr_pydantic.ColorMapEntryUpdate,
            pydantic_model_create=wfr_pydantic.ColorMapEntryBase,
        ),
    ]
).register_router(
    router
)
