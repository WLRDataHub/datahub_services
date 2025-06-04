from fastapi import APIRouter

from controller.endpoint_collections import EndpointCollection, EditableSubfield
import models.wfr_database as wfr_database
import models.wfr_pydantic as wfr_pydantic

router = APIRouter(tags=["Dictionary Section"], prefix='/DictionarySection')

EndpointCollection(
    name="DictionarySection",
    id_key="dictionary_section_id",
    database_model=wfr_database.DictionarySection,
    pydantic_model=wfr_pydantic.DictionarySection,
    pydantic_model_update=wfr_pydantic.DictionarySectionUpdate,
    pydantic_model_create=wfr_pydantic.DictionarySectionBase,
    pydantic_model_slim=wfr_pydantic.DictionarySection,
    # searchable_fields=[],
).with_subfields(
    [
        EditableSubfield(
            name="DictionaryItem",
            id_key="dictionary_item_id",
            database_model=wfr_database.DictionaryItem,
            pydantic_model=wfr_pydantic.DictionaryItem,
            pydantic_model_update=wfr_pydantic.DictionaryItemUpdate,
            pydantic_model_create=wfr_pydantic.DictionaryItemBase,
        ),
    ]
).register_router(
    router
)
