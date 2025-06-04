from fastapi import APIRouter

from controller.endpoint_collections import EndpointCollection, EditableSubfield
import models.wfr_database as wfr_database
import models.wfr_pydantic as wfr_pydantic


router = APIRouter(tags=["Taxonomy"], prefix='/Taxonomy')

EndpointCollection(
    name="Taxonomy",
    id_key="taxonomy_id",
    database_model=wfr_database.Taxonomy,
    pydantic_model=wfr_pydantic.Taxonomy,
    pydantic_model_update=wfr_pydantic.TaxonomyUpdate,
    pydantic_model_create=wfr_pydantic.TaxonomyBase,
    pydantic_model_slim=wfr_pydantic.Taxonomy,
    # searchable_fields=[],
).with_subfields(
    [
        EditableSubfield(
            name="TaxonomyItem",
            id_key="taxonomy_item_id",
            database_model=wfr_database.TaxonomyItem,
            pydantic_model=wfr_pydantic.TaxonomyItem,
            pydantic_model_update=wfr_pydantic.TaxonomyItemUpdate,
            pydantic_model_create=wfr_pydantic.TaxonomyItemBase,
        ),
    ]
).register_router(
    router
)
