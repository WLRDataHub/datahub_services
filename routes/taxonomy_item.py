from fastapi import APIRouter, Depends

from controller.db import use_unencrypted_session
from controller.endpoint_collections import EndpointCollection
import models.wfr_database as wfr_database
import models.wfr_pydantic as wfr_pydantic
from sqlalchemy.orm import Session

from models.wfr_database import TaxonomyItem

router = APIRouter(tags=["Taxonomy"], prefix='/TaxonomyItem')

EndpointCollection(
    name="TaxonomyItem",
    id_key="taxonomy_item_id",
    database_model=wfr_database.TaxonomyItem,
    pydantic_model=wfr_pydantic.TaxonomyItem,
    pydantic_model_update=wfr_pydantic.TaxonomyItemUpdate,
    pydantic_model_create=wfr_pydantic.TaxonomyItemBase,
).register_router(
    router
)


@router.get('/{taxonomy_item_id}/path')
async def get_taxonomy_item_path(taxonomy_item_id: int,
                                 db: Session = Depends(use_unencrypted_session)):
    path = []
    taxonomy_item = db.query(TaxonomyItem).filter_by(taxonomy_item_id=taxonomy_item_id).first()
    while taxonomy_item.parent_taxonomy_item:
        path.insert(0, taxonomy_item.taxonomy_item_name)
        taxonomy_item = taxonomy_item.parent_taxonomy_item
    path.insert(0, taxonomy_item.taxonomy_item_name)
    return path
