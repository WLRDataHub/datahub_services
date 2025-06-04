from typing import Optional, List

from fastapi import APIRouter, Query, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from controller.db import get_db
from controller.endpoint_collections import EndpointCollection, EditableSubfield, ManyToManySubfield
import models.wfr_database as wfr_database
import models.wfr_pydantic as wfr_pydantic
from models.wfr_database import DatasetCollection, Dataset, DatasetCollectionDataset, DatasetMetadata

router = APIRouter(tags=["Dataset"], prefix='/Dataset')


@router.get('/Dataset_Collections/counts', name='Get the data collection counts.')
async def get_data_collection_counts(
        search_terms: Optional[str] = Query(None, description="Search query term on the collection."),
        # filter_name: Optional[str] = Query(None),
        # filter_value: Optional[List[int]] = Query(None),
        db: Session = Depends(get_db)):
    if search_terms:
        where = DatasetMetadata.__ts_vector__.match(search_terms) | Dataset.__ts_vector__.match(search_terms)
    else:
        where = True
    subquery = (
        db.query(Dataset.dataset_id)
            .distinct(Dataset.dataset_id)
            .join(DatasetMetadata)
            .filter(where)
    )
    query = (select(DatasetCollection.name.label('data_collection_name'),
                    DatasetCollection.data_collection_id.label('data_collection_id'),
                    func.count(Dataset.dataset_id).label('dataset_count')
                    ).select_from(DatasetCollection)
             .join(DatasetCollectionDataset)
             .join(Dataset)
             .filter(Dataset.dataset_id.in_(subquery))
             # .filter(DatasetCollection.data_collection_id.in_(filter_value))
             .group_by(DatasetCollection.name, DatasetCollection.data_collection_id)
             .order_by(DatasetCollection.name))
    results = db.execute(query)
    return [{
        'data_collection_name': result.data_collection_name,
        'data_collection_id': result.data_collection_id,
        'count': result.dataset_count
    } for result in results]


EndpointCollection(
    name="Dataset",
    id_key="dataset_id",
    database_model=wfr_database.Dataset,
    pydantic_model=wfr_pydantic.Dataset,
    pydantic_model_update=wfr_pydantic.DatasetUpdate,
    pydantic_model_create=wfr_pydantic.DatasetBase,
    pydantic_model_slim=wfr_pydantic.Dataset,
    searchable=True,
    search_join=wfr_database.DatasetMetadata,

    filter_model=wfr_database.DatasetCollectionDataset,
    filter_field=wfr_database.DatasetCollectionDataset.data_collection_id

).with_subfields(
    [
        EditableSubfield(
            name="DatasetMetadata",
            id_key="dataset_metadata_id",
            database_model=wfr_database.DatasetMetadata,
            pydantic_model=wfr_pydantic.DatasetMetadata,
            pydantic_model_update=wfr_pydantic.DatasetMetadataUpdate,
            pydantic_model_create=wfr_pydantic.DatasetMetadataBase,
        ),
        EditableSubfield(
            name="GISService",
            id_key="service_id",
            database_model=wfr_database.GISService,
            pydantic_model=wfr_pydantic.GISService,
            pydantic_model_update=wfr_pydantic.GISServiceUpdate,
            pydantic_model_create=wfr_pydantic.GISServiceBase,
        ),
        ManyToManySubfield(
            name="TaxonomyItem",
            id_key="taxonomy_item_id",
            database_model=wfr_database.TaxonomyItem,
            pydantic_model=wfr_pydantic.TaxonomyItem,
            pydantic_model_update=wfr_pydantic.TaxonomyItemUpdate,
            pydantic_model_create=wfr_pydantic.TaxonomyItemBase,
            join_table=wfr_database.DatasetTaxonomyItem,
        )
    ]
).register_router(
    router
)
