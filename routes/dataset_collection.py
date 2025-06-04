import json
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import text, asc, desc
from sqlalchemy.orm import Session

from controller.db import get_db
from controller.db import use_unencrypted_session
from controller.endpoint_collections import EndpointCollection, ManyToManySubfield
import models.wfr_database as wfr_database
import models.wfr_pydantic as wfr_pydantic
from models.wfr_database import TaxonomyItem


router = APIRouter(tags=["Dataset Collection"], prefix='/DatasetCollection')


@router.get('/hierarchy', response_model=List[wfr_pydantic.DatasetCollection])
async def get_data_collection_hierarchy(
        skip: int = 0,
        limit: int = 100,
        ascending: bool = True,
        db: Session = Depends(get_db)):
    _ordering = asc(wfr_database.DatasetCollection.name) if ascending else desc(wfr_database.DatasetCollection.name)
    return (
        db.query(wfr_database.DatasetCollection)
            .filter(wfr_database.DatasetCollection.parent_dataset_collection_id.is_(None))
            .order_by(_ordering)
            .offset(skip)
            .limit(limit)
            .all()
    )


@router.get('/hierarchy/count')
async def get_data_collection_hierarchy_count(db: Session = Depends(get_db)):
    return (
        db.query(wfr_database.DatasetCollection)
            .filter(wfr_database.DatasetCollection.parent_dataset_collection_id.is_(None))
            .count()
    )


EndpointCollection(
    name="DatasetCollection",
    id_key="data_collection_id",
    database_model=wfr_database.DatasetCollection,
    pydantic_model=wfr_pydantic.DatasetCollection,
    pydantic_model_update=wfr_pydantic.DatasetCollectionUpdate,
    pydantic_model_create=wfr_pydantic.DatasetCollectionBase,
    pydantic_model_slim=wfr_pydantic.DatasetCollection,
    searchable=True,
).with_subfields(
    [
        ManyToManySubfield(
            name="Dataset",
            id_key="dataset_id",
            database_model=wfr_database.Dataset,
            pydantic_model=wfr_pydantic.Dataset,
            pydantic_model_update=wfr_pydantic.DatasetUpdate,
            pydantic_model_create=wfr_pydantic.DatasetBase,
            join_table=wfr_database.DatasetCollectionDataset,
        )
    ]
).register_router(
    router
)


def get_hierarchy(hierarchy, taxonomy_items):
    if taxonomy_items:
        if hierarchy:
            return hierarchy
        else:
            for taxa in taxonomy_items:
                if taxa.parent_taxonomy_item_id is None:
                    taxa_dict = taxa.__dict__
                    taxa_dict['children'] = []
                    hierarchy.append(taxa_dict)
    else:
        return hierarchy


def build_hierarchy(elements):
    # Create a dictionary to store parent-child relationships
    parent_child_map = {}
    root_elements = []
    taxonomy_item_id_name_map = {}

    for element in elements:
        element_id = element['taxonomy_item_id']
        parent_id = element['parent_taxonomy_item_id']
        taxonomy_item_name = element['taxonomy_item_name']
        taxonomy_item_id_name_map[element_id] = taxonomy_item_name
        if parent_id is None:
            root_elements.append(element_id)
        else:
            parent_child_map.setdefault(parent_id, []).append(element_id)

    # Recursive function to build hierarchy
    def build_subtree(element_id):
        children = parent_child_map.get(element_id, [])
        subtree = {
            'taxonomy_item_id': element_id,
            'taxonomy_item_name': taxonomy_item_id_name_map[element_id],
            'children': []
        }
        for child_id in children:
            child = build_subtree(child_id)
            subtree['children'].append(child)
        return subtree

    hierarchy = []
    for root_id in root_elements:
        hierarchy.append(build_subtree(root_id))

    return hierarchy


def add_datasets_to_hierarchy(hierarchy, datasets):
    def add_dataset_to_hierarchy(hierarchy, dataset):
        for node in hierarchy:
            if type(node) is dict and 'taxonomy_item_id' in node.keys():
                if node['taxonomy_item_id'] == dataset.taxonomy_item_id:
                    node['children'].append({
                        'dataset_id': dataset.dataset_id,
                        'dataset_name': dataset.name
                    })
                elif node['children']:
                    add_dataset_to_hierarchy(node['children'], dataset)

    for dataset in datasets:
        add_dataset_to_hierarchy(hierarchy, dataset)


def normalize_node(node, prefix):
    if type(node) is dict:
        if 'taxonomy_item_name' in node.keys():
            node['label'] = node['taxonomy_item_name']
            node['key'] = f'{prefix}/{node["taxonomy_item_name"]}'
            # del node['taxonomy_item_name']
            # del node['taxonomy_item_id']
        else:
            node['label'] = node['dataset_name']
            node['key'] = f'{node["dataset_id"]}'
            # node['key'] = f'{prefix}/{node["dataset_name"]}'
        if "children" in node.keys():
            for child in node["children"]:
                normalize_node(child, node['key'])


def normalize(nodelist):
    for node in nodelist:
        normalize_node(node, '')


@router.get('/{data_collection_id}/taxonomy/{taxonomy_id}/hierarchy')
async def get_data_collection_hierarchy(data_collection_id: int,
                                        taxonomy_id: int,
                                        db: Session = Depends(use_unencrypted_session)):
    query = f"""
        SELECT dataset.name, 
               dataset.dataset_id,
               taxonomy_item.taxonomy_item_name,
               taxonomy_item.taxonomy_item_id,
               taxonomy_item.parent_taxonomy_item_id
          FROM hub_catalog.dataset_collection,
               hub_catalog.dataset,
               hub_catalog.dataset_collection_dataset,
               hub_catalog.dataset_taxonomy_item,
               hub_catalog.taxonomy_item
         WHERE dataset_collection.data_collection_id={data_collection_id}
           AND dataset_collection_dataset.dataset_id = dataset.dataset_id
           AND dataset_collection_dataset.data_collection_id = dataset_collection.data_collection_id
           AND dataset_taxonomy_item.dataset_id = dataset.dataset_id
           AND dataset_taxonomy_item.taxonomy_item_id = taxonomy_item.taxonomy_item_id
           AND taxonomy_item.taxonomy_id = {taxonomy_id}
    """
    statement = text(query)
    datasets = db.execute(statement).fetchall()

    if datasets:
        taxonomy_items = db.query(TaxonomyItem).filter_by(taxonomy_id=taxonomy_id).all()
        taxonomy_item_dicts = [x.__dict__ for x in taxonomy_items]
        hierarchy = build_hierarchy(taxonomy_item_dicts)

        add_datasets_to_hierarchy(hierarchy, datasets)
        normalize(hierarchy)
        return hierarchy
    else:
        return []


@router.get('/{data_collection_id}/taxonomy')
async def get_data_collection_taxonomies(data_collection_id: int,
                                         db: Session = Depends(use_unencrypted_session)):
    query = f"""
        SELECT DISTINCT 
               taxonomy.taxonomy_id,
               taxonomy.taxonomy_name
          FROM hub_catalog.dataset_collection,
               hub_catalog.dataset,
               hub_catalog.dataset_collection_dataset,
               hub_catalog.dataset_taxonomy_item,
               hub_catalog.taxonomy_item,
               hub_catalog.taxonomy
         WHERE dataset_collection.data_collection_id={data_collection_id}
           AND dataset_collection_dataset.dataset_id = dataset.dataset_id
           AND dataset_collection_dataset.data_collection_id = dataset_collection.data_collection_id
           AND dataset_taxonomy_item.dataset_id = dataset.dataset_id
           AND dataset_taxonomy_item.taxonomy_item_id = taxonomy_item.taxonomy_item_id
           AND taxonomy_item.taxonomy_id = taxonomy.taxonomy_id
    """
    statement = text(query)
    taxonomies = db.execute(statement).fetchall()

    # Convert each Row object into a dictionary
    taxonomies_dict = []
    for row in taxonomies:
        taxonomies_dict.append({'taxonomy_id': row[0], 'taxonomy_name': row[1]})
    return taxonomies_dict
