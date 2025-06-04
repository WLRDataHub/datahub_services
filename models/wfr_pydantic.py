from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel

updatable_forward_refs = []


def deferred_ref_update(cls):
    updatable_forward_refs.append(cls)
    return cls


class Base(BaseModel):
    class Config:
        orm_mode = True


# --------------------------
# for creating a Dataset
class DatasetBase(Base):
    name: Optional[str]
    description: Optional[str]
    data_type: str
    file_path: str
    file_type: str
    url: Optional[str]
    report_title_template: Optional[str]
    report_description_template: Optional[str]
    last_update: datetime


# for updating a Dataset
class DatasetUpdate(DatasetBase):
    dataset_id: Optional[int]
    data_type: Optional[str]
    file_path: Optional[str]
    file_type: Optional[str]
    last_update: Optional[datetime]


# for getting a Dataset
@deferred_ref_update
class Dataset(DatasetBase):
    dataset_id: int

    dataset_metadata: List['DatasetMetadata']
    gis_services: List['GISService']
    taxonomy_items: List['TaxonomyItem']
    dataset_collections: List['DatasetCollectionSlim']


@deferred_ref_update
class DatasetSlim(DatasetBase):
    dataset_id: int

    taxonomy_items: List['TaxonomyItem']


# --------------------------
# for creating a DatasetMetadata
class DatasetMetadataBase(Base):
    dataset_id: int
    name: str
    text_value: Optional[str]
    float_value: Optional[float]
    int_value: Optional[int]
    bool_value: Optional[bool]


# for updating a DatasetMetadata
class DatasetMetadataUpdate(DatasetMetadataBase):
    dataset_metadata_id: Optional[int]
    dataset_id: Optional[int]
    name: Optional[str]


# for getting a DatasetMetadata
class DatasetMetadata(DatasetMetadataBase):
    dataset_metadata_id: int


# --------------------------
# for creating a ColorMap
class ColorMapBase(Base):
    name: str


class ColorMapUpdate(ColorMapBase):
    color_map_id: Optional[int]
    name: Optional[str]


@deferred_ref_update
class ColorMap(ColorMapBase):
    color_map_id: int

    color_map_entries: List['ColorMapEntry']


# --------------------------
# for creating a GisService
class GISServiceBase(Base):
    dataset_id: int
    name: Optional[str]
    description: Optional[str]
    service_type: str
    service_url: str
    layer_name: str
    layer_type: str
    color_map_id: Optional[int]


# for updating a GisService
class GISServiceUpdate(GISServiceBase):
    dataset_id: Optional[int]
    service_id: Optional[int]
    service_type: Optional[str]
    service_url: Optional[str]
    layer_name: Optional[str]
    layer_type: Optional[str]


# for getting a GisService
class GISService(GISServiceBase):
    service_id: int

    color_map: Optional[ColorMap]


# --------------------------
# for creating a ColorMapEntry
class ColorMapEntryBase(Base):
    color_map_id: int
    text_value: Optional[str]
    float_value: Optional[float]
    int_value: Optional[int]
    color: str


# for updating a ColorMapEntry
class ColorMapEntryUpdate(ColorMapEntryBase):
    color_map_entry_id: Optional[int]
    color_map_id: Optional[int]
    color: Optional[str]


# for getting a ColorMapEntry
class ColorMapEntry(ColorMapEntryBase):
    color_map_entry_id: int


# ----------------------------
# for creating a Taxonomy
class TaxonomyBase(Base):
    taxonomy_name: str


# for updating a Taxonomy
class TaxonomyUpdate(TaxonomyBase):
    taxonomy_id: Optional[int]


# for getting a Taxonomy
class Taxonomy(TaxonomyBase):
    taxonomy_id: int


# ----------------------------
# for creating a TaxonomyItem
class TaxonomyItemBase(Base):
    taxonomy_id: int
    taxonomy_item_name: str
    parent_taxonomy_item_id: Optional[int]


# for updating a TaxonomyItem
class TaxonomyItemUpdate(TaxonomyItemBase):
    taxonomy_item_id: Optional[int]
    taxonomy_id: Optional[int]
    taxonomy_item_name: Optional[str]


# for getting a TaxonomyItem
class TaxonomyItem(TaxonomyItemBase):
    taxonomy_item_id: int

    taxonomy: Taxonomy


# ----------------------------
# for creating a DatasetTaxonomyItem
class DatasetTaxonomyItemBase(Base):
    dataset_id: int
    taxonomy_item_id: int


# for updating a DatasetTaxonomyItem
class DatasetTaxonomyItemUpdate(DatasetTaxonomyItemBase):
    dataset_taxonomy_item_id: Optional[int]
    dataset_id: Optional[int]
    taxonomy_item_id: Optional[int]


# for getting a DatasetTaxonomyItem
class DatasetTaxonomyItem(DatasetTaxonomyItemBase):
    dataset_taxonomy_item_id: int

    taxonomy_item: TaxonomyItem


# ----------------------------
# for creating a DatasetCollection
class DatasetCollectionBase(Base):
    name: str
    description: Optional[str]
    parent_dataset_collection_id: Optional[int]


# for updating a DatasetCollection
class DatasetCollectionUpdate(DatasetCollectionBase):
    data_collection_id: Optional[int]
    name: Optional[str]


# for getting dataset collection slim information only
class DatasetCollectionSlim(DatasetCollectionBase):
    data_collection_id: int


# for getting dataset collection
class DatasetCollection(DatasetCollectionBase):
    data_collection_id: int

    # datasets: List[DatasetSlim]
    children: Optional[List[DatasetCollectionSlim]]



# ----------------------------
# for creating a User
class UserBase(Base):
    username: str
    # password_hash: str
    first_name: str
    last_name: str
    email: str
    is_verified: bool = False
    affiliation: str


class Account(BaseModel):
    username: str
    password: str
    firstname: str
    lastname: str
    agency: str
    email: str


class UserUpdate(UserBase):
    user_id: Optional[int]
    username: Optional[str]
    password_hash: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    email: Optional[str]
    is_verified: Optional[bool]
    affiliation: Optional[str]


@deferred_ref_update
class User(UserBase):
    user_id: int

    roles: List['Role']


# ----------------------------
# for creating a Role
class RoleBase(Base):
    name: str


class RoleUpdate(RoleBase):
    role_id: Optional[int]
    name: Optional[str]


class Role(RoleBase):
    role_id: int


# ----------------------------
# for creating a Workspace
class WorkspaceBase(Base):
    user_id: int
    workspace_name: str
    workspace_info: Dict


class WorkspaceUpdate(WorkspaceBase):
    user_id: Optional[int]
    workspace_name: Optional[str]
    workspace_info: Optional[Dict]
    last_used_date: Optional[datetime]


class Workspace(WorkspaceBase):
    workspace_id: int
    created_on: datetime
    last_used_date: Optional[datetime]

    user: User



# ----------------------------
# for creating a Dictionary Section
class DictionarySectionBase(Base):
    name: str


class DictionarySectionUpdate(DictionarySectionBase):
    dictionary_section_id: Optional[int]
    name: Optional[str]


@deferred_ref_update
class DictionarySection(DictionarySectionBase):
    dictionary_section_id: int

    # dictionary_items: List['DictionaryItem']


# ----------------------------
# for creating a Dictionary Item
class DictionaryItemBase(Base):
    dictionary_section_id: int
    name: str
    value: str


class DictionaryItemUpdate(DictionaryItemBase):
    dictionary_item_id: Optional[int]
    dictionary_section_id: Optional[int]
    name: Optional[str]
    value: Optional[str]


class DictionaryItem(DictionaryItemBase):
    dictionary_item_id: int



for model in updatable_forward_refs:
    model.update_forward_refs()
