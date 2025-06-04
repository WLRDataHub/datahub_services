from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator


class TSVector(TypeDecorator):
    impl = TSVECTOR


Base = declarative_base()
metadata = Base.metadata

schema = 'hub_catalog'


class Dataset(Base):
    __tablename__ = 'dataset'
    __table_args__ = {'schema': schema}

    dataset_id = Column(Integer, primary_key=True, server_default=text("nextval('dataset_dataset_id_seq'::regclass)"))
    name = Column(String(256), nullable=False)
    description = Column(Text)
    data_type = Column(String(32), nullable=False)
    file_path = Column(String(256))
    file_type = Column(String(32))
    url = Column(String(256))
    report_title_template = Column(String(256))
    report_description_template = Column(Text)
    last_update = Column(DateTime, server_default=text("now()"))
    __ts_vector__ = Column(TSVector(),
                           server_default=text("to_tsvector('english', (name || COALESCE(' ' || description, '')))"))

    dataset_metadata = relationship('DatasetMetadata', back_populates='dataset', lazy='joined')
    gis_services = relationship('GISService', back_populates='dataset', lazy='joined')
    taxonomy_items = relationship('TaxonomyItem', secondary='hub_catalog.dataset_taxonomy_item', lazy='joined')
    dataset_collections = relationship("DatasetCollection",
                                       secondary="hub_catalog.dataset_collection_dataset",
                                       back_populates="datasets",
                                       lazy='joined')


class DatasetCollection(Base):
    __tablename__ = 'dataset_collection'
    __table_args__ = {'schema': schema}

    data_collection_id = Column(Integer, primary_key=True,
                                server_default=text("nextval('dataset_collection_data_collection_id_seq'::regclass)"))
    name = Column(String(256))
    description = Column(Text)
    last_update = Column(DateTime, server_default=text("now()"))
    parent_dataset_collection_id = Column(ForeignKey('hub_catalog.dataset_collection.data_collection_id'))
    __ts_vector__ = Column(TSVector(),
                           server_default=text("to_tsvector('english', (name || COALESCE(' ' || description, '')))"))

    datasets = relationship("Dataset",
                            secondary="hub_catalog.dataset_collection_dataset",
                            back_populates="dataset_collections",
                            lazy='joined')
    children = relationship("DatasetCollection", lazy='joined', order_by="DatasetCollection.name")


class Taxonomy(Base):
    __tablename__ = 'taxonomy'
    __table_args__ = {'schema': schema}

    taxonomy_id = Column(Integer, primary_key=True,
                         server_default=text("nextval('taxonomy_taxonomy_id_seq'::regclass)"))
    taxonomy_name = Column(String(64))


class DatasetCollectionDataset(Base):
    __tablename__ = 'dataset_collection_dataset'
    __table_args__ = {'schema': schema}

    data_collection_dataset_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('dataset_collection_dataset_data_collection_dataset_id_seq'::regclass)"))
    data_collection_id = Column(ForeignKey('hub_catalog.dataset_collection.data_collection_id'), nullable=False)
    dataset_id = Column(ForeignKey('hub_catalog.dataset.dataset_id'), nullable=False)

    data_collection = relationship('DatasetCollection')
    dataset = relationship('Dataset')


class DatasetMetadata(Base):
    __tablename__ = 'dataset_metadata'
    __table_args__ = {'schema': schema}

    dataset_metadata_id = Column(Integer, primary_key=True,
                                 server_default=text("nextval('dataset_metadata_dataset_metadata_id_seq'::regclass)"))
    dataset_id = Column(ForeignKey('hub_catalog.dataset.dataset_id'))
    name = Column(String(128), nullable=False)
    text_value = Column(Text)
    float_value = Column(Float(53))
    int_value = Column(Integer)
    bool_value = Column(Boolean)
    __ts_vector__ = Column(TSVector(), server_default=text("to_tsvector('english', name || ' : ' || text_value)"))

    dataset = relationship('Dataset', lazy='joined')


class GISService(Base):
    __tablename__ = 'gis_service'
    __table_args__ = {'schema': schema}

    service_id = Column(Integer, primary_key=True,
                        server_default=text("nextval('gis_service_service_id_seq'::regclass)"))
    name = Column(String(256))
    description = Column(Text)
    service_type = Column(String(32), nullable=False)
    service_url = Column(String(256), nullable=False)
    layer_name = Column(String(128), nullable=False)
    layer_type = Column(String(32), nullable=False)
    dataset_id = Column(ForeignKey('hub_catalog.dataset.dataset_id'))
    color_map_id = Column(ForeignKey('hub_catalog.color_map.color_map_id'))

    dataset = relationship('Dataset')
    color_map = relationship('ColorMap', lazy='joined')


class ColorMap(Base):
    __tablename__ = 'color_map'
    __table_args__ = {'schema': schema}

    color_map_id = Column(Integer,
                          primary_key=True,
                          server_default=text("nextval('color_map_color_map_id_seq'::regclass)"))
    name = Column(String(128))

    color_map_entries = relationship('ColorMapEntry', back_populates="color_map", lazy='joined')


class ColorMapEntry(Base):
    __tablename__ = 'color_map_entry'
    __table_args__ = {'schema': schema}

    color_map_entry_id = Column(Integer,
                                primary_key=True,
                                server_default=text("nextval('color_map_entry_color_map_entry_id_seq'::regclass)"))
    color_map_id = Column(ForeignKey('hub_catalog.color_map.color_map_id'))
    text_value = Column(String(64))
    float_value = Column(Float(53))
    int_value = Column(Integer)
    color = Column(String(7))

    color_map = relationship('ColorMap', lazy='joined')


class TaxonomyItem(Base):
    __tablename__ = 'taxonomy_item'
    __table_args__ = {'schema': schema}

    taxonomy_item_id = Column(Integer, primary_key=True,
                              server_default=text("nextval('taxonomy_item_taxonomy_item_id_seq'::regclass)"))
    taxonomy_id = Column(ForeignKey('hub_catalog.taxonomy.taxonomy_id'), nullable=False)
    taxonomy_item_name = Column(String(256), nullable=False)
    parent_taxonomy_item_id = Column(ForeignKey('hub_catalog.taxonomy_item.taxonomy_item_id'))

    parent_taxonomy_item = relationship('TaxonomyItem', remote_side=[taxonomy_item_id])
    taxonomy = relationship('Taxonomy', lazy='joined')


class DatasetTaxonomyItem(Base):
    __tablename__ = 'dataset_taxonomy_item'
    __table_args__ = {'schema': schema}

    dataset_taxonomy_item_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('dataset_taxonomy_item_dataset_taxonomy_item_id_seq'::regclass)"))
    dataset_id = Column(ForeignKey('hub_catalog.dataset.dataset_id'), nullable=False)
    taxonomy_item_id = Column(ForeignKey('hub_catalog.taxonomy_item.taxonomy_item_id'), nullable=False)

    dataset = relationship('Dataset')
    taxonomy_item = relationship('TaxonomyItem', lazy='joined')


class User(Base):
    __tablename__ = 'useraccount'
    __table_args__ = {'schema': schema}

    user_id = Column(Integer, primary_key=True, server_default=text("nextval('users_user_id_seq'::regclass)"))
    username = Column(String(64))
    password_hash = Column(String(128))
    created_on = Column(DateTime, server_default=text("now()"))
    last_login = Column(DateTime, server_default=text("now()"))
    first_name = Column(String(64))
    last_name = Column(String(64))
    email = Column(String(128))
    is_verified = Column(Boolean, server_default=text("False"))
    affiliation = Column('agency', String(256))
    match_key = Column(UUID(as_uuid=True), server_default=text('uuid_generate_v1()'))

    roles = relationship("Role", secondary="hub_catalog.user_roles", back_populates="users")
    workspaces = relationship("Workspace", back_populates="user")


class Role(Base):
    __tablename__ = 'roles'
    __table_args__ = {'schema': schema}

    role_id = Column(Integer, primary_key=True, server_default=text("nextval('roles_role_id_seq'::regclass)"))
    name = Column(String(64))

    users = relationship(User, secondary="hub_catalog.user_roles", back_populates="roles")


class UserRole(Base):
    __tablename__ = 'user_roles'
    __table_args__ = {'schema': schema}

    user_role_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('user_roles_user_role_id_seq'::regclass)"))
    user_id = Column(ForeignKey('hub_catalog.useraccount.user_id'), nullable=False)
    role_id = Column(ForeignKey('hub_catalog.roles.role_id'), nullable=False)

    # Define a many-to-one relationship with the User model
    user = relationship("User")

    # Define a many-to-one relationship with the Role model
    role = relationship("Role")


class Workspace(Base):
    __tablename__ = 'workspace'
    __table_args__ = {'schema': schema}

    workspace_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('workspace_workspace_id_seq'::regclass)"))
    user_id = Column(ForeignKey('hub_catalog.useraccount.user_id'), nullable=False)
    workspace_name = Column(String(256), nullable=False)
    workspace_info = Column(JSON, nullable=False)
    created_on = Column(DateTime, server_default=text("now()"))
    last_used_date = Column(DateTime, nullable=False)

    # Define a many-to-one relationship with the User model
    user = relationship("User")


class DictionarySection(Base):
    __tablename__ = 'dictionary_section'
    __table_args__ = {'schema': schema}

    dictionary_section_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('dictionary_section_id_seq'::regclass)"))
    name = Column(String(128), nullable=False)

    dictionary_items = relationship("DictionaryItem",
                                    back_populates="dictionary_section",
                                    lazy='joined')


class DictionaryItem(Base):
    __tablename__ = 'dictionary_item'
    __table_args__ = {'schema': schema}

    dictionary_item_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('hub_catalog.dictionary_dictionary_id_seq'::regclass)"))
    dictionary_section_id = Column(ForeignKey('hub_catalog.dictionary_section.dictionary_section_id'), nullable=False)
    name = Column(String(128), nullable=False)
    value = Column(String(256), nullable=False)

    dictionary_section = relationship("DictionarySection")

                                                                                                                                                                    
class CKANPackageUser(Base):                                                                                                                                        
    __tablename__ = 'ckan_package_user'                                                                                                                             
    __table_args__ = {'schema': schema}                                                                                                                             
                                                                                                                                                                    
    ckan_package_user_id = Column(Integer, primary_key=True, server_default=text(                                                                                   
        "nextval('hub_catalog.ckan_package_user_ckan_package_user_id_seq'::regclass)"))                                                                             
    user_id = Column(ForeignKey('hub_catalog.useraccount.user_id'), nullable=False)                                                                                 
    package_id = Column(String(100), nullable=False)                                                                                                                
    updated_at = Column(DateTime, server_default=text("now()"))                                                                                                     
                                                                                                                                                                    
    # Define a many-to-one relationship with the User model                                                                                                         
    user = relationship("User")                                                                                                                                     
                                           