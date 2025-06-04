from dataclasses import dataclass
from typing import List, Literal, Optional, Union

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from fastapi_login.exceptions import InvalidCredentialsException
from pydantic import BaseModel
from sqlalchemy import Table, asc, desc, Column
from sqlalchemy.orm import Session

import controller.crud as crud
from controller.db import get_db
from controller.manager import manager
import models.wfr_database as wfr_database

class DefaultArgs(BaseModel):
    skip: int
    limit: Optional[int]
    """The key to sort the model by.
    """
    order_by: Optional[str]
    ascending: bool
    prefix: bool


def default_args(order_by: str):
    def _default_args(
            skip: int = 0,
            limit: Union[int, Literal["all"]] = 100,
            order_by: Optional[str] = Query(
                order_by, description="The key on the model to sort by"
            ),
            ascending: bool = False,
            prefix: bool = False,
    ) -> DefaultArgs:
        lim = None if limit == "all" else limit
        return DefaultArgs(
            **{"skip": skip, "limit": lim, "order_by": order_by, "ascending": ascending, "prefix": prefix}
        )

    return _default_args


def is_admin(user):
    if user and user.roles:
        for role in user.roles:
            if role.name == 'admin':
                return True
    return False


@dataclass
class EndpointCollection:
    name: str
    pydantic_model: BaseModel
    database_model: wfr_database.Base
    id_key: str
    sort_key: str = None
    # searchable_fields: List[str] = field(default_factory=list)
    searchable: bool = False
    search_join: wfr_database.Base = None
    filter_model: wfr_database.Base = None
    filter_field: Column = None

    @property
    def _sort_key(self):
        if self.sort_key is None:
            return self.id_key
        return self.sort_key

    # An optional sub-model that defines entries that are updateable
    pydantic_model_update: Optional[BaseModel] = None
    # An optional sub-model that describes a new entry's input.
    pydantic_model_create: Optional[BaseModel] = None
    # An optional sub-model primarily used for get_all responses.
    pydantic_model_slim: Optional[BaseModel] = None

    @property
    def _get_all_model(self):
        # if self.pydantic_model_slim is not None:
        #     return self.pydantic_model_slim
        return self.pydantic_model

    _subfields = []

    def _register_get(self, router: APIRouter):

        if self.searchable is True or self.search_join is not None:

            @router.get(
                "/count",
                response_model=int,
                name=f"Count all {self.name}",
            )
            def get_count(
                    search_terms: Optional[str] = Query(
                        None, description="Search query term on the collection."
                    ),
                    allowed_values: Optional[List[int]] = Query(
                        None, description="Allowed values"
                    ),
                    db: Session = Depends(get_db),
            ):
                if search_terms and search_terms.strip():
                    if self.search_join:
                        filter_cond = self.search_join.__ts_vector__.match(search_terms)
                        if self.searchable:
                            filter_cond = filter_cond | self.database_model.__ts_vector__.match(search_terms)
                    else:
                        filter_cond = self.database_model.__ts_vector__.match(search_terms)
                else:
                    filter_cond = True

                return crud.count_all(
                    db,
                    self.database_model,
                    search_join=self.search_join,
                    where=filter_cond,
                    id_key=self.id_key,
                    filter_model=self.filter_model,
                    filter_field=self.filter_field,
                    allowed_values=allowed_values
                )

            @router.get(
                "/",
                response_model=List[self.pydantic_model],
                name=f"Retrieve all {self.name}",
            )
            def get_all(
                    search_terms: Optional[str] = Query(
                        None, description="Search query term on the collection."
                    ),
                    allowed_values: Optional[List[int]] = Query(
                        None, description="Allowed values"
                    ),
                    skip: int = 0,
                    limit: int = 100,
                    order_by: Optional[str] = Query(
                        self.id_key, description="The key on the model to sort by"
                    ),
                    ascending: bool = True,
                    db: Session = Depends(get_db),
            ):

                if search_terms and search_terms.strip():
                    if self.search_join:
                        filter_cond = self.search_join.__ts_vector__.match(search_terms)
                        if self.searchable:
                            filter_cond = filter_cond | self.database_model.__ts_vector__.match(search_terms)
                    else:
                        filter_cond = self.database_model.__ts_vector__.match(search_terms)
                else:
                    filter_cond = True
                return crud.get_all(
                    db,
                    self.database_model,
                    search_join=self.search_join,
                    where = filter_cond,
                    filter_model=self.filter_model,
                    filter_field=self.filter_field,
                    allowed_values=allowed_values,
                    order_by=order_by,
                    ascending=ascending,
                    skip=skip,
                    limit=limit,
                    id_key=self.id_key
                )
        elif self.database_model.__tablename__ == 'useraccount':
            @router.get(
                "/",
                response_model=List[self.pydantic_model],
                name=f"Retrieve all {self.name}",
            )
            def get_all(
                    skip: int = 0,
                    limit: int = 100,
                    order_by: Optional[str] = Query(
                        self.id_key, description="The key on the model to sort by"
                    ),
                    ascending: bool = True,
                    db: Session = Depends(get_db),
                    user=Depends(manager),
            ):
                if not is_admin(user):
                    raise InvalidCredentialsException

                return crud.get_all(
                    db,
                    self.database_model,
                    order_by=order_by,
                    ascending=ascending,
                    skip=skip,
                    limit=limit
                )
        else:
            @router.get(
                "/",
                response_model=List[self.pydantic_model],
                name=f"Retrieve all {self.name}",
            )
            def get_all(
                    skip: int = 0,
                    limit: int = 100,
                    order_by: Optional[str] = Query(
                        self.id_key, description="The key on the model to sort by"
                    ),
                    ascending: bool = True,
                    db: Session = Depends(get_db),
            ):
                return crud.get_all(
                    db,
                    self.database_model,
                    order_by=order_by,
                    ascending=ascending,
                    skip=skip,
                    limit=limit
                )

        if self.database_model.__tablename__ == 'useraccount':
            @router.get(
                f"/{{{self.id_key}}}",
                response_model=self.pydantic_model,
                name=f"Retrieve {self.name} by ID",
            )
            def get_by_id(
                    item_id: int = Path(..., alias=self.id_key),
                    db=Depends(get_db),
                    user=Depends(manager),
            ):
                if not is_admin(user) and not user.user_id == item_id:
                    raise InvalidCredentialsException

                return crud.get_by_id(db, self.database_model, item_id, model_id_key=self.id_key)
        else:
            @router.get(
                f"/{{{self.id_key}}}",
                response_model=self.pydantic_model,
                name=f"Retrieve {self.name} by ID",
            )
            def get_by_id(
                    item_id: int = Path(..., alias=self.id_key),
                    db=Depends(get_db),
            ):
                return crud.get_by_id(db, self.database_model, item_id, model_id_key=self.id_key)

    def _register_put(self, router: APIRouter):
        if self.pydantic_model_update is not None:
            pydantic_model_update = self.pydantic_model_update

            @router.put(
                f"/{{{self.id_key}}}",
                response_model=self.pydantic_model,
                name=f"Update {self.name} by ID",
            )
            def update_by_id(
                    item_id: int = Path(..., alias=self.id_key),
                    copy: bool = Query(
                        False,
                        description="If set to true, specify all fields that you would like copied and copies the "
                                    "specified item.",
                    ),
                    updates: pydantic_model_update = Body(...),
                    db=Depends(get_db),
                    user=Depends(manager)
            ):
                if not is_admin(user):
                    if not self.database_model.__tablename__ == 'useraccount' or not user.user_id == item_id:
                        raise InvalidCredentialsException

                if copy:
                    original_item = crud.get_by_id(db, self.database_model, item_id, self.id_key)
                    copy = [original_item[k] for k in updates.keys()]
                    return crud.add_entry(db, self.database_model, copy)
                return crud.update_by_id(db, item_id, updates, self.database_model, model_id=self.id_key)

    def _register_post(self, router: APIRouter):
        pydantic_model_create = self.pydantic_model_create
        if self.pydantic_model_create is not None:

            if self.database_model.__tablename__ == 'useraccount':
                @router.post(
                    "/",
                    response_model=self.pydantic_model,
                    name=f"Create new {self.name}",
                )
                def create_new(new: pydantic_model_create,
                               db=Depends(get_db)):
                    return crud.add_entry(db, self.database_model, new)
            else:
                @router.post(
                    "/",
                    response_model=self.pydantic_model,
                    name=f"Create new {self.name}",
                )
                def create_new(new: pydantic_model_create,
                               db=Depends(get_db),
                               user=Depends(manager)):
                    if not is_admin(user):
                        raise InvalidCredentialsException
                    return crud.add_entry(db, self.database_model, new)

    def _register_delete(self, router: APIRouter):
        @router.delete(f"/{{{self.id_key}}}", name=f"Delete {self.name} by ID")
        def delete_by_id(
                item_id: int = Path(..., alias=self.id_key),
                db=Depends(get_db),
                user=Depends(manager)
        ):
            if not is_admin(user):
                if not self.database_model.__tablename__ == 'useraccount' or not user.user_id == item_id:
                    raise InvalidCredentialsException
            return crud.delete_by_id(db, self.database_model, item_id, self.id_key)

    @property
    def _register_methods(self):
        """All the route-able methods to be registered with.

        Returns:
            [type]: [description]
        """
        return [
            self._register_get,
            self._register_post,
            self._register_put,
            self._register_delete,
        ]

    def _add_subfield(self, subfield):
        self._subfields.append(subfield)

    def with_subfields(self, subfields):
        self._subfields = []
        for s in subfields:
            s.parent_endpoint = self
            self._add_subfield(s)
        return self

    def register_router(self, router: APIRouter):
        """Registers this endpoint collection with a FastAPI router.

        Args:
            router (APIRouter): [description]
        """
        for subfield in self._subfields:
            subfield.register_router(router)

        for register_method in self._register_methods:
            register_method(router)

    class Config:
        arbitrary_types_allowed = True


@dataclass
class EditableSubfield(EndpointCollection):
    parent_endpoint: EndpointCollection = None

    def _register_post(self, router: APIRouter):
        pydantic_model_create = self.pydantic_model_create

        @router.post(f"/{{{self.parent_endpoint.id_key}}}/{self.name}")
        def create_new(
                parent_item_id: int = Path(..., alias=self.parent_endpoint.id_key),
                new: pydantic_model_create = Body(..., alias=f"new_{self.name}"),
                db=Depends(get_db),
                user=Depends(manager)
        ):
            if not is_admin(user):
                if not super().__dict__['parent_endpoint'].name == 'User' or not user.user_id == parent_item_id:
                    raise InvalidCredentialsException

            setattr(new, self.parent_endpoint.id_key, parent_item_id)
            try:
                return crud.add_entry(db, self.database_model, new)
            except:
                raise HTTPException(
                    409,
                    f"The submitted new entry appears to violate a Database constriant. Please ensure that foreign keys point to valid rows.",
                )

    def _register_put(self, router: APIRouter):
        pydantic_model_update = self.pydantic_model_update

        @router.put(f"/{{{self.parent_endpoint.id_key}}}/{self.name}/{{{self.id_key}}}")
        def update_by_id(
                updated: pydantic_model_update = Body(...),
                parent_item_id: int = Path(..., alias=self.parent_endpoint.id_key),
                item_id: int = Path(..., alias=self.id_key),
                db=Depends(get_db),
                user=Depends(manager)
        ):
            if not is_admin(user):
                if not super().__dict__['parent_endpoint'].name == 'User' or not user.user_id == parent_item_id:
                    raise InvalidCredentialsException

            return crud.update_by_id(db, item_id, updated, self.database_model, model_id=self.id_key)

    def _register_delete(self, router: APIRouter):
        @router.delete(
            f"/{{{self.parent_endpoint.id_key}}}/{self.name}/{{{self.id_key}}}"
        )
        def delete_by_id(
                parent_item_id: int = Path(..., alias=self.parent_endpoint.id_key),
                item_id: int = Path(..., alias=self.id_key),
                db=Depends(get_db),
                user=Depends(manager)
        ):
            if not is_admin(user):
                if not super().__dict__['parent_endpoint'].name == 'User' or not user.user_id == parent_item_id:
                    raise InvalidCredentialsException

            return crud.delete_by_id(db, self.database_model, item_id)

    def _register_get(self, router: APIRouter):

        # @router.get(
        #     f"/{self.name}",
        #     response_model=List[self.pydantic_model],
        #     name=f"Retrieve all {self.name}",
        # )
        # def get_all(
        #         skip: int = 0,
        #         limit: int = 100,
        #         order_by: Optional[str] = Query(
        #             self.id_key, description="The key on the model to sort by"
        #         ),
        #         ascending: bool = True,
        #         db=Depends(get_db),
        # ):
        #     return crud.get_all(
        #             db,
        #             self.database_model,
        #             order_by=order_by,
        #             ascending=ascending,
        #             skip=skip,
        #             limit=limit
        #         )

        # @router.get(
        #     f"/{self.name}/{{{self.id_key}}}",
        #     response_model=self.pydantic_model,
        #     name=f"Retrieve {self.name} by ID",
        # )
        # def get_by_id(
        #         item_id: int = Path(..., alias=self.id_key),
        #         db=Depends(get_db),
        # ):
        #
        #     return crud.get_by_id(db, self.database_model, item_id, model_id_key=self.id_key)

        if super().__dict__['parent_endpoint'].name == 'User':
            @router.get(
                f"/{{{self.parent_endpoint.id_key}}}/{self.name}",
                response_model=List[self.pydantic_model],
                summary=f"Get all associated {self.name}",
            )
            def get_all_associated(
                    parent_item_id: int = Path(..., alias=self.parent_endpoint.id_key),
                    # args: DefaultArgs = Depends(default_args(self._sort_key)),
                    skip: int = 0,
                    limit: int = 100,
                    order_by: Optional[str] = Query(
                        self.id_key, description="The key on the model to sort by"
                    ),
                    ascending: bool = True,
                    db=Depends(get_db),
                    user=Depends(manager)
            ):
                if not is_admin(user) and not user.user_id == parent_item_id:
                    raise InvalidCredentialsException

                return crud.get_all(
                    db,
                    self.database_model,
                    where=(
                            getattr(self.database_model, self.parent_endpoint.id_key)
                            == parent_item_id
                    ),
                    # **args.dict(),
                    order_by=order_by,
                    ascending=ascending,
                    skip=skip,
                    limit=limit
                )
        else:
            @router.get(
                f"/{{{self.parent_endpoint.id_key}}}/{self.name}",
                response_model=List[self.pydantic_model],
                summary=f"Get all associated {self.name}",
            )
            def get_all_associated(
                    parent_item_id: int = Path(..., alias=self.parent_endpoint.id_key),
                    # args: DefaultArgs = Depends(default_args(self._sort_key)),
                    skip: int = 0,
                    limit: int = 100,
                    order_by: Optional[str] = Query(
                        self.id_key, description="The key on the model to sort by"
                    ),
                    ascending: bool = True,
                    db=Depends(get_db),
            ):
                return crud.get_all(
                    db,
                    self.database_model,
                    where=(
                            getattr(self.database_model, self.parent_endpoint.id_key)
                            == parent_item_id
                    ),
                    # **args.dict(),
                    order_by=order_by,
                    ascending=ascending,
                    skip=skip,
                    limit=limit
                )

    def register_router(self, router: APIRouter):
        for register_method in self._register_methods:
            register_method(router)


@dataclass
class ManyToManySubfield(EditableSubfield):
    join_table: Table = None

    def _register_get(self, router: APIRouter):
        pydantic_model = self.pydantic_model

        if super().__dict__['parent_endpoint'].__dict__['database_model'].__tablename__ == 'useraccount':
            @router.get(
                f"/{{{self.parent_endpoint.id_key}}}/{self.name}",
                description=f"""Get all {self.name} for {self.parent_endpoint.name}""",
                response_model=List[self.pydantic_model]
            )
            def get_all_associated(
                    parent_item_id: int = Path(..., alias=self.parent_endpoint.id_key),
                    skip: int = 0,
                    limit: int = 100,
                    order_by: Optional[str] = Query(
                        self.id_key, description="The key on the model to sort by"
                    ),
                    ascending: bool = True,
                    db=Depends(get_db),
                    user=Depends(manager)
            ):
                if not is_admin(user) and not user.user_id == parent_item_id:
                    raise InvalidCredentialsException

                if isinstance(order_by, str):
                    order_by = getattr(self.database_model, order_by)
                _ordering = asc(order_by) if ascending else desc(order_by)

                return db.query(self.database_model) \
                    .join(self.join_table) \
                    .filter(getattr(self.join_table, self.parent_endpoint.id_key) == parent_item_id,
                            getattr(self.join_table, self.id_key) == getattr(self.database_model, self.id_key)) \
                    .order_by(_ordering) \
                    .offset(skip) \
                    .limit(limit) \
                    .all()
        else:
            @router.get(
                f"/{{{self.parent_endpoint.id_key}}}/{self.name}",
                description=f"""Get all {self.name} for {self.parent_endpoint.name}""",
                response_model=List[self.pydantic_model],
            )
            def get_all_associated(
                    parent_item_id: int = Path(..., alias=self.parent_endpoint.id_key),
                    skip: int = 0,
                    limit: int = 100,
                    order_by: Optional[str] = Query(
                        self.id_key, description="The key on the model to sort by"
                    ),
                    ascending: bool = True,
                    db=Depends(get_db),
            ):
                if isinstance(order_by, str):
                    order_by = getattr(self.database_model, order_by)
                _ordering = asc(order_by) if ascending else desc(order_by)

                return db.query(self.database_model) \
                    .join(self.join_table) \
                    .filter(getattr(self.join_table, self.parent_endpoint.id_key) == parent_item_id,
                            getattr(self.join_table, self.id_key) == getattr(self.database_model, self.id_key)) \
                    .order_by(_ordering) \
                    .offset(skip) \
                    .limit(limit) \
                    .all()

    def _register_post(self, router: APIRouter):
        pydantic_model_create = self.pydantic_model_create

        @router.post(
            f"/{{{self.parent_endpoint.id_key}}}/{self.name}",
            description=f"""Create new {self.name} for {self.parent_endpoint.name}""",
            response_model=self.pydantic_model,
        )
        def create_new(
                parent_item_id: int = Path(..., alias=self.parent_endpoint.id_key),
                new: pydantic_model_create = Body(..., alias=self.name),
                db=Depends(get_db),
                user=Depends(manager)
        ):
            if not is_admin(user):
                raise InvalidCredentialsException

            new_entry = crud.add_entry(db, self.database_model, new)
            # Register the relationship with the join table
            new_relationship = self.join_table()
            setattr(new_relationship, self.parent_endpoint.id_key, parent_item_id)
            setattr(new_relationship, self.id_key, getattr(new_entry, self.id_key))
            db.add(new_relationship)
            db.commit()
            return new_entry

    def _register_patch(self, router: APIRouter):
        @router.patch(
            f"/{{{self.parent_endpoint.id_key}}}/{self.name}/{{{self.id_key}}}"
        )
        def link_items(
                parent_id: int = Path(..., alias=self.parent_endpoint.id_key),
                item_id: int = Path(..., alias=self.id_key),
                db=Depends(get_db),
                user=Depends(manager)
        ):
            if not is_admin(user):
                raise InvalidCredentialsException

            f"""Links an existing {self.parent_endpoint.name} instance with an existing {self.name} instance."""
            new_entry = self.join_table()
            setattr(new_entry, self.parent_endpoint.id_key, parent_id)
            setattr(new_entry, self.id_key, item_id)
            db.add(new_entry)
            db.commit()
            return True

    def _register_delete(self, router: APIRouter):
        @router.delete(
            f"/{{{self.parent_endpoint.id_key}}}/{self.name}/{{{self.id_key}}}"
        )
        def delete_link_between_items(
                parent_id: int = Path(..., alias=self.parent_endpoint.id_key),
                item_id: int = Path(..., alias=self.id_key),
                db=Depends(get_db),
                user=Depends(manager)
        ):
            if not is_admin(user):
                raise InvalidCredentialsException

            f"""
            Severs ties between a {self.parent_endpoint.name} entry and an
            existing {self.name} instance. **Note**: This endpoint does not remove either entry in their respective tables.
            """
            db.query(self.join_table).filter_by(
                **{
                    self.id_key: item_id,
                    self.parent_endpoint.id_key: parent_id,
                }
            ).delete()

            db.commit()
            return True

    @property
    def _register_methods(self):
        """All the route-able methods to be registered with.

        Returns:
            [type]: [description]
        """
        return [
            self._register_get,
            self._register_post,
            # self._register_put,
            self._register_delete,
            self._register_patch,
        ]


def register_endpoint(router: APIRouter):
    """Convenience decorator for registering an endpoint with a FastAPI router.

    Args:
        cls (EndpointCollection): [description]
        router (APIRouter): [description]

    Returns:
        [type]: [description]
    """

    def dec(cls: EndpointCollection):
        cls.register_router(router)
        return cls
