"""
Generic CRUD operations.

"""
import json
from typing import Optional, List

import sqlalchemy
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import desc, select, delete, text, asc

from controller.str_utils import change_case
import models.wfr_database as wfr_database
import models.wfr_pydantic as pydantic


def count_all(
        db: Session,
        orm_model: wfr_database.Base,
        id_key,
        search_join: wfr_database.Base = None,
        where: bool = True,

        filter_model: wfr_database.Base = None,
        filter_field=None,
        allowed_values: Optional[List[int]] = None,
) -> int:
    if search_join is not None:
        id_field = getattr(orm_model, id_key)
        subquery = db.query(id_field).distinct(id_field).join(search_join).filter(where)
        query = db.query(orm_model).filter(id_field.in_(subquery))

        if filter_model is not None and filter_field is not None and allowed_values is not None:
            subquery2 = db.query(id_field).distinct(id_field).join(filter_model).filter(
                filter_field.in_(allowed_values))
            query = query.filter(id_field.in_(subquery2))

        return query.count()
    else:
        return db.query(orm_model).filter(where).count()


def get_all(
        db: Session,
        orm_model: wfr_database.Base,
        search_join: wfr_database.Base = None,
        skip: int = 0,
        limit: int = 100,
        where: bool = True,
        order_by=None,
        ascending=False,
        endpoint_name: Optional[str] = None,
        use_cache=True,
        no_validation=False,
        prefix=False,
        id_key: str = None,

        filter_model: wfr_database.Base = None,
        filter_field=None,
        allowed_values: Optional[List[int]] = None,
) -> list:
    endpoint = endpoint_name if endpoint_name else orm_model.__name__
    if isinstance(order_by, str):
        order_by = getattr(orm_model, order_by)

    _ordering = asc(order_by) if ascending else desc(order_by)

    if search_join is not None:
        id_field = getattr(orm_model, id_key)
        subquery = db.query(id_field).distinct(id_field).join(search_join).filter(where)
        query = db.query(orm_model).filter(id_field.in_(subquery))

        if filter_model is not None and filter_field is not None and allowed_values is not None:
            subquery2 = db.query(id_field).distinct(id_field).join(filter_model).filter(
                filter_field.in_(allowed_values))
            query = query.filter(id_field.in_(subquery2))


        database_results = (
            query
                .order_by(_ordering)
                .offset(skip)
                .limit(limit)
                .all()
        )

        # database_results = (
        #     db.query(orm_model)
        #         .join(search_join)
        #         .filter(where)
        #         .order_by(_ordering)
        #         .offset(skip)
        #         .limit(limit)
        #         .all()
        # )
    else:
        database_results = (
            db.query(orm_model)
                .filter(where)
                .order_by(_ordering)
                .offset(skip)
                .limit(limit)
                .all()
        )

    if no_validation:
        return ORJSONResponse(jsonable_encoder(database_results))

    return database_results


def get_all_partial(
        db: Session,
        orm_model: wfr_database.Base,
        fields: List[str],
        query: str,
        order_by: Optional[str] = None,
        ascending=False,
        encrypted_fields: List[str] = [],
        additional_where: bool = True,
        multi: Optional[str] = False,
):
    _full_fields = [
        f"CONVERT(VARCHAR(MAX), DecryptByKey({encrypted_field}))"
        for encrypted_field in encrypted_fields
    ]

    # field_string = " + ' ' + ".join(fields + _full_fields)
    field_string = ' OR '.join(f"UPPER({x}) LIKE UPPER('{query}%') " for x in fields + _full_fields if x)
    _ordering = asc(order_by) if ascending else desc(order_by)

    return (
        db.query(orm_model)
            .where(additional_where)
            # .where(text(f"{field_string} LIKE '%{query}%'"))
            .where(text(field_string))
            .order_by(_ordering)
            .all()
    )


def get_by_id(db: Session, orm_model: wfr_database.Base, id: int, model_id_key=None):
    if model_id_key is None:
        model_id_key = change_case(orm_model.__tablename__) + "_id"

    model_id_key = sqlalchemy.inspect(orm_model).primary_key[0].name
    statement = select(orm_model).where(getattr(orm_model, model_id_key) == id)
    res = db.execute(statement).scalars().first()
    if res is None:
        raise HTTPException(404)
    return res


def delete_by_id(db: Session, orm_model: wfr_database.Base, id: int, model_id_key=None):
    if model_id_key is None:
        model_id_key = change_case(orm_model.__tablename__) + "_id"
    # Throws 404 if not found
    model_id_key = sqlalchemy.inspect(orm_model).primary_key[0].name
    get_by_id(db, orm_model, id, model_id_key=model_id_key)
    statement = delete(orm_model).where(getattr(orm_model, model_id_key) == id)
    try:
        db.execute(statement)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


def update_by_id(
        db: Session,
        id: int,
        new_entry: pydantic.BaseModel,
        db_model,
        id_prefix: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        model_id=None,
):
    if model_id is None:
        if id_prefix is None:
            id_prefix = change_case(db_model.__tablename__)
        model_id = f"{id_prefix}_id"
    else:
        model_id = model_id

    old = (
        db.query(db_model)
            .filter(getattr(db_model, model_id) == id)
            .one_or_none()
    )

    if old is None:
        raise HTTPException(404)

    new_items = new_entry.dict(exclude_unset=True).items()

    for k, v in new_items:
        setattr(old, k, v)

    try:
        db.add(old)
        db.commit()
        db.refresh(old)
        return old

    except Exception as e:
        db.rollback()
        raise HTTPException(500, e)


def add_entry(
        db: Session,
        db_model,
        new_entry: pydantic.BaseModel,
        encrypted=False,
        endpoint_name: Optional[str] = None,
):
    # open_symmkey(db)
    new_db_model = db_model(new_entry.dict())
    db.add(new_db_model)
    try:
        db.commit()
        db.refresh(new_db_model)

        return new_db_model
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            409,
            "The request conflicts with a Database constraint. Please ensure that all foreign keys point to valid rows."
            + str(e),
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(500, str(e))
