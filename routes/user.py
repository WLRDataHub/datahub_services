from fastapi import APIRouter

from controller.endpoint_collections import EndpointCollection, ManyToManySubfield, EditableSubfield
import models.wfr_database as wfr_database
import models.wfr_pydantic as wfr_pydantic


router = APIRouter(tags=["User"], prefix='/User')

EndpointCollection(
    name="User",
    id_key="user_id",
    database_model=wfr_database.User,
    pydantic_model=wfr_pydantic.User,
    pydantic_model_update=wfr_pydantic.UserUpdate,
    pydantic_model_create=wfr_pydantic.UserBase,
).with_subfields(
    [
        ManyToManySubfield(
            name="Role",
            id_key="role_id",
            database_model=wfr_database.Role,
            pydantic_model=wfr_pydantic.Role,
            pydantic_model_update=wfr_pydantic.RoleUpdate,
            pydantic_model_create=wfr_pydantic.RoleBase,
            join_table=wfr_database.UserRole,
        ),
        EditableSubfield(
            name="Workspace",
            id_key="workspace_id",
            database_model=wfr_database.Workspace,
            pydantic_model=wfr_pydantic.Workspace,
            pydantic_model_update=wfr_pydantic.WorkspaceUpdate,
            pydantic_model_create=wfr_pydantic.WorkspaceBase,
        ),
    ]
).register_router(
    router
)