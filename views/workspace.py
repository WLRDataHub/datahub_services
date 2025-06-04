from fastapi import FastAPI, APIRouter, HTTPException, Request, Body, Response
import json
from controller.workspace import WorkspaceFunctions
from controller.login import LoginFunctions
from controller.db import get_db
from fastapi import Depends
from sqlalchemy.orm import Session
from controller.manager import manager
from fastapi.encoders import jsonable_encoder
import datetime
from models.wfr_database import User



router = APIRouter(
    prefix='/v1/workspace',
    tags = ['workspace']
)

@router.post("/workspace")
async def workspace(data: str, user=Depends(manager), db: Session = Depends(get_db)):
    
    if user == None: 
         raise HTTPException(status_code=500, detail="User Not Found")
    
    workspace = json.loads(data)
    WorkspaceFunctions.save_workspace(workspace, user, db)

    return { "status" : "success", "message": "Workspace Saved"}


@router.get("/read_workspaces_by_user")
async def read_workspaces_by_user(user=Depends(manager), db: Session = Depends(get_db)):

    if user == None: 
         raise HTTPException(status_code=500, detail="User Not Found")
    
    workspaces = WorkspaceFunctions.read_workspaces_by_userid(user.user_id, db)

    if workspaces == None:
        raise HTTPException(status_code=500, detail="Workspace Not Found")

    if any([user.user_id != w.user_id for w in workspaces]):
        raise HTTPException(status_code=500, detail="Workspace Not Authorized")

    # workspaces = workspaces.sort_values(by='last_used_date', ascending=False) 

    results_json = json.dumps([jsonable_encoder(w) for w in workspaces])

    return { "status" : "success", "results" : results_json}


@router.put('/workspace_last_used/{workspaceId}')
def workspace_last_used(workspaceId: int, user=Depends(manager), db: Session = Depends(get_db)):
    if workspaceId == None: 
        raise HTTPException(status_code=500, detail="Workspace Not Found")
    
    if user == None:
        raise HTTPException(status_code=500, detail="User Not Found")
    
    # read workspace needs to grab pydantic models not db models
    workspace = WorkspaceFunctions.read_workspace(workspaceId, db)

    if workspace == None or (user.user_id != workspace.user_id):
        raise HTTPException(status_code=500, detail="Workspace Not Found")

    workspace.last_used_date = datetime.datetime.now()

    WorkspaceFunctions.update_workspace(workspace, db)

    return { "status" : "success" }


@router.delete("/workspace/{workspace_id}")
async def workspace(workspace_id: int, user=Depends(manager), db: Session = Depends(get_db)):

    if workspace_id == None: 
        raise HTTPException(status_code=500, detail="Workspace Not Found")
    
    if user == None:
        raise HTTPException(status_code=500, detail="User not found")

    workspace = WorkspaceFunctions.read_workspace(workspace_id, db)

    if workspace == None: 
        raise HTTPException(status_code=500, detail="Workspace Not Found")
    
    if user.user_id != workspace.user_id and (user.roles == None or (user.roles != None and any([r.name == 'admin' for r in user.roles]) == False)):
        raise HTTPException(status_code=500, detail="Workspace Not Authorized")

    WorkspaceFunctions.delete_workspace(workspace, db)

    return {"status": "success", "message": "Workspace Deleted"}




