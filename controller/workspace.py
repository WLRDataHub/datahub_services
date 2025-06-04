import os
import pandas as pd
import json
from passlib.context import CryptContext
import psycopg2
from datetime import datetime
from controller.login import LoginFunctions
from dotenv.main import load_dotenv
from sqlalchemy.orm import Session
from models.wfr_database import Workspace, User
from models.wfr_pydantic import WorkspaceUpdate



class WorkspaceFunctions:

    def save_workspace(workspace_json: json, user: User, db: Session):

        workspace = Workspace()
        workspace.created_on = datetime.now()
        workspace.user_id = user.user_id
        workspace.workspace_name = workspace_json['workspaceName']
        workspace.last_used_date = datetime.now()

        del workspace_json['access_token']
        del workspace_json['workspaceName']

        workspace_str = json.dumps(workspace_json)
        workspace_str = f"""{workspace_str}"""
        workspace_str = workspace_str.replace('\'', "''")
        workspace.workspace_info = workspace_str

        db.add(workspace)
        db.commit()

    def update_workspace(workspace: Workspace, db: Session):
        db_workspace = db.query(Workspace).filter(Workspace.workspace_id == workspace.workspace_id).first()

        db_workspace.user_id = workspace.user_id
        db_workspace.workspace_name = workspace.workspace_name
        db_workspace.last_used_date = workspace.last_used_date
        db_workspace.workspace_info = workspace.workspace_info

        db.commit()


    def read_workspaces_by_userid(userId: str, db = Session):
        return db.query(Workspace).filter(Workspace.user_id == userId).all()
    

    def read_workspace(workspaceId: int, db = Session):
        return db.query(Workspace).filter(Workspace.workspace_id == workspaceId).first()


    def delete_workspace(workspace: Workspace, db: Session):
        db.delete(workspace)
        db.commit()
