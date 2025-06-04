from fastapi_login import LoginManager

from controller.db import SessionLocal
from models.wfr_database import User
from sqlalchemy.orm import lazyload

SECRET = "z5_V3JG7E0=bZELa'^zk))AMtd*g"
manager = LoginManager(SECRET, '/staging-api/v1/Auth/login')

@manager.user_loader()
def get_user_by_username(username: str):
    return SessionLocal().query(User).filter(User.username == username).options(lazyload(User.roles)).first()