from fastapi import FastAPI,  APIRouter, Request, Body, Response
import pandas as pd
import json
import psycopg2
from passlib.context import CryptContext
import bcrypt
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import jwt
from dotenv.main import load_dotenv
import os
from models.wfr_database import User
from models.wfr_pydantic import Account, UserBase
from controller.db import SessionLocal, get_db
from fastapi import Depends
from sqlalchemy.orm import Session
from controller.manager import get_user_by_username



class LoginFunctions:
    
    def __init__(self):
        load_dotenv('./fastApi.env')

        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.columns = ["user_id", "username", "password_hash", "first_name", "last_name", "email", 
                        "agency", "is_verified", "created_on", "last_login", "match_key"]
        
        self.jwt_key= os.environ['token_key']
        self.jwt_algo = os.environ['token_algorithm']
        self.base_url = os.environ['backend_full_url']
        self.verification_email = os.environ['verification_email']
        self.verification_email_pwd = os.environ['verification_email_pwd']
        self.sender_email = os.environ['sender_email']
        
    def check_jwt(token: str):
        key = LoginFunctions().jwt_key
        jwt_algo = LoginFunctions().jwt_algo
        decoded = jwt.decode(token, key, algorithms=[jwt_algo])

        expiration = datetime.datetime.strptime(decoded['expiration_date'] + ' 11:59:59', '%Y-%m-%d %H:%M:%S')

        if expiration < datetime.datetime.today():
            return None
        
        return decoded['username']

    def create_user(account: Account, db: Session):
        password_hash = LoginFunctions.create_hash_password(account.password)
        password_hash = str(password_hash)[1:]
        password_hash = password_hash[1:-1]

        user = User()
        user.affiliation = account.agency
        user.username = account.username
        user.email = account.email
        user.first_name = account.firstname
        user.last_name = account.lastname
        user.password_hash = password_hash

        db.add(user)
        db.commit()

    def updateUser(user: User, db: Session):
        db_user = db.query(User).filter(User.username == user.username).first()

        db_user.username = user.username
        db_user.password_hash = user.password_hash
        db_user.last_login = user.last_login
        db_user.first_name = user.first_name
        db_user.last_name = user.last_name
        db_user.affiliation = user.affiliation
        db_user.email = user.email
        db_user.is_verified = user.is_verified

        db.commit()
        

    def update_last_login(username: str, db: Session):
        db_user = db.query(User).filter(User.username == username).first()
        db_user.last_login = datetime.datetime.now()
        db.commit()

    def create_hash_password(password):
        mySalt = bcrypt.gensalt()
        bytePwd = password.encode('utf-8')
        pwd_hash = bcrypt.hashpw(bytePwd, mySalt)

        return pwd_hash
    

    def send_verification_email(user: User, access_token: str):

        base_url =LoginFunctions().base_url
        verification_email = LoginFunctions().verification_email
        verification_email_pwd = LoginFunctions().verification_email_pwd

        sender_email = LoginFunctions().sender_email
        message = MIMEMultipart("")
        message["Subject"] = "Wildfire and Landscape Resilience Data Hub Explorer - Verify Account"
        message["From"] = sender_email
        message["To"] = verification_email

        html = f"""\
                <html>
                <body>
                    <p>Hello admin,
                    <br>
                    <br>
                        A new account has been added to the Wildfire and Landscape Resilience Data Hub Explorer. 
                    </p>
                    Their information is below: 
                    <ul>
                        <li><strong>Name:</strong> {user.first_name} {user.last_name}</li>
                        <li><strong>Username:</strong> {user.username}</li>
                        <li><strong>Email:</strong> {user.email}</li>
                        <li><strong>Affiliation:</strong> {user.affiliation}</li>
                    </ul>
                    <div>
                       If you wish to approve this account, click here: {base_url}/Auth/validate_account/{user.username}/{access_token}
                    </div>
                </body>
                </html>
                """ 

        # Create a secure SSL context
        context = ssl.create_default_context()
        smtp_server = "smtp.gmail.com"
        port = 465  # For starttls

        part2 = MIMEText(html, "html")

        # Attach both versions to the outgoing message
        message.attach(part2)

        try:
            smtp_server = smtplib.SMTP_SSL(smtp_server, port, context=context)
            smtp_server.ehlo()
            smtp_server.login(sender_email, verification_email_pwd)
            smtp_server.sendmail(sender_email, verification_email, message.as_string())     
            
            return { "status": 'success', "message": 'Email sent to admin for approval. You will recieve and email when your account has been approved.'}
        except smtplib.SMTPException as e:
            return { "status": 'success', "message": f'Email was not sent. Error: {e}'}


    def verify_account(username: str, db: Session):
        
        user = get_user_by_username(username)

        user.is_verified = True

        LoginFunctions.updateUser(user, db)

        LoginFunctions.send_approval_email(user)

        return { "status": "success", 'message': f"Account {username} has been verified. An email was sent to the account's provided email." }


    def send_approval_email(user:User):

        sender_email = LoginFunctions().sender_email
        verification_email = LoginFunctions().verification_email
        verification_email_pwd = LoginFunctions().verification_email_pwd

        message = MIMEMultipart("")
        message["Subject"] = "Wildfire and Landscape Resilience Data Hub Explorer Account Approved"
        message["From"] = sender_email

        message["To"] = user.email

        base_url = LoginFunctions().base_url

        html = f"""\
                <html>
                <body>
                    <p>Hello {user.first_name} {user.last_name},
                    <br>
                    <br>
                        Your Wildfire and Landscape Resilience Data Hub Explorer account has been approved. You now have access to the Data Hub. 
                    </p>
                    <div>
                       Here is the link to login and get started: {base_url}
                    </div>
                </body>
                </html>
                """ 

        # Create a secure SSL context
        context = ssl.create_default_context()
        smtp_server = "smtp.gmail.com"
        port = 465  # For starttls

        part2 = MIMEText(html, "html")

        # Attach both versions to the outgoing message
        message.attach(part2)

        try:
            smtp_server = smtplib.SMTP_SSL(smtp_server, port, context=context)
            smtp_server.ehlo()
            smtp_server.login(sender_email, verification_email_pwd)
            smtp_server.sendmail(sender_email, verification_email, message.as_string())  
            
            return { "status": 'success', "message": 'Email sent to admin for approval. You will recieve and email when your account has been approved.'}
        except smtplib.SMTPException as e:
            return { "status": 'success', "message": f'Email was not sent. Error: {e}'}
