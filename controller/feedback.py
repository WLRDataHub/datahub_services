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


class FeedbackFunctions:
    
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
        

    def send_feedback_email(user:User, feedback_text: str, feedback_topic: str):

        base_url =FeedbackFunctions().base_url
        verification_email = FeedbackFunctions().verification_email
        verification_email_pwd = FeedbackFunctions().verification_email_pwd

        sender_email = FeedbackFunctions().sender_email
        message = MIMEMultipart("")
        message["Subject"] = "Wildfire and Landscape Data Hub - Feedback"
        message["From"] = sender_email
        message["To"] = verification_email

        # message.attach(MIMEText(file))

        html = f"""\
                <html>
                <body>
                    <p>Hello admin,
                    <br>
                    <br>
                       {user.first_name} {user.last_name} submitted feedback about the WLR Data Hub. 
                    </p>
                    Their information is below: 
                    <ul>
                        <li><strong>Name:</strong> {user.first_name} {user.last_name}</li>
                        <li><strong>Username:</strong> {user.username}</li>
                        <li><strong>Email:</strong> {user.email}</li>
                        <li><strong>Agency:</strong> {user.affiliation}</li>
                    </ul>
                    <div>
                        Feedback Topic : {feedback_topic}
                    </div>
                    <div>
                        Feedback Reported
                    </div>
                    <div>
                        {feedback_text}
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
            
            return { "status": 'success', "message": 'Your feedback has been sent to the Data Hub Admins. Thank you for your thoughts!'}
        except smtplib.SMTPException as e:
            return { "status": 'success', "message": f'Your feedback was not sent. Please try again. Error: {e}'}





























