from controller.db import get_db
from controller.feedback import FeedbackFunctions
from controller.manager import manager
from sqlalchemy.orm import Session
from models.wfr_database import UserRole, Role, User
from fastapi import Depends, APIRouter
from fastapi import FastAPI, APIRouter, Request, Body, Response, UploadFile, File
from routes.ckan import send_email_via_sparcal


router = APIRouter(
    prefix='/v1/feedback',
    tags = ['Feedback']
)

def send_feedback_email(admin, user, feedback_topic, feedback_text):
    result = send_email_via_sparcal(
        to_email=admin.email,
        subject=f"Feedback for Task Force Data Hub from {user.first_name} {user.last_name}",
        body=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feedback Submission</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 15px 20px;
            border-radius: 6px;
            margin-bottom: 25px;
        }}
        .section {{
            margin-bottom: 25px;
        }}
        .label {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            display: block;
            font-size: 16px;
        }}
        .content {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }}
        ul {{
            list-style-type: none;
            padding-left: 0;
            margin: 0;
        }}
        li {{
            margin-bottom: 10px;
            padding-left: 10px;
        }}
        .field-label {{
            color: #666;
            width: 100px;
            display: inline-block;
        }}
        .field-value {{
            color: #333;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2 style="margin: 0;">Feedback for Task Force Data Hub</h2>
        </div>
        <div class="section">
            <span class="label">User Information</span>
            <div class="content">
                <ul>
                    <li>
                        <span class="field-label">Name:</span>
                        <span class="field-value">{user.first_name} {user.last_name}</span>
                    </li>
                    <li>
                        <span class="field-label">Username:</span>
                        <span class="field-value">{user.username}</span>
                    </li>
                    <li>
                        <span class="field-label">Email:</span>
                        <span class="field-value">{user.email}</span>
                    </li>
                    <li>
                        <span class="field-label">Affiliation:</span>
                        <span class="field-value">{user.affiliation}</span>
                    </li>
                </ul>
            </div>
        </div>
        <div class="section">
            <span class="label">Feedback Topic</span>
            <div class="content">{feedback_topic}</div>
        </div>
        <div class="section">
            <span class="label">Feedback Content</span>
            <div class="content">{feedback_text}</div>
        </div>
    </div>
</body>
</html>
        """
    )
    return result

@router.post("/feedback/{feedback_text}/topic/{feedback_topic}")
async def feedback(feedback_text: str, 
                   feedback_topic: str, 
                   user=Depends(manager), 
                   db: Session=Depends(get_db)):

    admin_users = (
        db.query(User)
        .join(UserRole, User.user_id == UserRole.user_id)
        .join(Role, UserRole.role_id == Role.role_id)
        .filter(Role.name == 'admin')
        .distinct()  # Ensure unique users even if they have multiple roles
        .all()
    )

    for admin in admin_users:
        send_feedback_email(admin, user, feedback_topic, feedback_text)

    return { 
             "status": 'success', 
             "message": 'Your feedback has been sent to the Data Hub Admins. Thank you for your thoughts!'
           } 
 
    # return FeedbackFunctions.send_feedback_email(user, feedback_text, feedback_topic)