
import os
import requests
import json

from datetime import datetime
from controller.manager import manager
from fastapi import Depends, APIRouter, HTTPException
from controller.db import get_db
from sqlalchemy.orm import Session, joinedload
from models.wfr_database import CKANPackageUser, Role, User
from dotenv.main import load_dotenv

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


load_dotenv('fastApi/.env')

CKAN_URL = os.environ.get('ckan_url')
CKAN_API_TOKEN = os.environ.get('ckan_api_token')
ORG = os.environ.get('dataset_org')
send_email_from=os.environ.get('send_email_from')
send_email_password=os.environ.get('send_email_password') 


router = APIRouter(tags=["CKAN"], prefix='/ckan')


def send_email_via_sparcal(to_email, subject, body):                                                                                                                                                                                                                                                                                  
    msg = MIMEMultipart()                                                                                                                                          
    msg['From'] = 'taskforce.datahub@ucsd.edu'                                                                                                                     
    msg['To'] = to_email                                                                                             
    msg['Subject'] = subject                                                                                                                                       
    msg.attach(MIMEText(body, 'html'))                                                                                                                             
                                                                                                                                                                   
    try:                                                                                                                                                           
        server = smtplib.SMTP("outmail.ucsd.edu", 25)                                                                                                              
        # server.set_debuglevel(1)  # Enable debug output                                                                                                            
        server.sendmail('taskforce.datahub@ucsd.edu', to_email, msg.as_string())     
        server.quit()                                                                                                                                              
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


def send_email_via_gmail(to_email, subject, body):

    from_email = send_email_from
    app_password = send_email_password 

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:  # Use SMTP_SSL for secure connection
            server.login(from_email, app_password)
            server.sendmail(from_email, to_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")



def send_approve_email(user, title):
    result = send_email_via_sparcal(
        to_email=user.email,
        subject="Your Dataset Submission to The Task Force Data Hub",
        body=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <p>Dear {user.first_name} {user.last_name},</p>
    <p>Thank you for submitting your dataset, <strong>“{title},”</strong> to the Task Force Data Hub.</p>
    <p>We are pleased to inform you that, after careful evaluation by our reviewers, your dataset meets our acceptance criteria and has been recognized for its high quality. As a result, we are delighted to include it in the Task Force Data Hub Catalog.</p>
    <p>We sincerely appreciate your valuable contribution and hope you will continue to support the Task Force Data Hub by sharing more high-quality datasets in the future.</p>
    <p>Best regards,</p>
    <p>The Task Force Data Hub Team</p>
</body>
</html>
        """
    )
    return result


def send_deny_email(user, title):
    result = send_email_via_sparcal(
        to_email=user.email,
        subject="Your Dataset Submission to The Task Force Data Hub",
        body=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <p>Dear {user.first_name} {user.last_name},</p>
    <p>Thank you for submitting your dataset, <strong>“{title}”</strong>, to the Task Force Data Hub. We appreciate your time and effort in contributing to our platform.</p>
    <p>After a thorough review by our team, we regret to inform you that your dataset does not currently meet the NDP acceptance criteria. While we are unable to include it in the Task Force Data Catalog at this time, we encourage you to review our guidelines and consider making revisions.</p>
    <p>We would be happy to review a revised submission, should you choose to update your dataset in line with our criteria. Your contributions are important to us, and we hope to see more of your work in the future.</p>
    <p>Best regards,</p>
    <p>The Task Force Data Hub Team</p>
</body>
</html>
        """
    )
    return result


def get_package(package_id, api_token):
    """
    Retrieve a CKAN dataset by its package_id using an API token.

    Args:
        package_id (str): The ID of the CKAN dataset to retrieve.
        api_token (str): The CKAN API token for authentication.

    Returns:
        dict: The dataset information as returned by the CKAN API.

    Raises:
        HTTPError: If the CKAN API returns an error.
        ValueError: If the API response is not successful.
    """
    # CKAN API endpoint for retrieving a package
    ckan_api_url = "https://wifire-data.sdsc.edu/api/3/action/package_show"

    # Headers for the API request
    headers = {
        "Authorization": api_token,
        "Content-Type": "application/json",
    }

    # Parameters for the API request
    params = {
        "id": package_id,
    }

    try:
        # Make the request to the CKAN API
        response = requests.get(ckan_api_url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        data = response.json()

        # Check if the CKAN API returned a successful response
        if not data.get("success", False):
            raise ValueError("CKAN API returned an unsuccessful response")

        # Return the dataset information
        return data["result"]
    except requests.exceptions.HTTPError as e:
        raise requests.exceptions.HTTPError(f"CKAN API error: {e.response.text}")
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")


@router.post("/package", include_in_schema=True)
async def save_dataset(package: dict, user=Depends(manager), db:Session=Depends(get_db)):
    try:
        if user == None: 
            raise HTTPException(status_code=500, detail="User Not Found")

        # only admin or publisher can create
        user = db.query(User).options(joinedload(User.roles)).filter(User.user_id == user.user_id).first()
        roles = [ role.name  for role in user.roles ]
        if not "admin" in roles and not "publisher" in roles :
            raise HTTPException(status_code=403, detail="Forbidden: not an admin or a publisher.")

        # CKAN API endpoint for package creation
        ckan_url = f"{CKAN_URL}/api/3/action/package_create"
        
        # Headers including the API key for authorization
        headers = {
            "Content-Type": "application/json",
            "Authorization": CKAN_API_TOKEN
        }
        
        # Make the POST request to CKAN API
        package["owner_org"] = ORG
        package["state"] = "draft"
        response = requests.post(ckan_url, json=package, headers=headers)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
	# Check if the CKAN API returned a success response
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail="Failed to create dataset in CKAN")

        # Create a new CKANPackageUser record
        package_user = CKANPackageUser()
        package_user.user_id = user.user_id  # Assign the user ID
        package_user.package_id = result['result']['id']  # Assign the package ID from CKAN

	# Save the record to the database
        try:
            db.add(package_user)
            db.commit()
        except Exception as db_error:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")

        return {"message": "Dataset created successfully", "package_id": result['result']['id']}
    
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with CKAN: {str(e)}")


@router.delete("/package/{package_id}", include_in_schema=True)
async def delete_dataset(
    package_id: str, 
    user=Depends(manager), 
    db: Session = Depends(get_db)
):
    try:
        # Check if the user is authenticated
        if user is None:
            raise HTTPException(status_code=401, detail="Unauthorized: User not found")

        package_user = db.query(CKANPackageUser).filter(
            CKANPackageUser.package_id == package_id,
            CKANPackageUser.user_id == user.user_id
        ).first()

        # only admin or owner can delete
        user = db.query(User).options(joinedload(User.roles)).filter(User.user_id == user.user_id).first()
        roles = [ role.name  for role in user.roles ]
        if not "admin" in roles :
            # Verify that the authenticated user is the owner of the package
            if not package_user:
                raise HTTPException(status_code=403, detail="Forbidden: You are not the owner of this package")

        # CKAN API endpoint for package deletion
        ckan_url = f"{CKAN_URL}/api/3/action/package_delete"
        
        # Headers including the API key for authorization
        headers = {
            "Content-Type": "application/json",
            "Authorization": CKAN_API_TOKEN
        }
        
        # Payload for the CKAN API request
        payload = {
            "id": package_id  # The ID of the package to delete
        }
        
        # Make the POST request to CKAN API
        response = requests.post(ckan_url, json=payload, headers=headers)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Check if the CKAN API returned a success response
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail="Failed to delete dataset in CKAN")

        # Delete the corresponding CKANPackageUser record from the database
        try:
            db.delete(package_user)
            db.commit()
        except Exception as db_error:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")

        # Return success response
        return {
            "message": "Dataset deleted successfully",
            "package_id": package_id
        }
    
    except requests.RequestException as e:
        # Handle errors from the CKAN API request
        raise HTTPException(status_code=500, detail=f"Error communicating with CKAN: {str(e)}")


@router.put("/package/{package_id}", include_in_schema=True)
async def update_dataset(
    package_id: str, 
    package: dict, 
    user=Depends(manager), 
    db: Session = Depends(get_db)
):
    try:
        # Check if the user is authenticated
        if user is None:
            raise HTTPException(status_code=401, detail="Unauthorized: User not found")

        # Verify that the authenticated user is the owner of the package
        package_user = db.query(CKANPackageUser).filter(
            CKANPackageUser.package_id == package_id,
            CKANPackageUser.user_id == user.user_id
        ).first()

        # Only admin or owner can edit; reload user to make sure roles are loaded
        user = db.query(User).options(joinedload(User.roles)).filter(User.user_id == user.user_id).first()
        roles = [ role.name  for role in user.roles ]    
        if not "admin" in roles:
            if not package_user:
                raise HTTPException(status_code=403, detail="Forbidden: You are not the owner of this package")

        # CKAN API endpoint for package update
        ckan_url = f"{CKAN_URL}/api/3/action/package_update"
        
        # Headers including the API key for authorization
        headers = {
            "Content-Type": "application/json",
            "Authorization": CKAN_API_TOKEN
        }
        
        # Payload for the CKAN API request
        payload = package
        payload["owner_org"] = ORG
        payload["id"] = package_id  # Add the package ID to the payload

        ckan_package = get_package(package_id, CKAN_API_TOKEN)
        if 'state' in payload.keys() and ckan_package['state'] != payload["state"]:
            raise HTTPException(status_code=403, detail="Forbidden: update the state")	

        # Make the POST request to CKAN API
        response = requests.post(ckan_url, json=payload, headers=headers)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Check if the CKAN API returned a success response
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail="Failed to update dataset in CKAN")

        # Update the `updated_at` field in the CKANPackageUser record
        try:
            package_user.updated_at = datetime.utcnow()  # Set the current timestamp
            db.commit()
        except Exception as db_error:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")

        # Return success response
        return {
            "message": "Dataset updated successfully",
            "package_id": package_id,
            "result": result["result"]
        }
    
    except requests.RequestException as e:
        # Handle errors from the CKAN API request
        raise HTTPException(status_code=500, detail=f"Error communicating with CKAN: {str(e)}")


@router.get("/draft-packages")
def get_draft_datasets(start:int=0, rows:int=1000, user=Depends(manager), db: Session = Depends(get_db)):
    """
    Retrieve all draft datasets from the CKAN API using the requests library.
    Returns empty list instead of error when no drafts found.
    """
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Check user roles
    user = db.query(User).options(joinedload(User.roles)).filter(User.user_id == user.user_id).first()
    roles = [role.name for role in user.roles]
    if "approver" not in roles and "admin" not in roles:
        raise HTTPException(status_code=403, detail="Forbidden: requires approver or admin role")

    ckan_url = f"{CKAN_URL}/api/3/action/package_search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": CKAN_API_TOKEN
    }
    params = {
        "q": "state:draft",
        "include_drafts": "true",
        "start": start,
        "rows": rows
    }

    try:
        response = requests.get(ckan_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        all_ckan_users = db.query(CKANPackageUser).options(joinedload(CKANPackageUser.user)).all()
        datasets = data.get("result", {"count": 0, "results": []}) 
        for dataset in datasets["results"]:
           ckan_package_user = next((cpu for cpu in all_ckan_users if cpu.package_id == dataset["id"]), None)
           if ckan_package_user:
               dataset['creator'] = {
                   'fullname': f"{ckan_package_user.user.first_name} {ckan_package_user.user.last_name}",
                   'email': f"{ckan_package_user.user.email}",
                   'affiliation': f"{ckan_package_user.user.affiliation}"
               } 

        if not data.get("success", False):
            raise HTTPException(status_code=500, detail="CKAN API returned unsuccessful response")

        # Ensure consistent return structure even when empty
        return data.get("result", {"count": 0, "results": []})

    except requests.exceptions.HTTPError as e:
        # Handle other 4xx/5xx errors
        if e.response.status_code == 404:
            return {"count": 0, "results": []}
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"CKAN API error: {e.response.text[:200]}"  # Truncate long errors
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/my-draft-packages")
def get_my_draft_datasets(
    start:int=0, 
    rows:int=1000,
    user=Depends(manager), 
    db: Session = Depends(get_db)  
):
    """
    Retrieve draft datasets from CKAN API filtered by packages associated with the current user
    """
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # 1. Get user's package IDs from database
        user_package_ids = db.query(CKANPackageUser.package_id)\
                            .filter(CKANPackageUser.user_id == user.user_id)\
                            .all()
        user_package_ids = [p[0] for p in user_package_ids]  # Extract strings from tuples

        # 2. Get all drafts from CKAN
        ckan_url = f"{CKAN_URL}/api/3/action/package_search"
        headers = {"Authorization": CKAN_API_TOKEN}
        
        response = requests.get(
            ckan_url,
            headers=headers,
            params={"q": "state:draft", "include_drafts": "true",  "start": start, "rows": rows}
        )

        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise HTTPException(status_code=502, detail="CKAN API response error")

        # 3. Filter results to only user's packages
        all_drafts = data["result"].get("results", [])
        user_drafts = [
            draft for draft in all_drafts 
            if draft["id"] in user_package_ids
        ]

        for dataset in user_drafts:
            dataset['creator'] = {
                'fullname': f"{user.first_name} {user.last_name}",
                 'email': f"{user.email}",
                 'affiliation': f"{user.affiliation}"
            } 

        # Return filtered results
        return {
            "count": len(user_drafts),
            "results": user_drafts
        }

    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"CKAN API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/approve-draft/{package_id}")
def approve_draft_package(
    package_id: str, 
    user=Depends(manager),
    db: Session = Depends(get_db) 
):
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Check if the user has the approver or admin role
    user = db.query(User).options(joinedload(User.roles)).filter(User.user_id == user.user_id).first()
    roles = [role.name for role in user.roles]
    if "approver" not in roles and "admin" not in roles:
        raise HTTPException(status_code=403, detail="Forbidden: User does not have the approver or admin role.")

    # CKAN API endpoint for updating a package
    ckan_url = f"{CKAN_URL}/api/3/action/package_patch"

    # Headers including the API key for authorization
    headers = {
        "Content-Type": "application/json",
        "Authorization": CKAN_API_TOKEN,
    }

    ckan_package = get_package(package_id, CKAN_API_TOKEN)
    if ckan_package['state'] != 'draft':
         raise HTTPException(status_code=403, detail=f"Forbidden: approve a package with the state \'{ckan_package['state']}\'.")	

    # Payload to update the package state to 'public'
    payload = {
        "id": package_id,
        "state": "active",  # CKAN uses 'active' for public datasets
    }

    try:
        # Make the request to the CKAN API
        response = requests.post(ckan_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Check if the CKAN API returned a successful response
        if not data.get("success", False):
            raise HTTPException(status_code=500, detail="CKAN API returned an unsuccessful response")

        # Send email
        package_user = db.query(CKANPackageUser)\
            .filter(CKANPackageUser.package_id == package_id)\
            .first()
    
        submitter = None
        if package_user:
            submitter = package_user.user
            send_approve_email(submitter, data["result"]["title"])

        # Return the updated package
        return data["result"]
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"CKAN API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.delete("/deny-draft/{package_id}")
def deny_draft_package(
    package_id: str, 
    user=Depends(manager),
    db: Session = Depends(get_db)  # Add database session dependency
):
    """
    Deny a draft package by deleting it from CKAN and the database.
    """
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Check if the user has the approver or admin role
    user = db.query(User).options(joinedload(User.roles)).filter(User.user_id == user.user_id).first()
    roles = [role.name for role in user.roles]
    if "approver" not in roles and "admin" not in roles:
        raise HTTPException(status_code=403, detail="Forbidden: User does not have the approver or admin role.")

    # CKAN API endpoint for deleting a package
    ckan_url = f"{CKAN_URL}/api/3/action/package_delete"

    # Headers including the API key for authorization
    headers = {
        "Content-Type": "application/json",
        "Authorization": CKAN_API_TOKEN,
    }

    ckan_package = get_package(package_id, CKAN_API_TOKEN)
    if ckan_package['state'] != 'draft':
         raise HTTPException(status_code=403, detail=f"Forbidden: deny a package with the state \'{ckan_package['state']}\'.")	

    # Payload to delete the package
    payload = {
        "id": package_id,
    }

    try:
        # Step 1: Delete the package from CKAN
        response = requests.post(ckan_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Check if the CKAN API returned a successful response
        if not data.get("success", False) and not ("details" in data.keys() and data["details"] == "An error occurred: 'NoneType' object is not subscriptable"):
            raise HTTPException(status_code=500, detail="CKAN API returned an unsuccessful response")

        # Send email
        package_user = db.query(CKANPackageUser)\
            .filter(CKANPackageUser.package_id == package_id)\
            .first()
    
        submitter = None
        if package_user:
            submitter = package_user.user
            send_deny_email(submitter, ckan_package["title"])

        # Step 2: Delete the package from the database table (CKANPackageUser)
        db.query(CKANPackageUser)\
          .filter(CKANPackageUser.package_id == package_id)\
          .delete()
        db.commit()  # Commit the transaction

        # Return a success message
        return {"message": f"Package {package_id} deleted successfully from CKAN and the database"}
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"CKAN API error: {e.response.text}")
    except Exception as e:
        db.rollback()  # Rollback the transaction in case of an error
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




