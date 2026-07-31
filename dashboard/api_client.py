"""
Credit Risk AI Platform
API Client
___________________________________________________
"""

import requests
from app.schemas import LoanApplication
from app.services import prediction_service
from app.services import explanation_service

API_URL = "http://127.0.0.1:8000"

# Check whether FastAPI is running
# _________________________________________________

def api_available():

    try:

        response = requests.get(f"{API_URL}/health",timeout=1)

        return response.status_code == 200

    except:

        return False


# PREDICTION REQUEST
# _____________________________________________________

def predict_loan(application):
    
    if api_available():

        response = requests.post(f"{API_URL}/predict",json=application)

        response.raise_for_status()

        return response.json()
        
    print("FastAPI not available. Using local prediction service...")

    application = LoanApplication(**application)

    return prediction_service(application)


# EXPLANATION REQUEST
# _____________________________________________________

def explain_loan(application):

    if api_available():

        response = requests.post(f"{API_URL}/explain",json=application)

        response.raise_for_status()

        return response.json()

    print("FastAPI not available. Using local explanation service...")

    application = LoanApplication(**application)

    return explanation_service(application)
    