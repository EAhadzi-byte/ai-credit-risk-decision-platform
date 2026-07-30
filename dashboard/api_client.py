"""
=========================================================
Credit Risk AI Platform
API Client
=========================================================
"""

import requests


API_URL = "http://127.0.0.1:8000"


# PREDICTION REQUEST
# _____________________________________________________

def predict_loan(application):

    response = requests.post(

        f"{API_URL}/predict",

        json=application)

    response.raise_for_status()

    return response.json()


# EXPLANATION REQUEST
# _____________________________________________________

def explain_loan(application):

    response = requests.post(

        f"{API_URL}/explain",

        json=application)

    response.raise_for_status()

    return response.json()