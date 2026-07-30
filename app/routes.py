"""
=========================================================
FastAPI Routes
=========================================================
"""

from fastapi import APIRouter, HTTPException

from app.schemas import ExplainResponse

from app.schemas import (

LoanApplication,

PredictionResponse)

from app.services import prediction_service

import app.services as services

router = APIRouter()


# HEALTH CHECK
# ____________________________________________________________

@router.get("/health")

def health():

    return {

        "status": "healthy",

        "api": "Explainable AI Credit Risk Platform"}

# HOME
# _______________________________________________________________

@router.get("/")

def home():

    return {

        "message": "Explainable AI Credit Risk Platform",

        "version": "1.0"}


# PREDICT
# _______________________________________________________________

@router.post(

    "/predict",

    response_model=PredictionResponse)

def predict(

    application: LoanApplication):

    try:

        result = services.prediction_service(application)

        return result

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e))

@router.post(
    "/explain",
    response_model=ExplainResponse)


# explain
# _______________________________________________________________

def explain(
    application: LoanApplication
):

    try:

        result = result = services.explain_service(application)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e))

    features = explain_application(matrix_D)

    prediction["top_features"] = features

    return prediction