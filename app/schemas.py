"""
=========================================================
Credit Risk AI Platform
FastAPI Schemas
---------------------------------------------------------
Request and Response Models
=========================================================
"""

from typing import List

from pydantic import BaseModel, Field

# LOAN APPLICATION REQUEST
# __________________________________________________________

class LoanApplication(BaseModel):

    """
    Raw borrower information supplied to the API.
    Mirrors preprocessing pipeline inputs.
    """

    loan_amnt: float = Field(
        ...,
        description="Requested loan amount")

    term: str = Field(
        ...,
        description="Loan term")

    grade: str = Field(
        ...,
        description="Lending Club grade")

    int_rate: float = Field(
        ...,
        description="Interest rate")

    annual_inc: float = Field(
        ...,
        description="Annual income")

    dti: float = Field(
        ...,
        description="Debt-to-income ratio")

    home_ownership: str = Field(
        ...,
        description="Home Ownership")

    revol_util: float = Field(
        ...,
        description="Revolving credit utilisation")

    title: str = Field(
        ...,
        description="Loan title")

    issue_d: str = Field(
        ...,
        description="Loan issue date")


    class Config:

        json_schema_extra = {

            "example": {

                "loan_amnt": 15000,

                "term": "36 months",

                "grade": "B",

                "int_rate": 11.99,

                "annual_inc": 65000,

                "dti": 18.5,

                "home_ownership": "MORTGAGE",

                "revol_util": 42.3,

                "title": "Debt consolidation",

                "issue_d": "2018-03-01"}}


# SHAP FEATURE CONTRIBUTION
# ______________________________________________________

class FeatureContribution(BaseModel):

    """
    Individual SHAP contribution.
    """

    feature: str

    shap_value: float


# PREDICTION RESPONSE
# __________________________________________________________

class PredictionResponse(BaseModel):

    """
    Prediction returned by /predict endpoint.
    """

    prediction: int

    default_probability: float

    confidence: float

    risk_class: str

    timestamp: str

class FeatureExplanation(BaseModel):
    feature: str
    shap_value: float

class ExplainResponse(PredictionResponse):
    top_features: List[FeatureExplanation]

# EXPLANATION RESPONSE
# ____________________________________________________________

class FeatureExplanation(BaseModel):
    
    feature: str
    
    shap_value: float

class ExplainResponse(BaseModel):

    """
    Prediction + SHAP explanation returned by /explain.
    """

    prediction: int

    default_probability: float

    confidence: float

    risk_class: str

    top_features: List[FeatureContribution]

    timestamp: str

    