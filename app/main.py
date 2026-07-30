"""
====================================================
Explainable AI Credit Risk Platform
FastAPI Application Entry Point
====================================================
"""

from fastapi import FastAPI

from app.routes import router


app = FastAPI(

    title="Explainable AI Credit Risk Platform",

    description="""
    AI-powered credit risk assessment platform.

    Features:

    - XGBoost credit risk prediction
    - NLP loan text intelligence
    - Macroeconomic feature integration
    - SHAP explainability
    """,

    version="1.0.0")


# Register all routes

app.include_router(router)