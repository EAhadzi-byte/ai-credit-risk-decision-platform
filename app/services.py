"""
=========================================================
Explainable AI Credit Risk Platform
Application Services

Connects FastAPI endpoints with ML inference pipeline.
=========================================================
"""

import pandas as pd

from src.inference.preprocessor import (preprocess_application)

from src.inference.predictor import (make_prediction)

from src.inference.explainer import (explain_prediction_json)

from src.inference.explainer import explain_application

# PREDICTION SERVICE
# _______________________________________________________________

def prediction_service(application):

    # Convert Pydantic object into dataframe

    borrower_df = pd.DataFrame([application.dict()])


    # Transform borrower data

    matrix_D = preprocess_application(borrower_df)

    # Generate prediction

    result = make_prediction(matrix_D)

    return result
    

def explain_service(application):
    """
    Predict loan risk and return SHAP explanations.
    """

    matrix_D = preprocess_application(application)

    prediction = make_prediction(matrix_D)

    prediction["top_features"] = explain_application(matrix_D)

    return prediction
    
# EXPLANATION SERVICE
# ___________________________________________________________________

def explanation_service(application):

    borrower_df = pd.DataFrame([application.dict()])


    matrix_D = preprocess_application(borrower_df)


    prediction = make_prediction(matrix_D)


    explanation = explain_prediction_json(matrix_D,prediction)

    return explanation



