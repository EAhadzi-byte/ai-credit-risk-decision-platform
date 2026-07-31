"""
=========================================================
Credit Risk AI Platform
Inference Pipeline
=========================================================
"""

from src.inference.preprocessor import preprocess_application
from src.inference.predictor import make_prediction
from src.inference.explainer import explain_prediction_json


def run_prediction_pipeline(application):
    """
    Complete inference pipeline.

    Parameters
    ----------
    application : pandas.DataFrame
        Single borrower application.

    Returns
    -------
    dict
        Prediction response.
    """

    # Step 1
    
    matrix_D = preprocess_application(application)

    # Step 2
    
    prediction = make_prediction(matrix_D)

    return prediction


def run_explainability_pipeline(application):
    """
    Complete explainability pipeline.

    Parameters
    ----------
    application : pandas.DataFrame
        Single borrower application.

    Returns
    -------
    dict
        Prediction + SHAP explanation.
    """

    # Step 1
    
    matrix_D = preprocess_application(application)

    # Step 2
    
    prediction = make_prediction(matrix_D)

    # Step 3
    
    explanation = explain_prediction_json(matrix_D,prediction)

    return explanation