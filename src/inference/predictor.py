"""
=========================================================
Credit Risk AI Platform
Prediction Engine
---------------------------------------------------------
Loads champion model and generates predictions
=========================================================
"""

import pandas as pd

from datetime import datetime


from src.data.loader import load_champion_model

# LOAD MODEL
# ________________________________________________________

model = load_champion_model()


# PREDICT DEFAULT PROBABILITY
# _________________________________________________________


def predict_probability(matrix_D):

    """
    Return probability of default.

    Input:
        Matrix D format

    Output:
        probabilities -> [[P(0), P(1)]]
    """
    
    probabilities = model.predict_proba(matrix_D)


    return probabilities


# CLASS PREDICTION
# _________________________________________________________


def predict_class(

        matrix_D,

        threshold=0.5):

    """
    Convert probability into class.
    """
    
    probability = predict_probability(matrix_D)
    
    prediction = (probability >= threshold).astype(int)
    
    return prediction


# COMPLETE PREDICTION
# ___________________________________________________________

def make_prediction(matrix_D, threshold=0.5):
    
    """
    Production prediction response.
    """

    probabilities = predict_probability(matrix_D)

    probability_low = float(probabilities[0][0])

    probability_high = float(probabilities[0][1])

    prediction = int(probability_high >= threshold)

    if prediction == 1:
        confidence = probability_high
        
        risk_class = "High Risk"
    else:
        confidence = probability_low
        
        risk_class = "Low Risk"

    response = {

        "prediction": prediction,

        "default_probability": round(probability_high, 4),

        "confidence": round(confidence, 4),

        "risk_class": risk_class,

        "timestamp": datetime.now().isoformat()}

    return response

# TEST
# _______________________________________________________

if __name__ == "__main__":

    print("Prediction Engine Ready")

    print("="*50)