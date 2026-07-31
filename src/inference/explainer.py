"""
=================================
Credit Risk AI Platform
SHAP Explainability Engine
---------------------------------
Provides

• Global explanations
• Local explanations
• Waterfall plots
• Summary plots
• Feature ranking
==================================
"""

import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.data.loader import (
    load_champion_model,
    load_shap_explainer,
    load_matrix_d_feature_names)

from src.config.feature_labels import FEATURE_LABELS

from src.inference.preprocessor import preprocess_application

from src.inference.predictor import make_prediction

# LOAD OBJECTS
# __________________________________________________________

model = load_champion_model()

explainer = shap.TreeExplainer(model)

feature_names = load_matrix_d_feature_names()["Feature_Name"].tolist()

# COMPUTE SHAP VALUES
# _________________________________________________________

def calculate_shap_values(matrix_D):
    
    """
    Calculate SHAP values.
    """
    return explainer(matrix_D)
    

# Explain prediction
# _________________________________________________________

def explain_prediction(matrix_D,index=0,top_n=10):
    """
    Returns the top SHAP features for one observation.
    """

    shap_values = calculate_shap_values(matrix_D)

    values = shap_values.values[index]

    explanation = pd.DataFrame({

        "Feature": feature_names,

        "SHAP_Value": values})

    explanation["Absolute"] = explanation["SHAP_Value"].abs()

    explanation.sort_values(

        "Absolute",

        ascending=False,

        inplace=True)

    explanation.reset_index(

        drop=True,

        inplace=True)

    return explanation.head(top_n)
    

# GLOBAL SUMMARY PLOT
# ______________________________________________________

def plot_summary(matrix_D):
    
    """
    SHAP Summary Plot.
    """

    shap_values = calculate_shap_values(matrix_D)

    shap.summary_plot(

        shap_values,

        matrix_D,

        feature_names=feature_names,

        show=False)

    plt.tight_layout()

    plt.show()


# BAR PLOT
# __________________________________________________________

def plot_bar(matrix_D):
    
    """
    Global SHAP bar chart.
    """

    shap_values = calculate_shap_values(matrix_D)

    shap.plots.bar(shap_values,max_display=20)


# FEATURE IMPORTANCE
# ________________________________________________________

def feature_importance(matrix_D):
    
    """
    Ranked SHAP importance.
    """

    shap_values = calculate_shap_values(matrix_D)

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance":

            abs(shap_values.values).mean(axis=0)})

    importance.sort_values(

        "Importance",ascending=False,inplace=True)

    importance.reset_index(drop=True,inplace=True)

    return importance


# WATERFALL PLOT
# _______________________________________________________

def waterfall_plot(

    matrix_D,

    index=0):
    
    """
    Local explanation.
    """

    shap_values = calculate_shap_values(matrix_D)

    shap.plots.waterfall(shap_values[index])


# FORCE PLOT
# ________________________________________________________

def force_plot(

    matrix_D,

    index=0):
    
    """
    Interactive force plot.
    """

    shap_values = calculate_shap_values(matrix_D)

    return shap.force_plot(

        explainer.expected_value,

        shap_values.values[index],

        matrix_D[index].toarray(),

        feature_names=feature_names)
    

# Explain application
# _________________________________________________________

def explain_application(matrix_D, top_n=10):

    """
    Generate SHAP explanation for the current borrower.
    """

    explanation = explain_prediction(matrix_D,index=0,top_n=top_n)

    top_features = []

    for _, row in explanation.iterrows():

        display_name = FEATURE_LABELS.get(row["Feature"],row["Feature"])

        direction = (
            "Increase Default Risk"
            
            if row["SHAP_Value"] > 0
            
            else "Decrease Default Risk")

        top_features.append({

            "feature": display_name,

            "shap_value": round(
                
                float(row["SHAP_Value"]), 4),

            "impact": direction})

    return top_features


# JSON EXPLANATION
# _____________________________________________________________

def explain_prediction_json(

    matrix_D,

    prediction_result,

    top_n=10):

    explanation = explain_prediction(

        matrix_D,

        index=0,

        top_n=top_n)

    top_features = []

    for _, row in explanation.iterrows():

        display_name = FEATURE_LABELS.get(row["Feature"],row["Feature"])

        direction = ("Increase Default Risk"
                     
                     if row["SHAP_Value"] > 0
                     
                     else "Decrease Default Risk")

    top_features.append({

        "feature": display_name,

        "shap_value": round(float(row["SHAP_Value"]), 4),

        "impact": direction})

    response = prediction_result.copy()

    response["top_features"] = top_features

    
    # Round SHAP values for readability
    
    explanation["SHAP_Value"] = explanation["SHAP_Value"].round(6)
    
    explanation["Absolute"] = explanation["Absolute"].round(6)

    # Determine feature impact direction
    
    impact = []

    for value in explanation["SHAP_Value"]:

        if value > 0:
            direction = "Increase Default Risk"
        else:
            direction = "Decrease Default Risk"

        impact.append(direction)

    explanation["Impact"] = impact
    
    return response
    
    
# COMPLETE JSON RESPONSE
# _________________________________________________________________

def explain_api_response(
    
    matrix_D,
    
    prediction,
    
    probability,
    
    index=0,
    
    top_n=10):
    
    """
    Complete response returned by FastAPI.
    """

    return {

        "prediction": int(prediction),

        "default_probability": round(
            float(probability),4),

        "top_features": explain_prediction_json(
            
            matrix_D,
            
            index=index,
            
            top_n=top_n)}



# COMPLETE REPORT
# _____________________________________________________

def generate_report(

    matrix_D,

    index=0):
    
    """
    Full explainability report.
    """

    print("Prediction Explanation")

    print("=" * 50)

    print(explain_prediction(matrix_D,index))

    waterfall_plot(matrix_D,index)


# TEST
# ______________________________________________________

if __name__ == "__main__":

    print("SHAP Explainability Engine Ready")

    print("=" * 40)