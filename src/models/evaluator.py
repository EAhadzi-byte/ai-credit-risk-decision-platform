"""
=========================================================
Credit Risk AI Platform
Reusable Model Evaluator
---------------------------------------------------------
Supports

• Accuracy
• Precision
• Recall
• F1
• ROC-AUC
• Confusion Matrix
• ROC Curve
• Precision-Recall Curve
=========================================================
"""

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay)

# EVALUATION METRICS
# _________________________________________

def evaluate_model(

    model,

    X_test,

    y_test,

    model_name="Model",

    dataset="Matrix"):
    
    """
    Returns evaluation metrics.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:,1]

    results = {

        "Model":model_name,

        "Dataset":dataset,

        "Accuracy":accuracy_score(

            y_test,

            predictions),

        "Precision":precision_score(

            y_test,

            predictions),

        "Recall":recall_score(

            y_test,

            predictions),

        "F1":f1_score(

            y_test,

            predictions),

        "ROC_AUC":roc_auc_score(

            y_test,

            probabilities)}

    return pd.DataFrame([results])


# CLASSIFICATION REPORT
# ______________________________________________________

def classification_report_df(

    model,

    X_test,

    y_test):
    
    """
    Returns classification report as DataFrame.
    """

    prediction = model.predict(X_test)

    report = classification_report(

        y_test,

        prediction,

        output_dict=True)

    return pd.DataFrame(report).transpose()


# CONFUSION MATRIX
# _______________________________________________________

def plot_confusion_matrix(

    model,

    X_test,

    y_test):
    
    """
    Plot confusion matrix.
    """

    prediction = model.predict(X_test)

    cm = confusion_matrix(y_test,prediction)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.show()


# ROC CURVE
# __________________________________________________

def plot_roc_curve(

    model,

    X_test,

    y_test):
    
    """
    ROC Curve.
    """

    RocCurveDisplay.from_estimator(

        model,

        X_test,

        y_test)

    plt.title("ROC Curve")

    plt.tight_layout()

    plt.show()


# PRECISION RECALL CURVE
# ______________________________________________________

def plot_precision_recall_curve(

    model,

    X_test,

    y_test):
    
    """
    Precision Recall Curve.
    """

    PrecisionRecallDisplay.from_estimator(

        model,

        X_test,

        y_test)

    plt.title("Precision Recall Curve")

    plt.tight_layout()

    plt.show()


# COMPLETE EVALUATION
# ____________________________________________________

def full_evaluation(

    model,

    X_test,

    y_test,

    model_name="Model",

    dataset="Matrix"):
    
    """
    Performs full evaluation.
    """

    results = evaluate_model(

        model,

        X_test,

        y_test,

        model_name,

        dataset)

    print(results)

    plot_confusion_matrix(

        model,

        X_test,

        y_test)

    plot_roc_curve(

        model,

        X_test,

        y_test)

    plot_precision_recall_curve(

        model,

        X_test,

        y_test)

    return results


# COMPARE MODELS
# __________________________________________________

def compare_models(

    *results):
    
    """
    Combine evaluation tables.
    """

    return pd.concat(

        results,

        ignore_index=True)


# TEST
# _________________________________________________

if __name__ == "__main__":

    print("Reusable Evaluator Ready")

    print("="*50)