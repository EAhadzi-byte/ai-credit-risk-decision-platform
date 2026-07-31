"""
=========================================================
Credit Risk AI Platform
Reusable Model Trainer
---------------------------------------------------------
Supports:

• Logistic Regression
• XGBoost
• SMOTE
• Hyperparameter Tuning
• Model Saving
=========================================================
"""

import joblib

import pandas as pd

from pathlib import Path

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import RandomizedSearchCV

from sklearn.base import clone

from sklearn.metrics import roc_auc_score

from src.config.pipeline_config import (RANDOM_STATE,MODEL_DIR)

from xgboost import XGBClassifier

# TRAIN MODEL
# ___________________________________________________

def train_model(
    
    model,
    
    X_train,
    
    y_train,
    
    use_smote=None,
    
    scale_pos_weight=None):

    """
    Train any sklearn-compatible model.
    """
    model = clone(model)

    # XGBoost
    # ------------------------------------------------------

    if isinstance(model, XGBClassifier):

        if scale_pos_weight is not None:

            model.set_params(scale_pos_weight=scale_pos_weight)

        model.fit(X_train,y_train)

        return model

    
    # Logistic Regression (or any model using SMOTE)
    # ------------------------------------------------------

    if use_smote is None:
        use_smote = False

    if use_smote:

        pipeline = Pipeline(

            [(

                    "smote",

                    SMOTE(
                        random_state=RANDOM_STATE)),

                (

                    "model",

                    model)])

        pipeline.fit(X_train,y_train)

        return pipeline

    # Standard Training
    # ------------------------------------------------------

    model.fit(X_train,y_train)

    return model


# RANDOM SEARCH
# _________________________________________________________

def hyperparameter_search(

    estimator,

    parameter_grid,

    X_train,

    y_train,

    cv,

    n_iter=30,

    scoring="roc_auc"):
    """
    RandomizedSearchCV wrapper.
    """

    search = RandomizedSearchCV(

        estimator=estimator,

        param_distributions=parameter_grid,

        n_iter=n_iter,

        scoring=scoring,

        cv=cv,

        random_state=RANDOM_STATE,

        verbose=2,

        n_jobs=-1,

        return_train_score=True)

    search.fit(X_train,y_train)

    return search

# SAVE MODEL
# ____________________________________________

def save_model(

    model,

    filename):
    
    """
    Save trained model.
    """

    filepath = MODEL_DIR / filename

    joblib.dump(

        model,

        filepath)

    print(f"Model saved to {filepath}")


# LOAD MODEL
# _________________________________________________

def load_model(

    filename):

    filepath = MODEL_DIR / filename

    return joblib.load(filepath)


# BEST MODEL
# _______________________________________________

def train_best_model(

    search,

    X_train,

    y_train):
    """
    Refit best estimator.
    """

    best_model = search.best_estimator_

    best_model.fit(

        X_train,

        y_train)

    return best_model


# FEATURE IMPORTANCE
# ________________________________________________

def get_feature_importance(

    model,

    feature_names):
    
    """
    Returns ranked feature importance.
    """

    importance = pd.DataFrame(

        {

            "Feature": feature_names,

            "Importance": model.feature_importances_})

    importance = importance.sort_values(

        "Importance",

        ascending=False)

    importance.reset_index(drop=True,inplace=True)

    return importance


# MODEL SUMMARY
# ______________________________________________________

def model_summary(

    model,

    X_test,

    y_test):
    
    """
    Quick ROC-AUC.
    """

    probability = model.predict_proba(

        X_test)[:,1]

    auc = roc_auc_score(

        y_test,

        probability)

    return {

        "Model":type(model).__name__,

        "ROC_AUC":round(

            auc,6)}

# TEST
# _______________________________________________________

if __name__ == "__main__":

    print("Reusable Model Trainer Ready")

    print("="*50)