"""
=========================================================
Charts
=========================================================
"""

import matplotlib.pyplot as plt

import pandas as pd

import streamlit as st

from feature_labels import FEATURE_LABELS

def shap_bar_chart(explanation):
    """
    Display SHAP feature importance.

    Expected API format

    {
        "top_features":[
            {
                "feature":"Interest Rate",
                "shap_value":0.82}]}
    """

    top_features = explanation.get("top_features", [])

    if len(top_features) == 0:

        st.warning("No SHAP explanation available.")

        return pd.DataFrame(columns=["feature", "shap_value"])

    shap_df = pd.DataFrame(top_features)

    shap_df["feature"] = shap_df["feature"].replace(FEATURE_LABELS)

    # Make sure required columns exist

    if "feature" not in shap_df.columns:

        shap_df["feature"] = "Unknown"

    if "shap_value" not in shap_df.columns:

        shap_df["shap_value"] = 0.0

    shap_df = shap_df.sort_values(

        by="shap_value",

        key=lambda x: x.abs(),

        ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [

        "#d62728" if value > 0 else "#2ca02c"

        for value in shap_df["shap_value"]]

    ax.barh(

        shap_df["feature"],

        shap_df["shap_value"],

        color=colors)

    ax.invert_yaxis()

    ax.set_xlabel("SHAP Contribution")

    ax.set_ylabel("Feature")

    ax.set_title("Top Factors Driving the Prediction")

    st.pyplot(fig)

    return shap_df