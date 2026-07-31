"""
=========================================================
Credit Risk AI Platform
Inference Preprocessor
---------------------------------------------------------
Transforms ONE borrower application into Matrix D
=========================================================
"""
import re
import pandas as pd

from scipy import sparse

from src.data.loader import (
    load_structured_preprocessor,
    load_tfidf_vectorizer,
    get_macro_features)

from src.config.pipeline_config import FEATURE_STORE_DIR


# Default input values
# ___________________________________________________

DEFAULT_VALUES = {

    "loan_amnt": 0,
    "term": "36 months",
    "grade": "C",
    "int_rate": 0.0,
    "annual_inc": 0.0,
    "dti": 0.0,
    "home_ownership": "RENT",
    "title": "",
    "revol_util": 0.0,
    "issue_d": "2018-01"}


# Load trained transformers
# _______________________________________________________

structured_preprocessor = load_structured_preprocessor()

tfidf_vectorizer = load_tfidf_vectorizer()


# Prepare borrower input
# ____________________________________________________________

def prepare_input(data):
    """
    Convert FastAPI LoanApplication request into a
    one-row DataFrame and ensure all required columns exist.
    """

    # Convert Pydantic model to DataFrame
    
    if not isinstance(data, pd.DataFrame):

        # Pydantic v2
        
        if hasattr(data, "model_dump"):
            data = pd.DataFrame([data.model_dump()])

        # Pydantic v1 fallback
        
        elif hasattr(data, "dict"):
            data = pd.DataFrame([data.dict()])

        # Already a dictionary
        
        elif isinstance(data, dict):
            data = pd.DataFrame([data])

        else:
            raise TypeError(
                f"Unsupported input type: {type(data)}")

    # Make a copy
    
    data = data.copy()

    # Add any missing columns
    
    for column, value in DEFAULT_VALUES.items():

        if column not in data.columns:

            data[column] = value

    return data


# Time features
# ---------------------------------------------------

def create_time_features(df):

    df = df.copy()

    df["DATE"] = pd.to_datetime(
        df["issue_d"],
        errors="coerce")

    df["Year"] = df["DATE"].dt.year

    df["Month"] = df["DATE"].dt.month

    df["Quarter"] = df["DATE"].dt.quarter

    df["Month_Name"] = df["DATE"].dt.month_name()

    df["YearMonth"] = (
        df["DATE"]
        .dt.to_period("M")
        .astype(str))

    return df


# Text cleaning
# ____________________________________________________

def clean_title(text):

    if pd.isna(text):

        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z ]",
        " ",
        text)

    text = re.sub(
        r"\s+",
        " ",
        text)

    return text.strip()


# Build Matrix D
# _______________________________________________________

def build_matrix_d(df):


    # Structured features

    structured = structured_preprocessor.transform(df)



    # Text features

    df["clean_title"] = (
        df["title"]
        .apply(clean_title))


    text = tfidf_vectorizer.transform(df["clean_title"])


    # Matrix C

    matrix_c = sparse.hstack(
        [
            structured,
            text
        ],format="csr")


    # Macro features

    issue_date = df["issue_d"].iloc[0]


    macro = get_macro_features(issue_date)


    macro = macro.drop(
        columns=[
            "Year",
            "Month"
        ],
        errors="ignore")


    macro_matrix = sparse.csr_matrix(macro.to_numpy(dtype="float64"))


    # Matrix D

    matrix_d = sparse.hstack(
        [
            matrix_c,
            macro_matrix
        ],format="csr")


    return matrix_d


# Final prediction preprocessing pipeline
# ______________________________________________________________________

def preprocess_application(borrower):

    borrower = prepare_input(borrower)


    matrix_d = build_matrix_d(borrower)


    # Safety check

    expected_features = 91


    if matrix_d.shape[1] != expected_features:

        raise ValueError(
            f"""
Feature mismatch.

Expected Matrix D:{expected_features}

Received:{matrix_d.shape[1]}""")
    
    return matrix_d


# Test
# ________________________________________________________

if __name__ == "__main__":

    print("Inference preprocessor ready")
    print("=" * 50)