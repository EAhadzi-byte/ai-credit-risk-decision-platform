"""
=========================================================
Credit Risk AI Platform
Reusable Data Loader
---------------------------------------------------------
Loads datasets, feature matrices, models and artefacts.
=========================================================
"""

from pathlib import Path
import joblib
import pandas as pd
from scipy import sparse
from datetime import datetime

import pandas as pd

from src.config.pipeline_config import (
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_STORE_DIR,
    MODEL_DIR,
    REGISTRY_DIR,
    MATRIX_A_TRAIN,
    MATRIX_A_TEST,
    MATRIX_C_TRAIN,
    MATRIX_C_TEST,
    MATRIX_D_TRAIN,
    MATRIX_D_TEST,
    MATRIX_D_FEATURES,
    Y_TRAIN,
    Y_TEST,
    CHAMPION_MODEL,
    SHAP_EXPLAINER,
    STRUCTURED_PREPROCESSOR,
    TFIDF_VECTORIZER,)


# GENERIC LOADERS
# ______________________________________________________

def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load any CSV file.
    """
    return pd.read_csv(filepath)


def load_parquet(filepath: Path) -> pd.DataFrame:
    """
    Load any parquet dataset.
    """
    return pd.read_parquet(filepath)


MODELS_DIR = Path("/Users/emmanuelahadzi/models")

def load_pickle(filepath):
    filepath = Path(filepath)

    # Try the provided path first
    if filepath.exists():
        return joblib.load(filepath)

    # Search the entire models directory
    matches = list(MODELS_DIR.rglob(filepath.name))

    if not matches:
        raise FileNotFoundError(
            f"{filepath.name} not found anywhere in {MODELS_DIR}")

    print(f"Found {filepath.name} at {matches[0]}")
    return joblib.load(matches[0])


def load_sparse(filepath: Path):
    """
    Load sparse matrix.
    """
    return sparse.load_npz(filepath)


# RAW DATA
# ________________________________________________________

def load_raw_dataset(filename: str):

    return load_csv(RAW_DATA_DIR / filename)


# PROCESSED DATA
# ________________________________________________________

def load_processed_dataset(filename: str):

    path = PROCESSED_DATA_DIR / filename

    if filename.endswith(".csv"):

        return load_csv(path)

    if filename.endswith(".parquet"):

        return load_parquet(path)

    raise ValueError("Unsupported file format.")


# FEATURE STORE
# ________________________________________________________

def load_feature_store(filename: str):

    path = FEATURE_STORE_DIR / filename

    if filename.endswith(".csv"):

        return load_csv(path)

    if filename.endswith(".parquet"):

        return load_parquet(path)

    if filename.endswith(".npz"):

        return load_sparse(path)

    raise ValueError("Unsupported feature file.")


# MATRIX LOADERS
# __________________________________________________________

def load_matrix_a(train=True):

    return load_pickle(

        MATRIX_A_TRAIN if train else MATRIX_A_TEST)


def load_matrix_c(train=True):

    return load_sparse(

        MATRIX_C_TRAIN if train else MATRIX_C_TEST)


def load_matrix_d(train=True):

    return load_sparse(

        MATRIX_D_TRAIN if train else MATRIX_D_TEST)


# LABELS
# __________________________________________________________

def load_target(train=True):

    return load_pickle(

        Y_TRAIN if train else Y_TEST)


# FEATURE NAMES
# __________________________________________________________

def load_matrix_d_feature_names():

    return load_csv(

        MATRIX_D_FEATURES)


# PREPROCESSORS
# ___________________________________________________________

def load_structured_preprocessor():

    return load_pickle(

        STRUCTURED_PREPROCESSOR)


def load_tfidf_vectorizer():

    return load_pickle(

        TFIDF_VECTORIZER)


# MODELS
# ___________________________________________________________

def load_champion_model():

    return load_pickle(

        CHAMPION_MODEL)


def load_shap_explainer():

    return load_pickle(

        SHAP_EXPLAINER)


# REGISTRY
# ________________________________________________________

def load_registry_file(filename: str):

    path = REGISTRY_DIR / filename

    if filename.endswith(".csv"):

        return load_csv(path)

    if filename.endswith(".json"):

        import json

        with open(path, "r") as f:

            return json.load(f)

    raise ValueError("Unsupported registry file.")


# INFORMATION
# ________________________________________________________

def describe_feature_store():

    files = sorted(

        FEATURE_STORE_DIR.glob("*"))

    summary = []

    for file in files:

        summary.append(

            {

                "File": file.name,

                "Size_MB": round(

                    file.stat().st_size / 1024 / 1024,2,),})

    return pd.DataFrame(summary)


# MACROECONOMIC FEATURE LOADER
# _________________________________________________________________

def get_macro_features(issue_date):

    macro_path = (FEATURE_STORE_DIR /"macroeconomic_feature_store.parquet")


    # Load production macro feature store

    macro_df = pd.read_parquet(macro_path)


    # Convert dates

    macro_df["DATE"] = pd.to_datetime(macro_df["DATE"])

    issue_date = pd.to_datetime(issue_date)

    # Create matching month

    issue_month = (issue_date.to_period("M").to_timestamp())

    # Find matching month

    macro_row = macro_df[macro_df["DATE"] == issue_month]
    
    if macro_row.empty:

        raise ValueError(f"No macroeconomic data found for {issue_month}")
  
    # Remove date column
    
    macro_row = macro_row.drop(columns=["DATE"],errors="ignore")

    # Keep only numeric columns
    
    macro_row = macro_row.select_dtypes(include=["number"])

    # Convert to float64
    
    macro_row = macro_row.astype("float64")

    return macro_row

# TEST
# ___________________________________________________________

if __name__ == "__main__":

    print("Reusable Loader")

    print("=" * 60)

    print(load_matrix_d(train=True).shape)

    print(load_target(train=True).shape)

    print(load_matrix_d_feature_names().head())