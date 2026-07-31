"""
=========================================================
Credit Risk AI Platform
Pipeline Configuration
---------------------------------------------------------
Author : Emmanuel Ahadzi
Project: Explainable AI Credit Risk Decision Platform
=========================================================
"""

from pathlib import Path

# PROJECT ROOT
# _______________________________________

# src/config/pipeline_config.py
#               ↑
# parents[0] -> config
# parents[1] -> src
# parents[2] -> project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# DIRECTORY STRUCTURE
# ________________________________________________

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

FEATURE_STORE_DIR = DATA_DIR / "feature_store"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

LOG_DIR = PROJECT_ROOT / "logs"

REGISTRY_DIR = MODEL_DIR / "registry"


# RANDOM STATE
# ________________________________________________________

RANDOM_STATE = 42

# 
# TRAIN / TEST SPLIT
# _______________________________________________________

TEST_SIZE = 0.20

# TF-IDF SETTINGS
# _______________________________________________________

MAX_FEATURES = 500

MIN_DF = 5

NGRAM_RANGE = (1, 2)

STOP_WORDS = "english"

SUBLINEAR_TF = True

MATRIX_D_FEATURE_COUNT = 91

# XGBOOST DEFAULT SETTINGS
# _____________________________________________________

XGB_DEFAULT_PARAMS = {

    "objective": "binary:logistic",

    "eval_metric": "auc",

    "random_state": RANDOM_STATE,

    "tree_method": "hist",

    "n_jobs": -1}


# FILE PATHS
# _________________________________________________________________

# ---------- Matrix A ----------

MATRIX_A_TRAIN = FEATURE_STORE_DIR / "matrix_A_train.pkl"

MATRIX_A_TEST = FEATURE_STORE_DIR / "matrix_A_test.pkl"

# ---------- Matrix C ----------

MATRIX_C_TRAIN = FEATURE_STORE_DIR / "matrix_C_train.npz"

MATRIX_C_TEST = FEATURE_STORE_DIR / "matrix_C_test.npz"

# ---------- Matrix D ----------

MATRIX_D_TRAIN = FEATURE_STORE_DIR / "matrix_D_train.npz"

MATRIX_D_TEST = FEATURE_STORE_DIR / "matrix_D_test.npz"

MATRIX_D_FEATURES = FEATURE_STORE_DIR / "matrix_D_feature_names.csv"

# ---------- Labels ----------

Y_TRAIN = FEATURE_STORE_DIR / "y_train.pkl"

Y_TEST = FEATURE_STORE_DIR / "y_test.pkl"

# ---------- Production Models ----------

CHAMPION_MODEL = MODEL_DIR / "champion_xgboost.pkl"

SHAP_EXPLAINER = MODEL_DIR / "shap_explainer.pkl"

STRUCTURED_PREPROCESSOR = MODEL_DIR / "structured_preprocessor.pkl"

TFIDF_VECTORIZER = MODEL_DIR / "preprocessing" / "tfidf_vectorizer.pkl"


# GOVERNANCE FILES
# __________________________________________________________________________

MODEL_METADATA = REGISTRY_DIR / "model_metadata.json"

PIPELINE_CONFIG = REGISTRY_DIR/ "pipeline_config.json"

MODEL_CHECKSUM = REGISTRY_DIR / "checksums.json"

MODEL_CARD = REGISTRY_DIR / "model_card.md"


# CREATE DIRECTORIES AUTOMATICALLY
# ________________________________________________________________________

DIRECTORIES = [

    RAW_DATA_DIR,

    INTERIM_DATA_DIR,

    PROCESSED_DATA_DIR,

    FEATURE_STORE_DIR,

    MODEL_DIR,

    REPORT_DIR,

    FIGURE_DIR,

    LOG_DIR,

    REGISTRY_DIR]

for directory in DIRECTORIES:

    directory.mkdir(parents=True, exist_ok=True)
    

# DISPLAY
# ______________________________________________________________

if __name__ == "__main__":

    print("Credit Risk Platform Configuration")

    print("=" * 50)

    print(f"Project Root : {PROJECT_ROOT}")

    print(f"Data Folder  : {DATA_DIR}")

    print(f"Models Folder: {MODEL_DIR}")

    print(f"Reports      : {REPORT_DIR}")

    print(f"Registry     : {REGISTRY_DIR}")