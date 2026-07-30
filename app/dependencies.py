"""
=========================================================
FastAPI Dependencies
=========================================================
"""

from functools import lru_cache

from src.data.loader import (
    load_champion_model,
    load_shap_explainer,
    load_structured_preprocessor,
    load_tfidf_vectorizer)


@lru_cache(maxsize=1)
def get_model():
    return load_champion_model()


@lru_cache(maxsize=1)
def get_explainer():
    return load_shap_explainer()


@lru_cache(maxsize=1)
def get_structured_preprocessor():
    return load_structured_preprocessor()


@lru_cache(maxsize=1)
def get_tfidf():
    return load_tfidf_vectorizer()