"""
=========================================================
Credit Risk AI Platform
Feature Matrix Builder
---------------------------------------------------------
Builds Matrix A, Matrix B, Matrix C and Matrix D
=========================================================
"""

import pandas as pd

from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.pipeline_config import (

    MAX_FEATURES,

    MIN_DF,

    NGRAM_RANGE,

    STOP_WORDS,

    SUBLINEAR_TF,)


# MATRIX A
# __________________________________________________________

def build_matrix_a(structured_features: pd.DataFrame):
    """
    Matrix A
    Structured borrower features only.
    """

    return structured_features.copy()


# MATRIX B
# _________________________________________________________

def create_tfidf_vectorizer():
    """
    Create TF-IDF Vectorizer.
    """

    return TfidfVectorizer(

        max_features=MAX_FEATURES,

        min_df=MIN_DF,

        stop_words=STOP_WORDS,

        ngram_range=NGRAM_RANGE,

        sublinear_tf=SUBLINEAR_TF)


def build_matrix_b(
    train_text,
    test_text):
    """
    Build TF-IDF matrices.
    """

    vectorizer = create_tfidf_vectorizer()

    matrix_B_train = vectorizer.fit_transform(train_text)

    matrix_B_test = vectorizer.transform(test_text)

    feature_names = vectorizer.get_feature_names_out()

    return (

        matrix_B_train,

        matrix_B_test,

        feature_names,

        vectorizer)


# MATRIX C
# ___________________________________________________________

def build_matrix_c(
    matrix_A_train,
    matrix_A_test,
    matrix_B_train,
    matrix_B_test):
    """
    Matrix C
    Structured + NLP
    """

    matrix_C_train = sparse.hstack(

        [sparse.csr_matrix(matrix_A_train),
         
         matrix_B_train]).tocsr()

    matrix_C_test = sparse.hstack(

        [

            sparse.csr_matrix(matrix_A_test),

            matrix_B_test

        ]).tocsr()

    return (

        matrix_C_train,

        matrix_C_test)


# MATRIX D
# _____________________________________________________

def build_matrix_d(
    
    matrix_C_train,
    
    matrix_C_test,
    
    macro_train,
    
    macro_test):
    
    """
    Matrix D

    Structured
    + NLP
    + FRED
    + GDELT
    """

    matrix_D_train = sparse.hstack(

        [

            matrix_C_train,

            sparse.csr_matrix(macro_train)

        ]).tocsr()

    matrix_D_test = sparse.hstack(

        [

            matrix_C_test,

            sparse.csr_matrix(macro_test)

        ]).tocsr()

    return (

        matrix_D_train,

        matrix_D_test)


# FEATURE NAMES
# ______________________________________________________

def combine_feature_names(
    
    structured_features,
    
    tfidf_features,
    
    macro_features):
    
    """
    Combine all feature names.
    """

    return (

        list(structured_features)

        +

        list(tfidf_features)

        +

        list(macro_features))


# VALIDATION
# _____________________________________________________

def validate_matrix(
    
    matrix,
    
    feature_names):
    
    """
    Validate matrix dimensions.
    """

    if matrix.shape[1] != len(feature_names):

        raise ValueError(

            "Feature names do not match matrix columns.")

    return True


# SUMMARY
# ______________________________________________________

def matrix_summary(
    
    name,
    
    matrix):

    density = (

        matrix.nnz /

        (matrix.shape[0] * matrix.shape[1]))

    summary = {

        "Matrix": name,

        "Rows": matrix.shape[0],

        "Columns": matrix.shape[1],

        "Density": round(density,6)}

    return summary


# EXPORT
# ____________________________________________________

def export_feature_names(
    
    feature_names,
    
    filepath):
    
    """
    Export feature names.
    """

    df = pd.DataFrame(

        {

            "Feature_ID":

                range(len(feature_names)),

            "Feature_Name":

                feature_names})

    df.to_csv(filepath,index=False)


# TEST
# _____________________________________________________

if __name__ == "__main__":

    print("Matrix Builder Ready")

    print("=" * 50)