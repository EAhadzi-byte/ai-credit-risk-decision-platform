"""
=========================================================
Pipeline Validation
=========================================================
"""

from scipy import sparse


def validate_matrix(matrix):

    if not sparse.issparse(matrix):

        raise TypeError("Matrix must be sparse.")

    if matrix.shape[0] == 0:

        raise ValueError("Empty matrix.")

    return True


def validate_prediction(probability):

    if probability < 0:

        raise ValueError("Probability below zero.")

    if probability > 1:

        raise ValueError("Probability above one.")

    return True


def validate_feature_count(matrix, expected):

    if matrix.shape[1] != expected:

        raise ValueError(

            f"Expected {expected} features.")

    return True