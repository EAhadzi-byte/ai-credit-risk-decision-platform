"""
=========================================================
Model Checksum
=========================================================
"""

import hashlib

import json

from src.config.pipeline_config import MODEL_CHECKSUM


def calculate_checksum(filepath):

    sha = hashlib.sha256()

    with open(filepath, "rb") as f:

        while True:

            block = f.read(4096)

            if not block:

                break

            sha.update(block)

    return sha.hexdigest()


def save_checksum(filepath):

    checksum = calculate_checksum(filepath)

    data = {

        "file": str(filepath),

        "checksum": checksum}

    with open(MODEL_CHECKSUM, "w") as f:

        json.dump(

            data,f,indent=4)

    return checksum


def verify_checksum(filepath):

    with open(MODEL_CHECKSUM, "r") as f:

        saved = json.load(f)

    current = calculate_checksum(filepath)

    return current == saved["checksum"]