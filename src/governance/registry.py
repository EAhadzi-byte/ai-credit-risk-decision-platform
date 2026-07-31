"""
=========================================================
Credit Risk AI Platform
Model Registry
=========================================================
"""

import json

from datetime import datetime

from src.config.pipeline_config import MODEL_METADATA


def load_registry():

    if MODEL_METADATA.exists():

        with open(MODEL_METADATA, "r") as f:

            return json.load(f)

    return []


def save_registry(registry):

    with open(MODEL_METADATA, "w") as f:

        json.dump(registry,f,indent=4)


def register_model(

    model_name,

    algorithm,

    dataset,

    roc_auc,

    precision,

    recall,

    f1,

    filepath):

    registry = load_registry()

    record = {

        "timestamp": datetime.now().isoformat(),

        "model_name": model_name,

        "algorithm": algorithm,

        "dataset": dataset,

        "roc_auc": float(roc_auc),

        "precision": float(precision),

        "recall": float(recall),

        "f1": float(f1),

        "filepath": str(filepath)}

    registry.append(record)

    save_registry(registry)

    print(f"{model_name} registered successfully.")


def get_best_model(metric="roc_auc"):

    registry = load_registry()

    if not registry:

        return None

    return max(registry, key=lambda x: x[metric])


if __name__ == "__main__":

    print(load_registry())