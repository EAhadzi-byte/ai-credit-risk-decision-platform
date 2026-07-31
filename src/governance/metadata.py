"""
=========================================================
Model Metadata
=========================================================
"""

import json

from datetime import datetime

from src.config.pipeline_config import PIPELINE_CONFIG


def save_metadata(metadata):

    metadata["created_at"] = datetime.now().isoformat()

    with open(PIPELINE_CONFIG, "w") as f:

        json.dump(metadata,f,indent=4)


def load_metadata():

    with open(PIPELINE_CONFIG, "r") as f:

        return json.load(f)