"""
=========================================================
Credit Risk AI Platform
Reusable Helper Functions
=========================================================
"""

import gc
import time
import joblib
import pandas as pd

from pathlib import Path



# TIMER
# ________________________________________________________

def start_timer():

    return time.time()


def stop_timer(start):

    elapsed = time.time() - start

    print(f"Completed in {elapsed:.2f} seconds.")

    return elapsed



# MEMORY
# ______________________________________________________

def clear_memory():

    gc.collect()


# SAVE OBJECT
# _____________________________________________________

def save_object(

    obj,

    filepath):

    joblib.dump(obj,filepath)


# LOAD OBJECT
# ____________________________________________________

def load_object(filepath):

    return joblib.load(filepath)
    

# SAVE DATAFRAME
# ______________________________________________________

def save_dataframe(

    df,filepath):

    if filepath.suffix == ".csv":

        df.to_csv(filepath,index=False)

    elif filepath.suffix == ".parquet":

        df.to_parquet(filepath,index=False)

    else:

        raise ValueError("Unsupported file format.")


# CREATE DIRECTORY
# _________________________________________________________

def create_directory(path):

    Path(path).mkdir(parents=True,exist_ok=True)


# FILE SIZE
# _______________________________________________________

def file_size(filepath):

    return round(

        Path(filepath).stat().st_size /

        1024 /1024,2)


# PRINT HEADER
# __________________________________________________________

def print_header(title):

    print(title)

    print("=" * 50)