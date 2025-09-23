# src/utils/io.py

import pandas as pd
import os

def save_dataframe_to_excel(df, path, sheet_name="Sheet1"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, sheet_name=sheet_name, index=False)

def save_dataframe_to_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

