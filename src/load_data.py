import pandas as pd
import os

def load_data(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    return pd.read_csv(file_path)
