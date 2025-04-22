# src/dataprocessing.py
import pandas as pd
import os

def load_data(path: str = None) -> pd.DataFrame:
    """
    Load the CSV. If no path given, fall back to ../data/Health_Sleep_Statistics.csv.
    """
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.abspath(
            os.path.join(base, "..", "data", "Health_Sleep_Statistics.csv")
        )
    return pd.read_csv(path)


def transform(df: pd.DataFrame, fit: bool = True):
    """
    Turn raw DataFrame into feature matrix X and target vector y.
    You’ll need to replace 'target_column' with the name of your label.
    """
    target_col = "Sleep Quality"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True)

    return X, y