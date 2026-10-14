"""
preprocessing.py
Shared data-loading and preprocessing code.
It is imported by train_model.py (training) AND app.py (prediction),
so the exact same cleaning steps are applied in both places.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "Sleep Disorder"
ID_COLUMNS = ["Person ID"]  # meaningless identifier -> never used as a feature

# Raw feature columns expected by the preprocessor (same names as the dataset)
RAW_FEATURES = [
    "Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level", "BMI Category",
    "Blood Pressure", "Heart Rate", "Daily Steps",
]

# Columns after cleaning (Blood Pressure is split into two numbers)
CATEGORICAL_COLUMNS = ["Gender", "Occupation", "BMI Category"]
NUMERICAL_COLUMNS = [
    "Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
    "Stress Level", "Heart Rate", "Daily Steps", "Systolic BP", "Diastolic BP",
]


def load_dataset(path):
    """Load the CSV.
    keep_default_na=False is IMPORTANT: in this dataset the text "None" is a real
    class label ("No Sleep Disorder"), but pandas would otherwise read it as NaN.
    """
    return pd.read_csv(path, keep_default_na=False)


def clean_features(X):
    """Row-wise cleaning that needs no learning from data:
    1. 'Normal Weight' and 'Normal' mean the same thing -> merge into 'Normal'
    2. 'Blood Pressure' like '126/83' -> two numeric columns
    """
    X = X.copy()
    X["BMI Category"] = X["BMI Category"].astype(str).str.strip().replace({"Normal Weight": "Normal"})
    bp = X["Blood Pressure"].astype(str).str.split("/", expand=True)
    X["Systolic BP"] = pd.to_numeric(bp[0], errors="coerce")
    X["Diastolic BP"] = pd.to_numeric(bp[1], errors="coerce")
    return X.drop(columns=["Blood Pressure"])


def build_preprocessor():
    """cleaning -> (scale numbers, one-hot encode categories)"""
    encode = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline([
        ("clean", FunctionTransformer(clean_features, validate=False)),
        ("encode", encode),
    ])
