"""
predict_utils.py - loading saved files, validating input and predicting.
Kept free of Streamlit code so it can be tested on its own.
"""
import json
import os
import joblib
import pandas as pd
from preprocessing import RAW_FEATURES  # also makes sure clean_features is importable for joblib

MODEL_PATH = "model/model.pkl"
PREPROCESSOR_PATH = "model/preprocessor.pkl"
METADATA_PATH = "model/metadata.json"

DISPLAY_NAMES = {"None": "No Sleep Disorder", "Insomnia": "Insomnia", "Sleep Apnea": "Sleep Apnea"}


def load_artifacts():
    """Load saved model, preprocessor and metadata. Raises FileNotFoundError if missing."""
    missing = [p for p in (MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, preprocessor, metadata


def validate_input(data, metadata):
    """Return a list of problems with the input (empty list = valid)."""
    problems = []
    for col in RAW_FEATURES:
        if col not in data or data[col] is None or data[col] == "":
            problems.append(f"Missing value for '{col}'.")
    if problems:
        return problems
    for col, (lo, hi) in metadata["ranges"].items():
        try:
            value = float(data[col])
        except (TypeError, ValueError):
            problems.append(f"'{col}' must be a number.")
            continue
        if not lo <= value <= hi:
            problems.append(f"'{col}' should be between {lo:g} and {hi:g} (range seen in training data).")
    for col in ("Gender", "Occupation", "BMI Category"):
        if data[col] not in metadata["options"][col]:
            problems.append(f"'{data[col]}' is not a known value for '{col}'.")
    try:
        sys_bp, dia_bp = (int(x) for x in str(data["Blood Pressure"]).split("/"))
        if sys_bp <= dia_bp:
            problems.append("Systolic blood pressure must be higher than diastolic.")
    except ValueError:
        problems.append("Blood Pressure must look like 120/80.")
    return problems


def predict(data, model, preprocessor):
    """data: dict with the raw feature names. Returns (label, {label: probability} or None)."""
    row = pd.DataFrame([data])[RAW_FEATURES]
    transformed = preprocessor.transform(row)
    label = str(model.predict(transformed)[0])
    probs = None
    if hasattr(model, "predict_proba"):
        probs = {str(c): float(p) for c, p in zip(model.classes_, model.predict_proba(transformed)[0])}
    return label, probs
