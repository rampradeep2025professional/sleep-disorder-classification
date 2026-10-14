"""
train_model.py - trains, compares and saves the sleep disorder models.
Run:  python train_model.py
Outputs: model/model.pkl, model/preprocessor.pkl, model/metadata.json,
         results/*.csv|txt, images/confusion_matrix_*.png, images/model_comparison.png
"""
import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (ID_COLUMNS, RAW_FEATURES, TARGET, build_preprocessor,
                           clean_features, load_dataset)

DATA_PATH = "data/dataset.csv"
RANDOM_STATE = 42
for folder in ["model", "results", "images"]:
    os.makedirs(folder, exist_ok=True)

# ---------------- 1. Load and inspect ----------------
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Dataset not found at {DATA_PATH}. Please place the CSV there.")
df = load_dataset(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows (all columns):", df.duplicated().sum())
print("Duplicate rows ignoring Person ID:", df.drop(columns=ID_COLUMNS).duplicated().sum())
print("\nUnique values per column:\n", df.nunique())

# ---------------- 2. Clean ----------------
# Person ID is only an identifier, so drop it. Then remove duplicate records:
# many rows are identical apart from the ID. If kept, copies would land in both
# the train and test sets and make test scores look better than they really are.
df = df.drop(columns=ID_COLUMNS)
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nRemoved {before - len(df)} duplicate records -> {len(df)} rows remain")
print("\nClass distribution:\n", df[TARGET].value_counts())

X = df[RAW_FEATURES]
y = df[TARGET]
cleaned = clean_features(X)
print("\nCategorical columns:", list(cleaned.select_dtypes(exclude="number").columns))
print("Numerical columns:", list(cleaned.select_dtypes(include="number").columns))

# ---------------- 3. Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")

# ---------------- 4. Models ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
labels = ["None", "Insomnia", "Sleep Apnea"]

rows, fitted, reports = [], {}, []
for name, clf in models.items():
    pipe = make_pipeline(build_preprocessor(), clf)
    # Cross-validation on the TRAINING data only (used to choose the model)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
        "CV F1 mean": cv_f1.mean(), "CV F1 std": cv_f1.std(),
    })
    fitted[name] = pipe
    reports.append(f"===== {name} =====\n{classification_report(y_test, pred, zero_division=0)}\n")
    cm = confusion_matrix(y_test, pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {name}"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"images/confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()

results = pd.DataFrame(rows).round(4)
print("\n=== Model comparison (test set; CV on training set) ===")
print(results.to_string(index=False))
results.to_csv("results/model_comparison.csv", index=False)
with open("results/classification_reports.txt", "w") as f:
    f.write("\n".join(reports))
print("\n" + "\n".join(reports))

# ---------------- 5. Pick best model ----------------
# Selection uses cross-validated macro-F1 on the training data (NOT the test
# set), so the test set stays an honest final check. Macro-F1 treats all 3
# classes equally, which matters because 'None' has more samples than the others.
# Ties are broken by the test macro-F1.
best_row = results.sort_values(["CV F1 mean", "F1 (macro)"], ascending=False).iloc[0]
best_name = best_row["Model"]
print(f"Selected model: {best_name}")
best_pipe = fitted[best_name]

results.set_index("Model")[["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]].plot(
    kind="bar", figsize=(10, 5), ylim=(0, 1.05))
plt.title("Model Comparison (Test Set)"); plt.xticks(rotation=20); plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=150); plt.close()

# ---------------- 6. Save ----------------
# model.pkl = classifier only, preprocessor.pkl = fitted preprocessing pipeline
preprocessor, classifier = best_pipe[0], best_pipe[-1]
joblib.dump(classifier, "model/model.pkl")
joblib.dump(preprocessor, "model/preprocessor.pkl")

best_pred = best_pipe.predict(X_test)
metadata = {
    "best_model": best_name,
    "classes": [str(c) for c in classifier.classes_],
    "metrics": {k: float(best_row[k]) for k in
                ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "CV F1 mean", "CV F1 std"]},
    "n_rows_after_dedup": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "supports_proba": hasattr(classifier, "predict_proba"),
    "options": {
        "Gender": sorted(df["Gender"].unique().tolist()),
        "Occupation": sorted(df["Occupation"].unique().tolist()),
        "BMI Category": sorted(cleaned["BMI Category"].unique().tolist()),
    },
    "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in
               ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
                "Stress Level", "Heart Rate", "Daily Steps"]},
    "bp_ranges": {"Systolic BP": [float(cleaned["Systolic BP"].min()), float(cleaned["Systolic BP"].max())],
                  "Diastolic BP": [float(cleaned["Diastolic BP"].min()), float(cleaned["Diastolic BP"].max())]},
    "model_comparison": results.to_dict(orient="records"),
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved model/model.pkl, model/preprocessor.pkl, model/metadata.json")
