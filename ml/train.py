r"""
train.py  --  Milestone M1, step 3 (the file you normally run)
Downloads the data if needed, preprocesses it, trains 4 models and prints
one comparison table. Then saves every model + a metrics JSON file.

Models trained:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost
  4. ANN  (scikit-learn MLPClassifier - we do NOT use PyTorch/TensorFlow)

Run from the project root:
  venv\Scripts\python.exe ml\train.py
"""

import json
import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import download_data
import preprocess

# Absolute project folder, so the script works no matter where you run it from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "ml", "models")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# 1. The four models. Each one is wrapped in the SAME preprocessing pipeline,
#    so a saved model can be used directly with raw patient numbers.
# --------------------------------------------------------------------------- #
def build_models():
    return {
        "logistic_regression": Pipeline([
            ("preprocess", preprocess.build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", preprocess.build_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "xgboost": Pipeline([
            ("preprocess", preprocess.build_preprocessor()),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "ann": Pipeline([
            ("preprocess", preprocess.build_preprocessor()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=400,
                early_stopping=True, random_state=RANDOM_STATE)),
        ]),
    }


# --------------------------------------------------------------------------- #
# 2. Score one model on the test patients.
# --------------------------------------------------------------------------- #
def evaluate(pipeline, X_test, y_test):
    """Return accuracy / precision / recall / F1 / AUC for one model."""
    y_pred = pipeline.predict(X_test)
    # probability of "disease" - needed for AUC
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:                                             # very old models only
        y_prob = pipeline.decision_function(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }


def main():
    started = time.time()

    # --- 0. Make sure the raw data exists (downloads it on first run) -------
    if not os.path.exists(os.path.join(ROOT, "data", "heart_combined.csv")):
        print("Data file missing - downloading it now...")
        download_data.download_sources()
        download_data.combine()

    # --- 1. Load + split + fit the preprocessor on the training data --------
    X, y, hospital = preprocess.load_combined()
    X_train, X_test, y_train, y_test, hosp_train, hosp_test = preprocess.split(X, y, hospital)
    preprocess.fit_and_save(X_train)                 # writes preprocessor.joblib

    print(f"\nPatients: {len(X)} total | train {len(X_train)} | test {len(X_test)}")
    print(f"Disease rate: {y.mean():.1%}\n")

    # --- 2. Train the 4 models and score them -------------------------------
    models = build_models()
    results = {}
    for name, pipeline in models.items():
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        results[name] = evaluate(pipeline, X_test, y_test)
        print(f"trained {name:<20} in {time.time() - t0:5.1f}s")

    # --- 3. One comparison table -------------------------------------------
    table = pd.DataFrame(results).T                      # models = rows
    table = table[["accuracy", "precision", "recall", "f1", "auc"]]  # tidy column order
    best_name = table["auc"].idxmax()                    # best model = highest AUC
    table.index = [
        (n + "  <-- BEST (by AUC)") if n == best_name else n for n in table.index
    ]
    print("\n================ MODEL COMPARISON (test set) ================")
    print(table.to_string())
    print("==============================================================\n")
    print("Best model:", best_name, "| AUC =", results[best_name]["auc"])

    # --- 4. Save every model + the metrics file -----------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    saved = []
    for name, pipeline in models.items():
        path = os.path.join(MODELS_DIR, f"model_{name}.joblib")
        joblib.dump(pipeline, path)
        saved.append(os.path.relpath(path, ROOT).replace("\\", "/"))

    best_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(models[best_name], best_path)            # same object, clear name
    saved.append(os.path.relpath(best_path, ROOT).replace("\\", "/"))

    metrics = {
        "dataset": {
            "name": "UCI Heart Disease (Cleveland + Hungarian + Switzerland + VA)",
            "patients": int(len(X)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "disease_rate": round(float(y.mean()), 4),
        },
        "best_model": best_name,
        "best_auc": results[best_name]["auc"],
        "metrics": results,
        "saved_files": saved,
        "train_seconds": round(time.time() - started, 1),
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics  ->", METRICS_FILE)
    print("Saved models   ->", ", ".join(saved))
    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
