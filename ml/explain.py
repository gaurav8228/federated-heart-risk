r"""
explain.py  --  Milestone M2  (SHAP explainability)

We explain the XGBoost model (NOT the ANN) because tree models have fast and
EXACT SHAP values via TreeExplainer, so the explanations are trustworthy.

This file:
  1. explain_patient(patient_dict) -> JSON with:
       risk_score   (0-100)
       risk_label   ("Low" < 40, "Moderate" 40-65, "High" > 65)
       key_factors  (top 5 features by |SHAP|, readable names, signed values)
     One-hot columns are merged back into their original feature, e.g. all
     "cp_1.0 / cp_2.0 / cp_3.0 / cp_4.0" columns become "Chest Pain Type".
  2. Summary plot for the test set   -> ml/explain/shap_summary.png
  3. Waterfall (force) plot for one patient -> ml/explain/waterfall_plot.png
  4. Deletion test: replace the top feature with the training median/mode,
     the predicted risk should DROP on average -> prints PASS or FAIL.

Run:  venv\Scripts\python.exe ml\explain.py
"""

import json
import os
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")          # draw pictures without opening any window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:                      # when FastAPI imports us as a package: ml.explain
    from . import preprocess
except ImportError:       # when run as a plain script: python ml\explain.py
    import preprocess

# Folders, built from this file's location (works from any folder)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "ml", "models")
EXPLAIN_DIR = os.path.join(ROOT, "ml", "explain")

XGBOOST_FILE = os.path.join(MODELS_DIR, "model_xgboost.joblib")
ACTIVE_FILE = os.path.join(MODELS_DIR, "active_model.joblib")   # used by the API later

# Machine column name -> readable name a doctor understands
READABLE = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting BP",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "ST Depression",
    "slope": "Slope",
    "ca": "Major Vessels",
    "thal": "Thalassemia",
}

# The example patient from the design diagram (52-year-old man)
SAMPLE_PATIENT = {
    "age": 52, "sex": 1, "cp": 1, "trestbps": 120, "chol": 240,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def ensure_active_model():
    """Copy the XGBoost pipeline to active_model.joblib (the backend uses it)."""
    if not os.path.exists(ACTIVE_FILE):
        os.makedirs(MODELS_DIR, exist_ok=True)
        shutil.copyfile(XGBOOST_FILE, ACTIVE_FILE)
        print("Saved active model ->", ACTIVE_FILE)


def load_model():
    """Load the saved pipeline: [preprocessing steps] + [XGBoost classifier]."""
    return joblib.load(ACTIVE_FILE)


def as_frame(patient_dict):
    """Turn a plain dictionary of patient numbers into a 1-row DataFrame."""
    row = pd.DataFrame([{c: patient_dict.get(c, np.nan)
                         for c in preprocess.FEATURE_COLS}])
    for c in preprocess.FEATURE_COLS:
        row[c] = pd.to_numeric(row[c], errors="coerce")   # missing -> NaN (imputed later)
    return row


# --------------------------------------------------------------------------- #
# Helpers: merge the one-hot columns back into one original feature
# --------------------------------------------------------------------------- #
def group_shap(shap_values, transformed_names):
    """
    The preprocessor creates 28 columns (5 numbers + 23 one-hot columns).
    Sum the SHAP values of columns that come from the SAME original feature,
    e.g. all cp_* columns -> one number for "Chest Pain Type".
    Returns (grouped_values, base_column_names) in the original column order.
    """
    index_of = {c: [] for c in preprocess.FEATURE_COLS}
    for j, name in enumerate(transformed_names):
        short = name.split("__", 1)[-1]              # "cat__cp_2.0" -> "cp_2.0"
        for base in preprocess.FEATURE_COLS:
            if short == base or short.startswith(base + "_"):
                index_of[base].append(j)
                break
    ordered = [c for c in preprocess.FEATURE_COLS if index_of[c]]
    grouped = np.column_stack([shap_values[:, index_of[c]].sum(axis=1)
                               for c in ordered])
    return grouped, ordered


# --------------------------------------------------------------------------- #
# 1. Explain one patient -> JSON (this is what the FastAPI backend will return)
# --------------------------------------------------------------------------- #
def explain_patient(patient_dict, model=None):
    """
    patient_dict example:
      {"age": 52, "sex": 1, "cp": 1, "trestbps": 120, "chol": 240, ...}
    Returns a JSON-ready dictionary with risk_score, risk_label, key_factors.
    """
    model = model or load_model()
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["model"]
    row = as_frame(patient_dict)

    # Predicted probability of heart disease (0.0 - 1.0)
    proba = float(model.predict_proba(row)[0, 1])
    risk_score = round(proba * 100, 1)
    if risk_score < 40:
        risk_label = "Low"
    elif risk_score <= 65:
        risk_label = "Moderate"
    else:
        risk_label = "High"

    # Exact SHAP values for this one patient
    explainer = shap.TreeExplainer(clf)
    shap_row = explainer.shap_values(np.asarray(pre.transform(row)))  # (1, 28)
    grouped, base_cols = group_shap(shap_row, list(pre.get_feature_names_out()))

    # Top 5 features by |SHAP| (positive = pushes risk up, negative = down)
    values = grouped[0]
    top = np.argsort(-np.abs(values))[:5]
    key_factors = [
        {"feature": READABLE[base_cols[i]], "value": round(float(values[i]), 4)}
        for i in top
    ]

    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "key_factors": key_factors,
    }


# --------------------------------------------------------------------------- #
# 2. Summary plot for the whole test set
# --------------------------------------------------------------------------- #
def make_summary_plot(model):
    """One picture: how every feature pushes the risk, for all test patients."""
    X, y, hospital = preprocess.load_combined()
    _, X_test, _, y_test, _, _ = preprocess.split(X, y, hospital)
    pre, clf = model.named_steps["preprocess"], model.named_steps["model"]

    shap_values = shap.TreeExplainer(clf).shap_values(np.asarray(pre.transform(X_test)))
    grouped, base_cols = group_shap(shap_values, list(pre.get_feature_names_out()))

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        grouped,
        features=X_test[base_cols],                       # used for the dot colour
        feature_names=[READABLE[c] for c in base_cols],
        show=False, max_display=len(base_cols),
    )
    plt.tight_layout()
    path = os.path.join(EXPLAIN_DIR, "shap_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved ->", path)


# --------------------------------------------------------------------------- #
# 3. Waterfall (force) plot for one example patient
# --------------------------------------------------------------------------- #
def make_waterfall_plot(model, patient_dict=None):
    """
    One picture for one patient: how the average patient risk is pushed up
    or down feature-by-feature to reach this patient's risk.
    (Waterfall = the modern version of SHAP's force plot.)
    """
    patient_dict = patient_dict or SAMPLE_PATIENT
    pre, clf = model.named_steps["preprocess"], model.named_steps["model"]
    row = as_frame(patient_dict)

    explainer = shap.TreeExplainer(clf)
    shap_row = explainer.shap_values(np.asarray(pre.transform(row)))
    # Merge one-hot columns back (13 original features) so the picture
    # shows readable names like "Chest Pain Type" instead of "cp_2.0".
    grouped, base_cols = group_shap(shap_row, list(pre.get_feature_names_out()))

    explanation = shap.Explanation(
        values=grouped[0],
        base_values=float(np.squeeze(explainer.expected_value)),  # average patient
        data=row[base_cols].to_numpy()[0],                        # the patient's values
        feature_names=[READABLE[c] for c in base_cols],
    )
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    path = os.path.join(EXPLAIN_DIR, "waterfall_plot.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", path)


# --------------------------------------------------------------------------- #
# 4. Deletion test - a quick sanity check that SHAP is meaningful
# --------------------------------------------------------------------------- #
def deletion_test(model):
    """
    For every test patient:
      a) note the predicted risk
      b) find the feature SHAP says matters most
      c) replace that feature with the training median (number) or
         most-common value (categorical)  -> "delete" it
      d) predict again
    If SHAP is honest, the risk should DROP on average for sick patients.
    """
    X, y, hospital = preprocess.load_combined()
    X_train, X_test, _, y_test, _, _ = preprocess.split(X, y, hospital)
    pre, clf = model.named_steps["preprocess"], model.named_steps["model"]

    medians = X_train[preprocess.NUMERIC_COLS].median()
    modes = X_train[preprocess.CATEGORICAL_COLS].mode().iloc[0]

    shap_values = shap.TreeExplainer(clf).shap_values(np.asarray(pre.transform(X_test)))
    grouped, base_cols = group_shap(shap_values, list(pre.get_feature_names_out()))
    top_feature = [base_cols[i] for i in np.argmax(np.abs(grouped), axis=1)]

    risk_before = model.predict_proba(X_test)[:, 1]
    risk_after = risk_before.copy()
    for i in range(len(X_test)):
        row = X_test.iloc[[i]].copy()
        feat = top_feature[i]
        row[feat] = medians[feat] if feat in preprocess.NUMERIC_COLS else modes[feat]
        risk_after[i] = model.predict_proba(row)[0, 1]

    sick = (y_test == 1).to_numpy()               # test patients who ARE sick
    avg_before = float(risk_before[sick].mean())
    avg_after = float(risk_after[sick].mean())
    avg_drop = avg_before - avg_after

    print(f"\nDeletion test on {int(sick.sum())} test patients with heart disease")
    print(f"  average risk before deleting top feature: {avg_before:.3f}")
    print(f"  average risk after  deleting top feature: {avg_after:.3f}")
    print(f"  average drop: {avg_drop:+.3f}")
    print("PASS" if avg_drop > 0 else "FAIL")

    return {
        "n_sick_patients": int(sick.sum()),
        "avg_risk_before": round(avg_before, 4),
        "avg_risk_after": round(avg_after, 4),
        "avg_drop": round(avg_drop, 4),
        "passed": bool(avg_drop > 0),
    }


# --------------------------------------------------------------------------- #
# Main: run everything once
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ensure_active_model()
    model = load_model()
    os.makedirs(EXPLAIN_DIR, exist_ok=True)

    print("=== 1. Explain one patient ===")
    print(json.dumps(explain_patient(SAMPLE_PATIENT, model=model), indent=2))

    print("\n=== 2. Summary plot (test set) ===")
    make_summary_plot(model)

    print("\n=== 3. Waterfall plot (one patient) ===")
    make_waterfall_plot(model)

    print("\n=== 4. Deletion test ===")
    deletion_test(model)
