r"""
preprocess.py  --  Milestone M1, step 2
Loads the combined data, splits it into train/test and builds the
preprocessing pipeline:

  * numeric columns   -> fill missing values with the median, then scale
  * categorical columns -> fill missing values with the most common value,
                           then one-hot encode

IMPORTANT: the pipeline is FIT on the training split only, so information
from the test patients never "leaks" into the training step.

Everything saved here lives in ml/models/ and is reused by train.py
(and later by the FastAPI backend and the federated-learning step).
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Where files live.  We build the paths from this file's location, so the
# script works no matter which folder you run it from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_CSV = os.path.join(ROOT, "data", "heart_combined.csv")
MODELS_DIR = os.path.join(ROOT, "ml", "models")
PREPROCESSOR_FILE = os.path.join(MODELS_DIR, "preprocessor.joblib")

# Columns used to predict heart disease.
# "source" (which UCI file the patient came from) and "hospital" are NOT
# features - hospital is only used later to split the data for federated
# learning, so the model cannot cheat by learning which hospital a patient is from.
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

TARGET_COL = "target"          # original UCI value 0..4 (0 = healthy)
TEST_SIZE = 0.2                # 20% of patients are kept aside for testing
RANDOM_STATE = 42              # fixed seed -> you get the same split every run


def load_combined():
    """Read data/heart_combined.csv and turn it into X (features) and y (label)."""
    df = pd.read_csv(COMBINED_CSV)

    # In the Switzerland/VA files a cholesterol of 0 really means "unknown".
    df.loc[df["chol"] == 0, "chol"] = pd.NA
    df.loc[df["trestbps"] == 0, "trestbps"] = pd.NA

    # The prediction target: 0 = no disease, 1..4 = disease present -> binary 0/1
    y = (df[TARGET_COL] > 0).astype(int)
    X = df[FEATURE_COLS].copy()

    # Make sure every column is numeric (missing values become NaN, handled later)
    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, y, df["hospital"]


def split(X, y, hospital):
    """Split into train (80%) and test (20%), keeping the same sick/healthy ratio."""
    return train_test_split(
        X, y, hospital,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor():
    """Build the preprocessing pipeline (nothing is fitted yet)."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # missing -> median value
        ("scaler", StandardScaler()),                   # scale to mean 0 / std 1
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # missing -> most common
        ("onehot", OneHotEncoder(handle_unknown="ignore")),    # numbers -> 0/1 columns
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])


def fit_and_save(X_train):
    """Fit the pipeline on the TRAINING split only and save it with joblib."""
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print("Saved preprocessor ->", PREPROCESSOR_FILE)
    return preprocessor


if __name__ == "__main__":
    X, y, hospital = load_combined()
    X_train, X_test, y_train, y_test, hosp_train, hosp_test = split(X, y, hospital)
    print(f"Total patients : {len(X)}  (train {len(X_train)} / test {len(X_test)})")
    print(f"Disease ratio  : {y.mean():.2%}")
    print(f"Hospitals      : {hospital.value_counts().to_dict()}")
    fit_and_save(X_train)
    print("Preprocessing ready.")
