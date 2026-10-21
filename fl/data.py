r"""
data.py  --  Milestone M5
Splits the heart-disease data for federated learning.

The rules:
  * A shared TEST set (~20%, stratified) is held out first.  NO hospital ever
    trains on it - it is only used to score the global model after each round.
  * From what is left (the "pool"), each hospital A / B / C takes up to 300 of
    its own patients to train on locally.  The three hospitals never share
    patient rows with each other - only model weights are shared later.
  * The preprocessing pipeline (from ml/preprocess.py) is FIT on the union of
    the hospitals' training patients only, so the test set never leaks in.

NOTE ON THE "300 per hospital" NUMBER:
  The dataset has 920 patients.  A strict held-out test set (184) plus
  3 x 300 = 900 training patients would need 1084 patients, which we do not
  have.  So we honour the more important rule - the test set is fully held out
  - and each hospital therefore uses every patient it still has in the pool
  (~240-250 each).  If the dataset were larger we would reach exactly 300.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Make the project root importable so we can reuse ml/preprocess.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml import preprocess  # build_preprocessor, FEATURE_COLS

COMBINED_CSV = os.path.join(ROOT, "data", "heart_combined.csv")
HOSPITALS = ["A", "B", "C"]
MAX_PER_HOSPITAL = 300
TEST_SIZE = 0.2
SEED = 42


def load_fl_data(seed=SEED):
    """
    Returns:
      preprocessor : fitted on the hospitals' training patients only
      clients      : list of {"hospital", "X" (2D array), "y" (array)}
      X_test       : transformed held-out test features
      y_test       : held-out test labels
    """
    # --- read the CSV -------------------------------------------------------
    df = pd.read_csv(COMBINED_CSV)
    # 0 cholesterol / 0 BP really means "unknown" in the source data
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan

    y = (df["target"] > 0).astype(int).to_numpy()     # 0 = healthy, 1 = disease
    hospital = df["hospital"].to_numpy()
    X = df[preprocess.FEATURE_COLS].apply(pd.to_numeric, errors="coerce")

    # --- 1) hold out the shared test set FIRST (stratified by label) -------
    X_pool, X_test, y_pool, y_test, hosp_pool, _ = train_test_split(
        X, y, hospital, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    # --- 2) give each hospital up to 300 of its own pool patients ----------
    rng = np.random.default_rng(seed)
    clients = []
    for h in HOSPITALS:
        mask = hosp_pool == h
        Xh, yh = X_pool[mask].reset_index(drop=True), y_pool[mask]
        n_take = min(MAX_PER_HOSPITAL, len(Xh))        # 300, or all we have
        idx = rng.choice(len(Xh), size=n_take, replace=False)
        clients.append({"hospital": h, "X": Xh.iloc[idx], "y": yh[idx]})
        print(f"  Hospital {h}: trains on {n_take} patients")

    # --- 3) fit the preprocessor on the UNION of client training data ------
    all_train = pd.concat([c["X"] for c in clients], ignore_index=True)
    preprocessor = preprocess.build_preprocessor()
    preprocessor.fit(all_train)                        # NEVER sees the test set

    # --- 4) turn everything into the numeric matrix the model uses ---------
    for c in clients:
        c["X"] = np.asarray(preprocessor.transform(c["X"]), dtype=np.float64)
    X_test = np.asarray(preprocessor.transform(X_test), dtype=np.float64)

    n_features = X_test.shape[1]
    print(f"  Held-out test set: {len(X_test)} patients")
    print(f"  Features after preprocessing: {n_features}")
    return preprocessor, clients, X_test, y_test


def union_of_clients(clients):
    """Stack every hospital's training data into one big matrix (for the
    centralized baseline, which is allowed to see everything)."""
    X = np.vstack([c["X"] for c in clients])
    y = np.concatenate([c["y"] for c in clients])
    return X, y
