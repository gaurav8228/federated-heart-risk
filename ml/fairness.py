r"""
fairness.py  --  Milestone M7 (LITE)
Scores the XGBoost active model on the SAME held-out test set as fl/data.py,
broken down by hospital, sex and age group.

    venv\Scripts\python.exe -m ml.fairness

For every group it reports n (size), recall, F1, AUC and accuracy; warns when
a group has fewer than 30 patients; sets AUC to null when a group has only one
class. Prints one table per group type plus the largest recall gap, then saves
the rows to the "fairness_metrics" table (replacing a group's old rows).
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml import preprocess
from backend.database import Base, SessionLocal, engine
from backend.models import FairnessMetric, ModelVersion

ACTIVE_MODEL = os.path.join(ROOT, "ml", "models", "active_model.joblib")
TEST_SIZE, SEED, MIN_GROUP = 0.2, 42, 30


def load_test_set():
    """Rebuild the exact held-out test set fl/data.py uses (raw features + groups)."""
    df = pd.read_csv(os.path.join(ROOT, "data", "heart_combined.csv"))
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    y = (df["target"] > 0).astype(int).to_numpy()
    hospital = df["hospital"].to_numpy()
    X = df[preprocess.FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    _, X_test, _, y_test, _, hosp_test = train_test_split(
        X, y, hospital, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    return X_test.reset_index(drop=True), y_test, hosp_test


def score(y_true, y_pred, y_prob):
    """Metrics for one group (AUC = None when the group has a single class)."""
    auc = None if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_prob))
    return {
        "n": int(len(y_true)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def main():
    X_test, y_test, hosp_test = load_test_set()
    model = joblib.load(ACTIVE_MODEL)                 # the XGBoost pipeline
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    age = X_test["age"].to_numpy()
    sex = X_test["sex"].to_numpy()
    groups = {
        "hospital": {h: np.where(hosp_test == h)[0] for h in ["A", "B", "C"]},
        "sex": {"female": np.where(sex == 0)[0], "male": np.where(sex == 1)[0]},
        "age": {"under 45": np.where(age < 45)[0],
                "45-60": np.where((age >= 45) & (age <= 60))[0],
                "over 60": np.where(age > 60)[0]},
    }

    results = {}
    for gtype, gdict in groups.items():
        print(f"\n=== Fairness by {gtype} ===")
        print(f"{'Group':<12}{'n':>5}{'Recall':>9}{'F1':>8}{'AUC':>8}{'Accuracy':>10}")
        rows = []
        for value, idx in gdict.items():
            m = score(y_test[idx], y_pred[idx], y_prob[idx])
            if m["n"] < MIN_GROUP:
                print(f"  warning: '{value}' has only {m['n']} patients "
                      f"(<{MIN_GROUP}) - metrics unreliable")
            auc_str = "null" if m["auc"] is None else f"{m['auc']:.4f}"
            print(f"{value:<12}{m['n']:>5}{m['recall']:>9.4f}{m['f1']:>8.4f}"
                  f"{auc_str:>8}{m['accuracy']:>10.4f}")
            rows.append((value, m))
        recalls = [m["recall"] for _, m in rows]
        print(f"  largest recall gap: {max(recalls) - min(recalls):.4f}")
        results[gtype] = rows

    save_to_db(results)


def save_to_db(results):
    """Replace each group type's old rows with the freshly computed ones."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.execute(text("ALTER TABLE fairness_metrics ADD COLUMN IF NOT EXISTS n integer"))
        db.commit()
        active = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()  # noqa: E712
        mvid = active.id if active else None
        for gtype, rows in results.items():
            db.query(FairnessMetric).filter(FairnessMetric.group_type == gtype).delete()
            for value, m in rows:
                db.add(FairnessMetric(model_version_id=mvid, group_type=gtype,
                                      group_value=value, recall=m["recall"],
                                      f1=m["f1"], auc=m["auc"], n=m["n"]))
        db.commit()
        print(f"\nSaved fairness rows to the database "
              f"({sum(len(r) for r in results.values())} rows).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
