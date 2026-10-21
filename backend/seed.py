r"""
seed.py  --  Milestone M3
One-time "fill the database with starter data" script.

It creates:
  * the tables (if missing)
  * two users:
        doctor  doctor@hospital.com / Doctor@123   (Dr. Smith)
        admin   admin@hospital.com  / Admin@123
  * three hospitals  (A, B, C) for the federated-learning step
  * one model_versions row pointing at the trained XGBoost model

Run it from the project root:
    venv\Scripts\python.exe -m backend.seed
"""

import json
import os

from . import models
from .auth import hash_password
from .database import Base, SessionLocal, engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_FILE = os.path.join(ROOT, "ml", "models", "metrics.json")
ACTIVE_MODEL = "ml/models/active_model.joblib"

# The two demo accounts (same as in SPEC.md / the design diagram)
USERS = [
    ("doctor@hospital.com", "Doctor@123", "Dr. Smith", "doctor"),
    ("admin@hospital.com", "Admin@123", "Hospital Admin", "admin"),
]

# The 3 federated-learning clients (each has ~307 patients in our data)
HOSPITALS = [("Hospital A", "A", 307), ("Hospital B", "B", 307), ("Hospital C", "C", 306)]


def main():
    Base.metadata.create_all(bind=engine)      # create tables if they do not exist
    db = SessionLocal()
    try:
        # --- 1. Users ---------------------------------------------------------
        for email, password, name, role in USERS:
            if db.query(models.User).filter(models.User.email == email).first():
                print("user already exists:", email)
                continue
            db.add(models.User(
                email=email,
                password_hash=hash_password(password),
                full_name=name,
                role=role,
            ))
            print("created user:", email, f"({role})")

        # --- 2. Hospitals -----------------------------------------------------
        for name, client_id, count in HOSPITALS:
            if db.query(models.Hospital).filter(models.Hospital.client_id == client_id).first():
                continue
            db.add(models.Hospital(name=name, client_id=client_id, patient_count=count))
            print("created hospital:", name)

        # --- 3. Register the trained model -------------------------------------
        if not db.query(models.ModelVersion).first():
            metrics = {}
            if os.path.exists(METRICS_FILE):
                with open(METRICS_FILE, encoding="utf-8") as f:
                    metrics = json.load(f)
            db.add(models.ModelVersion(
                name="HeartRisk XGBoost",
                algorithm="xgboost",
                version="1.0",
                metrics=metrics.get("metrics", {}).get("xgboost", {}),
                file_path=ACTIVE_MODEL,
                is_active=True,
            ))
            print("registered model: HeartRisk XGBoost v1.0")

        db.commit()
        print("\nSeed finished. Log in with:")
        print("  doctor  doctor@hospital.com / Doctor@123")
        print("  admin   admin@hospital.com  / Admin@123")
    finally:
        db.close()


if __name__ == "__main__":
    main()
