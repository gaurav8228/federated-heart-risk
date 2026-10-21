r"""
models.py  --  Milestone M3
The database tables (one class per table), as listed in SPEC.md section 3.

  users            doctors and admins who can log in
  hospitals        the 3 federated-learning clients (A, B, C)
  patients         one row per patient the doctor enters
  predictions      the model output: risk score + SHAP explanation + time
  fl_rounds        one row per federated-learning round (accuracy, epsilon...)
  model_versions   every trained model that was registered
  fairness_metrics fairness results per group (sex, age, hospital)

Tables are created automatically when the server starts (see main.py).
"""

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String,
)
from sqlalchemy.sql import func

from .database import Base


class User(Base):
    """Someone who can log in: a doctor or an admin."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)   # never the plain password
    full_name = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="doctor")  # "doctor" or "admin"
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Hospital(Base):
    """One federated-learning client (Hospital A, B or C)."""
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)   # "Hospital A"
    client_id = Column(String(20), unique=True, nullable=False)  # "A"
    patient_count = Column(Integer, default=0)


class Patient(Base):
    """A patient record entered by a doctor. Same fields as the ML features."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # who entered it
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)

    # The 13 heart-disease features (see ml/preprocess.py)
    age = Column(Float)
    sex = Column(Float)
    cp = Column(Float)          # chest pain type
    trestbps = Column(Float)    # resting blood pressure
    chol = Column(Float)        # cholesterol
    fbs = Column(Float)         # fasting blood sugar > 120
    restecg = Column(Float)     # resting ECG result
    thalach = Column(Float)     # maximum heart rate
    exang = Column(Float)       # exercise-induced angina
    oldpeak = Column(Float)     # ST depression
    slope = Column(Float)
    ca = Column(Float)          # number of major vessels
    thal = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    """One model prediction: risk score, label and the SHAP explanation."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    risk_score = Column(Float, nullable=False)      # 0 - 100
    risk_label = Column(String(20), nullable=False)  # Low / Moderate / High
    shap_values = Column(JSON, nullable=False)       # the key_factors list as JSON
    model_version = Column(String(50), default="1.0")

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FLRound(Base):
    """One federated-learning round (filled in later milestones, M5+)."""
    __tablename__ = "fl_rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, nullable=False)
    strategy = Column(String(20), default="fedavg")  # fedavg or fedprox
    global_accuracy = Column(Float)
    epsilon = Column(Float)                    # differential-privacy budget
    communication_cost_mb = Column(Float)      # how much data was exchanged
    status = Column(String(20), default="running")  # running / done / stopped

    started_at = Column(DateTime(timezone=True), server_default=func.now())
    finished_at = Column(DateTime(timezone=True), nullable=True)


class ModelVersion(Base):
    """A registered model (the seed script registers the XGBoost model)."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)          # "HeartRisk XGBoost"
    algorithm = Column(String(50), nullable=False)      # "xgboost"
    version = Column(String(50), default="1.0")
    metrics = Column(JSON)                              # accuracy, auc, ...
    file_path = Column(String(255))                     # ml/models/active_model.joblib
    is_active = Column(Boolean, default=False)          # used for predictions

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FairnessMetric(Base):
    """Fairness results per group (per hospital, per sex, per age group)."""
    __tablename__ = "fairness_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)
    group_type = Column(String(20), nullable=False)   # "hospital" / "sex" / "age"
    group_value = Column(String(50), nullable=False)  # "A", "1 (male)", "60+"
    recall = Column(Float)
    f1 = Column(Float)
    auc = Column(Float)
    n = Column(Integer)                              # group size (patients)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
