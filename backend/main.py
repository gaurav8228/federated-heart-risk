r"""
main.py  --  Milestone M3
The FastAPI web server (the "brain" that the React app talks to).

Start it with:   venv\Scripts\uvicorn.exe backend.main:app --reload
Then open:       http://localhost:8000/docs     (interactive API documentation)

What happens when the server starts (see lifespan below):
  1. Create the database tables (if they do not exist yet).
  2. Load the trained XGBoost model ONCE into memory (active_model.joblib),
     so every prediction is fast.

Endpoints
  GET  /health                  - is the server alive?
  POST /api/auth/login          - email + password -> JWT token
  POST /api/auth/signup         - public self-signup (always a doctor)
  GET  /api/auth/me             - who am I?
  POST /api/auth/register       - create a user (admins only)
  POST /api/predict             - predict heart-disease risk (JWT required)
  GET  /api/predictions         - my prediction history
  GET  /api/predictions/{id}    - one prediction with its SHAP explanation
  GET  /api/patients            - patient list (admins see everything)
"""

import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from typing import Literal

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

# Make sure we can import the ml/ folder (it sits next to backend/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.explain import READABLE, explain_patient   # our SHAP explanation function
from ml.preprocess import FEATURE_COLS             # the 13 patient fields

from . import auth
from .database import Base, engine, get_db
from .models import FLRound, FairnessMetric, ModelVersion, Patient, Prediction, User

MODEL_PATH = os.path.join(ROOT, "ml", "models", "active_model.joblib")

# The model lives here in memory after startup (loaded once, not per request).
model = None

# --- federated-learning runner (one background process at a time) ------------
VENV_PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
RUN_FL = os.path.join(ROOT, "fl", "run_fl.py")
FL_LOG = os.path.join(ROOT, "fl", "last_run.log")
_fl_process = None            # subprocess.Popen while a run is live, else None
_fl_lock = threading.Lock()


def pretty_names(key_factors):
    """
    Make sure BOTH /api/predict and /api/predictions/{id} show exactly the
    same readable names ("Exercise Angina", never "ExerciseAngina").
    READABLE in ml/explain.py is the single source of truth for names.
    """
    # map "ExerciseAngina" -> "Exercise Angina" (ignores spaces/case)
    lookup = {name.replace(" ", "").lower(): name for name in READABLE.values()}
    return [
        {**factor, "feature": lookup.get(factor["feature"].replace(" ", "").lower(),
                                         factor["feature"])}
        for factor in key_factors
    ]


# --------------------------------------------------------------------------- #
# Startup / shutdown
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts and once when it stops."""
    global model
    Base.metadata.create_all(bind=engine)        # create tables if missing
    try:                                          # add the fairness "n" column if missing
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE fairness_metrics ADD COLUMN IF NOT EXISTS n integer"))
    except Exception:
        pass
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "active_model.joblib not found - run: venv\\Scripts\\python.exe ml\\explain.py"
        )
    model = joblib.load(MODEL_PATH)              # load the XGBoost model ONCE
    print("HeartRisk API ready - model loaded from", MODEL_PATH)
    yield
    # (nothing to clean up on shutdown)


app = FastAPI(title="HeartRisk AI API", version="1.0", lifespan=lifespan)

# Allow the React dev server (npm run dev) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / response shapes (what the JSON body looks like)
# --------------------------------------------------------------------------- #
class RegisterBody(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "doctor"                     # "doctor" or "admin"


class SignupBody(BaseModel):
    """Public signup. Deliberately has NO "role" field - a client can never
    choose a role. Any "role" sent in the JSON is ignored by Pydantic."""
    full_name: str
    email: str
    password: str = Field(min_length=8)      # 422 if shorter than 8 chars


class PatientBody(BaseModel):
    """The same 13 numbers the ML model uses. All optional: missing = unknown."""
    age: float | None = None
    sex: float | None = None
    cp: float | None = None
    trestbps: float | None = None
    chol: float | None = None
    fbs: float | None = None
    restecg: float | None = None
    thalach: float | None = None
    exang: float | None = None
    oldpeak: float | None = None
    slope: float | None = None
    ca: float | None = None
    thal: float | None = None


def patient_to_dict(body: PatientBody) -> dict:
    """Convert the JSON body to the dictionary explain_patient() expects."""
    return {col: getattr(body, col) for col in FEATURE_COLS}


class FlStartBody(BaseModel):
    """Strictly validated FL-run options. Only these values are ever turned
    into command-line arguments - never free text, never a shell."""
    strategy: Literal["fedavg", "fedprox"]         # anything else -> 422
    rounds: int = Field(ge=1, le=100)              # 1..100         -> 422
    mu: float = Field(0.0, ge=0.0, le=1.0)         # 0..1           -> 422
    dp: bool = False                               # turn DP on/off
    clip: float = Field(1.0, gt=0.0)               # must be > 0    -> 422
    noise: float = Field(1.0, gt=0.0)              # must be > 0    -> 422
    secagg: bool = False                           # pairwise-mask the updates (M10)


# --------------------------------------------------------------------------- #
# 1. Health check (no login needed)
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# --------------------------------------------------------------------------- #
# 2. Login: email + password -> JWT token
# --------------------------------------------------------------------------- #
@app.post("/api/auth/login")
async def login(request: Request, db: Session = Depends(get_db)):
    """
    Accepts TWO formats so both work:
      * JSON         {"email": "...", "password": "..."}      <- React uses this
      * HTML form    username=...&password=...                <- Swagger UI uses this
    """
    content_type = request.headers.get("content-type", "")
    if "form" in content_type:
        form = await request.form()
        email = form.get("username") or form.get("email")
        password = form.get("password")
    else:
        body = await request.json()
        email, password = body.get("email"), body.get("password")

    user = db.query(User).filter(User.email == email).first()
    if user is None or not auth.verify_password(password or "", user.password_hash):
        raise HTTPException(status_code=401, detail="Wrong email or password")

    return {
        "access_token": auth.create_access_token(user),
        "token_type": "bearer",
        "role": user.role,
        "name": user.full_name,
    }


# --------------------------------------------------------------------------- #
# 3. Who am I? (needs a token)
# --------------------------------------------------------------------------- #
@app.get("/api/auth/me")
def me(user: User = Depends(auth.get_current_user)):
    return {"id": user.id, "email": user.email, "name": user.full_name, "role": user.role}


# --------------------------------------------------------------------------- #
# 3b. Public signup (no login needed) - always creates a doctor
# --------------------------------------------------------------------------- #
@app.post("/api/auth/signup", status_code=201)
def signup(body: SignupBody, db: Session = Depends(get_db)):
    """Let a visitor create their own account. The role is ALWAYS "doctor" -
    it is never read from the request, so nobody can sign up as an admin.
    Returns a token so the browser can log the user straight in."""
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(409, "This email is already registered")

    user = User(
        email=body.email,
        password_hash=auth.hash_password(body.password),
        full_name=body.full_name,
        role="doctor",                        # hard-coded, never from the client
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "access_token": auth.create_access_token(user),
        "token_type": "bearer",
        "role": user.role,
        "name": user.full_name,
    }


# --------------------------------------------------------------------------- #
# 4. Register a new user (admins only)
# --------------------------------------------------------------------------- #
@app.post("/api/auth/register", status_code=201)
def register(
    body: RegisterBody,
    admin: User = Depends(auth.require_admin),
    db: Session = Depends(get_db),
):
    if body.role not in ("doctor", "admin"):
        raise HTTPException(400, "role must be 'doctor' or 'admin'")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "This email is already registered")

    user = User(
        email=body.email,
        password_hash=auth.hash_password(body.password),
        full_name=body.full_name,
        role=body.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "email": user.email, "name": user.full_name, "role": user.role}


# --------------------------------------------------------------------------- #
# 5. Predict: the heart of the API (needs a token)
# --------------------------------------------------------------------------- #
@app.post("/api/predict")
def predict(
    body: PatientBody,
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """
    1. Save the patient (POST /api/predict does both patient + prediction).
    2. Ask the model + SHAP for a risk score and key factors.
    3. Save the prediction and return everything to the React app.
    """
    # --- a) store the patient ------------------------------------------------
    patient = Patient(user_id=user.id, **patient_to_dict(body))
    db.add(patient)
    db.commit()
    db.refresh(patient)

    # --- b) predict + explain (model was loaded once at startup) -------------
    result = explain_patient(patient_to_dict(body), model=model)

    # --- c) store the prediction (SHAP explanation as JSON) ------------------
    active = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()  # noqa: E712
    prediction = Prediction(
        user_id=user.id,
        patient_id=patient.id,
        risk_score=result["risk_score"],
        risk_label=result["risk_label"],
        shap_values=result,                      # risk + key_factors as JSON
        model_version=active.version if active else "1.0",
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    # --- d) send it back to the browser --------------------------------------
    return {
        "prediction_id": prediction.id,
        "patient_id": patient.id,
        "risk_score": result["risk_score"],
        "risk_label": result["risk_label"],
        "key_factors": pretty_names(result["key_factors"]),
        "created_at": prediction.created_at.isoformat(),
        "model_version": prediction.model_version,
    }


# --------------------------------------------------------------------------- #
# 6. Prediction history (needs a token)
# --------------------------------------------------------------------------- #
@app.get("/api/predictions")
def list_predictions(user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """The History page uses this. Admins see everybody's predictions."""
    q = db.query(Prediction)
    if user.role != "admin":
        q = q.filter(Prediction.user_id == user.id)
    rows = q.order_by(Prediction.created_at.desc()).limit(200).all()
    return [
        {
            "id": p.id,
            "patient_id": p.patient_id,
            "risk_score": p.risk_score,
            "risk_label": p.risk_label,
            "model_version": p.model_version,
            "created_at": p.created_at.isoformat(),
        }
        for p in rows
    ]


@app.get("/api/predictions/{prediction_id}")
def get_prediction(
    prediction_id: int,
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """One prediction in detail (including the SHAP explanation)."""
    p = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if p is None or (user.role != "admin" and p.user_id != user.id):
        raise HTTPException(404, "Prediction not found")
    patient = db.query(Patient).filter(Patient.id == p.patient_id).first()
    return {
        "id": p.id,
        "risk_score": p.risk_score,
        "risk_label": p.risk_label,
        "key_factors": pretty_names(p.shap_values.get("key_factors", [])) if p.shap_values else [],
        "model_version": p.model_version,
        "created_at": p.created_at.isoformat(),
        "patient": {col: getattr(patient, col) for col in FEATURE_COLS} if patient else None,
    }


# --------------------------------------------------------------------------- #
# 7. Patients (needs a token)
# --------------------------------------------------------------------------- #
@app.get("/api/patients")
def list_patients(user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """Admins see all patients; doctors see the ones they entered."""
    q = db.query(Patient)
    if user.role != "admin":
        q = q.filter(Patient.user_id == user.id)
    rows = q.order_by(Patient.created_at.desc()).limit(200).all()
    return [
        {
            "id": p.id,
            "created_at": p.created_at.isoformat(),
            **{col: getattr(p, col) for col in FEATURE_COLS},
        }
        for p in rows
    ]


# --------------------------------------------------------------------------- #
# 8. Federated-learning rounds (needs a token) - filled by fl/run_fl.py
# --------------------------------------------------------------------------- #
@app.get("/api/fl/rounds")
def fl_rounds(user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """
    The FL Training / Admin pages use this to show real accuracy and epsilon
    (privacy budget) once a federated run has been done. Newest first.
    """
    rows = db.query(FLRound).order_by(FLRound.id.desc()).limit(500).all()
    return [
        {
            "id": r.id,
            "round_number": r.round_number,
            "strategy": r.strategy,
            "global_accuracy": r.global_accuracy,
            "epsilon": r.epsilon,                       # NULL for non-DP runs
            "communication_cost_mb": r.communication_cost_mb,
            "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        }
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# 8b. Federated-learning control (start / stop / status)
# --------------------------------------------------------------------------- #
@app.post("/api/fl/start")
def fl_start(body: FlStartBody, admin: User = Depends(auth.require_admin)):
    """Launch fl/run_fl.py in the background (admins only). Builds the argv
    list from the validated fields only - no shell, no free-text arguments.
    Only one run may be live at a time (otherwise 409)."""
    global _fl_process
    with _fl_lock:
        if _fl_process is not None and _fl_process.poll() is None:
            raise HTTPException(409, "A run is already in progress")

        argv = [VENV_PYTHON, "-u", RUN_FL,                # "-u" = unbuffered log
                "--strategy", body.strategy,
                "--rounds", str(body.rounds)]
        if body.strategy == "fedprox":
            argv += ["--mu", str(body.mu)]
        if body.dp:
            argv += ["--dp", "--clip", str(body.clip), "--noise", str(body.noise)]
        if body.secagg:
            argv += ["--secagg"]

        with open(FL_LOG, "w", encoding="utf-8") as log:  # truncate + start fresh
            _fl_process = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT
            )
        return {"state": "running", "pid": _fl_process.pid}


@app.post("/api/fl/stop")
def fl_stop(admin: User = Depends(auth.require_admin)):
    """Terminate the running federated-learning process (admins only)."""
    global _fl_process
    with _fl_lock:
        if _fl_process is not None and _fl_process.poll() is None:
            _fl_process.terminate()
        _fl_process = None
    return {"state": "idle"}


@app.get("/api/fl/status")
def fl_status(user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """Running or idle + the latest round + the last 20 log lines (any user)."""
    running = _fl_process is not None and _fl_process.poll() is None
    latest = db.query(FLRound).order_by(FLRound.id.desc()).first()
    tail = []
    if os.path.exists(FL_LOG):
        with open(FL_LOG, "r", encoding="utf-8", errors="replace") as f:
            tail = f.read().splitlines()[-20:]
    return {
        "state": "running" if running else "idle",
        "latest_round": latest.round_number if latest else None,
        "log": tail,
    }


# --------------------------------------------------------------------------- #
# 9. Fairness metrics (needs a token) - filled by ml/fairness.py
# --------------------------------------------------------------------------- #
@app.get("/api/eval/fairness")
def eval_fairness(user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """Per-group fairness (hospital / sex / age) from ml/fairness.py."""
    rows = (db.query(FairnessMetric)
            .order_by(FairnessMetric.group_type, FairnessMetric.group_value).all())
    return [
        {
            "group_type": r.group_type,
            "group_value": r.group_value,
            "recall": r.recall,
            "f1": r.f1,
            "auc": r.auc,
            "n": r.n,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
