# HeartRisk AI — System Specification

Full-stack federated-learning heart-disease risk system.
Source of truth: architecture diagram `docs/HeartRisk AI Federated Learning System.png`.

**No code is written yet.** This file lists everything the system must contain.

---

## 0. Technology Stack (fixed by environment)

| Layer | Technology |
|---|---|
| Frontend | React (Node 22) |
| Backend | FastAPI + Uvicorn, PyJWT auth, python-dotenv |
| Database | PostgreSQL 18, `localhost:5432`, DB `heartrisk`, SQLAlchemy + psycopg2-binary |
| ML | pandas, scikit-learn, XGBoost, shap, joblib, numpy, matplotlib |
| Federated Learning | Flower (`flwr`) — FedAvg + FedProx |
| Secrets | `D:\heartRisk\.env` → `DATABASE_URL`, `SECRET_KEY` |
| Deployment | Docker Desktop (Dockerfile / docker-compose) |
| Not used | PyTorch, TensorFlow (disk space limit) — the "ANN (NN)" model must be a pure-numpy / scikit-learn MLP |

Repository root: `D:\heartRisk`, Python venv: `D:\heartRisk\venv`.
`.gitignore` must include at least: `.env`, `venv/`.

---

## 1. Components

### 1.1 Frontend (React) — "What the doctor sees and uses"
- Single-page app with a left sidebar navigation (Doctor/Admin identity at top): **Dashboard**, **New Prediction**, **History**, **FL Training**, **Settings**.
- Talks to the FastAPI backend over HTTP (REST + JWT bearer token).
- Components:
  - `App` / router / auth context (login state, role: doctor | admin)
  - `LoginPage`
  - `PredictionForm` (patient data form)
  - `PredictionResult` (risk score, key factors, SHAP chart)
  - `HistoryPage` (prediction table + View action)
  - `FLTrainingStatusPage` (rounds, accuracy, ε, communication cost, accuracy-over-rounds chart)
  - `AdminPage` (start/monitor FL training, control API UI)
  - `Sidebar` / `Layout` / shared UI (buttons, badges, tables, charts)

### 1.2 Backend (FastAPI) — "Handles requests, authentication and model inference"
Four internal services:
1. **Auth & Security** — JWT login and roles (doctor, admin).
2. **Prediction API** — `POST /predict`, returns risk score + SHAP explanation.
3. **FL Control API** — start / check training rounds (admin only).
4. **Database Service** — save users, patients, predictions, FL logs.

Other backend pieces:
- ML model loading service (loads trained `joblib` model + preprocessing artifacts).
- SHAP explainer service (computes per-prediction feature contributions).
- Static/frontend hosting (in Docker, backend or nginx serves the built React app).

### 1.3 Database (PostgreSQL) — "Stores users, patients, predictions and FL results"
See §3.

### 1.4 Federated Learning Layer (Flower)
- **Global Model** — initial model produced by the local ML pipeline, sent to clients.
- **Hospital A / Client 1** — local data (300 patients), trains locally for N epochs, sends model updates (not raw data).
- **Hospital B / Client 2** — local data (300 patients), trains locally, sends updates.
- **Hospital C / Client 3** — local data (300 patients), trains locally, sends updates.
- **Federated Server** — aggregates updates with **FedAvg** or **FedProx**, plus **+ DP** (differential privacy) and **+ SecAgg** (secure aggregation), writes per-round metrics to the DB.
- **Improved Global Model** — sent back to clients; loop repeats for multiple rounds.

### 1.5 ML Models & Pipeline (Python) — training, evaluation, explainability, fairness
Ordered stages:
1. **Data Preprocessing** — handle missing values, encode categoricals, scale features.
2. **Model Training** — Logistic Regression, Random Forest, XGBoost, ANN (numpy/sklearn MLP).
3. **Federated Learning** — FedAvg, FedProx strategies.
4. **Differential Privacy** — clip gradients, add Gaussian noise, track ε.
5. **SHAP Explainability** — summary plots, force plots, deletion test.
6. **Fairness Evaluation** — recall / F1 / AUC per client and per group; sex & age analysis.
7. **Model Evaluation** — accuracy, precision, recall, F1, AUC.

### 1.6 Key Features (from diagram)
1. **Federated Learning** — hospitals train locally, share model updates only.
2. **Differential Privacy** — adds noise to updates, controls ε (privacy budget).
3. **SHAP Explanations** — shows which features influenced the prediction.
4. **Fairness Analysis** — compare performance across hospitals, sex and age groups.
5. **Full-Stack Web App** — React frontend + FastAPI backend + PostgreSQL.
6. **Docker & Deployment** — easy setup and scaling.

### 1.7 Data Flow Summary (4 steps)
1. Doctor enters data (React).
2. Backend processes (FastAPI).
3. AI model predicts (Python + SHAP).
4. Result shown (React UI).
   Meanwhile, FL runs in the background (Flower) across hospital clients.

---

## 2. Pages / Screens (Frontend)

| # | Page | Route (planned) | Content |
|---|---|---|---|
| 1 | **Login Page** | `/login` | HeartRisk AI logo, email + password, Login button, "Don't have an account? Contact admin". Doctors and admins log in with secure authentication. |
| 2 | **Patient Data Form** ("New Prediction") | `/predict` | Fields: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Max Heart Rate, (plus remaining input features required by the trained model, e.g. resting ECG, exercise-induced angina, ST depression / oldpeak, slope, number of major vessels, thalassemia). "Predict Risk" button. |
| 3 | **Prediction Result** (same route, result panel) | `/predict` (result state) | Big **Risk Score** with category (e.g. `72% High Risk`), **Key Factors (SHAP)** bar list (Age, Max Heart Rate, Cholesterol, Chest Pain Type, ST Depression with signed values), **SHAP Explanation** waterfall/bar chart (values e.g. Age +0.38, Max Heart Rate +0.21, Cholesterol +0.16, Chest Pain Type −0.12, ST Depression −0.09 on a −0.2 … 0.4 axis). |
| 4 | **History Page** | `/history` | Table columns: Date & Time, Risk Score, Result (Low / Moderate / High badge), Actions (View). "View past predictions and reports." |
| 5 | **FL Training Status Page** (sidebar "FL Training") | `/fl-training` | Status card **FL Training Status** with "Training in Progress" chip; stat cards: **Round 12/100**, **Global Accuracy 87.4%**, **ε (Privacy Budget) 5.0**; **Communication Cost 12.3 MB**; **Accuracy over Rounds** line chart (x = rounds 0–100, y = accuracy 0.6–1.0). |
| 6 | **Admin Page** | `/admin` | Monitor FL rounds, accuracy, privacy (ε) and communication cost; start / stop training; select strategy (FedAvg / FedProx) and privacy settings. (Used together with page 5.) |
| 7 | **Dashboard** | `/` | Landing after login with quick links (New Prediction, History, FL Training). |
| 8 | **Settings** | `/settings` | Account/profile settings (minimal). |

Sidebar navigation items: **Dashboard, New Prediction, History, FL Training, Settings** (doctor identity shown at top, e.g. "Dr. Smith").

---

## 3. Database Tables (PostgreSQL)

| Table | Purpose | Key columns (minimum) |
|---|---|---|
| `users` | Doctors and admins | `id`, `email`, `password_hash`, `full_name`, `role` (`doctor`\|`admin`), `created_at` |
| `patients` | One row per patient, per hospital | `id`, `hospital_id`/`client_id`, `age`, `sex`, `chest_pain_type`, `resting_bp`, `cholesterol`, `max_heart_rate`, …other feature columns, `created_at` |
| `predictions` | Risk, SHAP, timestamp | `id`, `user_id`, `patient_id`, `risk_score`, `risk_label` (`Low`\|`Moderate`\|`High`), `shap_values` (JSON), `model_version`, `created_at` |
| `fl_rounds` | FL round log: round, accuracy, ε, cost | `id`, `round_number`, `strategy`, `global_accuracy`, `epsilon`, `communication_cost_mb`, `status`, `started_at`, `finished_at` |
| `model_versions` | Model info | `id`, `name`, `algorithm`, `version`, `metrics` (JSON), `file_path`, `is_active`, `created_at` |
| `hospitals` *(supporting)* | The 3 FL clients | `id`, `name` (Hospital A/B/C), `client_id`, `patient_count` |
| `fairness_metrics` *(supporting)* | Fairness results per group | `id`, `model_version`, `group_type` (`sex`\|`age`\|`hospital`), `group_value`, `recall`, `f1`, `auc` |

---

## 4. API Endpoints (FastAPI)

### Auth
| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/auth/register` | admin | Create doctor account (contact-admin flow) |
| POST | `/api/auth/login` | public | Email + password → returns JWT (access token, role) |
| GET | `/api/auth/me` | JWT | Current user info |

### Prediction
| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/predict` | JWT | Body = patient features → returns `risk_score`, `risk_label`, `key_factors`, `shap_values`, `prediction_id` (diagram: `POST /predict`) |
| GET | `/api/predictions` | JWT | List current user's prediction history (pagination, optional filters) |
| GET | `/api/predictions/{id}` | JWT | Single prediction detail (risk + SHAP) for History "View" |

### Patients
| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/patients` | JWT | Create/store patient record |
| GET | `/api/patients` | JWT | List patients (per hospital for admins) |

### Federated Learning control
| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/fl/start` | admin | Start FL training (params: strategy `fedavg`\|`fedprox`, rounds, ε, hospitals) — diagram "Start training rounds" |
| POST | `/api/fl/stop` | admin | Stop / pause a running FL job |
| GET | `/api/fl/status` | JWT | Current round, total rounds, global accuracy, ε, communication cost, status (`Training in Progress`) — diagram "Check training rounds" |
| GET | `/api/fl/rounds` | JWT | History of `fl_rounds` rows (for accuracy-over-rounds chart) |
| GET | `/api/fl/strategy` | JWT | Active strategy (FedAvg / FedProx) and DP settings |

### Model & evaluation
| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/models` | JWT | List `model_versions` + metrics (accuracy, precision, recall, F1, AUC) |
| GET | `/api/models/active` | JWT | Currently active model |
| GET | `/api/eval/fairness` | JWT | Fairness metrics per hospital / sex / age group |
| GET | `/api/eval/shap/{prediction_id}` | JWT | SHAP explanation for one prediction (summary / force data) |

### System
| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | public | Service health check |
| GET | `/api/summary` | JWT | Dashboard counts (predictions today, active model, FL status) |
| GET | `/docs`, `/openapi.json` | public | Auto API docs (FastAPI default) |
| `*` | `/assets/*`, `/` (SPA) | public | Serve built React app (production) |

---

## 5. Features checklist

### Core (M1–M4)
- [ ] F1 Data preprocessing: missing values, categorical encoding, feature scaling (fit on train only).
- [ ] F2 Train 4 model types: Logistic Regression, Random Forest, XGBoost, ANN (MLP — no PyTorch/TF).
- [ ] F3 Model evaluation: accuracy, precision, recall, F1, AUC; pick best; save with `joblib` + metadata into `model_versions`.
- [ ] F4 Persist trained artifacts under `ml/models/` and preprocessing pipeline (scaler/encoder).
- [ ] F5 SHAP explainability: per-prediction values for the UI key-factors list + summary plot (matplotlib PNG) and force-plot data (M2).
- [ ] F6 SHAP deletion test (sanity check that removing top feature lowers confidence).
- [ ] F7 FastAPI backend: JWT login with roles (doctor/admin), bcrypt-style password hashing.
- [ ] F8 `POST /api/predict` returning risk score + SHAP.
- [ ] F9 History endpoints backed by PostgreSQL (`users`, `patients`, `predictions`).
- [ ] F10 React frontend with the 8 pages in §2, sidebar navigation, charts (accuracy line, SHAP bars).

### Federated learning (M5)
- [ ] F11 Flower server + 3 hospital clients (Hospital A/B/C), each with **300 patients**.
- [ ] F12 Federated strategy **FedAvg**.
- [ ] F13 Federated strategy **FedProx** (μ parameter selectable).
- [ ] F14 Local training epochs per round; model updates only (never raw data).
- [ ] F15 Round metrics written to `fl_rounds` (round, accuracy, ε, communication cost).
- [ ] F16 Improved global model distributed back to clients each round.

### Privacy & fairness (M6–M7)
- [ ] F17 Differential privacy: gradient/update clipping, Gaussian noise, ε accounting displayed in UI.
- [ ] F18 (Optional per diagram) Secure aggregation note/aggregation alongside DP.
- [ ] F19 Fairness evaluation: recall / F1 / AUC **per client (hospital)** and **per group**.
- [ ] F20 Sex analysis and age-group analysis reported (page + `/api/eval/fairness`).

### Control & ops (M8–M9)
- [ ] F21 FL Control API: start / stop / status / rounds history (admin role).
- [ ] F22 Admin page: monitor FL rounds, accuracy, privacy ε, communication cost; start training; pick FedAvg/FedProx.
- [ ] F23 Docker: Dockerfiles (backend, frontend), `docker-compose.yml` with PostgreSQL, one-command startup.
- [ ] F24 `.gitignore` with `.env` and `venv`.
- [ ] F25 `.env` handling: `DATABASE_URL`, `SECRET_KEY` read via python-dotenv (never committed).

---

## 6. Milestone plan (stop after each one)

| Milestone | Deliverable | Command + test |
|---|---|---|
| **M1** | ML pipeline (preprocess → train LR/RF/XGB/ANN → evaluate → save best model) | run script; test: prints metrics table + saved model file |
| **M2** | SHAP explanations (values, summary plot, deletion test) | run script; test: prints top factors + PNG chart |
| **M3** | FastAPI backend (auth, predict, history, DB tables) | `uvicorn` command; test: login + `POST /api/predict` via `/docs` |
| **M4** | React frontend (8 pages, wired to backend) | `npm run dev`; test: log in, predict, see history |
| **M5** | Flower FL: FedAvg + FedProx, 3 hospitals × 300 patients | run FL script; test: rounds printed + `fl_rounds` rows |
| **M6** | Differential privacy (clip + noise + ε tracking) | run FL with DP; test: ε shown in status API/UI |
| **M7** | Fairness analysis (hospital, sex, age) | run eval script; test: fairness report/API output |
| **M8** | FL Control API + admin page | start training from UI; test: status updates live |
| **M9** | Docker (compose up, full stack) | `docker compose up --build`; test: app at localhost |

Each milestone ends with: exact run commands, a manual test, and an updated `PROGRESS.md`.

---

## 7. Out of scope / constraints

- No PyTorch, no TensorFlow (use scikit-learn `MLPClassifier` + numpy for ANN).
- No real PHI data — synthetic / open dataset (e.g. UCI heart-disease style, 13 features) for the 3×300 FL patients.
- Keep code simple and commented for a non-programmer.
