# HeartRisk AI — Progress Log

Read SPEC.md for the full specification. Rules: one milestone at a time, run
commands + a test after each, keep code simple and commented.

---

## M1 — ML pipeline  ✅ DONE

**What was built**

| File | What it does |
|---|---|
| `ml/download_data.py` | Downloads the 4 UCI Heart Disease files (Cleveland 303 + Hungarian 294 + Switzerland 123 + VA Long Beach 200 = **920 patients**) into `data/raw/`, then merges them into `data/heart_combined.csv` with a **`hospital` column (A / B / C)** for the later federated split. |
| `ml/preprocess.py` | Loads the data, makes the binary target (0 = healthy, 1 = disease), splits 80/20 (stratified, seed 42), builds and **fits the preprocessing pipeline on the training split only**: numeric → median imputer + standard scaler, categorical → most-frequent imputer + one-hot encoder. Saves `ml/models/preprocessor.joblib`. |
| `ml/train.py` | Trains **Logistic Regression, Random Forest, XGBoost, ANN (MLPClassifier)** (no PyTorch/TensorFlow), prints one comparison table (accuracy, precision, recall, F1, AUC), saves all 4 models + `best_model.joblib` (best by AUC) + `ml/models/metrics.json`. |

**Data notes**
- Cholesterol / resting BP values of `0` in the Switzerland/VA files mean "unknown" → converted to missing, then imputed.
- `source` and `hospital` are **not** model features (the model must not learn which hospital a patient is from).
- Patients per hospital: A = 307, B = 307, C = 306 (enough for 3 × 300 in M5).

**Results (test set, 184 patients)**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| logistic_regression | 0.8152 | 0.8269 | 0.8431 | 0.8350 | 0.8944 |
| random_forest | 0.7989 | 0.8037 | 0.8431 | 0.8230 | 0.9055 |
| xgboost | 0.8315 | 0.8257 | 0.8824 | 0.8531 | 0.8914 |
| **ann (BEST by AUC)** | 0.8098 | 0.8131 | 0.8529 | 0.8325 | **0.9059** |

**Also created:** `.gitignore` (ignores `.env`, `venv/`, `__pycache__/`, `ml/models/*.joblib`, `node_modules/`).

**How to run M1**

```bat
cd D:\heartRisk
venv\Scripts\python.exe ml\train.py
```

**Test:** the run prints the comparison table above, `Best model: ann | AUC = 0.9059`,
and creates `ml\models\best_model.joblib` + `ml\models\metrics.json`.

---

## M2 — SHAP explainability  ✅ DONE

**What was built** — `ml/explain.py` (explains **XGBoost**, not the ANN, because
TreeExplainer gives fast, exact SHAP values):

| Item | Output |
|---|---|
| `active_model.joblib` | Copy of the XGBoost pipeline (preprocessor + classifier) → `ml/models/active_model.joblib`. **The FastAPI backend will use this file for predictions in M3.** |
| `explain_patient(patient_dict)` | Function returning JSON: `risk_score` (0–100), `risk_label` (`Low` < 40, `Moderate` 40–65, `High` > 65), `key_factors` = top 5 features by \|SHAP\| with readable names (Age, Max Heart Rate, Cholesterol, Chest Pain Type, ST Depression …) and signed values. One-hot columns are merged back into their original feature. |
| Summary plot | `ml/explain/shap_summary.png` — every feature's impact on the whole test set (beeswarm). |
| Waterfall plot | `ml/explain/waterfall_plot.png` — step-by-step contributions for one example patient (modern version of the force plot; SHAP's matplotlib force plot overlaps its labels when 13 features are shown). |
| Deletion test | For each test patient: replace the top SHAP feature with the training median/mode, re-predict, average the risk drop over the 102 sick test patients → **+0.028 → PASS**. |

**Results (sample patient: 52, male, typical angina, BP 120, chol 240, max HR 150)**
```json
{
  "risk_score": 51.5,
  "risk_label": "Moderate",
  "key_factors": [
    {"feature": "Chest Pain Type", "value": -0.5554},
    {"feature": "Max Heart Rate",  "value":  0.3680},
    {"feature": "Exercise Angina", "value": -0.3108},
    {"feature": "ST Depression",   "value":  0.3047},
    {"feature": "Sex",             "value":  0.3004}
  ]
}
```

**How to run M2**

```bat
cd D:\heartRisk
venv\Scripts\python.exe ml\explain.py
```

**Test:** prints the JSON above, saves both PNGs in `ml\explain\`, and prints `PASS`
for the deletion test.

---

## M3 — FastAPI backend  ✅ DONE

**What was built** — `backend/` (small, commented files):

| File | What it does |
|---|---|
| `backend/database.py` | SQLAlchemy engine + session. Reads `DATABASE_URL` from `D:\heartRisk\.env` (postgres port **8228**, not 5432 — `.env` is the source of truth). |
| `backend/models.py` | All 7 tables: `users`, `hospitals`, `patients`, `predictions`, `fl_rounds`, `model_versions`, `fairness_metrics`. Created automatically on server startup. |
| `backend/auth.py` | Password hashing with **PBKDF2-SHA256** (stdlib `hashlib`, no extra package), JWT signed with `SECRET_KEY` from `.env`, **8-hour expiry**, roles `doctor` / `admin`. |
| `backend/main.py` | FastAPI app. Loads `ml/models/active_model.joblib` **once at startup**, creates tables, CORS for `http://localhost:5173`. |
| `backend/seed.py` | Creates the 2 demo users, 3 hospitals (A/B/C) and registers the XGBoost model in `model_versions`. |
| `backend/test_api.py` | 11-check smoke test (run it while the server is up). |
| `requirements.txt` | Pinned versions of everything — **all packages were already installed, nothing new to pip install**. |

**Endpoints**

| Method | Path | Who | Notes |
|---|---|---|---|
| GET | `/health` | anyone | `{"status": "ok", "model_loaded": true}` |
| POST | `/api/auth/login` | anyone | accepts JSON **and** form (so both React and Swagger's Authorize button work) |
| GET | `/api/auth/me` | JWT | current user |
| POST | `/api/auth/register` | **admin only** | create doctor/admin accounts |
| POST | `/api/predict` | JWT | saves patient + prediction (SHAP JSON) to PostgreSQL, returns risk + key factors |
| GET | `/api/predictions` | JWT | history list (doctors see their own, admins see all) |
| GET | `/api/predictions/{id}` | JWT | one prediction incl. SHAP + patient data |
| GET | `/api/patients` | JWT | patient list |

**How to run M3**

```bat
cd D:\heartRisk
venv\Scripts\python.exe -m backend.seed
venv\Scripts\uvicorn.exe backend.main:app --reload
```

Then open **http://localhost:8000/docs**, click **Authorize**, log in as
`doctor@hospital.com` / `Doctor@123`, expand `POST /api/predict`, click
"Try it out", paste the sample patient and Execute:

```json
{"age": 52, "sex": 1, "cp": 1, "trestbps": 120, "chol": 240, "fbs": 1,
 "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.5, "slope": 2,
 "ca": 0, "thal": 3}
```

Expected answer: `"risk_score": 51.5, "risk_label": "Moderate"` with 5 `key_factors`.
Server startup was verified here (`Application startup complete`).

---

## M3+ — Small fix: identical feature names  ✅ DONE

Both `POST /api/predict` and `GET /api/predictions/{id}` now pass their
`key_factors` through one shared helper `pretty_names()` in `backend/main.py`,
which forces every name to the single source of truth (`READABLE` in
`ml/explain.py`). "Exercise Angina" can never come back as "ExerciseAngina".

---

## M4 — React frontend  ✅ DONE

**What was built** — `frontend/` (Vite + JavaScript, React Router, plain CSS,
recharts — no other libraries). API base URL `http://localhost:8000`, JWT in
`localStorage`.

| File | What it does |
|---|---|
| `src/api.js` | One fetch wrapper: attaches the Bearer token, saves/clears the login, auto-logout on 401, friendly error messages. |
| `src/App.jsx` | Routes. No token → redirect to `/login`. |
| `src/styles.css` | All styling: blue & white medical theme, dark navy sidebar. |
| `src/components/Layout.jsx` | Sidebar with Dashboard / New Prediction / History / FL Training / Settings (+ **Admin** for admins), doctor's name on top, logout button. |
| `src/components/ResultPanel.jsx` | Big risk score + Low/Moderate/High badge, Key Factors bars, SHAP bar chart (red = pushes risk up, blue = pushes down). |
| `src/components/FlPanel.jsx` | FL Training Status card (chip, Round, Global Accuracy, ε, Communication Cost, accuracy-over-rounds chart, 3 hospital clients). |
| `src/pages/Login.jsx` | HeartRisk AI logo + heart icon, email, password, "Don't have an account? Contact admin". |
| `src/pages/Predict.jsx` | All 13 fields (numbers + labelled dropdowns with the codes the backend expects), "Predict Risk" button. |
| `src/pages/History.jsx` | Table: Date & Time, Risk Score, Result badge, View button. |
| `src/pages/PredictionDetail.jsx` | Full saved result (SHAP + patient data) via `GET /api/predictions/{id}`. |
| `src/pages/Dashboard.jsx` | Counts (total / high / moderate / low) + recent predictions. |
| `src/pages/FlTraining.jsx`, `Admin.jsx` | Full FL layout from the diagram using clearly labelled **MOCK data** in `src/mock/flData.js` (how to swap in the real API in M8 is documented inside that file). Admin has FedAvg/FedProx selector + Start/Stop buttons. |
| `src/pages/Settings.jsx` | Profile from `GET /api/auth/me` + logout button. |

`npm install` and `npm run build` were verified (build succeeds).

**How to run M4** (backend first, then frontend — two terminals):

```bat
:: terminal 1
cd D:\heartRisk
venv\Scripts\uvicorn.exe backend.main:app --reload

:: terminal 2
cd D:\heartRisk\frontend
npm run dev
```

**Test:** open **http://localhost:5173**, log in as
`doctor@hospital.com` / `Doctor@123` → Dashboard loads → **New Prediction**
→ **Predict Risk** → see **51.5% Moderate** with the SHAP chart → **History**
→ the same prediction is in the table → **View** opens it again.

---

## M5 — Flower federated learning (FedAvg + FedProx)  ✅ DONE

**What was built** — `fl/` (small, commented files):

| File | What it does |
|---|---|
| `fl/model.py` | Logistic regression in **numpy** (weights + bias arrays, gradient descent). FedAvg cannot average XGBoost trees, so the FL model is logistic regression. FedProx adds the `mu/2 * ‖w − w_global‖²` proximal term to the local loss. |
| `fl/data.py` | Loads `data/heart_combined.csv`, holds out a shared **stratified 20% test set (184 patients) that no client trains on**, gives each hospital up to 300 of its own pool patients, and fits the `ml/preprocess.py` pipeline on the union of client training data only. |
| `fl/run_fl.py` | The runner. Flower **server + 3 hospital client threads** (Ray is not installed, so the simulation mode is unavailable — but the real server + threaded clients works fine on Windows, no fallback needed). Prints per-round accuracy/F1/AUC/comm-cost, writes each round to `fl_rounds`, and prints a centralized-vs-FedAvg-vs-FedProx comparison. |

**Data split** (920 patients): held-out test 184 · Hospital A 250 · B 238 · C 248.
*Note:* 3 × 300 training + a disjoint 20 % test set would need 1084 patients, which
the 920-row dataset does not have. The important rule — the test set is never
trained on — is kept, so each hospital uses every pool patient it has (~240–250).

**Communication cost** is tiny (≈ 0.0013 MB/round) because a logistic model has
only 29 numbers (28 weights + 1 bias), unlike a big neural net.

**How to run M5**

```bat
cd D:\heartRisk
venv\Scripts\python.exe fl\run_fl.py --strategy fedavg  --rounds 20 --mu 0.0
venv\Scripts\python.exe fl\run_fl.py --strategy fedprox --rounds 20 --mu 0.1
```

**Results (20 rounds each, held-out test set)**

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Centralized LR (all data) | 0.7989 | 0.8230 | 0.8979 |
| FedAvg | 0.7989 | 0.8230 | 0.8977 |
| FedProx (μ=0.1) | 0.7989 | 0.8230 | 0.8974 |

AUC climbs from 0.35 (untrained) to 0.90 over 20 rounds. All three are nearly
identical because logistic regression is convex — FedAvg converges to the same
solution as centralized training, which is exactly what we want to see.

**Test:** each run prints per-round lines and the comparison table; the
`fl_rounds` table in pgAdmin gains **21 rows per run** (rounds 0–20) with
`strategy` = fedavg / fedprox, `global_accuracy`, `epsilon` = NULL, and
`communication_cost_mb` ≈ 0.0013. (Both strategies were verified here — 42 rows.)

---

## M6 — Differential privacy (clip + noise + ε)  ✅ DONE

**What was built**

| File | What it does |
|---|---|
| `fl/dp.py` | Two functions, plain numpy: `clip_and_noise()` (the client-side DP step) and `epsilon_for_rounds()` (a small **RDP accountant** — full formula + its limits documented in the file's comments). No Opacus/TF-Privacy. |
| `fl/run_fl.py` | New flags `--dp`, `--clip` (default 1.0), `--noise` (default 1.0), `--delta` (default 1e-5). When `--dp` is on, each hospital clips its weight update to `--clip`, adds Gaussian noise (`std = noise × clip`) before sending, and ε is printed and stored after every round. |
| `fl/dp_sweep.py` | Runs FedAvg (plain loop, no server → fast) with no DP and 4 noise levels, prints a privacy-vs-utility table. |
| `backend/main.py` | New endpoint **`GET /api/fl/rounds`** (JWT required) returns the `fl_rounds` rows so the frontend can show real ε in M8. Frontend untouched. |

**How the DP works** — after local training each hospital computes
`update = local − global`, clips it so its L2 norm ≤ `--clip`, adds
Gaussian noise (`std = noise × clip`), and returns `global + noisy update`.
The server just averages these. The accountant treats each round as one
Gaussian mechanism (noise multiplier σ = `--noise`) and composes over rounds
with RDP → `(ε, δ)` conversion, `δ = 1e-5`. Epsilon depends only on
(rounds, σ, δ); `clip` drives the accuracy-vs-noise trade-off, not ε.

**How to run M6**

```bat
cd D:\heartRisk
venv\Scripts\python.exe fl\run_fl.py --strategy fedavg --rounds 20 --dp --clip 1.0 --noise 1.0
venv\Scripts\python.exe fl\dp_sweep.py
```

**Privacy-vs-utility sweep (20 rounds, clip=1.0, δ=1e-5)**

| Noise | ε | Accuracy | F1 | AUC |
|---|---|---|---|---|
| 0.0 (no DP) | ∞ | 0.7989 | 0.8230 | 0.8977 |
| 0.5 | 91.51 | 0.7337 | 0.7200 | 0.8077 |
| 1.0 | 31.51 | 0.6739 | 0.6341 | 0.7478 |
| 2.0 | 13.26 | 0.6250 | 0.5548 | 0.7011 |
| 4.0 | 6.00 | 0.5652 | 0.4444 | 0.6521 |

Higher noise = more private (smaller ε) but less accurate — exactly the
trade-off differential privacy is about.

**Test:** run the DP command, then in pgAdmin query
`select round_number, global_accuracy, epsilon from fl_rounds where epsilon is not null order by id;`
— the DP rows show ε climbing from 0.00 to ~31.5; non-DP runs keep `epsilon` NULL.
(Verified here: DP rows have ε set, non-DP rows NULL.)

---

## M7 — Fairness analysis (hospital, sex, age)  ✅ DONE (LITE)

`ml/fairness.py` rebuilds the same held-out test set as `fl/data.py`, scores the
XGBoost active model only, and prints per-group tables (n, recall, F1, AUC, accuracy)
for hospital A/B/C, sex, and age (<45, 45-60, >60), warning on groups under 30 and
setting AUC to null for single-class groups, plus each group type's largest recall gap.
It saves rows to `fairness_metrics` (replacing a group's old rows each run; no schema
change needed). New endpoint `GET /api/eval/fairness` (JWT required) returns them.

**Run:** `venv\Scripts\python.exe -m ml.fairness`

---

## M8 — Frontend FL + fairness pages  ✅ DONE (LITE)

`src/mock/flData.js` now fetches real data — `fetchFlRuns()` calls `GET /api/fl/rounds`
and groups the per-round rows into runs (a new run starts when `round_number` resets to
0), and `fetchFairness()` calls `GET /api/eval/fairness`. The FL Training and Admin pages
show the selected run (round, global accuracy, ε, total communication cost, accuracy
line chart) with a **run dropdown**; ε renders as "Not private" when `null`. The
FedAvg/FedProx selector and Start/Stop buttons stay visible but **disabled**, labelled
"Run from terminal: fl\run_fl.py". Admin adds a **Fairness** section (one table per
hospital/sex/age with n, recall, F1, AUC and an amber "small n" badge under 30). Friendly
empty states cover no-runs / no-fairness. Small backend addition so the tables can show
group size: `fairness_metrics.n` column added to the model, saved by `ml/fairness.py`
(also `ALTER TABLE ... IF NOT EXISTS` at server start) and returned by the endpoint —
**re-run `venv\Scripts\python.exe -m ml.fairness` once to populate `n`**. `npm run build`
passes.

**Run:** `cd D:\heartRisk\frontend` then `npm run dev`.

---

## M9 — Packaging / README  ⬅ NEXT
