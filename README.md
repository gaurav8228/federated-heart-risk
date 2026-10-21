# HeartRisk AI

A full-stack heart-disease risk prediction system with **federated learning**,
**SHAP explainability**, **differential privacy**, and **fairness analysis**.
Predict a patient's risk from 13 clinical fields, see the SHAP "why", train a
shared model across three simulated hospitals without pooling patient data, and
audit both privacy budget and per-group fairness.

## Architecture

| Layer | Tech |
|---|---|
| Frontend | React + Vite (plain JS, React Router, recharts), JWT in localStorage |
| Backend | FastAPI (Python), PBKDF2 password hashing, JWT auth, CORS |
| Database | PostgreSQL (SQLAlchemy ORM) — users, hospitals, patients, predictions, fl_rounds, model_versions, fairness_metrics |
| ML | scikit-learn + XGBoost pipelines, numpy logistic regression, SHAP |
| Federated learning | Flower (real server + 3 threaded clients) — FedAvg & FedProx |
| Privacy | Differential privacy: client-side update clipping + Gaussian noise, plain-numpy RDP accountant for ε |
| Fairness | Per-group recall / F1 / AUC / n on the held-out test set (hospital, sex, age) |

## Setup
## Requirements

Python 3.13, Node.js 22, PostgreSQL 18 (with pgAdmin). Tested on Windows.

## Setup

1. **Get the code and install Python packages**
```bat
git clone <repo-url>
cd heartRisk
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
```

2. **Create the database.** In pgAdmin: right-click Databases, then Create, then Database, and name it `heartrisk`.

3. **Create `.env`** in the project root (copy `.env.example`, and use your own password and port):
```
DATABASE_URL=postgresql+psycopg2://<user>:<password>@localhost:<port>/heartrisk
SECRET_KEY=<any-long-random-string>
```
Never commit this file. Check your PostgreSQL port in pgAdmin (default 5432).

4. **Download the data and build the models** (model files are not stored in git):
```bat
venv\Scripts\python.exe ml\download_data.py
venv\Scripts\python.exe ml\train.py
venv\Scripts\python.exe ml\explain.py
```

5. **Seed the database and start the backend**
```bat
venv\Scripts\python.exe -m backend.seed
venv\Scripts\uvicorn.exe backend.main:app --reload
```

6. **Start the frontend** in a second terminal:
```bat
cd frontend
npm install
npm run dev
```
Open **http://localhost:5173**.

7. **Check everything works** (backend must be running):
```bat
venv\Scripts\python.exe backend\test_api.py
```
It should end with `ALL TESTS PASSED`.

## Accounts

Anyone can sign up from the login page; signup always creates a **doctor**.
Admin accounts come from the seed script or an existing admin.

## Admin: running FL from the UI

Log in as admin, open the Admin page, choose FedAvg or FedProx, set the rounds,
optionally tick DP and SecAgg, and press Start. The chart updates live; Stop halts the run.
You can also run FL from the terminal (below).

## Running federated learning

```bat
:: FedAvg
venv\Scripts\python.exe fl\run_fl.py --strategy fedavg --rounds 20 --mu 0.0

:: FedProx
venv\Scripts\python.exe fl\run_fl.py --strategy fedprox --rounds 20 --mu 0.1

:: FedAvg with differential privacy (clip + Gaussian noise)
venv\Scripts\python.exe fl\run_fl.py --strategy fedavg --rounds 20 --dp --clip 1.0 --noise 1.0

:: FedAvg with secure aggregation (pairwise-masking simulation; works with or without --dp)
venv\Scripts\python.exe fl\run_fl.py --strategy fedavg --rounds 20 --secagg

:: SecAgg self-test (masked sum == plain sum)
venv\Scripts\python.exe fl\secagg.py

:: privacy / utility sweep (several noise levels, fast, no server)
venv\Scripts\python.exe fl\dp_sweep.py
```

Each run writes one row per round to `fl_rounds` (accuracy, ε, communication
cost). The FL Training page and the Admin page read them via `GET /api/fl/rounds`.

## Running fairness analysis

```bat
venv\Scripts\python.exe -m ml.fairness
```

Prints per-group tables (n, recall, F1, AUC, accuracy), warns when a group has
fewer than 30 patients, and saves rows to `fairness_metrics`. The Admin page
reads them via `GET /api/eval/fairness`.

## Demo logins

| Role | Email | Password |
|---|---|---|
| Doctor | `doctor@hospital.com` | `Doctor@123` |
| Admin | `admin@hospital.com` | `Admin@123` |

## Limitations

- **~250 patients per hospital.** The UCI dataset has only 920 rows split across
  three hospitals, so each client trains on roughly 240–250 patients. Metrics are
  correspondingly noisy, and some fairness groups (e.g. a single hospital + sex +
  age slice) fall under 30 patients.
- **Simulated federated learning on one machine.** All three hospital clients run
  as threads on this machine against a local Flower server. There is no real
  network, no real multi-node deployment, and no hospital-to-hospital isolation
  beyond the code structure.
- **Simple RDP accountant.** The privacy accountant in `fl/dp.py` is a
  from-scratch, single-mechanism Gaussian RDP composition — a readable teaching
  implementation, not a library like Google's DP libraries. Treat the reported ε
  as indicative, not a formal guarantee.
- **SecAgg is a simulation, not a cryptographic protocol.** `fl/secagg.py`
  implements the pairwise-masking building block only: clients add/subtract a
  shared per-pair mask so the masks cancel in the summed update. There is **no
  key exchange and no dropout handling**, so it demonstrates the idea rather
  than providing real secure aggregation.
- **DP lowers accuracy.** Adding noise to updates visibly reduces test accuracy
  (the sweep trades ε for accuracy), which is expected but worth knowing before
  relying on a private run for clinical numbers.
- **Docker is future work.** The app runs as three separate services (React, FastAPI, PostgreSQL) started directly on the host; containerization is planned.
