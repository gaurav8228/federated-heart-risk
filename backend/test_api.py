r"""
test_api.py  --  Milestone M3 smoke test (run while the server is up)

    venv\Scripts\python.exe backend\test_api.py
"""
import sys
import uuid

import httpx

BASE = "http://localhost:8000"

SAMPLE_PATIENT = {
    "age": 52, "sex": 1, "cp": 1, "trestbps": 120, "chol": 240,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3,
}


def main():
    c = httpx.Client(base_url=BASE, timeout=30)
    ok = True
    # unique suffix so user-creating tests pass on EVERY run (no duplicate email)
    tag = uuid.uuid4().hex[:8]
    dr2_email = f"dr2_{tag}@hospital.com"
    signup_email = f"signup_{tag}@hospital.com"
    sneaky_email = f"sneaky_{tag}@hospital.com"

    def check(label, condition, extra=""):
        nonlocal ok
        print(("PASS  " if condition else "FAIL  ") + label + ("  " + str(extra) if extra else ""))
        ok = ok and condition

    # 1. health
    r = c.get("/health")
    check("GET /health", r.status_code == 200 and r.json()["status"] == "ok", r.json())

    # 2. login (JSON - what React will use)
    r = c.post("/api/auth/login", json={"email": "doctor@hospital.com", "password": "Doctor@123"})
    check("POST /api/auth/login (JSON)", r.status_code == 200, {k: v for k, v in r.json().items() if k != "access_token"})
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. login (form - what the Swagger "Authorize" button uses)
    r = c.post("/api/auth/login", data={"username": "admin@hospital.com", "password": "Admin@123"})
    check("POST /api/auth/login (form)", r.status_code == 200)
    admin_token = r.json()["access_token"]

    # 4. who am I
    r = c.get("/api/auth/me", headers=headers)
    check("GET /api/auth/me", r.status_code == 200 and r.json()["role"] == "doctor", r.json())

    # 5. wrong password is refused
    r = c.post("/api/auth/login", json={"email": "doctor@hospital.com", "password": "nope"})
    check("wrong password -> 401", r.status_code == 401)

    # 6. no token is refused
    r = c.get("/api/predictions")
    check("no token -> 401", r.status_code == 401)

    # 7. predict (the important one)
    r = c.post("/api/predict", json=SAMPLE_PATIENT, headers=headers)
    check("POST /api/predict", r.status_code == 200, r.json())
    pred_id = r.json().get("prediction_id")

    # 8. history
    r = c.get("/api/predictions", headers=headers)
    check("GET /api/predictions", r.status_code == 200 and len(r.json()) >= 1, f"{len(r.json())} rows")

    # 9. one prediction with SHAP
    r = c.get(f"/api/predictions/{pred_id}", headers=headers)
    check("GET /api/predictions/{id}", r.status_code == 200 and len(r.json()["key_factors"]) == 5,
          r.json().get("key_factors"))

    # 10. patients
    r = c.get("/api/patients", headers=headers)
    check("GET /api/patients", r.status_code == 200 and len(r.json()) >= 1, f"{len(r.json())} rows")

    # 11. register needs admin
    r = c.post("/api/auth/register",
               json={"email": dr2_email, "password": "Test@123",
                     "full_name": "Dr Two", "role": "doctor"}, headers=headers)
    check("register as doctor -> 403", r.status_code == 403)
    r = c.post("/api/auth/register",
               json={"email": dr2_email, "password": "Test@123",
                     "full_name": "Dr Two", "role": "doctor"},
               headers={"Authorization": f"Bearer {admin_token}"})
    check("register as admin -> 201", r.status_code == 201, r.json())

    # 12. public signup creates a doctor and returns a token
    r = c.post("/api/auth/signup",
               json={"full_name": "Dr Signup", "email": signup_email,
                     "password": "Signup@123"})
    check("POST /api/auth/signup -> 201 doctor",
          r.status_code == 201 and r.json()["role"] == "doctor", r.json())

    # 13. duplicate email -> 409 (reuses the SAME email as test 12)
    r = c.post("/api/auth/signup",
               json={"full_name": "Dr Signup", "email": signup_email,
                     "password": "Signup@123"})
    check("signup duplicate email -> 409", r.status_code == 409)

    # 14. sending role=admin cannot make an admin
    r = c.post("/api/auth/signup",
               json={"full_name": "Sneaky", "email": sneaky_email,
                     "password": "Sneaky@123", "role": "admin"})
    check("signup with role=admin still -> doctor",
          r.status_code == 201 and r.json()["role"] == "doctor", r.json())

    # 15. a doctor cannot start an FL run
    r = c.post("/api/fl/start", json={"strategy": "fedavg", "rounds": 1},
               headers=headers)
    check("doctor POST /api/fl/start -> 403", r.status_code == 403)

    # 16. an invalid strategy is rejected (422)
    r = c.post("/api/fl/start", json={"strategy": "nonsense", "rounds": 1},
               headers={"Authorization": f"Bearer {admin_token}"})
    check("invalid strategy -> 422", r.status_code == 422)

    # 17. status is idle or running
    r = c.get("/api/fl/status", headers=headers)
    check("GET /api/fl/status -> idle/running",
          r.status_code == 200 and r.json()["state"] in ("idle", "running"), r.json())

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
