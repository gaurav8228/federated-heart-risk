// New Prediction: the patient form (13 fields) + the result after Predict.
import { useState } from "react";
import { apiFetch } from "../api";
import ResultPanel from "../components/ResultPanel";

// The starting values of the form (the sample patient from the design diagram)
const EMPTY = {
  age: "52", sex: "1", cp: "1", trestbps: "120", chol: "240",
  fbs: "1", restecg: "0", thalach: "150", exang: "0",
  oldpeak: "1.5", slope: "2", ca: "0", thal: "3",
};

export default function Predict() {
  const [form, setForm] = useState(EMPTY);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // one small helper so every field can use the same onChange
  const set = (key) => (e) => setForm({ ...form, [key]: e.target.value });

  async function submit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    setResult(null);
    try {
      // send the numbers; empty string -> null (the model treats it as unknown)
      const body = {};
      Object.entries(form).forEach(([k, v]) => {
        body[k] = v === "" ? null : Number(v);
      });
      setResult(await apiFetch("/api/predict", { method: "POST", body: JSON.stringify(body) }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h1>New Patient Prediction</h1>
      <p>Enter the patient's details and the AI will estimate the heart-disease risk.</p>

      {error && <div className="error-banner">{error}</div>}

      <div className="card">
        <form onSubmit={submit}>
          <div className="form-grid">
            <label className="field">
              <span>Age (years)</span>
              <input type="number" value={form.age} onChange={set("age")} required />
            </label>

            <label className="field">
              <span>Sex</span>
              <select value={form.sex} onChange={set("sex")}>
                <option value="0">0 - Female</option>
                <option value="1">1 - Male</option>
              </select>
            </label>

            <label className="field">
              <span>Chest Pain Type</span>
              <select value={form.cp} onChange={set("cp")}>
                <option value="1">1 - Typical Angina</option>
                <option value="2">2 - Atypical Angina</option>
                <option value="3">3 - Non-Anginal Pain</option>
                <option value="4">4 - Asymptomatic</option>
              </select>
            </label>

            <label className="field">
              <span>Resting BP (mm Hg)</span>
              <input type="number" value={form.trestbps} onChange={set("trestbps")} />
            </label>

            <label className="field">
              <span>Cholesterol (mg/dl)</span>
              <input type="number" value={form.chol} onChange={set("chol")} />
            </label>

            <label className="field">
              <span>Max Heart Rate (bpm)</span>
              <input type="number" value={form.thalach} onChange={set("thalach")} />
            </label>

            <label className="field">
              <span>Fasting Blood Sugar &gt; 120 mg/dl</span>
              <select value={form.fbs} onChange={set("fbs")}>
                <option value="0">0 - No</option>
                <option value="1">1 - Yes</option>
              </select>
            </label>

            <label className="field">
              <span>Resting ECG</span>
              <select value={form.restecg} onChange={set("restecg")}>
                <option value="0">0 - Normal</option>
                <option value="1">1 - ST-T Wave Abnormality</option>
                <option value="2">2 - Left Ventricular Hypertrophy</option>
              </select>
            </label>

            <label className="field">
              <span>Exercise Angina</span>
              <select value={form.exang} onChange={set("exang")}>
                <option value="0">0 - No</option>
                <option value="1">1 - Yes</option>
              </select>
            </label>

            <label className="field">
              <span>ST Depression (oldpeak)</span>
              <input type="number" step="0.1" value={form.oldpeak} onChange={set("oldpeak")} />
            </label>

            <label className="field">
              <span>Slope of Peak Exercise ST Segment</span>
              <select value={form.slope} onChange={set("slope")}>
                <option value="1">1 - Upsloping</option>
                <option value="2">2 - Flat</option>
                <option value="3">3 - Downsloping</option>
              </select>
            </label>

            <label className="field">
              <span>Major Vessels (0-3)</span>
              <select value={form.ca} onChange={set("ca")}>
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
              </select>
            </label>

            <label className="field">
              <span>Thalassemia</span>
              <select value={form.thal} onChange={set("thal")}>
                <option value="3">3 - Normal</option>
                <option value="6">6 - Fixed Defect</option>
                <option value="7">7 - Reversible Defect</option>
              </select>
            </label>
          </div>

          <button className="btn mt" disabled={busy}>
            {busy ? "Predicting..." : "Predict Risk"}
          </button>
        </form>
      </div>

      <ResultPanel result={result} />
    </div>
  );
}
