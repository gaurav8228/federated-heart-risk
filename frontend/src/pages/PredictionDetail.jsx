// PredictionDetail: the full saved result (GET /api/predictions/{id}).
// Opened from the History page's "View" button.
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { apiFetch } from "../api";
import ResultPanel from "../components/ResultPanel";

export default function PredictionDetail() {
  const { id } = useParams();                  // the number in the URL
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    apiFetch(`/api/predictions/${id}`)
      .then(setData)
      .catch((err) => setError(err.message));
  }, [id]);

  if (error) return <div className="error-banner">{error}</div>;
  if (!data) return <p>Loading...</p>;

  return (
    <div>
      <div className="row-between">
        <h1>Prediction #{data.id}</h1>
        <Link to="/history">← Back to History</Link>
      </div>
      <p className="muted">
        {new Date(data.created_at).toLocaleString()} · model v{data.model_version}
      </p>

      <ResultPanel result={{ risk_score: data.risk_score, risk_label: data.risk_label, key_factors: data.key_factors }} />

      {data.patient && (
        <div className="card">
          <h2>Patient Data</h2>
          <table>
            <tbody>
              <tr><td>Age</td><td>{data.patient.age}</td><td>Sex</td><td>{data.patient.sex === 1 ? "Male" : "Female"}</td></tr>
              <tr><td>Chest Pain Type</td><td>{data.patient.cp}</td><td>Resting BP</td><td>{data.patient.trestbps}</td></tr>
              <tr><td>Cholesterol</td><td>{data.patient.chol}</td><td>Max Heart Rate</td><td>{data.patient.thalach}</td></tr>
              <tr><td>Fasting Blood Sugar</td><td>{data.patient.fbs}</td><td>Resting ECG</td><td>{data.patient.restecg}</td></tr>
              <tr><td>Exercise Angina</td><td>{data.patient.exang}</td><td>ST Depression</td><td>{data.patient.oldpeak}</td></tr>
              <tr><td>Slope</td><td>{data.patient.slope}</td><td>Major Vessels</td><td>{data.patient.ca}</td></tr>
              <tr><td>Thalassemia</td><td>{data.patient.thal}</td><td></td><td></td></tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
