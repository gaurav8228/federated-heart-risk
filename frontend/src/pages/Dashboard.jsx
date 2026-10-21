// Dashboard: quick counts + the last predictions (from GET /api/predictions).
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { apiFetch } from "../api";
import Badge from "../components/Badge";

export default function Dashboard() {
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    apiFetch("/api/predictions")
      .then(setPredictions)
      .catch((err) => setError(err.message));
  }, []);

  // simple counts, computed from the list we already have
  const total = predictions.length;
  const high = predictions.filter((p) => p.risk_label === "High").length;
  const moderate = predictions.filter((p) => p.risk_label === "Moderate").length;
  const low = predictions.filter((p) => p.risk_label === "Low").length;
  const recent = predictions.slice(0, 5);

  return (
    <div>
      <h1>Dashboard</h1>
      {error && <div className="error-banner">{error}</div>}

      <div className="stat-grid">
        <div className="stat"><div className="label">Total Predictions</div><div className="value">{total}</div></div>
        <div className="stat"><div className="label">High Risk</div><div className="value" style={{ color: "#dc2626" }}>{high}</div></div>
        <div className="stat"><div className="label">Moderate Risk</div><div className="value" style={{ color: "#d97706" }}>{moderate}</div></div>
        <div className="stat"><div className="label">Low Risk</div><div className="value" style={{ color: "#16a34a" }}>{low}</div></div>
      </div>

      <div className="card">
        <div className="row-between">
          <h2 style={{ margin: 0 }}>Recent Predictions</h2>
          <Link to="/predict"><button className="btn btn-sm">+ New Prediction</button></Link>
        </div>
        <table>
          <thead>
            <tr><th>Date & Time</th><th>Risk Score</th><th>Result</th></tr>
          </thead>
          <tbody>
            {recent.length === 0 && (
              <tr><td colSpan="3" className="muted">No predictions yet.</td></tr>
            )}
            {recent.map((p) => (
              <tr key={p.id}>
                <td>{new Date(p.created_at).toLocaleString()}</td>
                <td>{p.risk_score}%</td>
                <td><Badge label={p.risk_label} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
