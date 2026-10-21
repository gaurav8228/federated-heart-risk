// History: the list of past predictions (GET /api/predictions).
// "View" opens the full result page for that prediction.
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiFetch } from "../api";
import Badge from "../components/Badge";

export default function History() {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    apiFetch("/api/predictions")
      .then(setRows)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <div>
      <h1>Prediction History</h1>
      <p>View past predictions and reports.</p>

      {error && <div className="error-banner">{error}</div>}

      <div className="card">
        <table>
          <thead>
            <tr>
              <th>Date & Time</th><th>Risk Score</th><th>Result</th><th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && !error && (
              <tr><td colSpan="4" className="muted">No predictions yet.</td></tr>
            )}
            {rows.map((p) => (
              <tr key={p.id}>
                <td>{new Date(p.created_at).toLocaleString()}</td>
                <td>{p.risk_score}%</td>
                <td><Badge label={p.risk_label} /></td>
                <td>
                  <button className="btn btn-sm" onClick={() => navigate(`/history/${p.id}`)}>
                    View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
