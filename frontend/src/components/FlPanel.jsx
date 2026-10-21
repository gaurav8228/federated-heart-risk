// FlPanel: the "FL Training Status" dashboard.
// Shown on the FL Training page (read-only) and on the Admin page (with the
// disabled FedAvg/FedProx selector + Start/Stop buttons). Data is REAL, from
// GET /api/fl/rounds (see src/mock/flData.js -> fetchFlRuns).
import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function FlPanel({ status, history, hospitals, controls, runs, selectedRunId, onRunChange }) {
  const epsilon =
    status.epsilon === null || status.epsilon === undefined ? "Not private" : status.epsilon;

  return (
    <div>
      {/* header with the status chip */}
      <div className="row-between" style={{ marginBottom: 14 }}>
        <h1 style={{ margin: 0 }}>FL Training Status</h1>
        <span className="chip">
          <span className="chip-dot" /> {status.status}
        </span>
      </div>

      {/* run selector - pick which federated run to view */}
      {runs && runs.length > 0 && (
        <label className="field" style={{ maxWidth: 360, marginBottom: 14 }}>
          <span>Run</span>
          <select value={selectedRunId} onChange={(e) => onRunChange(e.target.value)}>
            {runs.map((r) => (
              <option key={r.id} value={String(r.id)}>{r.label}</option>
            ))}
          </select>
        </label>
      )}

      {/* the 4 stat boxes from the design diagram */}
      <div className="stat-grid">
        <div className="stat">
          <div className="label">Round</div>
          <div className="value">
            {status.currentRound} <small>/ {status.totalRounds}</small>
          </div>
        </div>
        <div className="stat">
          <div className="label">Global Accuracy</div>
          <div className="value">{status.globalAccuracy}%</div>
        </div>
        <div className="stat">
          <div className="label">ε (Privacy Budget)</div>
          <div className="value">{epsilon}</div>
        </div>
        <div className="stat">
          <div className="label">Communication Cost</div>
          <div className="value">
            {status.communicationCostMb} <small>MB</small>
          </div>
        </div>
      </div>

      {/* accuracy over rounds line chart */}
      <div className="card">
        <h2>Accuracy over Rounds</h2>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={history} margin={{ left: -20, right: 16, top: 8 }}>
            <XAxis dataKey="round" label={{ value: "Round", position: "insideBottom", offset: -4 }} />
            <YAxis domain={[0.6, 1.0]} />
            <ReferenceLine y={0.87} stroke="#cbd5e1" strokeDasharray="4 4" />
            <Tooltip />
            <Line type="monotone" dataKey="accuracy" stroke="#2563eb" strokeWidth={2} dot />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* the 3 hospital clients */}
      <div className="card">
        <h2>Hospital Clients</h2>
        <table>
          <thead>
            <tr>
              <th>Hospital</th><th>Client #</th><th>Patients</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            {hospitals.map((h) => (
              <tr key={h.name}>
                <td>{h.name}</td>
                <td>{h.clients}</td>
                <td>{h.patients}</td>
                <td><span className="chip">{h.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* optional controls (only passed in by the Admin page) */}
      {controls}
    </div>
  );
}
