// Admin page: FL dashboard with START/STOP controls (admins only) plus a
// Fairness section. While a run is live it polls /api/fl/status every 3s and
// refreshes the accuracy chart.
import { useEffect, useState } from "react";
import { getRole } from "../api";
import FlPanel from "../components/FlPanel";
import {
  fetchFairness, fetchFlRuns, fetchFlStatus, hospitalClients, startFlRun, stopFlRun,
} from "../mock/flData";

const GROUP_ORDER = ["hospital", "sex", "age"];
const GROUP_TITLES = { hospital: "By Hospital", sex: "By Sex", age: "By Age Group" };
const fmt = (v) => (v === null || v === undefined ? "–" : Number(v).toFixed(4));

function FairnessSection({ fairness }) {
  if (fairness === null)
    return (
      <div className="card">
        <h2>Fairness</h2>
        <div className="info-banner">Loading fairness metrics…</div>
      </div>
    );
  const types = GROUP_ORDER.filter((t) => fairness[t] && fairness[t].length);
  if (!types.length)
    return (
      <div className="card">
        <h2>Fairness</h2>
        <div className="info-banner">
          No fairness metrics yet. Generate them from the terminal:
          <br />
          <code>venv\Scripts\python.exe -m ml.fairness</code>
        </div>
      </div>
    );

  return (
    <div className="card">
      <h2>Fairness</h2>
      {types.map((type) => (
        <div key={type} style={{ marginBottom: 20 }}>
          <h3>{GROUP_TITLES[type]}</h3>
          <table>
            <thead>
              <tr>
                <th>Group</th><th>n</th><th>Recall</th><th>F1</th><th>AUC</th>
              </tr>
            </thead>
            <tbody>
              {fairness[type].map((g) => {
                const small = g.n !== null && g.n !== undefined && g.n < 30;
                return (
                  <tr key={g.group_value}>
                    <td>
                      {g.group_value}{" "}
                      {small && <span className="badge badge-moderate">small n</span>}
                    </td>
                    <td>{g.n ?? "–"}</td>
                    <td>{fmt(g.recall)}</td>
                    <td>{fmt(g.f1)}</td>
                    <td>{g.auc === null || g.auc === undefined ? "null" : fmt(g.auc)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

export default function Admin() {
  const [runs, setRuns] = useState(null);
  const [fairness, setFairness] = useState(null);
  const [selected, setSelected] = useState("");
  const [status, setStatus] = useState(null);   // { state, latest_round, log }
  const [error, setError] = useState("");

  // FL control form
  const [strategy, setStrategy] = useState("fedavg");
  const [rounds, setRounds] = useState(20);
  const [mu, setMu] = useState(0.1);
  const [dp, setDp] = useState(false);
  const [clip, setClip] = useState(1.0);
  const [noise, setNoise] = useState(1.0);
  const [secagg, setSecagg] = useState(false);

  const running = !!status && status.state === "running";

  function loadRuns() {
    fetchFlRuns()
      .then((r) => {
        setRuns(r);
        if (r.length) setSelected((prev) => prev || String(r[0].id));
      })
      .catch(() => setRuns([]));
  }
  function loadStatus() {
    fetchFlStatus().then(setStatus).catch(() => {});
  }

  useEffect(() => {
    loadRuns();
    loadStatus();
    fetchFairness().then(setFairness).catch(() => setFairness({}));
  }, []);

  // poll every 3s while a run is live, to refresh status + accuracy chart
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => {
      loadStatus();
      loadRuns();
    }, 3000);
    return () => clearInterval(t);
  }, [running]);

  // Only admins may open this page (the sidebar also hides the link)
  if (getRole() !== "admin")
    return <div className="error-banner">You need an admin account to open this page.</div>;

  async function start() {
    setError("");
    try {
      await startFlRun({
        strategy,
        rounds: Number(rounds),
        mu: Number(mu),
        dp,
        clip: Number(clip),
        noise: Number(noise),
        secagg,
      });
      loadStatus();
      loadRuns();
    } catch (e) {
      setError(e.message);   // e.g. "A run is already in progress"
    }
  }
  async function stop() {
    setError("");
    try {
      await stopFlRun();
      loadStatus();
      loadRuns();
    } catch (e) {
      setError(e.message);
    }
  }

  const run = runs && runs.length ? runs.find((r) => String(r.id) === selected) || runs[0] : null;

  const controls = (
    <div className="card">
      <h2>FL Controls</h2>
      {error && <div className="error-banner">{error}</div>}
      <div className="info-banner">
        {running ? "A federated run is in progress…" : "Idle — ready to start."}
      </div>
      <div className="row">
        <label className="field" style={{ maxWidth: 220 }}>
          <span>Strategy</span>
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)} disabled={running}>
            <option value="fedavg">FedAvg</option>
            <option value="fedprox">FedProx</option>
          </select>
        </label>
        <label className="field" style={{ maxWidth: 130 }}>
          <span>Rounds</span>
          <input type="number" min="1" max="100" value={rounds}
                 onChange={(e) => setRounds(e.target.value)} disabled={running} />
        </label>
        {strategy === "fedprox" && (
          <label className="field" style={{ maxWidth: 130 }}>
            <span>μ (mu)</span>
            <input type="number" step="0.1" min="0" max="1" value={mu}
                   onChange={(e) => setMu(e.target.value)} disabled={running} />
          </label>
        )}
      </div>
      <div className="row">
        <label className="field" style={{ maxWidth: 240 }}>
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input type="checkbox" checked={dp} onChange={(e) => setDp(e.target.checked)}
                   disabled={running} />
            Differential privacy
          </span>
        </label>
        {dp && (
          <>
            <label className="field" style={{ maxWidth: 130 }}>
              <span>Clip</span>
              <input type="number" step="0.1" min="0.1" value={clip}
                     onChange={(e) => setClip(e.target.value)} disabled={running} />
            </label>
            <label className="field" style={{ maxWidth: 130 }}>
              <span>Noise</span>
              <input type="number" step="0.1" min="0.1" value={noise}
                     onChange={(e) => setNoise(e.target.value)} disabled={running} />
            </label>
          </>
        )}
        <label className="field" style={{ maxWidth: 200 }}>
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input type="checkbox" checked={secagg} onChange={(e) => setSecagg(e.target.checked)}
                   disabled={running} />
            SecAgg (secure aggregation)
          </span>
        </label>
        <button className="btn mt" onClick={start} disabled={running}>Start Training</button>
        <button className="btn btn-danger mt" onClick={stop} disabled={!running}>Stop</button>
      </div>
      {status && status.log && status.log.length > 0 && (
        <pre style={{ marginTop: 14, maxHeight: 200, overflow: "auto", fontSize: 12,
                      background: "#0f172a", color: "#e2e8f0", padding: 10, borderRadius: 8 }}>
          {status.log.join("\n")}
        </pre>
      )}
    </div>
  );

  const flBlock =
    runs === null ? (
      <div className="info-banner">Loading federated runs…</div>
    ) : (
      <FlPanel
        status={run || { status: running ? "Running" : "Idle", currentRound: 0, totalRounds: 0,
                         globalAccuracy: "–", epsilon: null, communicationCostMb: "–" }}
        history={run ? run.history : []}
        hospitals={hospitalClients}
        controls={controls}
        runs={runs}
        selectedRunId={run ? String(run.id) : ""}
        onRunChange={setSelected}
      />
    );

  return (
    <div>
      {flBlock}
      <FairnessSection fairness={fairness} />
    </div>
  );
}
