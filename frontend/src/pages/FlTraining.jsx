// FL Training page: read-only view of real federated-training runs (everyone,
// including doctors). While a run is live it polls /api/fl/status every 3s and
// refreshes the accuracy chart. Controls only appear on the Admin page.
import { useEffect, useState } from "react";
import FlPanel from "../components/FlPanel";
import { fetchFlRuns, fetchFlStatus, hospitalClients } from "../mock/flData";

export default function FlTraining() {
  const [runs, setRuns] = useState(null);   // null = still loading
  const [error, setError] = useState("");
  const [selected, setSelected] = useState("");
  const [status, setStatus] = useState(null);

  function loadRuns() {
    fetchFlRuns()
      .then((r) => {
        setRuns(r);
        if (r.length) setSelected((prev) => prev || String(r[0].id));
      })
      .catch((e) => {
        setRuns([]);
        setError(e.message);
      });
  }

  useEffect(() => {
    loadRuns();
    fetchFlStatus().then(setStatus).catch(() => {});
  }, []);

  const running = !!status && status.state === "running";
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => {
      fetchFlStatus().then(setStatus).catch(() => {});
      loadRuns();
    }, 3000);
    return () => clearInterval(t);
  }, [running]);

  if (runs === null) return <div className="info-banner">Loading federated runs…</div>;
  if (error) return <div className="error-banner">{error}</div>;
  if (!runs.length)
    return (
      <div className="info-banner">
        No federated runs yet. An admin can start one from the Admin page.
      </div>
    );

  const run = runs.find((r) => String(r.id) === selected) || runs[0];

  return (
    <FlPanel
      status={run}
      history={run.history}
      hospitals={hospitalClients}
      runs={runs}
      selectedRunId={String(run.id)}
      onRunChange={setSelected}
    />
  );
}
