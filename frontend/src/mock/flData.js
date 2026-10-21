// ---------------------------------------------------------------------------
// flData.js - REAL data for the FL Training + Admin pages (M8).
//
//   fetchFlRuns()    -> GET /api/fl/rounds, grouped into runs (newest first)
//   fetchFairness()  -> GET /api/eval/fairness, grouped by group_type
//   fetchFlStatus()  -> GET /api/fl/status  (running/idle + latest round + log)
//   startFlRun()     -> POST /api/fl/start  (admin only)
//   stopFlRun()      -> POST /api/fl/stop   (admin only)
//   hospitalClients  -> static roster of the 3 federated hospitals
// ---------------------------------------------------------------------------

import { apiFetch } from "../api";

export const hospitalClients = [
  { name: "Hospital A", clients: 1, patients: 250, status: "Idle" },
  { name: "Hospital B", clients: 2, patients: 238, status: "Idle" },
  { name: "Hospital C", clients: 3, patients: 248, status: "Idle" },
];

// The endpoint returns one row per round, newest first. A new run begins
// whenever round_number drops back to 0, so walk oldest -> newest and cut.
function groupRuns(rows) {
  const asc = [...rows].sort((a, b) => a.id - b.id);
  const runs = [];
  let current = [];
  let prev = -1;
  for (const row of asc) {
    if (row.round_number <= prev) {
      runs.push(current);
      current = [];
    }
    current.push(row);
    prev = row.round_number;
  }
  if (current.length) runs.push(current);

  const shaped = runs.map((r) => {
    const last = r[r.length - 1];
    const isDp = r.some((x) => x.epsilon !== null && x.epsilon !== undefined);
    const name = last.strategy === "fedprox" ? "FedProx" : "FedAvg";
    const comm = r.reduce((s, x) => s + (x.communication_cost_mb || 0), 0);
    return {
      id: last.id,
      label: `${name} ${isDp ? "(DP)" : "(not private)"} - ${last.round_number} rounds`,
      strategy: last.strategy,
      isDp,
      status: last.status,
      currentRound: last.round_number,
      totalRounds: last.round_number,
      globalAccuracy: +(last.global_accuracy * 100).toFixed(1),
      epsilon: last.epsilon,                          // null -> "Not private"
      communicationCostMb: +comm.toFixed(4),
      history: r.map((x) => ({ round: x.round_number, accuracy: x.global_accuracy })),
    };
  });
  return shaped.reverse();                            // newest run first
}

export async function fetchFlRuns() {
  const rows = await apiFetch("/api/fl/rounds");
  return groupRuns(rows);
}

export async function fetchFlStatus() {
  return apiFetch("/api/fl/status");
}

export async function startFlRun(params) {
  return apiFetch("/api/fl/start", { method: "POST", body: JSON.stringify(params) });
}

export async function stopFlRun() {
  return apiFetch("/api/fl/stop", { method: "POST" });
}

// canonical display order for each group type
const GROUP_ORDER = {
  hospital: ["A", "B", "C"],
  sex: ["female", "male"],
  age: ["under 45", "45-60", "over 60"],
};

export async function fetchFairness() {
  const rows = await apiFetch("/api/eval/fairness");
  const byType = {};
  for (const r of rows) (byType[r.group_type] ||= []).push(r);
  for (const type of Object.keys(byType)) {
    const order = GROUP_ORDER[type] || [];
    byType[type].sort((a, b) => {
      const ia = order.indexOf(a.group_value);
      const ib = order.indexOf(b.group_value);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
  }
  return byType;
}
