// ResultPanel: how a prediction is displayed (used by the Predict page
// and by the History "View" page).
//   * big risk score + coloured Low/Moderate/High badge
//   * Key Factors list with bars (red = pushes risk up, blue = pushes down)
//   * a SHAP bar chart made with recharts
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import Badge from "./Badge";

export default function ResultPanel({ result }) {
  if (!result) return null;

  // Largest |value| sets the width of the bars, so they always look balanced.
  const maxValue = Math.max(...result.key_factors.map((f) => Math.abs(f.value)), 0.01);

  // Data for the recharts chart (one bar per feature)
  const chartData = result.key_factors.map((f) => ({
    name: f.feature,
    value: f.value,
    fill: f.value >= 0 ? "#ef4444" : "#3b82f6", // red up / blue down
  }));

  return (
    <div className="card">
      <h2>Prediction Result</h2>

      {/* big risk score */}
      <div className="risk-box">
        <div>
          <div className="score-label">Risk Score</div>
          <div className="score">{result.risk_score}%</div>
        </div>
        <Badge label={result.risk_label} />
      </div>

      {/* key factors with bars */}
      <h3>Key Factors (SHAP)</h3>
      {result.key_factors.map((f) => (
        <div className="factor" key={f.feature}>
          <div className="top">
            <span>{f.feature}</span>
            <span>{f.value > 0 ? "+" : ""}{f.value.toFixed(2)}</span>
          </div>
          <div className="track">
            <div
              className={"fill " + (f.value >= 0 ? "pos" : "neg")}
              style={{ width: `${(Math.abs(f.value) / maxValue) * 100}%` }}
            />
          </div>
        </div>
      ))}

      {/* SHAP chart (recharts) */}
      <h3 className="mt">SHAP Explanation</h3>
      <ResponsiveContainer width="100%" height={230}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 30, right: 20 }}>
          <XAxis type="number" domain={["dataMin", "dataMax"]} />
          <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
          <ReferenceLine x={0} />
          <Tooltip />
          <Bar dataKey="value" radius={[0, 6, 6, 0]}>
            {chartData.map((d) => (
              <Cell key={d.name} fill={d.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
