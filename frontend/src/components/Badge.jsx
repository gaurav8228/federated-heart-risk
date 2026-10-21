// Badge: the coloured Low / Moderate / High pill.
export default function Badge({ label }) {
  const cls =
    label === "Low" ? "badge-low" : label === "Moderate" ? "badge-moderate" : "badge-high";
  return <span className={`badge ${cls}`}>{label}</span>;
}
