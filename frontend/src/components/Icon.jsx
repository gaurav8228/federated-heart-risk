// Small SVG icons for the sidebar (no icon library needed).
export default function Icon({ name }) {
  const paths = {
    dashboard: "M3 3h7v9H3zM14 3h7v5h-7zM14 12h7v9h-7zM3 16h7v5H3z",
    predict: "M12 5v14M5 12h14",
    history: "M12 8v4l3 2M12 21a9 9 0 100-18 9 9 0 000 18z",
    chart: "M3 3v18h18M7 15l4-4 3 3 5-6",
    settings:
      "M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 15a1.7 1.7 0 00.3 1.9l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-2.9 1.2V21a2 2 0 11-4 0v-.1A1.7 1.7 0 007 19.4l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00-1.2-2.9H3a2 2 0 110-4h.1A1.7 1.7 0 004.6 7l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.9.3H9.4A1.7 1.7 0 0010.6 3V3a2 2 0 114 0v.1a1.7 1.7 0 002.9 1.2l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.9v.1a1.7 1.7 0 001.6 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z",
    admin: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z",
  };
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
         strokeLinecap="round" strokeLinejoin="round">
      <path d={paths[name] || paths.dashboard} />
    </svg>
  );
}

// The HeartRisk AI logo: a white heart inside a blue circle.
export function HeartLogo({ size = 40 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40">
      <circle cx="20" cy="20" r="20" fill="#2563eb" />
      <path
        d="M20 29s-9-5.8-9-12.2C11 13.5 13.2 12 15.5 12c1.8 0 3.4 1 4.5 2.6C21.1 13 22.7 12 24.5 12 26.8 12 29 13.5 29 16.8 29 23.2 20 29 20 29z"
        fill="#fff"
      />
    </svg>
  );
}
