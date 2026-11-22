import { useEffect, useState } from "react";

/**
 * Phase 1 shell. It reports the live result of the backend health check and
 * nothing else: the dashboard, prediction form and analytics views arrive in
 * milestones 12 to 14.
 */
export default function App() {
  const [health, setHealth] = useState({ state: "loading" });

  useEffect(() => {
    let cancelled = false;

    fetch("/api/health")
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (!cancelled) setHealth({ state: "ready", data });
      })
      .catch((error) => {
        if (!cancelled) setHealth({ state: "error", message: error.message });
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main style={styles.page}>
      <div style={styles.panel}>
        <p style={styles.eyebrow}>EduPredict</p>
        <h1 style={styles.heading}>Student performance analytics</h1>
        <p style={styles.body}>
          Project skeleton only. The dashboard, prediction form and analytics views
          are built in later milestones, once the model produces real numbers.
        </p>
        <div style={styles.status}>
          <span style={styles.label}>API status</span>
          <span className="tabular" style={styles.value}>
            {describe(health)}
          </span>
        </div>
      </div>
    </main>
  );
}

function describe(health) {
  if (health.state === "loading") return "checking /api/health";
  if (health.state === "error") return `unreachable — ${health.message}`;
  return `${health.data.status} · v${health.data.version} · model not loaded`;
}

const styles = {
  page: {
    minHeight: "100%",
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "center",
    padding: "var(--space-7) var(--space-4)",
  },
  panel: {
    width: "100%",
    maxWidth: "560px",
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    boxShadow: "var(--shadow)",
    padding: "var(--space-6)",
  },
  eyebrow: {
    fontSize: "var(--text-xs)",
    letterSpacing: "0.04em",
    textTransform: "uppercase",
    color: "var(--text-muted)",
    marginBottom: "var(--space-2)",
  },
  heading: {
    fontSize: "var(--text-xl)",
    fontWeight: 600,
    letterSpacing: "-0.01em",
    marginBottom: "var(--space-3)",
  },
  body: {
    fontSize: "var(--text-base)",
    color: "var(--text-muted)",
    marginBottom: "var(--space-5)",
  },
  status: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "var(--space-4)",
    borderTop: "1px solid var(--border)",
    paddingTop: "var(--space-3)",
  },
  label: {
    fontSize: "var(--text-sm)",
    color: "var(--text-muted)",
  },
  value: {
    fontSize: "var(--text-sm)",
    color: "var(--text)",
  },
};
