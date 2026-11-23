import { NavLink, Route, Routes } from "react-router-dom";

import { useApi } from "./hooks/useApi";
import { api } from "./services/api";
import Analytics from "./pages/Analytics";
import Dashboard from "./pages/Dashboard";
import Model from "./pages/Model";
import Predict from "./pages/Predict";
import Students from "./pages/Students";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/predict", label: "Predict" },
  { to: "/students", label: "Records" },
  { to: "/analytics", label: "Analytics" },
  { to: "/model", label: "Model" },
];

export default function App() {
  const health = useApi(() => api.health(), []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <p className="brand-name">EduPredict</p>
            <p className="brand-sub">Grade estimates with error bars</p>
          </div>
        </div>

        <nav className="nav" aria-label="Sections">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-link ${isActive ? "is-active" : ""}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <footer className="sidebar-foot">
          <p className="foot-line">
            <span className={`dot ${health.data?.model_loaded ? "dot-ok" : "dot-warn"}`} />
            {health.status === "ready"
              ? health.data.model_loaded
                ? `${health.data.models} models loaded`
                : "models unavailable"
              : health.status === "error"
                ? "API unreachable"
                : "checking API"}
          </p>
          <p className="foot-note">
            Built for CBSE secondary schools. Trained on 1,044 public Portuguese records;
            marks are mapped to the 0-100 scale, and estimates are statistical, not
            decisions.
          </p>
        </footer>
      </aside>

      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/students" element={<Students />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/model" element={<Model />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}

function NotFound() {
  return (
    <div className="page">
      <p className="eyebrow">404</p>
      <h1 className="page-title">No such page</h1>
      <p className="page-desc">
        Use the navigation on the left: overview, prediction, records, analytics, model.
      </p>
    </div>
  );
}
