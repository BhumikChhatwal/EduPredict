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

  const status =
    health.status === "ready"
      ? health.data.model_loaded
        ? `${health.data.models} models loaded`
        : "models unavailable"
      : health.status === "error"
        ? "API unreachable"
        : "checking API";

  return (
    <div className="app-shell">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">EduPredict</span>
        </NavLink>

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

        <p className="topbar-status">
          <span className={`dot ${health.data?.model_loaded ? "dot-ok" : "dot-warn"}`} />
          {status}
        </p>
      </header>

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

      <footer className="site-foot">
        <p>
          Built for CBSE secondary schools. Trained on 1,044 public Portuguese records;
          marks are mapped to the 0-100 scale, and every estimate is statistical, not a
          decision.
        </p>
        <p className="site-foot-meta">Estimates with measured error</p>
      </footer>
    </div>
  );
}

function NotFound() {
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <p className="eyebrow">404</p>
          <h1 className="page-title">No such page</h1>
        </div>
        <div className="page-head-aside">
          <p className="page-desc">
            Use the navigation above: overview, prediction, records, analytics, model.
          </p>
        </div>
      </header>
    </div>
  );
}
