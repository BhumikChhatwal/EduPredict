import { useState } from "react";

import {
  DataTable,
  ErrorNote,
  Loading,
  PageHeader,
  formatNumber,
  humanize,
} from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

/**
 * The model page is a rendering of the artifact metadata, not a hand-written
 * description. Every number here comes from `ml/artifacts/models/*.json`, which the
 * serializer wrote from the same fitted pipeline that answers predictions. If the
 * artefacts are retrained, this page changes with them.
 */
export default function Model() {
  const info = useApi(() => api.modelInfo(), []);
  const [activeKey, setActiveKey] = useState(null);

  if (info.status === "loading") return <Loading label="Loading model metadata" />;
  if (info.status === "error") return <ErrorNote error={info.error} />;

  const data = info.data;
  const models = data.models;
  const active = models.find((model) => keyOf(model) === activeKey) ?? models[0];

  return (
    <div className="page">
      <PageHeader
        eyebrow="Model"
        title="What is actually deployed"
        description="Four fitted pipelines, two per subject. This page reads the saved metadata: the chosen family, its parameters, the measured error, and how the estimates behave by band."
      />

      <div className="grid grid-2">
        {models.map((model) => (
          <article className="card model-summary" key={keyOf(model)}>
            <header className="model-summary-head">
              <div>
                <p className="eyebrow">
                  {model.subject === "mat" ? "Mathematics" : "Portuguese"} /{" "}
                  {model.feature_set.replace("_", " ")}
                </p>
                <h2 className="frame-title">{model.family}</h2>
              </div>
              <span className="tag">{model.chosen === "tuned" ? "tuned" : "untuned"}</span>
            </header>

            <div className="model-figures">
              <Figure label="Test RMSE" value={formatNumber(model.metrics.test_rmse, 2)} />
              <Figure label="Test MAE" value={formatNumber(model.metrics.test_mae, 2)} />
              <Figure label="CV RMSE" value={formatNumber(model.metrics.cv_rmse, 2)} />
              <Figure
                label="Within 2 marks"
                value={`${(model.metrics.within_2_marks * 100).toFixed(0)}%`}
              />
            </div>

            <dl className="kv-list">
              <div className="kv">
                <dt>Features</dt>
                <dd>
                  {model.features.count} raw / {model.features.encoded_count} encoded
                </dd>
              </div>
              <div className="kv">
                <dt>Rows</dt>
                <dd>
                  {model.rows_train} train / {model.rows_test} test
                </dd>
              </div>
              <div className="kv">
                <dt>Parameters</dt>
                <dd>{formatParams(model.params)}</dd>
              </div>
              <div className="kv">
                <dt>R2</dt>
                <dd>{formatNumber(model.metrics.test_r2, 3)}</dd>
              </div>
            </dl>
          </article>
        ))}
      </div>

      <section className="card">
        <h2 className="frame-title">Chosen configuration</h2>
        <p className="frame-subtitle">
          The winner of the cross-validated comparison, refit on the full training split.
          The dummy mean regressor is the floor every model had to beat.
        </p>
        <DataTable
          rows={models}
          rowKey={keyOf}
          columns={[
            { key: "subject", label: "Subject", render: (row) => row.subject.toUpperCase() },
            {
              key: "feature_set",
              label: "Variant",
              render: (row) => row.feature_set.replace("_", " "),
            },
            { key: "family", label: "Model" },
            {
              key: "cv_rmse",
              label: "CV RMSE",
              align: "right",
              render: (row) => formatNumber(row.metrics.cv_rmse, 3),
            },
            {
              key: "baseline",
              label: "Floor",
              align: "right",
              render: (row) => formatNumber(row.metrics.baseline_cv_rmse, 3),
            },
            {
              key: "test_rmse",
              label: "Test RMSE",
              align: "right",
              render: (row) => formatNumber(row.metrics.test_rmse, 3),
            },
            {
              key: "test_r2",
              label: "Test R2",
              align: "right",
              render: (row) => formatNumber(row.metrics.test_r2, 3),
            },
            {
              key: "worst",
              label: "Worst error",
              align: "right",
              render: (row) => formatNumber(row.metrics.worst_error, 1),
            },
          ]}
        />
        <p className="frame-footnote">
          Held-out rows were scored once. The gap between CV and test RMSE is the usual
          optimism of resampling; the test figure is the one to trust.
        </p>
      </section>

      <div className="grid grid-2">
        <section className="card">
          <header className="frame-head">
            <div>
              <h2 className="frame-title">Error by predicted band</h2>
              <p className="frame-subtitle">
                On the held-out rows, grouped by the band the model assigned.
              </p>
            </div>
            <div className="frame-actions">
              <select
                value={keyOf(active)}
                onChange={(event) => setActiveKey(event.target.value)}
                aria-label="Model"
              >
                {models.map((model) => (
                  <option key={keyOf(model)} value={keyOf(model)}>
                    {model.subject.toUpperCase()} / {model.feature_set.replace("_", " ")}
                  </option>
                ))}
              </select>
            </div>
          </header>

          <DataTable
            rows={bandRows(active)}
            rowKey={(row) => row.band}
            columns={[
              { key: "band", label: "Band" },
              { key: "n", label: "Rows", align: "right" },
              {
                key: "mae",
                label: "MAE",
                align: "right",
                render: (row) => formatNumber(row.mae, 2),
              },
              {
                key: "mean_residual",
                label: "Mean bias",
                align: "right",
                render: (row) => formatSigned(row.mean_residual),
              },
            ]}
          />
          <p className="frame-footnote">
            Bias is predicted minus actual. The early models pull toward the middle: a
            student who fails is estimated too high, a top student too low. Read a band
            as a rough signal, and check its row count before trusting its error.
          </p>
        </section>

        <section className="card">
          <h2 className="frame-title">Top permutation importance</h2>
          <p className="frame-subtitle">
            {active.subject.toUpperCase()} / {active.feature_set.replace("_", " ")} -
            increase in RMSE when one input is shuffled.
          </p>
          <div className="importance-list">
            {active.feature_importance.map((row) => {
              const max = Math.max(
                ...active.feature_importance.map((item) => Math.abs(item.importance)),
                0.0001,
              );
              return (
                <div className="importance-row" key={row.feature}>
                  <span className="importance-name">{humanize(row.feature)}</span>
                  <div className="importance-track">
                    <span
                      className="importance-fill"
                      style={{ width: `${(Math.abs(row.importance) / max) * 100}%` }}
                    />
                  </div>
                  <span className="importance-value tabular">
                    {formatSigned(row.importance)}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="frame-footnote">
            Dependence, not causation. Correlated inputs split their credit, which is why
            absences and failures can both look modest while the model still leans on them.
          </p>
        </section>
      </div>

      <div className="grid grid-2">
        <section className="card">
          <h2 className="frame-title">Risk band policy</h2>
          <p className="frame-subtitle">{data.risk_band_policy}</p>
          <ul className="plain-list">
            {data.risk_bands.map((band) => (
              <li key={band.label}>
                <strong>{band.label.replace("_", " ")}</strong> - {bandRange(band)}
              </li>
            ))}
          </ul>
          <p className="frame-footnote">
            A band is a coarse triage signal. On the early models the estimate shrinks
            toward the middle, so the two extreme bands are the least reliable.
          </p>
        </section>

        <section className="card">
          <h2 className="frame-title">Feature sets</h2>
          <dl className="kv-list">
            {Object.entries(data.feature_sets).map(([name, description]) => (
              <div className="kv" key={name}>
                <dt>{name.replace("_", " ")}</dt>
                <dd>{description}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>

      <section className="card limitations">
        <h2 className="frame-title">Limitations carried by every estimate</h2>
        <ul className="plain-list">
          {data.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
        <p className="frame-footnote">
          Artifact {active.model_version}, metadata schema {active.metadata_schema},
          target {active.target} on a {active.target_range[0]}-{active.target_range[1]}{" "}
          scale, {active.rows_train + active.rows_test} rows in this subject, seed{" "}
          {active.split.seed}, last written{" "}
          {formatDate(active.trained_at)}. Runtime: scikit-learn{" "}
          {active.runtime.scikit_learn}, Python {active.runtime.python}.
        </p>
      </section>
    </div>
  );
}

function keyOf(model) {
  return `${model.subject}__${model.feature_set}`;
}

function bandRows(model) {
  return Object.entries(model.bands).map(([band, stats]) => ({
    band: band.replace("_", " "),
    ...stats,
  }));
}

function bandRange(band) {
  if (band.min === 0) return `under ${band.max}`;
  if (band.max_inclusive) return `${band.min} and above`;
  return `${band.min} to under ${band.max}`;
}

function formatParams(params) {
  const entries = Object.entries(params ?? {});
  if (entries.length === 0) return "library defaults";
  return entries.map(([name, value]) => `${name}=${value}`).join(", ");
}

function formatSigned(value) {
  if (value === null || value === undefined) return "-";
  return `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
}

function formatDate(value) {
  return new Date(value).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function Figure({ label, value }) {
  return (
    <div className="model-figure">
      <span className="model-figure-label">{label}</span>
      <span className="model-figure-value tabular">{value}</span>
    </div>
  );
}
