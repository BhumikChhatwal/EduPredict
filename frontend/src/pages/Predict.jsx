import { useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  BandBadge,
  PageHeader,
  formatNumber,
  humanize,
} from "../components/ui";
import { DEFAULT_ATTRIBUTES, FIELD_GROUPS, GRADE_FIELDS } from "../data/formFields";
import { SUBJECT_LABELS, subjectLabel } from "../data/subjects";
import { api } from "../services/api";

const ACCENT = "#8fb6d9";
const MUTED = "#6e737a";

export default function Predict() {
  const [subject, setSubject] = useState("mat");
  const [featureSet, setFeatureSet] = useState("early");
  const [attributes, setAttributes] = useState(DEFAULT_ATTRIBUTES);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  function update(name, value) {
    setAttributes((current) => ({ ...current, [name]: value }));
  }

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError(null);

    const payload = { ...attributes };
    if (featureSet === "early") {
      delete payload.G1;
      delete payload.G2;
    }

    try {
      const response = await api.predict({
        subject,
        feature_set: featureSet,
        attributes: payload,
      });
      setResult(response);
    } catch (requestError) {
      setError(requestError);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <PageHeader
        eyebrow="Predict"
        title="Estimate a final grade"
        description="Fill in what is known about the student. The estimate is reported as CBSE marks out of 100 with the board's grade letter, beside the model's measured error and the attributes that moved it. The model was trained on Portuguese school records, so read it as an early-warning aid for a CBSE classroom, not a calibrated local prediction."
      />

      <div className="predict-layout">
        <form className="card predict-form" onSubmit={submit}>
          <div className="form-toolbar">
            <label className="control">
              <span className="control-label">Subject</span>
              <select
                value={subject}
                onChange={(event) => setSubject(event.target.value)}
              >
                {Object.entries(SUBJECT_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>

            <div className="control">
              <span className="control-label">Feature set</span>
              <div className="segmented" role="group" aria-label="Feature set">
                {[
                  ["early", "Early"],
                  ["with_grades", "With grades"],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    className={featureSet === value ? "is-active" : ""}
                    onClick={() => setFeatureSet(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <p className="form-note">
            {featureSet === "early"
              ? "Early: no period grades. This is the variant usable at intake."
              : "With grades: uses G1 and G2. More accurate, but largely restates two earlier assessments."}
          </p>

          {featureSet === "with_grades" ? (
            <fieldset className="field-group">
              <legend>Period grades</legend>
              <div className="field-grid">
                {GRADE_FIELDS.map((field) => (
                  <Field
                    key={field.name}
                    field={field}
                    value={attributes[field.name]}
                    onChange={update}
                  />
                ))}
              </div>
            </fieldset>
          ) : null}

          {FIELD_GROUPS.map((group) => (
            <fieldset className="field-group" key={group.title}>
              <legend>{group.title}</legend>
              <p className="group-note">{group.description}</p>
              <div className="field-grid">
                {group.fields.map((field) => (
                  <Field
                    key={field.name}
                    field={field}
                    value={attributes[field.name]}
                    onChange={update}
                  />
                ))}
              </div>
            </fieldset>
          ))}

          <div className="form-actions">
            <button className="button" type="submit" disabled={busy}>
              {busy ? "Estimating..." : "Estimate grade"}
            </button>
            <span className="form-disclaimer">
              An estimate from a small dataset, not a guaranteed outcome.
            </span>
          </div>

          {error ? (
            <p className="inline-error" role="alert">
              {error.message}
            </p>
          ) : null}
        </form>

        <div className="result-column">
          {result ? <Result result={result} /> : <EmptyResult />}
        </div>
      </div>
    </div>
  );
}

function Field({ field, value, onChange }) {
  const numericOptions =
    field.type === "select" && typeof field.options[0]?.value === "number";

  return (
    <label className="control">
      <span className="control-label">{field.label}</span>
      {field.type === "select" ? (
        <select
          value={value}
          onChange={(event) =>
            onChange(
              field.name,
              numericOptions ? Number(event.target.value) : event.target.value,
            )
          }
        >
          {field.options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          type="number"
          min={field.min}
          max={field.max}
          value={value}
          onChange={(event) => onChange(field.name, Number(event.target.value))}
        />
      )}
      {field.help ? <span className="control-help">{field.help}</span> : null}
    </label>
  );
}

function EmptyResult() {
  return (
    <div className="card result-empty">
      <p className="eyebrow">Result</p>
      <h2 className="result-empty-title">No estimate yet</h2>
      <p className="result-empty-body">
        Submit the form to see the estimated CBSE marks, the grade letter and band they
        fall in, the model's measured error, and which attributes raised or lowered this
        particular estimate.
      </p>
      <ul className="plain-list">
        <li>The estimate is a statistical prediction, not the marks a student will get.</li>
        <li>Contributions are correlational, not causal, and not advice.</li>
        <li>Attendance is never predicted; recorded absences is only an input.</li>
      </ul>
    </div>
  );
}

function Result({ result }) {
  const estimate = result.prediction;
  const context = result.context;
  const contributions = result.contributions.slice(0, 10);
  const chartData = [...contributions].reverse();
  const maxAbs = Math.max(
    1,
    ...contributions.map((row) => Math.abs(row.delta)),
  );

  return (
    <div className="result-stack">
      <div className="card result-card">
        <div className="result-head">
          <div>
            <p className="eyebrow">
              Estimate - {subjectLabel(result.subject)} / {result.feature_set.replace("_", " ")}
            </p>
            <p className="result-grade tabular">
              {formatNumber(estimate.marks, 1)}
              <span className="result-scale">/100</span>
            </p>
            <p className="result-interval">
              {formatNumber(estimate.interval[0], 1)} to{" "}
              {formatNumber(estimate.interval[1], 1)} marks at the model's typical error
            </p>
          </div>
          <div className="result-badge">
            <span className="tag">CBSE {estimate.grade_letter}</span>
            <BandBadge band={estimate.band} />
            <p className="result-model">
              {result.family} v{result.model_version}
            </p>
          </div>
        </div>

        <p className="result-basis">{estimate.interval_basis}.</p>

        {estimate.distance_to_band_edge !== null ? (
          <p className="result-edge">
            {formatNumber(estimate.distance_to_band_edge, 1)} marks from the nearest band
            boundary
            {estimate.distance_to_band_edge < context.test_rmse
              ? " - within the model's typical error, so the band is not a firm call."
              : "."}
          </p>
        ) : null}

        <div className="context-grid">
          <ContextItem label="Held-out RMSE" value={formatNumber(context.test_rmse, 2)} />
          <ContextItem label="Held-out MAE" value={formatNumber(context.test_mae, 2)} />
          <ContextItem
            label="Within 10 marks"
            value={`${(context.within_10_marks * 100).toFixed(0)}%`}
          />
          <ContextItem label="CV RMSE" value={formatNumber(context.cv_rmse, 2)} />
          <ContextItem label="Training rows" value={context.rows_train} />
          <ContextItem label="Test rows" value={context.rows_test} />
        </div>
      </div>

      <div className="card">
        <h2 className="frame-title">What moved this estimate</h2>
        <p className="frame-subtitle">
          Each bar replaces one attribute with its typical training value and re-runs the
          model. Bars show the difference that attribute made.
        </p>
        <div style={{ height: Math.max(220, chartData.length * 26) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
            >
              <XAxis
                type="number"
                domain={[-maxAbs, maxAbs]}
                tick={{ fontSize: 11, fill: MUTED }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <YAxis
                type="category"
                dataKey="feature"
                width={116}
                tick={{ fontSize: 11, fill: "#a6a6a1" }}
                tickLine={false}
                axisLine={false}
                tickFormatter={humanize}
              />
              <Tooltip
                formatter={(value) => [
                  `${value > 0 ? "+" : ""}${value.toFixed(2)} marks`,
                  "effect",
                ]}
                labelFormatter={humanize}
                contentStyle={tooltipStyle}
              />
              <ReferenceLine x={0} stroke="rgba(242, 242, 240, 0.3)" />
              <Bar dataKey="delta" isAnimationActive={false} barSize={13}>
                {chartData.map((row) => (
                  <Cell
                    key={row.feature}
                    fill={row.delta >= 0 ? ACCENT : MUTED}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
            <p className="frame-footnote">{result.contribution_basis}</p>
      </div>

      <div className="card limitations">
        <h2 className="frame-title">What this estimate does not claim</h2>
        <ul className="plain-list">
          {result.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function ContextItem({ label, value }) {
  return (
    <div className="context-item">
      <span className="context-label">{label}</span>
      <span className="context-value tabular">{value}</span>
    </div>
  );
}

const tooltipStyle = {
  fontSize: 12,
  background: "#15161a",
  border: "1px solid rgba(242, 242, 240, 0.18)",
  borderRadius: 2,
  color: "#f2f2f0",
};
