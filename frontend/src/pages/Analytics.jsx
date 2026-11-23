import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import CorrelationHeatmap from "../components/CorrelationHeatmap";
import {
  ChartFrame,
  DataTable,
  ErrorNote,
  Loading,
  PageHeader,
  formatNumber,
  humanize,
} from "../components/ui";
import { SUBJECT_LABELS, subjectLabel } from "../data/subjects";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const ACCENT = "#8fb6d9";
const MUTED = "#6e737a";

export default function Analytics() {
  const [subject, setSubject] = useState("mat");
  const [featureSet, setFeatureSet] = useState("early");
  const [attribute, setAttribute] = useState("studytime");

  const options = useApi(() => api.options(), []);
  const correlation = useApi(() => api.correlation(subject), [subject]);
  const importance = useApi(
    () => api.featureImportance(subject, featureSet, 12),
    [subject, featureSet],
  );
  const comparison = useApi(() => api.modelComparison(), []);
  const subjects = useApi(() => api.subjectComparison(), []);
  const breakdown = useApi(
    () => Promise.all([api.attributes("mat", attribute), api.attributes("por", attribute)]),
    [attribute],
  );

  // Hooks must run in the same order on every render, so the derivation below stays
  // above the loading/error guards rather than after them.
  const breakdownRows = useMemo(() => {
    if (!breakdown.data) return [];
    const [maths, portuguese] = breakdown.data;
    const values = new Set([
      ...maths.groups.map((group) => group.value),
      ...portuguese.groups.map((group) => group.value),
    ]);

    return [...values]
      .sort((a, b) => (typeof a === "number" ? a - b : String(a).localeCompare(String(b))))
      .map((value) => {
        const matGroup = maths.groups.find((group) => group.value === value);
        const porGroup = portuguese.groups.find((group) => group.value === value);
        return {
          value: String(value),
          mat: matGroup?.mean ?? null,
          por: porGroup?.mean ?? null,
          matN: matGroup?.n ?? 0,
          porN: porGroup?.n ?? 0,
        };
      });
  }, [breakdown.data]);

  if (options.status === "loading") return <Loading label="Loading analytics" />;
  if (options.status === "error") return <ErrorNote error={options.error} />;

  return (
    <div className="page">
      <PageHeader
        eyebrow="Analytics"
        title="What the data and the models actually show"
        description="Descriptive statistics from the validated dataset, plus the cross-validation and held-out records produced by the training pipeline. Grades are shown as CBSE marks out of 100; errors are in the same units."
      />

      <div className="grid grid-2">
        <ChartFrame
          title="Correlation matrix"
          subtitle="Numeric attributes, including all three grade periods"
          actions={
            <SubjectToggle subject={subject} onChange={setSubject} />
          }
        >
          {correlation.status === "loading" ? <Loading /> : null}
          {correlation.status === "error" ? <ErrorNote error={correlation.error} /> : null}
          {correlation.data ? (
            <CorrelationHeatmap
              labels={correlation.data.labels}
              matrix={correlation.data.matrix}
            />
          ) : null}
        </ChartFrame>

        <ChartFrame
          title="Permutation importance"
          subtitle={
            importance.data
              ? `${subjectLabel(subject)} / ${featureSet.replace("_", " ")} - held-out rows n=${importance.data.n}`
              : "Held-out rows"
          }
          actions={
            <div className="inline-controls">
              <SubjectToggle subject={subject} onChange={setSubject} />
              <select
                value={featureSet}
                onChange={(event) => setFeatureSet(event.target.value)}
                aria-label="Feature set"
              >
                <option value="early">Early</option>
                <option value="with_grades">With grades</option>
              </select>
            </div>
          }
          footnote="Increase in RMSE, in CBSE marks, when one feature is shuffled. It measures dependence, not causation, and is unstable when features are correlated."
        >
          {importance.status === "loading" ? <Loading /> : null}
          {importance.status === "error" ? <ErrorNote error={importance.error} /> : null}
          {importance.data ? (
            <div className="chart-height">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={importance.data.rows}
                  layout="vertical"
                  margin={{ top: 4, right: 16, bottom: 4, left: 10 }}
                >
                  <XAxis
                    type="number"
                    tick={{ fontSize: 11, fill: MUTED }}
                    tickLine={false}
                    axisLine={false}
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
                      `+${value.toFixed(3)} RMSE`,
                      "when shuffled",
                    ]}
                    labelFormatter={humanize}
                    contentStyle={tooltipStyle}
                  />
                  <Bar dataKey="importance" isAnimationActive={false} barSize={12}>
                    {importance.data.rows.map((row) => (
                      <Cell
                        key={row.feature}
                        fill={row.importance > 0 ? ACCENT : MUTED}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : null}
        </ChartFrame>
      </div>

      <ChartFrame
        title="Mean final marks by attribute"
        subtitle={
          breakdown.data
            ? `${humanize(attribute)} - Mathematics n=${breakdown.data[0].n}, Language n=${breakdown.data[1].n}`
            : "Group means"
        }
        actions={
          <select
            value={attribute}
            onChange={(event) => setAttribute(event.target.value)}
            aria-label="Attribute"
          >
            {options.data.breakdown_attributes.map((name) => (
              <option key={name} value={name}>
                {humanize(name)}
              </option>
            ))}
          </select>
        }
        footnote="Descriptive group means with no adjustment for anything else. Hover a bar to see the group size; a mean from a handful of students is noise."
      >
        {breakdown.status === "loading" ? <Loading /> : null}
        {breakdown.status === "error" ? <ErrorNote error={breakdown.error} /> : null}
        {breakdown.data ? (
          <div className="chart-height">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={breakdownRows}
                margin={{ top: 8, right: 8, bottom: 4, left: -18 }}
              >
                <CartesianGrid stroke="rgba(242, 242, 240, 0.07)" vertical={false} />
                <XAxis
                  dataKey="value"
                  tick={{ fontSize: 11, fill: MUTED }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(242, 242, 240, 0.18)" }}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fontSize: 11, fill: MUTED }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value, name, entry) => {
                    const count = name === "Mathematics" ? entry.payload.matN : entry.payload.porN;
                    return [`${formatNumber(value, 2)} (n=${count})`, name];
                  }}
                />
                <Legend iconType="square" iconSize={8} wrapperStyle={{ fontSize: 12, color: "#a6a6a1" }} />
                <Bar dataKey="mat" name="Mathematics" fill={ACCENT} isAnimationActive={false} />
                <Bar dataKey="por" name="Language" fill={MUTED} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : null}
      </ChartFrame>

      <ChartFrame
        title="Chosen model per subject and variant"
                  subtitle="5-fold CV on training rows, beside the single held-out evaluation, in CBSE marks"
          footnote="Held-out scores were computed once, on the fixed 80/20 split, and are reported as recorded. They are not a second estimate. RMSE and MAE are multiplied by five from the dataset's 0-20 grade points."
      >
        {comparison.status === "loading" ? <Loading /> : null}
        {comparison.status === "error" ? <ErrorNote error={comparison.error} /> : null}
        {comparison.data ? (
          <DataTable
            rows={comparison.data.chosen}
            rowKey={(row) => `${row.subject}-${row.feature_set}`}
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
                render: (row) => formatNumber(row.cv_rmse, 3),
              },
              {
                key: "baseline_cv_rmse",
                label: "Baseline",
                align: "right",
                render: (row) => formatNumber(row.baseline_cv_rmse, 3),
              },
              {
                key: "test_rmse",
                label: "Test RMSE",
                align: "right",
                render: (row) => formatNumber(row.test_rmse, 3),
              },
              {
                key: "test_mae",
                label: "Test MAE",
                align: "right",
                render: (row) => formatNumber(row.test_mae, 3),
              },
              {
                key: "test_r2",
                label: "Test R2",
                align: "right",
                render: (row) => formatNumber(row.test_r2, 3),
              },
              {
                key: "within_10_marks",
                label: "Within 10",
                align: "right",
                render: (row) => `${(row.within_10_marks * 100).toFixed(0)}%`,
              },
            ]}
          />
        ) : null}
      </ChartFrame>

      <div className="grid grid-2">
        <ChartFrame
          title="Every candidate, compared"
          subtitle="Same folds for every model, mean baseline as the floor, in CBSE marks"
          footnote={comparison.data?.selection_rule}
        >
          {comparison.data ? (
            <DataTable
              rows={comparison.data.candidates}
              rowKey={(row) => `${row.subject}-${row.feature_set}-${row.name}`}
              columns={[
                {
                  key: "subject",
                  label: "Set",
                  render: (row) =>
                    `${subjectLabel(row.subject)} / ${row.feature_set.replace("_", " ")}`,
                },
                {
                  key: "name",
                  label: "Model",
                  render: (row) =>
                    row.baseline ? `${row.name} (baseline)` : row.selected ? `${row.name} *` : row.name,
                },
                {
                  key: "cv_rmse",
                  label: "CV RMSE",
                  align: "right",
                  render: (row) =>
                    `${formatNumber(row.cv_rmse, 3)} +-${formatNumber(row.cv_rmse_std, 2)}`,
                },
                {
                  key: "cv_mae",
                  label: "MAE",
                  align: "right",
                  render: (row) => formatNumber(row.cv_mae, 3),
                },
                {
                  key: "cv_r2",
                  label: "R2",
                  align: "right",
                  render: (row) => formatNumber(row.cv_r2, 3),
                },
              ]}
            />
          ) : comparison.status === "loading" ? (
            <Loading />
          ) : (
            <ErrorNote error={comparison.error} />
          )}
        </ChartFrame>

        <ChartFrame
          title="Subject comparison"
          subtitle="Two populations, described separately"
          footnote={subjects.data?.caveat}
        >
          {subjects.data ? (
            <DataTable
              rows={subjects.data.subjects}
              rowKey={(row) => row.subject}
              columns={[
                { key: "subject", label: "Subject", render: (row) => subjectLabel(row.subject) },
                { key: "rows", label: "Rows", align: "right" },
                {
                  key: "mean",
                  label: "Mean marks",
                  align: "right",
                  render: (row) => formatNumber(row.mean, 2),
                },
                {
                  key: "median",
                  label: "Median",
                  align: "right",
                  render: (row) => formatNumber(row.median, 2),
                },
                {
                  key: "pass_rate",
                  label: "Pass rate",
                  align: "right",
                  render: (row) => `${(row.pass_rate * 100).toFixed(1)}%`,
                },
                { key: "at_risk", label: "Below 33", align: "right" },
                {
                  key: "mean_absences",
                  label: "Absences",
                  align: "right",
                  render: (row) => formatNumber(row.mean_absences, 1),
                },
              ]}
            />
          ) : subjects.status === "loading" ? (
            <Loading />
          ) : (
            <ErrorNote error={subjects.error} />
          )}
        </ChartFrame>
      </div>
    </div>
  );
}

function SubjectToggle({ subject, onChange }) {
  return (
    <div className="segmented" role="group" aria-label="Subject">
      {Object.entries(SUBJECT_LABELS).map(([value, label]) => (
        <button
          key={value}
          type="button"
          className={subject === value ? "is-active" : ""}
          onClick={() => onChange(value)}
        >
          {label}
        </button>
      ))}
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
