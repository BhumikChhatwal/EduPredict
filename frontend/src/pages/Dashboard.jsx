import { useMemo } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  BandBadge,
  ChartFrame,
  DataTable,
  ErrorNote,
  Loading,
  MetricCard,
  PageHeader,
  formatNumber,
} from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const ACCENT = "#1f5aa6";
const MUTED = "#8794a5";

const BAND_ORDER = ["critical", "watch", "on_track", "strong"];
const BAND_LABELS = {
  critical: "Critical (< 10)",
  watch: "Watch (10-13)",
  on_track: "On track (14-16)",
  strong: "Strong (17+)",
};

export default function Dashboard() {
  const summary = useApi(() => api.summary(), []);
  const distribution = useApi(() => api.distribution(), []);

  const histogram = useMemo(() => {
    if (!distribution.data) return [];
    const maths = distribution.data.subjects.find((item) => item.subject === "mat");
    const portuguese = distribution.data.subjects.find((item) => item.subject === "por");
    if (!maths || !portuguese) return [];

    return maths.counts.map((row, index) => ({
      grade: row.grade,
      mat: row.count,
      por: portuguese.counts[index]?.count ?? 0,
    }));
  }, [distribution.data]);

  if (summary.status === "loading") return <Loading label="Loading dashboard" />;
  if (summary.status === "error") return <ErrorNote error={summary.error} />;

  const data = summary.data;
  const mat = data.grades.mat;
  const por = data.grades.por;

  return (
    <div className="page">
      <PageHeader
        eyebrow="Overview"
        title="Student performance"
        description="Two public datasets, four models, one honest error figure. Every number on this page is computed from the validated dataset and the saved training records."
        actions={
          <Link className="button" to="/predict">
            Estimate a grade
          </Link>
        }
      />

      <div className="grid grid-4">
        <MetricCard
          label="Students"
          value={data.students.total.toLocaleString()}
          detail={`${data.students.mat} maths / ${data.students.por} Portuguese`}
        />
        <MetricCard
          label="Mean final grade"
          value={`${formatNumber(mat.mean)} / ${formatNumber(por.mean)}`}
          detail="Maths / Portuguese, out of 20"
        />
        <MetricCard
          label="Below the pass mark"
          value={data.at_risk.total}
          detail="Actual G3 under 10, both subjects"
        />
        <MetricCard
          label="Predictions logged"
          value={data.predictions_logged}
          detail="Stored locally; survives restart"
        />
      </div>

      <div className="grid grid-2">
        <ChartFrame
          title="Final grade distribution"
          subtitle={`Counts per grade point - maths n=${data.students.mat}, Portuguese n=${data.students.por}`}
          footnote="The two files are separate populations and are never pooled. Grades are discrete integers on a 0-20 scale; the pass mark is 10."
        >
          <div className="chart-height">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogram} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                <CartesianGrid stroke="#eef1f4" vertical={false} />
                <XAxis
                  dataKey="grade"
                  tick={{ fontSize: 11, fill: MUTED }}
                  tickLine={false}
                  axisLine={{ stroke: "#e2e5ea" }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: MUTED }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelFormatter={(grade) => `Grade ${grade}`}
                />
                <Legend iconType="square" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="mat" name="Maths" fill={ACCENT} isAnimationActive={false} />
                <Bar dataKey="por" name="Portuguese" fill={MUTED} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartFrame>

        <ChartFrame
          title="Actual risk bands"
          subtitle={`Bands applied to the observed final grade of ${data.students.total} records`}
          footnote="Bands are fixed product cut-offs, not learned thresholds and not an institutional standard."
        >
          <div className="band-chart">
            {BAND_ORDER.map((band) => (
              <div className="band-row" key={band}>
                <span className="band-row-label">{BAND_LABELS[band]}</span>
                {["mat", "por"].map((subject) => {
                  const count = data.bands[subject][band];
                  const total = data.students[subject];
                  return (
                    <div className="band-track-group" key={subject}>
                      <div className="band-track">
                        <span
                          className={`band-fill band-fill-${band}`}
                          style={{ width: `${Math.max(1, (count / total) * 100)}%` }}
                        />
                      </div>
                      <span className="band-count tabular">
                        {subject.toUpperCase()} {count}
                        <span className="band-share">
                          {" "}
                          ({(100 * (count / total)).toFixed(0)}%)
                        </span>
                      </span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </ChartFrame>
      </div>

      <ChartFrame
        title="Recent predictions"
        subtitle="Newest first, read from the local SQLite history"
        footnote="Submissions are stored with their attributes and the model version that answered, so an estimate can be audited later."
      >
        <DataTable
          rows={data.recent_predictions}
          rowKey={(row) => row.id}
          empty="No predictions yet. Submit one from the Predict page and it will appear here."
          columns={[
            { key: "id", label: "ID", align: "right" },
            {
              key: "subject",
              label: "Subject",
              render: (row) =>
                `${row.subject.toUpperCase()} / ${row.feature_set.replace("_", " ")}`,
            },
            {
              key: "predicted_g3",
              label: "Estimate",
              align: "right",
              render: (row) => formatNumber(row.predicted_g3, 1),
            },
            {
              key: "risk_band",
              label: "Band",
              render: (row) => <BandBadge band={row.risk_band} />,
            },
            { key: "family", label: "Model" },
            {
              key: "created_at",
              label: "Logged",
              align: "right",
              render: (row) => new Date(row.created_at).toLocaleString(),
            },
          ]}
        />
      </ChartFrame>
    </div>
  );
}

const tooltipStyle = {
  fontSize: 12,
  border: "1px solid #e2e5ea",
  borderRadius: 6,
  boxShadow: "0 1px 2px rgba(16, 27, 45, 0.06)",
};
