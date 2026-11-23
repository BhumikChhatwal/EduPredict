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
import { subjectLabel } from "../data/subjects";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const ACCENT = "#8fb6d9";
const MUTED = "#6e737a";

const BAND_ORDER = ["critical", "watch", "on_track", "strong"];
// Band cuts are CBSE marks out of 100; the lowest starts at the board's 33% pass mark.
const BAND_LABELS = {
  critical: "Critical (under 33)",
  watch: "Watch (33-50)",
  on_track: "On track (51-70)",
  strong: "Strong (71+)",
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
        description="A CBSE secondary-school view of one public dataset, four models, one honest error figure. Every number here is computed from the validated records and the saved training artifacts; grades are shown on the CBSE 0-100 marks scale."
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
          detail={`${data.students.mat} Mathematics / ${data.students.por} Language`}
        />
        <MetricCard
          label="Mean final marks"
          value={`${formatNumber(mat.mean)} / ${formatNumber(por.mean)}`}
          detail="Mathematics / Language, out of 100"
        />
        <MetricCard
          label="Below the pass mark"
          value={data.at_risk.total}
          detail="Actual marks under 33, both subjects"
        />
        <MetricCard
          label="Predictions logged"
          value={data.predictions_logged}
          detail="Recorded in the API's SQLite history"
        />
      </div>

      <div className="grid grid-2">
        <ChartFrame
          title="Final grade distribution"
          subtitle={`Students per 5-mark step - Mathematics n=${data.students.mat}, Language n=${data.students.por}`}
          footnote="The two cohorts are separate populations and are never pooled. Marks are the dataset's native 0-20 grades multiplied by five, plotted in 5-mark steps; the CBSE pass mark is 33."
        >
          <div className="chart-height">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogram} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                <CartesianGrid stroke="rgba(242, 242, 240, 0.07)" vertical={false} />
                <XAxis
                  dataKey="grade"
                  tick={{ fontSize: 11, fill: MUTED }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(242, 242, 240, 0.18)" }}
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
                <Legend iconType="square" iconSize={8} wrapperStyle={{ fontSize: 12, color: "#a6a6a1" }} />
                <Bar dataKey="mat" name="Mathematics" fill={ACCENT} isAnimationActive={false} />
                <Bar dataKey="por" name="Language" fill={MUTED} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartFrame>

        <ChartFrame
          title="Actual risk bands"
          subtitle={`Bands applied to the observed final grade of ${data.students.total} records`}
          footnote="Bands are fixed product cut-offs on CBSE marks, not learned thresholds and not a board standard."
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
        subtitle="Newest first, read from the API's SQLite history"
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
                `${subjectLabel(row.subject)} / ${row.feature_set.replace("_", " ")}`,
            },
            {
              key: "predicted_marks",
              label: "Marks",
              align: "right",
              render: (row) => formatNumber(row.predicted_marks, 1),
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
  background: "#15161a",
  border: "1px solid rgba(242, 242, 240, 0.18)",
  borderRadius: 2,
  color: "#f2f2f0",
};
