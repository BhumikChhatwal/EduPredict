import { useState } from "react";

import {
  BandBadge,
  DataTable,
  ErrorNote,
  Loading,
  PageHeader,
  formatNumber,
} from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const BANDS = [
  ["", "All bands"],
  ["critical", "Critical"],
  ["watch", "Watch"],
  ["on_track", "On track"],
  ["strong", "Strong"],
];

export default function Students() {
  const [subject, setSubject] = useState("");
  const [band, setBand] = useState("");
  const [sort, setSort] = useState("student_id");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState(null);

  const students = useApi(
    () =>
      api.students({
        subject: subject || undefined,
        band: band || undefined,
        sort,
        search: search || undefined,
        page,
        pageSize: 20,
      }),
    [subject, band, sort, search, page],
  );

  const detail = useApi(
    () => (selected ? api.student(selected) : Promise.resolve(null)),
    [selected],
  );
  const history = useApi(
    () => (selected ? api.studentPredictions(selected) : Promise.resolve([])),
    [selected],
  );

  function change(setter) {
    return (event) => {
      setter(event.target.value);
      setPage(1);
      setSelected(null);
    };
  }

  return (
    <div className="page">
      <PageHeader
        eyebrow="Records"
        title="Student records"
        description="The public UCI rows, described by their actual final grade. IDs are generated row indices - the dataset contains no student identifiers."
      />

      <div className="card">
        <div className="filters">
          <label className="control">
            <span className="control-label">Subject</span>
            <select value={subject} onChange={change(setSubject)}>
              <option value="">Both</option>
              <option value="mat">Mathematics</option>
              <option value="por">Portuguese</option>
            </select>
          </label>

          <label className="control">
            <span className="control-label">Actual band</span>
            <select value={band} onChange={change(setBand)}>
              {BANDS.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>

          <label className="control">
            <span className="control-label">Sort</span>
            <select value={sort} onChange={change(setSort)}>
              <option value="student_id">ID</option>
              <option value="grade_desc">Highest grade</option>
              <option value="grade_asc">Lowest grade</option>
              <option value="absences_desc">Most absences</option>
            </select>
          </label>

          <label className="control control-grow">
            <span className="control-label">Search by ID</span>
            <input
              type="search"
              placeholder="e.g. MAT-0001"
              value={search}
              onChange={change(setSearch)}
            />
          </label>
        </div>

        {students.status === "loading" ? <Loading label="Loading records" /> : null}
        {students.status === "error" ? <ErrorNote error={students.error} /> : null}

        {students.data ? (
          <>
            <DataTable
              rows={students.data.items}
              rowKey={(row) => row.student_id}
              selectedKey={selected}
              onRowClick={(row) => setSelected(row.student_id)}
              empty="No records match these filters."
              columns={[
                { key: "student_id", label: "ID" },
                { key: "subject", label: "Subject", render: (row) => row.subject.toUpperCase() },
                { key: "age", label: "Age", align: "right" },
                { key: "sex", label: "Sex" },
                { key: "studytime", label: "Study", align: "right" },
                { key: "failures", label: "Failures", align: "right" },
                { key: "absences", label: "Absences", align: "right" },
                { key: "G1", label: "G1", align: "right" },
                { key: "G2", label: "G2", align: "right" },
                { key: "G3", label: "G3", align: "right" },
                {
                  key: "band",
                  label: "Band",
                  render: (row) => <BandBadge band={row.band} prefix="Actual" />,
                },
              ]}
            />

            <div className="pagination">
              <span className="pagination-info">
                {(students.data.page - 1) * students.data.page_size + 1}-
                {Math.min(
                  students.data.page * students.data.page_size,
                  students.data.total,
                )}{" "}
                of {students.data.total}
              </span>
              <div className="pagination-actions">
                <button
                  type="button"
                  className="button button-secondary"
                  disabled={students.data.page <= 1}
                  onClick={() => setPage((current) => current - 1)}
                >
                  Previous
                </button>
                <span className="pagination-page">
                  Page {students.data.page} / {students.data.pages}
                </span>
                <button
                  type="button"
                  className="button button-secondary"
                  disabled={students.data.page >= students.data.pages}
                  onClick={() => setPage((current) => current + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        ) : null}
      </div>

      {selected ? (
        <div className="grid grid-2">
          <div className="card">
            <div className="detail-head">
              <div>
                <p className="eyebrow">Record</p>
                <h2 className="frame-title">{selected}</h2>
              </div>
              <button
                type="button"
                className="button button-secondary"
                onClick={() => setSelected(null)}
              >
                Close
              </button>
            </div>

            {detail.status === "loading" ? <Loading label="Loading record" /> : null}
            {detail.status === "error" ? <ErrorNote error={detail.error} /> : null}
            {detail.data ? (
              <>
                <p className="detail-note">{detail.data.note}</p>
                <div className="attribute-grid">
                  {Object.entries(detail.data.attributes).map(([name, value]) => (
                    <div className="attribute-row" key={name}>
                      <span className="attribute-name">{name}</span>
                      <span className="attribute-value tabular">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : null}
          </div>

          <div className="card">
            <h2 className="frame-title">Prediction history</h2>
            <p className="frame-subtitle">
              Estimates stored against this ID. A record usually has none: submissions
              are optional and only appear when the ID was supplied.
            </p>

            {history.status === "loading" ? <Loading /> : null}
            {history.status === "error" ? <ErrorNote error={history.error} /> : null}
            {history.data ? (
              <DataTable
                rows={history.data}
                rowKey={(row) => row.id}
                empty="No predictions stored for this record."
                columns={[
                  { key: "id", label: "ID", align: "right" },
                  {
                    key: "feature_set",
                    label: "Variant",
                    render: (row) => row.feature_set.replace("_", " "),
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
                  {
                    key: "created_at",
                    label: "Logged",
                    align: "right",
                    render: (row) => new Date(row.created_at).toLocaleString(),
                  },
                ]}
              />
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
