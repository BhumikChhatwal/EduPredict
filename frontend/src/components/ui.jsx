/** Small shared presentational pieces. No data fetching lives here. */

export function ChartFrame({ title, subtitle, footnote, children, actions }) {
  return (
    <section className="card chart-frame">
      <header className="frame-head">
        <div>
          <h2 className="frame-title">{title}</h2>
          {subtitle ? <p className="frame-subtitle">{subtitle}</p> : null}
        </div>
        {actions ? <div className="frame-actions">{actions}</div> : null}
      </header>
      <div className="frame-body">{children}</div>
      {footnote ? <p className="frame-footnote">{footnote}</p> : null}
    </section>
  );
}

export function MetricCard({ label, value, detail }) {
  return (
    <div className="card metric">
      <p className="metric-label">{label}</p>
      <p className="metric-value tabular">{value}</p>
      {detail ? <p className="metric-detail">{detail}</p> : null}
    </div>
  );
}

const BAND_LABELS = {
  critical: "Critical",
  watch: "Watch",
  on_track: "On track",
  strong: "Strong",
};

export function BandBadge({ band, prefix }) {
  return (
    <span className={`badge band-${band}`}>
      {prefix ? <span className="badge-prefix">{prefix}</span> : null}
      {BAND_LABELS[band] ?? band}
    </span>
  );
}

export function DataTable({ columns, rows, rowKey, onRowClick, selectedKey, empty }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} className={column.align ? `align-${column.align}` : ""}>
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const key = rowKey ? rowKey(row) : JSON.stringify(row);
            return (
              <tr
                key={key}
                className={`${onRowClick ? "is-clickable" : ""} ${
                  selectedKey === key ? "is-selected" : ""
                }`}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
              >
                {columns.map((column) => (
                  <td
                    key={column.key}
                    className={column.align ? `align-${column.align}` : ""}
                  >
                    {column.render ? column.render(row) : row[column.key]}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.length === 0 ? <p className="empty">{empty ?? "Nothing to show."}</p> : null}
    </div>
  );
}

export function Loading({ label = "Loading" }) {
  return <p className="loading">{label}...</p>;
}

export function ErrorNote({ error }) {
  return (
    <div className="error-note" role="alert">
      <p className="error-title">That request did not complete</p>
      <p className="error-body">{error?.message ?? "Unknown error."}</p>
    </div>
  );
}

export function PageHeader({ eyebrow, title, description, actions }) {
  return (
    <header className="page-head">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="page-title">{title}</h1>
      </div>
      {description || actions ? (
        <div className="page-head-aside">
          {description ? <p className="page-desc">{description}</p> : null}
          {actions ? <div className="page-actions">{actions}</div> : null}
        </div>
      ) : null}
    </header>
  );
}

export function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toFixed(digits);
}

export function humanize(feature) {
  return String(feature).replaceAll("_", " ");
}
