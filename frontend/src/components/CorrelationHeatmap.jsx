/**
 * Correlation matrix as a small SVG grid.
 *
 * Recharts has no heatmap primitive, and pulling in a charting library for one grid
 * would be a dependency for decoration. Cell colour carries the sign, opacity the
 * strength; exact values are available in the cell tooltip so the display stays
 * readable instead of packing 256 numbers onto a page.
 */
export default function CorrelationHeatmap({ labels, matrix }) {
  const cell = 30;
  const pad = 96;
  const size = pad + labels.length * cell + 4;

  return (
    <div className="heatmap-wrap">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="heatmap"
        role="img"
        aria-label="Correlation matrix of numeric attributes"
      >
        {labels.map((label, row) => (
          <text
            key={`row-${label}`}
            x={pad - 8}
            y={pad + row * cell + cell / 2 + 3}
            textAnchor="end"
            className="heatmap-label"
          >
            {label}
          </text>
        ))}

        {labels.map((label, column) => (
          <text
            key={`col-${label}`}
            x={pad + column * cell + cell / 2}
            y={pad - 10}
            textAnchor="start"
            transform={`rotate(-55 ${pad + column * cell + cell / 2} ${pad - 10})`}
            className="heatmap-label"
          >
            {label}
          </text>
        ))}

        {matrix.map((row, rowIndex) =>
          row.map((value, columnIndex) => {
            const magnitude = value === null ? 0 : Math.min(1, Math.abs(value));
            const fill = value === null
              ? "rgba(242, 242, 240, 0.06)"
              : value >= 0
                ? "var(--accent)"
                : "var(--critical)";
            return (
              <rect
                key={`${rowIndex}-${columnIndex}`}
                x={pad + columnIndex * cell}
                y={pad + rowIndex * cell}
                width={cell - 2}
                height={cell - 2}
                rx={2}
                fill={fill}
                fillOpacity={value === null ? 1 : 0.1 + magnitude * 0.78}
              >
                <title>
                  {`${labels[rowIndex]} vs ${labels[columnIndex]}: ${
                    value === null ? "n/a" : value.toFixed(3)
                  }`}
                </title>
              </rect>
            );
          }),
        )}

        {matrix.map((row, rowIndex) =>
          row.map((value, columnIndex) =>
            magnitudeVisible(value) ? (
              <text
                key={`value-${rowIndex}-${columnIndex}`}
                x={pad + columnIndex * cell + (cell - 2) / 2}
                y={pad + rowIndex * cell + cell / 2 + 2}
                textAnchor="middle"
                className={`heatmap-value ${Math.abs(value) > 0.45 ? "is-strong" : ""}`}
              >
                {value.toFixed(2).replace("0.", ".")}
              </text>
            ) : null,
          ),
        )}
      </svg>
      <p className="heatmap-note">
        Blue is positive, red is negative; opacity shows strength. Hover a cell for its
        exact value. Correlation is linear association only - and G1, G2 and G3 are three
        periods of the same course, so their cells are strong by construction.
      </p>
    </div>
  );
}

function magnitudeVisible(value) {
  return value !== null && Math.abs(value) >= 0.35;
}
