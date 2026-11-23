/**
 * The only module that knows about HTTP.
 *
 * Every page goes through `request`, so error handling, JSON parsing and the base path
 * exist in one place. A failed request becomes an `ApiError` with a human-readable
 * message: FastAPI's validation detail is an array of field errors, and rendering that
 * raw would put `{"loc": [...]}` in front of a user.
 */

const BASE = "/api";

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, { method = "GET", body } = {}) {
  let response;

  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(
      "Could not reach the API. Start the backend on port 8000 and reload.",
      null,
      null,
    );
  }

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    throw new ApiError(describeError(payload, response.status), response.status, payload);
  }

  return payload;
}

function describeError(payload, status) {
  const detail = payload?.detail;

  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const field = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "request";
        return `${field}: ${item.msg}`;
      })
      .join("; ");
  }

  if (status === 503) return "The model is not available.";
  return `Request failed with status ${status}.`;
}

export const api = {
  health: () => request("/health"),
  modelInfo: () => request("/model/info"),

  summary: () => request("/dashboard/summary"),

  options: () => request("/analytics/options"),
  distribution: (subject) =>
    request(`/analytics/distribution${subject ? `?subject=${subject}` : ""}`),
  correlation: (subject) => request(`/analytics/correlation?subject=${subject}`),
  featureImportance: (subject, featureSet, limit = 14) =>
    request(
      `/analytics/feature-importance?subject=${subject}&feature_set=${featureSet}&limit=${limit}`,
    ),
  modelComparison: () => request("/analytics/model-comparison"),
  subjectComparison: () => request("/analytics/subject-comparison"),
  attributes: (subject, attribute) =>
    request(`/analytics/attributes?subject=${subject}&attribute=${attribute}`),

  students: ({ subject, band, search, sort, page, pageSize }) => {
    const query = new URLSearchParams();
    if (subject) query.set("subject", subject);
    if (band) query.set("band", band);
    if (search) query.set("search", search);
    if (sort) query.set("sort", sort);
    query.set("page", String(page ?? 1));
    query.set("page_size", String(pageSize ?? 20));
    return request(`/students?${query.toString()}`);
  },
  student: (id) => request(`/students/${encodeURIComponent(id)}`),
  studentPredictions: (id) =>
    request(`/students/${encodeURIComponent(id)}/predictions`),

  recentPredictions: (limit = 8) => request(`/predictions/recent?limit=${limit}`),
  predict: (payload) => request("/predict", { method: "POST", body: payload }),
};
