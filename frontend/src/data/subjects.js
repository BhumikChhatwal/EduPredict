/**
 * CBSE subject labels for the two cohorts the models were fitted on.
 *
 * `mat` maps to Mathematics cleanly. `por` is the dataset's Portuguese-language course,
 * so it is shown as "Language" rather than as English or Hindi: those are CBSE papers,
 * but the training cohort is neither, and naming it after one would claim a curriculum
 * match the data does not have.
 */
export const SUBJECT_LABELS = {
  mat: "Mathematics",
  por: "Language",
};

export function subjectLabel(code) {
  return SUBJECT_LABELS[code] ?? String(code).toUpperCase();
}
