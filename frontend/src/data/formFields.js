/**
 * Form field definitions for the prediction form.
 *
 * Values and domains mirror the backend schema exactly (see `app/schemas/prediction.py`
 * and `ml/config.py`). Option labels are plain-language versions of the data
 * dictionary's codes, because "Medu = 3" means nothing without the legend.
 */

const yesNo = [
  { value: "no", label: "No" },
  { value: "yes", label: "Yes" },
];

const ordinal = (labels) =>
  labels.map((label, index) => ({ value: index + 1, label }));

const education = [
  { value: 0, label: "0 - none" },
  { value: 1, label: "1 - primary" },
  { value: 2, label: "2 - up to 9th grade" },
  { value: 3, label: "3 - secondary" },
  { value: 4, label: "4 - higher education" },
];

export const FIELD_GROUPS = [
  {
    title: "Academic history",
    description: "What the school already knows before the final period.",
    fields: [
      {
        name: "school",
        label: "School",
        type: "select",
        options: [
          { value: "GP", label: "GP - Gabriel Pereira" },
          { value: "MS", label: "MS - Mousinho da Silveira" },
        ],
      },
      { name: "age", label: "Age", type: "number", min: 15, max: 22 },
      {
        name: "studytime",
        label: "Weekly study time",
        type: "select",
        options: ordinal(["< 2 hours", "2 to 5 hours", "5 to 10 hours", "> 10 hours"]),
      },
      {
        name: "failures",
        label: "Past class failures",
        type: "select",
        options: [
          { value: 0, label: "None" },
          { value: 1, label: "1" },
          { value: 2, label: "2" },
          { value: 3, label: "3" },
          { value: 4, label: "4 or more (bucketed at 3+)" },
        ],
      },
      {
        name: "absences",
        label: "Recorded absences",
        type: "number",
        min: 0,
        max: 93,
        help: "Reported count for the year. This is an input, never a prediction target.",
      },
      {
        name: "schoolsup",
        label: "Extra school support",
        type: "select",
        options: yesNo,
      },
      {
        name: "higher",
        label: "Wants higher education",
        type: "select",
        options: yesNo,
      },
      {
        name: "paid",
        label: "Paid classes",
        type: "select",
        options: yesNo,
      },
      {
        name: "reason",
        label: "Reason for choosing the school",
        type: "select",
        options: [
          { value: "course", label: "Course preference" },
          { value: "home", label: "Close to home" },
          { value: "reputation", label: "School reputation" },
          { value: "other", label: "Other" },
        ],
      },
    ],
  },
  {
    title: "Family and household",
    description: "Background attributes. Descriptive, not levers.",
    fields: [
      { name: "sex", label: "Sex", type: "select", options: [
        { value: "F", label: "Female" },
        { value: "M", label: "Male" },
      ] },
      { name: "address", label: "Home area", type: "select", options: [
        { value: "U", label: "Urban" },
        { value: "R", label: "Rural" },
      ] },
      { name: "famsize", label: "Family size", type: "select", options: [
        { value: "GT3", label: "More than 3" },
        { value: "LE3", label: "3 or fewer" },
      ] },
      { name: "Pstatus", label: "Parents living together", type: "select", options: [
        { value: "T", label: "Together" },
        { value: "A", label: "Apart" },
      ] },
      { name: "Medu", label: "Mother's education", type: "select", options: education },
      { name: "Fedu", label: "Father's education", type: "select", options: education },
      { name: "Mjob", label: "Mother's occupation", type: "select", options: [
        { value: "at_home", label: "At home" },
        { value: "health", label: "Health" },
        { value: "services", label: "Services" },
        { value: "teacher", label: "Teacher" },
        { value: "other", label: "Other" },
      ] },
      { name: "Fjob", label: "Father's occupation", type: "select", options: [
        { value: "at_home", label: "At home" },
        { value: "health", label: "Health" },
        { value: "services", label: "Services" },
        { value: "teacher", label: "Teacher" },
        { value: "other", label: "Other" },
      ] },
      { name: "guardian", label: "Guardian", type: "select", options: [
        { value: "mother", label: "Mother" },
        { value: "father", label: "Father" },
        { value: "other", label: "Other" },
      ] },
      { name: "famsup", label: "Family educational support", type: "select", options: yesNo },
      {
        name: "famrel",
        label: "Family relationship quality",
        type: "select",
        options: ordinal(["1 - very bad", "2 - bad", "3 - fair", "4 - good", "5 - excellent"]),
      },
    ],
  },
  {
    title: "Lifestyle and routine",
    description:
      "Sensitive attributes. The model uses them; the interface never presents them as advice.",
    fields: [
      {
        name: "traveltime",
        label: "Travel time to school",
        type: "select",
        options: ordinal(["< 15 min", "15 to 30 min", "30 to 60 min", "> 1 hour"]),
      },
      {
        name: "freetime",
        label: "Free time after school",
        type: "select",
        options: ordinal(["1 - very low", "2 - low", "3 - moderate", "4 - high", "5 - very high"]),
      },
      {
        name: "goout",
        label: "Going out with friends",
        type: "select",
        options: ordinal(["1 - very low", "2 - low", "3 - moderate", "4 - high", "5 - very high"]),
      },
      {
        name: "Dalc",
        label: "Weekday alcohol use",
        type: "select",
        options: ordinal(["1 - very low", "2 - low", "3 - moderate", "4 - high", "5 - very high"]),
      },
      {
        name: "Walc",
        label: "Weekend alcohol use",
        type: "select",
        options: ordinal(["1 - very low", "2 - low", "3 - moderate", "4 - high", "5 - very high"]),
      },
      {
        name: "health",
        label: "Current health status",
        type: "select",
        options: ordinal(["1 - very bad", "2 - bad", "3 - fair", "4 - good", "5 - very good"]),
      },
      { name: "activities", label: "Extra-curricular activities", type: "select", options: yesNo },
      { name: "nursery", label: "Attended nursery school", type: "select", options: yesNo },
      { name: "internet", label: "Internet at home", type: "select", options: yesNo },
      { name: "romantic", label: "In a romantic relationship", type: "select", options: yesNo },
    ],
  },
];

export const GRADE_FIELDS = [
  {
    name: "G1",
    label: "First period grade (0-20)",
    type: "number",
    min: 0,
    max: 20,
  },
  {
    name: "G2",
    label: "Second period grade (0-20)",
    type: "number",
    min: 0,
    max: 20,
  },
];

export const DEFAULT_ATTRIBUTES = {
  school: "GP",
  sex: "F",
  age: 17,
  address: "U",
  famsize: "GT3",
  Pstatus: "T",
  Medu: 3,
  Fedu: 2,
  Mjob: "other",
  Fjob: "other",
  reason: "course",
  guardian: "mother",
  traveltime: 1,
  studytime: 2,
  failures: 0,
  schoolsup: "no",
  famsup: "yes",
  paid: "no",
  activities: "yes",
  nursery: "yes",
  higher: "yes",
  internet: "yes",
  romantic: "no",
  famrel: 4,
  freetime: 3,
  goout: 2,
  Dalc: 1,
  Walc: 2,
  health: 3,
  absences: 4,
  G1: 12,
  G2: 13,
};
