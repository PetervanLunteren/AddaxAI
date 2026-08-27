/**
 * What a person can record on a count row besides the number: sex, life
 * stage and behaviour. Mirrors `backend/app/core/observation_attributes.py`,
 * the same pairing as confidence.ts / confidence.py. The backend validates
 * against its copy, so a value that is not there is refused with a 422.
 *
 * Values are Camtrap DP's own (American spelling on the wire); the labels
 * are what the dropdowns show. Empty string in a select means unknown and
 * is sent as null.
 */

export interface AttributeOption {
  value: string;
  label: string;
}

export const SEX_OPTIONS: AttributeOption[] = [
  { value: "female", label: "Female" },
  { value: "male", label: "Male" },
];

export const LIFE_STAGE_OPTIONS: AttributeOption[] = [
  { value: "adult", label: "Adult" },
  { value: "subadult", label: "Subadult" },
  { value: "juvenile", label: "Juvenile" },
];

export const BEHAVIOR_OPTIONS: AttributeOption[] = [
  { value: "traveling", label: "Travelling" },
  { value: "foraging", label: "Foraging" },
  { value: "resting", label: "Resting" },
  { value: "vigilance", label: "Vigilance" },
  { value: "drinking", label: "Drinking" },
  { value: "grooming", label: "Grooming" },
  { value: "courtship", label: "Courtship" },
  { value: "nursing", label: "Nursing" },
  { value: "aggression", label: "Aggression" },
  { value: "marking", label: "Marking" },
];

/** The three dropdowns of a count row, in display order. */
export const OBSERVATION_ATTRIBUTES = [
  { field: "sex", label: "Sex", options: SEX_OPTIONS },
  // "Age" on screen: the panel is narrow and "Life stage" truncates. The
  // export column keeps Camtrap DP's `life_stage`.
  { field: "life_stage", label: "Age", options: LIFE_STAGE_OPTIONS },
  { field: "behavior", label: "Behaviour", options: BEHAVIOR_OPTIONS },
] as const;

export type ObservationAttributeField =
  (typeof OBSERVATION_ATTRIBUTES)[number]["field"];
