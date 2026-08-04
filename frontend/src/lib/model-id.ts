/**
 * The "no model" sentinel the optional model dropdowns carry, and the
 * one conversion back to what the API accepts.
 *
 * A Radix Select item cannot hold an empty string, so a form that offers
 * "No classification model" needs a real value to put in its place. That
 * value is form state only: the backend knows model ids or null, and
 * nothing else.
 *
 * This lives in lib rather than beside ModelSelect so importing it does
 * not drag a component in, and so the payload builders that need it are
 * not importing a dropdown to get a string function.
 */

/** Form-level value meaning "the user chose no model". */
export const NO_MODEL_VALUE = "none";

/**
 * Map a form-level model id to what the API accepts: the none sentinel
 * and empty values become null. Use at every payload build site that
 * sends `classification_model_id` or `embedding_model_id` to the
 * backend, the same way `toApiCountryCode` is used for the ALL sentinel.
 *
 * Letting the sentinel through stores the literal string "none" as the
 * project's model. The backend then reports it as an unknown model, and
 * the UI shows "none — Needs weights and environment" on every attempt
 * to run an analysis, so a project with no classifier cannot be run at
 * all until the value is corrected.
 */
export function toApiModelId(id: string | null | undefined): string | null {
  return !id || id === NO_MODEL_VALUE ? null : id;
}
