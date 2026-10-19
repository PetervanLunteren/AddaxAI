/**
 * The species selection a form carries for its classification model: the
 * geofence country and US state, and the excluded label names. The country
 * is only the preset the list was built from; the user may edit the list
 * after, so nothing re-derives one from the other. They stay matched by one
 * rule instead: a change of classification model clears all three, and the
 * user picks the country again (submit already requires that for a
 * geofenced model).
 *
 * Before this rule the country survived a model switch while the list was
 * pruned to the names the new model happened to share. A SpeciesNet run
 * then said "Australia" and ran with 4 of 2,498 labels excluded
 * (2026-10-17). Every form with a classification model dropdown calls this
 * from the dropdown's change handler, never from an effect: an effect keyed
 * on the model id also fires on load and on restore, where nothing changed.
 */

import type { FieldValues, Path, PathValue, UseFormSetValue } from "react-hook-form";

const CLEARED = {
  country_code: null,
  state_code: null,
  excluded_classes: [] as string[],
};

/**
 * Clear the species selection via the form's setValue, marked dirty like
 * any other edit. The form `T` must carry these field names; the casts are
 * safe because every consumer's schema includes them (same shape as
 * `restoreAdvancedDefaults`).
 */
export function clearSpeciesSelection<T extends FieldValues>(
  setValue: UseFormSetValue<T>,
): void {
  for (const [key, value] of Object.entries(CLEARED)) {
    setValue(key as Path<T>, value as PathValue<T, Path<T>>, { shouldDirty: true });
  }
}
