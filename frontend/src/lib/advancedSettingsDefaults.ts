/**
 * Factory defaults for the advanced analysis settings, shared by the project
 * Settings page and the folder-run model step so their "Restore defaults"
 * buttons and "is anything non-default?" checks stay in sync.
 *
 * Scope is the tuning params only. The model/species selection (classification
 * model, embedding model, country/state, excluded classes) is a deliberate
 * user choice, not a default, so it is intentionally excluded. Note the
 * The rollup threshold is fixed policy (backend
 * app/core/confidence.py), not a setting, so it is not here.
 */

import {
  DEFAULT_CLASSIFICATION_GATE,
  DEFAULT_COUNTING_THRESHOLD,
} from "./confidence";
import type {
  FieldValues,
  Path,
  PathValue,
  UseFormSetValue,
} from "react-hook-form";

export const ADVANCED_SETTINGS_DEFAULTS = {
  detection_model_id: "MD5A-0-0",
  video_fps: 1.0,
  counting_threshold: DEFAULT_COUNTING_THRESHOLD,
  classification_gate: DEFAULT_CLASSIFICATION_GATE,
  event_smoothing: true,
  smoothing_strength: "normal",
  taxonomic_rollup: true,
  independence_interval: 1800,
  detection_batch_size: null,
  classification_batch_size: null,
  embedding_batch_size: null,
  detection_augment: false,
  detection_image_size: null,
} as const;

/**
 * Choices for the "Detection image size" select, shared by the project
 * Settings page and the folder-run model step. The value is the long-edge
 * pixel size MegaDetector resizes to before detection; the "default" sentinel
 * maps to null (omit the flag, MD uses its model-native size). Only larger
 * steps are offered, since the reason to change it is spotting small or
 * distant animals, and MegaDetector advises against changing it otherwise.
 */
export const DETECTION_IMAGE_SIZE_DEFAULT = "default";
export const DETECTION_IMAGE_SIZE_OPTIONS: readonly {
  value: string;
  label: string;
}[] = [
  { value: DETECTION_IMAGE_SIZE_DEFAULT, label: "Model default (1280)" },
  { value: "1920", label: "1920" },
  { value: "2560", label: "2560" },
];

/**
 * Choices for the "Video frame rate" select, shared by the project Settings
 * page and the folder-run model step so they can't drift. Values are the
 * sampling rate in fps; the sub-1 range is labelled as "1 frame every N
 * seconds" because it reads clearer than a fractional fps.
 */
export const VIDEO_FPS_OPTIONS: readonly { value: string; label: string }[] = [
  { value: "0.1", label: "1 frame every 10 seconds" },
  { value: "0.25", label: "1 frame every 4 seconds" },
  { value: "0.5", label: "1 frame every 2 seconds" },
  { value: "1", label: "1 frame per second" },
  { value: "2", label: "2 frames per second" },
  { value: "3", label: "3 frames per second" },
  { value: "4", label: "4 frames per second" },
  { value: "10", label: "10 frames per second" },
];

type AdvancedKey = keyof typeof ADVANCED_SETTINGS_DEFAULTS;

const ADVANCED_KEYS = Object.keys(ADVANCED_SETTINGS_DEFAULTS) as AdvancedKey[];

/** True if any advanced setting in `values` deviates from its default.
 * Keys absent from `values` are skipped: not every consumer form carries
 * every advanced setting (the folder-run setup step has no
 * counting_threshold; that lives on the project Settings page only). */
export function isAnyAdvancedNonDefault(values: Record<string, unknown>): boolean {
  return ADVANCED_KEYS.some(
    (key) => key in values && values[key] !== ADVANCED_SETTINGS_DEFAULTS[key],
  );
}

/**
 * Reset every advanced setting to its default via the form's setValue. Marks
 * fields dirty so the change is treated like any other edit (the user still
 * has to save / run to apply it). The form `T` must carry these field names;
 * the casts are safe because every consumer's schema includes them.
 */
export function restoreAdvancedDefaults<T extends FieldValues>(
  setValue: UseFormSetValue<T>,
): void {
  for (const key of ADVANCED_KEYS) {
    setValue(
      key as Path<T>,
      ADVANCED_SETTINGS_DEFAULTS[key] as PathValue<T, Path<T>>,
      { shouldDirty: true },
    );
  }
}
