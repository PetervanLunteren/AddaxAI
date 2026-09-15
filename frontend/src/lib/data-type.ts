/**
 * The "Data type" toggle: camera trap or underwater.
 *
 * One rule, shared by the three setup forms (folder-run setup, create
 * project, project settings):
 *
 *   - A form's data type is its detector's catalog `domain`. Nothing else
 *     is stored on the project, so the two can never disagree.
 *   - The toggle shows only the detectors and classifiers of that type.
 *     Choosing a type swaps any model that does not fit it
 *     (`applyDataType`).
 *   - The user's last choice is remembered on this machine and is where a
 *     new folder run or project starts. An existing project or a resumed
 *     run shows its own type, read from its detector.
 *
 * Why the filtering is not cosmetic: classification runs on every box that
 * is not a person or vehicle, and a box whose top class is "blank" is
 * never loaded. A camera trap classifier on an underwater detector would
 * drop sharks without a word. The project API refuses that pair too
 * (`_refuse_mixed_domains` in backend `routers/projects.py`).
 */

import type {
  FieldValues,
  Path,
  PathValue,
  UseFormGetValues,
  UseFormSetValue,
} from "react-hook-form";

import type { ModelInfo } from "../api/types";
import { loadLastUsedSettings, saveLastUsedSettings } from "./folderRunSettings";
import { NO_MODEL_VALUE } from "./model-id";

export type DataType = "camera_trap" | "underwater";

/** Toggle order and labels. Values are the catalog's `domain` values. */
export const DATA_TYPE_OPTIONS: readonly { value: DataType; label: string }[] = [
  { value: "camera_trap", label: "Camera trap" },
  { value: "underwater", label: "Underwater" },
];

/** The detector a form moves to when its detector does not fit the chosen
 *  type: the general-purpose model of each type. */
export const DEFAULT_DETECTOR: Record<DataType, string> = {
  camera_trap: "MD5A-0-0",
  underwater: "CFD-NANO-1-0",
};

const FALLBACK_DATA_TYPE: DataType = "camera_trap";

function isDataType(value: unknown): value is DataType {
  return value === "camera_trap" || value === "underwater";
}

/** The data type a model is for, or null when the catalog does not say. */
export function dataTypeOf(model: ModelInfo | null | undefined): DataType | null {
  return isDataType(model?.domain) ? model.domain : null;
}

/** Models of one data type. A model without a domain is in neither list;
 *  the catalog test requires one on every detector and classifier. */
export function modelsForDataType(models: ModelInfo[], dataType: DataType): ModelInfo[] {
  return models.filter((m) => m.domain === dataType);
}

/** The user's last choice on this machine, or null before they made one. */
export function savedDataType(): DataType | null {
  const saved = loadLastUsedSettings()?.data_type;
  return isDataType(saved) ? saved : null;
}

/** The user's last choice, camera trap when there is none. For display
 *  only while a form's detector is unknown; a new form squares itself with
 *  `savedDataType()` so that, before any choice was made, the restored
 *  detector decides (an underwater user from before the toggle keeps
 *  their fish detector). */
export function loadDataType(): DataType {
  return savedDataType() ?? FALLBACK_DATA_TYPE;
}

/** The fields a data type switch touches. Every setup form carries them;
 *  the form type itself stays generic, as in `restoreAdvancedDefaults`. */
type DataTypeFields = {
  detection_model_id: string;
  classification_model_id?: string | null;
  excluded_classes: string[];
  country_code?: string | null;
  state_code?: string | null;
};

/**
 * Put a form on `dataType`: keep every model that fits it, swap the rest.
 * A detector of another type becomes the type's default detector; a
 * classifier of another type becomes none, and its species selection goes
 * with it. Remembers the choice for the next new run or project.
 *
 * Idempotent, so it is also how a new form is squared with the saved
 * choice once the model lists have loaded.
 */
export function applyDataType<T extends FieldValues>(
  dataType: DataType,
  form: { getValues: UseFormGetValues<T>; setValue: UseFormSetValue<T> },
  models: { detectors: ModelInfo[]; classifiers: ModelInfo[] },
): void {
  saveLastUsedSettings({ data_type: dataType });
  const set = (key: keyof DataTypeFields, value: DataTypeFields[keyof DataTypeFields]) =>
    form.setValue(key as Path<T>, value as PathValue<T, Path<T>>, { shouldDirty: true });

  const detectorId = form.getValues("detection_model_id" as Path<T>) as string;
  const detector = models.detectors.find((m) => m.model_id === detectorId);
  if (dataTypeOf(detector) !== dataType) {
    const fits = modelsForDataType(models.detectors, dataType);
    const next =
      fits.find((m) => m.model_id === DEFAULT_DETECTOR[dataType]) ?? fits[0];
    if (next) set("detection_model_id", next.model_id);
  }

  const classifierId = form.getValues("classification_model_id" as Path<T>) as
    | string
    | null
    | undefined;
  if (classifierId && classifierId !== NO_MODEL_VALUE) {
    const classifier = models.classifiers.find((m) => m.model_id === classifierId);
    if (dataTypeOf(classifier) !== dataType) {
      set("classification_model_id", NO_MODEL_VALUE);
      set("excluded_classes", []);
      set("country_code", null);
      set("state_code", null);
    }
  }
}

/** True when switching to `dataType` would drop the chosen classifier.
 *  The project Settings page asks before it does. */
export function switchDropsClassifier(
  dataType: DataType,
  classifierId: string | null | undefined,
  classifiers: ModelInfo[],
): boolean {
  if (!classifierId || classifierId === NO_MODEL_VALUE) return false;
  const classifier = classifiers.find((m) => m.model_id === classifierId);
  return dataTypeOf(classifier) !== dataType;
}
