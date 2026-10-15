/**
 * The data type: camera trap or underwater.
 *
 * One device-global preference, picked on the first-run setup screen and
 * changed in View -> Data type. Stored in localStorage, the same shape as
 * `species-name-mode.ts`: a module-level cache so non-React code can read
 * it synchronously, and a reload on change so every filtered list
 * re-renders consistently. Changing it is rare: next to nobody analyses
 * both camera trap and underwater footage.
 *
 * What it does:
 *
 *   - It decides what first-run setup installs (the detector and the
 *     environments; see `routers/setup.py`).
 *   - It picks the detector a new folder run or new project starts on,
 *     and filters the model pickers to that type.
 *
 * What it does not do: decide what an existing project is. A project's
 * data type is its detector's catalog `domain`, read from the project
 * itself, so the two can never disagree and a user whose preference says
 * camera trap still sees an underwater project exactly as it is.
 *
 * Why the filtering is not cosmetic: classification runs on every box
 * that is not a person or vehicle, and a box whose top class is "blank"
 * is never loaded. A camera trap classifier on an underwater detector
 * would drop sharks without a word. The project API refuses that pair
 * too (`_refuse_mixed_domains` in backend `routers/projects.py`).
 */

import { useSyncExternalStore } from "react";
import type {
  FieldValues,
  Path,
  PathValue,
  UseFormGetValues,
  UseFormSetValue,
} from "react-hook-form";

import type { ModelInfo } from "../api/types";
import { NO_MODEL_VALUE } from "./model-id";

export type DataType = "camera_trap" | "underwater";

/** Toggle order and labels. Values are the catalog's `domain` values. */
export const DATA_TYPE_OPTIONS: readonly { value: DataType; label: string }[] = [
  { value: "camera_trap", label: "Camera trap" },
  { value: "underwater", label: "Underwater" },
];

/** The detector a new form starts on, per type: the general-purpose
 *  model of each. The same two ids are the setup wizard's defaults. */
export const DEFAULT_DETECTOR: Record<DataType, string> = {
  camera_trap: "MD5A-0-0",
  underwater: "CFD-NANO-1-0",
};

const STORAGE_KEY = "addaxai:data-type";
const FALLBACK_DATA_TYPE: DataType = "camera_trap";

function isDataType(value: unknown): value is DataType {
  return value === "camera_trap" || value === "underwater";
}

function readStored(): DataType {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return isDataType(stored) ? stored : FALLBACK_DATA_TYPE;
  } catch {
    // localStorage can throw in locked-down contexts; fall back.
    return FALLBACK_DATA_TYPE;
  }
}

let current: DataType = readStored();
const listeners = new Set<() => void>();

/**
 * The user's data type, read once. Camera trap until they say otherwise,
 * which is what keeps every existing install working exactly as before.
 *
 * This is the snapshot: right for an effect, a form's initial values, or
 * any non-React caller. A component that *renders* the data type uses
 * `useDataType` instead, or it will not re-render when the setup screen
 * changes it.
 */
export function getDataType(): DataType {
  return current;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * The data type, as a React value. A plain module variable read during
 * render is invisible to React: the setup screen changes the type without
 * reloading, and anything that had already rendered would keep showing
 * the old answer until something else happened to re-render it. This is
 * the subscription React asks for (`useSyncExternalStore`), so every
 * reader updates together.
 */
export function useDataType(): DataType {
  return useSyncExternalStore(subscribe, getDataType);
}

/**
 * Store the choice and tell every subscriber.
 *
 * `reload` is for changing it from the menu of a running app. The
 * subscription above keeps every reader of the value in step by itself,
 * but a form that is already filled in is a different matter: its models
 * were chosen for the old type, and a reload is the simplest way to
 * start them over, the same bargain `species-name-mode` makes. The setup
 * screen leaves it off: there is no filled-in form there, and a reload
 * would restart the install poll for nothing.
 */
export function setDataType(
  dataType: DataType,
  { reload = false }: { reload?: boolean } = {},
): void {
  if (dataType === current) return;
  current = dataType;
  try {
    localStorage.setItem(STORAGE_KEY, dataType);
  } catch {
    // Ignore persistence failure; the in-memory value still applies.
  }
  listeners.forEach((listener) => listener());
  if (reload) window.location.reload();
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

/** The fields squaring a form with a data type touches. Every setup form
 *  carries them; the form type itself stays generic, as in
 *  `restoreAdvancedDefaults`. */
type DataTypeFields = {
  detection_model_id: string;
  classification_model_id?: string | null;
  excluded_classes: string[];
  country_code?: string | null;
  state_code?: string | null;
};

/**
 * Square a new form with the user's data type: keep every model that
 * fits it, swap the rest. A detector of another type becomes the type's
 * default detector; a classifier of another type becomes none, and its
 * species selection goes with it.
 *
 * This is for restored values, not for a choice made in the form: the
 * last-used settings of a run from before the preference changed can
 * carry models of the other type, and offering them would let the user
 * start a run the backend refuses. Idempotent, so it is safe to run on
 * every mount.
 */
export function fitFormToDataType<T extends FieldValues>(
  dataType: DataType,
  form: { getValues: UseFormGetValues<T>; setValue: UseFormSetValue<T> },
  models: { detectors: ModelInfo[]; classifiers: ModelInfo[] },
): void {
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
