/**
 * Last-used analysis settings, persisted in localStorage. One store shared by
 * both flows so "pre-fill from what I did last time" stays a single concept:
 *
 *   - The folder-run wizard exposes every field and writes them all on Start
 *     analysis, so folder run 2 starts identical to folder run 1.
 *   - The create-project dialog only exposes model + species, so it reads and
 *     writes just that subset (a merge — it never touches batch sizes,
 *     threshold, etc.). Creating a project therefore updates the shared model
 *     choice but leaves the folder-run-only params alone.
 *
 * Every field is optional and writes merge into the existing record, so a
 * subset write from the project dialog is safe and each field has a single
 * source of truth (no duplicated stores).
 *
 * Resuming an existing folder run seeds from that run's own project row
 * instead — the run's actual settings must win over "last used".
 *
 * The folder path is deliberately not stored: it's run-specific. Saved on
 * commit (Start analysis / project create), not on every keystroke, so we only
 * remember settings actually used. Missing models are NOT validated here; the
 * forms' model-status badges flag anything needing setup rather than silently
 * swapping the user's choice.
 */

import type { SeparateGroupBy } from "../api/folder-runs";

const KEY = "addaxai.folderRun.lastSettings";

export interface PersistedAnalysisSettings {
  detection_model_id?: string;
  classification_model_id?: string | null;
  embedding_model_id?: string | null;
  excluded_classes?: string[];
  country_code?: string | null;
  state_code?: string | null;
  detection_batch_size?: number | null;
  classification_batch_size?: number | null;
  embedding_batch_size?: number | null;
  counting_threshold?: number;
  classification_gate?: number;
  video_fps?: number;
  event_smoothing?: boolean;
  smoothing_strength?: "mild" | "normal" | "aggressive";
  taxonomic_rollup?: boolean;
  independence_interval?: number;
}

/**
 * Merge `patch` into the stored settings. The folder-run wizard passes the
 * full set (so it overwrites everything); the project dialog passes only the
 * model + species fields (leaving the rest intact).
 */
export function saveLastUsedSettings(
  patch: Partial<PersistedAnalysisSettings>,
): void {
  try {
    const prev = loadLastUsedSettings() ?? {};
    localStorage.setItem(KEY, JSON.stringify({ ...prev, ...patch }));
  } catch {
    // localStorage can be unavailable / full; remembering settings is
    // a convenience, not load-bearing, so swallow and move on.
  }
}

export function loadLastUsedSettings(): PersistedAnalysisSettings | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return parsed as PersistedAnalysisSettings;
    }
    return null;
  } catch {
    return null;
  }
}

// ── Save outputs step ───────────────────────────────────────────────
//
// Same idea as above: remember the last-used output choices so the next
// run's Save step starts where the user left off. Saved on Save (the
// moment the user commits the outputs), not on every toggle. The output
// folder is deliberately not stored — it's derived per run from the
// source folder, not a sticky preference.

const SAVE_OUTPUTS_KEY = "addaxai.folderRun.lastSaveOutputs";

export interface PersistedSaveOutputsSettings {
  exportEnabled: boolean;
  csv: boolean;
  xlsx: boolean;
  recognitionJson: boolean;
  mediaEnabled: boolean;
  groupBy: SeparateGroupBy;
  groupEvents: boolean;
  speciesLast: boolean;
  copyEmpties: boolean;
  drawBoxes: boolean;
  blur: boolean;
  /** Media-output confidence slider. Absent in objects stored by older
   * versions; readers fall back to the 0.2 default. */
  mediaConfidence?: number;
}

export function saveLastUsedSaveOutputs(
  settings: PersistedSaveOutputsSettings,
): void {
  try {
    localStorage.setItem(SAVE_OUTPUTS_KEY, JSON.stringify(settings));
  } catch {
    // Convenience only; ignore storage failures.
  }
}

export function loadLastUsedSaveOutputs(): PersistedSaveOutputsSettings | null {
  try {
    const raw = localStorage.getItem(SAVE_OUTPUTS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return parsed as PersistedSaveOutputsSettings;
    }
    return null;
  } catch {
    return null;
  }
}
