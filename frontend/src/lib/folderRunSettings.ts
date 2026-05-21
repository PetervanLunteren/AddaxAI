/**
 * Last-used folder-run settings, persisted in localStorage.
 *
 * Folder mode is a single-device Electron flow, so the user's last
 * model + threshold choices are remembered locally and used to seed
 * the Setup form for the next brand-new run. Resuming an existing run
 * always seeds from that run's own project row instead — the run's
 * actual settings must win over "last used".
 *
 * The folder path is deliberately not stored: it's run-specific.
 *
 * Saved on Start analysis (the moment the user commits a run), not on
 * every keystroke, so we only remember settings that were actually
 * used. Missing models are NOT validated here: we keep whatever was
 * saved and let the Setup form's model-status badges flag anything
 * that needs setup, rather than silently swapping the user's choice.
 */

const KEY = "addaxai.folderRun.lastSettings";

export interface PersistedFolderRunSettings {
  detection_model_id: string;
  classification_model_id: string | null;
  embedding_model_id: string | null;
  excluded_classes: string[];
  country_code: string | null;
  state_code: string | null;
  detection_batch_size: number | null;
  classification_batch_size: number | null;
  embedding_batch_size: number | null;
  detection_threshold: number;
  video_fps: number;
  event_smoothing: boolean;
  smoothing_strength: "mild" | "normal" | "aggressive";
  taxonomic_rollup: boolean;
}

export function saveLastUsedSettings(
  settings: PersistedFolderRunSettings,
): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    // localStorage can be unavailable / full; remembering settings is
    // a convenience, not load-bearing, so swallow and move on.
  }
}

export function loadLastUsedSettings(): PersistedFolderRunSettings | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return parsed as PersistedFolderRunSettings;
    }
    return null;
  } catch {
    return null;
  }
}
