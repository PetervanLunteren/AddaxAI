/**
 * Base captions for settings that appear in BOTH the folder-run wizard and the
 * project settings page. One source of truth so the two surfaces stay in sync
 * (they had drifted: e.g. smoothing was one sentence in one place and a long
 * paragraph in the other).
 *
 * Each surface may append a short context note. Project settings change data
 * after a run, so it adds timing notes ("applies retroactively" / "new
 * analyses only"); the folder-run wizard sets everything before the single
 * run, so it uses the base caption alone.
 */
export const SETTING_CAPTIONS = {
  detectionThreshold:
    "Hide detections below this confidence score. Verified observations are always included.",
  videoFrameRate:
    "How many frames per second to extract from videos for detection. Higher values find more but take longer.",
  independenceInterval:
    "Files at the same camera within this window are merged into one event.",
  smoothing:
    "Cleans up species labels across an event, nudging the odd one out toward the rest.",
  taxonomicRollup:
    "When the model isn't sure of the exact species, it falls back to a broader group it's confident about, like genus or family.",
} as const;
