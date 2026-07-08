import {
  DEFAULT_CLASSIFICATION_GATE,
  DEFAULT_COUNTING_THRESHOLD,
} from "./confidence";

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
    "Hide detections below this confidence score from counts and views. " +
    "Verified observations are always included. " +
    `The default is ${DEFAULT_COUNTING_THRESHOLD}.`,
  classificationGate:
    "Detections below this confidence are not identified to species and " +
    "skip label review, but are still saved and exported. " +
    `The default is ${DEFAULT_CLASSIFICATION_GATE}.`,
  videoFrameRate:
    "How many frames per second to extract from videos for detection. Higher values find more but take longer. One frame per second is a good default.",
  independenceInterval:
    "Files at the same camera within this window are merged into one event. The default is 30 minutes.",
  smoothing:
    "Looks at all photos grouped into one event and changes an odd-one-out label to match the rest. Example: a burst that is mostly red deer with one stray roe deer gets the stray corrected to red deer.",
  taxonomicRollup:
    "When the model can't confidently name the exact species, it labels the animal with a broader group instead, such as genus or family. Example: unsure between two deer species, it labels the animal 'deer' rather than guessing one. On by default.",
} as const;
