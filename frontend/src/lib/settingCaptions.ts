import {
  DEFAULT_CLASSIFICATION_GATE,
  DEFAULT_COUNTING_THRESHOLD,
  formatConfidencePct,
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
/**
 * The three batch-size captions share one message: the default is picked for
 * the hardware, and it is a power-user knob, not a speed dial (raising it does
 * not make CPU inference faster, and too large a value crashes a GPU run). One
 * helper keeps them identical so they cannot drift, as they had.
 */
const batchSizeCaption = (subject: string): string =>
  `How many ${subject} at once. The default is set for your hardware. ` +
  "Leave it unless you know what you are doing.";

export const SETTING_CAPTIONS = {
  detectionThreshold:
    "Hide detections below this confidence score from counts and views. " +
    "Verified observations are always included. " +
    `The default is ${formatConfidencePct(DEFAULT_COUNTING_THRESHOLD)}.`,
  classificationGate:
    "Detections below this confidence are not identified to species and " +
    "skip label review, but are still saved and exported. " +
    `The default is ${formatConfidencePct(DEFAULT_CLASSIFICATION_GATE)}.`,
  mediaFilter:
    "Which files the AI looks at. Use this when a camera recorded videos by mistake: skipping them is much faster than analysing them. Images and videos is the default.",
  videoFrameRate:
    "How many frames per second to extract from videos for detection. Higher values find more but take longer. One frame per second is a good default.",
  independenceInterval:
    "Files at the same camera within this window are merged into one event. The default is 30 minutes.",
  smoothing:
    "Looks at all photos grouped into one event and changes an odd-one-out label to match the rest. Example: a burst that is mostly red deer with one stray roe deer gets the stray corrected to red deer.",
  taxonomicRollup:
    "When the model can't confidently name the exact species, it labels the animal with a broader group instead, such as genus or family. Example: unsure between two deer species, it labels the animal 'deer' rather than guessing one. On by default.",
  detectionImageSize:
    "The pixel size images are resized to before detection. Larger can find small or distant animals but is slower and uses more memory. The model default is best in most cases.",
  imageAugmentation:
    "A slower detection mode that can find a few more animals in difficult images, but may add false positives. Off by default.",
  detectionBatchSize: batchSizeCaption("images the detection model handles"),
  classificationBatchSize: batchSizeCaption(
    "crops the classification model handles",
  ),
  embeddingBatchSize: batchSizeCaption("crops the embedding model handles"),
} as const;
