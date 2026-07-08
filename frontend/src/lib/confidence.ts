/**
 * Single source of truth for the app's confidence defaults.
 *
 * Mirrors ``backend/app/core/confidence.py`` — keep the two files in
 * sync. One constant per concept; every form default, filter seed,
 * and advisory line references these.
 *
 * MD_OUTPUT_CONFIDENCE_THRESHOLD — what MegaDetector writes (its own
 * internal default). Everything above it exists in the database and
 * the data exports.
 *
 * DEFAULT_CLASSIFICATION_GATE — default detection confidence above
 * which animal crops are classified and embedded.
 *
 * DEFAULT_COUNTING_THRESHOLD — default detection confidence for
 * counting / visualization: the project threshold, the save step's
 * media confidence, and the labels grid's seeded filter. Below it most
 * detections are false positives, which is also why the grid's noise
 * advisory sits at the same value.
 *
 * The slider scale constants (0.01–1.00) live with the shared
 * ConfidenceSlider component.
 */

/**
 * Render a 0-1 confidence as a whole-percent string for display
 * (sliders, captions). Mirrors the backend format_confidence_pct. Data
 * (URL params, API payloads, exports) keeps the raw 0-1 value.
 */
export function formatConfidencePct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export const MD_OUTPUT_CONFIDENCE_THRESHOLD = 0.005;
export const DEFAULT_CLASSIFICATION_GATE = 0.1;
export const DEFAULT_COUNTING_THRESHOLD = 0.2;

/**
 * Detection-confidence noise advice, shared by every detection slider
 * (the filter ranges and the save step's media slider). Frontend-only
 * (UI copy, no backend mirror). The trigger value equals the counting
 * default, so both move together if that default ever changes.
 */
export const DETECTION_CONFIDENCE_ADVICE = {
  value: DEFAULT_COUNTING_THRESHOLD,
  message:
    "The lower the detection confidence, the more false positives to " +
    "expect. Review with care.",
} as const;
