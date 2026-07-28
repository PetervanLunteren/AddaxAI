import type { CaptionedOption } from "../components/ui/captioned-select";

/**
 * Smoothing dropdown options, shared by the project settings and folder-run
 * forms so the wording and order stay identical (one source of truth).
 *
 * "off" maps to event_smoothing=false at each call site; the other three set
 * smoothing_strength (matching the `mild | normal | aggressive` enum and the
 * backend SMOOTHING_PRESETS).
 */
export const SMOOTHING_LEVELS: readonly CaptionedOption[] = [
  {
    value: "off",
    label: "Off",
    caption: "Keeps the AI's labels as is, outliers included.",
  },
  {
    value: "mild",
    label: "Mild",
    caption: "Fixes only the clearest outliers.",
  },
  {
    value: "normal",
    label: "Normal",
    caption: "A good default for most projects.",
  },
  {
    value: "aggressive",
    label: "Aggressive",
    caption:
      "Replaces more outliers with the dominant species. Best when multi-species events are rare.",
  },
];
