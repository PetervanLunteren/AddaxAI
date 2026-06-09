/**
 * Shared constants for the Labels view-options popover.
 *
 * Lives in a separate file from LabelsSettings.tsx so the
 * component file exports only the component (otherwise React
 * Fast Refresh complains: `react-refresh/only-export-components`).
 */

export const LABELS_MAX_DETECTIONS_OPTIONS = [
  { value: 5000, label: "5,000 labels (fastest)" },
  { value: 10000, label: "10,000 labels" },
  { value: 20000, label: "20,000 labels (default)" },
  { value: 35000, label: "35,000 labels" },
  { value: 50000, label: "50,000 labels (slowest)" },
];

export const LABELS_MAX_DETECTIONS_DEFAULT = 20000;
