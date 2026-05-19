/**
 * Shared constants for the Observations view-options popover.
 *
 * Lives in a separate file from ObservationsSettings.tsx so the
 * component file exports only the component (otherwise React
 * Fast Refresh complains: `react-refresh/only-export-components`).
 */

export const OBSERVATIONS_MAX_DETECTIONS_OPTIONS = [
  { value: 5000, label: "5,000 detections (fastest)" },
  { value: 10000, label: "10,000 detections" },
  { value: 20000, label: "20,000 detections (default)" },
  { value: 35000, label: "35,000 detections" },
  { value: 50000, label: "50,000 detections (slowest)" },
];

export const OBSERVATIONS_MAX_DETECTIONS_DEFAULT = 20000;
