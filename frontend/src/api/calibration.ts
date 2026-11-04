/**
 * Calibration endpoints: per-label score curves from "Is this a <label>?" answers.
 */

import { api } from "../lib/api-client";

export type CalibrationAnswerValue = "right" | "wrong" | "unsure";

export interface CalibrationPoint {
  score: number;
  p: number;
  lower: number;
  upper: number;
}

export interface CalibrationThreshold {
  target: number;
  /** Score from which the target holds; null when not reached. */
  score: number | null;
}

export interface CalibrationLabel {
  label: string;
  common_name: string;
  scientific_name: string;
  detections: number;
  right: number;
  wrong: number;
  unsure: number;
  right_needed: number;
  wrong_needed: number;
  thresholds: CalibrationThreshold[] | null;
  curve: CalibrationPoint[] | null;
  no_error_bound: number | null;
}

export interface CalibrationResponse {
  labels: CalibrationLabel[];
}

export interface CalibrationSampleItem {
  detection_id: string;
  score: number;
}

export const calibrationApi = {
  get: (projectId: string) =>
    api.get<CalibrationResponse>(
      `/api/calibration?project_id=${encodeURIComponent(projectId)}`,
    ),
  sample: (projectId: string, label: string) =>
    api.get<CalibrationSampleItem[]>(
      `/api/calibration/sample?project_id=${encodeURIComponent(projectId)}` +
        `&label=${encodeURIComponent(label)}`,
    ),
  answer: (detectionId: string, answer: CalibrationAnswerValue) =>
    api.post<void>("/api/calibration/answers", {
      detection_id: detectionId,
      answer,
    }),
};
