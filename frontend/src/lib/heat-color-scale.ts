/**
 * Heat color scale for the observation rate map.
 *
 * Gradient per FRONTEND_CONVENTIONS.md:
 *   #f9f871 (light yellow, low)  ->  #0f6064 (dark teal, high)
 *
 * Ported from AddaxAI-Connect's color-scale.ts. Renamed functions
 * from "detection rate" to "rate" / "heat" since WebUI counts
 * observations (MaxN per event), not raw detections.
 */

import chroma from "chroma-js";

const GRADIENT = chroma.scale(["#f9f871", "#0f6064"]).mode("lab");

/**
 * Pick a color for a single rate value against a normalization max.
 *
 * Zero always returns the lightest yellow, so deployments with effort
 * but no observations are visually distinct from high-rate hotspots.
 */
export function getRateColor(rate: number, maxRate?: number): string {
  if (rate <= 0) {
    return "#f9f871";
  }
  const normalized =
    maxRate && maxRate > 0 ? Math.min(rate / maxRate, 1.0) : 0.5;
  return GRADIENT(normalized).hex();
}

export interface RateScaleDomain {
  min: number;
  max: number;
  p33: number;
  p66: number;
}

/**
 * Compute min/max + p33/p66 percentiles over the *non-zero* rates.
 *
 * Using percentiles instead of raw max prevents a single outlier from
 * flattening the gradient. p66 is the recommended normalization max
 * when coloring markers.
 */
export function calculateRateDomain(rates: number[]): RateScaleDomain {
  const nonZero = rates.filter((r) => r > 0);
  if (nonZero.length === 0) {
    return { min: 0, max: 0, p33: 0, p66: 0 };
  }

  const sorted = [...nonZero].sort((a, b) => a - b);
  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    p33: sorted[Math.floor(sorted.length * 0.33)],
    p66: sorted[Math.floor(sorted.length * 0.66)],
  };
}
