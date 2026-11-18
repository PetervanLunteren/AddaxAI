/**
 * Heat color scale: a value mapped onto a colour ramp, for the
 * observation rate map, the Explore mini map and the Deployment
 * timeline heatmap.
 *
 * The app's gradient is the teal one (FRONTEND_CONVENTIONS.md):
 *   #f9f871 (light yellow, low)  ->  #0f6064 (dark teal, high)
 * The Map page alone offers two more, because its background changes:
 * dark teal sinks into satellite imagery, and there the highest rates
 * are the ones that must stand out.
 */

import chroma from "chroma-js";

/** Every ramp the app knows. The first is the default. */
export const RATE_RAMPS = [
  { id: "teal", label: "Teal", scale: chroma.scale(["#f9f871", "#0f6064"]).mode("lab") },
  // ColorBrewer yellow-orange-red, stopped at its bright red: its last
  // quarter runs into a dark burgundy that sinks into dark imagery as
  // much as teal does, and there the highest rates must stand out.
  { id: "heat", label: "Heat", scale: chroma.scale("YlOrRd").padding([0, 0.25]) },
  // Colour-blind safe, and the usual choice in scientific figures.
  { id: "viridis", label: "Viridis", scale: chroma.scale("viridis") },
  // The two classic heat map looks, offered because people know them.
  // Rainbow ("jet") has uneven steps and fails colour-blind readers, so
  // it is a choice, never a default. Thermal ("hot") starts at black.
  {
    id: "rainbow",
    label: "Rainbow",
    scale: chroma.scale(["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000"]),
  },
  {
    id: "thermal",
    label: "Thermal",
    scale: chroma.scale(["#000000", "#ff0000", "#ffff00", "#ffffff"]),
  },
] as const;

export type RateRamp = (typeof RATE_RAMPS)[number]["id"];

export function isRateRamp(value: unknown): value is RateRamp {
  return RATE_RAMPS.some((ramp) => ramp.id === value);
}

function scaleOf(ramp: RateRamp) {
  return RATE_RAMPS.find((r) => r.id === ramp)!.scale;
}

/**
 * The colour of one value. The ramp runs from zero to `cap`; a value
 * at or above the cap gets the last colour.
 *
 * Zero gets the first colour, so a site with effort but no
 * observations is visually distinct from a hotspot.
 */
export function getRateColor(rate: number, cap?: number, ramp: RateRamp = "teal"): string {
  const scale = scaleOf(ramp);
  if (rate <= 0) return scale(0).hex();
  return scale(cap && cap > 0 ? Math.min(rate / cap, 1) : 0.5).hex();
}

/** The ramp as a CSS gradient, for a legend bar. */
export function rampGradient(ramp: RateRamp, direction: string): string {
  // Eleven stops: enough to follow a ramp with several colours.
  const stops = Array.from({ length: 11 }, (_, i) => scaleOf(ramp)(i / 10).hex());
  return `linear-gradient(${direction}, ${stops.join(", ")})`;
}

export interface RateScaleDomain {
  /** The highest value. */
  max: number;
  /** Where the ramp ends: the 95th percentile of the non-zero values. */
  cap: number;
}

/**
 * The scale for a set of values, over the non-zero ones.
 *
 * The ramp ends at the 95th percentile instead of the maximum, so one
 * outlier does not make every other value pale. With fewer than 20
 * values the 95th percentile is the maximum itself. A legend must
 * print `cap`, with a "+" when `max` lies above it.
 */
export function calculateRateDomain(rates: number[]): RateScaleDomain {
  const sorted = rates.filter((r) => r > 0).sort((a, b) => a - b);
  if (sorted.length === 0) return { max: 0, cap: 0 };
  return {
    max: sorted[sorted.length - 1],
    cap: sorted[Math.floor(sorted.length * 0.95)],
  };
}
