/**
 * How a count reads against survey effort: the one place that scales,
 * rounds and names a rate, so no chart, popup or legend can disagree
 * with another.
 *
 * A rate is `count / trap nights x N`, and N is the user's rate unit:
 * 1, 7, 30 or 100 trap nights. It is a device-global display preference,
 * changed in View -> Rate unit and stored in localStorage, the same
 * shape as `species-name-mode.ts`: a module-level cache so any code can
 * read it synchronously, and a reload on change so every number on
 * screen flips at once.
 *
 * The backend and the exports always speak per 100 trap nights. Only
 * what is shown is scaled, and only here.
 *
 * With no trap nights (a project without capture dates yet) there is no
 * effort to divide by, so the wildlife chart shows the plain count and
 * says so, instead of a count under a rate axis.
 */

export type RateUnit = "night" | "week" | "month" | "hundred";

interface RateUnitSpec {
  /** Trap nights the rate is expressed per. */
  nights: number;
  /** Decimals shown. More for a smaller unit, so every unit carries the
   *  same precision and a rare species does not round to zero. */
  decimals: number;
  /** Reads after a number: "0.49 per 7 trap nights". */
  label: string;
  /** For the activity clock's centre, where the full label does not fit. */
  short: string;
}

export const RATE_UNITS: Record<RateUnit, RateUnitSpec> = {
  night: { nights: 1, decimals: 4, label: "per trap night", short: "/ night" },
  week: { nights: 7, decimals: 3, label: "per 7 trap nights", short: "/ 7 nights" },
  month: { nights: 30, decimals: 3, label: "per 30 trap nights", short: "/ 30 nights" },
  hundred: { nights: 100, decimals: 2, label: "per 100 trap nights", short: "/ 100 nights" },
};

const STORAGE_KEY = "addaxai:rate-unit";
const DEFAULT_RATE_UNIT: RateUnit = "hundred";

export function isRateUnit(value: unknown): value is RateUnit {
  return typeof value === "string" && Object.hasOwn(RATE_UNITS, value);
}

/** The unit a stored value stands for. Anything that is not a unit,
 *  a missing value included, is the default. */
export function parseRateUnit(stored: unknown): RateUnit {
  return isRateUnit(stored) ? stored : DEFAULT_RATE_UNIT;
}

function readStored(): RateUnit {
  try {
    return parseRateUnit(localStorage.getItem(STORAGE_KEY));
  } catch {
    // localStorage can throw in locked-down contexts; fall back to default.
    return DEFAULT_RATE_UNIT;
  }
}

let current: RateUnit = readStored();

export function getRateUnit(): RateUnit {
  return current;
}

export function setRateUnit(unit: RateUnit): void {
  current = unit;
  try {
    localStorage.setItem(STORAGE_KEY, unit);
  } catch {
    // Ignore persistence failure; the in-memory value still applies.
  }
  // A full reload redraws every rate everywhere, the Leaflet popups and
  // legend included, which are built outside React. Changing the unit is
  // rare, so the cost is acceptable and the result is simple.
  window.location.reload();
}

function roundTo(value: number, decimals: number): number {
  return +value.toFixed(decimals);
}

/** A count against its effort, in the rate unit. Callers check for
 *  effort first: there is no rate over zero trap nights. */
export function rateValue(
  count: number,
  trapNights: number,
  unit: RateUnit = current,
): number {
  const { nights, decimals } = RATE_UNITS[unit];
  return roundTo((count / trapNights) * nights, decimals);
}

/** A rate the backend already divided (always per 100 trap nights), in
 *  the rate unit. */
export function rateFromPer100(ratePer100: number, unit: RateUnit = current): number {
  const { nights, decimals } = RATE_UNITS[unit];
  return roundTo((ratePer100 * nights) / 100, decimals);
}

/** A rate as text: thousands separators, the unit's decimals at most,
 *  no trailing zeros. */
export function formatRate(value: number, unit: RateUnit = current): string {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: RATE_UNITS[unit].decimals,
  });
}

export function rateLabel(unit: RateUnit = current): string {
  return RATE_UNITS[unit].label;
}

/** A backend rate written out with its unit: "0.49 per 7 trap nights".
 *  For the map popups and the info sheets. */
export function describeRatePer100(ratePer100: number, unit: RateUnit = current): string {
  return `${formatRate(rateFromPer100(ratePer100, unit), unit)} ${rateLabel(unit)}`;
}

export type WildlifeCountMode = "events" | "max_n";

const UNIT: Record<WildlifeCountMode, { axis: string; one: string; many: string }> = {
  events: { axis: "Independent events", one: "event", many: "events" },
  max_n: { axis: "Observations", one: "individual", many: "individuals" },
};

export function wildlifeBarValue(
  count: number,
  trapNights: number,
  unit: RateUnit = current,
): number {
  return trapNights > 0 ? rateValue(count, trapNights, unit) : count;
}

export function wildlifeAxisTitle(
  mode: WildlifeCountMode,
  trapNights: number,
  unit: RateUnit = current,
): string {
  const { axis } = UNIT[mode];
  return trapNights > 0 ? `${axis} ${rateLabel(unit)}` : axis;
}

/** Tooltip line: the rate, with the count and the effort it was made
 * from, so a reader can see why a bar is high or low. */
export function wildlifeTooltipLabel(
  mode: WildlifeCountMode,
  count: number,
  trapNights: number,
  unit: RateUnit = current,
): string {
  const words = UNIT[mode];
  const plain = `${count.toLocaleString()} ${count === 1 ? words.one : words.many}`;
  if (trapNights <= 0) return plain;
  const rate = formatRate(rateValue(count, trapNights, unit), unit);
  const nights = `${trapNights.toLocaleString()} trap night${trapNights === 1 ? "" : "s"}`;
  return `${rate} ${rateLabel(unit)} (${plain} over ${nights})`;
}
