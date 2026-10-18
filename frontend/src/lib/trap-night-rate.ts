/**
 * How a count reads against survey effort on the dashboard's wildlife
 * chart: the bar value, the axis title and the tooltip line, from one
 * place so the three cannot disagree.
 *
 * With no trap nights (a project without capture dates yet) there is no
 * effort to divide by, so the chart shows the plain count and says so,
 * instead of a count under a "per 100 trap nights" axis.
 */

export type WildlifeCountMode = "events" | "max_n";

const UNIT: Record<WildlifeCountMode, { axis: string; one: string; many: string }> = {
  events: { axis: "Independent events", one: "event", many: "events" },
  max_n: { axis: "Observations", one: "individual", many: "individuals" },
};

export function wildlifeBarValue(count: number, trapNights: number): number {
  return trapNights > 0 ? +((count / trapNights) * 100).toFixed(2) : count;
}

export function wildlifeAxisTitle(
  mode: WildlifeCountMode,
  trapNights: number,
): string {
  const { axis } = UNIT[mode];
  return trapNights > 0 ? `${axis} per 100 trap nights` : axis;
}

/** Tooltip line: the rate, with the count and the effort it was made
 * from, so a reader can see why a bar is high or low. */
export function wildlifeTooltipLabel(
  mode: WildlifeCountMode,
  count: number,
  trapNights: number,
): string {
  const unit = UNIT[mode];
  const plain = `${count.toLocaleString()} ${count === 1 ? unit.one : unit.many}`;
  if (trapNights <= 0) return plain;
  const rate = wildlifeBarValue(count, trapNights).toLocaleString();
  const nights = `${trapNights.toLocaleString()} trap night${trapNights === 1 ? "" : "s"}`;
  return `${rate} per 100 trap nights (${plain} over ${nights})`;
}
