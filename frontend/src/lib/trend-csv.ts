/**
 * The Trend page's two downloads: the chart's periods and the species
 * table, each as the rows its card shows.
 *
 * Numbers are in the user's rate unit, like on screen, and
 * `per_trap_nights` names that unit (1, 7, 30 or 100). The filters are
 * repeated on every row, so a file still says what it is of when it is
 * opened a year later, and it stays a plain table.
 */

import type { RateSummary, SpeciesRate, TrendPeriod } from "../api/statistics";
import { toCsv, type CsvCell } from "./csv";
import { RATE_UNITS, getRateUnit, rateFromPer100, type RateUnit } from "./trap-night-rate";

/** The filters a download was made under, as text for its columns. */
export interface TrendCsvScope {
  dateFrom: string;
  dateTo: string;
  /** Site names, or "all". */
  sites: string;
  /** Label names, or "all". */
  labels: string;
}

const SUMMARY_HEADERS = ["n_sites", "mean", "sd", "ci_low", "ci_high"];
const SCOPE_HEADERS = ["per_trap_nights", "date_from", "date_to", "sites", "labels"];

function summaryCells(summary: RateSummary, unit: RateUnit): CsvCell[] {
  const scaled = (value: number | null) =>
    value === null ? null : rateFromPer100(value, unit);
  return [
    summary.n_sites,
    scaled(summary.mean),
    scaled(summary.sd),
    scaled(summary.ci_low),
    scaled(summary.ci_high),
  ];
}

function scopeCells(scope: TrendCsvScope, unit: RateUnit): CsvCell[] {
  return [RATE_UNITS[unit].nights, scope.dateFrom, scope.dateTo, scope.sites, scope.labels];
}

export function trendPeriodsCsv(
  periods: TrendPeriod[],
  scope: TrendCsvScope,
  unit: RateUnit = getRateUnit(),
): string {
  return toCsv(
    ["period", "period_start", "period_end", ...SUMMARY_HEADERS, ...SCOPE_HEADERS],
    periods.map((p) => [
      p.period,
      p.start,
      p.end,
      ...summaryCells(p, unit),
      ...scopeCells(scope, unit),
    ]),
  );
}

export function speciesRatesCsv(
  species: SpeciesRate[],
  scope: TrendCsvScope,
  unit: RateUnit = getRateUnit(),
): string {
  return toCsv(
    ["label", "common_name", "scientific_name", ...SUMMARY_HEADERS, ...SCOPE_HEADERS],
    species.map((s) => [
      s.label,
      s.common_name,
      s.scientific_name,
      ...summaryCells(s, unit),
      ...scopeCells(scope, unit),
    ]),
  );
}
