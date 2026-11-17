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
import { downloadBlob } from "./download";
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

/** Names for a filter column, or "all" without a filter. */
function namesOrAll(ids: string[] | undefined, names: Record<string, string>): string {
  return ids?.length ? ids.map((id) => names[id] ?? id).join("; ") : "all";
}

/**
 * The filter columns of a download. The dates are the date filter when
 * one is set, else the first and last day with effort, so the file always
 * says which days it covers.
 */
export function trendCsvScope(filters: {
  dateFrom?: string;
  dateTo?: string;
  firstDate?: string | null;
  lastDate?: string | null;
  siteIds?: string[];
  siteNames: Record<string, string>;
  labelIds?: string[];
  labelNames: Record<string, string>;
}): TrendCsvScope {
  return {
    dateFrom: filters.dateFrom ?? filters.firstDate ?? "",
    dateTo: filters.dateTo ?? filters.lastDate ?? "",
    sites: namesOrAll(filters.siteIds, filters.siteNames),
    labels: namesOrAll(filters.labelIds, filters.labelNames),
  };
}

/** Saves a download named after what it holds and its dates, so a
 *  "before" and an "after" file can be told apart. */
export function saveTrendCsv(name: string, scope: TrendCsvScope, csv: string): void {
  const dates = scope.dateFrom && scope.dateTo ? `-${scope.dateFrom}-to-${scope.dateTo}` : "";
  downloadBlob(
    new Blob([csv], { type: "text/csv;charset=utf-8" }),
    `addaxai-${name}${dates}.csv`,
  );
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
