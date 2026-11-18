/**
 * "Download CSV" on the Insights pages: the numbers a page shows, as a
 * plain table (the rule is in FRONTEND_CONVENTIONS.md).
 *
 * Every file repeats the filters it was made under as columns on every
 * row, so it still says what it is of when it is opened a year later,
 * and it stays one plain table. Rates are in the user's rate unit, like
 * on screen, and `per_trap_nights` names that unit (1, 7, 30 or 100).
 *
 * Each view has one pure builder here, from its API response to CSV
 * text. Raw data is not this module's business: that is the Export page.
 */

import type { PerformanceResponse } from "../api/performance";
import type {
  ObservationRateMapFeature,
  RateSummary,
  SpeciesRate,
  TrendPeriod,
} from "../api/statistics";
import { toCsv, type CsvCell } from "./csv";
import { downloadBlob } from "./download";
import { RANK_OPTIONS, type TaxonomicRank } from "./taxonomic-rank";
import { RATE_UNITS, getRateUnit, rateFromPer100, type RateUnit } from "./trap-night-rate";

// ---------------------------------------------------------------------------
// Shared by every view
// ---------------------------------------------------------------------------

/** The filters a file was made under: column name and value, in order. */
export type CsvFilters = [name: string, value: CsvCell][];

/** A view's rows with its filters appended to every row. */
export function viewCsv(headers: string[], rows: CsvCell[][], filters: CsvFilters): string {
  const values = filters.map(([, value]) => value);
  return toCsv(
    [...headers, ...filters.map(([name]) => name)],
    rows.map((row) => [...row, ...values]),
  );
}

/** Names for a filter column, or "all" without a filter. */
export function namesOrAll(ids: string[] | undefined, names: Record<string, string>): string {
  return ids?.length ? ids.map((id) => names[id] ?? id).join("; ") : "all";
}

/** The dates a download covers, for its file name. */
export interface FileDates {
  from: string;
  to: string;
}

/**
 * `addaxai-<name>.<extension>`, with the dates in the name when the view
 * has both, so a "before" and an "after" file can be told apart. The one
 * place that builds a download's file name, for data and for images.
 */
export function downloadFileName(
  name: string,
  extension: "csv" | "png",
  dates?: FileDates,
): string {
  const range = dates?.from && dates.to ? `-${dates.from}-to-${dates.to}` : "";
  return `addaxai-${name}${range}.${extension}`;
}

/** A label as part of a file name: "V. vulpes" becomes "v-vulpes". */
export function fileSlug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

export function saveCsv(name: string, csv: string, dates?: FileDates): void {
  downloadBlob(
    new Blob([csv], { type: "text/csv;charset=utf-8" }),
    downloadFileName(name, "csv", dates),
  );
}

// ---------------------------------------------------------------------------
// Rates: Trend, Species rates and the Map
// ---------------------------------------------------------------------------

/** The filters of a rate view, as text for its columns. */
export interface RateCsvScope {
  /** Empty when there is no date filter and nothing to date. */
  dateFrom: string;
  dateTo: string;
  /** Site names, or "all". */
  sites: string;
  /** Label names, or "all". */
  labels: string;
}

/**
 * The filter columns of a rate view. The dates are the date filter when
 * one is set, else the first and last day with effort when the view
 * knows them, so the file says which days it covers.
 */
export function rateCsvScope(filters: {
  dateFrom?: string;
  dateTo?: string;
  firstDate?: string | null;
  lastDate?: string | null;
  siteIds?: string[];
  siteNames: Record<string, string>;
  labelIds?: string[];
  labelNames: Record<string, string>;
}): RateCsvScope {
  return {
    dateFrom: filters.dateFrom ?? filters.firstDate ?? "",
    dateTo: filters.dateTo ?? filters.lastDate ?? "",
    sites: namesOrAll(filters.siteIds, filters.siteNames),
    labels: namesOrAll(filters.labelIds, filters.labelNames),
  };
}

function rateFilters(scope: RateCsvScope, unit: RateUnit): CsvFilters {
  return [
    ["per_trap_nights", RATE_UNITS[unit].nights],
    ["date_from", scope.dateFrom],
    ["date_to", scope.dateTo],
    ["sites", scope.sites],
    ["labels", scope.labels],
  ];
}

const SUMMARY_HEADERS = ["n_sites", "mean", "sd", "ci_low", "ci_high"];

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

/** Trend: one row per period. */
export function trendPeriodsCsv(
  periods: TrendPeriod[],
  scope: RateCsvScope,
  unit: RateUnit = getRateUnit(),
): string {
  return viewCsv(
    ["period", "period_start", "period_end", ...SUMMARY_HEADERS],
    periods.map((p) => [p.period, p.start, p.end, ...summaryCells(p, unit)]),
    rateFilters(scope, unit),
  );
}

/** Species rates: one row per species. */
export function speciesRatesCsv(
  species: SpeciesRate[],
  scope: RateCsvScope,
  unit: RateUnit = getRateUnit(),
): string {
  return viewCsv(
    ["label", "common_name", "scientific_name", ...SUMMARY_HEADERS],
    species.map((s) => [s.label, s.common_name, s.scientific_name, ...summaryCells(s, unit)]),
    rateFilters(scope, unit),
  );
}

/**
 * Map: one row per site, every site in the filter and not only those in
 * view. `site_first_date` and `site_last_date` are the site's whole
 * monitoring range across its deployments: unlike the trap nights and
 * the observations, they are not cut to the date filter.
 */
export function mapSitesCsv(
  sites: ObservationRateMapFeature[],
  scope: RateCsvScope,
  unit: RateUnit = getRateUnit(),
): string {
  return viewCsv(
    [
      "site", "latitude", "longitude", "deployments",
      "site_first_date", "site_last_date",
      "trap_nights", "observations", "rate",
    ],
    sites.map((s) => [
      s.site_name, s.latitude, s.longitude, s.deployment_count,
      s.earliest_start_local, s.latest_end_local,
      s.trap_nights, s.observation_count, rateFromPer100(s.rate_per_100, unit),
    ]),
    rateFilters(scope, unit),
  );
}

// ---------------------------------------------------------------------------
// Classification performance: Confusion matrix and Class performance
// ---------------------------------------------------------------------------

/**
 * The filter columns of the two performance views. The rank is written
 * as the page labels it: its value "all" means "Most specific", and
 * would read as "every rank" in a file.
 */
export function performanceFilters(filters: {
  siteIds?: string[];
  siteNames: Record<string, string>;
  taxonomicRank: TaxonomicRank;
  topN: string;
}): CsvFilters {
  const rank = RANK_OPTIONS.find((o) => o.value === filters.taxonomicRank)?.label;
  return [
    ["sites", namesOrAll(filters.siteIds, filters.siteNames)],
    ["taxonomic_rank", (rank ?? filters.taxonomicRank).toLowerCase()],
    ["top_n", filters.topN],
  ];
}

/**
 * Confusion matrix in long format: one row per verified and predicted
 * pair with its count, zeros included. Always counts, whatever the view
 * mode, and no totals: both follow from the counts, and a pivot gives
 * the grid back. Classes carry their names from the response, not the
 * common or scientific display name, so the file is the same whoever
 * downloads it.
 */
export function confusionMatrixCsv(data: PerformanceResponse, filters: CsvFilters): string {
  return viewCsv(
    ["verified", "predicted", "count"],
    data.classes.flatMap((verified, i) =>
      data.classes.map((predicted, j) => [verified, predicted, data.matrix[i][j]]),
    ),
    filters,
  );
}

/**
 * Class performance: the table's rows (classes with support), then its
 * two average rows. `row_type` tells them apart, so one filter removes
 * the averages before summing or plotting. Metrics are fractions, not
 * percent text, rounded to 4 decimals (the table shows 3 digits); one
 * that cannot be computed is an empty cell.
 */
export function classPerformanceCsv(data: PerformanceResponse, filters: CsvFilters): string {
  const metric = (value: number | null) => (value === null ? null : +value.toFixed(4));
  const average = (name: string, p: number | null, r: number | null, f1: number | null) =>
    ["average", name, null, null, null, metric(p), metric(r), metric(f1)];
  return viewCsv(
    ["row_type", "class", "common_name", "scientific_name", "support", "precision", "recall", "f1"],
    [
      ...data.per_class
        .filter((c) => c.support > 0)
        .map((c) => [
          "class", c.class_name, c.common_name, c.scientific_name,
          c.support, metric(c.precision), metric(c.recall), metric(c.f1),
        ]),
      average("macro average", data.macro_precision, data.macro_recall, data.macro_f1),
      average("weighted average", data.weighted_precision, data.weighted_recall, data.weighted_f1),
    ],
    filters,
  );
}
