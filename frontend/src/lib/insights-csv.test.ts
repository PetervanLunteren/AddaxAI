/**
 * A downloaded file must carry the numbers the page shows, in the unit
 * the user chose, and say which unit and filters that was. The Trend and
 * Species rates cases are older than the shared module and pin that its
 * arrival did not change those two files.
 */

import { describe, expect, it } from "vitest";
import type { PerformanceResponse } from "../api/performance";
import type { ObservationRateMapFeature } from "../api/statistics";
import {
  classPerformanceCsv,
  confusionMatrixCsv,
  downloadFileName,
  fileSlug,
  mapSitesCsv,
  performanceFilters,
  rateCsvScope,
  speciesRatesCsv,
  trendPeriodsCsv,
  viewCsv,
} from "./insights-csv";

const scope = {
  dateFrom: "2024-01-01",
  dateTo: "2024-01-14",
  sites: "Site A; Site B",
  labels: "all",
};

describe("trend downloads", () => {
  it("writes a period per row in the chosen unit, with the filters", () => {
    const csv = trendPeriodsCsv(
      [
        {
          period: "2024-W01", start: "2024-01-01", end: "2024-01-07",
          n_sites: 3, mean: 100, sd: 100, ci_low: 0, ci_high: 348.4338,
        },
        {
          period: "2024-W02", start: "2024-01-08", end: "2024-01-14",
          n_sites: 0, mean: null, sd: null, ci_low: null, ci_high: null,
        },
      ],
      scope,
      "week",
    );

    expect(csv.split("\r\n")).toEqual([
      "period,period_start,period_end,n_sites,mean,sd,ci_low,ci_high,per_trap_nights,date_from,date_to,sites,labels",
      "2024-W01,2024-01-01,2024-01-07,3,7,7,0,24.39,7,2024-01-01,2024-01-14,Site A; Site B,all",
      "2024-W02,2024-01-08,2024-01-14,0,,,,,7,2024-01-01,2024-01-14,Site A; Site B,all",
      "",
    ]);
  });

  it("writes a species per row and quotes a name with a comma", () => {
    const csv = speciesRatesCsv(
      [
        {
          label: "deer", label_taxonomy_id: "t1", common_name: "Deer, red",
          scientific_name: "Cervus elaphus",
          n_sites: 1, mean: 50, sd: null, ci_low: null, ci_high: null,
        },
      ],
      scope,
      "hundred",
    );

    expect(csv.split("\r\n")[1]).toBe(
      'deer,"Deer, red",Cervus elaphus,1,50,,,,100,2024-01-01,2024-01-14,Site A; Site B,all',
    );
  });
});

describe("the filters of a download", () => {
  const names = {
    siteNames: { a: "Site A", b: "Site B" },
    labelNames: { t1: "Red deer" },
  };

  it("names the picked sites and labels", () => {
    expect(
      rateCsvScope({
        ...names,
        dateFrom: "2024-01-01",
        dateTo: "2024-01-14",
        siteIds: ["a", "b"],
        labelIds: ["t1", "t2"],
      }),
    ).toEqual({
      dateFrom: "2024-01-01",
      dateTo: "2024-01-14",
      sites: "Site A; Site B",
      // A label without a known name keeps its id, never an empty cell.
      labels: "Red deer; t2",
    });
  });

  it("says all without a filter and falls back to the days with effort", () => {
    expect(
      rateCsvScope({ ...names, firstDate: "2023-05-02", lastDate: "2023-09-30", siteIds: [] }),
    ).toEqual({ dateFrom: "2023-05-02", dateTo: "2023-09-30", sites: "all", labels: "all" });
  });

  it("leaves the dates empty when there is nothing to date", () => {
    expect(rateCsvScope({ ...names, firstDate: null, lastDate: null })).toMatchObject({
      dateFrom: "",
      dateTo: "",
    });
  });
});

describe("every view", () => {
  it("repeats the filters on every row", () => {
    expect(
      viewCsv(["a"], [[1], [2]], [["sites", "all"], ["top_n", "20"]]).split("\r\n"),
    ).toEqual(["a,sites,top_n", "1,all,20", "2,all,20", ""]);
  });

  it("puts the dates in the file name only when there are both", () => {
    expect(downloadFileName("map-sites", "csv")).toBe("addaxai-map-sites.csv");
    expect(downloadFileName("map-sites", "csv", { from: "", to: "" })).toBe(
      "addaxai-map-sites.csv",
    );
    // Half a date range is no range: the name would claim days it cannot.
    expect(downloadFileName("activity-overlap", "png", { from: "2024-01-01", to: "" })).toBe(
      "addaxai-activity-overlap.png",
    );
    // A species in a file name has no spaces or dots.
    expect(fileSlug("V. vulpes")).toBe("v-vulpes");
    expect(fileSlug("Odocoileus virginianus (deer)")).toBe("odocoileus-virginianus-deer");
    // Data and image of one view differ in the extension only.
    const dates = { from: "2024-01-01", to: "2024-03-31" };
    expect(downloadFileName("trend-week", "csv", dates)).toBe(
      "addaxai-trend-week-2024-01-01-to-2024-03-31.csv",
    );
    expect(downloadFileName("trend-week", "png", dates)).toBe(
      "addaxai-trend-week-2024-01-01-to-2024-03-31.png",
    );
  });
});

describe("map download", () => {
  const site = (over: Partial<ObservationRateMapFeature>): ObservationRateMapFeature => ({
    site_id: "s1",
    site_name: "Wolf Hill",
    latitude: -43.3,
    longitude: 172.6,
    deployment_count: 2,
    earliest_start_local: "2024-01-01",
    latest_end_local: "2024-03-31",
    trap_nights: 28,
    observation_count: 14,
    rate_per_100: 50,
    species_breakdown: [],
    ...over,
  });

  it("writes a site per row with its rate in the chosen unit", () => {
    const lines = mapSitesCsv(
      [site({}), site({ site_name: "Pine Cove, north", latest_end_local: null, rate_per_100: 0 })],
      { dateFrom: "", dateTo: "", sites: "all", labels: "all" },
      "week",
    ).split("\r\n");

    expect(lines[0]).toBe(
      "site,latitude,longitude,deployments,site_first_date,site_last_date," +
        "trap_nights,observations,rate,per_trap_nights,date_from,date_to,sites,labels",
    );
    // 50 per 100 trap nights is 3.5 per 7.
    expect(lines[1]).toBe("Wolf Hill,-43.3,172.6,2,2024-01-01,2024-03-31,28,14,3.5,7,,,all,all");
    // A site still running has no last date; a name with a comma is quoted.
    expect(lines[2]).toBe(
      '"Pine Cove, north",-43.3,172.6,2,2024-01-01,,28,14,0,7,,,all,all',
    );
  });
});

describe("classification performance downloads", () => {
  const data = {
    classes: ["deer", "pig"],
    matrix: [
      [8, 2],
      [0, 5],
    ],
    grand_total: 15,
    per_class: [
      { class_name: "deer", common_name: "Red deer", scientific_name: "Cervus elaphus",
        support: 10, precision: 1, recall: 0.8, f1: 0.888889 },
      { class_name: "pig", common_name: "Feral pig", scientific_name: "Sus scrofa",
        support: 5, precision: 0.714286, recall: 1, f1: 0.833333 },
      { class_name: "goat", common_name: "Goat", scientific_name: "Capra hircus",
        support: 0, precision: null, recall: null, f1: null },
    ],
    macro_precision: 0.857143,
    macro_recall: 0.9,
    macro_f1: 0.861111,
    weighted_precision: 0.904762,
    weighted_recall: 0.866667,
    weighted_f1: null,
  } as unknown as PerformanceResponse;
  const filters = performanceFilters({
    siteIds: ["a"],
    siteNames: { a: "Site A" },
    taxonomicRank: "species",
    topN: "20",
  });

  it("writes the rank as the page labels it", () => {
    const rank = performanceFilters({ siteNames: {}, taxonomicRank: "all", topN: "all" });

    // "all" is "Most specific", not every rank.
    expect(rank).toEqual([
      ["sites", "all"],
      ["taxonomic_rank", "most specific"],
      ["top_n", "all"],
    ]);
  });

  it("writes the confusion matrix as one row per pair, zeros included", () => {
    const lines = confusionMatrixCsv(data, filters).split("\r\n");

    expect(lines).toEqual([
      "verified,predicted,count,sites,taxonomic_rank,top_n",
      "deer,deer,8,Site A,species,20",
      "deer,pig,2,Site A,species,20",
      "pig,deer,0,Site A,species,20",
      "pig,pig,5,Site A,species,20",
      "",
    ]);
    // The counts add up to everything the matrix covers.
    const total = lines.slice(1, -1).reduce((sum, line) => sum + Number(line.split(",")[2]), 0);
    expect(total).toBe(data.grand_total);
  });

  it("writes the classes with support, then the averages marked as such", () => {
    expect(classPerformanceCsv(data, filters).split("\r\n")).toEqual([
      "row_type,class,common_name,scientific_name,support,precision,recall,f1," +
        "sites,taxonomic_rank,top_n",
      // Metrics rounded to 4 decimals: 0.888889 and 0.714286 in the response.
      "class,deer,Red deer,Cervus elaphus,10,1,0.8,0.8889,Site A,species,20",
      "class,pig,Feral pig,Sus scrofa,5,0.7143,1,0.8333,Site A,species,20",
      // Goat had no verified detections: not in the table, not in the file.
      "average,macro average,,,,0.8571,0.9,0.8611,Site A,species,20",
      // A metric that cannot be computed is an empty cell.
      "average,weighted average,,,,0.9048,0.8667,,Site A,species,20",
      "",
    ]);
  });
});
