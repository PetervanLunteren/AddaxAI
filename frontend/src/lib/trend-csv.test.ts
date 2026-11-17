/**
 * A downloaded file must carry the numbers the card shows, in the unit
 * the user chose, and say which unit and filters that was.
 */

import { describe, expect, it } from "vitest";
import { speciesRatesCsv, trendPeriodsCsv } from "./trend-csv";

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
