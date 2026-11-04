import { describe, expect, it } from "vitest";
import { exceedsChartYears, yearSpan } from "./date-span";

describe("yearSpan", () => {
  it("reads the years of zero-padded dates", () => {
    expect(yearSpan("0025-01-01", "4500-01-01")).toEqual({ first: 25, last: 4500 });
    expect(yearSpan("2024-03-01", "2024-03-02")).toEqual({ first: 2024, last: 2024 });
  });
});

describe("exceedsChartYears", () => {
  it("allows 30 years and refuses 31", () => {
    expect(exceedsChartYears({ first: 1995, last: 2024 })).toBe(false);
    expect(exceedsChartYears({ first: 1994, last: 2024 })).toBe(true);
  });

  it("allows a single year", () => {
    expect(exceedsChartYears({ first: 2024, last: 2024 })).toBe(false);
  });
});
