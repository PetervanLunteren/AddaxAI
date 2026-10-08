import { describe, expect, it } from "vitest";
import { effortLevel, weekOfYear, weeklyCameras } from "./effort-calendar";

describe("weekOfYear", () => {
  it("puts 1-7 January in week 0 and 31 December in week 52", () => {
    expect(weekOfYear(Date.UTC(2024, 0, 1))).toBe(0);
    expect(weekOfYear(Date.UTC(2024, 0, 7))).toBe(0);
    expect(weekOfYear(Date.UTC(2024, 0, 8))).toBe(1);
    expect(weekOfYear(Date.UTC(2024, 11, 31))).toBe(52);
  });
});

describe("weeklyCameras", () => {
  it("applies each count until the next change point", () => {
    const rows = weeklyCameras([
      { date: "2024-01-01", count: 2 },
      { date: "2024-01-08", count: 3 },
      { date: "2024-01-15", count: 0 },
    ]);

    expect(rows).toHaveLength(1);
    expect(rows[0].year).toBe(2024);
    expect(rows[0].weeks.slice(0, 3)).toEqual([2, 3, 0]);
  });

  it("takes the busiest day of a week", () => {
    const rows = weeklyCameras([
      { date: "2024-01-01", count: 1 },
      { date: "2024-01-03", count: 4 },
      { date: "2024-01-04", count: 1 },
      { date: "2024-01-08", count: 0 },
    ]);

    expect(rows[0].weeks[0]).toBe(4);
  });

  it("keeps a year without effort as a row of zeros", () => {
    const rows = weeklyCameras([
      { date: "2022-06-01", count: 1 },
      { date: "2022-06-02", count: 0 },
      { date: "2024-06-01", count: 1 },
      { date: "2024-06-02", count: 0 },
    ]);

    expect(rows.map((r) => r.year)).toEqual([2022, 2023, 2024]);
    expect(rows[1].weeks.every((w) => w === 0)).toBe(true);
  });

  it("runs across the new year into the next row", () => {
    const rows = weeklyCameras([
      { date: "2023-12-30", count: 1 },
      { date: "2024-01-03", count: 0 },
    ]);

    expect(rows.map((r) => r.year)).toEqual([2023, 2024]);
    expect(rows[0].weeks[52]).toBe(1);
    expect(rows[1].weeks[0]).toBe(1);
  });

  it("returns nothing without effort", () => {
    expect(weeklyCameras([])).toEqual([]);
    expect(weeklyCameras([{ date: "2024-01-01", count: 0 }])).toEqual([]);
  });
});

describe("effortLevel", () => {
  it("maps a share of the busiest week to 1-4, and nothing to 0", () => {
    expect(effortLevel(0, 10)).toBe(0);
    expect(effortLevel(1, 10)).toBe(1);
    expect(effortLevel(5, 10)).toBe(2);
    expect(effortLevel(10, 10)).toBe(4);
  });
});
