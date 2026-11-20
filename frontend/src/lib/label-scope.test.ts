import { describe, expect, it } from "vitest";
import { scopeFilters, scopeText } from "./label-scope";

const NAMES = { a: "Kenai North", b: "Kenai South" };
const text = (scope: Parameters<typeof scopeText>[0], showSites = true) =>
  scopeText(scope, NAMES, showSites, "en-GB");

describe("scopeFilters", () => {
  it("keeps sites and dates and leaves the selection behind", () => {
    const filters = {
      site_ids: ["a"],
      date_from: "2026-06-01",
      date_to: "2026-08-31",
      labels: ["canis"],
      verified: false,
      min_confidence: 0.4,
      max_confidence: 0.9,
      min_label_confidence: 0.5,
      max_label_confidence: 0.8,
      flagged: "flagged" as const,
      favorited: "favorited" as const,
    };
    expect(scopeFilters(filters)).toEqual({
      site_ids: ["a"],
      date_from: "2026-06-01",
      date_to: "2026-08-31",
    });
  });
});

describe("scopeText", () => {
  it("says everything when nothing is set", () => {
    expect(text({})).toBe("in all sites, all dates");
    expect(text({ site_ids: [] })).toBe("in all sites, all dates");
  });

  it("names one site and counts several", () => {
    expect(text({ site_ids: ["a"] })).toBe("in Kenai North, all dates");
    expect(text({ site_ids: ["a", "b"] })).toBe("in 2 sites, all dates");
    expect(text({ site_ids: ["a", "b", "null"] })).toBe("in 3 sites, all dates");
  });

  it("names the no-site option as the filter does", () => {
    expect(text({ site_ids: ["null"] })).toBe("in (no site), all dates");
  });

  it("counts a site whose name has not loaded", () => {
    expect(text({ site_ids: ["unknown"] })).toBe("in 1 site, all dates");
  });

  it("names the dates that are set", () => {
    expect(text({ date_from: "2026-06-01", date_to: "2026-08-31" })).toBe(
      "in all sites, 1 Jun 2026 to 31 Aug 2026",
    );
    expect(text({ date_from: "2026-06-01" })).toBe("in all sites, from 1 Jun 2026");
    expect(text({ date_to: "2026-08-31" })).toBe("in all sites, until 31 Aug 2026");
  });

  it("names sites and dates together", () => {
    expect(
      text({ site_ids: ["a", "b"], date_from: "2026-06-01", date_to: "2026-08-31" }),
    ).toBe("in 2 sites, 1 Jun 2026 to 31 Aug 2026");
  });

  it("leaves the sites out for a folder run, which has none", () => {
    expect(text({}, false)).toBe("all dates");
    expect(text({ date_from: "2026-06-01" }, false)).toBe("from 1 Jun 2026");
  });
});
