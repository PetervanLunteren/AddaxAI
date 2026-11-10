import { describe, expect, it } from "vitest";
import {
  activityOverlapHref,
  exploreHref,
  foldTagsIntoSites,
  insightsMapHref,
} from "./explore-filters";

describe("foldTagsIntoSites", () => {
  const sites = [
    { id: "s1", tags: { habitat: "forest" } },
    { id: "s2", tags: { habitat: "grass" } },
    { id: "s3", tags: null },
  ];

  it("means no site filter when neither is set", () => {
    expect(foldTagsIntoSites(sites, undefined, undefined)).toBeUndefined();
    expect(foldTagsIntoSites(sites, [], [])).toBeUndefined();
  });

  it("passes chosen sites through when no tags are set", () => {
    expect(foldTagsIntoSites(sites, ["s3"], undefined)).toEqual(["s3"]);
  });

  it("turns tags into the sites that carry them", () => {
    expect(foldTagsIntoSites(sites, undefined, ["habitat:forest"])).toEqual(["s1"]);
  });

  it("keeps only chosen sites that also carry a chosen tag", () => {
    expect(foldTagsIntoSites(sites, ["s1", "s2"], ["habitat:grass"])).toEqual(["s2"]);
  });

  it("gives an empty list, never 'all sites', when nothing matches", () => {
    expect(foldTagsIntoSites(sites, ["s3"], ["habitat:forest"])).toEqual([]);
    expect(foldTagsIntoSites(sites, undefined, ["habitat:desert"])).toEqual([]);
  });
});

describe("exploreHref", () => {
  it("puts the label ids in the same parameter the Map page uses", () => {
    expect(exploreHref("p1", ["lion-a", "lion-b"])).toBe(
      "/projects/p1/dashboard/explore?labels=lion-a%2Clion-b",
    );
  });
});

describe("insights links", () => {
  const scope = {
    labelTaxonomyIds: ["lion-a"],
    siteIds: ["s1", "s2"],
    dateFrom: "2026-01-01",
    dateTo: "2026-06-30",
  };

  it("takes labels, sites and dates to the map", () => {
    const params = new URLSearchParams(insightsMapHref("p1", scope).split("?")[1]);
    expect(insightsMapHref("p1", scope)).toMatch(/^\/projects\/p1\/insights\/map\?/);
    expect(params.get("labels")).toBe("lion-a");
    expect(params.get("site_ids")).toBe("s1,s2");
    expect(params.get("date_from")).toBe("2026-01-01");
    expect(params.get("date_to")).toBe("2026-06-30");
  });

  it("takes sites and dates to activity overlap, never the labels", () => {
    const href = activityOverlapHref("p1", scope);
    const params = new URLSearchParams(href.split("?")[1]);
    expect(href).toMatch(/^\/projects\/p1\/insights\/activity-overlap\?/);
    expect(params.has("labels")).toBe(false);
    expect(params.get("site_ids")).toBe("s1,s2");
    expect(params.get("date_from")).toBe("2026-01-01");
    expect(params.get("date_to")).toBe("2026-06-30");
  });

  it("leaves out what the scope does not set", () => {
    expect(activityOverlapHref("p1", {})).toBe("/projects/p1/insights/activity-overlap?");
  });
});
