import { describe, expect, it } from "vitest";
import { exploreHref, foldTagsIntoSites } from "./explore-filters";

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
