import { describe, expect, it } from "vitest";

import { BASE_LAYERS, isBaseLayer } from "./basemap-styles";

describe("BASE_LAYERS", () => {
  it("keeps the stored ids and adds the topographic map", () => {
    expect(BASE_LAYERS.map((layer) => layer.id)).toEqual(["positron", "satellite", "osm", "topo"]);
  });

  it("has unique names: the layers control tells layers apart by name", () => {
    const names = BASE_LAYERS.map((layer) => layer.name);
    expect(new Set(names).size).toBe(names.length);
  });

  it("gives every raster background its tiles and its credit line", () => {
    for (const layer of BASE_LAYERS) {
      if (layer.id === "positron") continue; // the street map component
      expect(layer.url).toMatch(/^https:\/\/.*\{z\}/);
      expect(layer.attribution.length).toBeGreaterThan(0);
    }
  });
});

describe("isBaseLayer", () => {
  it("accepts the known backgrounds and nothing else", () => {
    expect(["positron", "satellite", "osm", "topo"].every(isBaseLayer)).toBe(true);
    expect(isBaseLayer("carto")).toBe(false);
    expect(isBaseLayer(null)).toBe(false);
  });
});
