import { beforeEach, describe, expect, it, vi } from "vitest";

import { readLabelsSettings } from "./labels-settings";

beforeEach(() => {
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
  });
});

describe("readLabelsSettings", () => {
  it("reads a sort saved as events as the sort by sequence", () => {
    localStorage.setItem(
      "addaxai:labelsSettings",
      JSON.stringify({ sort: "events", filesSort: "events", tileSize: "L" }),
    );
    expect(readLabelsSettings()).toEqual({
      sort: "sequences",
      filesSort: "sequences",
      tileSize: "L",
    });
  });
});
