import { beforeEach, describe, expect, it, vi } from "vitest";
import { loadLastUsedSettings, saveLastUsedSettings } from "./folderRunSettings";

const KEY = "addaxai.folderRun.lastSettings";

beforeEach(() => {
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
  });
});

describe("loadLastUsedSettings", () => {
  it("drops the species selection of a record saved before version 2", () => {
    localStorage.setItem(
      KEY,
      JSON.stringify({
        classification_model_id: "SPECIESNET-v4-0-2-A",
        country_code: "AUS",
        state_code: null,
        excluded_classes: ["capybara"],
        video_fps: 2,
      }),
    );

    expect(loadLastUsedSettings()).toEqual({
      classification_model_id: "SPECIESNET-v4-0-2-A",
      video_fps: 2,
    });
  });

  it("restores a record it saved itself whole, without the version key", () => {
    saveLastUsedSettings({
      classification_model_id: "SPECIESNET-v4-0-2-A",
      country_code: "AUS",
      excluded_classes: ["capybara"],
    });

    expect(loadLastUsedSettings()).toEqual({
      classification_model_id: "SPECIESNET-v4-0-2-A",
      country_code: "AUS",
      excluded_classes: ["capybara"],
    });
  });
});
