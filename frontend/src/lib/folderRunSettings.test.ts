import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  loadLastUsedSaveOutputs,
  loadLastUsedSettings,
  saveLastUsedSaveOutputs,
  saveLastUsedSettings,
} from "./folderRunSettings";

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
        detection_augment: true,
      }),
    );

    expect(loadLastUsedSettings()).toEqual({
      classification_model_id: "SPECIESNET-v4-0-2-A",
      detection_augment: true,
    });
  });

  it("drops the frame rate of a record saved before version 3, and nothing else", () => {
    localStorage.setItem(
      KEY,
      JSON.stringify({
        version: 2,
        country_code: "AUS",
        excluded_classes: ["capybara"],
        counting_threshold: 0.2,
        video_fps: 1,
      }),
    );

    expect(loadLastUsedSettings()).toEqual({
      country_code: "AUS",
      excluded_classes: ["capybara"],
      counting_threshold: 0.2,
    });
  });

  it("keeps a frame rate chosen since, also across a project dialog's subset write", () => {
    saveLastUsedSettings({ video_fps: 1 });
    saveLastUsedSettings({ classification_model_id: "SPECIESNET-v4-0-2-A" });

    expect(loadLastUsedSettings()).toEqual({
      classification_model_id: "SPECIESNET-v4-0-2-A",
      video_fps: 1,
    });
  });

  it("does not hand an old frame rate back through a subset write", () => {
    localStorage.setItem(KEY, JSON.stringify({ version: 2, video_fps: 1 }));
    saveLastUsedSettings({ classification_model_id: "SPECIESNET-v4-0-2-A" });

    expect(loadLastUsedSettings()).toEqual({
      classification_model_id: "SPECIESNET-v4-0-2-A",
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

describe("choices saved before events became sequences", () => {
  it("keeps smoothing off", () => {
    localStorage.setItem(KEY, JSON.stringify({ event_smoothing: false, version: 3 }));
    expect(loadLastUsedSettings()).toEqual({ sequence_smoothing: false });
  });

  it("keeps the files of a sequence apart", () => {
    const SAVE_KEY = "addaxai.folderRun.lastSaveOutputs";
    localStorage.setItem(SAVE_KEY, JSON.stringify({ groupEvents: false, blur: true }));
    expect(loadLastUsedSaveOutputs()).toEqual({ groupSequences: false, blur: true });
  });

  it("lets a newer choice win and stops writing the old name", () => {
    localStorage.setItem(KEY, JSON.stringify({ event_smoothing: false, version: 3 }));
    saveLastUsedSettings({ sequence_smoothing: true });
    expect(loadLastUsedSettings()).toEqual({ sequence_smoothing: true });
    expect(localStorage.getItem(KEY)).not.toContain("event_smoothing");

    const SAVE_KEY = "addaxai.folderRun.lastSaveOutputs";
    localStorage.setItem(SAVE_KEY, JSON.stringify({ groupEvents: false }));
    saveLastUsedSaveOutputs({ groupSequences: true });
    expect(loadLastUsedSaveOutputs()).toEqual({ groupSequences: true });
    expect(localStorage.getItem(SAVE_KEY)).not.toContain("groupEvents");
  });
});
