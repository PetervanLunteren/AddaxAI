import { describe, expect, it } from "vitest";
import { advancedNonDefaultKeys } from "./advancedSettingsDefaults";

describe("advancedNonDefaultKeys", () => {
  const changed = { sequence_smoothing: false, video_fps: 1 };

  it("counts a classifier-only setting when a classifier is chosen", () => {
    expect(
      advancedNonDefaultKeys({ ...changed, classification_model_id: "EUR-DF-v1-5" }),
    ).toEqual(["video_fps", "sequence_smoothing"]);
  });

  it("leaves classifier-only settings out without a classifier", () => {
    // Their rows are hidden then, so the count would name nothing visible.
    for (const none of ["none", null, undefined]) {
      expect(
        advancedNonDefaultKeys({ ...changed, classification_model_id: none }),
      ).toEqual(["video_fps"]);
    }
  });
});
