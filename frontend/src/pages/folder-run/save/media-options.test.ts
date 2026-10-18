import { describe, expect, it } from "vitest";

import {
  excludedLabelIds,
  mediaRequestFields,
  nothingInScope,
  type SeparateState,
} from "./media-options";

const ALL = ["a", "b", "c"];

function separate(overrides: Partial<SeparateState> = {}): SeparateState {
  return {
    enabled: true,
    groupBy: "flat",
    groupEvents: true,
    speciesLast: false,
    copyEmpties: false,
    mediaThreshold: 0.2,
    includedLabelIds: [],
    likedOnly: false,
    verifiedOnly: false,
    ...overrides,
  };
}

describe("excludedLabelIds", () => {
  it("sends the complement of a real subset", () => {
    expect(
      excludedLabelIds(separate({ includedLabelIds: ["b"] }), ALL),
    ).toEqual(["a", "c"]);
  });

  it("sends no filter when everything is picked", () => {
    expect(excludedLabelIds(separate({ includedLabelIds: ALL }), ALL)).toEqual([]);
  });

  it("reads an empty selection as every label", () => {
    expect(excludedLabelIds(separate(), ALL)).toEqual([]);
  });

  it("sends no filter before the label tree has loaded", () => {
    expect(excludedLabelIds(separate({ includedLabelIds: ["a"] }), [])).toEqual(
      [],
    );
  });
});

describe("nothingInScope", () => {
  it("is false with nothing picked: empty labels mean all", () => {
    expect(nothingInScope(separate())).toBe(false);
  });

  it("is true when the deployment picker has nothing ticked", () => {
    expect(nothingInScope(separate({ deploymentIds: [] }))).toBe(true);
    expect(
      nothingInScope(separate({ deploymentIds: [], includedLabelIds: ["a"] })),
    ).toBe(true);
  });

  it("treats an absent deployment list as every deployment", () => {
    expect(nothingInScope(separate({ deploymentIds: undefined }))).toBe(false);
  });
});

describe("mediaRequestFields", () => {
  const on = { enabled: true };
  const off = { enabled: false };

  it("passes the file filters and the deployments through unchanged", () => {
    const fields = mediaRequestFields(
      separate({
        likedOnly: true,
        verifiedOnly: true,
        dateFrom: "2025-01-01",
        dateTo: "2025-01-31",
        deploymentIds: ["d1"],
        includedLabelIds: ["a"],
      }),
      on,
      off,
      ALL,
      "scientific",
    );
    expect(fields).toEqual({
      media_threshold: 0.2,
      separate_group_by: "flat",
      group_events: true,
      separate_species_last: false,
      include_empty: false,
      excluded_label_ids: ["b", "c"],
      draw_bboxes: true,
      anonymise: false,
      name_mode: "scientific",
      favorited_only: true,
      verified_only: true,
      date_from: "2025-01-01",
      date_to: "2025-01-31",
      deployment_ids: ["d1"],
    });
  });

  it("omits the deployment list when every deployment is in", () => {
    const fields = mediaRequestFields(separate(), off, off, ALL, "common");
    expect(fields.deployment_ids).toBeUndefined();
    expect(fields.date_from).toBeUndefined();
    expect(fields.favorited_only).toBe(false);
    expect(fields.excluded_label_ids).toEqual([]);
  });

  it("drops boxes, blur, empties and the label filter when the media card is off", () => {
    const fields = mediaRequestFields(
      separate({ enabled: false, copyEmpties: true, includedLabelIds: ["a"] }),
      on,
      on,
      ALL,
      "common",
    );
    expect(fields.draw_bboxes).toBe(false);
    expect(fields.anonymise).toBe(false);
    expect(fields.include_empty).toBe(false);
    expect(fields.excluded_label_ids).toEqual([]);
  });
});
