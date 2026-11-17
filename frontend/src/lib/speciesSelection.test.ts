import { describe, expect, it } from "vitest";
import { clearSpeciesSelection, mappedLabels } from "./speciesSelection";

describe("clearSpeciesSelection", () => {
  it("clears country, state, exclusions and the taxa list together, as dirty edits", () => {
    const calls: unknown[][] = [];
    const setValue = (...args: unknown[]) => {
      calls.push(args);
    };

    clearSpeciesSelection(setValue as never);

    expect(calls).toEqual([
      ["country_code", null, { shouldDirty: true }],
      ["state_code", null, { shouldDirty: true }],
      ["excluded_classes", [], { shouldDirty: true }],
      ["taxa_mapping", null, { shouldDirty: true }],
    ]);
  });
});

describe("mappedLabels", () => {
  it("lists each label of a taxa list once, in file order", () => {
    expect(
      mappedLabels({
        file_name: "mara.csv",
        rows: [
          { input_latin: "accipitridae", user_taxon_name: "bird of prey" },
          { input_latin: "aepyceros", user_taxon_name: "impala" },
          { input_latin: "strigiformes", user_taxon_name: "bird of prey" },
        ],
      }),
    ).toEqual(["bird of prey", "impala"]);
  });
});
