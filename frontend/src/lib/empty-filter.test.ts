import { describe, expect, it } from "vitest";
import { reconcileEmptyFilter } from "./empty-filter";

describe("reconcileEmptyFilter", () => {
  it("clears the box filters when Empty only is picked", () => {
    const prev = {
      labels: ["fox"],
      min_confidence: 0.4,
      max_confidence: 0.9,
      min_label_confidence: 0.5,
      max_label_confidence: 0.8,
      site_ids: ["s1"],
    };
    expect(reconcileEmptyFilter(prev, { ...prev, empty: "show_only" })).toEqual({
      empty: "show_only",
      site_ids: ["s1"],
      labels: undefined,
      min_confidence: undefined,
      max_confidence: undefined,
      min_label_confidence: undefined,
      max_label_confidence: undefined,
    });
  });

  it("drops Empty only when a label is chosen", () => {
    const prev = { empty: "show_only" as const };
    expect(reconcileEmptyFilter(prev, { ...prev, labels: ["fox"] })).toEqual({
      labels: ["fox"],
      empty: undefined,
    });
  });

  it("drops Empty only when a confidence range is set", () => {
    const prev = { empty: "show_only" as const };
    expect(
      reconcileEmptyFilter(prev, { ...prev, min_label_confidence: 0.5 }),
    ).toEqual({ min_label_confidence: 0.5, empty: undefined });
  });

  it("keeps Empty only when an unrelated filter changes", () => {
    const prev = { empty: "show_only" as const };
    const next = { ...prev, site_ids: ["s1"], verification: "verified" as const };
    expect(reconcileEmptyFilter(prev, next)).toBe(next);
  });

  it("keeps Empty only when a box filter is cleared", () => {
    const prev = { empty: "show_only" as const, labels: [] };
    const next = { empty: "show_only" as const, labels: undefined };
    expect(reconcileEmptyFilter(prev, next)).toBe(next);
  });

  it("leaves Hide empty combinable with box filters", () => {
    const prev = { empty: "hide" as const };
    const next = { ...prev, labels: ["fox"], min_confidence: 0.5 };
    expect(reconcileEmptyFilter(prev, next)).toBe(next);
  });
});
