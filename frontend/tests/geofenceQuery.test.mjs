import assert from "node:assert/strict";
import test from "node:test";
import { shouldFetchModelGeofence } from "../src/components/taxonomy/geofenceQuery.ts";

test("does not request geofence rules for a detector-class alias", () => {
  assert.equal(shouldFetchModelGeofence("custom-detector", true), false);
});

test("requests geofence rules for a normal selected classifier", () => {
  assert.equal(shouldFetchModelGeofence("speciesnet", false), true);
});

test("does not request geofence rules before a model is selected", () => {
  assert.equal(shouldFetchModelGeofence(undefined, false), false);
  assert.equal(shouldFetchModelGeofence("", false), false);
});
