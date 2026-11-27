import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";

// The real client starts the browser logger, which needs a window.
vi.mock("./api-client", () => ({ api: {} }));

import { projectQueryOptions } from "../api/projects";
import { invalidateProjectData } from "./invalidate-project";

describe("invalidateProjectData", () => {
  // The Labels viewers draw boxes from the project's threshold. When this
  // refresh missed their key, they kept the old threshold for minutes
  // after a change (audit 1, 2026-11-21).
  it("refreshes the project row the readers use", () => {
    const client = new QueryClient();
    const { queryKey } = projectQueryOptions("p1");
    client.setQueryData(queryKey, { id: "p1" } as never);

    invalidateProjectData(client, "p1");

    expect(client.getQueryState(queryKey)?.isInvalidated).toBe(true);
  });

  // A new independence interval changes every count. The map and the info
  // sheets kept the old numbers for minutes when this refresh missed them.
  it("refreshes the map and the site and deployment info sheets", () => {
    const client = new QueryClient();
    const keys = [
      ["statistics", "summary", "p1"],
      ["observation-rate-map", "p1"],
      ["sites", "site-1", "info"],
      ["deployments", "dep-1", "info"],
    ];
    for (const key of keys) client.setQueryData(key, {});
    client.setQueryData(["sites", "site-1", "photo"], {});

    invalidateProjectData(client, "p1");

    for (const key of keys) {
      expect(client.getQueryState(key)?.isInvalidated, key.join("/")).toBe(true);
    }
    expect(client.getQueryState(["sites", "site-1", "photo"])?.isInvalidated).toBe(false);
  });
});
