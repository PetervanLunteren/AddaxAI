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
});
