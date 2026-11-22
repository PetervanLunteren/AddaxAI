import { QueryClient, QueryObserver } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VERIFY_SETTLE_MS, invalidateAfterVerify } from "./invalidate-label-queries";

const PROJECT = "p1";

/** A client with one mounted query per key, each counting its fetches. */
async function clientWith(keys: unknown[][]) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  const fetches = new Map<string, number>();
  for (const key of keys) {
    const name = JSON.stringify(key);
    fetches.set(name, 0);
    // A subscribed observer makes the query active, as a mounted
    // component does, so an invalidation refetches it.
    new QueryObserver(client, {
      queryKey: key,
      queryFn: async () => {
        fetches.set(name, (fetches.get(name) ?? 0) + 1);
        return null;
      },
    }).subscribe(() => {});
  }
  await vi.advanceTimersByTimeAsync(0);
  return { client, fetches };
}

describe("invalidateAfterVerify", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const keys = [
    ["labels-progress", PROJECT],
    ["cohorts", PROJECT],
    ["label-tree", PROJECT, "detection", {}],
    ["label-colors", PROJECT],
    ["statistics", PROJECT],
    ["event-filter-options", PROJECT],
    ["events", PROJECT],
  ];

  const stale = (client: QueryClient, key: unknown[]) =>
    client.getQueryCache().find({ queryKey: key })!.state.isInvalidated;
  const fetched = (fetches: Map<string, number>, key: unknown[]) =>
    fetches.get(JSON.stringify(key));

  it("refetches what a verify moves once, after the clicks stop", async () => {
    const { client, fetches } = await clientWith(keys);
    invalidateAfterVerify(client, PROJECT, true);
    await vi.advanceTimersByTimeAsync(VERIFY_SETTLE_MS - 1);
    invalidateAfterVerify(client, PROJECT, true);
    await vi.advanceTimersByTimeAsync(VERIFY_SETTLE_MS - 1);
    // Still inside the second click's window: only the first load.
    expect(fetched(fetches, ["labels-progress", PROJECT])).toBe(1);

    await vi.advanceTimersByTimeAsync(1);
    expect(fetched(fetches, ["labels-progress", PROJECT])).toBe(2);
    expect(fetched(fetches, ["cohorts", PROJECT])).toBe(2);
    expect(fetched(fetches, ["label-tree", PROJECT, "detection", {}])).toBe(2);
  });

  it("leaves the species queries alone: a verify adds or removes no species", async () => {
    const { client, fetches } = await clientWith(keys);
    invalidateAfterVerify(client, PROJECT, true);
    await vi.advanceTimersByTimeAsync(VERIFY_SETTLE_MS);
    expect(fetched(fetches, ["label-colors", PROJECT])).toBe(1);
    expect(fetched(fetches, ["statistics", PROJECT])).toBe(1);
  });

  it("keeps the label tree when the Verified filter is off", async () => {
    const { client, fetches } = await clientWith(keys);
    invalidateAfterVerify(client, PROJECT, false);
    await vi.advanceTimersByTimeAsync(VERIFY_SETTLE_MS);
    expect(fetched(fetches, ["label-tree", PROJECT, "detection", {}])).toBe(1);
    expect(fetched(fetches, ["labels-progress", PROJECT])).toBe(2);
  });

  it("marks the event queries stale without refetching them", async () => {
    const { client, fetches } = await clientWith(keys);
    invalidateAfterVerify(client, PROJECT, true);
    await vi.advanceTimersByTimeAsync(VERIFY_SETTLE_MS);
    expect(stale(client, ["event-filter-options", PROJECT])).toBe(true);
    expect(stale(client, ["events", PROJECT])).toBe(true);
    expect(fetched(fetches, ["event-filter-options", PROJECT])).toBe(1);
    expect(fetched(fetches, ["events", PROJECT])).toBe(1);
  });
});
