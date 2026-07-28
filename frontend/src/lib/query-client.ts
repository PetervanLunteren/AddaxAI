/**
 * TanStack Query client configuration.
 *
 * Following DEVELOPERS.md principles:
 * - Explicit configuration
 * - Crash early on errors
 */

import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 30 * 60 * 1000, // 30 minutes (formerly cacheTime)
      retry: 1, // Only retry once
      refetchOnWindowFocus: true,
      // Throw errors to surface them (crash early principle)
      throwOnError: false, // Let components handle errors
    },
    mutations: {
      // Throw errors to surface them
      throwOnError: false,
    },
  },
});
