import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

/**
 * Running app version, served by /health and backed by the repo-root
 * VERSION file (see backend/app/__init__.py). Cached for the session:
 * the value cannot change without a backend restart. `null` until the
 * first answer, so callers can tell "unknown" from a real version.
 */
export function useAppVersion(): string | null {
  const { data } = useQuery({
    queryKey: ["health"],
    queryFn: () => api.get<{ version: string }>("/health"),
    staleTime: Infinity,
  });
  return data?.version ?? null;
}
