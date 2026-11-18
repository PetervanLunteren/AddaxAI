/**
 * Shared options builder for site multi-select filters.
 *
 * The reserved NO_SITE_SENTINEL option is prepended only when the
 * project actually has deployments with no assigned site — so users
 * never see a filter they cannot use. The backend's site_ids_filter()
 * already translates NO_SITE_SENTINEL to `Deployment.site_id IS NULL`,
 * so selecting it works end-to-end on every events / statistics
 * endpoint without any additional wiring.
 */

import { NO_SITE_SENTINEL } from "./filter-url";

interface SiteLike {
  id: string;
  name: string;
}

export interface SiteFilterOption {
  value: string;
  label: string;
}

/** What the reserved "deployments without a site" option is called. */
export const NO_SITE_LABEL = "(no site)";

export function buildSiteOptions(
  sites: SiteLike[] | undefined,
  noSiteCount: number,
): SiteFilterOption[] {
  const base = (sites ?? []).map((s) => ({ value: s.id, label: s.name }));
  if (noSiteCount > 0) {
    return [{ value: NO_SITE_SENTINEL, label: NO_SITE_LABEL }, ...base];
  }
  return base;
}
