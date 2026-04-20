/**
 * URL search-params helpers for filter state.
 *
 * Lets pages persist filter state in the URL via react-router-dom's
 * useSearchParams. Multi-value fields are encoded as comma-separated
 * lists (e.g. ?site_ids=a,b,c), same convention as the Verify page.
 *
 * Site filter sentinel: the string "null" inside a `site_ids` list
 * is a reserved token meaning "deployments whose site_id IS NULL"
 * (deployment-agnostic batches). The backend expands it to a SQL
 * IS NULL clause via `app.api.crud.deployment.site_ids_filter`.
 * Import `NO_SITE_SENTINEL` rather than typing the literal.
 */

export const NO_SITE_SENTINEL = "null" as const;

export type FilterFieldKind = "string" | "string[]" | "date";
export type FilterSchema = Record<string, FilterFieldKind>;

/**
 * Parse a URLSearchParams object into a typed filter values object,
 * driven by a schema describing each field's kind.
 */
export function filtersFromSearchParams(
  params: URLSearchParams,
  schema: FilterSchema
): Record<string, string | string[]> {
  const out: Record<string, string | string[]> = {};
  for (const [key, kind] of Object.entries(schema)) {
    const raw = params.get(key);
    if (raw === null || raw === "") continue;
    if (kind === "string[]") {
      const parts = raw.split(",").map((s) => s.trim()).filter(Boolean);
      if (parts.length > 0) out[key] = parts;
    } else {
      // string and date are both stored as plain strings
      out[key] = raw;
    }
  }
  return out;
}

/**
 * Serialise a filter values object into a URLSearchParams. Empty/undefined
 * fields are omitted entirely so the URL stays clean.
 */
export function filtersToSearchParams(
  values: Record<string, string | string[] | undefined>,
  schema: FilterSchema
): URLSearchParams {
  const params = new URLSearchParams();
  for (const [key, kind] of Object.entries(schema)) {
    const v = values[key];
    if (v === undefined || v === "") continue;
    if (kind === "string[]") {
      if (Array.isArray(v) && v.length > 0) {
        params.set(key, v.join(","));
      }
    } else if (typeof v === "string") {
      params.set(key, v);
    }
  }
  return params;
}
