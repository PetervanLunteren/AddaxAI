/**
 * Global, per-user preference for how species names render: the common
 * name (default) or the scientific name. Stored in localStorage so it is
 * device-global and survives reloads.
 *
 * The preference is read synchronously by `resolveSpeciesName` from a
 * module-level cache, so plain (non-React) utilities can resolve names
 * without prop-drilling the mode. Changing the mode is a deliberate, rare
 * action: `setSpeciesNameMode` persists and reloads the page, which is the
 * simplest way to guarantee every rendered name flips consistently.
 */

export type SpeciesNameMode = "common" | "scientific";

const STORAGE_KEY = "addaxai:species-name-mode";

function readStored(): SpeciesNameMode {
  try {
    return localStorage.getItem(STORAGE_KEY) === "scientific"
      ? "scientific"
      : "common";
  } catch {
    // localStorage can throw in locked-down contexts; fall back to default.
    return "common";
  }
}

let current: SpeciesNameMode = readStored();

export function getSpeciesNameMode(): SpeciesNameMode {
  return current;
}

export function setSpeciesNameMode(mode: SpeciesNameMode): void {
  current = mode;
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // Ignore persistence failure; the in-memory value still applies.
  }
  // A full reload re-renders every name everywhere with no stale caches.
  // Toggling is rare, so the cost is acceptable and the result is simple
  // and guaranteed-consistent.
  window.location.reload();
}

/** Fields any species-named object exposes. All optional so callers can
 *  pass detections, summaries, or aggregate rows interchangeably. */
export interface SpeciesNamed {
  common_name?: string | null;
  scientific_name?: string | null;
  label?: string | null;
  category?: string | null;
}

/**
 * Pick the right taxonomy-id -> name map (common or scientific) for the
 * active mode from a summary that carries both. Returns an empty object
 * when neither map is present so callers can index safely.
 */
export function speciesLabelMap(
  obj: {
    common_labels?: Record<string, string> | null;
    scientific_labels?: Record<string, string> | null;
  },
  mode: SpeciesNameMode = current,
): Record<string, string> {
  return (
    (mode === "scientific" ? obj.scientific_labels : obj.common_labels) ?? {}
  );
}

/**
 * Resolve the display name for a species-named object under the active
 * (or given) mode. Common mode prefers `common_name`; scientific mode
 * prefers `scientific_name`. Both fall back through the other name, the
 * raw label, and finally the capitalised category, so a missing field
 * never renders blank.
 */
export function resolveSpeciesName(
  obj: SpeciesNamed,
  mode: SpeciesNameMode = current,
): string {
  const common = obj.common_name || undefined;
  const scientific = obj.scientific_name || undefined;
  const label = obj.label || undefined;
  const category = obj.category || undefined;
  const ordered =
    mode === "scientific"
      ? [scientific, common, label, category]
      : [common, scientific, label, category];
  const picked = ordered.find((v): v is string => Boolean(v)) ?? "";
  // Format like normalizeLabel: underscores -> spaces, capitalise the
  // first letter. Idempotent on the already-formatted common / scientific
  // names; fixes raw lowercase values that arrive unformatted, e.g. the
  // dashboard's taxon-rank keys ("felidae" -> "Felidae") and detector
  // categories ("person" -> "Person").
  return picked.replace(/_/g, " ").replace(/\b\w/, (c) => c.toUpperCase());
}
