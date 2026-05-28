/**
 * Species color mapping utility
 *
 * Generates consistent colors for species using a deterministic hash.
 * The same label string always produces the same color, regardless of
 * what other labels are on the page. No global mutable state.
 *
 * Gradient: #0f6064 (dark teal) -> #f9f871 (light yellow)
 */
import chroma from 'chroma-js';

// Species color gradient (single source of truth)
// Previous: chroma.scale(['#0f6064', '#f9f871'])
const speciesScale = chroma.scale(['#0f6064', '#f9f871']);

// Aliases map: taxonomy UUID -> label name (and vice versa).
// Both keys resolve to the same hash-based color because the
// hash is computed from whichever key was registered first.
const aliasCache: Map<string, string> = new Map();

/**
 * Register aliases so that a taxonomy UUID and its display name
 * resolve to the same color. Call this when event data is loaded.
 */
export function setSpeciesContext(
  _speciesList: string[],
  aliases?: Record<string, string>,
): void {
  if (aliases) {
    for (const [key, alias] of Object.entries(aliases)) {
      const k = key.toLowerCase();
      const a = alias.toLowerCase();
      if (a) {
        aliasCache.set(k, a);
        aliasCache.set(a, k);
      }
    }
  }
}

/** Deterministic hash of a string to a 0-1 gradient position. */
function hashToPosition(str: string): number {
  // FNV-1a hash for even distribution
  let hash = 2166136261;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) % 1000) / 1000;
}

/** Get the canonical key for color lookup (follows aliases). */
function canonicalKey(species: string): string {
  const lower = species.toLowerCase();
  // If this key has an alias, pick whichever is shorter (more likely
  // the human-readable name) to hash. This ensures UUID and name
  // produce the same color.
  const alias = aliasCache.get(lower);
  if (alias) {
    return lower.length <= alias.length ? lower : alias;
  }
  return lower;
}

export function getSpeciesColor(species: string): string {
  const position = hashToPosition(canonicalKey(species));
  return speciesScale(position).hex();
}

export function getSpeciesColors(speciesList: string[]): string[] {
  return speciesList.map(species => getSpeciesColor(species));
}

export function getSpeciesColorWithAlpha(species: string, alpha: number = 0.8): string {
  const position = hashToPosition(canonicalKey(species));
  return speciesScale(position).alpha(alpha).css();
}

export function getSpeciesChartColors(species: string, backgroundAlpha: number = 0.8): {
  borderColor: string;
  backgroundColor: string;
} {
  return {
    borderColor: getSpeciesColor(species),
    backgroundColor: getSpeciesColorWithAlpha(species, backgroundAlpha),
  };
}

/** Pick a foreground color (white or near-black) that meets WCAG AA
 * contrast against the given background.
 *
 * Pass the ACTUAL rendered background colour (hex string) rather than a
 * species key: there were chips where the background was derived from
 * `label_taxonomy_id` while the text was derived from `label`, the two
 * keys hashed to different positions on the scale, and white-on-yellow
 * slipped through unreadable. Sourcing the bg directly makes the
 * mismatch impossible. Threshold is 4.5 (AA, normal text); chips below
 * that get dark text.
 */
export function getContrastTextColor(bgHex: string): string {
  return chroma.contrast(bgHex, "white") >= 4.5 ? "white" : "#1f2937";
}

/** @deprecated use `getContrastTextColor(getSpeciesColor(key))` so the
 * bg and the text are computed from the same key. Retained briefly so
 * older call sites compile while they migrate. */
export function getSpeciesTextColor(species: string): string {
  return getContrastTextColor(getSpeciesColor(species));
}
