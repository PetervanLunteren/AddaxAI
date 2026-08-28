/**
 * Frontend path-math helpers used by the bulk-relink dialog, the
 * remember-last-prefix feature in the single-relink dialog, and the
 * deployments / queue / file-detail breadcrumb displays.
 *
 * Paths arrive from the backend in their native form: forward slashes
 * on macOS / Linux, backslashes on Windows. Every helper here treats
 * both separators as equivalent for parsing. Helpers that produce a
 * path back out (`replacePrefix`, `deriveSubstitution`) preserve the
 * input's separator style so the result can round-trip to the
 * backend without breaking filesystem operations on Windows.
 */

const SEP_RE = /[/\\]/;
const TRAIL_SEP_RE = /[/\\]+$/;

/** Index of the last `/` or `\` in `path`, or -1 if neither is present. */
function lastSepIndex(path: string): number {
  return Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
}

/**
 * Heuristic: Windows path if it contains a backslash and no forward
 * slash. Anything else (POSIX, mixed, empty) gets forward slash. Mixed
 * cases shouldn't occur in practice; the backend never produces them.
 */
function pickSeparator(path: string): string {
  return path.includes("\\") && !path.includes("/") ? "\\" : "/";
}

/**
 * Leaf segment of a path, with trailing separators stripped. Returns
 * the empty string when given an empty / null path. Handles both
 * separator styles so it works on Windows paths too.
 */
export function basename(path: string | null | undefined): string {
  if (!path) return "";
  const trimmed = path.replace(TRAIL_SEP_RE, "");
  const i = lastSepIndex(trimmed);
  return i >= 0 ? trimmed.slice(i + 1) : trimmed;
}

/**
 * Everything before the leaf segment, with the trailing separator
 * stripped: the folder a file lives in. Empty when the path has no
 * separator. Same separator handling as `basename`.
 */
export function dirname(path: string | null | undefined): string {
  if (!path) return "";
  const trimmed = path.replace(TRAIL_SEP_RE, "");
  const i = lastSepIndex(trimmed);
  return i > 0 ? trimmed.slice(0, i) : "";
}

/** Split a path into segments, treating `/` and `\` as equivalent. */
export function splitPath(path: string): string[] {
  return path.split(SEP_RE);
}

/**
 * Compute the longest common prefix of a list of paths, snapped to a
 * directory boundary (i.e., truncated at the last slash).
 *
 * Returns an empty string if there's no useful common prefix
 * (no shared parent above the root).
 *
 * Example:
 *   /Volumes/Drive/site_001/dep001
 *   /Volumes/Drive/site_001/dep002
 *   /Volumes/Drive/site_002/dep001
 *   → "/Volumes/Drive/"
 */
export function longestCommonPrefix(paths: string[]): string {
  if (paths.length === 0) return "";
  if (paths.length === 1) {
    // Parent directory of the single path
    const i = lastSepIndex(paths[0]);
    if (i <= 0) return "";
    return paths[0].slice(0, i + 1);
  }

  let prefix = paths[0];
  for (let i = 1; i < paths.length; i++) {
    while (prefix && !paths[i].startsWith(prefix)) {
      prefix = prefix.slice(0, -1);
    }
    if (!prefix) return "";
  }

  // Snap to the last separator so we don't break in the middle of a name.
  const i = lastSepIndex(prefix);
  if (i <= 0) return "";
  return prefix.slice(0, i + 1);
}

/**
 * Replace the leading `oldPrefix` of `path` with `newPrefix`. Trailing
 * separators on either prefix are ignored, so the result is well-formed
 * regardless of how the user typed them.
 *
 * `oldPrefix` may be the whole of `path`, which is what happens when the
 * folder that went missing is the deployment's own folder rather than an
 * ancestor of it. That is the common case: one folder renamed or moved.
 * Requiring a trailing separator on both sides used to fail there, since
 * `/a/deployment_001` does not start with `/a/deployment_001/`. The path
 * came back unchanged and the caller then asked the backend to relink the
 * deployment to the very folder that had gone missing, which it always
 * refused with "Folder not found".
 *
 * If `path` is not `oldPrefix` or below it, returns `path` unchanged.
 */
export function replacePrefix(
  path: string,
  oldPrefix: string,
  newPrefix: string
): string {
  // Preserve each prefix's own separator style so a Windows path
  // survives the substitution as a Windows path.
  const oldSep = pickSeparator(oldPrefix);
  const newSep = pickSeparator(newPrefix);
  const stripTrailing = (p: string, sep: string) =>
    p.endsWith(sep) ? p.slice(0, -sep.length) : p;
  const oldBase = stripTrailing(oldPrefix, oldSep);
  const newBase = stripTrailing(newPrefix, newSep);

  if (path === oldBase) return newBase;

  const oldNorm = oldBase + oldSep;
  if (!path.startsWith(oldNorm)) return path;
  return newBase + newSep + path.slice(oldNorm.length);
}

export interface PathItem {
  id: string;
  folder_path: string;
}

export interface PrefixGroup<T extends PathItem> {
  /** Common prefix all items in this group share. Always ends with "/". */
  prefix: string;
  items: T[];
}

/**
 * Return the leaf (last non-empty path segment) of a path. Trailing
 * separators are ignored. Alias of `basename` kept for callers that
 * already use it.
 */
export function leafName(path: string): string {
  return basename(path);
}

/**
 * Diff two paths at the component level so callers can render the
 * common prefix / suffix in a muted style and highlight the parts
 * that actually differ.
 *
 * Returns arrays of path segments (not joined strings) so the caller
 * can control separator rendering and avoid slash-escaping surprises.
 *
 * Example: diffing
 *   /a/b/project_Kenya/Kifaru Plains
 *   /a/b/project_Kenya/Kifaru Plains2
 * yields prefixParts = ["", "a", "b", "project_Kenya"],
 * oldMidParts = ["Kifaru Plains"], newMidParts = ["Kifaru Plains2"],
 * suffixParts = [].
 */
export function diffPaths(
  oldPath: string,
  newPath: string
): {
  prefixParts: string[];
  oldMidParts: string[];
  newMidParts: string[];
  suffixParts: string[];
} {
  const oldParts = splitPath(oldPath);
  const newParts = splitPath(newPath);

  let p = 0;
  while (
    p < oldParts.length &&
    p < newParts.length &&
    oldParts[p] === newParts[p]
  ) {
    p++;
  }

  let s = 0;
  while (
    p + s < oldParts.length &&
    p + s < newParts.length &&
    oldParts[oldParts.length - 1 - s] === newParts[newParts.length - 1 - s]
  ) {
    s++;
  }

  return {
    prefixParts: oldParts.slice(0, p),
    oldMidParts: oldParts.slice(p, oldParts.length - s),
    newMidParts: newParts.slice(p, newParts.length - s),
    suffixParts: oldParts.slice(oldParts.length - s),
  };
}

/**
 * Group items by their common path prefix. Subdivides aggressively so
 * each returned group has the *deepest* meaningful prefix, not just the
 * global LCP across all items.
 *
 * Algorithm: compute the global LCP, then split items by their next
 * path component beyond that LCP. If everyone agrees on the next
 * component, return a single group with the global LCP. Otherwise,
 * return one group per divergent subtree, each with its own recomputed
 * (deeper) LCP.
 *
 * Used by the bulk-relink dialog so the user sees e.g.
 * `/Downloads/example-data/project_A/` instead of a useless
 * `/Downloads/` when broken deployments span multiple projects.
 */
export function groupByPrefix<T extends PathItem>(items: T[]): PrefixGroup<T>[] {
  if (items.length === 0) return [];
  if (items.length === 1) {
    return [{ prefix: longestCommonPrefix([items[0].folder_path]), items }];
  }

  const globalPrefix = longestCommonPrefix(items.map((i) => i.folder_path));

  // Split by the next path component after the global LCP.
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const remainder = item.folder_path.slice(globalPrefix.length);
    const nextComponent = splitPath(remainder)[0] ?? "";
    if (!groups.has(nextComponent)) groups.set(nextComponent, []);
    groups.get(nextComponent)!.push(item);
  }

  // Everyone agrees on the next component → one group with the global LCP.
  if (groups.size === 1) {
    return [{ prefix: globalPrefix, items }];
  }

  // Divergence → one group per subtree, each with its own deeper LCP.
  return Array.from(groups.values()).map((groupItems) => ({
    prefix: longestCommonPrefix(groupItems.map((i) => i.folder_path)),
    items: groupItems,
  }));
}

/**
 * Given an old absolute path and a new absolute path that share the
 * same suffix (e.g. /old/site/dep001 → /new/site/dep001), return the
 * longest common suffix and the (oldPrefix, newPrefix) pair derived
 * from stripping that suffix off both ends.
 *
 * Used after a single-relink to remember "the user moved /old → /new"
 * so the next single-relink can pre-fill the new path.
 *
 * Returns null if the paths don't share any directory-aligned suffix.
 */
export function deriveSubstitution(
  oldPath: string,
  newPath: string
): { oldPrefix: string; newPrefix: string } | null {
  const oldSep = pickSeparator(oldPath);
  const newSep = pickSeparator(newPath);
  const oldParts = splitPath(oldPath);
  const newParts = splitPath(newPath);

  // Walk from the end while parts match
  let suffixLen = 0;
  while (
    suffixLen < oldParts.length &&
    suffixLen < newParts.length &&
    oldParts[oldParts.length - 1 - suffixLen] ===
      newParts[newParts.length - 1 - suffixLen]
  ) {
    suffixLen++;
  }

  if (suffixLen === 0) return null;

  // Re-emit prefixes using each input's native separator so the result
  // round-trips to the backend in the right form.
  const oldPrefix =
    oldParts.slice(0, oldParts.length - suffixLen).join(oldSep) + oldSep;
  const newPrefix =
    newParts.slice(0, newParts.length - suffixLen).join(newSep) + newSep;

  // Refuse trivially-empty prefixes
  if (oldPrefix === oldSep || newPrefix === newSep || oldPrefix === newPrefix) {
    return null;
  }

  return { oldPrefix, newPrefix };
}
