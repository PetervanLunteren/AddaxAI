/**
 * Frontend path-math helpers used by the bulk-relink dialog and the
 * remember-last-prefix feature in the single-relink dialog.
 *
 * All paths are POSIX-style absolute strings (e.g.,
 * "/Volumes/Drive/project/site_001/dep001"). We don't bother with
 * Windows-style backslashes — the rest of the app already standardizes
 * on forward slashes.
 */

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
    const lastSlash = paths[0].lastIndexOf("/");
    if (lastSlash <= 0) return "";
    return paths[0].slice(0, lastSlash + 1);
  }

  let prefix = paths[0];
  for (let i = 1; i < paths.length; i++) {
    while (prefix && !paths[i].startsWith(prefix)) {
      prefix = prefix.slice(0, -1);
    }
    if (!prefix) return "";
  }

  // Snap to last slash so we don't break in the middle of a folder name
  const lastSlash = prefix.lastIndexOf("/");
  if (lastSlash <= 0) return "";
  return prefix.slice(0, lastSlash + 1);
}

/**
 * Replace the leading `oldPrefix` of `path` with `newPrefix`. Both
 * prefixes are normalized to end with a slash before substitution so
 * the result is well-formed regardless of how the user typed them.
 *
 * If `path` does not start with `oldPrefix`, returns `path` unchanged.
 */
export function replacePrefix(
  path: string,
  oldPrefix: string,
  newPrefix: string
): string {
  const oldNorm = oldPrefix.endsWith("/") ? oldPrefix : oldPrefix + "/";
  const newNorm = newPrefix.endsWith("/") ? newPrefix : newPrefix + "/";
  if (!path.startsWith(oldNorm)) return path;
  return newNorm + path.slice(oldNorm.length);
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
 * slashes are ignored. "/a/b/c/" → "c".
 */
export function leafName(path: string): string {
  if (!path) return "";
  const trimmed = path.replace(/\/+$/, "");
  const lastSlash = trimmed.lastIndexOf("/");
  return lastSlash >= 0 ? trimmed.slice(lastSlash + 1) : trimmed;
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
  const oldParts = oldPath.split("/");
  const newParts = newPath.split("/");

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
 * Shorten a path in the middle while preserving the leaf segment,
 * since the leaf is what users recognize. Returns the original path
 * unchanged if it's already shorter than `maxLen`.
 */
export function truncateMiddle(path: string, maxLen = 60): string {
  if (path.length <= maxLen) return path;
  const lastSlash = path.lastIndexOf("/");
  const leaf = lastSlash >= 0 ? path.slice(lastSlash) : "";
  const keepStart = Math.max(1, maxLen - leaf.length - 1);
  if (leaf.length + 4 > maxLen) {
    return path.slice(0, maxLen - 1) + "…";
  }
  return path.slice(0, keepStart) + "…" + leaf;
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
    const nextComponent = remainder.split("/")[0] ?? "";
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
  const oldParts = oldPath.split("/");
  const newParts = newPath.split("/");

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

  const oldPrefix = oldParts.slice(0, oldParts.length - suffixLen).join("/") + "/";
  const newPrefix = newParts.slice(0, newParts.length - suffixLen).join("/") + "/";

  // Refuse trivially-empty prefixes
  if (oldPrefix === "/" || newPrefix === "/" || oldPrefix === newPrefix) {
    return null;
  }

  return { oldPrefix, newPrefix };
}
