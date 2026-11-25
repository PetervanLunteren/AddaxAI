/**
 * Frontend path-math helpers used by the bulk-relink dialog, the
 * remember-last-prefix feature in the single-relink dialog, and the
 * deployments / queue / file-detail breadcrumb displays.
 *
 * Paths arrive from the backend in their native form: forward slashes
 * on macOS / Linux, backslashes on Windows. Every helper here treats
 * both separators as equivalent for parsing. `replacePrefix`, which
 * produces a path back out, preserves the input's separator style so
 * the result can round-trip to the backend without breaking filesystem
 * operations on Windows.
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

/** Split a path into segments, treating `/` and `\` as equivalent. */
export function splitPath(path: string): string[] {
  return path.split(SEP_RE);
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
  // Strip ALL trailing separators before picking a separator style.
  // Callers append "/" to a prefix regardless of platform
  // (RelinkGroupBanner), so a Windows prefix can arrive as
  // `C:\...\TrailCam/`. Picking the separator first saw that foreign
  // slash, chose "/", and the startsWith check below could never match
  // the backslash path: the old path came back unchanged and the
  // backend was asked to relink every deployment to the very folder
  // that had gone missing. Stripping first leaves a pure native prefix,
  // and preserves each prefix's own separator style so a Windows path
  // survives the substitution as a Windows path.
  const oldBase = oldPrefix.replace(TRAIL_SEP_RE, "");
  const newBase = newPrefix.replace(TRAIL_SEP_RE, "");
  const oldSep = pickSeparator(oldBase);
  const newSep = pickSeparator(newBase);

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
