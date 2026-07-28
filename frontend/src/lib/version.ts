/**
 * Version parsing, comparison and display.
 *
 * Single source of truth for anything that has to reason about an app
 * version string. Two call sites need it: the check-for-updates dialog
 * (installed vs latest GitHub release) and the model info sheet
 * (installed vs a model's `min_app_version`). Both used to compare raw
 * strings, which orders "7.0.10" below "7.0.9" because it compares
 * character by character.
 *
 * Comparison follows semver 2.0.0 precedence (spec sections 11.2-11.4):
 * numeric core compared numerically, build metadata ignored, a
 * prerelease sorts below its own release, prerelease identifiers
 * compared field by field.
 *
 * Version strings this app actually produces:
 *   - "7.0.1-beta.23"  a tagged release build (CI writes it from the tag)
 *   - "0.0.0-dev"      the committed marker for untagged dev builds
 *   - "0.0.0+unknown"  backend fallback when the VERSION file is missing
 *   - "(dev)"          browser dev server, no Electron to ask
 *   - "(unknown)"      Electron's getVersion() rejected
 *
 * The last two are not versions. Every function here reports that
 * honestly rather than guessing: `parseVersion` and `compareVersions`
 * return null, and `formatVersion` renders them untouched.
 */

export interface ParsedVersion {
  major: number;
  minor: number;
  patch: number;
  /** Dot-separated prerelease identifiers; empty for a stable release. */
  prerelease: string[];
}

/**
 * major.minor.patch, an optional leading "v", an optional "-prerelease"
 * and an optional "+build". Build metadata is matched so it does not
 * fail the parse, but is discarded: semver 10 says it is ignored when
 * determining precedence.
 */
const VERSION_RE =
  /^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/;

/** Parse a version string. Returns null when it is not a version. */
export function parseVersion(raw: string): ParsedVersion | null {
  const match = VERSION_RE.exec(raw.trim());
  if (!match) return null;
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
    prerelease: match[4] ? match[4].split(".") : [],
  };
}

function sign(n: number): -1 | 0 | 1 {
  return n < 0 ? -1 : n > 0 ? 1 : 0;
}

/**
 * Compare two prerelease identifiers (semver 11.4.1-11.4.3): all-digit
 * identifiers compare numerically, anything else compares as ASCII text,
 * and a numeric identifier always ranks below a non-numeric one.
 */
function compareIdentifier(a: string, b: string): -1 | 0 | 1 {
  const aNumeric = /^\d+$/.test(a);
  const bNumeric = /^\d+$/.test(b);
  if (aNumeric && bNumeric) return sign(Number(a) - Number(b));
  if (aNumeric) return -1;
  if (bNumeric) return 1;
  return a < b ? -1 : a > b ? 1 : 0;
}

function comparePrerelease(a: string[], b: string[]): -1 | 0 | 1 {
  // Semver 11.3: a version with a prerelease ranks below the same
  // version without one, so 7.0.1-beta.23 < 7.0.1.
  if (a.length === 0 && b.length === 0) return 0;
  if (a.length === 0) return 1;
  if (b.length === 0) return -1;

  const shared = Math.min(a.length, b.length);
  for (let i = 0; i < shared; i++) {
    const result = compareIdentifier(a[i], b[i]);
    if (result !== 0) return result;
  }
  // Semver 11.4.4: when every shared field is equal, more fields wins.
  return sign(a.length - b.length);
}

/**
 * Compare two version strings.
 *
 * Returns -1 when a ranks below b, 0 when they are equal, 1 when a
 * ranks above b, and null when either side is not a version (a "(dev)"
 * placeholder, an empty string). Callers must handle null rather than
 * treating it as a comparison result.
 */
export function compareVersions(a: string, b: string): -1 | 0 | 1 | null {
  const left = parseVersion(a);
  const right = parseVersion(b);
  if (!left || !right) return null;
  if (left.major !== right.major) return sign(left.major - right.major);
  if (left.minor !== right.minor) return sign(left.minor - right.minor);
  if (left.patch !== right.patch) return sign(left.patch - right.patch);
  return comparePrerelease(left.prerelease, right.prerelease);
}

/**
 * Does `current` satisfy a model's `min_app_version`?
 *
 * Compares release cores only and ignores any prerelease suffix, which
 * is deliberately NOT strict semver. The model catalog authors
 * `min_app_version` as the release cycle in which a model became
 * available ("7.0.1" on all 39 entries today), while every build the
 * app has ever shipped is a prerelease of that cycle
 * ("7.0.1-beta.23"). Strict semver ranks a prerelease below its own
 * release, so comparing strictly would tell every beta tester to
 * update to a version that has never been released.
 *
 * Returns null when either side is not a version, so the caller decides
 * what to do when the check cannot be made.
 */
export function satisfiesMinVersion(
  current: string,
  min: string
): boolean | null {
  const installed = parseVersion(current);
  const required = parseVersion(min);
  if (!installed || !required) return null;
  if (installed.major !== required.major)
    return installed.major > required.major;
  if (installed.minor !== required.minor)
    return installed.minor > required.minor;
  if (installed.patch !== required.patch)
    return installed.patch > required.patch;
  return true;
}

/**
 * Render a version for display. Real versions get a "v" prefix; the
 * "(dev)" and "(unknown)" placeholders carry their own brackets and
 * render as-is, so the UI never shows "v(dev)".
 */
export function formatVersion(raw: string): string {
  if (!parseVersion(raw)) return raw;
  return `v${raw.replace(/^v/, "")}`;
}
