/**
 * Latest published release, and how the installed build compares to it.
 *
 * Two callers need the same answer and must never disagree: the startup
 * toast, which decides whether to say anything at all, and the
 * check-for-updates dialog, which shows the detail. They share one
 * react-query key, so a launch that shows the toast costs a single
 * request and opening the dialog afterwards renders from cache.
 */

import { useQuery } from "@tanstack/react-query";
import { compareVersions, parseVersion } from "@/lib/version";

export interface GitHubRelease {
  tag_name: string;
  html_url: string;
}

// Deliberately not /releases/latest: GitHub documents that one as "the
// most recent non-prerelease, non-draft release", so the moment a
// release is flagged as a pre-release in the GitHub UI it becomes
// invisible there and users get told they are up to date on an older
// build. We ship betas, so take the newest release whatever its flag.
// Drafts are not returned to unauthenticated callers, so they cannot
// leak in.
const RELEASES_API =
  "https://api.github.com/repos/PetervanLunteren/AddaxAI/releases?per_page=1";

// Where a user goes to get the new version. Not the GitHub release
// page: that is a wall of assets and checksums written for developers,
// and most people running AddaxAI are ecologists. The site owns the
// download story and can change it without an app release.
export const DOWNLOAD_URL = "https://addaxai.com";

function normalize(v: string): string {
  return v.replace(/^v/, "").trim();
}

export interface LatestReleaseState {
  /** Newest published release version, without a leading "v". */
  latest: string | null;
  /** The installed version, normalised the same way. */
  current: string;
  /** Where to send the user to download it. */
  downloadUrl: string;
  upToDate: boolean;
  /** Installed build is newer than anything released (a dev build). */
  ahead: boolean;
  /** A newer release exists and the user can act on it. */
  updateAvailable: boolean;
  isLoading: boolean;
  error: Error | null;
}

/**
 * @param currentVersion installed version, as Electron reports it. May
 *   be a placeholder like "(dev)"; every derived flag stays false then,
 *   rather than guessing.
 * @param enabled whether to hit the network at all.
 */
export function useLatestRelease(
  currentVersion: string,
  enabled: boolean,
): LatestReleaseState {
  const { data, isLoading, error } = useQuery({
    queryKey: ["latest-release"],
    queryFn: async (): Promise<GitHubRelease> => {
      const res = await fetch(RELEASES_API);
      if (!res.ok) {
        throw new Error(`GitHub returned ${res.status}`);
      }
      const releases: GitHubRelease[] = await res.json();
      if (releases.length === 0) {
        throw new Error("no published releases found");
      }
      return releases[0];
    },
    enabled,
    staleTime: 60_000, // Don't re-fetch within a minute.
  });

  const latest = data ? normalize(data.tag_name) : null;
  const current = normalize(currentVersion);

  // Exact string equality is the only thing that claims "up to date",
  // which keeps that branch as safe as it was before the comparator
  // existed. The comparator only splits the remaining cases into
  // "ahead of the latest release" and "an update exists", so a bug in
  // it can never hide an available update from a user.
  const upToDate = latest !== null && latest === current;
  const ahead =
    latest !== null && !upToDate && compareVersions(current, latest) === 1;

  // The toast acts on this, so it has to be certain. compareVersions
  // returns null when either side is not a version, and parseVersion
  // rejects the "(dev)" / "(unknown)" placeholders, so a build that
  // cannot name itself never nags anybody.
  const updateAvailable =
    latest !== null &&
    !upToDate &&
    !ahead &&
    parseVersion(current) !== null &&
    compareVersions(current, latest) === -1;

  return {
    latest,
    current,
    downloadUrl: DOWNLOAD_URL,
    upToDate,
    ahead,
    updateAvailable,
    isLoading,
    error: (error as Error) ?? null,
  };
}
