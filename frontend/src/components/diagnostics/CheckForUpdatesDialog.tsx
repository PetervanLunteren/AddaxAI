/**
 * Check for updates dialog.
 *
 * Fetches the latest GitHub release on open and compares to the
 * runtime app version. No background polling — only checks when the
 * user explicitly clicks the menu item, so we don't spam GitHub's
 * unauthenticated rate limit (60 req/hour/IP).
 */

import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight } from "lucide-react";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { compareVersions, formatVersion } from "@/lib/version";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

interface CheckForUpdatesDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  currentVersion: string;
}

interface GitHubRelease {
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

function normalize(v: string): string {
  return v.replace(/^v/, "").trim();
}

export function CheckForUpdatesDialog({
  open,
  onOpenChange,
  currentVersion,
}: CheckForUpdatesDialogProps) {
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
    enabled: open,
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

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Check for updates</DialogTitle>
          <DialogDescription>
            Compares your installed version to the latest release on
            GitHub.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 text-sm">
          <Row label="Installed" value={formatVersion(current)} />

          {isLoading && (
            <Row label="Latest" value="checking..." muted />
          )}

          {error && (
            <Callout variant="error" size="compact">
              Could not check for updates: {(error as Error).message}.
              Check your internet connection.
            </Callout>
          )}

          {data && latest && (
            <>
              <Row label="Latest" value={formatVersion(latest)} />
              {upToDate ? (
                <Callout variant="success" size="compact">
                  You're on the latest version.
                </Callout>
              ) : ahead ? (
                <Callout variant="info" size="compact">
                  You're running a build that is newer than the latest
                  release.
                </Callout>
              ) : (
                <Callout variant="info">
                  <div className="space-y-2">
                    <div>A newer version is available.</div>
                    <a
                      href={data.html_url}
                      className="inline-flex items-center gap-1 text-primary hover:underline"
                    >
                      View release notes and download
                      <ArrowUpRight className="h-3.5 w-3.5" />
                    </a>
                  </div>
                </Callout>
              )}
            </>
          )}
        </div>

        <DialogFooter>
          <Button onClick={() => onOpenChange(false)}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface RowProps {
  label: string;
  value: string;
  muted?: boolean;
}

function Row({ label, value, muted }: RowProps) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className={muted ? "text-muted-foreground" : "font-mono"}>
        {value}
      </span>
    </div>
  );
}
