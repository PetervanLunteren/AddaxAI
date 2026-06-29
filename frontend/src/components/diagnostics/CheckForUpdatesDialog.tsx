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
  name: string;
  published_at: string;
  prerelease: boolean;
}

const RELEASES_API =
  "https://api.github.com/repos/PetervanLunteren/AddaxAI-WebUI/releases/latest";

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
      return res.json();
    },
    enabled: open,
    staleTime: 60_000, // Don't re-fetch within a minute.
  });

  const latest = data ? normalize(data.tag_name) : null;
  const current = normalize(currentVersion);
  const upToDate = latest !== null && latest === current;

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
          <Row label="Installed" value={`v${current}`} />

          {isLoading && (
            <Row label="Latest" value="checking..." muted />
          )}

          {error && (
            <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive/90">
              Could not reach GitHub: {(error as Error).message}.
              Check your internet connection.
            </div>
          )}

          {data && latest && (
            <>
              <Row label="Latest" value={`v${latest}`} />
              {upToDate ? (
                <Callout variant="success" size="compact">
                  You're on the latest version.
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
