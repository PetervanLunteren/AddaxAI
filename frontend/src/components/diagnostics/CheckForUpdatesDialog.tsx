/**
 * Check for updates dialog.
 *
 * Shows how the installed build compares to the newest published
 * release. Opened from the Help menu, or from the startup toast that
 * appears when an update is waiting (see MenuCommands).
 *
 * The comparison itself lives in useLatestRelease, shared with that
 * toast so the two can never tell the user different things. Both read
 * one query, so a launch costs one request whether or not the dialog is
 * opened; there is still no background polling, which keeps us clear of
 * GitHub's unauthenticated rate limit (60 req/hour/IP).
 */

import { ArrowUpRight } from "lucide-react";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { formatVersion } from "@/lib/version";
import { useLatestRelease } from "@/hooks/useLatestRelease";
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

export function CheckForUpdatesDialog({
  open,
  onOpenChange,
  currentVersion,
}: CheckForUpdatesDialogProps) {
  const {
    latest,
    current,
    downloadUrl,
    upToDate,
    ahead,
    isLoading,
    error,
  } = useLatestRelease(currentVersion, open);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Check for updates</DialogTitle>
          <DialogDescription>
            Compares your installed version to the latest available
            release.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 text-sm">
          <Row label="Installed" value={formatVersion(current)} />

          {isLoading && (
            <Row label="Latest" value="checking..." muted />
          )}

          {error && (
            <Callout variant="error" size="compact">
              Could not check for updates: {error.message}. Check your
              internet connection.
            </Callout>
          )}

          {latest && (
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
                      href={downloadUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-primary hover:underline"
                    >
                      Download the latest version
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
