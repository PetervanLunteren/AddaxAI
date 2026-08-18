/**
 * "Update now" for a drifted analysis environment.
 *
 * Clicking the button opens a confirm step first: the rebuild is long
 * (tens of minutes) and cannot be cancelled once started, so the user
 * gets the warning while backing out is still possible (beta feedback
 * from Saul: the old flow only said this after the rebuild had already
 * begun). Confirming triggers the backend's designated drift-rebuild
 * path (POST /api/setup/install-env with force_envs) and the modal
 * becomes blocking (no X, no ESC, no backdrop dismiss, no cancel).
 * Progress comes from polling /api/setup/status. The rebuild wipes the
 * env before recreating it, so there is nothing sensible to cancel
 * into; the user waits until it finishes or fails.
 *
 * This is the single guarded env-build path (_install_state on the
 * backend rejects a concurrent install with 409), so it does not
 * reintroduce the old toast-vs-Settings rebuild race.
 *
 * The rebuild rewrites the env's YAML-hash sentinel, and
 * GET /api/ml/updates re-reads that sentinel on every request, so the
 * notice is gone for good once this finishes. Dismissing it here is
 * only about closing the toast the user is looking at.
 */

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertCircle, Check, RefreshCw } from "lucide-react";
import { setupApi } from "../../api/setup";
import { Button } from "../ui/button";
import { Progress } from "../ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

type Phase = "idle" | "confirm" | "rebuilding" | "done" | "error";

interface EnvRebuildButtonProps {
  /** Env names to wipe and rebuild, e.g. ["pytorch"]. */
  envNames: string[];
  /** Called when the user closes the modal after a successful rebuild. */
  onDone?: () => void;
}

export function EnvRebuildButton({ envNames, onDone }: EnvRebuildButtonProps) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  // Whether we have observed the backend actually building, so a status poll
  // that arrives before the thread starts isn't mistaken for "done".
  const sawInProgress = useRef(false);

  const { data: status } = useQuery({
    queryKey: ["setup-status", "env-rebuild"],
    queryFn: setupApi.getStatus,
    enabled: phase === "rebuilding",
    refetchInterval: phase === "rebuilding" ? 1500 : false,
  });

  useEffect(() => {
    if (phase !== "rebuilding" || !status) return;
    if (status.install_in_progress) {
      sawInProgress.current = true;
      return;
    }
    // Not in progress: only conclude once we've seen it start (avoids the
    // POST -> first-poll gap before the worker flips the flag).
    if (!sawInProgress.current) return;
    if (status.error) {
      setError(status.error);
      setPhase("error");
    } else {
      setPhase("done");
    }
  }, [status, phase]);

  // Warn before closing the window while the rebuild is running. The
  // env is wiped first, so quitting mid-rebuild leaves a broken env
  // and sends the user through the first-run wizard on next launch.
  useEffect(() => {
    if (phase !== "rebuilding") return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [phase]);

  const start = async () => {
    setError(null);
    sawInProgress.current = false;
    setPhase("rebuilding");
    try {
      await setupApi.rebuildEnvs(envNames);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not start the rebuild",
      );
      setPhase("error");
    }
  };

  const pct = Math.round(status?.progress_pct ?? 0);
  const modalOpen = phase !== "idle";

  return (
    <>
      <Button
        size="sm"
        variant="outline"
        className="h-7 w-full px-2 text-xs"
        onClick={() => setPhase("confirm")}
      >
        Update now
      </Button>

      <Dialog open={modalOpen}>
        <DialogContent className="max-w-lg" nonDismissable>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {phase === "rebuilding" && (
                <RefreshCw className="h-4 w-4 animate-spin" />
              )}
              Updating analysis environment
            </DialogTitle>
            <DialogDescription>
              The environment is wiped and rebuilt to match this app
              version. This can take 10 to 30 minutes depending on your
              machine and internet connection. Keep the app open so it can
              finish, but quitting is safe: your projects and data are
              untouched, and the rebuild simply starts over the next time
              you open AddaxAI.
            </DialogDescription>
          </DialogHeader>

          <ul className="text-xs text-muted-foreground space-y-0.5">
            {envNames.map((name) => (
              <li key={name} className="font-mono truncate">
                env-{name}
              </li>
            ))}
          </ul>

          {phase === "rebuilding" && (
            <div className="space-y-3">
              <div className="space-y-2 min-w-0">
                <Progress value={pct} className="h-2" />
                <div className="flex justify-end text-sm">
                  <span className="text-muted-foreground">{pct}%</span>
                </div>
              </div>
              <div className="bg-muted/50 rounded-md px-3 py-2 min-w-0">
                <p className="text-[11px] leading-none text-muted-foreground font-mono truncate">
                  {status?.message || "Starting rebuild..."}
                </p>
              </div>
            </div>
          )}

          {phase === "done" && (
            <div className="flex items-center gap-2 text-sm text-primary">
              <Check className="h-4 w-4 shrink-0" />
              <span>Environment updated.</span>
            </div>
          )}

          {phase === "error" && (
            <div className="flex items-start gap-2 text-sm text-destructive">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              {/* Same classes as the other two surfaces that render a
                  backend install error. min-w-0 is needed here and not
                  there: this span is a flex child, and without it the
                  long paths and hashes in pip output push it 350px past
                  the dialog edge instead of wrapping. */}
              <span className="min-w-0 break-words whitespace-pre-line">
                {error || "Rebuild failed."}
              </span>
            </div>
          )}

          {phase === "confirm" && (
            <>
              <p className="text-sm text-muted-foreground">
                Not a good moment? Cancel and update later, the notice
                comes back the next time you open the app.
              </p>
              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPhase("idle")}
                >
                  Cancel
                </Button>
                <Button size="sm" onClick={start}>
                  Update now
                </Button>
              </div>
            </>
          )}

          {(phase === "done" || phase === "error") && (
            <div className="flex justify-end gap-2">
              {phase === "error" && (
                <Button variant="outline" size="sm" onClick={start}>
                  Try again
                </Button>
              )}
              <Button
                size="sm"
                onClick={() => {
                  if (phase === "done") {
                    onDone?.();
                  }
                  setPhase("idle");
                }}
              >
                Close
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
