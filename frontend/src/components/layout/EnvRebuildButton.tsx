/**
 * One-click "Update now" for a drifted analysis environment.
 *
 * Triggers the backend's designated drift-rebuild path
 * (POST /api/setup/install-env with force_envs) and shows live progress by
 * polling /api/setup/status. This is the single guarded env-build path
 * (_install_state on the backend rejects a concurrent install with 409), so it
 * does not reintroduce the old toast-vs-Settings rebuild race.
 *
 * Note: the drift list itself is computed once at app launch, so the rebuild
 * updates the env's YAML-hash sentinel (it won't reappear next launch) but the
 * notice only disappears after the parent dismisses it.
 */

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { RefreshCw, Check, AlertCircle } from "lucide-react";
import { setupApi } from "../../api/setup";
import { Button } from "../ui/button";

type Phase = "idle" | "rebuilding" | "done" | "error";

interface EnvRebuildButtonProps {
  /** Env names to wipe and rebuild, e.g. ["pytorch"]. */
  envNames: string[];
  /** Called once the rebuild finishes successfully. */
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
      onDone?.();
    }
  }, [status, phase, onDone]);

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

  if (phase === "rebuilding") {
    const pct = Math.round(status?.progress_pct ?? 0);
    return (
      <div className="space-y-1">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <RefreshCw className="h-3.5 w-3.5 animate-spin" />
          <span className="truncate">
            {status?.message || "Starting rebuild..."}
          </span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
          <div
            className="h-full rounded-full bg-primary transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    );
  }

  if (phase === "done") {
    return (
      <div className="flex items-center gap-2 text-xs text-primary">
        <Check className="h-3.5 w-3.5" />
        <span>Environment updated. Restart the app to finish.</span>
      </div>
    );
  }

  if (phase === "error") {
    return (
      <div className="space-y-1">
        <div className="flex items-start gap-2 text-xs text-destructive">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{error || "Rebuild failed."}</span>
        </div>
        <Button
          size="sm"
          variant="outline"
          className="h-7 w-full px-2 text-xs"
          onClick={start}
        >
          Try again
        </Button>
      </div>
    );
  }

  return (
    <Button
      size="sm"
      variant="outline"
      className="h-7 w-full px-2 text-xs"
      onClick={start}
    >
      Update now
    </Button>
  );
}
