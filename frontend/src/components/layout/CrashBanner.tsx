/**
 * Banner shown on app launch when the previous shutdown was not clean.
 *
 * "Not clean" means Electron's `before-quit` handler never ran on the
 * previous session — typical causes are a hard crash, OOM kill, panic,
 * or power loss. We surface it once per session with a clear CTA to
 * export a diagnostic report. Dismissed state is per-session only;
 * we don't persist it because the user might dismiss in a hurry on one
 * launch and want it back on a later one.
 */

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle, Download, X } from "lucide-react";
import { diagnosticsApi } from "../../api/diagnostics";
import { Button } from "../ui/button";

export function CrashBanner() {
  const [dismissed, setDismissed] = useState(false);

  const { data } = useQuery({
    queryKey: ["last-launch-status"],
    queryFn: diagnosticsApi.getLastLaunchStatus,
    staleTime: Infinity, // Once per session is enough.
  });

  const download = useMutation({
    mutationFn: () => diagnosticsApi.downloadDiagnosticZip(),
  });

  if (
    dismissed ||
    !data ||
    !data.snapshot_present ||
    data.previous_shutdown_clean
  ) {
    return null;
  }

  return (
    <div
      role="alert"
      className="border-b text-white"
      style={{ backgroundColor: "#882000" }}
    >
      <div className="mx-auto max-w-7xl px-4 py-2 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <div className="flex-1 text-sm">
            AddaxAI didn't shut down cleanly last time. If this is unexpected,
            export a diagnostic report and email it to support.
          </div>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => download.mutate()}
            disabled={download.isPending}
            className="shrink-0"
          >
            <Download className="h-3.5 w-3.5 mr-1.5" />
            {download.isPending ? "Building..." : "Export report"}
          </Button>
          <button
            onClick={() => setDismissed(true)}
            aria-label="Dismiss"
            className="rounded p-1 hover:bg-white/10 shrink-0"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
