/**
 * The reprocess "how the DB changed" summary, shared by the project
 * Settings page and the folder-run "Refine results" slideout.
 *
 * `showSummary(results)` shows a dismissible toast panel with a "See the
 * effect" link that opens the SaveResultsModal, but only when a number
 * moved. A save that changed nothing (a rename, a threshold step that
 * crossed no score) gets a plain toast: a modal of "No change." told the
 * user nothing, and the link on every save taught them to ignore it.
 * `summaryUI` is the panel + modal, rendered by the host.
 *
 * `savedMessage` is the toast lead text so each surface can phrase it for
 * its own context ("Settings saved!" vs "Changes applied").
 */

import { useCallback, useRef, useState } from "react";
import { Check, X } from "lucide-react";
import { toast } from "sonner";

import {
  type Protection,
  SaveResultsModal,
  type SaveResults,
} from "../components/projects/SaveResultsModal";
import { fetchProtection } from "../lib/reprocessStats";
import {
  ALL_METRICS,
  hasChanges,
  metricsPhrase,
  type SaveMetric,
} from "../lib/saveMetrics";

export function useReprocessSummary(
  projectId: string,
  savedMessage = "Settings saved!",
  metrics: SaveMetric[] = ALL_METRICS,
): {
  showSummary: (results: SaveResults) => void;
  summaryUI: React.ReactNode;
} {
  const [saveResults, setSaveResults] = useState<SaveResults | null>(null);
  const [toastResults, setToastResults] = useState<SaveResults | null>(null);
  const [protection, setProtection] = useState<Protection | null>(null);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const dismissToast = useCallback(() => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    setToastResults(null);
  }, []);

  const showSummary = useCallback(
    (results: SaveResults) => {
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
      if (!hasChanges(results, metrics)) {
        setToastResults(null);
        toast.success(savedMessage);
        return;
      }
      setToastResults(results);
      toastTimerRef.current = setTimeout(() => setToastResults(null), 5000);
      // Load the verified/confirmed share for the modal's footer lines.
      // Best-effort: on failure the modal just omits the lines.
      if (projectId) {
        fetchProtection(projectId)
          .then(setProtection)
          .catch(() => setProtection(null));
      }
    },
    [projectId, savedMessage, metrics],
  );

  const summaryUI = (
    <>
      {toastResults && (
        <div
          className="fixed bottom-6 right-6 z-50 flex items-center gap-3 rounded-lg border bg-popover px-4 py-3 text-popover-foreground shadow-lg"
          style={{ animation: "toast-slide-up 0.2s ease-out" }}
        >
          <Check className="h-4 w-4 flex-shrink-0 text-primary" />
          <span className="text-sm">
            {savedMessage}{" "}
            <button
              onClick={() => {
                setSaveResults(toastResults);
                dismissToast();
              }}
              className="font-medium text-primary hover:underline"
            >
              See the effect on {metricsPhrase(metrics)}
            </button>
          </span>
          <button
            onClick={dismissToast}
            className="ml-1 text-muted-foreground hover:text-foreground"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {saveResults && (
        <SaveResultsModal
          open={saveResults !== null}
          onOpenChange={(open) => !open && setSaveResults(null)}
          results={saveResults}
          metrics={metrics}
          protection={protection}
        />
      )}
    </>
  );

  return { showSummary, summaryUI };
}
