/**
 * The reprocess "how the DB changed" summary, shared by the project
 * Settings page and the folder-run "Refine results" slideout.
 *
 * `showSummary(results)` always shows a dismissible toast panel with a
 * "See effect" link that opens the SaveResultsModal, whether or not any
 * count changed: the modal tells the story either way, including the
 * share of labels/counts that are verified and therefore left untouched.
 * `summaryUI` is the panel + modal, rendered by the host.
 *
 * `savedMessage` is the toast lead text so each surface can phrase it for
 * its own context ("Settings saved!" vs "Changes applied").
 */

import { useCallback, useRef, useState } from "react";
import { Check, X } from "lucide-react";

import {
  type Protection,
  SaveResultsModal,
  type SaveResults,
} from "../components/projects/SaveResultsModal";
import { fetchProtection } from "../lib/reprocessStats";
import { ALL_METRICS, type SaveMetric } from "../lib/saveMetrics";

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
    [projectId],
  );

  const summaryUI = (
    <>
      {toastResults && (
        <div
          className="fixed bottom-6 right-6 z-50 flex items-center gap-3 rounded-lg border border-gray-200 bg-white px-4 py-3 shadow-lg"
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
              See effect
            </button>
          </span>
          <button
            onClick={dismissToast}
            className="ml-1 text-gray-400 hover:text-gray-600"
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
