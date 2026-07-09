/**
 * The reprocess "how the DB changed" summary, shared by the project
 * Settings page and the folder-run "Refine results" slideout.
 *
 * `showSummary(results)` compares the before/after snapshots: when a
 * count changed it shows a dismissible toast panel with a "See effect"
 * link that opens the SaveResultsModal; when nothing changed it falls
 * back to a plain success toast (no empty modal). `summaryUI` is the
 * panel + modal, rendered by the host.
 *
 * `savedMessage` is the toast lead text so each surface can phrase it
 * for its own context ("Settings saved!" vs "Changes applied").
 */

import { useCallback, useRef, useState } from "react";
import { Check, X } from "lucide-react";
import { toast } from "sonner";

import {
  SaveResultsModal,
  type SaveResults,
} from "../components/projects/SaveResultsModal";
import {
  ALL_METRICS,
  METRIC_FIELD,
  type SaveMetric,
} from "../lib/saveMetrics";

function snapshotsDiffer(
  before: { total: number; labels: { label: string; count: number }[] },
  after: { total: number; labels: { label: string; count: number }[] },
): boolean {
  if (before.total !== after.total) return true;
  const beforeByLabel = new Map(before.labels.map((l) => [l.label, l.count]));
  const afterByLabel = new Map(after.labels.map((l) => [l.label, l.count]));
  const allLabels = new Set([...beforeByLabel.keys(), ...afterByLabel.keys()]);
  for (const label of allLabels) {
    if ((beforeByLabel.get(label) ?? 0) !== (afterByLabel.get(label) ?? 0)) {
      return true;
    }
  }
  return false;
}

export function useReprocessSummary(
  savedMessage = "Settings saved!",
  metrics: SaveMetric[] = ALL_METRICS,
): {
  showSummary: (results: SaveResults) => void;
  summaryUI: React.ReactNode;
} {
  const [saveResults, setSaveResults] = useState<SaveResults | null>(null);
  const [toastResults, setToastResults] = useState<SaveResults | null>(null);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const dismissToast = useCallback(() => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    setToastResults(null);
  }, []);

  /** When before/after stats are identical the change didn't touch any
   * counts (e.g. a setting that only affects future analyses), so fall
   * back to a plain toast without the "See effect" link that would open
   * an empty diff modal. Equal totals aren't enough: relabel / rollup
   * changes can shuffle counts between labels and net to zero at the
   * aggregate level, so also compare per-label counts. */
  const showSummary = useCallback(
    (results: SaveResults) => {
      // Only diff the metrics we actually show, so a change in a hidden
      // card never triggers a "See effect" that opens on unchanged ones.
      const changed = metrics.some((metric) => {
        const { before, after } = results[METRIC_FIELD[metric]];
        return snapshotsDiffer(before, after);
      });
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
      if (!changed) {
        setToastResults(null);
        toast.success(savedMessage);
        return;
      }
      setToastResults(results);
      toastTimerRef.current = setTimeout(() => setToastResults(null), 5000);
    },
    [savedMessage, metrics],
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
        />
      )}
    </>
  );

  return { showSummary, summaryUI };
}
