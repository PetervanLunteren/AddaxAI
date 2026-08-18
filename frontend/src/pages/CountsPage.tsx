/**
 * Counts page — thin wrapper around `VerifyView`.
 *
 * One of the two verification pages (the other is `LabelsPage`). The
 * Counts page is the ecological record: review each event's media in the
 * gallery and confirm the species and counts (the data model and exports
 * keep the standard term "observation"). Provides the canonical
 * research-projects page chrome and hands `projectId` to `VerifyView`,
 * which owns the event gallery, state, queries, and the event detail modal.
 */

import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { eventsApi } from "../api/events";
import { BrokenFolderBanner } from "../components/deployments/BrokenFolderBanner";
import { VerifyView } from "../components/verify/VerifyView";
import { WideModeContext, useWideMode } from "../components/verify/wide-mode";
import { cn } from "../lib/utils";

export default function CountsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const wideMode = useWideMode();

  // Wide mode drops the readable-width cap so the gallery fills the
  // content area; the normal view keeps the app-wide max-w-7xl. The
  // toggle lives in the view's toolbar (VerifyView).
  const shell = (vertical: string) =>
    cn(
      "px-4 sm:px-6 lg:px-8",
      vertical,
      wideMode.wide ? "w-full" : "mx-auto max-w-7xl",
    );

  // Drive the subtitle from the unfiltered event count. Same query key
  // as VerifyView's internal totalCountData, so the TanStack cache
  // serves both subscribers from one fetch.
  const { data: totalCountData } = useQuery({
    queryKey: ["event-count", projectId],
    queryFn: () => eventsApi.count(projectId!),
    enabled: !!projectId,
  });
  const totalEvents = totalCountData?.count ?? 0;

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className={shell("py-4")}>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Counts</h1>
            <p className="text-sm text-muted-foreground">
              {totalEvents > 0
                ? "Check the AI's counts, adjust any that are wrong (optional)"
                : "Run a deployment analysis to get started"}
            </p>
          </div>
        </div>
      </header>

      <main className={shell("py-8")}>
        <BrokenFolderBanner projectId={projectId!} />
        <WideModeContext.Provider value={wideMode}>
          <VerifyView projectId={projectId!} />
        </WideModeContext.Provider>
      </main>
    </div>
  );
}
