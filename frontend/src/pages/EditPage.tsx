/**
 * Edit page — thin wrapper around `VerifyView`.
 *
 * Provides the canonical research-projects page chrome (full-viewport
 * background, top header with title + subtitle + DiagnosticReportButton,
 * `<main>` container at `max-w-7xl`) and hands `projectId` from the
 * URL to `VerifyView`, which owns all the page body, state, queries,
 * and modals.
 *
 * The same `VerifyView` is also mounted inline inside the folder-run
 * stepper Edit step (`FolderRunEditStep`). Keeping the body in one
 * component means filter logic, sort modes, modals, and pagination
 * behave identically across the two flows. (The shared component keeps
 * the `Verify*` name because it operates on the `verified` data; only
 * the page/route is renamed to "Edit".)
 */

import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { eventsApi } from "../api/events";
import { DiagnosticReportButton } from "../components/diagnostics/DiagnosticReportButton";
import { VerifyView } from "../components/verify/VerifyView";

export default function EditPage() {
  const { projectId } = useParams<{ projectId: string }>();

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
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Edit</h1>
              <p className="text-sm text-muted-foreground">
                {totalEvents > 0
                  ? "Correct the AI's predictions"
                  : "Run a deployment analysis to get started"}
              </p>
            </div>
            <DiagnosticReportButton />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <VerifyView projectId={projectId!} />
      </main>
    </div>
  );
}
