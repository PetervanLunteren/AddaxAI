/**
 * Labels page — thin wrapper around `LabelsView`.
 *
 * One of the two verification pages (the other is `ObservationsPage`).
 * Labels is the per-detection label-cleanup workspace: fix the AI's
 * species labels via the crop grid, similarity sort, and cohort
 * relabel. Provides the canonical research-projects page chrome and
 * hands `projectId` to `LabelsView`.
 */

import { useParams } from "react-router-dom";
import { DiagnosticReportButton } from "../components/diagnostics/DiagnosticReportButton";
import { SpeciesNameToggle } from "../components/layout/SpeciesNameToggle";
import { LabelsView } from "../components/verify/LabelsView";

export default function LabelsPage() {
  const { projectId } = useParams<{ projectId: string }>();

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Labels</h1>
              <p className="text-sm text-muted-foreground">
                Fix the AI's species labels
              </p>
            </div>
            <div className="flex items-center gap-2">
              <SpeciesNameToggle />
              <DiagnosticReportButton />
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <LabelsView projectId={projectId!} />
      </main>
    </div>
  );
}
