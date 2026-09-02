/**
 * Labels page — thin wrapper around `LabelsView`.
 *
 * One of the two verification pages (the other is `ObservationsPage`).
 * Labels is the per-detection label-cleanup workspace: fix the AI's
 * species labels via the crop grid, similarity sort, and cohort
 * relabel. Provides the canonical research-projects page chrome and
 * hands `projectId` to `LabelsView`.
 */

import { useNavigate, useParams } from "react-router-dom";
import { BrokenFolderBanner } from "../components/deployments/BrokenFolderBanner";
import { LabelsView } from "../components/verify/LabelsView";
import { WideModeContext, useWideMode } from "../components/verify/wide-mode";
import { cn } from "../lib/utils";

export default function LabelsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const wideMode = useWideMode();

  // Wide mode drops the readable-width cap so the grid fills the content
  // area; the normal view keeps the app-wide max-w-7xl. The toggle lives
  // in the view's toolbar (LabelsTab), so it shows in folder-run too.
  const shell = (vertical: string) =>
    cn(
      "px-4 sm:px-6 lg:px-8",
      vertical,
      wideMode.wide ? "w-full" : "mx-auto max-w-7xl",
    );

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className={shell("py-4")}>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Labels</h1>
            <p className="text-sm text-muted-foreground">
              Check the AI's labels, fix any that are wrong (optional)
            </p>
          </div>
        </div>
      </header>

      <main className={shell("py-8")}>
        <BrokenFolderBanner projectId={projectId!} />
        <WideModeContext.Provider value={wideMode}>
          <LabelsView
            projectId={projectId!}
            // The Files tab note offers to open the detection threshold.
            // In projects mode that is the settings page.
            onEditThreshold={() => navigate(`/projects/${projectId}/settings`)}
          />
        </WideModeContext.Provider>
      </main>
    </div>
  );
}
