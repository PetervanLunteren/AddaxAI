/**
 * Folder-run stepper layout.
 *
 * Shared chrome for all five folder-run steps: breadcrumbs, header,
 * step progress, and the Outlet that renders the current step. Lives
 * on two URL shapes:
 *
 * - `/folder-runs/new/*`: a brand-new run. The `:runId` is unknown
 *   until step 1 creates the project. The layout passes `runId =
 *   undefined` to child steps; only step 1 (Choose folder) handles
 *   this case, the others redirect to `/` if they land here without
 *   an id.
 * - `/folder-runs/:runId/*`: a created or resumed run. The layout
 *   fetches the run state and exposes it to children via the
 *   FolderRunContext.
 *
 * The current step is derived from the URL pathname rather than from
 * the backend's persisted step. The backend's step is used at the
 * top of the resume flow (when /folder-runs/:runId redirects to the
 * persisted step) but day-to-day stepper navigation is URL-driven.
 */

import { createContext, useContext } from "react";
import { Outlet, useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { HomeButton } from "../../components/layout/HomeButton";
import { StepProgress } from "../../components/folder-run/StepProgress";
import { useLabelColors } from "../../hooks/useLabelColors";
import { WideModeContext, useWideMode } from "../../components/verify/wide-mode";
import {
  folderRunsApi,
  type FolderRunResponse,
  type FolderRunStep,
} from "../../api/folder-runs";

interface FolderRunContextValue {
  runId: string | undefined;
  run: FolderRunResponse | undefined;
  isLoading: boolean;
}

const FolderRunContext = createContext<FolderRunContextValue | null>(null);

export function useFolderRun(): FolderRunContextValue {
  const value = useContext(FolderRunContext);
  if (!value) {
    throw new Error(
      "useFolderRun must be used inside a FolderRunLayout route",
    );
  }
  return value;
}

/** Step inferred from the current URL. Used to drive the visual
 * progress indicator. Defaults to "setup" because the brand-new path
 * (/folder-runs/new) renders the merged Setup page. */
function stepFromPath(pathname: string): FolderRunStep {
  if (pathname.endsWith("/labels")) return "labels";
  if (pathname.endsWith("/save")) return "save";
  return "setup";
}

export function FolderRunLayout() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  // A folder run is a project too: species colours for its Labels step.
  useLabelColors(runId);

  const { data: run, isLoading } = useQuery({
    queryKey: ["folder-run", runId],
    queryFn: () => folderRunsApi.get(runId!),
    enabled: !!runId,
  });

  const currentStep = stepFromPath(window.location.pathname);

  // URL-only navigation: clicking a chip just moves the user. The
  // backend step ("furthest reached") is not regressed when going
  // backward, and forward chip-nav past it is disabled by the chip
  // itself, so we don't need to PATCH the backend here.
  const handleStepClick = (step: FolderRunStep) => {
    if (!runId) return;
    navigate(`/folder-runs/${runId}/${step}`);
  };

  // One width for the whole flow. Every step shares the wide shell so
  // nothing jumps — not between steps, and not when the Edit grid or
  // Summary dashboard expand in place. The canvas steps embed the
  // verify grid / dashboard / save preview, which sit at this same 7xl
  // width in research-projects mode too. The Setup form keeps its two
  // equal-column rows here as well, just at the wider width.
  // Wide mode (Labels toolbar toggle) un-caps just the Edit/Labels step
  // to the full window width. Owned here, above the step, so un-capping
  // is a plain max-width change: no transform/full-bleed, which would
  // break the step's fixed/sticky bars (verify, relabel, prev/next).
  const wideMode = useWideMode();
  const mainMaxWidth =
    currentStep === "labels" && wideMode.wide ? "max-w-none" : "max-w-7xl";

  return (
    <FolderRunContext.Provider value={{ runId, run, isLoading }}>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
        {/* Standard page header (matches Projects and the project pages):
            logo + title + subtitle on the left. The logo links home; the
            stepper sits in its own band below. */}
        <header className="relative z-40 border-b bg-white/80 backdrop-blur-sm">
          <div className="mx-auto max-w-7xl px-4 py-3 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <HomeButton />
                <img
                  src="/branding/logo-mark.png"
                  alt="AddaxAI"
                  className="h-12 w-12 shrink-0"
                />
                <div>
                  <h1 className="text-2xl font-bold tracking-tight">
                    Analyse a folder
                  </h1>
                  <p className="text-sm text-muted-foreground">
                    Run a quick, one-off analysis on a folder
                  </p>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Stepper band — where you are in the run. Same width as the
            page content below so the indicator lines up with each step. */}
        <div className="border-b bg-white/60 backdrop-blur-sm">
          <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
            <StepProgress
              current={currentStep}
              furthest={run?.step}
              onStepClick={runId ? handleStepClick : undefined}
            />
          </div>
        </div>

        <main
          className={`mx-auto ${mainMaxWidth} px-4 py-8 sm:px-6 lg:px-8`}
        >
          <WideModeContext.Provider value={wideMode}>
            <Outlet />
          </WideModeContext.Provider>
        </main>
      </div>
    </FolderRunContext.Provider>
  );
}
