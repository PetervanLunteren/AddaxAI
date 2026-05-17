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
import { Outlet, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Breadcrumbs } from "../../components/layout/Breadcrumbs";
import { StepProgress } from "../../components/folder-run/StepProgress";
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
 * progress indicator. Defaults to "folder" so the new-run path
 * (which has no :stepSlug yet at /folder-runs/new) still shows the
 * first dot lit. */
function stepFromPath(pathname: string): FolderRunStep {
  if (pathname.endsWith("/model")) return "model";
  if (pathname.endsWith("/run")) return "run";
  if (pathname.endsWith("/overview")) return "overview";
  if (pathname.endsWith("/review")) return "review";
  if (pathname.endsWith("/save")) return "save";
  return "folder";
}

export function FolderRunLayout() {
  const { runId } = useParams<{ runId: string }>();

  const { data: run, isLoading } = useQuery({
    queryKey: ["folder-run", runId],
    queryFn: () => folderRunsApi.get(runId!),
    enabled: !!runId,
  });

  const currentStep = stepFromPath(window.location.pathname);

  // Steps 4 (Overview), 5 (Review) and 6 (Save) need horizontal
  // room: Overview embeds the Dashboard, Review embeds the verify
  // grid, Save renders a two-column options + preview layout. The
  // form-shaped earlier steps stay narrow.
  const mainMaxWidth =
    currentStep === "overview" ||
    currentStep === "review" ||
    currentStep === "save"
      ? "max-w-7xl"
      : "max-w-3xl";

  return (
    <FolderRunContext.Provider value={{ runId, run, isLoading }}>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
        <Breadcrumbs />
        <header className="border-b bg-white/80 backdrop-blur-sm">
          <div className="mx-auto max-w-5xl px-4 py-6 sm:px-6 lg:px-8">
            <div className="mb-6">
              <h1 className="text-2xl font-bold tracking-tight">
                Analyse a folder
              </h1>
              <p className="text-sm text-muted-foreground">
                Run AI on one folder and save results you can use right
                away.
              </p>
            </div>
            <StepProgress current={currentStep} />
          </div>
        </header>

        <main
          className={`mx-auto ${mainMaxWidth} px-4 py-8 sm:px-6 lg:px-8`}
        >
          <Outlet />
        </main>
      </div>
    </FolderRunContext.Provider>
  );
}
