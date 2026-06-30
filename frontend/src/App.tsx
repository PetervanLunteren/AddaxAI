/**
 * Main App component.
 *
 * Following DEVELOPERS.md principles:
 * - Simple, clear structure
 * - Type hints everywhere
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useLocation,
  useParams,
} from "react-router-dom";
import { X } from "lucide-react";
import { toast } from "sonner";
import { queryClient } from "./lib/query-client";
import { AppLayout } from "./components/layout/AppLayout";
import { ProjectsPage } from "./pages/ProjectsPage";
import { AnalysesPage } from "./pages/AnalysesPage";
import DashboardPage from "./pages/DashboardPage";
import { MapPage } from "./pages/MapPage";
import { DeploymentTimelinePage } from "./pages/DeploymentTimelinePage";
import { ActivityOverlapPage } from "./pages/ActivityOverlapPage";
import { ConfusionMatrixPage } from "./pages/ConfusionMatrixPage";
import { PerClassPerformancePage } from "./pages/PerClassPerformancePage";
import LabelsPage from "./pages/LabelsPage";
import CountsPage from "./pages/CountsPage";
import ExportPage from "./pages/ExportPage";
import SettingsPage from "./pages/SettingsPage";
import SetupPage from "./pages/SetupPage";
import { SitesPage } from "./pages/SitesPage";
import { DeploymentsPage } from "./pages/DeploymentsPage";
import { HomePage } from "./pages/HomePage";
import { FolderRunLayout } from "./pages/folder-run/FolderRunLayout";
import { FolderRunModelStep } from "./pages/folder-run/FolderRunModelStep";
import { FolderRunOverviewStep } from "./pages/folder-run/FolderRunOverviewStep";
import { FolderRunLabelsStep } from "./pages/folder-run/FolderRunLabelsStep";
import { FolderRunCountsStep } from "./pages/folder-run/FolderRunCountsStep";
import { FolderRunSaveStep } from "./pages/folder-run/FolderRunSaveStep";
import { FolderRunResumeIndex } from "./pages/folder-run/FolderRunResumeIndex";
import { Button } from "./components/ui/button";
import { CrashBanner } from "./components/layout/CrashBanner";
import { MenuCommands } from "./components/layout/MenuCommands";
import { EnvRebuildButton } from "./components/layout/EnvRebuildButton";
import { Toaster } from "./components/ui/sonner";
import { api } from "./lib/api-client";
import { setupApi } from "./api/setup";
import { projectsApi } from "./api/projects";
import AboutPage from "./pages/AboutPage";

interface ModelUpdate {
  model_id: string;
  friendly_name: string;
  emoji: string;
}

interface DriftedEnv {
  env_name: string;
}

interface ModelUpdatesResponse {
  new_models: ModelUpdate[];
  refreshed_models?: ModelUpdate[];
  drifted_models?: ModelUpdate[];
  drifted_envs?: DriftedEnv[];
  checked_at: string | null;
}

function ModelUpdateToast() {
  const [dismissed, setDismissed] = useState(false);
  const [busyIds, setBusyIds] = useState<Set<string>>(new Set());

  // Fetch model updates once on app load.
  const { data: updates } = useQuery({
    queryKey: ["model-updates"],
    queryFn: () => api.get<ModelUpdatesResponse>("/api/ml/updates"),
    staleTime: Infinity,
  });

  const newModels = updates?.new_models ?? [];
  const driftedModels = updates?.drifted_models ?? [];
  const driftedEnvs = updates?.drifted_envs ?? [];
  const hasDrift = driftedModels.length > 0 || driftedEnvs.length > 0;
  const hasNew = newModels.length > 0;
  const hasAnything = hasNew || hasDrift;

  // Auto-dismiss after 10 s only when there's nothing actionable.
  // Drift entries have buttons the user is supposed to interact with;
  // those stay visible until the user dismisses them.
  useEffect(() => {
    if (!hasAnything || hasDrift) return;
    const timer = setTimeout(() => setDismissed(true), 10000);
    return () => clearTimeout(timer);
  }, [hasAnything, hasDrift]);

  if (dismissed || !hasAnything) {
    return null;
  }

  const markBusy = (id: string) => {
    setBusyIds((prev) => new Set(prev).add(id));
  };

  const handleRedownload = async (model: ModelUpdate) => {
    if (busyIds.has(model.model_id)) return;
    markBusy(model.model_id);
    try {
      await api.post(`/api/ml/models/${model.model_id}/redownload`, {});
      toast.success(`Re-downloading ${model.friendly_name}`, {
        description:
          "Running in the background. Restart the app once it's done.",
      });
    } catch (err) {
      toast.error(`Re-download failed for ${model.friendly_name}`, {
        description: err instanceof Error ? err.message : String(err),
      });
    }
  };

  // Env drift "Update now" goes through EnvRebuildButton, which uses the
  // backend's single guarded drift-rebuild path (install-env force_envs). The
  // env is global, so this needs no project context (the old version sent users
  // to a project's Settings page, which broke when no project was open).
  return (
    <div
      className="fixed bottom-4 right-4 z-50 w-96 rounded-lg border bg-white p-4 shadow-lg animate-in slide-in-from-bottom-5"
      role="alert"
    >
      <div className="flex items-start gap-3">
        <div className="flex-1 space-y-3">
          {hasNew && (
            <div>
              <div className="font-semibold text-sm mb-2">
                New {newModels.length === 1 ? "model" : "models"} available
              </div>
              <ul className="text-sm text-muted-foreground space-y-1">
                {newModels.slice(0, 3).map((model) => (
                  <li key={model.model_id}>
                    {model.emoji} {model.friendly_name}
                  </li>
                ))}
                {newModels.length > 3 && (
                  <li className="italic">+ {newModels.length - 3} more</li>
                )}
              </ul>
            </div>
          )}

          {driftedModels.length > 0 && (
            <div>
              <div className="font-semibold text-sm mb-2">
                Update{driftedModels.length === 1 ? "" : "s"} available
              </div>
              <ul className="text-sm text-muted-foreground space-y-1.5">
                {driftedModels.map((model) => (
                  <li
                    key={model.model_id}
                    className="flex items-center justify-between gap-2"
                  >
                    <span className="truncate">
                      {model.emoji} {model.friendly_name}
                    </span>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 px-2 text-xs"
                      disabled={busyIds.has(model.model_id)}
                      onClick={() => handleRedownload(model)}
                    >
                      {busyIds.has(model.model_id)
                        ? "Started"
                        : "Re-download"}
                    </Button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {driftedEnvs.length > 0 && (
            <div>
              <div className="font-semibold text-sm mb-2">
                Environment update{driftedEnvs.length === 1 ? "" : "s"}{" "}
                available
              </div>
              <p className="text-sm text-muted-foreground mb-2">
                The analysis environment ships a newer version than the
                one installed on this machine. Rebuild it to match this
                app version.
              </p>
              <ul className="text-xs text-muted-foreground space-y-0.5 mb-2">
                {driftedEnvs.map((env) => (
                  <li key={env.env_name} className="font-mono truncate">
                    env-{env.env_name}
                  </li>
                ))}
              </ul>
              <EnvRebuildButton
                envNames={driftedEnvs.map((env) => env.env_name)}
                onDone={() => setDismissed(true)}
              />
            </div>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setDismissed(true)}
          className="h-6 w-6 p-0 shrink-0"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

/**
 * Fallback screen rendered when the backend stops responding to the
 * setup-status poll. Without this, the SetupGate would just return
 * null forever, leaving the user staring at a blank window with no
 * indication that anything went wrong (e.g. backend crashed during
 * startup, alembic migration failed, port collision).
 */
function BackendDownScreen({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="max-w-md text-center space-y-4">
        <h1 className="text-2xl font-bold tracking-tight">
          Backend not responding
        </h1>
        <p className="text-sm text-muted-foreground">
          AddaxAI's backend stopped responding. This usually means it
          crashed during startup or hit a database migration error.
          Check the log at{" "}
          <code className="text-xs">~/AddaxAI/logs/backend.log</code> (or{" "}
          <code className="text-xs">
            %USERPROFILE%\AddaxAI\logs\backend.log
          </code>{" "}
          on Windows) and report the issue if it persists.
        </p>
        <Button onClick={onRetry}>Retry now</Button>
      </div>
    </div>
  );
}

/**
 * Full-app gate. While the first-run setup wizard hasn't completed, every
 * route except /setup redirects to /setup. Once setup is ready, /setup
 * itself redirects out. Status is polled cheaply (every 5s here; the
 * SetupPage itself polls more aggressively at 1.5s while the wizard is
 * open).
 */
function SetupGate({ children }: { children: ReactNode }) {
  const location = useLocation();
  const onSetupRoute = location.pathname.startsWith("/setup");
  // Sticky-ready flag. Once the backend reports `ready=true` at least
  // once in this session, we trust that setup is done and ignore later
  // transient `ready=false` polls. Real resets bounce the whole app
  // (Electron quit + relaunch), so the flag resets naturally and the
  // wizard still triggers on a genuinely-empty install. Without this,
  // any operation that briefly removes a default model weights file
  // (e.g. the drift toast's force-redownload) flips setup-status to
  // not-ready for the duration of the download and yanks the user back
  // to the wizard mid-task.
  const everReadyRef = useRef(false);

  const { data, isLoading, isError, errorUpdatedAt, dataUpdatedAt, refetch } = useQuery({
    queryKey: ["setup-status"],
    queryFn: setupApi.getStatus,
    refetchInterval: 5000,
  });

  if (data?.ready) {
    everReadyRef.current = true;
  }
  const effectivelyReady = Boolean(data?.ready || everReadyRef.current);

  // Detect a persistently-down backend. A single flaky fetch shouldn't
  // fire this: we require either the very first fetch to fail (no
  // dataUpdatedAt yet) or the error to have continued for >= 15s with
  // no successful poll in between. The refetchInterval of 5s drives
  // the re-renders that re-evaluate this on the wall clock.
  const now = Date.now();
  const hasRecentData = dataUpdatedAt > 0 && now - dataUpdatedAt < 15_000;
  const persistentError =
    isError && errorUpdatedAt > 0 && !hasRecentData;

  if (persistentError) {
    return <BackendDownScreen onRetry={() => refetch()} />;
  }

  // Don't render anything until we know the setup state. Avoids a flash
  // of the projects page before redirecting to /setup.
  if (isLoading || !data) {
    return null;
  }

  if (!effectivelyReady && !onSetupRoute) {
    return <Navigate to="/setup" replace />;
  }

  if (effectivelyReady && onSetupRoute) {
    return <Navigate to="/" replace />;
  }

  // Crash banner is intentionally suppressed for the current beta —
  // it fires too eagerly and creates noise. The detection logic
  // (sentinel files + last-launch snapshot in Electron, banner
  // component in React) is kept fully wired so we can flip the flag
  // back to true to re-enable without re-implementing anything.
  const SHOW_CRASH_BANNER = false;
  return (
    <>
      {effectivelyReady && SHOW_CRASH_BANNER && <CrashBanner />}
      {children}
    </>
  );
}

/**
 * Project-index redirect. Sends users with imported data straight to
 * the Dashboard; brand-new projects (no files yet) land on the
 * Analyses page so the next step is obvious. Renders nothing while
 * the stats query is in flight to avoid a Dashboard-then-Analyses
 * flash.
 */
function ProjectIndexRoute() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data, isLoading, isError } = useQuery({
    queryKey: ["project-stats", projectId],
    queryFn: () => projectsApi.getWithStats(projectId!),
    enabled: !!projectId,
  });

  if (isLoading) return null;
  const hasData = !isError && (data?.file_count ?? 0) > 0;
  return <Navigate to={hasData ? "dashboard" : "process"} replace />;
}

/**
 * Shows a toast when the Electron main process finishes auto-saving a
 * download to the Downloads folder (it no longer pops a Save dialog).
 * One toast per file, with a reveal-in-folder action. No-op in the
 * browser, where `electronAPI` is undefined and downloads behave normally.
 */
function DownloadCompleteToasts() {
  useEffect(() => {
    const api = window.electronAPI;
    if (!api?.onDownloadComplete) return;
    return api.onDownloadComplete(({ filename, path, success }) => {
      if (!success) {
        toast.error(`Could not save ${filename}`);
        return;
      }
      toast.success(`Saved ${filename} to Downloads`, {
        action: {
          label: "Show in folder",
          onClick: () => void api.showItemInFolder(path),
        },
      });
    });
  }, []);
  return null;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <SetupGate>
          <Routes>
            <Route path="/setup" element={<SetupPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/" element={<HomePage />} />

            {/* New folder-run: project id does not exist yet. The
                Setup page handles the no-runId case by creating a
                project on Start analysis. */}
            <Route path="/folder-runs/new" element={<FolderRunLayout />}>
              <Route index element={<FolderRunModelStep />} />
            </Route>

            {/* Existing / resumed folder run. Hitting the bare id
                redirects to the persisted step. */}
            <Route path="/folder-runs/:runId" element={<FolderRunLayout />}>
              <Route index element={<FolderRunResumeIndex />} />
              <Route path="setup" element={<FolderRunModelStep />} />
              <Route path="summary" element={<FolderRunOverviewStep />} />
              <Route path="labels" element={<FolderRunLabelsStep />} />
              <Route path="counts" element={<FolderRunCountsStep />} />
              <Route path="save" element={<FolderRunSaveStep />} />
            </Route>

            <Route path="/projects" element={<ProjectsPage />} />

            {/* Project routes with sidebar */}
            <Route path="/projects/:projectId" element={<AppLayout />}>
              <Route index element={<ProjectIndexRoute />} />
              <Route path="process" element={<AnalysesPage />} />
              <Route path="labels" element={<LabelsPage />} />
              <Route path="counts" element={<CountsPage />} />
              <Route path="dashboard" element={<DashboardPage />} />
              <Route path="insights" element={<Navigate to="map" replace />} />
              <Route path="insights/map" element={<MapPage />} />
              <Route path="insights/timeline" element={<DeploymentTimelinePage />} />
              <Route path="insights/activity-overlap" element={<ActivityOverlapPage />} />
              <Route path="insights/confusion-matrix" element={<ConfusionMatrixPage />} />
              <Route path="insights/per-class-performance" element={<PerClassPerformancePage />} />
              <Route path="sites" element={<SitesPage />} />
              <Route path="deployments" element={<DeploymentsPage />} />
              <Route path="export" element={<ExportPage />} />
              <Route path="settings" element={<SettingsPage />} />
            </Route>
          </Routes>
        </SetupGate>

        {/* Native menu command bridge (Electron only) + global toasts */}
        <MenuCommands />
        <ModelUpdateToast />
        <DownloadCompleteToasts />
        <Toaster />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
