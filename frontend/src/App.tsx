/**
 * Main App component.
 *
 * Following DEVELOPERS.md principles:
 * - Simple, clear structure
 * - Type hints everywhere
 */

import { useEffect, useState, type ReactNode } from "react";
import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useLocation,
} from "react-router-dom";
import { X } from "lucide-react";
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
import VerifyPage from "./pages/VerifyPage";
import ExportPage from "./pages/ExportPage";
import SettingsPage from "./pages/SettingsPage";
import SetupPage from "./pages/SetupPage";
import { SitesPage } from "./pages/SitesPage";
import { DeploymentsPage } from "./pages/DeploymentsPage";
import { Button } from "./components/ui/button";
import { CrashBanner } from "./components/layout/CrashBanner";
import { Toaster } from "./components/ui/sonner";
import { api } from "./lib/api-client";
import { setupApi } from "./api/setup";

interface ModelUpdate {
  model_id: string;
  friendly_name: string;
  emoji: string;
}

interface ModelUpdatesResponse {
  new_models: ModelUpdate[];
  checked_at: string | null;
}

function ModelUpdateToast() {
  const [showToast, setShowToast] = useState(false);

  // Fetch model updates once on app load
  const { data: updates } = useQuery({
    queryKey: ["model-updates"],
    queryFn: () => api.get<ModelUpdatesResponse>("/api/ml/updates"),
    staleTime: Infinity, // Only check once per session
  });

  // Show toast if new models found
  useEffect(() => {
    if (updates?.new_models && updates.new_models.length > 0) {
      setShowToast(true);
      // Auto-dismiss after 10 seconds
      const timer = setTimeout(() => setShowToast(false), 10000);
      return () => clearTimeout(timer);
    }
  }, [updates]);

  if (!showToast || !updates?.new_models || updates.new_models.length === 0) {
    return null;
  }

  return (
    <div
      className="fixed bottom-4 right-4 z-50 w-96 rounded-lg border bg-white p-4 shadow-lg animate-in slide-in-from-bottom-5"
      role="alert"
    >
      <div className="flex items-start gap-3">
        <div className="flex-1">
          <div className="font-semibold text-sm mb-2">
            New {updates.new_models.length === 1 ? "model" : "models"} available
          </div>
          <ul className="text-sm text-muted-foreground space-y-1">
            {updates.new_models.slice(0, 3).map((model) => (
              <li key={model.model_id}>{model.emoji} {model.friendly_name}</li>
            ))}
            {updates.new_models.length > 3 && (
              <li className="italic">+ {updates.new_models.length - 3} more</li>
            )}
          </ul>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowToast(false)}
          className="h-6 w-6 p-0 shrink-0"
        >
          <X className="h-4 w-4" />
        </Button>
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

  const { data, isLoading } = useQuery({
    queryKey: ["setup-status"],
    queryFn: setupApi.getStatus,
    refetchInterval: 5000,
  });

  // Don't render anything until we know the setup state. Avoids a flash
  // of the projects page before redirecting to /setup.
  if (isLoading || !data) {
    return null;
  }

  if (!data.ready && !onSetupRoute) {
    return <Navigate to="/setup" replace />;
  }

  if (data.ready && onSetupRoute) {
    return <Navigate to="/projects" replace />;
  }

  // Crash banner only renders once setup is ready: while the wizard is
  // open, the user has more pressing things to look at.
  return (
    <>
      {data.ready && <CrashBanner />}
      {children}
    </>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <SetupGate>
          <Routes>
            <Route path="/setup" element={<SetupPage />} />
            <Route path="/" element={<Navigate to="/projects" replace />} />
            <Route path="/projects" element={<ProjectsPage />} />

            {/* Project routes with sidebar */}
            <Route path="/projects/:projectId" element={<AppLayout />}>
              <Route index element={<Navigate to="analyses" replace />} />
              <Route path="analyses" element={<AnalysesPage />} />
              <Route path="verify" element={<VerifyPage />} />
              <Route path="review" element={<Navigate to="../verify" replace />} />
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

        {/* Global toast notifications */}
        <ModelUpdateToast />
        <Toaster />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
