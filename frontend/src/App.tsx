/**
 * Main App component.
 *
 * Following DEVELOPERS.md principles:
 * - Simple, clear structure
 * - Type hints everywhere
 */

import { useEffect, useState } from "react";
import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { X } from "lucide-react";
import { queryClient } from "./lib/query-client";
import { AppLayout } from "./components/layout/AppLayout";
import { ProjectsPage } from "./pages/ProjectsPage";
import { AnalysesPage } from "./pages/AnalysesPage";
import ImagesPage from "./pages/ImagesPage";
import DashboardPage from "./pages/DashboardPage";
import SettingsPage from "./pages/SettingsPage";
import { Button } from "./components/ui/button";
import { api } from "./lib/api-client";

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

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<ProjectsPage />} />

          {/* Project routes with sidebar */}
          <Route path="/projects/:projectId" element={<AppLayout />}>
            <Route index element={<Navigate to="analyses" replace />} />
            <Route path="analyses" element={<AnalysesPage />} />
            <Route path="dashboard" element={<DashboardPage />} />
            <Route path="images" element={<ImagesPage />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>
        </Routes>

        {/* Global toast notification */}
        <ModelUpdateToast />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
