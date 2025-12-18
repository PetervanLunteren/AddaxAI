/**
 * Main App component.
 *
 * Following DEVELOPERS.md principles:
 * - Simple, clear structure
 * - Type hints everywhere
 */

import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { queryClient } from "./lib/query-client";
import { AppLayout } from "./components/layout/AppLayout";
import { ProjectsPage } from "./pages/ProjectsPage";
import { AnalysesPage } from "./pages/AnalysesPage";
import ImagesPage from "./pages/ImagesPage";
import DashboardPage from "./pages/DashboardPage";

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
            <Route path="settings" element={<div className="p-8">Settings - Coming soon</div>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
