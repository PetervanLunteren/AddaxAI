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
import { SitesPage } from "./pages/SitesPage";
import { DeploymentsPage } from "./pages/DeploymentsPage";

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<ProjectsPage />} />

          {/* Project routes with sidebar */}
          <Route path="/projects/:projectId" element={<AppLayout />}>
            <Route index element={<Navigate to="sites" replace />} />
            <Route path="sites" element={<SitesPage />} />
            <Route path="files" element={<div className="p-8">Files page - Coming soon</div>} />
            <Route path="images" element={<div className="p-8">Images page - Coming soon</div>} />
            <Route path="deployments" element={<DeploymentsPage />} />
            <Route path="detections" element={<div className="p-8">Detections page - Coming soon</div>} />
            <Route path="settings" element={<div className="p-8">Settings page - Coming soon</div>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
