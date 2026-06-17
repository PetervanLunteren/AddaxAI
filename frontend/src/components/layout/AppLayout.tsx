/**
 * App Layout with Sidebar
 */

import { Outlet, useParams } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { DeploymentHealthToast } from "./DeploymentHealthToast";
import { ModelSetupRequiredDialog } from "../models/ModelSetupRequiredDialog";

export function AppLayout() {
  const { projectId } = useParams<{ projectId: string }>();
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="ml-64 flex-1 bg-gradient-to-br from-slate-50 to-slate-100">
        {/* No breadcrumb here: the sidebar already shows the project
            name and a "Back to projects" link. The logo links Home. */}
        <Outlet />
      </main>
      <DeploymentHealthToast />
      {projectId && <ModelSetupRequiredDialog projectId={projectId} />}
    </div>
  );
}
