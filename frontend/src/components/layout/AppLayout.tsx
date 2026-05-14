/**
 * App Layout with Sidebar
 */

import { Outlet, useParams } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Breadcrumbs } from "./Breadcrumbs";
import { DeploymentHealthToast } from "./DeploymentHealthToast";
import { ModelSetupRequiredDialog } from "../models/ModelSetupRequiredDialog";

export function AppLayout() {
  const { projectId } = useParams<{ projectId: string }>();
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="ml-64 flex-1 bg-gradient-to-br from-slate-50 to-slate-100">
        <Breadcrumbs />
        <Outlet />
      </main>
      <DeploymentHealthToast />
      {projectId && <ModelSetupRequiredDialog projectId={projectId} />}
    </div>
  );
}
