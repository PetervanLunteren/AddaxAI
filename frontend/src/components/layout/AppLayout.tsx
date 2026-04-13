/**
 * App Layout with Sidebar
 */

import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { DeploymentHealthToast } from "./DeploymentHealthToast";

export function AppLayout() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="ml-64 flex-1 bg-gradient-to-br from-slate-50 to-slate-100">
        <Outlet />
      </main>
      <DeploymentHealthToast />
    </div>
  );
}
