/**
 * App Layout with Sidebar
 */

import { useState } from "react";
import { Outlet, useParams } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { SidebarCollapseContext } from "./sidebar-context";
import { DeploymentHealthToast } from "./DeploymentHealthToast";
import { ModelSetupRequiredDialog } from "../models/ModelSetupRequiredDialog";
import { cn } from "../../lib/utils";
import { useLabelColors } from "../../hooks/useLabelColors";

const COLLAPSE_KEY = "addaxai:sidebar-collapsed";

export function AppLayout() {
  const { projectId } = useParams<{ projectId: string }>();
  // Species colours for every page under this project.
  useLabelColors(projectId);
  // Collapsed = icon-only rail. Persisted so the choice sticks across
  // reloads. Owned here (not in Sidebar) because the content margin has
  // to track the rail width.
  const [collapsed, setCollapsed] = useState<boolean>(
    () => localStorage.getItem(COLLAPSE_KEY) === "true",
  );
  const toggleCollapsed = () => {
    setCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem(COLLAPSE_KEY, String(next));
      return next;
    });
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar collapsed={collapsed} onToggleCollapsed={toggleCollapsed} />
      {/* Margins track the rail widths in Sidebar.tsx. 72px, not 64px,
          so the collapsed rail's icons keep the same centre as the
          expanded one's. */}
      <main
        className={cn(
          "flex-1 bg-gradient-to-br from-slate-50 to-slate-100 transition-[margin] duration-200",
          collapsed ? "ml-[72px]" : "ml-64",
        )}
      >
        {/* No breadcrumb here: the sidebar already shows the project
            name and a "Back to projects" link. The logo links Home. */}
        <SidebarCollapseContext.Provider value={collapsed}>
          <Outlet />
        </SidebarCollapseContext.Provider>
      </main>
      <DeploymentHealthToast />
      {projectId && <ModelSetupRequiredDialog projectId={projectId} />}
    </div>
  );
}
