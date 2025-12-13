/**
 * Sidebar Navigation Component
 */

import { NavLink, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Camera, Play, BarChart3, Image, Settings } from "lucide-react";
import { projectsApi } from "../../api/projects";
import { cn } from "../../lib/utils";

interface NavItem {
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
}

export function Sidebar() {
  const { projectId } = useParams<{ projectId: string }>();

  const { data: project } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  const navItems: NavItem[] = [
    { to: `/projects/${projectId}/analyses`, icon: Play, label: "New Analysis" },
    { to: `/projects/${projectId}/dashboard`, icon: BarChart3, label: "Dashboard" },
    { to: `/projects/${projectId}/images`, icon: Image, label: "Images" },
    { to: `/projects/${projectId}/settings`, icon: Settings, label: "Settings" },
  ];

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 border-r bg-white">
      {/* Logo/Brand */}
      <div className="flex h-16 items-center gap-3 border-b px-6">
        <div className="rounded-lg bg-primary p-2">
          <Camera className="h-5 w-5 text-primary-foreground" />
        </div>
        <div>
          <h1 className="text-lg font-bold">AddaxAI</h1>
          <p className="text-xs text-muted-foreground">Camera Trap Analysis</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-1 p-4">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              )
            }
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </NavLink>
        ))}
      </nav>

      {/* Project Info at Bottom */}
      <div className="absolute bottom-0 left-0 right-0 border-t bg-muted/30 p-4">
        <p className="text-xs font-medium text-muted-foreground">Current Project</p>
        <p className="truncate text-sm font-semibold">
          {project?.name || "Loading..."}
        </p>
        <NavLink
          to="/projects"
          className="mt-2 text-xs text-primary hover:underline"
        >
          ← Back to Projects
        </NavLink>
      </div>
    </aside>
  );
}
