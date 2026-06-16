/**
 * Sidebar Navigation Component
 */

import { useState } from "react";
import { NavLink, useLocation, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  CardSim,
  ChevronRight,
  Download,
  GanttChartSquare,
  Grid3x3,
  LayoutDashboard,
  Layers,
  Lightbulb,
  LineChart,
  Map,
  MapPin,
  Pencil,
  Play,
  Settings,
  Table2,
} from "lucide-react";
import { projectsApi } from "../../api/projects";
import { cn } from "../../lib/utils";

interface NavItem {
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  /**
   * Optional children that render indented under the parent item.
   * When a parent has children, its path should be a route that
   * redirects to the first child (so clicking the parent still
   * "does something"). We don't bother with collapse/expand state
   * for v1: with one child per parent the full subtree is fine
   * to show. Add collapse later if a parent ever grows past ~4
   * children.
   */
  children?: NavItem[];
}

export function Sidebar() {
  const { projectId } = useParams<{ projectId: string }>();

  const { data: project } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  // "Work" pages — daily-flow actions and views, end-to-end pipeline
  // from queueing an analysis through verifying and exporting results.
  const workItems: NavItem[] = [
    { to: `/projects/${projectId}/process`, icon: Play, label: "Process" },
    { to: `/projects/${projectId}/labels`, icon: Pencil, label: "Labels" },
    { to: `/projects/${projectId}/counts`, icon: Layers, label: "Counts" },
    { to: `/projects/${projectId}/dashboard`, icon: LayoutDashboard, label: "Dashboard" },
    {
      to: `/projects/${projectId}/insights`,
      icon: Lightbulb,
      label: "Insights",
      children: [
        {
          to: `/projects/${projectId}/insights/map`,
          icon: Map,
          label: "Map",
        },
        {
          to: `/projects/${projectId}/insights/timeline`,
          icon: GanttChartSquare,
          label: "Deployment timeline",
        },
        {
          to: `/projects/${projectId}/insights/activity-overlap`,
          icon: LineChart,
          label: "Activity overlap",
        },
        {
          to: `/projects/${projectId}/insights/confusion-matrix`,
          icon: Grid3x3,
          label: "Confusion matrix",
        },
        {
          to: `/projects/${projectId}/insights/per-class-performance`,
          icon: Table2,
          label: "Per-class performance",
        },
      ],
    },
    { to: `/projects/${projectId}/export`, icon: Download, label: "Export" },
  ];

  // "Config" pages — set up monitoring locations and deployments.
  const configItems: NavItem[] = [
    { to: `/projects/${projectId}/sites`, icon: MapPin, label: "Sites" },
    { to: `/projects/${projectId}/deployments`, icon: CardSim, label: "Deployments" },
  ];

  // Utility pages — settings only for now.
  const utilityItems: NavItem[] = [
    { to: `/projects/${projectId}/settings`, icon: Settings, label: "Settings" },
  ];

  const renderNavLink = (item: NavItem) => {
    if (!item.children || item.children.length === 0) {
      return <LeafNavLink key={item.to} item={item} />;
    }
    return <CollapsibleNavGroup key={item.to} item={item} />;
  };

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 border-r bg-white">
      {/* Logo/Brand */}
      <div className="flex h-16 items-center border-b px-4">
        <img
          src="/branding/logo-wordmark.png"
          alt="AddaxAI"
          className="h-10 w-auto"
        />
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-1 p-4">
        {workItems.map(renderNavLink)}
        <div className="my-2 border-t" />
        {configItems.map(renderNavLink)}
        <div className="my-2 border-t" />
        {utilityItems.map(renderNavLink)}
      </nav>

      {/* Project Info at Bottom */}
      <div className="absolute bottom-0 left-0 right-0 border-t bg-muted/30 p-4">
        <p className="text-xs font-medium text-muted-foreground">Current project</p>
        <p className="truncate text-sm font-semibold">
          {project?.name || "Loading..."}
        </p>
        <NavLink
          to="/projects"
          className="mt-2 text-xs text-primary hover:underline"
        >
          ← Back to research projects
        </NavLink>
      </div>
    </aside>
  );
}

// ---------------------------------------------------------------------------
// Nav item building blocks
// ---------------------------------------------------------------------------

const parentLinkClass = (isActive: boolean) =>
  cn(
    "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
    isActive
      ? "bg-primary/10 text-primary"
      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
  );

const childLinkClass = (isActive: boolean) =>
  cn(
    "flex items-center gap-2 rounded-md py-1.5 pl-7 pr-3 text-sm transition-colors",
    isActive
      ? "bg-primary/10 text-primary font-medium"
      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
  );

function LeafNavLink({ item }: { item: NavItem }) {
  return (
    <NavLink to={item.to} className={({ isActive }) => parentLinkClass(isActive)}>
      <item.icon className="h-4 w-4" />
      {item.label}
    </NavLink>
  );
}

/**
 * Parent nav item with a nested list of child items. The parent row
 * is a toggle button (never navigates on its own) showing a chevron
 * that rotates with the expanded state. Children render as NavLinks
 * underneath when expanded.
 *
 * Expanded state is persisted to localStorage per parent path so a
 * user's show/hide preference sticks across reloads and sessions.
 * Default on first visit: expanded (so new users immediately see
 * the available children instead of discovering them via a click).
 */
function CollapsibleNavGroup({ item }: { item: NavItem }) {
  const location = useLocation();
  const storageKey = `addaxai:sidebar-expand:${item.to}`;
  const [expanded, setExpanded] = useState<boolean>(() => {
    const saved = localStorage.getItem(storageKey);
    return saved === null ? true : saved === "true";
  });

  const toggle = () => {
    const next = !expanded;
    setExpanded(next);
    localStorage.setItem(storageKey, String(next));
  };

  const isActiveParent = location.pathname.startsWith(item.to);

  return (
    <div>
      <button
        type="button"
        onClick={toggle}
        aria-expanded={expanded}
        className={parentLinkClass(isActiveParent)}
      >
        <item.icon className="h-4 w-4" />
        <span className="flex-1 text-left">{item.label}</span>
        <ChevronRight
          className={cn(
            "h-4 w-4 shrink-0 opacity-60 transition-transform",
            expanded && "rotate-90",
          )}
        />
      </button>
      {expanded && (
        <div className="mt-1 flex flex-col gap-0.5">
          {item.children?.map((child) => (
            <NavLink
              key={child.to}
              to={child.to}
              end
              className={({ isActive }) => childLinkClass(isActive)}
            >
              <child.icon className="h-3.5 w-3.5" />
              {child.label}
            </NavLink>
          ))}
        </div>
      )}
    </div>
  );
}
