/**
 * Sidebar Navigation Component
 *
 * Two widths: full (labels + icons) and a collapsed icon-only rail.
 * Collapsed state is owned by AppLayout (the content margin tracks the
 * rail width) and passed in. In the rail, leaf items show their label
 * as a hover tooltip and the Insights group opens its children in a
 * hover flyout.
 */

import { useRef, useState } from "react";
import { NavLink, useLocation, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  CardSim,
  ChevronRight,
  Download,
  GanttChartSquare,
  Grid3x3,
  LayoutDashboard,
  Lightbulb,
  LineChart,
  Map,
  MapPin,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Sparkles,
  Table2,
  Tag,
  Tally5,
} from "lucide-react";
import { projectsApi } from "../../api/projects";
import { cn } from "../../lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";

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

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

export function Sidebar({ collapsed, onToggleCollapsed }: SidebarProps) {
  const { projectId } = useParams<{ projectId: string }>();

  const { data: project } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  // "Work" pages — daily-flow actions and views, end-to-end pipeline
  // from queueing an analysis through verifying and exporting results.
  const workItems: NavItem[] = [
    { to: `/projects/${projectId}/dashboard`, icon: LayoutDashboard, label: "Dashboard" },
    { to: `/projects/${projectId}/process`, icon: Sparkles, label: "Process" },
    { to: `/projects/${projectId}/labels`, icon: Tag, label: "Labels" },
    { to: `/projects/${projectId}/counts`, icon: Tally5, label: "Counts" },
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
      return <LeafNavLink key={item.to} item={item} collapsed={collapsed} />;
    }
    return <CollapsibleNavGroup key={item.to} item={item} collapsed={collapsed} />;
  };

  return (
    <TooltipProvider delayDuration={200}>
      <aside
        className={cn(
          "fixed left-0 top-0 flex h-screen flex-col border-r bg-white transition-[width] duration-200",
          collapsed ? "w-16" : "w-64",
        )}
      >
        {/* Logo/Brand */}
        {/* Height matches the page-header band in the content area (uniform
            py-4 + title + subtitle across pages) so the top divider runs
            straight across the sidebar and the content. */}
        <div
          className={cn(
            "flex h-[85px] items-center border-b",
            collapsed ? "justify-center px-0" : "px-4",
          )}
        >
          <NavLink to="/" aria-label="Home" title="Home">
            <img
              src={
                collapsed
                  ? "/branding/logo-mark.png"
                  : "/branding/logo-wordmark.png"
              }
              alt="AddaxAI"
              className={collapsed ? "h-9 w-9" : "h-14 w-auto"}
            />
          </NavLink>
        </div>

        {/* Current project + back-link. Sits between the logo and the nav
            (mirrors AddaxAI-Connect) so the workspace context lives at the
            top, where users expect it. In the rail the name has no room,
            so it collapses to just the back arrow with a tooltip. */}
        {projectId &&
          (collapsed ? (
            <div className="shrink-0 border-b p-4">
              <Tooltip>
                <TooltipTrigger asChild>
                  <NavLink
                    to="/projects"
                    aria-label="Back to projects"
                    className={utilityRowClass(true)}
                  >
                    <ArrowLeft className="h-4 w-4 shrink-0" />
                  </NavLink>
                </TooltipTrigger>
                <TooltipContent side="right">Back to projects</TooltipContent>
              </Tooltip>
            </div>
          ) : (
            <div className="shrink-0 border-b px-4 py-4">
              <div className="border-l-[3px] border-primary pl-3">
                <p className="truncate text-base font-bold leading-tight text-primary">
                  {project?.name || "Loading..."}
                </p>
                <NavLink
                  to="/projects"
                  className="mt-1 inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
                >
                  <ArrowLeft className="h-3 w-3" />
                  Back to projects
                </NavLink>
              </div>
            </div>
          ))}

        {/* Navigation */}
        <nav className="flex flex-1 flex-col gap-1 overflow-y-auto p-4">
          {workItems.map(renderNavLink)}
          <div className="my-2 border-t" />
          {configItems.map(renderNavLink)}
          <div className="my-2 border-t" />
          {utilityItems.map(renderNavLink)}
        </nav>

        {/* Collapse / expand toggle. */}
        <div className="shrink-0 border-t p-4">
          <SidebarToggle collapsed={collapsed} onToggle={onToggleCollapsed} />
        </div>
      </aside>
    </TooltipProvider>
  );
}

// ---------------------------------------------------------------------------
// Nav item building blocks
// ---------------------------------------------------------------------------

// Collapsed centers the icon on the rail's centerline (one clean
// vertical axis); expanded left-aligns the icon + label.
const parentLinkClass = (isActive: boolean, collapsed = false) =>
  cn(
    "flex w-full items-center gap-3 rounded-lg py-2 text-sm font-medium transition-colors",
    collapsed ? "justify-center px-0" : "px-3",
    isActive
      ? "bg-primary/10 text-primary"
      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
  );

// The muted, non-navigating row style shared by the collapse toggle and
// the collapsed back-to-projects arrow.
const utilityRowClass = (collapsed = false) =>
  cn(
    "flex w-full items-center gap-3 rounded-lg py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground",
    collapsed ? "justify-center px-0" : "px-3",
  );

const childLinkClass = (isActive: boolean) =>
  cn(
    "flex items-center gap-2 rounded-md py-1.5 pl-7 pr-3 text-sm transition-colors",
    isActive
      ? "bg-primary/10 text-primary font-medium"
      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
  );

// Child link inside the collapsed-rail flyout: no deep indent (the
// flyout is its own panel), icon + full label.
const flyoutChildClass = (isActive: boolean) =>
  cn(
    "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors",
    isActive
      ? "bg-primary/10 text-primary font-medium"
      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
  );

function LeafNavLink({ item, collapsed }: { item: NavItem; collapsed: boolean }) {
  const link = (
    <NavLink to={item.to} className={({ isActive }) => parentLinkClass(isActive, collapsed)}>
      <item.icon className="h-4 w-4 shrink-0" />
      {!collapsed && item.label}
    </NavLink>
  );
  if (!collapsed) return link;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{link}</TooltipTrigger>
      <TooltipContent side="right">{item.label}</TooltipContent>
    </Tooltip>
  );
}

/**
 * Parent nav item with a nested list of child items.
 *
 * Expanded: the parent row is a toggle button (never navigates on its
 * own) showing a chevron that rotates with the expanded state; children
 * render as NavLinks underneath. Expanded/collapsed of the group is
 * persisted to localStorage per parent path. Default on first visit:
 * expanded.
 *
 * Collapsed rail: renders as a single icon that opens the children in a
 * hover flyout instead (see CollapsedNavGroupFlyout).
 */
function CollapsibleNavGroup({ item, collapsed }: { item: NavItem; collapsed: boolean }) {
  const location = useLocation();
  const isActiveParent = location.pathname.startsWith(item.to);

  const storageKey = `addaxai:sidebar-expand:${item.to}`;
  const [expanded, setExpanded] = useState<boolean>(() => {
    const saved = localStorage.getItem(storageKey);
    return saved === null ? true : saved === "true";
  });

  if (collapsed) {
    return <CollapsedNavGroupFlyout item={item} isActiveParent={isActiveParent} />;
  }

  const toggle = () => {
    const next = !expanded;
    setExpanded(next);
    localStorage.setItem(storageKey, String(next));
  };

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

/**
 * Collapsed-rail rendering of a nav group: an icon button that opens its
 * children in a flyout to the right. Opens on hover (with a small close
 * delay so moving the cursor across the gap to the panel doesn't dismiss
 * it) and on click/focus for keyboard and touch.
 */
function CollapsedNavGroupFlyout({
  item,
  isActiveParent,
}: {
  item: NavItem;
  isActiveParent: boolean;
}) {
  const [open, setOpen] = useState(false);
  const closeTimer = useRef<number | undefined>(undefined);

  const openNow = () => {
    window.clearTimeout(closeTimer.current);
    setOpen(true);
  };
  const closeSoon = () => {
    window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => setOpen(false), 120);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={item.label}
          onMouseEnter={openNow}
          onMouseLeave={closeSoon}
          className={parentLinkClass(isActiveParent, true)}
        >
          <item.icon className="h-4 w-4 shrink-0" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        side="right"
        align="start"
        sideOffset={8}
        className="w-56 p-2"
        onMouseEnter={openNow}
        onMouseLeave={closeSoon}
      >
        <p className="px-2 pb-1 text-xs font-semibold text-muted-foreground">
          {item.label}
        </p>
        <div className="flex flex-col gap-0.5">
          {item.children?.map((child) => (
            <NavLink
              key={child.to}
              to={child.to}
              end
              onClick={() => setOpen(false)}
              className={({ isActive }) => flyoutChildClass(isActive)}
            >
              <child.icon className="h-4 w-4" />
              {child.label}
            </NavLink>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}

/**
 * The collapse/expand control at the foot of the sidebar. In the rail it
 * shows just the icon with a tooltip; expanded it shows an icon + label.
 */
function SidebarToggle({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: () => void;
}) {
  const button = (
    <button
      type="button"
      onClick={onToggle}
      aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      className={utilityRowClass(collapsed)}
    >
      {collapsed ? (
        <PanelLeftOpen className="h-4 w-4 shrink-0" />
      ) : (
        <PanelLeftClose className="h-4 w-4 shrink-0" />
      )}
      {!collapsed && "Collapse"}
    </button>
  );
  if (!collapsed) return button;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{button}</TooltipTrigger>
      <TooltipContent side="right">Expand sidebar</TooltipContent>
    </Tooltip>
  );
}
