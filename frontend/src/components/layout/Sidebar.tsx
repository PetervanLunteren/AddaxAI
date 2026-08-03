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
import {
  NavLink,
  useLocation,
  useMatch,
  useParams,
  useResolvedPath,
} from "react-router-dom";
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
import { useBrokenDeployments } from "../../hooks/useBrokenDeployments";
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
  /**
   * Draw an attention dot on the item's icon. Rendered by `LeafNavLink`
   * only, so it does nothing on an item that has `children`.
   */
  dot?: boolean;
}

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

export function Sidebar({ collapsed, onToggleCollapsed }: SidebarProps) {
  const { projectId } = useParams<{ projectId: string }>();
  // Shares the deployments query the health toast already runs, so this
  // subscribes to a cache entry rather than adding a request.
  const brokenDeployments = useBrokenDeployments(projectId);

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
          label: "Class performance",
        },
      ],
    },
    { to: `/projects/${projectId}/export`, icon: Download, label: "Export" },
  ];

  // "Config" pages — set up monitoring locations and deployments.
  //
  // Deployments carries a dot when any of them have lost their files. The
  // startup toast says it loudly once; the dot is what keeps saying it,
  // and it points at the page that fixes it.
  const configItems: NavItem[] = [
    { to: `/projects/${projectId}/sites`, icon: MapPin, label: "Sites" },
    {
      to: `/projects/${projectId}/deployments`,
      icon: CardSim,
      label: "Deployments",
      dot: brokenDeployments.length > 0,
    },
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
      {/* Rail width is 72px, not the usual 64px, so that collapsing does
          not shift the icons sideways. Expanded, an icon centre sits at
          nav `p-4` + row `px-3` + half a 16px icon = 36px. A 72px rail
          with the same `p-4` centres its icons on 36px too, so they stay
          put. Keep the two in step if either padding changes. */}
      <aside
        className={cn(
          "fixed left-0 top-0 flex h-screen flex-col border-r bg-white transition-[width] duration-200",
          collapsed ? "w-[72px]" : "w-64",
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
        {/* Both variants are pinned to the same height so the nav below
            starts at the same y in either state and the icons do not
            walk up the screen when the rail collapses. */}
        {projectId &&
          (collapsed ? (
            <div className="flex h-[77px] shrink-0 items-center border-b px-4">
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
            <div className="flex h-[77px] shrink-0 flex-col justify-center border-b px-4">
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
//
// `h-9` rather than `py-2`: an expanded row is sized by its `text-sm`
// label (20px line box + padding), a collapsed row only holds the 16px
// icon. Padding alone would make the rail's rows 4px shorter each and
// walk every icon up the screen. A fixed height keeps both states on
// the same 40px pitch (36px row + `gap-1`).
const parentLinkClass = (isActive: boolean, collapsed = false) =>
  cn(
    "flex h-9 w-full items-center gap-3 rounded-lg text-sm font-medium transition-colors",
    collapsed ? "justify-center px-0" : "px-3",
    isActive
      ? "bg-primary/10 text-primary"
      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
  );

// The muted, non-navigating row style shared by the collapse toggle and
// the collapsed back-to-projects arrow.
const utilityRowClass = (collapsed = false) =>
  cn(
    "flex h-9 w-full items-center gap-3 rounded-lg text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground",
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

/**
 * Active-state check using the same primitives NavLink uses internally.
 *
 * We resolve `isActive` ourselves so `className` is always a plain
 * string. NavLink also accepts a `({ isActive }) => string` callback,
 * but that shape breaks the moment the link is wrapped in a Radix
 * `asChild` trigger: Slot clones the child and merges props before
 * NavLink ever runs, and its className merge is a string join, so the
 * callback gets stringified into the class attribute and every style
 * is lost.
 */
function useIsActive(to: string): boolean {
  const resolved = useResolvedPath(to);
  return useMatch({ path: resolved.pathname, end: false }) !== null;
}

// What the dot means, in words. The dot alone is colour-only, which says
// nothing to a screen reader and nothing in the collapsed rail's tooltip.
const DOT_LABEL = "needs attention";

function LeafNavLink({ item, collapsed }: { item: NavItem; collapsed: boolean }) {
  const isActive = useIsActive(item.to);
  const link = (
    <NavLink to={item.to} className={parentLinkClass(isActive, collapsed)}>
      {/* The dot is an absolute overlay, not a sibling of the icon, for two
          reasons: `parentLinkClass` sets no `relative`, so it needs its own
          positioned ancestor, and in the collapsed rail `justify-center`
          would centre an icon+dot pair and knock this one icon off the
          centreline the rail width exists to protect. */}
      <span className="relative shrink-0">
        <item.icon className="h-4 w-4" />
        {item.dot && (
          <span
            className="absolute -right-1 -top-1 h-2 w-2 rounded-full bg-red-600 ring-2 ring-white"
            aria-hidden
          />
        )}
      </span>
      {!collapsed && item.label}
      {item.dot && <span className="sr-only">, {DOT_LABEL}</span>}
    </NavLink>
  );
  if (!collapsed) return link;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{link}</TooltipTrigger>
      <TooltipContent side="right">
        {item.dot ? `${item.label} (${DOT_LABEL})` : item.label}
      </TooltipContent>
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
