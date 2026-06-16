/**
 * Breadcrumb registry and useBreadcrumbs hook.
 *
 * The registry maps route patterns (the same shape react-router uses)
 * to a resolver that returns the breadcrumb trail for that route.
 * Dynamic segments like `:projectId` are resolved by reading the
 * project's name out of the TanStack Query cache; if the cache is
 * cold, the slug falls back to the id, which the next render replaces
 * once the query resolves.
 *
 * Adding a new breadcrumb: append a pattern + resolver to ROUTES.
 * Patterns are matched in order, first hit wins.
 *
 * Home and Setup intentionally have no entry — they are top-level
 * destinations where a "Home > Home" crumb adds nothing.
 */

import { useQuery } from "@tanstack/react-query";
import { matchPath, useLocation } from "react-router-dom";
import { projectsApi } from "../api/projects";
import { queryClient } from "./query-client";
import type { ProjectResponse } from "../api/types";

export interface BreadcrumbItem {
  label: string;
  /** Omit `to` on the final item — it renders as the current page. */
  to?: string;
}

interface BreadcrumbRoute {
  pattern: string;
  /** Builds the trail. Params come from react-router's matchPath. */
  resolve: (params: Record<string, string | undefined>) => BreadcrumbItem[];
}

function projectName(projectId: string | undefined): string {
  if (!projectId) return "";
  const cached = queryClient.getQueryData<ProjectResponse>([
    "projects",
    projectId,
  ]);
  return cached?.name ?? projectId;
}

const HOME: BreadcrumbItem = { label: "Home", to: "/" };
const RESEARCH_PROJECTS: BreadcrumbItem = {
  label: "Research projects",
  to: "/projects",
};

// Labels for insights sub-pages. Kept here (not derived from the
// sidebar) so the breadcrumb wording stays stable even if the sidebar
// label drifts; copy is part of the trail's value.
const INSIGHTS_LABELS: Record<string, string> = {
  map: "Map",
  timeline: "Deployment timeline",
  "activity-overlap": "Activity overlap",
  "confusion-matrix": "Confusion matrix",
  "per-class-performance": "Per-class performance",
};

// Labels for project sub-pages (single-segment ones).
const PROJECT_PAGE_LABELS: Record<string, string> = {
  dashboard: "Dashboard",
  process: "Process",
  labels: "Labels",
  counts: "Counts",
  sites: "Sites",
  deployments: "Deployments",
  export: "Export",
  settings: "Settings",
};

// Labels for folder-run stepper steps. Same step ids as the
// FolderRunStep type and the URL slugs.
const FOLDER_RUN_STEP_LABELS: Record<string, string> = {
  setup: "Setup",
  labels: "Labels",
  counts: "Counts",
  summary: "Summary",
  save: "Save",
};

const ROUTES: BreadcrumbRoute[] = [
  {
    // Insights leaf, e.g. /projects/:id/insights/activity-overlap
    pattern: "/projects/:projectId/insights/:insightSlug",
    resolve: ({ projectId, insightSlug }) => [
      HOME,
      RESEARCH_PROJECTS,
      {
        label: projectName(projectId),
        to: `/projects/${projectId}`,
      },
      { label: "Insights", to: `/projects/${projectId}/insights` },
      { label: INSIGHTS_LABELS[insightSlug ?? ""] ?? (insightSlug ?? "") },
    ],
  },
  {
    // Insights index (redirects to /insights/map, but a brief hop
    // through this URL is possible).
    pattern: "/projects/:projectId/insights",
    resolve: ({ projectId }) => [
      HOME,
      RESEARCH_PROJECTS,
      {
        label: projectName(projectId),
        to: `/projects/${projectId}`,
      },
      { label: "Insights" },
    ],
  },
  {
    // Project sub-page (dashboard, verify, sites, etc.)
    pattern: "/projects/:projectId/:pageSlug",
    resolve: ({ projectId, pageSlug }) => [
      HOME,
      RESEARCH_PROJECTS,
      {
        label: projectName(projectId),
        to: `/projects/${projectId}`,
      },
      { label: PROJECT_PAGE_LABELS[pageSlug ?? ""] ?? (pageSlug ?? "") },
    ],
  },
  {
    pattern: "/projects/:projectId",
    resolve: ({ projectId }) => [
      HOME,
      RESEARCH_PROJECTS,
      { label: projectName(projectId) },
    ],
  },
  {
    pattern: "/projects",
    resolve: () => [HOME, { label: "Research projects" }],
  },
  {
    pattern: "/folder-runs/new",
    resolve: () => [HOME, { label: "Analyse a folder" }],
  },
  {
    pattern: "/folder-runs/:runId/:stepSlug",
    resolve: ({ stepSlug }) => [
      HOME,
      { label: "Analyse a folder" },
      { label: FOLDER_RUN_STEP_LABELS[stepSlug ?? ""] ?? (stepSlug ?? "") },
    ],
  },
  {
    pattern: "/folder-runs/:runId",
    resolve: () => [HOME, { label: "Analyse a folder" }],
  },
];

/**
 * Compute breadcrumb items for the current URL.
 *
 * Returns an empty array on routes with no registered pattern
 * (including `/` and `/setup`); the Breadcrumbs component renders
 * nothing in that case.
 */
export function useBreadcrumbs(): BreadcrumbItem[] {
  const location = useLocation();

  // Find the matching route (matchPath is not a hook, so this loop is safe).
  let matched: {
    route: BreadcrumbRoute;
    params: Record<string, string | undefined>;
  } | null = null;
  for (const route of ROUTES) {
    const match = matchPath({ path: route.pattern, end: true }, location.pathname);
    if (match) {
      matched = {
        route,
        params: match.params as Record<string, string | undefined>,
      };
      break;
    }
  }

  // Subscribe to the project query so the crumb re-renders once the name
  // loads. Without this the resolver reads a cold cache on a direct
  // navigation (e.g. opening /projects/:id/settings in a fresh tab) and
  // shows the raw project id forever. The fetch dedupes with the page's
  // own identical ["projects", projectId] query. Only project routes carry
  // a projectId param (folder-run routes use runId and show a static label).
  const projectId = matched?.params?.projectId;
  useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
    staleTime: 5 * 60_000,
  });

  if (!matched) return [];
  return matched.route.resolve(matched.params);
}
