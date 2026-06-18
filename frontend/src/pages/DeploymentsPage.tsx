/**
 * Deployments metadata management page.
 *
 * Table-based view of all deployments in a project with sorting, filters, and edit actions.
 */

import { useState, useMemo, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams, useSearchParams, useNavigate } from "react-router-dom";
import { Search, MoreVertical, Pencil, Trash2, ArrowUp, ArrowDown, Tent, AlertTriangle, Plus, Info, Scissors } from "lucide-react";
import { deploymentsApi } from "../api/deployments";
import { sitesApi } from "../api/sites";
import { basename } from "../lib/path-utils";
import type { DeploymentResponse, DeploymentStatsOnly } from "../api/types";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { FilterBar, type FilterFieldDef, type FilterValues } from "../components/ui/filter-bar";
import { TagPills } from "../components/ui/tag-pills";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../components/ui/tooltip";
import { filtersFromSearchParams, filtersToSearchParams, NO_SITE_SENTINEL, type FilterSchema } from "../lib/filter-url";
import { formatShortDate } from "../lib/datetime";
import { DeploymentInfoSheet } from "../components/deployments/DeploymentInfoSheet";
import { EditDeploymentDialog } from "../components/deployments/EditDeploymentDialog";
import {
  DeleteDeploymentDialog,
  type DeleteDeploymentTarget,
} from "../components/deployments/DeleteDeploymentDialog";
import {
  SplitDeploymentDialog,
  type SplitDeploymentTarget,
} from "../components/deployments/SplitDeploymentDialog";
import { RelinkGroupBanner } from "../components/deployments/RelinkGroupBanner";

type SortField =
  | "folder"
  | "site_name"
  | "period"
  | "file_count"
  | "notes"
  | "tag_count";
type SortDir = "asc" | "desc";

/**
 * Leaf segment of a folder path. Used as a deployment's primary
 * identifier in the table ("deployment_001" out of
 * `/data/project/site/deployment_001`, also `C:\...\deployment_001`).
 * Falls back to a single dash when folder_path is null (legacy /
 * unlinked deployments).
 */
function folderBasename(path: string | null): string {
  return basename(path) || "-";
}

/**
 * Render a deployment's period as a single readable string.
 *
 * Null end date (no files loaded yet) → just the start date. Same-day
 * deployments collapse to one date plus "(1 day)". Multi-day
 * deployments show start-end with an inclusive day count.
 */
function formatPeriod(start: string, end: string | null): string {
  const s = formatShortDate(start);
  if (!end) return s;
  if (start === end) return `${s} (1 day)`;
  const days =
    Math.round(
      (new Date(end).getTime() - new Date(start).getTime()) / 86_400_000,
    ) + 1;
  return `${s} - ${formatShortDate(end)} (${days} days)`;
}

const FILTER_SCHEMA: FilterSchema = {
  search: "string",
  site_ids: "string[]",
  date_from: "date",
  date_to: "date",
  tag_keys: "string[]",
};

interface DeploymentRow extends DeploymentResponse {
  /** Rendered site name, `-` when the deployment has no site. */
  site_name: string;
  /** Leaf of folder_path, used as the primary table identifier. */
  folder_basename: string;
  file_count: number;
  event_count: number;
  detection_count: number;
}

function SortIcon({ field, sortField, sortDir }: { field: SortField; sortField: SortField; sortDir: SortDir }) {
  if (field !== sortField) return null;
  return sortDir === "asc"
    ? <ArrowUp className="ml-1 inline h-3.5 w-3.5" />
    : <ArrowDown className="ml-1 inline h-3.5 w-3.5" />;
}


export function DeploymentsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const filters = useMemo(
    () => filtersFromSearchParams(searchParams, FILTER_SCHEMA),
    [searchParams]
  );
  const search = (filters.search as string | undefined) ?? "";
  const siteIds = (filters.site_ids as string[] | undefined) ?? [];
  const dateFrom = (filters.date_from as string | undefined) ?? "";
  const dateTo = (filters.date_to as string | undefined) ?? "";
  const tagKeysFilter = (filters.tag_keys as string[] | undefined) ?? [];

  const [sortField, setSortField] = useState<SortField>("period");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [editingDeployment, setEditingDeployment] = useState<DeploymentResponse | null>(null);
  const [deletingDeployment, setDeletingDeployment] = useState<DeleteDeploymentTarget | null>(null);
  const [splittingDeployment, setSplittingDeployment] = useState<SplitDeploymentTarget | null>(null);
  const [infoDeploymentId, setInfoDeploymentId] = useState<string | null>(null);

  // Deep link support: `?info=<id>` auto-opens the info sheet for a
  // specific deployment (used by the queue's "already a deployment"
  // banner). Strip the param after reading so back/refresh doesn't
  // repeatedly re-open the sheet.
  useEffect(() => {
    const id = searchParams.get("info");
    if (id) {
      setInfoDeploymentId(id);
      const next = new URLSearchParams(searchParams);
      next.delete("info");
      setSearchParams(next, { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFilterChange = (next: FilterValues) => {
    setSearchParams(filtersToSearchParams(next, FILTER_SCHEMA));
  };

  const { data: deployments, isLoading: deploymentsLoading } = useQuery({
    queryKey: ["deployments", projectId],
    queryFn: () => deploymentsApi.list({ projectId: projectId! }),
    enabled: !!projectId,
  });

  const { data: bulkStats } = useQuery({
    queryKey: ["deployment-stats", projectId],
    queryFn: () => deploymentsApi.getBulkStats(projectId!),
    enabled: !!projectId,
  });

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId,
  });

  const siteMap = useMemo(() => {
    const map = new Map<string, string>();
    if (sites) {
      for (const s of sites) {
        map.set(s.id, s.name);
      }
    }
    return map;
  }, [sites]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  };

  // Merge deployments with site names and stats. Renders a dash when
  // the deployment has no site assigned.
  const rows: DeploymentRow[] = useMemo(() => {
    if (!deployments) return [];
    return deployments.map((d) => {
      const stats: DeploymentStatsOnly = bulkStats?.[d.id] ?? {
        file_count: 0,
        event_count: 0,
        detection_count: 0,
      };
      const siteName = d.site_id ? (siteMap.get(d.site_id) ?? "Unknown") : "(no site)";
      return {
        ...d,
        site_name: siteName,
        folder_basename: folderBasename(d.folder_path),
        file_count: stats.file_count,
        event_count: stats.event_count,
        detection_count: stats.detection_count,
      };
    });
  }, [deployments, bulkStats, siteMap]);

  // Filter and sort
  const filtered = useMemo(() => {
    let result = [...rows];

    // Site filter (multi-select). The reserved NO_SITE_SENTINEL token
    // matches deployments with site_id IS NULL.
    if (siteIds.length > 0) {
      const wantsNoSite = siteIds.includes(NO_SITE_SENTINEL);
      const realIds = new Set(siteIds.filter((id) => id !== NO_SITE_SENTINEL));
      result = result.filter((d) => {
        if (wantsNoSite && d.site_id == null) return true;
        if (d.site_id != null && realIds.has(d.site_id)) return true;
        return false;
      });
    }

    // Date range filter
    if (dateFrom) {
      result = result.filter((d) => d.start_date_local >= dateFrom);
    }
    if (dateTo) {
      result = result.filter((d) => d.start_date_local <= dateTo);
    }

    // Text search
    if (search) {
      const q = search.toLowerCase();
      result = result.filter((d) => {
        if (d.folder_basename.toLowerCase().includes(q)) return true;
        if (d.site_name.toLowerCase().includes(q)) return true;
        if (d.id.toLowerCase().includes(q)) return true;
        if (d.notes && d.notes.toLowerCase().includes(q)) return true;
        for (const [k, v] of Object.entries(d.tags ?? {})) {
          if (k.toLowerCase().includes(q) || v.toLowerCase().includes(q)) {
            return true;
          }
        }
        return false;
      });
    }

    // Tag keys multi-select (deployment has at least one of the selected keys)
    if (tagKeysFilter.length > 0) {
      const set = new Set(tagKeysFilter);
      result = result.filter((d) => {
        for (const k of Object.keys(d.tags ?? {})) {
          if (set.has(k)) return true;
        }
        return false;
      });
    }

    // Sort. `period` sorts by start date (ISO strings lex-sort chronologically).
    const getSortValue = (
      d: DeploymentRow,
      field: SortField,
    ): string | number | null => {
      if (field === "tag_count") return Object.keys(d.tags ?? {}).length;
      if (field === "period") return d.start_date_local;
      if (field === "folder") return d.folder_basename;
      return d[field];
    };

    result.sort((a, b) => {
      let aVal = getSortValue(a, sortField);
      let bVal = getSortValue(b, sortField);

      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;

      if (typeof aVal === "string") aVal = aVal.toLowerCase();
      if (typeof bVal === "string") bVal = (bVal as string).toLowerCase();

      if (aVal < bVal) return sortDir === "asc" ? -1 : 1;
      if (aVal > bVal) return sortDir === "asc" ? 1 : -1;
      return 0;
    });

    return result;
  }, [rows, siteIds, dateFrom, dateTo, search, tagKeysFilter, sortField, sortDir]);

  if (!projectId) {
    return <div>Project ID missing</div>;
  }

  const isLoading = deploymentsLoading;
  const headClass = "cursor-pointer select-none hover:text-foreground";

  // Broken deployments grouped by their deepest missing ancestor.
  // Grouping happens on the backend because only the filesystem knows
  // which ancestor was actually renamed — see /api/deployments/group-broken.
  const brokenItems = useMemo(
    () =>
      (deployments ?? [])
        .filter((d) => d.folder_status === "needs_relink" && d.folder_path)
        .map((d) => ({ id: d.id, folder_path: d.folder_path! })),
    [deployments]
  );

  const brokenGroupsQuery = useQuery({
    queryKey: ["broken-groups", projectId, brokenItems],
    queryFn: () => deploymentsApi.groupBroken({ items: brokenItems }),
    enabled: brokenItems.length > 0,
    staleTime: 30_000,
  });

  const brokenGroups = brokenGroupsQuery.data?.groups ?? [];

  const deploymentsById = useMemo(() => {
    const map = new Map<string, DeploymentResponse>();
    for (const d of deployments ?? []) map.set(d.id, d);
    return map;
  }, [deployments]);

  // Distinct tag keys across all loaded deployments, sorted alphabetically.
  const tagKeyOptions = (() => {
    const set = new Set<string>();
    for (const d of deployments ?? []) {
      for (const k of Object.keys(d.tags ?? {})) {
        if (k.trim()) set.add(k);
      }
    }
    return Array.from(set)
      .sort((a, b) => a.localeCompare(b))
      .map((k) => ({ value: k, label: k }));
  })();

  const filterFields: FilterFieldDef[] = [
    { kind: "search", key: "search", label: "Search", placeholder: "Search names, IDs, etc..." },
    {
      kind: "multi-select",
      key: "site_ids",
      label: "Sites",
      options: [
        { value: NO_SITE_SENTINEL, label: "(no site)" },
        ...(sites ?? []).map((s) => ({ value: s.id, label: s.name })),
      ],
      placeholder: "All sites",
      summary: (n) => `${n} site${n > 1 ? "s" : ""}`,
    },
    {
      kind: "date_range",
      key: "date_from",
      toKey: "date_to",
      label: "Date range",
    },
    {
      kind: "multi-select",
      key: "tag_keys",
      label: "Tag keys",
      options: tagKeyOptions,
      placeholder: "All tag keys",
      summary: (n) => `${n} tag${n > 1 ? "s" : ""}`,
    },
  ];

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Deployments</h1>
              <p className="text-sm text-muted-foreground">
                Camera deployment periods across all sites
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={() => navigate(`/projects/${projectId}/process`)}>
                <Plus className="mr-2 h-4 w-4" />
                New deployment
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Filter bar */}
        <div className="mb-6">
          <FilterBar
            value={filters}
            onChange={handleFilterChange}
            fields={filterFields}
          />
        </div>

        {/* One recovery banner per group of missing folders. */}
        {brokenGroups.length > 0 && (
          <div className="mb-6">
            {brokenGroups.map((g) => (
              <RelinkGroupBanner
                key={g.prefix}
                group={g}
                projectId={projectId}
                siteNames={Object.fromEntries(siteMap.entries())}
                deploymentsById={deploymentsById}
              />
            ))}
          </div>
        )}

        {isLoading ? (
          <div className="text-center py-12 text-muted-foreground">Loading deployments...</div>
        ) : filtered.length > 0 ? (
          <div className="rounded-lg border bg-white">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className={headClass} onClick={() => toggleSort("folder")}>
                    Folder<SortIcon field="folder" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("site_name")}>
                    Site<SortIcon field="site_name" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("period")}>
                    Period<SortIcon field="period" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("file_count")}>
                    Files<SortIcon field="file_count" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("notes")}>
                    Notes<SortIcon field="notes" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("tag_count")}>
                    Tags<SortIcon field="tag_count" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((dep) => (
                  <TableRow
                    key={dep.id}
                    onClick={() => setInfoDeploymentId(dep.id)}
                    className="cursor-pointer"
                  >
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        {dep.folder_status === "needs_relink" && (
                          <TooltipProvider delayDuration={100}>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <AlertTriangle className="h-4 w-4 text-destructive shrink-0" />
                              </TooltipTrigger>
                              <TooltipContent className="max-w-xs">
                                AddaxAI can't find this folder on disk. It may
                                have been renamed, moved, or is on a
                                disconnected drive. Use the banner at the top
                                of the page to reconnect it.
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )}
                        <span>{dep.folder_basename}</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {dep.site_name}
                    </TableCell>
                    <TableCell className="text-muted-foreground tabular-nums">
                      {formatPeriod(dep.start_date_local, dep.end_date_local)}
                    </TableCell>
                    <TableCell className="text-muted-foreground tabular-nums">{dep.file_count}</TableCell>
                    <TableCell className="text-muted-foreground max-w-[300px] truncate">
                      {dep.notes ? (dep.notes.length > 50 ? `${dep.notes.slice(0, 50)}\u2026` : dep.notes) : "\u2014"}
                    </TableCell>
                    <TableCell className="max-w-[320px]">
                      <TagPills tags={dep.tags} />
                    </TableCell>
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => setInfoDeploymentId(dep.id)}>
                            <Info className="mr-2 h-4 w-4" />
                            Info
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => setEditingDeployment(dep)}>
                            <Pencil className="mr-2 h-4 w-4" />
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={() => setSplittingDeployment({
                              id: dep.id,
                              folder_path: dep.folder_path,
                            })}
                            disabled={!dep.folder_path}
                          >
                            <Scissors className="mr-2 h-4 w-4" />
                            Split
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={() => setDeletingDeployment({
                              id: dep.id,
                              site_name: dep.site_name,
                              start_date_local: dep.start_date_local,
                            })}
                          >
                            <Trash2 className="mr-2 h-4 w-4" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : deployments && deployments.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Tent className="h-12 w-12 mb-3 text-muted-foreground/30" />
              <p className="text-muted-foreground">
                No deployments yet. Add deployments from the Analyses page.
              </p>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Search className="h-12 w-12 mb-3 text-muted-foreground/30" />
              <p className="text-muted-foreground">No deployments match your filters.</p>
            </CardContent>
          </Card>
        )}
      </main>

      {editingDeployment && (
        <EditDeploymentDialog
          deployment={editingDeployment}
          projectId={projectId}
          open={!!editingDeployment}
          onOpenChange={(open) => !open && setEditingDeployment(null)}
          onSplit={() => {
            const dep = editingDeployment;
            setEditingDeployment(null);
            setSplittingDeployment({
              id: dep.id,
              folder_path: dep.folder_path,
            });
          }}
        />
      )}

      <DeleteDeploymentDialog
        deployment={deletingDeployment}
        projectId={projectId}
        open={!!deletingDeployment}
        onOpenChange={(open) => !open && setDeletingDeployment(null)}
      />

      <SplitDeploymentDialog
        deployment={splittingDeployment}
        projectId={projectId}
        open={!!splittingDeployment}
        onOpenChange={(open) => !open && setSplittingDeployment(null)}
      />

      <DeploymentInfoSheet
        deploymentId={infoDeploymentId}
        open={infoDeploymentId !== null}
        onOpenChange={(open) => !open && setInfoDeploymentId(null)}
        onEdit={() => {
          if (!infoDeploymentId) return;
          const dep = deploymentsById.get(infoDeploymentId);
          setInfoDeploymentId(null);
          if (dep) setEditingDeployment(dep);
        }}
        onSplit={() => {
          if (!infoDeploymentId) return;
          const dep = deploymentsById.get(infoDeploymentId);
          setInfoDeploymentId(null);
          if (dep) {
            setSplittingDeployment({
              id: dep.id,
              folder_path: dep.folder_path,
            });
          }
        }}
        onDelete={() => {
          if (!infoDeploymentId) return;
          const row = rows.find((r) => r.id === infoDeploymentId);
          setInfoDeploymentId(null);
          if (row) {
            setDeletingDeployment({
              id: row.id,
              site_name: row.site_name,
              start_date_local: row.start_date_local,
            });
          }
        }}
      />

    </div>
  );
}

export default DeploymentsPage;
