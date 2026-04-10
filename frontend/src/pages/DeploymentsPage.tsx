/**
 * Deployments metadata management page.
 *
 * Table-based view of all deployments in a project with sorting, filters, and edit actions.
 */

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { Search, MoreVertical, Pencil, ArrowUp, ArrowDown, Tent } from "lucide-react";
import { deploymentsApi } from "../api/deployments";
import { sitesApi } from "../api/sites";
import type { DeploymentResponse, DeploymentStatsOnly, SiteResponse } from "../api/types";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import { EditDeploymentDialog } from "../components/deployments/EditDeploymentDialog";
import { cn } from "../lib/utils";

type SortField =
  | "site_name"
  | "start_date"
  | "end_date"
  | "camera_model"
  | "camera_serial"
  | "file_count"
  | "event_count"
  | "detection_count";
type SortDir = "asc" | "desc";

interface DeploymentRow extends DeploymentResponse {
  site_name: string;
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
  const [search, setSearch] = useState("");
  const [siteFilter, setSiteFilter] = useState<string>("all");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [sortField, setSortField] = useState<SortField>("start_date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [editingDeployment, setEditingDeployment] = useState<DeploymentResponse | null>(null);

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

  // Merge deployments with site names and stats
  const rows: DeploymentRow[] = useMemo(() => {
    if (!deployments) return [];
    return deployments.map((d) => {
      const stats: DeploymentStatsOnly = bulkStats?.[d.id] ?? {
        file_count: 0,
        event_count: 0,
        detection_count: 0,
      };
      return {
        ...d,
        site_name: siteMap.get(d.site_id) ?? "Unknown",
        file_count: stats.file_count,
        event_count: stats.event_count,
        detection_count: stats.detection_count,
      };
    });
  }, [deployments, bulkStats, siteMap]);

  // Filter and sort
  const filtered = useMemo(() => {
    let result = [...rows];

    // Site filter
    if (siteFilter !== "all") {
      result = result.filter((d) => d.site_id === siteFilter);
    }

    // Date range filter
    if (dateFrom) {
      result = result.filter((d) => d.start_date >= dateFrom);
    }
    if (dateTo) {
      result = result.filter((d) => d.start_date <= dateTo);
    }

    // Text search
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(
        (d) =>
          d.site_name.toLowerCase().includes(q) ||
          (d.camera_model && d.camera_model.toLowerCase().includes(q)) ||
          (d.camera_serial && d.camera_serial.toLowerCase().includes(q)) ||
          (d.notes && d.notes.toLowerCase().includes(q)) ||
          (d.folder_path && d.folder_path.toLowerCase().includes(q))
      );
    }

    // Sort
    result.sort((a, b) => {
      let aVal: string | number | null = a[sortField];
      let bVal: string | number | null = b[sortField];

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
  }, [rows, siteFilter, dateFrom, dateTo, search, sortField, sortDir]);

  if (!projectId) {
    return <div>Project ID missing</div>;
  }

  const isLoading = deploymentsLoading;
  const headClass = "cursor-pointer select-none hover:text-foreground";

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
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Filter bar */}
        <div className="mb-6 flex flex-wrap items-center gap-4">
          <Select value={siteFilter} onValueChange={setSiteFilter}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="All sites" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All sites</SelectItem>
              {sites?.map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  {s.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="flex items-center gap-2">
            <Input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="w-40"
              placeholder="From"
            />
            <span className="text-muted-foreground text-sm">to</span>
            <Input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="w-40"
              placeholder="To"
            />
          </div>

          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search camera, serial, notes..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>

          <span className="text-sm text-muted-foreground">
            {filtered.length} {filtered.length === 1 ? "deployment" : "deployments"}
          </span>
        </div>

        {isLoading ? (
          <div className="text-center py-12 text-muted-foreground">Loading deployments...</div>
        ) : filtered.length > 0 ? (
          <div className="rounded-lg border bg-white">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className={headClass} onClick={() => toggleSort("site_name")}>
                    Site<SortIcon field="site_name" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("start_date")}>
                    Start date<SortIcon field="start_date" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("end_date")}>
                    End date<SortIcon field="end_date" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("camera_model")}>
                    Camera<SortIcon field="camera_model" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("camera_serial")}>
                    Serial<SortIcon field="camera_serial" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={cn(headClass, "text-right")} onClick={() => toggleSort("file_count")}>
                    Files<SortIcon field="file_count" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={cn(headClass, "text-right")} onClick={() => toggleSort("event_count")}>
                    Events<SortIcon field="event_count" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={cn(headClass, "text-right")} onClick={() => toggleSort("detection_count")}>
                    Detections<SortIcon field="detection_count" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((dep) => (
                  <TableRow key={dep.id}>
                    <TableCell className="font-medium">{dep.site_name}</TableCell>
                    <TableCell className="tabular-nums">{dep.start_date}</TableCell>
                    <TableCell className="text-muted-foreground tabular-nums">
                      {dep.end_date || "\u2014"}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {dep.camera_model || "\u2014"}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {dep.camera_serial || "\u2014"}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{dep.file_count}</TableCell>
                    <TableCell className="text-right tabular-nums">{dep.event_count}</TableCell>
                    <TableCell className="text-right tabular-nums">{dep.detection_count}</TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => setEditingDeployment(dep)}>
                            <Pencil className="mr-2 h-4 w-4" />
                            Edit
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
        />
      )}
    </div>
  );
}

export default DeploymentsPage;
