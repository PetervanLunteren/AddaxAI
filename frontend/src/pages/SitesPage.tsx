/**
 * Sites metadata management page.
 *
 * Table-based view of all sites in a project with sorting, search, and edit actions.
 */

import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { Search, MoreVertical, Pencil, Trash2, ArrowUp, ArrowDown, MapPin } from "lucide-react";
import { sitesApi } from "../api/sites";
import type { SiteWithStats, SiteResponse } from "../api/types";
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
import { AddSiteModal } from "../components/analyses/AddSiteModal";
import { cn } from "../lib/utils";

type SortField = "name" | "latitude" | "longitude" | "elevation_m" | "habitat_type" | "deployment_count";
type SortDir = "asc" | "desc";

function SortIcon({ field, sortField, sortDir }: { field: SortField; sortField: SortField; sortDir: SortDir }) {
  if (field !== sortField) return null;
  return sortDir === "asc"
    ? <ArrowUp className="ml-1 inline h-3.5 w-3.5" />
    : <ArrowDown className="ml-1 inline h-3.5 w-3.5" />;
}

export function SitesPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [sortField, setSortField] = useState<SortField>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [editingSite, setEditingSite] = useState<SiteResponse | null>(null);

  const { data: sites, isLoading } = useQuery({
    queryKey: ["sites-with-stats", projectId],
    queryFn: () => sitesApi.listWithStats(projectId!),
    enabled: !!projectId,
  });

  const deleteMutation = useMutation({
    mutationFn: (siteId: string) => sitesApi.delete(siteId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sites-with-stats", projectId] });
      queryClient.invalidateQueries({ queryKey: ["sites", projectId] });
    },
  });

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  };

  const filtered = useMemo(() => {
    if (!sites) return [];
    let result = [...sites];

    // Text search
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(
        (s) =>
          s.name.toLowerCase().includes(q) ||
          (s.habitat_type && s.habitat_type.toLowerCase().includes(q)) ||
          (s.notes && s.notes.toLowerCase().includes(q))
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
  }, [sites, search, sortField, sortDir]);

  if (!projectId) {
    return <div>Project ID missing</div>;
  }

  const headClass = "cursor-pointer select-none hover:text-foreground";

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Sites</h1>
              <p className="text-sm text-muted-foreground">
                Manage monitoring locations for this project
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Filter bar */}
        <div className="mb-6 flex items-center gap-4">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search sites..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>
          <span className="text-sm text-muted-foreground">
            {filtered.length} {filtered.length === 1 ? "site" : "sites"}
          </span>
        </div>

        {isLoading ? (
          <div className="text-center py-12 text-muted-foreground">Loading sites...</div>
        ) : filtered.length > 0 ? (
          <div className="rounded-lg border bg-white">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className={headClass} onClick={() => toggleSort("name")}>
                    Name<SortIcon field="name" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("latitude")}>
                    Latitude<SortIcon field="latitude" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("longitude")}>
                    Longitude<SortIcon field="longitude" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("elevation_m")}>
                    Elevation<SortIcon field="elevation_m" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("habitat_type")}>
                    Habitat type<SortIcon field="habitat_type" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={cn(headClass, "text-right")} onClick={() => toggleSort("deployment_count")}>
                    Deployments<SortIcon field="deployment_count" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((site) => (
                  <TableRow key={site.id}>
                    <TableCell className="font-medium">{site.name}</TableCell>
                    <TableCell className="text-muted-foreground tabular-nums">
                      {site.latitude != null ? site.latitude.toFixed(4) : "\u2014"}
                    </TableCell>
                    <TableCell className="text-muted-foreground tabular-nums">
                      {site.longitude != null ? site.longitude.toFixed(4) : "\u2014"}
                    </TableCell>
                    <TableCell className="text-muted-foreground tabular-nums">
                      {site.elevation_m != null ? `${site.elevation_m}m` : "\u2014"}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {site.habitat_type || "\u2014"}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {site.deployment_count}
                    </TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => setEditingSite(site)}>
                            <Pencil className="mr-2 h-4 w-4" />
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={() => {
                              if (confirm("Are you sure you want to delete this site? All its deployments, files, and detections will be permanently removed.")) {
                                deleteMutation.mutate(site.id);
                              }
                            }}
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
        ) : sites && sites.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <MapPin className="h-12 w-12 mb-3 text-muted-foreground/30" />
              <p className="text-muted-foreground">
                No sites yet. Sites are created when you add deployments.
              </p>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Search className="h-12 w-12 mb-3 text-muted-foreground/30" />
              <p className="text-muted-foreground">No sites match your search.</p>
            </CardContent>
          </Card>
        )}
      </main>

      {editingSite && (
        <AddSiteModal
          projectId={projectId}
          site={editingSite}
          open={!!editingSite}
          onOpenChange={(open) => !open && setEditingSite(null)}
        />
      )}
    </div>
  );
}

export default SitesPage;
