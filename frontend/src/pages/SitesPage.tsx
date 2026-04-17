/**
 * Sites metadata management page.
 *
 * Table-based view of all sites in a project with sorting, search, and edit actions.
 */

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams, useSearchParams } from "react-router-dom";
import { Search, MoreVertical, Pencil, Trash2, ArrowUp, ArrowDown, MapPin, Plus, Info } from "lucide-react";
import { sitesApi } from "../api/sites";
import type { SiteWithStats, SiteResponse } from "../api/types";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { FilterBar, type FilterFieldDef, type FilterValues } from "../components/ui/filter-bar";
import { TagPills } from "../components/ui/tag-pills";
import { filtersFromSearchParams, filtersToSearchParams, type FilterSchema } from "../lib/filter-url";
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
import { DeleteSiteDialog } from "../components/sites/DeleteSiteDialog";
import { SiteInfoSheet } from "../components/sites/SiteInfoSheet";

type SortField = "name" | "elevation_m" | "habitat_type" | "deployment_count" | "notes" | "tag_count";
type SortDir = "asc" | "desc";

const FILTER_SCHEMA: FilterSchema = {
  search: "string",
  habitat: "string[]",
  has_deployments: "string",
  tag_keys: "string[]",
};

function SortIcon({ field, sortField, sortDir }: { field: SortField; sortField: SortField; sortDir: SortDir }) {
  if (field !== sortField) return null;
  return sortDir === "asc"
    ? <ArrowUp className="ml-1 inline h-3.5 w-3.5" />
    : <ArrowDown className="ml-1 inline h-3.5 w-3.5" />;
}

export function SitesPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const filters = useMemo(
    () => filtersFromSearchParams(searchParams, FILTER_SCHEMA),
    [searchParams]
  );
  const search = (filters.search as string | undefined) ?? "";
  const habitatFilter = (filters.habitat as string[] | undefined) ?? [];
  const hasDeploymentsFilter = (filters.has_deployments as string | undefined) ?? "";
  const tagKeysFilter = (filters.tag_keys as string[] | undefined) ?? [];

  const [sortField, setSortField] = useState<SortField>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [editingSite, setEditingSite] = useState<SiteResponse | null>(null);
  const [deletingSite, setDeletingSite] = useState<SiteWithStats | null>(null);
  const [infoSiteId, setInfoSiteId] = useState<string | null>(null);
  const [createSiteOpen, setCreateSiteOpen] = useState(false);

  const { data: sites, isLoading } = useQuery({
    queryKey: ["sites-with-stats", projectId],
    queryFn: () => sitesApi.listWithStats(projectId!),
    enabled: !!projectId,
  });

  const handleFilterChange = (next: FilterValues) => {
    // "any" is the default for the has_deployments select; drop it from
    // the URL so it doesn't show up as an active filter chip.
    const cleaned = { ...next };
    if (cleaned.has_deployments === "any") {
      delete cleaned.has_deployments;
    }
    setSearchParams(filtersToSearchParams(cleaned, FILTER_SCHEMA));
  };

  // Distinct habitat types and tag keys derived from loaded sites,
  // sorted alphabetically. Empty/null habitat types are skipped.
  const habitatOptions = useMemo(() => {
    const set = new Set<string>();
    for (const s of sites ?? []) {
      if (s.habitat_type && s.habitat_type.trim()) set.add(s.habitat_type);
    }
    return Array.from(set)
      .sort((a, b) => a.localeCompare(b))
      .map((h) => ({ value: h, label: h }));
  }, [sites]);

  const tagKeyOptions = useMemo(() => {
    const set = new Set<string>();
    for (const s of sites ?? []) {
      for (const k of Object.keys(s.tags ?? {})) {
        if (k.trim()) set.add(k);
      }
    }
    return Array.from(set)
      .sort((a, b) => a.localeCompare(b))
      .map((k) => ({ value: k, label: k }));
  }, [sites]);

  const filterFields: FilterFieldDef[] = [
    { kind: "search", key: "search", label: "Search", placeholder: "Search..." },
    {
      kind: "multi-select",
      key: "habitat",
      label: "Habitat",
      options: habitatOptions,
      placeholder: "All habitats",
      summary: (n) => `${n} habitat${n > 1 ? "s" : ""}`,
    },
    {
      kind: "select",
      key: "has_deployments",
      label: "Deployments",
      options: [
        { value: "any", label: "Any" },
        { value: "with", label: "With deployments" },
        { value: "without", label: "Without deployments" },
      ],
      placeholder: "Any",
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
      result = result.filter((s) => {
        if (s.name.toLowerCase().includes(q)) return true;
        if (s.habitat_type && s.habitat_type.toLowerCase().includes(q)) return true;
        if (s.notes && s.notes.toLowerCase().includes(q)) return true;
        for (const [k, v] of Object.entries(s.tags ?? {})) {
          if (k.toLowerCase().includes(q) || v.toLowerCase().includes(q)) {
            return true;
          }
        }
        return false;
      });
    }

    // Habitat type multi-select
    if (habitatFilter.length > 0) {
      const set = new Set(habitatFilter);
      result = result.filter((s) => s.habitat_type !== null && set.has(s.habitat_type));
    }

    // Has-deployments select (any / with / without)
    if (hasDeploymentsFilter === "with") {
      result = result.filter((s) => s.deployment_count > 0);
    } else if (hasDeploymentsFilter === "without") {
      result = result.filter((s) => s.deployment_count === 0);
    }

    // Tag keys multi-select (site has at least one of the selected keys)
    if (tagKeysFilter.length > 0) {
      const set = new Set(tagKeysFilter);
      result = result.filter((s) => {
        for (const k of Object.keys(s.tags ?? {})) {
          if (set.has(k)) return true;
        }
        return false;
      });
    }

    // Sort
    result.sort((a, b) => {
      let aVal: string | number | null = sortField === "tag_count"
        ? Object.keys(a.tags ?? {}).length
        : a[sortField];
      let bVal: string | number | null = sortField === "tag_count"
        ? Object.keys(b.tags ?? {}).length
        : b[sortField];

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
  }, [sites, search, habitatFilter, hasDeploymentsFilter, tagKeysFilter, sortField, sortDir]);

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
            <Button onClick={() => setCreateSiteOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              New site
            </Button>
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
                  <TableHead className={headClass} onClick={() => toggleSort("elevation_m")}>
                    Elevation<SortIcon field="elevation_m" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("habitat_type")}>
                    Habitat type<SortIcon field="habitat_type" sortField={sortField} sortDir={sortDir} />
                  </TableHead>
                  <TableHead className={headClass} onClick={() => toggleSort("deployment_count")}>
                    Deployments<SortIcon field="deployment_count" sortField={sortField} sortDir={sortDir} />
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
                {filtered.map((site) => (
                  <TableRow
                    key={site.id}
                    onClick={() => setInfoSiteId(site.id)}
                    className="cursor-pointer"
                  >
                    <TableCell className="font-medium">{site.name}</TableCell>
                    <TableCell className="text-muted-foreground tabular-nums">
                      {site.elevation_m != null ? `${site.elevation_m}m` : "\u2014"}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {site.habitat_type || "\u2014"}
                    </TableCell>
                    <TableCell className="text-muted-foreground tabular-nums">
                      {site.deployment_count}
                    </TableCell>
                    <TableCell className="text-muted-foreground max-w-[300px] truncate">
                      {site.notes ? (site.notes.length > 50 ? `${site.notes.slice(0, 50)}\u2026` : site.notes) : "\u2014"}
                    </TableCell>
                    <TableCell className="max-w-[320px]">
                      <TagPills tags={site.tags} />
                    </TableCell>
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => setInfoSiteId(site.id)}>
                            <Info className="mr-2 h-4 w-4" />
                            Info
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => setEditingSite(site)}>
                            <Pencil className="mr-2 h-4 w-4" />
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={() => setDeletingSite(site)}
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

      <AddSiteModal
        projectId={projectId}
        open={createSiteOpen}
        onOpenChange={setCreateSiteOpen}
      />

      <DeleteSiteDialog
        site={deletingSite}
        open={!!deletingSite}
        onOpenChange={(open) => !open && setDeletingSite(null)}
      />

      <SiteInfoSheet
        siteId={infoSiteId}
        open={infoSiteId !== null}
        onOpenChange={(open) => !open && setInfoSiteId(null)}
        onEdit={() => {
          if (!infoSiteId) return;
          const site = sites?.find((s) => s.id === infoSiteId);
          setInfoSiteId(null);
          if (site) setEditingSite(site);
        }}
        onDelete={() => {
          if (!infoSiteId) return;
          const site = sites?.find((s) => s.id === infoSiteId);
          setInfoSiteId(null);
          if (site) setDeletingSite(site);
        }}
      />
    </div>
  );
}

export default SitesPage;
