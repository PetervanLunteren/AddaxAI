/**
 * Dashboard page with statistics and charts
 */
import { useState, useMemo } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  type ChartOptions,
} from "chart.js";
import { Eye, FileImage, Layers, FolderOpen, MapPin, CalendarDays } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import {
  FilterBar,
  type FilterFieldDef,
  type FilterValues,
} from "../components/ui/filter-bar";
import { BugReportButton } from "../components/diagnostics/BugReportButton";
import { DashboardAboutPopover } from "../components/dashboard/DashboardAboutPopover";
import { statisticsApi } from "../api/statistics";
import { sitesApi } from "../api/sites";
import { useNoSiteDeployments } from "../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../lib/site-filter-options";
import { normalizeLabel } from "../utils/labels";
import { getSpeciesColor, getSpeciesColorWithAlpha, setSpeciesContext } from "../utils/species-colors";
import { RANK_OPTIONS } from "../lib/taxonomic-rank";
import {
  type DateRange,
  ActivityPatternChart,
  DetectionTrendChart,
  AlertCounters,
  VerificationProgressChart,
} from "../components/dashboard";

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend);

export default function DashboardPage() {
  const { projectId } = useParams<{ projectId: string }>();

  const [dateRange, setDateRange] = useState<DateRange>({
    startDate: null,
    endDate: null,
  });
  const [selectedSiteIds, setSelectedSiteIds] = useState<string[]>([]);
  const [selectedTagPairs, setSelectedTagPairs] = useState<string[]>([]);
  const [taxonomicRank, setTaxonomicRank] = useState("all");
  const [speciesCountMode, setSpeciesCountMode] = useState<"events" | "max_n">("events");

  // Fetch sites for filter options
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId,
  });
  const { data: noSite } = useNoSiteDeployments(projectId);

  const siteOptions = useMemo(
    () => buildSiteOptions(sites, noSite?.count ?? 0),
    [sites, noSite],
  );

  // All unique tag key:value combos across the project's sites. Each
  // entry is "key:value"; the label shows them with a space for
  // readability ("habitat: forest"). Sorted alphabetically.
  const tagPairOptions = useMemo(() => {
    const set = new Set<string>();
    for (const s of sites ?? []) {
      for (const [k, v] of Object.entries(s.tags ?? {})) {
        const key = k.trim();
        const value = String(v ?? "").trim();
        if (key && value) set.add(`${key}:${value}`);
      }
    }
    return Array.from(set)
      .sort((a, b) => a.localeCompare(b))
      .map((pair) => {
        const idx = pair.indexOf(":");
        const key = pair.slice(0, idx);
        const val = pair.slice(idx + 1);
        return { value: pair, label: `${key}: ${val}` };
      });
  }, [sites]);

  // Resolve the active tag-pair filter into a set of matching site
  // IDs. A site matches if it has at least one of the picked pairs.
  // When the tag filter is empty, returns null (meaning "no tag
  // constraint, all sites pass").
  const tagFilteredSiteIds = useMemo<Set<string> | null>(() => {
    if (selectedTagPairs.length === 0) return null;
    const picks = new Set(selectedTagPairs);
    const matched = new Set<string>();
    for (const s of sites ?? []) {
      for (const [k, v] of Object.entries(s.tags ?? {})) {
        const pair = `${k}:${String(v)}`;
        if (picks.has(pair)) {
          matched.add(s.id);
          break;
        }
      }
    }
    return matched;
  }, [sites, selectedTagPairs]);

  // Effective site IDs sent to the statistics API. If tag filter is
  // active and the user also picked specific sites, intersect the two.
  // Tag-only filter resolves to all matching sites; no filter
  // resolves to undefined (server-side default = all sites).
  const effectiveSiteIds = useMemo<string[] | undefined>(() => {
    if (tagFilteredSiteIds === null) {
      return selectedSiteIds.length > 0 ? selectedSiteIds : undefined;
    }
    if (selectedSiteIds.length === 0) {
      return Array.from(tagFilteredSiteIds);
    }
    return selectedSiteIds.filter((id) => tagFilteredSiteIds.has(id));
  }, [selectedSiteIds, tagFilteredSiteIds]);

  const siteIdsParam = effectiveSiteIds?.join(",") || undefined;

  // Fetch overview
  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ["statistics", "overview", projectId, siteIdsParam, dateRange.startDate, dateRange.endDate],
    queryFn: () =>
      statisticsApi.getOverview(projectId!, siteIdsParam, dateRange.startDate ?? undefined, dateRange.endDate ?? undefined),
    enabled: !!projectId,
  });

  // Fetch species distribution
  const { data: species, isLoading: speciesLoading } = useQuery({
    queryKey: ["statistics", "species", projectId, siteIdsParam, dateRange.startDate, dateRange.endDate, taxonomicRank, speciesCountMode],
    queryFn: () =>
      statisticsApi.getSpeciesDistribution(projectId!, siteIdsParam, dateRange.startDate ?? undefined, dateRange.endDate ?? undefined, taxonomicRank, speciesCountMode),
    enabled: !!projectId,
  });

  // Set species color context
  useMemo(() => {
    if (species && species.length > 0) {
      const allSpecies = species.map((s) => s.species);
      allSpecies.push("animal", "person", "vehicle", "empty");
      setSpeciesContext(allSpecies);
    }
  }, [species]);

  // Trap nights for normalization
  const trapNights = overview?.trap_nights ?? 0;
  const norm = (n: number) => +(n / trapNights * 100).toFixed(2);

  // Format large numbers with K/M suffixes
  const compact = (n: number): string => {
    if (n >= 1_000_000) return `${+(n / 1_000_000).toFixed(1)}M`;
    if (n >= 10_000) return `${+(n / 1_000).toFixed(1)}K`;
    return n.toLocaleString();
  };

  // Summary cards (raw counts, not normalized)
  const summaryCards = [
    { title: "Sites", value: overview?.total_sites ?? 0, icon: MapPin, color: "#0f6064" },
    { title: "Deployments", value: overview?.total_deployments ?? 0, icon: FolderOpen, color: "#0f6064" },
    { title: "Trap nights", value: overview?.trap_nights ?? 0, icon: CalendarDays, color: "#0f6064" },
    { title: "Events", value: overview?.total_events ?? 0, icon: Layers, color: "#0f6064" },
    { title: "Files", value: overview?.total_files ?? 0, icon: FileImage, color: "#0f6064" },
    { title: "Observations", value: overview?.total_observations ?? 0, icon: Eye, color: "#0f6064" },
  ];

  // Species bar chart
  const speciesAxisLabel = speciesCountMode === "events"
    ? "Independent events per 100 trap nights"
    : "Observations (MaxN) per 100 trap nights";

  const speciesData = {
    labels: species?.map((s) => normalizeLabel(s.species)) ?? [],
    datasets: [
      {
        label: speciesAxisLabel,
        data: species?.map((s) => norm(s.count)) ?? [],
        backgroundColor: species?.map((s) => getSpeciesColorWithAlpha(s.species, 0.8)) ?? [],
        borderColor: species?.map((s) => getSpeciesColor(s.species)) ?? [],
        borderWidth: 1,
      },
    ],
  };

  const speciesOptions: ChartOptions<"bar"> = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: false },
    },
    scales: {
      x: {
        beginAtZero: true,
        title: { display: true, text: speciesAxisLabel },
      },
    },
  };

  // FilterBar uses a single FilterValues object; we keep the three
  // useStates as the source of truth (children components consume them
  // individually) and translate in both directions. "all" rank is the
  // default so it never appears in the FilterValues — selecting any
  // explicit rank surfaces a chip the user can dismiss.
  const filterValues: FilterValues = {
    sites: selectedSiteIds.length > 0 ? selectedSiteIds : undefined,
    tag_pairs: selectedTagPairs.length > 0 ? selectedTagPairs : undefined,
    date_from: dateRange.startDate ?? undefined,
    date_to: dateRange.endDate ?? undefined,
    // rank is always set so the Select renders the value (not the
    // greyed-out placeholder). FilterBar's `defaultValue: "all"`
    // means the chip stays hidden when sitting at the default.
    rank: taxonomicRank,
  };

  const handleFilterChange = (next: FilterValues) => {
    const sitesNext = next.sites;
    setSelectedSiteIds(Array.isArray(sitesNext) ? sitesNext : []);
    const tagNext = next.tag_pairs;
    setSelectedTagPairs(Array.isArray(tagNext) ? tagNext : []);
    setDateRange({
      startDate: typeof next.date_from === "string" ? next.date_from : null,
      endDate: typeof next.date_to === "string" ? next.date_to : null,
    });
    setTaxonomicRank(typeof next.rank === "string" ? next.rank : "all");
  };

  const filterFields: FilterFieldDef[] = [
    {
      kind: "multi-select",
      key: "sites",
      label: "Sites",
      options: siteOptions,
      placeholder: "All sites",
      summary: (n) => `${n} site${n > 1 ? "s" : ""}`,
    },
    {
      kind: "multi-select",
      key: "tag_pairs",
      label: "Site tags",
      options: tagPairOptions,
      placeholder: "Any tags",
      summary: (n) => `${n} tag${n > 1 ? "s" : ""}`,
    },
    {
      kind: "date_range",
      key: "date_from",
      toKey: "date_to",
      label: "Date range",
      min: overview?.first_file_date ?? undefined,
      max: overview?.last_file_date ?? undefined,
    },
    {
      kind: "select",
      key: "rank",
      label: "Taxonomic rank",
      options: RANK_OPTIONS,
      defaultValue: "all",
    },
  ];

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
              <p className="text-sm text-muted-foreground">
                Project overview with statistics and trends
              </p>
            </div>
            <BugReportButton />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        {/* Filter bar (canonical, same style as Sites / Deployments / etc.) */}
        <FilterBar
          value={filterValues}
          onChange={handleFilterChange}
          fields={filterFields}
        />
        {/* Summary Cards */}
      <div className="grid gap-4 grid-cols-2 md:grid-cols-3 xl:grid-cols-6">
        {summaryCards.map((card) => (
          <Card key={card.title}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">{card.title}</p>
                  <p className="text-2xl font-bold mt-1">
                    {overviewLoading ? "..." : compact(card.value)}
                  </p>
                </div>
                <div className="p-3 rounded-lg" style={{ backgroundColor: `${card.color}20` }}>
                  <card.icon className="h-6 w-6" style={{ color: card.color }} />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Row 1: Species detected + Detection trend */}
      <div className="grid gap-6 grid-cols-1 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-1.5">
                  <CardTitle className="text-lg">Taxa detected</CardTitle>
                  <DashboardAboutPopover
                    what={
                      <>
                        <p>
                          Top 10 taxa in the filtered project view, ranked by
                          either how often they appear (Frequency) or how
                          many individuals were observed (Abundance). Switch
                          with the dropdown.
                        </p>
                      </>
                    }
                    how={
                      <>
                        <p>
                          Frequency counts the number of independent events
                          containing the taxon. Abundance sums MaxN across
                          those events. Events come from files captured
                          close together in time, grouped by the project's
                          independence interval. Frequency is the basis for
                          RAI in camera-trap ecology.
                        </p>
                      </>
                    }
                  />
                </div>
                <p className="text-sm text-muted-foreground">
                  {speciesCountMode === "events"
                    ? "Top 10 by number of independent events"
                    : "Top 10 by total observations (MaxN sum)"}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Select value={speciesCountMode} onValueChange={(v) => setSpeciesCountMode(v as "events" | "max_n")}>
                  <SelectTrigger className="w-[140px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="events">Frequency</SelectItem>
                    <SelectItem value="max_n">Abundance</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              {speciesLoading ? (
                <div className="flex items-center justify-center h-full">
                  <p className="text-muted-foreground">Loading...</p>
                </div>
              ) : species && species.length > 0 ? (
                <Bar data={speciesData} options={speciesOptions} />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <p className="text-muted-foreground">No species data available</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
        <DetectionTrendChart
          dateRange={dateRange}
          projectId={projectId!}
          siteIds={siteIdsParam}
          trapNights={trapNights}
          taxonomicRank={taxonomicRank}
        />
      </div>

      {/* Row 2: Activity pattern + Detection categories + Verification progress */}
      <div className="grid gap-6 grid-cols-1 md:grid-cols-3">
        <ActivityPatternChart
          dateRange={dateRange}
          projectId={projectId!}
          siteIds={siteIdsParam}
          trapNights={trapNights}
          taxonomicRank={taxonomicRank}
        />
        <AlertCounters
          projectId={projectId!}
          siteIds={siteIdsParam}
          dateFrom={dateRange.startDate ?? undefined}
          dateTo={dateRange.endDate ?? undefined}
        />
        <VerificationProgressChart
          projectId={projectId!}
          siteIds={siteIdsParam}
          dateFrom={dateRange.startDate ?? undefined}
          dateTo={dateRange.endDate ?? undefined}
        />
      </div>
      </main>
    </div>
  );
}
