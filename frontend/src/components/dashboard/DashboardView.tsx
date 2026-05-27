/**
 * Project Dashboard body — pure presentational component.
 *
 * Mounted by:
 * - ``DashboardPage`` (the research-projects route ``/projects/:id/dashboard``
 *   wraps it in its own ``min-h-screen`` + ``<header>`` chrome).
 * - ``FolderRunOverviewStep`` (the folder-run Overview step renders it
 *   inline under the stepper's existing header).
 *
 * Single source of truth: changing what the dashboard shows (new
 * filter, new chart, new query) automatically applies in both
 * places.
 */

import { useState, useMemo } from "react";
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
import {
  Eye,
  FileImage,
  Layers,
  FolderOpen,
  MapPin,
  CalendarDays,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import {
  FilterBar,
  type FilterFieldDef,
  type FilterValues,
} from "../ui/filter-bar";
import { DashboardAboutPopover } from "./DashboardAboutPopover";
import { MissingDatesBanner } from "./MissingDatesWarning";
import { statisticsApi } from "../../api/statistics";
import { sitesApi } from "../../api/sites";
import { projectsApi } from "../../api/projects";
import { useNoSiteDeployments } from "../../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../../lib/site-filter-options";
import { normalizeLabel } from "../../utils/labels";
import {
  getSpeciesColor,
  getSpeciesColorWithAlpha,
  setSpeciesContext,
} from "../../utils/species-colors";
import { RANK_OPTIONS } from "../../lib/taxonomic-rank";
import {
  type DateRange,
  ActivityPatternChart,
  DetectionTrendChart,
  AlertCounters,
  VerificationProgressChart,
} from ".";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
);

export function DashboardView({ projectId }: { projectId: string }) {
  const [dateRange, setDateRange] = useState<DateRange>({
    startDate: null,
    endDate: null,
  });
  const [selectedSiteIds, setSelectedSiteIds] = useState<string[]>([]);
  const [selectedTagPairs, setSelectedTagPairs] = useState<string[]>([]);
  const [taxonomicRank, setTaxonomicRank] = useState("all");
  const [speciesCountMode, setSpeciesCountMode] = useState<
    "events" | "max_n"
  >("max_n");

  // Fetch sites for filter options
  // Folder runs are a single deployment with no site, so site / site-tag
  // filters and the site / deployment count cards are meaningless there.
  const { data: project } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId),
    enabled: !!projectId,
  });
  const isFolderRun = project?.mode === "folder_run";

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId && !isFolderRun,
  });
  const { data: noSite } = useNoSiteDeployments(projectId);

  const siteOptions = useMemo(
    () => buildSiteOptions(sites, noSite?.count ?? 0),
    [sites, noSite],
  );

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

  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: [
      "statistics",
      "overview",
      projectId,
      siteIdsParam,
      dateRange.startDate,
      dateRange.endDate,
    ],
    queryFn: () =>
      statisticsApi.getOverview(
        projectId,
        siteIdsParam,
        dateRange.startDate ?? undefined,
        dateRange.endDate ?? undefined,
      ),
    enabled: !!projectId,
  });

  const { data: species, isLoading: speciesLoading } = useQuery({
    queryKey: [
      "statistics",
      "species",
      projectId,
      siteIdsParam,
      dateRange.startDate,
      dateRange.endDate,
      taxonomicRank,
      speciesCountMode,
    ],
    queryFn: () =>
      statisticsApi.getSpeciesDistribution(
        projectId,
        siteIdsParam,
        dateRange.startDate ?? undefined,
        dateRange.endDate ?? undefined,
        taxonomicRank,
        speciesCountMode,
      ),
    enabled: !!projectId,
  });

  useMemo(() => {
    if (species && species.length > 0) {
      const allSpecies = species.map((s) => s.species);
      allSpecies.push("animal", "person", "vehicle", "empty");
      setSpeciesContext(allSpecies);
    }
  }, [species]);

  const trapNights = overview?.trap_nights ?? 0;
  const norm = (n: number) => +((n / trapNights) * 100).toFixed(2);

  const compact = (n: number): string => {
    if (n >= 1_000_000) return `${+(n / 1_000_000).toFixed(1)}M`;
    if (n >= 10_000) return `${+(n / 1_000).toFixed(1)}K`;
    return n.toLocaleString();
  };

  const summaryCards = [
    // Site + deployment counts are always 1 / 0 for a folder run, so
    // they're omitted there.
    ...(isFolderRun
      ? []
      : [
          { title: "Sites", value: overview?.total_sites ?? 0, icon: MapPin, color: "#0f6064" },
          { title: "Deployments", value: overview?.total_deployments ?? 0, icon: FolderOpen, color: "#0f6064" },
        ]),
    { title: "Trap nights", value: overview?.trap_nights ?? 0, icon: CalendarDays, color: "#0f6064" },
    { title: "Events", value: overview?.total_events ?? 0, icon: Layers, color: "#0f6064" },
    { title: "Files", value: overview?.total_files ?? 0, icon: FileImage, color: "#0f6064" },
    { title: "Observations", value: overview?.total_observations ?? 0, icon: Eye, color: "#0f6064" },
  ];

  const speciesAxisLabel =
    speciesCountMode === "events"
      ? "Independent events per 100 trap nights"
      : "Observations per 100 trap nights";

  const speciesData = {
    labels: species?.map((s) => normalizeLabel(s.species)) ?? [],
    datasets: [
      {
        label: speciesAxisLabel,
        data: species?.map((s) => norm(s.count)) ?? [],
        backgroundColor:
          species?.map((s) => getSpeciesColorWithAlpha(s.species, 0.8)) ?? [],
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

  const filterValues: FilterValues = {
    sites: selectedSiteIds.length > 0 ? selectedSiteIds : undefined,
    tag_pairs:
      selectedTagPairs.length > 0 ? selectedTagPairs : undefined,
    date_from: dateRange.startDate ?? undefined,
    date_to: dateRange.endDate ?? undefined,
    rank: taxonomicRank,
  };

  const handleFilterChange = (next: FilterValues) => {
    const sitesNext = next.sites;
    setSelectedSiteIds(Array.isArray(sitesNext) ? sitesNext : []);
    const tagNext = next.tag_pairs;
    setSelectedTagPairs(Array.isArray(tagNext) ? tagNext : []);
    setDateRange({
      startDate:
        typeof next.date_from === "string" ? next.date_from : null,
      endDate: typeof next.date_to === "string" ? next.date_to : null,
    });
    setTaxonomicRank(
      typeof next.rank === "string" ? next.rank : "all",
    );
  };

  const filterFields: FilterFieldDef[] = [
    // Site + site-tag filters don't apply to a single-deployment,
    // siteless folder run.
    ...(isFolderRun
      ? []
      : ([
          {
            kind: "multi-select",
            key: "sites",
            label: "Sites",
            options: siteOptions,
            placeholder: "All sites",
            summary: (n: number) => `${n} site${n > 1 ? "s" : ""}`,
          },
          {
            kind: "multi-select",
            key: "tag_pairs",
            label: "Site tags",
            options: tagPairOptions,
            placeholder: "Any tags",
            summary: (n: number) => `${n} tag${n > 1 ? "s" : ""}`,
          },
        ] as FilterFieldDef[])),
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
    <div className="space-y-6">
      <FilterBar
        value={filterValues}
        onChange={handleFilterChange}
        fields={filterFields}
      />

      <MissingDatesBanner projectId={projectId} />

      <div
        className={`grid gap-4 grid-cols-2 ${
          isFolderRun
            ? "md:grid-cols-2 xl:grid-cols-4"
            : "md:grid-cols-3 xl:grid-cols-6"
        }`}
      >
        {summaryCards.map((card) => (
          <Card key={card.title}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">
                    {card.title}
                  </p>
                  <p className="text-2xl font-bold mt-1">
                    {overviewLoading ? "..." : compact(card.value)}
                  </p>
                </div>
                <div
                  className="p-3 rounded-lg"
                  style={{ backgroundColor: `${card.color}20` }}
                >
                  <card.icon
                    className="h-6 w-6"
                    style={{ color: card.color }}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 grid-cols-1 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-1.5">
                  <CardTitle className="text-lg">Taxa detected</CardTitle>
                  <DashboardAboutPopover
                    what={
                      <p>
                        Top 10 taxa in the filtered project view,
                        ranked by either how often they appear
                        (Frequency) or how many individuals were
                        observed (Abundance). Switch with the
                        dropdown.
                      </p>
                    }
                    how={
                      <p>
                        Frequency counts the number of independent
                        events containing the taxon. Abundance sums
                        each event's MaxN across those events. MaxN
                        is an event's count of a taxon: the most
                        individuals of it visible in any single frame
                        of that event, so the same animals are not
                        re-counted frame to frame. Events come from
                        files captured close together in time,
                        grouped by the project's independence
                        interval. Frequency is the basis for RAI in
                        camera-trap ecology.
                      </p>
                    }
                  />
                </div>
                <p className="text-sm text-muted-foreground">
                  {speciesCountMode === "events"
                    ? "Top 10 by number of independent events"
                    : "Top 10 by total observations"}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Select
                  value={speciesCountMode}
                  onValueChange={(v) =>
                    setSpeciesCountMode(v as "events" | "max_n")
                  }
                >
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
                  <p className="text-muted-foreground">
                    No species data available
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
        <DetectionTrendChart
          dateRange={dateRange}
          projectId={projectId}
          siteIds={siteIdsParam}
          trapNights={trapNights}
          taxonomicRank={taxonomicRank}
        />
      </div>

      <div className="grid gap-6 grid-cols-1 md:grid-cols-3">
        <ActivityPatternChart
          dateRange={dateRange}
          projectId={projectId}
          siteIds={siteIdsParam}
          trapNights={trapNights}
          taxonomicRank={taxonomicRank}
        />
        <AlertCounters
          projectId={projectId}
          siteIds={siteIdsParam}
          dateFrom={dateRange.startDate ?? undefined}
          dateTo={dateRange.endDate ?? undefined}
        />
        <VerificationProgressChart
          projectId={projectId}
          siteIds={siteIdsParam}
          dateFrom={dateRange.startDate ?? undefined}
          dateTo={dateRange.endDate ?? undefined}
        />
      </div>
    </div>
  );
}
