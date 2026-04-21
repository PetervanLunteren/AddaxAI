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
import { Eye, FileImage, Info, Layers, FolderOpen, MapPin, CalendarDays } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Tooltip as UITooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../components/ui/tooltip";
import { statisticsApi } from "../api/statistics";
import { sitesApi } from "../api/sites";
import { useNoSiteDeployments } from "../hooks/useNoSiteDeployments";
import { buildSiteOptions } from "../lib/site-filter-options";
import { normalizeLabel } from "../utils/labels";
import { getSpeciesColor, getSpeciesColorWithAlpha, setSpeciesContext } from "../utils/species-colors";
import {
  type DateRange,
  DashboardFilters,
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
  const [taxonomicRank, setTaxonomicRank] = useState("all");
  const [speciesCountMode, setSpeciesCountMode] = useState<"events" | "max_n">("events");

  // Derive comma-separated site IDs for API calls
  const siteIdsParam = selectedSiteIds.length > 0 ? selectedSiteIds.join(",") : undefined;

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

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
              <p className="text-sm text-muted-foreground">
                Project overview with statistics and trends. Observation counts are based on MaxN per event, the peak number of individuals per species visible in a single image within an event, summed across all events.
              </p>
            </div>
            <DashboardFilters
              siteOptions={siteOptions}
              selectedSiteIds={selectedSiteIds}
              onSiteIdsChange={setSelectedSiteIds}
              dateRange={dateRange}
              onDateRangeChange={setDateRange}
              minDate={overview?.first_file_date}
              maxDate={overview?.last_file_date}
              taxonomicRank={taxonomicRank}
              onTaxonomicRankChange={setTaxonomicRank}
            />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
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
                  <TooltipProvider delayDuration={200}>
                    <UITooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-4 w-4 text-muted-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent side="bottom" className="max-w-sm p-3 space-y-2">
                        <p><span className="font-semibold">Frequency</span> counts how many independent events a species appeared in, regardless of how many individuals were in each event. A single cow in one event counts the same as 20 cows in another. It answers &ldquo;how often does this species show up?&rdquo;</p>
                        <p><span className="font-semibold">Abundance</span> sums the MaxN values, so it reflects how many individuals were observed. 20 cows in one event contributes 20 to the total. It answers &ldquo;how many individuals were observed?&rdquo;</p>
                        <p>Both are standard metrics in camera trap ecology. Frequency (event count) is what RAI is based on.</p>
                      </TooltipContent>
                    </UITooltip>
                  </TooltipProvider>
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
          trapNights={trapNights}
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
