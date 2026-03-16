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
import { Maximize, FileImage, Images, FolderOpen, MapPin, Moon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { statisticsApi } from "../api/statistics";
import { sitesApi } from "../api/sites";
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
  const [taxonomicRank, setTaxonomicRank] = useState("species");

  // Derive comma-separated site IDs for API calls
  const siteIdsParam = selectedSiteIds.length > 0 ? selectedSiteIds.join(",") : undefined;

  // Fetch sites for filter options
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId,
  });

  const siteOptions = useMemo(
    () => (sites ?? []).map((s) => ({ value: s.id, label: s.name })),
    [sites]
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
    queryKey: ["statistics", "species", projectId, siteIdsParam, dateRange.startDate, dateRange.endDate, taxonomicRank],
    queryFn: () =>
      statisticsApi.getSpeciesDistribution(projectId!, siteIdsParam, dateRange.startDate ?? undefined, dateRange.endDate ?? undefined, taxonomicRank),
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

  // Summary cards (raw counts, not normalized)
  const summaryCards = [
    { title: "Detections", value: overview?.total_detections ?? 0, icon: Maximize, color: "#0f6064" },
    { title: "Files", value: overview?.total_files ?? 0, icon: FileImage, color: "#0f6064" },
    { title: "Events", value: overview?.total_events ?? 0, icon: Images, color: "#0f6064" },
    { title: "Deployments", value: overview?.total_deployments ?? 0, icon: FolderOpen, color: "#0f6064" },
    { title: "Sites", value: overview?.total_sites ?? 0, icon: MapPin, color: "#0f6064" },
    { title: "Trap nights", value: overview?.trap_nights ?? 0, icon: Moon, color: "#0f6064" },
  ];

  // Species bar chart (normalized)
  const speciesData = {
    labels: species?.map((s) => normalizeLabel(s.species)) ?? [],
    datasets: [
      {
        label: "Per 100 trap nights",
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
        title: { display: true, text: "Per 100 trap nights" },
      },
    },
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Project overview with statistics and trends
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

      {/* Summary Cards */}
      <div className="grid gap-4 grid-cols-2 md:grid-cols-3 lg:grid-cols-6">
        {summaryCards.map((card) => (
          <Card key={card.title}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">{card.title}</p>
                  <p className="text-2xl font-bold mt-1">
                    {overviewLoading ? "..." : card.value.toLocaleString()}
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
            <CardTitle className="text-lg">Taxa detected</CardTitle>
            <p className="text-sm text-muted-foreground">
              Top 10 most frequently observed
            </p>
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
    </div>
  );
}
