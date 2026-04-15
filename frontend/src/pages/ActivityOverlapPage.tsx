/**
 * Plots → Activity overlap page.
 *
 * Page-wide 1- or 2-species temporal activity comparison. Owns its
 * filter state, persists it in the URL, fetches the new activity-overlap
 * endpoint, and renders the chart + Δ readout + per-species legend with
 * diel badges and sample-size warnings.
 *
 * The science (KDE, Δ, bootstrap CI, diel classification) lives on the
 * backend in `app.ml.activity_analysis` so this file is purely
 * presentational.
 */

import { useMemo } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Info, Loader2 } from "lucide-react";

import { statisticsApi } from "../api/statistics";
import type {
  ActivityOverlapResponse,
  DielClass,
  SampleSizeWarning,
  SpeciesActivity,
} from "../api/statistics";
import {
  ActivityOverlapChart,
  SPECIES_A_COLOR,
  SPECIES_B_COLOR,
} from "../components/plots/ActivityOverlapChart";
import {
  ActivityOverlapFilterBar,
  type ActivityOverlapPageFilters,
  type TimeAxis,
} from "../components/plots/ActivityOverlapFilterBar";
import { Badge } from "../components/ui/badge";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../components/ui/tooltip";
import {
  filtersFromSearchParams,
  filtersToSearchParams,
  type FilterSchema,
} from "../lib/filter-url";
import { normalizeLabel } from "../utils/labels";

const FILTER_SCHEMA: FilterSchema = {
  species_a: "string",
  species_b: "string",
  site_ids: "string[]",
  date_from: "date",
  date_to: "date",
  time_axis: "string",
  bands_visible: "string",
};

const DIEL_LABEL: Record<DielClass, string> = {
  diurnal: "Diurnal",
  nocturnal: "Nocturnal",
  crepuscular: "Crepuscular",
  cathemeral: "Cathemeral",
};

const DIEL_TOOLTIP_RULE =
  "≥ 70% of activity density falls in this phase. Threshold per Bennie et al. 2014.";

const SAMPLE_WARNING_LABEL: Record<SampleSizeWarning, string> = {
  low_n_30: "too few detections, interpret with caution",
  low_n_50: "small sample, using Δ₁",
  low_n_75: "Δ₄ may be unreliable below n=75",
};

const SAMPLE_WARNING_CRITICAL: Record<SampleSizeWarning, boolean> = {
  low_n_30: true,
  low_n_50: false,
  low_n_75: false,
};

function parseTimeAxis(raw: string | undefined): TimeAxis {
  return raw === "sun" ? "sun" : "clock";
}

function parseBandsVisible(raw: string | undefined): boolean {
  return raw !== "false"; // default true
}

function formatPhase(phase: string, value: number): string {
  return `${phase} ${(value * 100).toFixed(0)}%`;
}

function dominantPhase(
  diel: DielClass,
  byPhase: Record<string, number>,
): string {
  if (diel === "diurnal") return formatPhase("day", byPhase.day ?? 0);
  if (diel === "nocturnal") return formatPhase("night", byPhase.night ?? 0);
  if (diel === "crepuscular") return formatPhase("twilight", byPhase.twilight ?? 0);
  return "mixed";
}

interface SpeciesLegendProps {
  species: SpeciesActivity;
  swatchColor: string;
}

function SpeciesLegend({ species, swatchColor }: SpeciesLegendProps) {
  const warning = species.sample_size_warning;
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span
        aria-hidden="true"
        className="inline-block h-3 w-3 rounded-full"
        style={{ backgroundColor: swatchColor }}
      />
      <span className="font-medium">{normalizeLabel(species.label)}</span>
      <span className="tabular-nums text-muted-foreground">
        n = {species.n}
      </span>
      <TooltipProvider delayDuration={200}>
        <UITooltip>
          <TooltipTrigger asChild>
            <Badge variant="secondary" className="cursor-help font-normal">
              {DIEL_LABEL[species.diel_class]} ·{" "}
              {dominantPhase(species.diel_class, species.diel_density_by_phase)}
            </Badge>
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-xs">
            <p>{DIEL_TOOLTIP_RULE}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              day {(species.diel_density_by_phase.day ?? 0) * 100} %
              · twilight {(species.diel_density_by_phase.twilight ?? 0) * 100} %
              · night {(species.diel_density_by_phase.night ?? 0) * 100} %
            </p>
          </TooltipContent>
        </UITooltip>
      </TooltipProvider>
      {warning && (
        <Badge
          variant={SAMPLE_WARNING_CRITICAL[warning] ? "destructive" : "outline"}
          className={
            SAMPLE_WARNING_CRITICAL[warning]
              ? undefined
              : "border-amber-300 bg-amber-50 text-amber-900"
          }
        >
          {SAMPLE_WARNING_LABEL[warning]}
        </Badge>
      )}
    </div>
  );
}

interface OverlapReadoutProps {
  data: ActivityOverlapResponse;
}

function OverlapReadout({ data }: OverlapReadoutProps) {
  if (!data.overlap) return null;
  const { delta, ci_low, ci_high, bootstrap_reps, min_n, delta_estimator } =
    data.overlap;
  const symbol = delta_estimator === "delta1" ? "Δ₁" : "Δ₄";
  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 rounded-md border bg-muted/30 px-4 py-3 text-sm">
      <span className="text-base font-semibold tabular-nums">
        {symbol} = {delta.toFixed(3)}
      </span>
      <span className="text-muted-foreground tabular-nums">
        95% CI: {ci_low.toFixed(3)} – {ci_high.toFixed(3)}
      </span>
      <span className="text-xs text-muted-foreground">
        bootstrap {bootstrap_reps} reps · n_min = {min_n}
      </span>
    </div>
  );
}

export function ActivityOverlapPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();

  const filters = useMemo<ActivityOverlapPageFilters>(() => {
    const parsed = filtersFromSearchParams(searchParams, FILTER_SCHEMA);
    return {
      speciesA: (parsed.species_a as string | undefined) ?? null,
      speciesB: (parsed.species_b as string | undefined) ?? null,
      siteIds: (parsed.site_ids as string[] | undefined) ?? [],
      dateFrom: (parsed.date_from as string | undefined) ?? null,
      dateTo: (parsed.date_to as string | undefined) ?? null,
      timeAxis: parseTimeAxis(parsed.time_axis as string | undefined),
      bandsVisible: parseBandsVisible(parsed.bands_visible as string | undefined),
    };
  }, [searchParams]);

  const handleFiltersChange = (next: ActivityOverlapPageFilters) => {
    setSearchParams(
      filtersToSearchParams(
        {
          species_a: next.speciesA ?? undefined,
          species_b: next.speciesB ?? undefined,
          site_ids: next.siteIds.length > 0 ? next.siteIds : undefined,
          date_from: next.dateFrom ?? undefined,
          date_to: next.dateTo ?? undefined,
          time_axis: next.timeAxis === "sun" ? "sun" : undefined,
          bands_visible: next.bandsVisible ? undefined : "false",
        },
        FILTER_SCHEMA,
      ),
    );
  };

  const enabled = !!projectId && !!filters.speciesA;

  const { data, isLoading, isFetching } = useQuery({
    queryKey: [
      "statistics",
      "activity-overlap",
      projectId,
      filters.speciesA,
      filters.speciesB,
      filters.siteIds.join(","),
      filters.dateFrom,
      filters.dateTo,
    ],
    queryFn: () =>
      statisticsApi.getActivityOverlap(projectId!, {
        speciesA: filters.speciesA!,
        speciesB: filters.speciesB ?? undefined,
        siteIds: filters.siteIds.length > 0 ? filters.siteIds : undefined,
        dateFrom: filters.dateFrom ?? undefined,
        dateTo: filters.dateTo ?? undefined,
      }),
    enabled,
  });

  if (!projectId) {
    return <div>Project ID missing</div>;
  }

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold tracking-tight">Activity overlap</h1>
          <p className="text-sm text-muted-foreground">
            Compare daily activity patterns between species using a von
            Mises kernel density estimate. The Δ overlap coefficient
            (Ridout &amp; Linkie 2009) and bootstrap 95% CI are shown
            when two species are selected.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-4 py-8 sm:px-6 lg:px-8">
        <ActivityOverlapFilterBar
          projectId={projectId}
          filters={filters}
          onChange={handleFiltersChange}
        />

        {!enabled && (
          <div className="rounded-lg border border-dashed bg-muted/20 p-12 text-center text-sm text-muted-foreground">
            Pick a species in the dropdowns above to start.
          </div>
        )}

        {enabled && (isLoading || isFetching) && !data && (
          <div className="flex items-center justify-center gap-2 rounded-lg border bg-card p-12 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Computing activity densities...
          </div>
        )}

        {enabled && data && (
          <div className="space-y-4 rounded-lg border bg-card p-4">
            <ActivityOverlapChart
              data={data}
              timeAxis={filters.timeAxis}
              bandsVisible={filters.bandsVisible}
            />

            {data.overlap && <OverlapReadout data={data} />}

            <div className="space-y-2 border-t pt-3">
              <SpeciesLegend
                species={data.species_a}
                swatchColor={SPECIES_A_COLOR}
              />
              {data.species_b && (
                <SpeciesLegend
                  species={data.species_b}
                  swatchColor={SPECIES_B_COLOR}
                />
              )}
            </div>

            <div className="flex items-center gap-1.5 border-t pt-3 text-xs text-muted-foreground">
              <Info className="h-3.5 w-3.5" />
              Independence interval:{" "}
              {Math.round(data.independence_interval_seconds / 60)} min
              (project setting)
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default ActivityOverlapPage;
