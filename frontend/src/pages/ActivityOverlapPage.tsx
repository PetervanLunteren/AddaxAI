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

import { useEffect, useMemo } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Info, Loader2 } from "lucide-react";

import { sitesApi } from "../api/sites";
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
import {
  PlotExplainer,
  type PlotReference,
} from "../components/plots/PlotExplainer";
import {
  InsightsFilterChips,
  buildSiteNameMap,
  dateChips,
  siteChips,
} from "../components/plots/InsightsFilterChips";
import { NoSiteBanner } from "../components/deployments/NoSiteBanner";
import { MissingDatesBanner } from "../components/dashboard/MissingDatesWarning";
import { Badge } from "../components/ui/badge";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../components/ui/tooltip";
import { formatCameraDate } from "../lib/datetime";
import {
  filtersFromSearchParams,
  filtersToSearchParams,
  type FilterSchema,
} from "../lib/filter-url";
import { resolveSpeciesName } from "../lib/species-name-mode";

const FILTER_SCHEMA: FilterSchema = {
  species_a: "string",
  species_b: "string",
  site_ids: "string[]",
  date_from: "date",
  date_to: "date",
  time_axis: "string",
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

// -- Scientific explainer content for the "About this view" section --

const EXPLAINER_REFERENCES: PlotReference[] = [
  {
    citation:
      "Ridout, M. S., & Linkie, M. (2009). Estimating overlap of daily " +
      "activity patterns from camera trap data. Journal of Agricultural, " +
      "Biological, and Environmental Statistics, 14(3), 322–337.",
    url: "https://link.springer.com/article/10.1198/jabes.2009.08038",
  },
  {
    citation:
      "Vazquez, C., Rowcliffe, J. M., Spoelstra, K., & Jansen, P. A. " +
      "(2019). Comparing diel activity patterns of wildlife across " +
      "latitudes and seasons: time transformations using day length. " +
      "Methods in Ecology and Evolution, 10(12), 2057–2066.",
    url: "https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13290",
  },
  {
    citation:
      "Bennie, J. J., Duffy, J. P., Inger, R., & Gaston, K. J. (2014). " +
      "Biogeography of time partitioning in mammals. PNAS, 111(38), " +
      "13727–13732.",
    url: "https://www.pnas.org/doi/10.1073/pnas.1216063110",
  },
];

function parseTimeAxis(raw: string | undefined): TimeAxis {
  return raw === "sun" ? "sun" : "clock";
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
  /** Common name for this species' key, when known (the overlap response
   *  only carries the scientific-preferring key). */
  commonName?: string | null;
}

function SpeciesLegend({ species, swatchColor, commonName }: SpeciesLegendProps) {
  const warning = species.sample_size_warning;
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span
        aria-hidden="true"
        className="inline-block h-3 w-3 rounded-full"
        style={{ backgroundColor: swatchColor }}
      />
      <span className="font-medium">
        {resolveSpeciesName({
          scientific_name: species.label,
          common_name: commonName,
        })}
      </span>
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
              day {((species.diel_density_by_phase.day ?? 0) * 100).toFixed(1)} %
              · twilight{" "}
              {((species.diel_density_by_phase.twilight ?? 0) * 100).toFixed(1)} %
              · night {((species.diel_density_by_phase.night ?? 0) * 100).toFixed(1)} %
            </p>
          </TooltipContent>
        </UITooltip>
      </TooltipProvider>
      {warning && (
        <Badge
          variant="outline"
          className="border-transparent text-white"
          style={{
            backgroundColor: SAMPLE_WARNING_CRITICAL[warning]
              ? "#882000"
              : "#71b7ba",
          }}
        >
          {SAMPLE_WARNING_LABEL[warning]}
        </Badge>
      )}
      {species.dropped_polar > 0 && (
        <Badge
          variant="outline"
          className="border-transparent text-white"
          style={{ backgroundColor: "#71b7ba" }}
        >
          {species.dropped_polar} dropped (polar date)
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
    };
  }, [searchParams]);

  // Species A and B auto-pick: each slot's decision is persisted via
  // sessionStorage, keyed per project. Survives navigating away and
  // back within a session, but clears on app/tab close so a fresh
  // session re-evaluates the most-observed defaults.
  //
  // Storage value semantics (one key per slot):
  //   - missing  = never attempted (fire auto-pick when URL is empty)
  //   - ""       = attempted, the slot is deliberately empty now
  //                (zero candidates, or user cleared the picker)
  //   - "Tiger"  = the current value
  //
  // Two write helpers below: `writeUrl` updates only the URL,
  // `writeFilters` updates URL + storage for both slots. Restore-
  // from-session uses `writeUrl` so filling A from storage does not
  // poison B's "never decided" sentinel (and vice versa). User
  // changes and auto-pick decisions go through `writeFilters` so the
  // decision they make is durable.
  const storageKeyA = projectId
    ? `addaxai:plots:activity-overlap:species-a:${projectId}`
    : null;
  const storageKeyB = projectId
    ? `addaxai:plots:activity-overlap:species-b:${projectId}`
    : null;

  const readStored = (key: string | null): string | null => {
    if (!key) return null;
    try {
      return sessionStorage.getItem(key);
    } catch {
      return null;
    }
  };

  const writeStored = (key: string | null, value: string) => {
    if (!key) return;
    try {
      sessionStorage.setItem(key, value);
    } catch {
      /* ignore — private mode or quota, not fatal */
    }
  };

  const writeUrl = (
    next: ActivityOverlapPageFilters,
    options?: { replace?: boolean },
  ) => {
    setSearchParams(
      filtersToSearchParams(
        {
          species_a: next.speciesA ?? undefined,
          species_b: next.speciesB ?? undefined,
          site_ids: next.siteIds.length > 0 ? next.siteIds : undefined,
          date_from: next.dateFrom ?? undefined,
          date_to: next.dateTo ?? undefined,
          time_axis: next.timeAxis === "sun" ? "sun" : undefined,
        },
        FILTER_SCHEMA,
      ),
      options,
    );
  };

  const writeFilters = (
    next: ActivityOverlapPageFilters,
    options?: { replace?: boolean },
  ) => {
    writeUrl(next, options);
    writeStored(storageKeyA, next.speciesA ?? "");
    writeStored(storageKeyB, next.speciesB ?? "");
  };

  const handleFiltersChange = (next: ActivityOverlapPageFilters) => {
    writeFilters(next);
  };

  // Only trigger the species-distribution API call when at least one
  // slot has no URL value AND no storage decision. A stored "" means
  // "deliberately empty" and blocks auto-pick for that slot.
  const hasStoredA = readStored(storageKeyA) !== null;
  const hasStoredB = readStored(storageKeyB) !== null;
  const siteIdsCsv = filters.siteIds.length > 0 ? filters.siteIds.join(",") : undefined;
  const shouldAutoPickA =
    !!projectId && filters.speciesA === null && !hasStoredA;
  const shouldAutoPickB =
    !!projectId && filters.speciesB === null && !hasStoredB;

  const { data: autoPickCandidates } = useQuery({
    queryKey: [
      "statistics",
      "species",
      "auto-pick",
      projectId,
      siteIdsCsv,
      filters.dateFrom,
      filters.dateTo,
    ],
    queryFn: () =>
      statisticsApi.getSpeciesDistribution(
        projectId!,
        siteIdsCsv,
        filters.dateFrom ?? undefined,
        filters.dateTo ?? undefined,
      ),
    enabled: shouldAutoPickA || shouldAutoPickB,
  });

  // Species distribution again, but always on, purely to map the
  // scientific-preferring key (what the overlap response and pickers use)
  // to its common name so the legend + chart can follow the display
  // preference. Same query key as SpeciesPicker, so this is a cache hit
  // (no extra request).
  const { data: speciesNameList } = useQuery({
    queryKey: [
      "statistics",
      "species",
      "picker",
      projectId,
      siteIdsCsv,
      filters.dateFrom ?? undefined,
      filters.dateTo ?? undefined,
      undefined,
    ],
    queryFn: () =>
      statisticsApi.getSpeciesDistribution(
        projectId!,
        siteIdsCsv,
        filters.dateFrom ?? undefined,
        filters.dateTo ?? undefined,
      ),
    enabled: !!projectId,
  });

  const commonNameByKey = useMemo(
    () =>
      new Map(
        (speciesNameList ?? []).map((s) => [s.species, s.common_name]),
      ),
    [speciesNameList],
  );

  // Restore-from-session: URL empty but session remembers a species →
  // write it back to the URL. URL-only update so we do not overwrite
  // the "never decided" sentinel for any slot we did not just restore.
  useEffect(() => {
    if (!projectId) return;
    if (filters.speciesA !== null && filters.speciesB !== null) return;
    const storedA = readStored(storageKeyA);
    const storedB = readStored(storageKeyB);
    const nextA =
      filters.speciesA === null && storedA && storedA !== ""
        ? storedA
        : filters.speciesA;
    const nextB =
      filters.speciesB === null && storedB && storedB !== ""
        ? storedB
        : filters.speciesB;
    if (nextA !== filters.speciesA || nextB !== filters.speciesB) {
      writeUrl(
        { ...filters, speciesA: nextA, speciesB: nextB },
        { replace: true },
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, filters.speciesA, filters.speciesB]);

  // Auto-pick once per session per project. A = most-observed species;
  // B = next most-observed species not equal to A, so the chart
  // immediately shows an overlap instead of an empty B picker. Writes
  // URL + storage for both slots in one go. A null result writes ""
  // ("deliberately empty") so we do not retry.
  useEffect(() => {
    if (!autoPickCandidates) return;
    if (!shouldAutoPickA && !shouldAutoPickB) return;
    const sorted = [...autoPickCandidates].sort((a, b) => b.count - a.count);
    let nextA = filters.speciesA;
    let nextB = filters.speciesB;
    if (shouldAutoPickA) {
      nextA = sorted.length > 0 ? sorted[0].species : null;
    }
    if (shouldAutoPickB) {
      const candidate = sorted.find((s) => s.species !== nextA);
      nextB = candidate ? candidate.species : null;
    }
    writeFilters(
      { ...filters, speciesA: nextA, speciesB: nextB },
      { replace: true },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shouldAutoPickA, shouldAutoPickB, autoPickCandidates]);

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
      filters.timeAxis,
    ],
    queryFn: () =>
      statisticsApi.getActivityOverlap(projectId!, {
        speciesA: filters.speciesA!,
        speciesB: filters.speciesB ?? undefined,
        siteIds: filters.siteIds.length > 0 ? filters.siteIds : undefined,
        dateFrom: filters.dateFrom ?? undefined,
        dateTo: filters.dateTo ?? undefined,
        timeAxis: filters.timeAxis,
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
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Activity overlap</h1>
              <p className="text-sm text-muted-foreground">
                Compare daily activity patterns between species
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-4 py-8 sm:px-6 lg:px-8">
        {filters.timeAxis === "sun" && data && data.anchor_sun_bands === null && (
          <NoSiteBanner
            projectId={projectId}
            message="Sun-time mode needs at least one camera site with GPS coordinates. Assign a location to a site or switch the time axis to clock-time."
          />
        )}
        <MissingDatesBanner projectId={projectId} />
        <ActivityOverlapFilterBar
          projectId={projectId}
          filters={filters}
          onChange={handleFiltersChange}
        />
        <ActivityOverlapChips
          projectId={projectId}
          filters={filters}
          onChange={handleFiltersChange}
        />

        {!enabled && (
          <div className="rounded-lg border bg-card p-8 text-center space-y-2">
            <div className="text-sm font-medium text-foreground">
              No activity to show
            </div>
            <div className="text-sm text-muted-foreground max-w-xl mx-auto">
              Pick a label in the dropdowns above to start.
            </div>
          </div>
        )}

        {enabled && (isLoading || isFetching) && !data && (
          <div className="flex items-center justify-center gap-2 rounded-lg border bg-card p-12 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Computing activity densities...
          </div>
        )}

        {enabled && data && (
          <div className="flex h-[600px] flex-col space-y-4 rounded-lg border bg-card p-4">
            <div className="min-h-0 flex-1">
              <ActivityOverlapChart
                data={data}
                speciesAName={resolveSpeciesName({
                  scientific_name: data.species_a.label,
                  common_name: commonNameByKey.get(data.species_a.label),
                })}
                speciesBName={
                  data.species_b
                    ? resolveSpeciesName({
                        scientific_name: data.species_b.label,
                        common_name: commonNameByKey.get(data.species_b.label),
                      })
                    : undefined
                }
              />
            </div>

            {data.overlap && <OverlapReadout data={data} />}

            <div className="space-y-2 border-t pt-3">
              <SpeciesLegend
                species={data.species_a}
                swatchColor={SPECIES_A_COLOR}
                commonName={commonNameByKey.get(data.species_a.label)}
              />
              {data.species_b && (
                <SpeciesLegend
                  species={data.species_b}
                  swatchColor={SPECIES_B_COLOR}
                  commonName={commonNameByKey.get(data.species_b.label)}
                />
              )}
            </div>

            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t pt-3 text-xs text-muted-foreground">
              <Info className="h-3.5 w-3.5" />
              <span>Timezone: {data.project_timezone}</span>
              <span aria-hidden="true">·</span>
              <span>
                Independence interval:{" "}
                {Math.round(data.independence_interval_seconds / 60)} min
              </span>
              {data.time_axis === "clock" && data.sun_bands_reference_date && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>
                    Sun bands drawn for midpoint (
                    {formatCameraDate(data.sun_bands_reference_date)})
                  </span>
                </>
              )}
            </div>
          </div>
        )}

        <PlotExplainer
          plotKey="activity-overlap"
          what={
            <p>
              Two smooth curves, one per species, showing how often
              each species was detected at each hour of the day. Each
              curve is normalised so the area under it sums to 1. The
              peaks show when the species is active, not how many
              detections there were. When two species are picked, the
              shaded region between the curves is the overlap
              coefficient Δ. Rug ticks at the bottom show the raw
              detection times behind each curve.
            </p>
          }
          how={
            <p>
              Detections are grouped into events using the project's
              independence interval, and the MaxN per species in each
              event becomes the sample count at that event's time of
              day. Samples are fit with a von Mises circular kernel
              density on a 240-point grid over [0, 24) hours (κ = 5).
              The overlap coefficient Δ (Ridout &amp; Linkie 2009) =
              ∫ min(f<sub>a</sub>, f<sub>b</sub>) dt over the full day, with the label
              flipping between Δ<sub>1</sub> and Δ<sub>4</sub> based on
              the smaller sample size (Δ<sub>4</sub> for min-N ≥ 50,
              Δ<sub>1</sub> below) following the `overlap` R package
              convention. A 95% percentile bootstrap CI comes from
              1000 resamples with a fixed seed. Diel classification
              follows Bennie et al. 2014: when ≥ 70% of activity
              density falls in one phase (day, twilight, or night, as
              defined by the project's sun bands), the species gets
              that label, otherwise it is labelled cathemeral. When
              sun time is selected, each detection's clock hour is
              transformed via the Vazquez et al. 2019 double-anchored
              mapping: that day's sunrise and sunset (from astral, in
              the project's timezone) are stretched or compressed to
              match the dataset's mean sunrise and sunset. Observations
              on dates without a defined sunrise (polar night or day)
              are dropped and counted in the legend.
            </p>
          }
          references={EXPLAINER_REFERENCES}
        />
      </main>
    </div>
  );
}

export default ActivityOverlapPage;

/**
 * Active-filter chip row scoped to the data filters. Species pickers
 * and the time-axis toggle are excluded — they're required selections
 * or display modes rather than removable filters. "Clear all" resets
 * sites and date range while keeping the species and time axis.
 */
function ActivityOverlapChips({
  projectId,
  filters,
  onChange,
}: {
  projectId: string;
  filters: ActivityOverlapPageFilters;
  onChange: (next: ActivityOverlapPageFilters) => void;
}) {
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId,
  });

  const chips = useMemo(() => {
    const siteNames = buildSiteNameMap(sites);
    return [
      ...siteChips(filters.siteIds, siteNames, (next) =>
        onChange({ ...filters, siteIds: next }),
      ),
      ...dateChips(
        filters.dateFrom,
        filters.dateTo,
        () => onChange({ ...filters, dateFrom: null }),
        () => onChange({ ...filters, dateTo: null }),
      ),
    ];
  }, [filters, sites, onChange]);

  return (
    <InsightsFilterChips
      chips={chips}
      onClearAll={() =>
        onChange({ ...filters, siteIds: [], dateFrom: null, dateTo: null })
      }
    />
  );
}
