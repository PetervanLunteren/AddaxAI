/**
 * Confusion matrix insight view.
 *
 * Per-detection comparison of what the model originally predicted vs
 * what the user ended up labelling. Only verified detections count.
 * Rolling up to a higher taxonomic rank is supported via the rank
 * filter; large class lists collapse a tail into an 'other' bucket
 * via the top-N filter.
 */

import { useMemo } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { performanceApi } from "../api/performance";
import { sitesApi } from "../api/sites";
import { ConfusionMatrix } from "../components/plots/ConfusionMatrix";
import {
  buildSiteNameMap,
  siteChips,
} from "../components/plots/InsightsFilterChips";
import {
  PerformanceFilterBar,
  type MatrixMode,
  type PerformancePageFilters,
  type TopN,
} from "../components/plots/PerformanceFilterBar";
import { PlotExplainer } from "../components/plots/PlotExplainer";
import {
  filtersFromSearchParams,
  filtersToSearchParams,
  type FilterSchema,
} from "../lib/filter-url";
import { DEFAULT_TAXONOMIC_RANK, isTaxonomicRank } from "../lib/taxonomic-rank";
import { PageHeader } from "../components/layout/PageHeader";
import { DATA_CSV, DownloadMenu } from "../components/plots/DownloadMenu";
import { confusionMatrixCsv, performanceFilters, saveCsv } from "../lib/insights-csv";

const FILTER_SCHEMA: FilterSchema = {
  site_ids: "string[]",
  taxonomic_rank: "string",
  top_n: "string",
  mode: "string",
};

function isTopN(value: string | undefined): value is TopN {
  return value === "10" || value === "20" || value === "all";
}

function isMode(value: string | undefined): value is MatrixMode {
  return value === "counts" || value === "recall" || value === "precision";
}

export function ConfusionMatrixPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();

  const filters = useMemo<PerformancePageFilters>(() => {
    const parsed = filtersFromSearchParams(searchParams, FILTER_SCHEMA);
    const rawRank = parsed.taxonomic_rank as string | undefined;
    const rawMode = parsed.mode as string | undefined;
    return {
      siteIds: (parsed.site_ids as string[] | undefined) ?? [],
      taxonomicRank: isTaxonomicRank(rawRank) ? rawRank : DEFAULT_TAXONOMIC_RANK,
      topN: isTopN(parsed.top_n as string | undefined)
        ? (parsed.top_n as TopN)
        : "20",
      mode: isMode(rawMode) ? rawMode : "counts",
    };
  }, [searchParams]);

  const handleFiltersChange = (next: PerformancePageFilters) => {
    setSearchParams(
      filtersToSearchParams(
        {
          site_ids: next.siteIds,
          taxonomic_rank: next.taxonomicRank,
          top_n: next.topN,
          mode: next.mode === "counts" ? undefined : next.mode,
        },
        FILTER_SCHEMA,
      ),
    );
  };

  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId!),
    enabled: !!projectId,
  });
  const chips = useMemo(() => {
    const siteNames = buildSiteNameMap(sites);
    return siteChips(filters.siteIds, siteNames, (next) =>
      handleFiltersChange({ ...filters, siteIds: next }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, sites]);
  const clearAllDataFilters = () =>
    handleFiltersChange({ ...filters, siteIds: [] });

  const siteKey = filters.siteIds.slice().sort().join(",");
  const enabled = !!projectId;

  const { data, isLoading, isFetching } = useQuery({
    enabled,
    queryKey: [
      "statistics",
      "performance",
      projectId,
      siteKey,
      filters.taxonomicRank,
      filters.topN,
    ],
    queryFn: () =>
      performanceApi.get(projectId!, {
        siteIds: filters.siteIds,
        taxonomicRank: filters.taxonomicRank,
        topN: filters.topN,
      }),
  });

  if (!projectId) return null;

  return (
    <div className="min-h-screen">
      <PageHeader
        title="Confusion matrix"
        subtitle="Agreement between AI and human labels"
        actions={
          // Nothing verified yet means nothing to save: the page shows
          // its empty state instead of a view.
          <DownloadMenu
            items={
              data?.grand_total
                ? [
                    {
                      label: DATA_CSV,
                      onSelect: () =>
                        saveCsv(
                          "confusion-matrix",
                          confusionMatrixCsv(
                            data,
                            performanceFilters({
                              siteIds: filters.siteIds,
                              siteNames: buildSiteNameMap(sites),
                              taxonomicRank: filters.taxonomicRank,
                              topN: filters.topN,
                            }),
                          ),
                        ),
                    },
                  ]
                : []
            }
          />
        }
      />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <PerformanceFilterBar
          projectId={projectId}
          filters={filters}
          onChange={handleFiltersChange}
          showModeToggle
          chips={chips}
          onClearAll={clearAllDataFilters}
        />

        <ConfusionMatrix
          data={data}
          loading={isLoading || isFetching}
          mode={filters.mode}
        />

        <PlotExplainer
          plotKey="confusion-matrix"
          what={
            <p>
              One cell per detection. Row is the current label after human
              verification. Column is what the AI originally predicted.
              Diagonal cells are agreements. Off-diagonal cells show which
              class the AI mistook for which. Cell colour is darker
              when the cell dominates its row (or its column in precision
              mode), so the strong pairs stand out.
            </p>
          }
          how={
            <p>
              Only verified detections count. The predicted column comes from
              the raw AI output captured at analysis time and is never
              changed by rollup, smoothing, or relabels. Classes are grouped
              at the selected taxonomic rank. The top-N filter keeps the
              biggest classes and folds the rest into an "Other" row and
              column so the totals still add up.
            </p>
          }
        />
      </main>
    </div>
  );
}

export default ConfusionMatrixPage;
