/**
 * Per-class performance insight view.
 *
 * Precision, recall, F1, and support per class, plus macro and weighted
 * averages, computed from the same verified-detection pairs as the
 * confusion matrix.
 */

import { useMemo } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { performanceApi } from "../api/performance";
import { sitesApi } from "../api/sites";
import { PerClassPerformanceTable } from "../components/plots/PerClassPerformanceTable";
import {
  InsightsFilterChips,
  buildSiteNameMap,
  siteChips,
} from "../components/plots/InsightsFilterChips";
import {
  PerformanceFilterBar,
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

const FILTER_SCHEMA: FilterSchema = {
  site_ids: "string[]",
  taxonomic_rank: "string",
  top_n: "string",
};

function isTopN(value: string | undefined): value is TopN {
  return value === "10" || value === "20" || value === "all";
}

export function PerClassPerformancePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();

  const filters = useMemo<PerformancePageFilters>(() => {
    const parsed = filtersFromSearchParams(searchParams, FILTER_SCHEMA);
    const rawRank = parsed.taxonomic_rank as string | undefined;
    return {
      siteIds: (parsed.site_ids as string[] | undefined) ?? [],
      taxonomicRank: isTaxonomicRank(rawRank) ? rawRank : DEFAULT_TAXONOMIC_RANK,
      topN: isTopN(parsed.top_n as string | undefined)
        ? (parsed.top_n as TopN)
        : "20",
      // Report is inherently a ratio view — the mode is ignored here.
      mode: "counts",
    };
  }, [searchParams]);

  const handleFiltersChange = (next: PerformancePageFilters) => {
    setSearchParams(
      filtersToSearchParams(
        {
          site_ids: next.siteIds,
          taxonomic_rank: next.taxonomicRank,
          top_n: next.topN,
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
    <>
      <header className="border-b bg-white/80 backdrop-blur-sm px-4 py-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">
                Per-class performance
              </h1>
              <p className="text-sm text-muted-foreground">
                Precision, recall, and F1 score per class
              </p>
            </div>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <PerformanceFilterBar
          projectId={projectId}
          filters={filters}
          onChange={handleFiltersChange}
        />
        <InsightsFilterChips chips={chips} onClearAll={clearAllDataFilters} />

        <PerClassPerformanceTable
          data={data}
          loading={isLoading || isFetching}
        />

        <PlotExplainer
          plotKey="per-class-performance"
          what={
            <p>
              One row per class. Support counts the verified detections for
              each class. Precision shows how often the AI was right when
              it predicted the class. Recall shows how often the AI caught
              the real detections of the class. F1 combines precision and
              recall into one number. Macro averages give every class the
              same weight. Weighted averages give bigger classes more
              weight.
            </p>
          }
          how={
            <p>
              Same verified detections as the confusion matrix. The AI
              prediction is the raw top-1 saved at analysis time, not the
              post-processed label. Classes are grouped at the selected
              taxonomic rank. The top-N filter keeps the biggest classes
              and puts the rest in an "Other" row. The Other row shows in
              the table but is not counted in the macro or weighted
              averages.
            </p>
          }
        />
      </main>
    </>
  );
}

export default PerClassPerformancePage;
