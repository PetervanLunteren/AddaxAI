/**
 * Classification report insight view.
 *
 * Per-class precision / recall / F1 / support derived from the same
 * data as the confusion matrix, plus macro and weighted averages. Row
 * click deep-links to the Verify page filtered to that class.
 */

import { useMemo } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { performanceApi } from "../api/performance";
import { ClassificationReportTable } from "../components/plots/ClassificationReportTable";
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

export function ClassificationReportPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

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

  const handleRowClick = (args: {
    className: string;
    taxonomyId: string | null;
  }) => {
    if (args.className === "other") return;
    const params = new URLSearchParams();
    if (args.taxonomyId) params.set("labels", args.taxonomyId);
    const qs = params.toString();
    navigate(`/projects/${projectId}/verify${qs ? `?${qs}` : ""}`);
  };

  return (
    <>
      <header className="border-b bg-white/80 backdrop-blur-sm px-4 py-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <h1 className="text-2xl font-bold tracking-tight">
            Classification report
          </h1>
          <p className="text-sm text-muted-foreground">
            Per-class precision, recall, and F1 derived from the confusion matrix.
          </p>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <PerformanceFilterBar
          projectId={projectId}
          filters={filters}
          onChange={handleFiltersChange}
        />

        <ClassificationReportTable
          data={data}
          loading={isLoading || isFetching}
          onRowClick={handleRowClick}
        />

        <PlotExplainer
          plotKey="classification-report"
          what={
            <p>
              One row per class. Support is the number of verified detections
              with that class as the current label. Precision is the share of
              detections predicted as this class that really were it. Recall
              is the share of real detections of this class the classifier
              caught. F1 is the harmonic mean of precision and recall. The F1
              column uses the project status palette, so weak classes are
              easy to spot. Macro averages treat every class the same.
              Weighted averages use support. Click a row to open the matching
              detections in the Verify page.
            </p>
          }
          how={
            <p>
              The metrics come from the same verified-detection pairs as the
              confusion matrix, computed server-side. Predictions come from
              the raw classifier output captured at analysis time. Classes
              are grouped at the selected taxonomic rank. The top-N filter
              folds smaller classes into an "Other" row, which is shown but
              not clickable.
            </p>
          }
        />
      </main>
    </>
  );
}

export default ClassificationReportPage;
