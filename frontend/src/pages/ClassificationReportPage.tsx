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

import { performanceApi, type PerformanceRank } from "../api/performance";
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

const FILTER_SCHEMA: FilterSchema = {
  site_ids: "string[]",
  date_from: "date",
  date_to: "date",
  rank: "string",
  top_n: "string",
};

function isRank(value: string | undefined): value is PerformanceRank {
  return (
    value === "class" ||
    value === "order" ||
    value === "family" ||
    value === "genus" ||
    value === "species"
  );
}

function isTopN(value: string | undefined): value is TopN {
  return value === "10" || value === "20" || value === "50" || value === "all";
}

export function ClassificationReportPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const filters = useMemo<PerformancePageFilters>(() => {
    const parsed = filtersFromSearchParams(searchParams, FILTER_SCHEMA);
    return {
      siteIds: (parsed.site_ids as string[] | undefined) ?? [],
      dateFrom: (parsed.date_from as string | undefined) ?? null,
      dateTo: (parsed.date_to as string | undefined) ?? null,
      rank: isRank(parsed.rank as string | undefined)
        ? (parsed.rank as PerformanceRank)
        : "species",
      topN: isTopN(parsed.top_n as string | undefined)
        ? (parsed.top_n as TopN)
        : "20",
    };
  }, [searchParams]);

  const handleFiltersChange = (next: PerformancePageFilters) => {
    setSearchParams(
      filtersToSearchParams(
        {
          site_ids: next.siteIds,
          date_from: next.dateFrom ?? undefined,
          date_to: next.dateTo ?? undefined,
          rank: next.rank,
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
      filters.dateFrom,
      filters.dateTo,
      filters.rank,
      filters.topN,
    ],
    queryFn: () =>
      performanceApi.get(projectId!, {
        siteIds: filters.siteIds,
        dateFrom: filters.dateFrom ?? undefined,
        dateTo: filters.dateTo ?? undefined,
        rank: filters.rank,
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

        {data && (data.skipped_no_prediction > 0 || data.skipped_unverified > 0) && (
          <div className="text-xs text-muted-foreground">
            {data.skipped_no_prediction > 0 && (
              <p>
                {data.skipped_no_prediction} verified detection
                {data.skipped_no_prediction === 1 ? "" : "s"} excluded because no
                original machine prediction is on record.
              </p>
            )}
            {data.skipped_unverified > 0 && (
              <p>
                {data.skipped_unverified} detection
                {data.skipped_unverified === 1 ? "" : "s"} in the filtered range
                are not yet verified; only verified detections count toward the
                report.
              </p>
            )}
          </div>
        )}

        <PlotExplainer
          plotKey="classification-report"
          what={
            <p>
              Each row reports how the classifier performed on a given class.
              Support is the number of verified detections with that class as the
              current label, precision is the fraction of detections predicted as
              this class that really were it, recall is the fraction that really
              were this class that the classifier caught, and F1 combines
              precision and recall as their harmonic mean. The F1 column is
              colour-scaled across the project status palette so strong and weak
              classes are easy to spot. Macro averages ignore class size;
              weighted averages use support. Click any row to open the
              underlying detections in the Verify page.
            </p>
          }
          how={
            <p>
              Metrics are computed server-side from the same per-detection
              confusion counts as the matrix. Only verified detections count.
              Predictions come from the original_label column captured at JSON
              load time. Classes roll up to the chosen taxonomic rank via the
              label_taxonomy table; the top-N filter folds smaller classes into
              an "other" bucket, which always appears in the table but is not
              clickable.
            </p>
          }
          caveats={
            <ul className="list-disc space-y-1 pl-5">
              <li>
                <span className="font-medium text-foreground">Detection vs classification errors.</span>{" "}
                Rows for animal, person, and vehicle reflect detector decisions,
                not classifier ones. A low recall on animal doesn't mean the
                species classifier missed animals, it means the detector didn't
                box them.
              </li>
              <li>
                <span className="font-medium text-foreground">Missed animals are invisible.</span>{" "}
                The report scores classifications against a ground truth that
                only exists when the detector drew a box and a human reviewed it.
                Animals missed by the detector don't appear at all.
              </li>
              <li>
                <span className="font-medium text-foreground">Class imbalance.</span>{" "}
                Macro averages weight every class equally even when some have
                very small support. Tiny classes can look misleadingly bad; lean
                on the weighted average when the dataset is skewed.
              </li>
              <li>
                <span className="font-medium text-foreground">Detector-only projects.</span>{" "}
                With no classifier configured the report reduces to animal,
                person, vehicle. Attach a classification model to get
                species-level metrics.
              </li>
            </ul>
          }
          settings={[
            {
              label: "Detection threshold",
              detail:
                "Low-confidence detections are hidden from the Verify page but stay eligible for this report once verified.",
            },
            {
              label: "Taxonomic rollup",
              detail:
                "Turning it off keeps labels at the leaf. Turning it on means family- or genus-level rolled-up labels land in the same row as their taxon.",
            },
            {
              label: "Excluded classes",
              detail:
                "Classes in the project exclusion list never reach the database and so never appear in the report.",
            },
          ]}
        />
      </main>
    </>
  );
}

export default ClassificationReportPage;
