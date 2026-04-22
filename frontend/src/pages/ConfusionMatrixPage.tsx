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
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { performanceApi, type PerformanceRank } from "../api/performance";
import { ConfusionMatrix } from "../components/plots/ConfusionMatrix";
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

export function ConfusionMatrixPage() {
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

  const handleCellClick = (args: {
    rowClass: string;
    rowTaxonomyId: string | null;
    colClass: string;
  }) => {
    const params = new URLSearchParams();
    if (args.rowTaxonomyId) params.set("labels", args.rowTaxonomyId);
    if (args.colClass !== "other") params.set("original_label", args.colClass);
    const qs = params.toString();
    navigate(`/projects/${projectId}/verify${qs ? `?${qs}` : ""}`);
  };

  return (
    <>
      <header className="border-b bg-white/80 backdrop-blur-sm px-4 py-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <h1 className="text-2xl font-bold tracking-tight">Confusion matrix</h1>
          <p className="text-sm text-muted-foreground">
            Where the classifier and the human labels agree, and where they don't.
          </p>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <PerformanceFilterBar
          projectId={projectId}
          filters={filters}
          onChange={handleFiltersChange}
        />

        <ConfusionMatrix
          data={data}
          loading={isLoading || isFetching}
          onCellClick={handleCellClick}
        />

        {data && (data.skipped_no_prediction > 0 || data.skipped_unverified > 0) && (
          <div className="text-xs text-muted-foreground">
            {data.skipped_no_prediction > 0 && (
              <p>
                {data.skipped_no_prediction} verified detection
                {data.skipped_no_prediction === 1 ? "" : "s"} excluded because no
                original machine prediction is on record. Re-run analysis on
                affected deployments to populate them.
              </p>
            )}
            {data.skipped_unverified > 0 && (
              <p>
                {data.skipped_unverified} detection
                {data.skipped_unverified === 1 ? "" : "s"} in the filtered range
                are not yet verified; only verified detections count toward the matrix.
              </p>
            )}
          </div>
        )}

        <PlotExplainer
          plotKey="confusion-matrix"
          what={
            <p>
              Each bounding box contributes one cell. The row is the current label
              (after human verification or relabel), the column is what the
              classifier originally predicted. The diagonal is agreements, the
              off-diagonal cells are the species the classifier confuses for which.
              Cell colour scales per-row from light to dark teal so the dominant
              prediction for each true class stands out. Click a non-zero cell to
              open the underlying detections in the Verify page.
            </p>
          }
          how={
            <p>
              Only verified detections count. Predictions come from the
              original_label column captured at JSON load time and are never
              modified by rollup, smoothing, or user relabels. Classes roll up to
              the chosen taxonomic rank via the label_taxonomy table. The
              largest classes by support sit in the fixed head; everything outside
              the top-N falls into an "other" row and column so totals stay
              conserved.
            </p>
          }
          caveats={
            <ul className="list-disc space-y-1 pl-5">
              <li>
                <span className="font-medium text-foreground">Detection errors vs classification errors.</span>{" "}
                Cells on the animal, person, or vehicle row reflect detector decisions
                rather than classifier ones. The detector decides whether a box
                shows up at all; the classifier only speaks on the crops it gets.
              </li>
              <li>
                <span className="font-medium text-foreground">Missed animals are invisible.</span>{" "}
                If the detector never drew a box, there is no row here at all.
                This matrix counts misclassifications, not missed observations.
              </li>
              <li>
                <span className="font-medium text-foreground">Non-label classes are stripped.</span>{" "}
                Detections classified as blank, empty, vide, bait, etc. never
                reach the database and so never appear here.
              </li>
              <li>
                <span className="font-medium text-foreground">Detector-only projects.</span>{" "}
                Projects with no classifier see a three-class matrix of animal,
                person, vehicle. Attach a classification model to get
                species-level resolution.
              </li>
            </ul>
          }
          settings={[
            {
              label: "Detection threshold",
              detail:
                "Low-confidence detections get hidden from the Verify page but stay eligible here once verified.",
            },
            {
              label: "Taxonomic rollup",
              detail:
                "Turning it off keeps labels at the leaf (raw classifier top-1). Turning it on means the matrix may show family- or genus-level rolled-up labels.",
            },
            {
              label: "Excluded classes",
              detail:
                "Classes in the project exclusion list are stripped before detections land in the database.",
            },
          ]}
        />
      </main>
    </>
  );
}

export default ConfusionMatrixPage;
