/**
 * Classification performance endpoint (confusion matrix + report).
 */

import { api } from "../lib/api-client";
import type { TaxonomicRank } from "../lib/taxonomic-rank";

export interface ClassMetrics {
  class_name: string;
  common_name: string;
  scientific_name: string;
  support: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
}

export interface PerformanceResponse {
  taxonomic_rank: TaxonomicRank;
  classes: string[];
  class_scientific_names: string[];
  class_common_names: string[];
  class_taxonomy_ids: (string | null)[];
  matrix: number[][];
  row_totals: number[];
  col_totals: number[];
  grand_total: number;
  per_class: ClassMetrics[];
  macro_precision: number | null;
  macro_recall: number | null;
  macro_f1: number | null;
  weighted_precision: number | null;
  weighted_recall: number | null;
  weighted_f1: number | null;
  skipped_no_prediction: number;
  skipped_unverified: number;
  has_classifier: boolean;
  top_n_applied: number | null;
  other_bucket_present: boolean;
}

export interface PerformanceFilters {
  siteIds?: string[];
  dateFrom?: string;
  dateTo?: string;
  taxonomicRank?: TaxonomicRank;
  /** Integer or the literal 'all'. Defaults to 20 server-side. */
  topN?: string;
}

export const performanceApi = {
  get: (projectId: string, filters: PerformanceFilters = {}) => {
    const params = new URLSearchParams();
    params.set("project_id", projectId);
    if (filters.siteIds?.length) params.set("site_ids", filters.siteIds.join(","));
    if (filters.dateFrom) params.set("date_from", filters.dateFrom);
    if (filters.dateTo) params.set("date_to", filters.dateTo);
    if (filters.taxonomicRank) params.set("taxonomic_rank", filters.taxonomicRank);
    if (filters.topN) params.set("top_n", filters.topN);
    return api.get<PerformanceResponse>(
      `/api/statistics/performance?${params.toString()}`
    );
  },
};
