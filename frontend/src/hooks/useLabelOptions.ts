/**
 * Hook to build label options for the unified label picker.
 *
 * Fetches labels from the classification model's taxonomy (or from
 * project label stats for SpeciesNet), merges in any project-specific
 * custom labels, and combines them with the always-available "person"
 * and "vehicle" options.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { modelsApi } from "../api/models";
import { projectsApi } from "../api/projects";

export interface LabelOption {
  value: string;
  category: "animal" | "person" | "vehicle";
  label: string | null;
  isCustom?: boolean;
  customId?: string;
}

const GENERAL_OPTIONS: LabelOption[] = [
  { value: "person", category: "person", label: null },
  { value: "vehicle", category: "vehicle", label: null },
];

export function useLabelOptions(
  classificationModelId: string | null,
  projectId: string
) {
  const isSpeciesNet =
    classificationModelId?.toUpperCase().includes("SPECIESNET") ?? false;
  const hasTaxonomyModel = !!classificationModelId && !isSpeciesNet;

  // Taxonomy-based models (EUR-DF, NAM-ADS, etc.)
  const {
    data: taxonomy,
    isLoading: taxonomyLoading,
  } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: hasTaxonomyModel,
  });

  // SpeciesNet fallback: distinct labels already in the project
  const {
    data: labelStats,
    isLoading: statsLoading,
  } = useQuery({
    queryKey: ["project-label-stats", projectId],
    queryFn: () => projectsApi.getLabelStats(projectId),
    enabled: isSpeciesNet,
  });

  // Custom labels added by the user for this project
  const {
    data: customLabels,
    isLoading: customLoading,
  } = useQuery({
    queryKey: ["custom-labels", projectId],
    queryFn: () => projectsApi.getCustomLabels(projectId),
    enabled: !!projectId,
  });

  const isLoading =
    (hasTaxonomyModel && taxonomyLoading) ||
    (isSpeciesNet && statsLoading) ||
    (!!projectId && customLoading);

  const options = useMemo(() => {
    const result: LabelOption[] = [...GENERAL_OPTIONS];

    if (hasTaxonomyModel && taxonomy?.all_classes) {
      for (const cls of taxonomy.all_classes) {
        result.push({ value: cls, category: "animal", label: cls });
      }
    } else if (isSpeciesNet && labelStats) {
      for (const stat of labelStats) {
        if (stat.label) {
          result.push({
            value: stat.label,
            category: "animal",
            label: stat.label,
          });
        }
      }
    }

    // Append custom labels, deduplicating against already-present names
    if (customLabels) {
      const existingNames = new Set(result.map((o) => o.value.toLowerCase()));
      for (const cl of customLabels) {
        if (!existingNames.has(cl.name.toLowerCase())) {
          result.push({
            value: cl.name,
            category: "animal",
            label: cl.name,
            isCustom: true,
            customId: cl.id,
          });
          existingNames.add(cl.name.toLowerCase());
        }
      }
    }

    return result;
  }, [hasTaxonomyModel, taxonomy, isSpeciesNet, labelStats, customLabels]);

  return { options, isLoading };
}
