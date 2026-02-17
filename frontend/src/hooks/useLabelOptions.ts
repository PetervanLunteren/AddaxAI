/**
 * Hook to build label options for the unified label picker.
 *
 * Fetches species from the classification model's taxonomy (or from
 * project species stats for SpeciesNet), and combines them with
 * the always-available "person" and "vehicle" options.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { modelsApi } from "../api/models";
import { projectsApi } from "../api/projects";

export interface LabelOption {
  value: string;
  category: "animal" | "person" | "vehicle";
  species: string | null;
}

const GENERAL_OPTIONS: LabelOption[] = [
  { value: "person", category: "person", species: null },
  { value: "vehicle", category: "vehicle", species: null },
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

  // SpeciesNet fallback: distinct species already in the project
  const {
    data: speciesStats,
    isLoading: statsLoading,
  } = useQuery({
    queryKey: ["project-species-stats", projectId],
    queryFn: () => projectsApi.getSpeciesStats(projectId),
    enabled: isSpeciesNet,
  });

  const isLoading =
    (hasTaxonomyModel && taxonomyLoading) || (isSpeciesNet && statsLoading);

  const options = useMemo(() => {
    const result: LabelOption[] = [...GENERAL_OPTIONS];

    if (hasTaxonomyModel && taxonomy?.all_classes) {
      for (const cls of taxonomy.all_classes) {
        result.push({ value: cls, category: "animal", species: cls });
      }
    } else if (isSpeciesNet && speciesStats) {
      for (const stat of speciesStats) {
        if (stat.species) {
          result.push({
            value: stat.species,
            category: "animal",
            species: stat.species,
          });
        }
      }
    }

    return result;
  }, [hasTaxonomyModel, taxonomy, isSpeciesNet, speciesStats]);

  return { options, isLoading };
}
