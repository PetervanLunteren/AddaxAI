/**
 * Hook to build label options for the unified label picker.
 *
 * Fetches species from the classification model's taxonomy (or from
 * project species stats for SpeciesNet), merges in any project-specific
 * custom species, and combines them with the always-available "person"
 * and "vehicle" options.
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

  // Custom species added by the user for this project
  const {
    data: customSpecies,
    isLoading: customLoading,
  } = useQuery({
    queryKey: ["custom-species", projectId],
    queryFn: () => projectsApi.getCustomSpecies(projectId),
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

    // Append custom species, deduplicating against already-present names
    if (customSpecies) {
      const existingNames = new Set(result.map((o) => o.value.toLowerCase()));
      for (const cs of customSpecies) {
        if (!existingNames.has(cs.name.toLowerCase())) {
          result.push({ value: cs.name, category: "animal", species: cs.name });
          existingNames.add(cs.name.toLowerCase());
        }
      }
    }

    return result;
  }, [hasTaxonomyModel, taxonomy, isSpeciesNet, speciesStats, customSpecies]);

  return { options, isLoading };
}
