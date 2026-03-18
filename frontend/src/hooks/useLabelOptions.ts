/**
 * Hook to build label options for the unified label picker.
 *
 * Fetches labels from the classification model's taxonomy (or from
 * project label stats for SpeciesNet), merges in any project-specific
 * custom labels, and combines them with the always-available "person"
 * and "vehicle" options. Each option is annotated with a taxonomy
 * string (e.g. "mammalia › carnivora › felidae") when available.
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
  /** Taxonomy string for display, e.g. "mammalia › carnivora › felidae" */
  taxonomyCaption?: string | null;
}

const GENERAL_OPTIONS: LabelOption[] = [
  { value: "person", category: "person", label: null },
  { value: "vehicle", category: "vehicle", label: null },
];

/** Build a display string from taxonomy fields, joining non-empty ranks with " › ". */
function buildTaxonomyCaption(
  entry: {
    taxon_class: string | null;
    taxon_order: string | null;
    taxon_family: string | null;
    taxon_genus: string | null;
    taxon_species: string | null;
  } | undefined,
): string | null {
  if (!entry) return null;
  const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
  const parts = [
    entry.taxon_class,
    entry.taxon_order,
    entry.taxon_family,
    entry.taxon_genus,
    entry.taxon_species,
  ].filter(Boolean).map((s) => capitalize(s as string));
  return parts.length > 0 ? parts.join(" › ") : null;
}

export function useLabelOptions(
  classificationModelId: string | null,
  projectId: string
) {
  const isSpeciesNet =
    classificationModelId?.toUpperCase().includes("SPECIESNET") ?? false;
  const hasClassificationModel = !!classificationModelId && classificationModelId !== "none";
  const hasTaxonomyModel = hasClassificationModel && !isSpeciesNet;

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
    enabled: hasClassificationModel && isSpeciesNet,
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

  // Taxonomy fields for all labels (model + custom)
  const {
    data: taxonomyMap,
  } = useQuery({
    queryKey: ["label-taxonomy-map", projectId],
    queryFn: () => projectsApi.getLabelTaxonomyMap(projectId),
    enabled: !!projectId,
  });

  const isLoading =
    (hasTaxonomyModel && taxonomyLoading) ||
    (isSpeciesNet && statsLoading) ||
    (!!projectId && customLoading);

  const options = useMemo(() => {
    const result: LabelOption[] = GENERAL_OPTIONS.map((o) => ({
      ...o,
      taxonomyCaption: buildTaxonomyCaption(taxonomyMap?.[o.value]),
    }));

    if (!hasClassificationModel) {
      // Detection-only projects: add "animal" alongside "person" and "vehicle"
      result.push({ value: "animal", category: "animal", label: null });
    } else if (hasTaxonomyModel && taxonomy?.all_classes) {
      for (const cls of taxonomy.all_classes) {
        result.push({
          value: cls,
          category: "animal",
          label: cls,
          taxonomyCaption: buildTaxonomyCaption(taxonomyMap?.[cls]),
        });
      }
    } else if (isSpeciesNet && labelStats) {
      for (const stat of labelStats) {
        if (stat.label) {
          result.push({
            value: stat.label,
            category: "animal",
            label: stat.label,
            taxonomyCaption: buildTaxonomyCaption(taxonomyMap?.[stat.label]),
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
            taxonomyCaption: buildTaxonomyCaption(taxonomyMap?.[cl.name]),
          });
          existingNames.add(cl.name.toLowerCase());
        }
      }
    }

    return result;
  }, [hasClassificationModel, hasTaxonomyModel, taxonomy, isSpeciesNet, labelStats, customLabels, taxonomyMap]);

  return { options, isLoading };
}
