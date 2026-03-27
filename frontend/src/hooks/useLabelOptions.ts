/**
 * Hook to build label options for the unified label picker.
 *
 * Fetches labels from the classification model's taxonomy, merges in any
 * project-specific custom labels, and combines them with the always-available
 * "person" and "vehicle" options. Each option is annotated with a taxonomy
 * string (e.g. "mammalia > carnivora > felidae") when available.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { modelsApi } from "../api/models";
import { projectsApi } from "../api/projects";

export interface LabelOption {
  value: string;
  /** Latin display name (e.g., "G. camelopardalis"). Falls back to capitalized value. */
  displayName: string;
  category: "animal" | "person" | "vehicle";
  label: string | null;
  isCustom?: boolean;
  customId?: string;
  /** Taxonomy string for display, e.g. "mammalia > carnivora > felidae" */
  taxonomyCaption?: string | null;
}

const GENERAL_OPTIONS: LabelOption[] = [
  { value: "person", displayName: "Person", category: "person", label: null },
  { value: "vehicle", displayName: "Vehicle", category: "vehicle", label: null },
];

/** Read display_name from the backend taxonomy map, with capitalize fallback. */
function getDisplayName(
  rawLabel: string,
  entry: { display_name?: string | null } | undefined,
): string {
  return entry?.display_name || rawLabel.charAt(0).toUpperCase() + rawLabel.slice(1);
}

/** Build a display string from taxonomy fields, joining non-empty ranks with " > ". */
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
  return parts.length > 0 ? parts.join(" \u203A ") : null;
}

export function useLabelOptions(
  classificationModelId: string | null,
  projectId: string
) {
  const hasClassificationModel = !!classificationModelId && classificationModelId !== "none";

  // Taxonomy-based models (all classification models now use taxonomy)
  const {
    data: taxonomy,
    isLoading: taxonomyLoading,
  } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: hasClassificationModel,
    staleTime: Infinity,
  });

  // Custom labels added by the user for this project
  const {
    data: customLabels,
    isLoading: customLoading,
  } = useQuery({
    queryKey: ["custom-labels", projectId],
    queryFn: () => projectsApi.getCustomLabels(projectId),
    enabled: !!projectId,
    staleTime: Infinity,
  });

  // Taxonomy fields for all labels (model + custom)
  const {
    data: taxonomyMap,
  } = useQuery({
    queryKey: ["label-taxonomy-map", projectId],
    queryFn: () => projectsApi.getLabelTaxonomyMap(projectId),
    enabled: !!projectId,
    staleTime: Infinity,
  });

  const isLoading =
    (hasClassificationModel && taxonomyLoading) ||
    (!!projectId && customLoading);

  const options = useMemo(() => {
    const result: LabelOption[] = GENERAL_OPTIONS.map((o) => ({
      ...o,
      taxonomyCaption: buildTaxonomyCaption(taxonomyMap?.[o.value]),
    }));

    if (!hasClassificationModel) {
      // Detection-only projects: add "animal" alongside "person" and "vehicle"
      result.push({
        value: "animal", displayName: "Animal",
        category: "animal", label: null,
      });
    } else if (taxonomy?.all_classes) {
      for (const cls of taxonomy.all_classes) {
        const entry = taxonomyMap?.[cls];
        result.push({
          value: cls,
          displayName: getDisplayName(cls, entry),
          category: "animal",
          label: cls,
          taxonomyCaption: buildTaxonomyCaption(entry),
        });
      }
    }

    // Merge custom labels: if a custom label name already exists (e.g. from
    // model taxonomy), mark that entry as custom so it appears in the
    // "Custom labels" section with an edit button. Otherwise, append it as
    // a new entry.
    if (customLabels) {
      const existingByName = new Map(result.map((o, i) => [o.value.toLowerCase(), i]));
      for (const cl of customLabels) {
        const idx = existingByName.get(cl.name.toLowerCase());
        if (idx !== undefined) {
          const entry = taxonomyMap?.[cl.name];
          result[idx] = {
            ...result[idx],
            isCustom: true,
            customId: cl.id,
            displayName: getDisplayName(cl.name, entry),
            taxonomyCaption: buildTaxonomyCaption(entry) ?? result[idx].taxonomyCaption,
          };
        } else {
          const entry = taxonomyMap?.[cl.name];
          result.push({
            value: cl.name,
            displayName: getDisplayName(cl.name, entry),
            category: "animal",
            label: cl.name,
            isCustom: true,
            customId: cl.id,
            taxonomyCaption: buildTaxonomyCaption(entry),
          });
          existingByName.set(cl.name.toLowerCase(), result.length - 1);
        }
      }
    }

    return result;
  }, [hasClassificationModel, taxonomy, customLabels, taxonomyMap]);

  return { options, isLoading };
}
