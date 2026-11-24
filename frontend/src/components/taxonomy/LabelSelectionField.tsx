/**
 * Label Selection Field
 *
 * Compact inline control that surfaces the geofence country filter which used
 * to be hidden inside SpeciesSelectionModal. Country dropdown on the left
 * (geofence models only) and an "X of Y included · Edit species" summary on
 * the right. Picking a country applies immediately (recomputes excluded
 * classes from the geofence); "Edit species" opens the species tree for
 * manual tweaks.
 *
 * Replaces the duplicated "Select labels" button + modal wiring across the
 * create-project modal, project settings, and folder-run step 1.
 *
 * The dropdown also offers a custom SpeciesNet taxa list (.csv), the file
 * MegaDetector's restrict_to_taxa_list reads. It is a choice instead of a
 * country, as Dan Morris runs it: picking one clears the country and the
 * exclusions, and the field then shows the file and its label count.
 */

import { useState, useMemo, useCallback, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, ChevronsUpDown, Loader2, SlidersHorizontal } from "lucide-react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "../ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../ui/command";
import { cn } from "../../lib/utils";
import { modelsApi } from "../../api/models";
import type { TaxaMapping } from "../../api/types";
import { ApiError } from "../../lib/api-client";
import { ALL_LABELS_CODE, mappedLabels } from "../../lib/speciesSelection";
import { SpeciesSelectionModal } from "./SpeciesSelectionModal";

/** First entry in the country dropdown: run with every label. */
const ALL_LABELS_LABEL = "All labels";

/** Second entry: load a custom taxa list instead of a country. */
const TAXA_LIST_LABEL = "Custom taxa list (.csv)";

/** Button prompt while no choice has been made yet. */
const CHOOSE_PROMPT = "Select a country";

/**
 * Model-aware helper text for the label-selection control. Geofence models
 * (e.g. SpeciesNet) narrow by country; other models pick species directly.
 * Shared so the three call sites (create project, project settings, folder
 * run) stay in sync, and reusing the cached geofence query means calling this
 * next to the field adds no extra request.
 */
export function useLabelSelectionCaption(modelId: string, hasTaxaList: boolean): string {
  const { data: geofence } = useQuery({
    queryKey: ["model-geofence", modelId],
    queryFn: () => modelsApi.getModelGeofence(modelId),
    enabled: !!modelId,
    staleTime: Infinity,
  });
  if (hasTaxaList) return "Your taxa list sets which labels the AI can give.";
  return geofence?.has_geofence && geofence.countries
    ? "Filter species to the country where your cameras are."
    : "Pick which species the AI can identify, to cut false positives.";
}

interface LabelSelectionFieldProps {
  modelId: string;
  /** Currently excluded label names. */
  excludedClasses: string[];
  /** All label names in the model taxonomy (model all_classes). Used to count
   *  included labels and to ignore stale exclusions left over from a previously
   *  selected model. */
  allClasses: string[];
  /** Current country code from the parent form. `ALL_LABELS_CODE` means
   *  the user explicitly chose all labels; null means no choice yet. */
  countryCode?: string | null;
  /** Current state code from the parent form. */
  stateCode?: string | null;
  /** Called when the excluded label set changes (country preselect or refine). */
  onExclusionChange: (classes: string[]) => void;
  /** Called when the user picks a country/state (or `ALL_LABELS_CODE`). */
  onLocationChange: (country: string | null, state: string | null) => void;
  /** The loaded taxa list, or null when the project uses a country. */
  taxaMapping: TaxaMapping | null | undefined;
  /** Called when a taxa list is loaded, changed or removed. */
  onTaxaMappingChange: (mapping: TaxaMapping | null) => void;
  /** Validation error from the parent form, rendered under the control. */
  error?: string;
}

export function LabelSelectionField({
  modelId,
  excludedClasses,
  allClasses,
  countryCode,
  stateCode,
  onExclusionChange,
  onLocationChange,
  taxaMapping,
  onTaxaMappingChange,
  error,
}: LabelSelectionFieldProps) {
  const [modalOpen, setModalOpen] = useState(false);
  const [locationOpen, setLocationOpen] = useState(false);
  const [applying, setApplying] = useState(false);
  // A refused file and every bad row in it, shown until the next pick:
  // the user fixes those rows in the file, so they need to stay readable.
  const [refused, setRefused] = useState<{ fileName: string; problems: string[] } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Countries list for the dropdown (only present for geofence models).
  const { data: geofence } = useQuery({
    queryKey: ["model-geofence", modelId],
    queryFn: () => modelsApi.getModelGeofence(modelId),
    enabled: !!modelId,
    staleTime: Infinity,
  });

  const hasGeofence = geofence?.has_geofence && geofence.countries;

  // Build merged location options: countries + US states nested under USA.
  const locationOptions = useMemo(() => {
    if (!hasGeofence) return [];
    const options: {
      key: string;
      label: string;
      searchValue: string;
      country: string;
      state: string | null;
    }[] = [];
    const usaDisplayName = Object.entries(geofence.countries!).find(
      ([, code]) => code === "USA",
    )?.[0];

    for (const [name, code] of Object.entries(geofence.countries!)) {
      if (code === "USA" && geofence.us_states) {
        // "United States" as a general entry (all states).
        options.push({ key: "USA", label: name, searchValue: name, country: "USA", state: null });
        // Each state as "United States (California)" etc.
        for (const [stateName, stateCode] of Object.entries(geofence.us_states)) {
          const stateLabel = `${usaDisplayName ?? "United States"} (${stateName})`;
          options.push({
            key: `USA:${stateCode}`,
            label: stateLabel,
            searchValue: `${name} ${stateName}`,
            country: "USA",
            state: stateCode,
          });
        }
      } else {
        options.push({ key: code, label: name, searchValue: name, country: code, state: null });
      }
    }
    return options;
  }, [hasGeofence, geofence]);

  // Composite key + label for the current selection. Three states:
  // a real country/state, the explicit "All labels" choice, or no
  // choice yet (prompt).
  // A stored project keeps no country with a taxa list, and forms read a
  // stored null back as "All labels", so the list decides first.
  const allLabelsChosen = !taxaMapping && countryCode === ALL_LABELS_CODE;
  const selectedLocationKey =
    countryCode && !allLabelsChosen
      ? stateCode
        ? `${countryCode}:${stateCode}`
        : countryCode
      : null;
  const selectedLocationLabel = locationOptions.find(
    (o) => o.key === selectedLocationKey,
  )?.label;
  const displayLabel = taxaMapping
    ? taxaMapping.file_name
    : allLabelsChosen
      ? ALL_LABELS_LABEL
      : (selectedLocationLabel ?? CHOOSE_PROMPT);
  const isPrompt = !taxaMapping && !allLabelsChosen && !selectedLocationLabel;

  // Apply a country/state pick: recompute excluded classes from the geofence.
  // The country is written together with the list it produced, never
  // before: a failed request leaves both as they were, so the dropdown can
  // not show a country the list does not reflect.
  const applyLocation = useCallback(
    async (country: string | null, state: string | null) => {
      setLocationOpen(false);
      setRefused(null);
      if (!country || country === ALL_LABELS_CODE) {
        onTaxaMappingChange(null);
        onLocationChange(country, state);
        onExclusionChange([]);
        return;
      }
      setApplying(true);
      try {
        const res = await modelsApi.getModelGeofence(modelId, country, state ?? undefined);
        onTaxaMappingChange(null);
        onLocationChange(country, state);
        onExclusionChange(res.excluded_labels ?? []);
      } catch {
        toast.error("Could not load the species list for that country");
      } finally {
        setApplying(false);
      }
    },
    [modelId, onExclusionChange, onLocationChange, onTaxaMappingChange],
  );

  // Load a taxa list. Like a country pick, nothing changes until the
  // backend has read the whole file: a refused file leaves the current
  // choice as it was and lists its bad rows.
  const openTaxaListPicker = () => {
    setLocationOpen(false);
    fileInputRef.current?.click();
  };
  const loadTaxaList = async (file: File) => {
    setApplying(true);
    setRefused(null);
    try {
      const mapping = await modelsApi.readTaxaList(modelId, file.name, await file.text());
      onTaxaMappingChange(mapping);
      onLocationChange(null, null);
      onExclusionChange([]);
    } catch (e) {
      const detail = e instanceof ApiError ? e.detail : null;
      const problems =
        detail && typeof detail === "object" && "problems" in detail
          ? (detail as { problems: string[] }).problems
          : [];
      setRefused({
        fileName: file.name,
        problems: problems.length > 0 ? problems : ["The file could not be read."],
      });
    } finally {
      setApplying(false);
    }
  };

  // Count only exclusions that exist in the current model. Switching models can
  // leave stale exclusions from a previous (larger) taxonomy; ignoring them
  // keeps the count correct and non-negative instead of e.g. "-2046 of 30".
  const allClassesSet = useMemo(() => new Set(allClasses), [allClasses]);
  const totalSpeciesCount = allClasses.length;
  const excludedInModel = useMemo(
    () => excludedClasses.filter((c) => allClassesSet.has(c)).length,
    [excludedClasses, allClassesSet],
  );
  const includedCount = totalSpeciesCount - excludedInModel;

  // Shared status text so the geofence and non-geofence branches read the
  // same: "All species included" when nothing is excluded, otherwise the
  // included / total count.
  const statusText =
    excludedInModel === 0
      ? "All species included"
      : `${includedCount} of ${totalSpeciesCount} included`;

  // Geofence caption: the shared status plus an "Edit species" link that opens
  // the species tree (the country button above does not open it). With a
  // taxa list there is no tree to edit: the file is the selection, so the
  // caption counts its labels and offers to change or remove it.
  const linkClass = "text-primary font-medium hover:underline";
  const labelCount = taxaMapping ? mappedLabels(taxaMapping).length : 0;
  const summary = taxaMapping ? (
    <p className="pl-3 text-xs text-muted-foreground">
      {labelCount} {labelCount === 1 ? "label" : "labels"}{" "}
      <button type="button" onClick={openTaxaListPicker} className={linkClass}>
        · Change
      </button>{" "}
      <button
        type="button"
        onClick={() => onTaxaMappingChange(null)}
        className={linkClass}
      >
        · Remove
      </button>
    </p>
  ) : (
    <p className="pl-3 text-xs text-muted-foreground">
      {statusText}{" "}
      <button
        type="button"
        onClick={() => setModalOpen(true)}
        className={linkClass}
      >
        · Edit species
      </button>
    </p>
  );

  return (
    <>
      <div className="space-y-1">
        {hasGeofence ? (
          <>
          <Popover open={locationOpen} onOpenChange={setLocationOpen}>
            <PopoverTrigger asChild>
              <Button
                type="button"
                variant="outline"
                role="combobox"
                size="sm"
                className="h-9 w-full justify-between"
              >
                {applying && (
                  <Loader2 className="h-3.5 w-3.5 shrink-0 mr-1.5 animate-spin" />
                )}
                <span
                  className={cn(
                    "truncate flex-1 text-left",
                    isPrompt && "text-muted-foreground",
                  )}
                  title={displayLabel}
                >
                  {displayLabel}
                </span>
                <ChevronsUpDown className="ml-1.5 h-3 w-3 shrink-0 opacity-50" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[400px] p-0">
              <Command>
                <CommandInput placeholder="Search countries or states..." />
                <CommandList className="max-h-[300px] overflow-y-auto">
                  <CommandEmpty>No location found.</CommandEmpty>
                  <CommandGroup>
                    <CommandItem
                      key="__all__"
                      value="Do not filter show all labels"
                      onSelect={() => applyLocation(ALL_LABELS_CODE, null)}
                    >
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          allLabelsChosen ? "opacity-100" : "opacity-0",
                        )}
                      />
                      {ALL_LABELS_LABEL}
                    </CommandItem>
                    <CommandItem
                      key="__taxa_list__"
                      value="Custom taxa list csv file"
                      onSelect={openTaxaListPicker}
                    >
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          taxaMapping ? "opacity-100" : "opacity-0",
                        )}
                      />
                      {TAXA_LIST_LABEL}
                    </CommandItem>
                    {locationOptions.map((option) => (
                      <CommandItem
                        key={option.key}
                        value={option.searchValue}
                        onSelect={() => applyLocation(option.country, option.state)}
                      >
                        <Check
                          className={cn(
                            "mr-2 h-4 w-4",
                            selectedLocationKey === option.key ? "opacity-100" : "opacity-0",
                          )}
                        />
                        {option.label}
                      </CommandItem>
                    ))}
                  </CommandGroup>
                </CommandList>
              </Command>
            </PopoverContent>
          </Popover>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              // Cleared so picking the same file again still fires.
              e.target.value = "";
              if (file) void loadTaxaList(file);
            }}
          />
          {/* No status line while nothing is chosen: "All species
              included · Edit species" would contradict the "Select a
              country" prompt above, and manual species tweaks made
              before a country pick would be clobbered by the geofence
              recompute anyway. */}
          {!isPrompt && summary}
          {refused && (
            <Callout variant="error" size="compact" title={`Could not use ${refused.fileName}`}>
              <ul className="max-h-40 overflow-y-auto">
                {refused.problems.map((p) => (
                  <li key={p}>{p}</li>
                ))}
              </ul>
            </Callout>
          )}
          {error && (
            <p className="pl-3 text-xs font-medium text-destructive">
              {error}
            </p>
          )}
          </>
        ) : (
          // No geofence: no country dropdown, so mirror the geofence layout
          // with a "Select species" button on top (names the action) and the
          // shared status caption below. The button opens the species tree.
          <>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 w-full justify-between"
              onClick={() => setModalOpen(true)}
            >
              <span className="truncate flex-1 text-left">Select species</span>
              <SlidersHorizontal className="ml-1.5 h-3.5 w-3.5 shrink-0 opacity-50" />
            </Button>
            <p className="pl-3 text-xs text-muted-foreground">{statusText}</p>
          </>
        )}
      </div>

      <SpeciesSelectionModal
        modelId={modelId}
        excludedClasses={excludedClasses}
        allClasses={allClasses}
        onExclusionChange={onExclusionChange}
        open={modalOpen}
        onOpenChange={setModalOpen}
      />
    </>
  );
}
