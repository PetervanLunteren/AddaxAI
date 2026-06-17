/**
 * Label Selection Field
 *
 * Compact inline control that surfaces the geofence country filter which used
 * to be hidden inside SpeciesSelectionModal. Country dropdown on the left
 * (geofence models only) and an "X of Y included · Refine" summary on the
 * right. Picking a country applies immediately (recomputes excluded classes
 * from the geofence); "Refine" opens the species tree for manual tweaks.
 *
 * Replaces the duplicated "Select labels" button + modal wiring across the
 * create-project modal, project settings, and folder-run step 1.
 */

import { useState, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, ChevronsUpDown, Loader2 } from "lucide-react";
import { Button } from "../ui/button";
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
import { SpeciesSelectionModal } from "./SpeciesSelectionModal";

/** Shown in the dropdown when no country is selected (all labels kept). */
const NO_FILTER_LABEL = "All labels included";

interface LabelSelectionFieldProps {
  modelId: string;
  /** Currently excluded label names. */
  excludedClasses: string[];
  /** Total number of labels in the model taxonomy (all_classes length). */
  totalSpeciesCount: number;
  /** Current country code from the parent form. */
  countryCode?: string | null;
  /** Current state code from the parent form. */
  stateCode?: string | null;
  /** Called when the excluded label set changes (country preselect or refine). */
  onExclusionChange: (classes: string[]) => void;
  /** Called when the user picks a country/state. */
  onLocationChange: (country: string | null, state: string | null) => void;
}

export function LabelSelectionField({
  modelId,
  excludedClasses,
  totalSpeciesCount,
  countryCode,
  stateCode,
  onExclusionChange,
  onLocationChange,
}: LabelSelectionFieldProps) {
  const [modalOpen, setModalOpen] = useState(false);
  const [locationOpen, setLocationOpen] = useState(false);
  const [applying, setApplying] = useState(false);

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

  // Composite key + label for the current selection.
  const selectedLocationKey = countryCode
    ? stateCode
      ? `${countryCode}:${stateCode}`
      : countryCode
    : null;
  const selectedLocationLabel = locationOptions.find(
    (o) => o.key === selectedLocationKey,
  )?.label;
  const displayLabel = selectedLocationLabel ?? NO_FILTER_LABEL;

  // Apply a country/state pick: recompute excluded classes from the geofence.
  const applyLocation = useCallback(
    async (country: string | null, state: string | null) => {
      setLocationOpen(false);
      onLocationChange(country, state);
      if (!country) {
        onExclusionChange([]);
        return;
      }
      setApplying(true);
      try {
        const res = await modelsApi.getModelGeofence(modelId, country, state ?? undefined);
        onExclusionChange(res.excluded_labels ?? []);
      } finally {
        setApplying(false);
      }
    },
    [modelId, onExclusionChange, onLocationChange],
  );

  const includedCount = totalSpeciesCount - excludedClasses.length;

  const summary = (
    <p className="pl-3 text-xs text-muted-foreground">
      {includedCount} of {totalSpeciesCount} included{" "}
      <button
        type="button"
        onClick={() => setModalOpen(true)}
        className="text-primary font-medium hover:underline"
      >
        · Refine
      </button>
    </p>
  );

  return (
    <>
      <div className="space-y-1">
        {hasGeofence && (
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
                <span className="truncate flex-1 text-left" title={displayLabel}>
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
                      key="__none__"
                      value="Do not filter show all labels"
                      onSelect={() => applyLocation(null, null)}
                    >
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          !selectedLocationKey ? "opacity-100" : "opacity-0",
                        )}
                      />
                      {NO_FILTER_LABEL}
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
        )}
        {summary}
      </div>

      <SpeciesSelectionModal
        modelId={modelId}
        excludedClasses={excludedClasses}
        onExclusionChange={onExclusionChange}
        open={modalOpen}
        onOpenChange={setModalOpen}
      />
    </>
  );
}
