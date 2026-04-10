/**
 * Species Selection Modal
 *
 * Modal dialog containing the SpeciesSelector tree for excluding species.
 * Keeps the settings page clean by hiding the complex tree UI until needed.
 * Uses a working-copy pattern: changes are only applied on "Apply".
 *
 * When a model has geofence support, shows a compact country/state filter
 * above the species tree so users can narrow labels by geography.
 */

import { useState, useCallback, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, ChevronsUpDown, Flag } from "lucide-react";
import { SpeciesSelector } from "./SpeciesSelector";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
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
import type { GeofenceResponse } from "../../api/types";

interface SpeciesSelectionModalProps {
  modelId: string;
  excludedClasses: string[];
  onExclusionChange: (classes: string[]) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  totalSpeciesCount: number;
  /** Current country code from project settings. */
  countryCode?: string | null;
  /** Current state code from project settings. */
  stateCode?: string | null;
  /** Called when the user changes country/state inside the modal. */
  onLocationChange?: (country: string | null, state: string | null) => void;
}

export function SpeciesSelectionModal({
  modelId,
  excludedClasses,
  onExclusionChange,
  open,
  onOpenChange,
  totalSpeciesCount,
  countryCode,
  stateCode,
  onLocationChange,
}: SpeciesSelectionModalProps) {
  const [workingExcluded, setWorkingExcluded] = useState<string[]>([]);
  const [workingCountry, setWorkingCountry] = useState<string | null>(null);
  const [workingState, setWorkingState] = useState<string | null>(null);
  const [locationOpen, setLocationOpen] = useState(false);
  // Track whether the user actively changed the country (vs initial load)
  const [countryUserChanged, setCountryUserChanged] = useState(false);

  // Fetch geofence data (countries list)
  const { data: geofence } = useQuery({
    queryKey: ["model-geofence", modelId],
    queryFn: () => modelsApi.getModelGeofence(modelId),
    enabled: open && !!modelId,
    staleTime: Infinity,
  });

  // Fetch filtered labels when country changes
  const { data: geofenceFiltered } = useQuery<GeofenceResponse>({
    queryKey: ["model-geofence-filtered", modelId, workingCountry, workingState],
    queryFn: () =>
      modelsApi.getModelGeofence(
        modelId,
        workingCountry ?? undefined,
        workingState ?? undefined,
      ),
    enabled: open && !!workingCountry && workingCountry !== "NONE",
    staleTime: Infinity,
  });

  // Re-initialize working copies each time the modal opens
  useEffect(() => {
    if (open) {
      setWorkingExcluded(excludedClasses);
      setWorkingCountry(countryCode ?? null);
      setWorkingState(stateCode ?? null);
      setCountryUserChanged(false);
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  // When country changes, auto-compute excluded classes from geofence
  const handleCountryChange = useCallback(
    (country: string) => {
      setWorkingCountry(country);
      setCountryUserChanged(true);
      if (country !== "USA") {
        setWorkingState(null);
      }
    },
    [],
  );

  // When geofence-filtered data arrives after a user-initiated country
  // change, update the working excluded list. On modal open (initial
  // load), the saved excluded_classes are used instead.
  useEffect(() => {
    if (countryUserChanged && geofenceFiltered?.excluded_labels) {
      setWorkingExcluded(geofenceFiltered.excluded_labels);
    }
  }, [geofenceFiltered, countryUserChanged]);

  const handleStateChange = useCallback((state: string) => {
    setWorkingState(state);
  }, []);

  const handleApply = useCallback(() => {
    onExclusionChange(workingExcluded);
    if (onLocationChange) {
      onLocationChange(workingCountry, workingState);
    }
    onOpenChange(false);
  }, [workingExcluded, workingCountry, workingState, onExclusionChange, onLocationChange, onOpenChange]);

  const handleCancel = useCallback(() => {
    onOpenChange(false);
  }, [onOpenChange]);

  const hasGeofence = geofence?.has_geofence && geofence.countries;

  // Build merged location options: countries + US states under USA
  const locationOptions = useMemo(() => {
    if (!hasGeofence) return [];
    const options: { key: string; label: string; searchValue: string; country: string; state: string | null }[] = [];
    const usaDisplayName = Object.entries(geofence.countries!).find(([, code]) => code === "USA")?.[0];

    for (const [name, code] of Object.entries(geofence.countries!)) {
      if (code === "USA" && geofence.us_states) {
        // Add "United States" as a general entry (all states)
        options.push({
          key: "USA",
          label: name,
          searchValue: name,
          country: "USA",
          state: null,
        });
        // Add each state as "United States (California)" etc.
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

  // Derive the display label and composite key for the current selection
  const selectedLocationKey = workingCountry
    ? workingState ? `${workingCountry}:${workingState}` : workingCountry
    : null;
  const selectedLocationLabel = locationOptions.find((o) => o.key === selectedLocationKey)?.label;

  // Country/state filter rendered inside the search bar row
  const countryFilter = hasGeofence ? (
    <Popover open={locationOpen} onOpenChange={setLocationOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          size="sm"
          className={cn(
            "h-9 w-full justify-between",
            !selectedLocationKey &&
              "border-primary bg-primary/10 text-primary font-medium hover:bg-primary/15 hover:text-primary",
          )}
        >
          <Flag
            className={cn(
              "h-3.5 w-3.5 shrink-0 mr-1.5",
              selectedLocationKey ? "opacity-60" : "opacity-100",
            )}
          />
          <span className="truncate flex-1 text-left">
            {selectedLocationLabel ?? "Preselect species by country (recommended)"}
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
                onSelect={() => {
                  setWorkingCountry(null);
                  setWorkingState(null);
                  setWorkingExcluded([]);
                  setLocationOpen(false);
                }}
              >
                <Check
                  className={cn(
                    "mr-2 h-4 w-4",
                    !selectedLocationKey ? "opacity-100" : "opacity-0",
                  )}
                />
                No geographic preselection (all labels included)
              </CommandItem>
              {locationOptions.map((option) => (
                <CommandItem
                  key={option.key}
                  value={option.searchValue}
                  onSelect={() => {
                    handleCountryChange(option.country);
                    if (option.state) {
                      handleStateChange(option.state);
                    } else {
                      setWorkingState(null);
                    }
                    setLocationOpen(false);
                  }}
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
  ) : undefined;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Configure label selection</DialogTitle>
          <DialogDescription>
            Select which labels to include in classifications
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 min-h-0">
          <SpeciesSelector
            modelId={modelId}
            excludedClasses={workingExcluded}
            onExclusionChange={setWorkingExcluded}
            fillHeight
            searchRowExtra={countryFilter}
          />
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button onClick={handleApply}>Apply</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
