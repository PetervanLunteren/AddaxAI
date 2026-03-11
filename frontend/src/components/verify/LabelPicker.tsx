/**
 * Unified label picker — command palette for detection labels.
 *
 * Opens a centered dialog with searchable groups: pinned shortcuts,
 * general labels (person/vehicle), and species from the classification
 * model. An "Add new label" action at the bottom opens the TaxonomySheet
 * slideout for creating a new custom species with optional GBIF lookup.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronsUpDown, Pencil, Plus } from "lucide-react";
import { cn } from "../../lib/utils";
import { getCategoryColor } from "../../lib/detection-utils";
import { projectsApi } from "../../api/projects";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import {
  Command,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../ui/command";
import { TaxonomySheet } from "./TaxonomySheet";
import type { LabelOption } from "../../hooks/useLabelOptions";
import type { CustomSpeciesResponse } from "../../api/types";

interface PinnedOption {
  key: number;
  option: LabelOption;
}

interface LabelPickerProps {
  value: string | null;
  onSelect: (option: LabelOption) => void;
  options: LabelOption[];
  isLoading?: boolean;
  forceOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  pinnedOptions?: PinnedOption[];
  hideDot?: boolean;
  hideLabel?: boolean;
  projectId?: string;
}

export function LabelPicker({
  value,
  onSelect,
  options,
  isLoading,
  forceOpen,
  onOpenChange,
  pinnedOptions,
  hideDot,
  hideLabel,
  projectId,
}: LabelPickerProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [taxonomySpecies, setTaxonomySpecies] = useState<CustomSpeciesResponse | null>(null);
  const [taxonomySheetOpen, setTaxonomySheetOpen] = useState(false);
  const [pendingOption, setPendingOption] = useState<LabelOption | null>(null);
  const [createName, setCreateName] = useState("");
  const queryClient = useQueryClient();

  // Fetch custom species for edit lookups
  const { data: customSpeciesList } = useQuery({
    queryKey: ["custom-species", projectId],
    queryFn: () => projectsApi.getCustomSpecies(projectId!),
    enabled: !!projectId,
  });

  const customSpeciesMap = useMemo(() => {
    const map = new Map<string, CustomSpeciesResponse>();
    if (customSpeciesList) {
      for (const cs of customSpeciesList) map.set(cs.id, cs);
    }
    return map;
  }, [customSpeciesList]);

  // When forceOpen changes to true, open the dialog
  useEffect(() => {
    if (forceOpen) setOpen(true);
  }, [forceOpen]);

  const handleOpenChange = (next: boolean) => {
    setOpen(next);
    if (!next) setSearch("");
    onOpenChange?.(next);
  };

  const handleSelect = useCallback(
    (option: LabelOption) => {
      onSelect(option);
      handleOpenChange(false);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [onSelect]
  );

  // Manual filtering
  const searchLower = search.toLowerCase().trim();

  const filteredPinned = pinnedOptions?.filter(
    ({ option }) => !searchLower || option.value.toLowerCase().includes(searchLower)
  );

  const generalOptions = options.filter((o) => o.species === null);
  const modelSpecies = options.filter((o) => o.species !== null && !o.isCustom);
  const customSpeciesOpts = options.filter((o) => o.species !== null && o.isCustom);

  const filteredGeneral = generalOptions.filter(
    (o) => !searchLower || o.value.toLowerCase().includes(searchLower)
  );
  const filteredModelSpecies = modelSpecies.filter(
    (o) => !searchLower || o.value.toLowerCase().includes(searchLower)
  );
  const filteredCustomSpecies = customSpeciesOpts.filter(
    (o) => !searchLower || o.value.toLowerCase().includes(searchLower)
  );

  const hasResults =
    (filteredPinned && filteredPinned.length > 0) ||
    filteredGeneral.length > 0 ||
    filteredModelSpecies.length > 0 ||
    filteredCustomSpecies.length > 0;
  const showAddNew = !!projectId && (!searchLower || !hasResults);

  const handleAddNew = useCallback(() => {
    if (!projectId) return;
    setCreateName(search.trim());
    setOpen(false);
    setSearch("");
    setTaxonomySpecies(null);
    setTaxonomySheetOpen(true);
  }, [projectId, search]);

  const handleTaxonomySheetCreated = useCallback(
    (created: CustomSpeciesResponse) => {
      queryClient.invalidateQueries({ queryKey: ["custom-species", projectId] });
      queryClient.invalidateQueries({ queryKey: ["species-tree"] });
      const option: LabelOption = {
        value: created.name,
        category: "animal",
        species: created.name,
      };
      // Defer onSelect until sheet close to avoid unmount issues
      setPendingOption(option);
    },
    [projectId, queryClient]
  );

  const handleTaxonomySheetClose = useCallback(
    (isOpen: boolean) => {
      if (!isOpen) {
        setTaxonomySheetOpen(false);
        setTaxonomySpecies(null);
        setCreateName("");
        if (pendingOption) {
          onSelect(pendingOption);
          setPendingOption(null);
        }
        onOpenChange?.(false);
      }
    },
    [pendingOption, onSelect, onOpenChange]
  );

  // Trigger button
  const currentOption = options.find((o) => o.value === value);
  const displayLabel = value ?? "Select label...";
  const dotColor = currentOption
    ? getCategoryColor(currentOption.category)
    : undefined;

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        className="h-6 px-1.5 gap-1 text-xs font-medium justify-start"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(true);
        }}
      >
        {dotColor && !hideDot && (
          <div
            className="w-2 h-2 rounded-full shrink-0"
            style={{ backgroundColor: dotColor }}
          />
        )}
        {!hideLabel && (
          <span className="truncate max-w-[120px]">{displayLabel}</span>
        )}
        <ChevronsUpDown className="h-3 w-3 opacity-50 shrink-0" />
      </Button>

      <Dialog open={open} onOpenChange={handleOpenChange}>
        <DialogContent
          className="max-w-md overflow-hidden p-0"
          onClick={(e) => e.stopPropagation()}
        >
          <DialogTitle className="sr-only">Select label</DialogTitle>
          <Command shouldFilter={false}>
            <CommandInput
              placeholder="Search labels..."
              value={search}
              onValueChange={setSearch}
            />
            <CommandList onWheel={(e) => e.stopPropagation()}>
              {/* Quick labels (pinned) */}
              {filteredPinned && filteredPinned.length > 0 && (
                <CommandGroup heading="Quick labels">
                  {filteredPinned.map(({ key, option: opt }) => (
                    <CommandItem
                      key={`pinned-${key}`}
                      value={`${key}-${opt.value}`}
                      onSelect={() => handleSelect(opt)}
                    >
                      <code className="bg-zinc-100 text-zinc-500 px-1 rounded text-[10px] mr-1.5">
                        {key}
                      </code>
                      <div
                        className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                        style={{
                          backgroundColor: getCategoryColor(opt.category),
                        }}
                      />
                      {opt.value}
                      <Check
                        className={cn(
                          "ml-auto h-3 w-3",
                          value === opt.value ? "opacity-100" : "opacity-0"
                        )}
                      />
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}

              {/* General options */}
              {filteredGeneral.length > 0 && (
                <CommandGroup heading="General">
                  {filteredGeneral.map((opt) => (
                    <CommandItem
                      key={opt.value}
                      value={opt.value}
                      onSelect={() => handleSelect(opt)}
                    >
                      <div
                        className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                        style={{
                          backgroundColor: getCategoryColor(opt.category),
                        }}
                      />
                      {opt.value}
                      <Check
                        className={cn(
                          "ml-auto h-3 w-3",
                          value === opt.value ? "opacity-100" : "opacity-0"
                        )}
                      />
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}

              {/* Model species */}
              {filteredModelSpecies.length > 0 && (
                <CommandGroup heading="Species">
                  {filteredModelSpecies.map((opt) => (
                    <CommandItem
                      key={opt.value}
                      value={opt.value}
                      onSelect={() => handleSelect(opt)}
                    >
                      <div
                        className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                        style={{
                          backgroundColor: getCategoryColor(opt.category),
                        }}
                      />
                      {opt.value}
                      <Check
                        className={cn(
                          "ml-auto h-3 w-3",
                          value === opt.value ? "opacity-100" : "opacity-0"
                        )}
                      />
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}

              {/* Custom species */}
              {filteredCustomSpecies.length > 0 && (
                <CommandGroup heading="Custom species">
                  {filteredCustomSpecies.map((opt) => (
                    <CommandItem
                      key={opt.value}
                      value={opt.value}
                      onSelect={() => handleSelect(opt)}
                      className="group"
                    >
                      <div
                        className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                        style={{
                          backgroundColor: getCategoryColor(opt.category),
                        }}
                      />
                      {opt.value}
                      <span className="ml-auto flex items-center gap-0.5">
                        <button
                          type="button"
                          className="p-0.5 rounded hover:bg-accent opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={(e) => {
                            e.stopPropagation();
                            const cs = customSpeciesMap.get(opt.customId!);
                            if (cs) {
                              setTaxonomySpecies(cs);
                              setTaxonomySheetOpen(true);
                            }
                          }}
                        >
                          <Pencil className="h-3 w-3 text-muted-foreground" />
                        </button>
                        <Check
                          className={cn(
                            "h-3 w-3 group-hover:invisible",
                            value === opt.value ? "opacity-100" : "opacity-0"
                          )}
                        />
                      </span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}

              {/* Add new label */}
              {showAddNew && (
                <CommandGroup>
                  <CommandItem onSelect={handleAddNew}>
                    <Plus className="h-4 w-4 mr-1.5 text-muted-foreground" />
                    {search.trim()
                      ? <>Add new label for &ldquo;{search.trim()}&rdquo;</>
                      : "Add new label"}
                  </CommandItem>
                </CommandGroup>
              )}

              {/* Empty state */}
              {!showAddNew &&
                filteredGeneral.length === 0 &&
                filteredModelSpecies.length === 0 &&
                filteredCustomSpecies.length === 0 &&
                (!filteredPinned || filteredPinned.length === 0) && (
                  <div className="py-6 text-center text-sm text-muted-foreground">
                    {isLoading ? "Loading..." : "No label found."}
                  </div>
                )}
            </CommandList>
          </Command>
        </DialogContent>
      </Dialog>

      {projectId && (
        <TaxonomySheet
          species={taxonomySpecies}
          projectId={projectId}
          initialName={createName}
          open={taxonomySheetOpen}
          onOpenChange={handleTaxonomySheetClose}
          onCreated={handleTaxonomySheetCreated}
        />
      )}
    </>
  );
}
