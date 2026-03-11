/**
 * Unified label picker — command palette for detection labels.
 *
 * Opens a centered dialog with searchable groups: pinned shortcuts,
 * general labels (person/vehicle), and labels from the classification
 * model. An "Add new label" action at the bottom opens the TaxonomySheet
 * slideout for creating a new custom label with optional GBIF lookup.
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
import type { CustomLabelResponse } from "../../api/types";

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
  const [taxonomyLabel, setTaxonomyLabel] = useState<CustomLabelResponse | null>(null);
  const [taxonomySheetOpen, setTaxonomySheetOpen] = useState(false);
  const [pendingOption, setPendingOption] = useState<LabelOption | null>(null);
  const [createName, setCreateName] = useState("");
  const queryClient = useQueryClient();

  // Fetch custom labels for edit lookups
  const { data: customLabelsList } = useQuery({
    queryKey: ["custom-labels", projectId],
    queryFn: () => projectsApi.getCustomLabels(projectId!),
    enabled: !!projectId,
  });

  const customLabelsMap = useMemo(() => {
    const map = new Map<string, CustomLabelResponse>();
    if (customLabelsList) {
      for (const cl of customLabelsList) map.set(cl.id, cl);
    }
    return map;
  }, [customLabelsList]);

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

  const generalOptions = options.filter((o) => o.label === null);
  const modelLabels = options.filter((o) => o.label !== null && !o.isCustom);
  const customLabelOpts = options.filter((o) => o.label !== null && o.isCustom);

  const filteredGeneral = generalOptions.filter(
    (o) => !searchLower || o.value.toLowerCase().includes(searchLower)
  );
  const filteredModelLabels = modelLabels.filter(
    (o) => !searchLower || o.value.toLowerCase().includes(searchLower)
  );
  const filteredCustomLabels = customLabelOpts.filter(
    (o) => !searchLower || o.value.toLowerCase().includes(searchLower)
  );

  const hasResults =
    (filteredPinned && filteredPinned.length > 0) ||
    filteredGeneral.length > 0 ||
    filteredModelLabels.length > 0 ||
    filteredCustomLabels.length > 0;
  const showAddNew = !!projectId && (!searchLower || !hasResults);

  const handleAddNew = useCallback(() => {
    if (!projectId) return;
    setCreateName(search.trim());
    setOpen(false);
    setSearch("");
    setTaxonomyLabel(null);
    setTaxonomySheetOpen(true);
  }, [projectId, search]);

  const handleTaxonomySheetCreated = useCallback(
    (created: CustomLabelResponse) => {
      queryClient.invalidateQueries({ queryKey: ["custom-labels", projectId] });
      queryClient.invalidateQueries({ queryKey: ["label-tree"] });
      const option: LabelOption = {
        value: created.name,
        category: "animal",
        label: created.name,
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
        setTaxonomyLabel(null);
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

              {/* Model labels */}
              {filteredModelLabels.length > 0 && (
                <CommandGroup heading="Labels">
                  {filteredModelLabels.map((opt) => (
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

              {/* Custom labels */}
              {filteredCustomLabels.length > 0 && (
                <CommandGroup heading="Custom labels">
                  {filteredCustomLabels.map((opt) => (
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
                            const cl = customLabelsMap.get(opt.customId!);
                            if (cl) {
                              setTaxonomyLabel(cl);
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
                filteredModelLabels.length === 0 &&
                filteredCustomLabels.length === 0 &&
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
          customLabel={taxonomyLabel}
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
