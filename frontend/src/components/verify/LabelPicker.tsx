/**
 * Unified label picker — command palette for detection labels.
 *
 * Opens a centered dialog with searchable groups: pinned shortcuts,
 * general labels (person/vehicle), and labels from the classification
 * model. An "Add new label" action at the bottom opens the TaxonomySheet
 * slideout for creating a new custom label with optional GBIF lookup.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronsUpDown, Pencil, Plus, type LucideIcon } from "lucide-react";
import { toast } from "sonner";
import { cn } from "../../lib/utils";
import { getCategoryColor } from "../../lib/detection-utils";
import { getSpeciesColor } from "../../utils/species-colors";
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

/** Get the dot color for a label option: species color for labels, category color for general. */
function getLabelDotColor(opt: LabelOption): string {
  return opt.label ? getSpeciesColor(opt.value) : getCategoryColor(opt.category);
}

/** Clean up a label for display: replace underscores, capitalize first letter. */
function formatLabel(name: string): string {
  const cleaned = name.replace(/[_-]+/g, " ").trim();
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

function TaxonomyCaption({
  commonName,
  caption,
}: {
  commonName?: string | null;
  caption?: string | null;
}) {
  const parts: string[] = [];
  if (commonName) parts.push(formatLabel(commonName));
  if (caption) parts.push(caption);
  const text = parts.length > 0 ? parts.join(" · ") : "no taxonomy";
  return (
    <span className="text-[10px] text-muted-foreground truncate">
      {text}
    </span>
  );
}

const RECENT_LABELS_MAX = 3;

function getRecentLabelKeys(projectId?: string): string[] {
  if (!projectId) return [];
  try {
    const raw = localStorage.getItem(`addaxai-recent-labels-${projectId}`);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function pushRecentLabel(value: string, projectId?: string): void {
  if (!projectId) return;
  const key = `addaxai-recent-labels-${projectId}`;
  const current = getRecentLabelKeys(projectId).filter((v) => v !== value);
  current.unshift(value);
  localStorage.setItem(key, JSON.stringify(current.slice(0, RECENT_LABELS_MAX)));
}

export interface PinnedOption {
  key: number;
  option: LabelOption;
}

interface LabelPickerProps {
  value: string | null;
  /** Display name override for the trigger button (falls back to formatted value). */
  displayName?: string | null;
  onSelect: (option: LabelOption) => void;
  options: LabelOption[];
  isLoading?: boolean;
  forceOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  pinnedOptions?: PinnedOption[];
  hideDot?: boolean;
  hideLabel?: boolean;
  projectId?: string;
  /** Icon rendered on the trigger button. Defaults to ChevronsUpDown
   *  (combobox idiom). The modal sidebar uses `Tag` for the always-
   *  visible active-species picker so it reads as "set label" rather
   *  than "expand combobox". */
  triggerIcon?: LucideIcon;
  /** Tooltip on the trigger button. Defaults to the current value's
   *  display label; pass a sentence-style hint for the always-visible
   *  sidebar control. */
  triggerTitle?: string;
  /** Suppress the trigger button. The picker becomes just the dialog,
   *  driven entirely by `forceOpen` from a sibling button. Used by
   *  the BulkActionBar, where the Relabel button next to the picker
   *  is the trigger and a second "Select label..." button would be
   *  redundant. */
  headless?: boolean;
}

export function LabelPicker({
  value,
  displayName,
  onSelect,
  options,
  isLoading,
  forceOpen,
  onOpenChange,
  pinnedOptions,
  hideDot,
  hideLabel,
  projectId,
  triggerIcon: TriggerIcon = ChevronsUpDown,
  triggerTitle,
  headless,
}: LabelPickerProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [taxonomyLabel, setTaxonomyLabel] = useState<CustomLabelResponse | null>(null);
  const [taxonomySheetOpen, setTaxonomySheetOpen] = useState(false);
  const pendingOptionRef = useRef<LabelOption | null>(null);
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

  // Recent labels (persisted per project in localStorage)
  const [recentKeys, setRecentKeys] = useState<string[]>(() =>
    getRecentLabelKeys(projectId)
  );

  const handleSelect = useCallback(
    (option: LabelOption) => {
      pushRecentLabel(option.value, projectId);
      setRecentKeys(getRecentLabelKeys(projectId));
      onSelect(option);
      handleOpenChange(false);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [onSelect, projectId]
  );

  // Manual filtering
  const searchLower = search.toLowerCase().trim();

  const matchesSearch = (o: LabelOption) =>
    !searchLower
    || o.value.toLowerCase().includes(searchLower)
    || o.displayName.toLowerCase().includes(searchLower);

  const filteredPinned = pinnedOptions?.filter(
    ({ option }) => matchesSearch(option)
  );

  const generalOptions = options.filter((o) => o.label === null);
  const modelLabels = options.filter((o) => o.label !== null && !o.isCustom);
  const customLabelOpts = options.filter((o) => o.label !== null && o.isCustom);

  // Build recent options from localStorage keys, excluding pinned labels
  const pinnedValues = useMemo(
    () => new Set(pinnedOptions?.map(({ option }) => option.value) ?? []),
    [pinnedOptions]
  );
  const recentOptions = useMemo(() => {
    const optMap = new Map(options.map((o) => [o.value, o]));
    return recentKeys
      .filter((k) => optMap.has(k) && !pinnedValues.has(k))
      .map((k) => optMap.get(k)!);
  }, [recentKeys, options, pinnedValues]);
  const filteredRecent = recentOptions.filter(matchesSearch);

  const filteredGeneral = generalOptions.filter(matchesSearch);
  const allFilteredModelLabels = modelLabels.filter(matchesSearch);
  const filteredModelLabels = allFilteredModelLabels.slice(0, 50);
  const hasMoreModelLabels = allFilteredModelLabels.length > 50;
  const filteredCustomLabels = customLabelOpts.filter(matchesSearch);

  const showAddNew = !!projectId;

  const handleAddNew = useCallback(() => {
    if (!projectId) return;
    setCreateName(search.trim());
    setOpen(false);
    setSearch("");
    // Tell the parent the picker closed. Without this a CONTROLLED
    // parent (BulkActionBar via relabelOpen) keeps thinking the picker
    // is open, and re-opens it the next time the user selects a crop.
    onOpenChange?.(false);
    setTaxonomyLabel(null);
    setTaxonomySheetOpen(true);
  }, [projectId, search, onOpenChange]);

  const handleTaxonomySheetCreated = useCallback(
    (created: CustomLabelResponse) => {
      queryClient.invalidateQueries({ queryKey: ["custom-labels", projectId] });
      queryClient.invalidateQueries({ queryKey: ["label-tree"] });
      const option: LabelOption = {
        value: created.name,
        category: "animal",
        label: created.name,
      };
      // Store in both state and ref. The ref is read by
      // handleTaxonomySheetClose in the same tick (before React
      // re-renders), so the close handler always sees the latest value.
      pendingOptionRef.current = option;
    },
    [projectId, queryClient]
  );

  const handleTaxonomySheetClose = useCallback(
    (isOpen: boolean) => {
      if (!isOpen) {
        setTaxonomySheetOpen(false);
        setTaxonomyLabel(null);
        setCreateName("");
        const pending = pendingOptionRef.current;
        if (pending) {
          onSelect(pending);
          toast.success(`Label "${pending.value}" created and applied`);
          pendingOptionRef.current = null;
        }
        onOpenChange?.(false);
      }
    },
    [onSelect, onOpenChange]
  );

  // Trigger button
  const currentOption = options.find((o) => o.value === value);
  const displayLabel = displayName ? formatLabel(displayName) : value ? formatLabel(value) : "Select label...";
  const dotColor = currentOption
    ? getLabelDotColor(currentOption)
    : value
      ? getSpeciesColor(value)
      : undefined;

  return (
    <>
      {!headless && (
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-1.5 gap-1 text-xs font-medium justify-start"
          title={triggerTitle ?? displayLabel}
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
            <span className="truncate max-w-[180px]">{displayLabel}</span>
          )}
          <TriggerIcon className="h-3 w-3 opacity-50 shrink-0" />
        </Button>
      )}

      <Dialog open={open} onOpenChange={handleOpenChange}>
        <DialogContent
          className="max-w-xl overflow-hidden p-0"
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
                      value={`${key}-${opt.value} ${opt.displayName}`}
                      onSelect={() => handleSelect(opt)}
                      className="odd:bg-muted/40"
                    >
                      <code className="bg-zinc-100 text-zinc-500 px-1 rounded text-[10px] mr-1.5">
                        {key}
                      </code>
                      <div
                        className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                        style={{
                          backgroundColor: getLabelDotColor(opt),
                        }}
                      />
                      <div className="flex flex-col min-w-0">
                        <span>{opt.displayName}</span>
                        <TaxonomyCaption commonName={opt.label} caption={opt.taxonomyCaption} />
                      </div>
                      <Check
                        className={cn(
                          "ml-auto h-3 w-3 shrink-0",
                          value === opt.value ? "opacity-100" : "opacity-0"
                        )}
                      />
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}

              {/* Recent labels */}
              {filteredRecent.length > 0 && (
                <CommandGroup heading="Recent">
                  {filteredRecent.map((opt) => (
                    <CommandItem
                      key={`recent-${opt.value}`}
                      value={`recent-${opt.value} ${opt.displayName}`}
                      onSelect={() => handleSelect(opt)}
                      className="odd:bg-muted/40"
                    >
                      <div
                        className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                        style={{
                          backgroundColor: getLabelDotColor(opt),
                        }}
                      />
                      <div className="flex flex-col min-w-0">
                        <span>{opt.displayName}</span>
                        <TaxonomyCaption commonName={opt.label} caption={opt.taxonomyCaption} />
                      </div>
                      <Check
                        className={cn(
                          "ml-auto h-3 w-3 shrink-0",
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
                      value={`${opt.value} ${opt.displayName}`}
                      onSelect={() => handleSelect(opt)}
                      className="odd:bg-muted/40"
                    >
                      <div
                        className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                        style={{
                          backgroundColor: getLabelDotColor(opt),
                        }}
                      />
                      <div className="flex flex-col min-w-0">
                        <span>{opt.displayName}</span>
                        <TaxonomyCaption commonName={opt.label} caption={opt.taxonomyCaption} />
                      </div>
                      <Check
                        className={cn(
                          "ml-auto h-3 w-3 shrink-0",
                          value === opt.value ? "opacity-100" : "opacity-0"
                        )}
                      />
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}

              {/* Model labels */}
              {filteredModelLabels.length > 0 && (
                <CommandGroup heading="Model labels">
                  {filteredModelLabels.map((opt) => (
                    <CommandItem
                      key={opt.value}
                      value={`${opt.value} ${opt.displayName}`}
                      onSelect={() => handleSelect(opt)}
                      className="odd:bg-muted/40"
                    >
                      <div
                        className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                        style={{
                          backgroundColor: getLabelDotColor(opt),
                        }}
                      />
                      <div className="flex flex-col min-w-0">
                        <span>{opt.displayName}</span>
                        <TaxonomyCaption commonName={opt.label} caption={opt.taxonomyCaption} />
                      </div>
                      <Check
                        className={cn(
                          "ml-auto h-3 w-3 shrink-0",
                          value === opt.value ? "opacity-100" : "opacity-0"
                        )}
                      />
                    </CommandItem>
                  ))}
                  {hasMoreModelLabels && (
                    <div className="px-2 py-2 text-center text-[11px] text-muted-foreground">
                      Showing 50 of {allFilteredModelLabels.length} labels. Type to search.
                    </div>
                  )}
                </CommandGroup>
              )}

              {/* Custom labels */}
              {filteredCustomLabels.length > 0 && (
                <CommandGroup heading="Custom labels">
                  {filteredCustomLabels.map((opt) => (
                    <CommandItem
                      key={opt.value}
                      value={`${opt.value} ${opt.displayName}`}
                      onSelect={() => handleSelect(opt)}
                      className="group odd:bg-muted/40"
                    >
                      <div
                        className="w-2 h-2 rounded-full shrink-0 mr-1.5"
                        style={{
                          backgroundColor: getLabelDotColor(opt),
                        }}
                      />
                      <div className="flex flex-col min-w-0">
                        <span>{opt.displayName}</span>
                        <TaxonomyCaption commonName={opt.label} caption={opt.taxonomyCaption} />
                      </div>
                      <span className="ml-auto flex items-center gap-0.5 shrink-0">
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
                filteredRecent.length === 0 &&
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
