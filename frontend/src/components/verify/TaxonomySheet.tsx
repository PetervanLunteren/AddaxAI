/**
 * TaxonomySheet — right-side slideout for adding/editing custom species.
 *
 * Users search GBIF to look up taxonomy, which places the species in
 * the hierarchical filter tree. Suggestion cards auto-fill the five
 * taxonomy fields. Fields can also be edited manually.
 *
 * Supports two modes:
 * - Edit: species prop is a CustomSpeciesResponse — updates existing entry
 * - Create: species is null — creates a new custom species on save
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronDown, ChevronRight, Loader2, Pencil, Search } from "lucide-react";
import { toast } from "sonner";
import { cn } from "../../lib/utils";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "../ui/sheet";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { projectsApi } from "../../api/projects";
import type {
  CustomSpeciesResponse,
  CustomSpeciesUpdate,
  GBIFSuggestion,
} from "../../api/types";

interface TaxonomySheetProps {
  species: CustomSpeciesResponse | null;
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Pre-filled name for create mode (when species is null). */
  initialName?: string;
  /** Called after a new custom species is created (create mode only). */
  onCreated?: (created: CustomSpeciesResponse) => void;
}

function hasTaxonomy(species: CustomSpeciesResponse | null): boolean {
  if (!species) return false;
  return !!(
    species.taxon_class ||
    species.taxon_order ||
    species.taxon_family ||
    species.taxon_genus ||
    species.taxon_species
  );
}

export function TaxonomySheet({
  species,
  projectId,
  open,
  onOpenChange,
  initialName,
  onCreated,
}: TaxonomySheetProps) {
  const queryClient = useQueryClient();

  const [speciesName, setSpeciesName] = useState("");
  const [gbifQuery, setGbifQuery] = useState("");
  const [taxonClass, setTaxonClass] = useState("");
  const [taxonOrder, setTaxonOrder] = useState("");
  const [taxonFamily, setTaxonFamily] = useState("");
  const [taxonGenus, setTaxonGenus] = useState("");
  const [taxonSpecies, setTaxonSpecies] = useState("");

  const [suggestions, setSuggestions] = useState<GBIFSuggestion[]>([]);
  const [fetching, setFetching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const [taxonomyExpanded, setTaxonomyExpanded] = useState(false);
  const [selectedGbifKey, setSelectedGbifKey] = useState<number | null>(null);

  const isCreateMode = species === null;
  const isEditing = hasTaxonomy(species);
  const gbifInputRef = useRef<HTMLInputElement>(null);

  // Reset form when species changes or sheet opens in create mode
  useEffect(() => {
    if (!open) return;
    if (species) {
      setSpeciesName(species.name);
      setGbifQuery(species.name);
      setTaxonClass(species.taxon_class ?? "");
      setTaxonOrder(species.taxon_order ?? "");
      setTaxonFamily(species.taxon_family ?? "");
      setTaxonGenus(species.taxon_genus ?? "");
      setTaxonSpecies(species.taxon_species ?? "");
    } else {
      setSpeciesName(initialName ?? "");
      setGbifQuery(initialName ?? "");
      setTaxonClass("");
      setTaxonOrder("");
      setTaxonFamily("");
      setTaxonGenus("");
      setTaxonSpecies("");
    }
    setSuggestions([]);
    setHasSearched(false);
    setTaxonomyExpanded(false);
    setSelectedGbifKey(null);
  }, [species, open, initialName]);

  const fetchSuggestions = useCallback(async (query: string) => {
    const trimmed = query.trim();
    if (!trimmed) {
      setSuggestions([]);
      return;
    }
    setFetching(true);
    try {
      const results = await projectsApi.gbifSuggest(trimmed);
      setSuggestions(results);
      setHasSearched(true);
    } catch {
      toast.error("Could not reach GBIF");
    } finally {
      setFetching(false);
    }
  }, []);

  // Auto-search on mount (when sheet opens with a name)
  useEffect(() => {
    const name = species?.name ?? initialName;
    if (open && name && !hasSearched) {
      fetchSuggestions(name);
    }
  }, [open, species, initialName, hasSearched, fetchSuggestions]);

  const handleGbifSearch = () => {
    fetchSuggestions(gbifQuery);
  };

  const handlePickSuggestion = (s: GBIFSuggestion) => {
    setTaxonClass(s.taxon_class ?? "");
    setTaxonOrder(s.taxon_order ?? "");
    setTaxonFamily(s.taxon_family ?? "");
    setTaxonGenus(s.taxon_genus ?? "");
    setTaxonSpecies(s.taxon_species ?? "");
    setSelectedGbifKey(s.gbif_key);
  };

  const updateMutation = useMutation({
    mutationFn: (data: CustomSpeciesUpdate) =>
      projectsApi.updateCustomSpecies(projectId, species!.id, data),
    onSuccess: () => {
      toast.success("Taxonomy saved");
      queryClient.invalidateQueries({ queryKey: ["custom-species", projectId] });
      queryClient.invalidateQueries({ queryKey: ["species-tree"] });
      onOpenChange(false);
    },
    onError: () => {
      toast.error("Failed to save taxonomy");
    },
  });

  const createMutation = useMutation({
    mutationFn: async (name: string) => {
      const created = await projectsApi.createCustomSpecies(projectId, name);
      // Immediately update taxonomy fields
      const updated = await projectsApi.updateCustomSpecies(projectId, created.id, {
        taxon_class: taxonClass || null,
        taxon_order: taxonOrder || null,
        taxon_family: taxonFamily || null,
        taxon_genus: taxonGenus || null,
        taxon_species: taxonSpecies || null,
      });
      return updated;
    },
    onSuccess: (created) => {
      toast.success(`Label "${created.name}" created`);
      queryClient.invalidateQueries({ queryKey: ["custom-species", projectId] });
      queryClient.invalidateQueries({ queryKey: ["species-tree"] });
      onCreated?.(created);
      onOpenChange(false);
    },
    onError: () => {
      toast.error("Failed to create label");
    },
  });

  const handleSave = () => {
    const trimmedName = speciesName.trim();
    if (!trimmedName) {
      toast.error("Name is required");
      return;
    }

    if (isCreateMode) {
      createMutation.mutate(trimmedName);
    } else {
      updateMutation.mutate({
        name: trimmedName !== species?.name ? trimmedName || null : undefined,
        taxon_class: taxonClass || null,
        taxon_order: taxonOrder || null,
        taxon_family: taxonFamily || null,
        taxon_genus: taxonGenus || null,
        taxon_species: taxonSpecies || null,
      });
    }
  };

  const isPending = updateMutation.isPending || createMutation.isPending;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="sm:max-w-xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>
            {isCreateMode ? "Add new label" : isEditing ? "Edit custom label" : "Add custom label"}
          </SheetTitle>
          <SheetDescription>
            {isCreateMode
              ? "Create a new custom label. Search GBIF to fill in taxonomy automatically, or enter fields manually. If this label has no taxonomy (e.g. \"bait\" or \"setup\"), leave all fields blank."
              : "Add taxonomic information to your custom species. Search GBIF to fill in the fields automatically, or enter them manually. Scientific names often give better search results. If this label has no taxonomy (e.g. \"bait\" or \"setup\"), leave all fields blank."}
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-6 py-4">
          {/* Name */}
          <div>
            <label className="text-sm font-medium">Name</label>
            <Input
              value={speciesName}
              onChange={(e) => setSpeciesName(e.target.value)}
              placeholder="e.g. spotted hyena"
              autoFocus={isCreateMode}
            />
          </div>

          {/* GBIF search */}
          <div className="space-y-3">
            <p className="text-xs font-medium text-muted-foreground">
              GBIF lookup
            </p>
            <div className="flex gap-2">
              <Input
                ref={gbifInputRef}
                value={gbifQuery}
                onChange={(e) => setGbifQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleGbifSearch();
                }}
                placeholder="Search taxonomy..."
              />
              <Button
                variant="outline"
                size="default"
                onClick={handleGbifSearch}
                disabled={fetching || !gbifQuery.trim()}
                className="shrink-0"
              >
                {fetching ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Search className="h-4 w-4" />
                )}
                <span className="ml-1.5">GBIF</span>
              </Button>
            </div>

            {/* Suggestions */}
            {suggestions.length > 0 && (
              <div className="rounded-md border divide-y">
                {suggestions.map((s) => {
                  const isSelected = selectedGbifKey === s.gbif_key;
                  return (
                    <button
                      key={s.gbif_key}
                      type="button"
                      className={cn(
                        "w-full text-left px-2.5 py-2 transition-colors first:rounded-t-md last:rounded-b-md",
                        !isSelected && "hover:bg-accent"
                      )}
                      style={isSelected ? { backgroundColor: "rgba(15, 96, 100, 0.08)" } : undefined}
                      onClick={() => handlePickSuggestion(s)}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm">{s.canonical_name}</span>
                        <span className="flex items-center gap-1.5">
                          {isSelected && (
                            <Check className="h-3.5 w-3.5" style={{ color: "#0f6064" }} />
                          )}
                          <span className="text-[10px] text-muted-foreground shrink-0">
                            {s.rank}
                          </span>
                        </span>
                      </div>
                      <div className="text-[11px] text-muted-foreground">
                        {[s.taxon_class, s.taxon_order, s.taxon_family, s.taxon_genus, s.taxon_species]
                          .filter(Boolean)
                          .join(" › ")}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            {hasSearched && !fetching && suggestions.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-2">
                No results found
              </p>
            )}
          </div>

          {/* Taxonomy status + editable fields */}
          <div className="space-y-3">
            {(() => {
              const parts = [taxonClass, taxonOrder, taxonFamily, taxonGenus, taxonSpecies].filter(Boolean);
              const hasParts = parts.length > 0;

              return (
                <>
                  {/* Status indicator */}
                  {hasParts ? (
                    <div
                      className="flex items-start gap-2 rounded-md border px-3 py-2"
                      style={{ backgroundColor: "rgba(15, 96, 100, 0.08)", borderColor: "rgba(15, 96, 100, 0.25)" }}
                    >
                      <Check className="h-4 w-4 mt-0.5 shrink-0" style={{ color: "#0f6064" }} />
                      <div className="min-w-0">
                        <p className="text-sm font-medium" style={{ color: "#0f6064" }}>Taxonomy set</p>
                        <p className="text-xs truncate" style={{ color: "rgba(15, 96, 100, 0.75)" }}>
                          {parts.join(" › ")}
                        </p>
                      </div>
                      <button
                        type="button"
                        className="ml-auto shrink-0 p-1 rounded transition-colors"
                        style={{ color: "#0f6064" }}
                        onClick={() => setTaxonomyExpanded(!taxonomyExpanded)}
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 rounded-md bg-zinc-50 border border-zinc-200 px-3 py-2">
                      <span className="text-sm text-muted-foreground">
                        Taxonomy not set
                      </span>
                      <button
                        type="button"
                        className="ml-auto shrink-0 p-1 rounded hover:bg-zinc-100 transition-colors"
                        onClick={() => setTaxonomyExpanded(!taxonomyExpanded)}
                      >
                        <Pencil className="h-3.5 w-3.5 text-muted-foreground" />
                      </button>
                    </div>
                  )}

                  {taxonomyExpanded && (
                    <div className="space-y-3">
                      <div>
                        <label className="text-sm font-medium">Class</label>
                        <Input
                          value={taxonClass}
                          onChange={(e) => setTaxonClass(e.target.value)}
                          placeholder="e.g. Mammalia"
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">Order</label>
                        <Input
                          value={taxonOrder}
                          onChange={(e) => setTaxonOrder(e.target.value)}
                          placeholder="e.g. Carnivora"
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">Family</label>
                        <Input
                          value={taxonFamily}
                          onChange={(e) => setTaxonFamily(e.target.value)}
                          placeholder="e.g. Hyaenidae"
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">Genus</label>
                        <Input
                          value={taxonGenus}
                          onChange={(e) => setTaxonGenus(e.target.value)}
                          placeholder="e.g. Crocuta"
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">Species</label>
                        <Input
                          value={taxonSpecies}
                          onChange={(e) => setTaxonSpecies(e.target.value)}
                          placeholder="e.g. Crocuta crocuta"
                        />
                      </div>
                    </div>
                  )}
                </>
              );
            })()}
          </div>
        </div>

        <SheetFooter>
          <Button
            onClick={handleSave}
            disabled={isPending || !speciesName.trim()}
          >
            {isPending && (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            )}
            {isCreateMode ? "Create" : "Save"}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
