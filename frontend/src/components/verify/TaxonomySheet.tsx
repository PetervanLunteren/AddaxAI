/**
 * TaxonomySheet — right-side slideout for adding/editing custom species.
 *
 * Users search GBIF to look up taxonomy, which places the species in
 * the hierarchical filter tree. Suggestion cards auto-fill the five
 * taxonomy fields. Fields can also be edited manually.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Loader2, Pencil, Search } from "lucide-react";
import { toast } from "sonner";
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

  const isEditing = hasTaxonomy(species);
  const gbifInputRef = useRef<HTMLInputElement>(null);

  // Reset form when species changes
  useEffect(() => {
    if (species) {
      setSpeciesName(species.name);
      setGbifQuery(species.name);
      setTaxonClass(species.taxon_class ?? "");
      setTaxonOrder(species.taxon_order ?? "");
      setTaxonFamily(species.taxon_family ?? "");
      setTaxonGenus(species.taxon_genus ?? "");
      setTaxonSpecies(species.taxon_species ?? "");
      setSuggestions([]);
      setHasSearched(false);
      setTaxonomyExpanded(false);
    }
  }, [species]);

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

  // Auto-search on mount (when sheet opens with a species)
  useEffect(() => {
    if (open && species && !hasSearched) {
      fetchSuggestions(species.name);
    }
  }, [open, species, hasSearched, fetchSuggestions]);

  const handleGbifSearch = () => {
    fetchSuggestions(gbifQuery);
  };

  const handlePickSuggestion = (s: GBIFSuggestion) => {
    setTaxonClass(s.taxon_class ?? "");
    setTaxonOrder(s.taxon_order ?? "");
    setTaxonFamily(s.taxon_family ?? "");
    setTaxonGenus(s.taxon_genus ?? "");
    setTaxonSpecies(s.taxon_species ?? "");
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

  const handleSave = () => {
    const trimmedName = speciesName.trim();
    updateMutation.mutate({
      name: trimmedName !== species?.name ? trimmedName || null : undefined,
      taxon_class: taxonClass || null,
      taxon_order: taxonOrder || null,
      taxon_family: taxonFamily || null,
      taxon_genus: taxonGenus || null,
      taxon_species: taxonSpecies || null,
    });
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="sm:max-w-xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>
            {isEditing ? "Edit custom label" : "Add custom label"}
          </SheetTitle>
          <SheetDescription>
            Add taxonomic information to your custom species. Search
            GBIF to fill in the fields
            automatically, or enter them manually. Scientific names
            often give better search results. If this label has no taxonomy
            (e.g. &ldquo;bait&rdquo; or &ldquo;setup&rdquo;), leave all
            fields blank.
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-6 py-4">
          {/* Name */}
          <div>
            <label className="text-sm font-medium">Name</label>
            <Input
              value={speciesName}
              onChange={(e) => setSpeciesName(e.target.value)}
              placeholder="Species name"
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
                {suggestions.map((s) => (
                  <button
                    key={s.gbif_key}
                    type="button"
                    className="w-full text-left px-2.5 py-2 hover:bg-accent transition-colors first:rounded-t-md last:rounded-b-md"
                    onClick={() => handlePickSuggestion(s)}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm">{s.canonical_name}</span>
                      <span className="text-[10px] text-muted-foreground shrink-0">
                        {s.rank}
                      </span>
                    </div>
                    <div className="text-[11px] text-muted-foreground">
                      {[s.taxon_class, s.taxon_order, s.taxon_family, s.taxon_genus, s.taxon_species]
                        .filter(Boolean)
                        .join(" › ")}
                    </div>
                  </button>
                ))}
              </div>
            )}

            {hasSearched && !fetching && suggestions.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-2">
                No results found
              </p>
            )}
          </div>

          {/* Taxonomy summary / editable fields */}
          <div className="space-y-3">
            {(() => {
              const parts = [taxonClass, taxonOrder, taxonFamily, taxonGenus, taxonSpecies].filter(Boolean);
              const hasParts = parts.length > 0;

              return (
                <>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
                      onClick={() => setTaxonomyExpanded(!taxonomyExpanded)}
                    >
                      {taxonomyExpanded ? (
                        <ChevronDown className="h-3.5 w-3.5" />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5" />
                      )}
                      Taxonomy
                    </button>
                    {!taxonomyExpanded && hasParts && (
                      <button
                        type="button"
                        className="flex items-center gap-1.5 group"
                        onClick={() => setTaxonomyExpanded(true)}
                      >
                        <span className="text-xs text-muted-foreground">
                          {parts.join(" › ")}
                        </span>
                        <Pencil className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                      </button>
                    )}
                    {!taxonomyExpanded && !hasParts && (
                      <span className="text-xs text-muted-foreground italic">
                        not set
                      </span>
                    )}
                  </div>

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
            disabled={updateMutation.isPending}
          >
            {updateMutation.isPending && (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            )}
            Save
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
