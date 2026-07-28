/**
 * Shared taxonomic rank constants.
 *
 * Mirrors app/ml/taxonomic_rank.py on the backend. Strings are the wire
 * contract; keep them in sync.
 */

export type TaxonomicRank =
  | "all"
  | "class"
  | "order"
  | "family"
  | "genus"
  | "species";

export const RANK_OPTIONS: { value: TaxonomicRank; label: string }[] = [
  { value: "all", label: "Most specific" },
  { value: "species", label: "Species" },
  { value: "genus", label: "Genus" },
  { value: "family", label: "Family" },
  { value: "order", label: "Order" },
  { value: "class", label: "Class" },
];

export const DEFAULT_TAXONOMIC_RANK: TaxonomicRank = "all";

export function isTaxonomicRank(value: unknown): value is TaxonomicRank {
  return (
    value === "all" ||
    value === "class" ||
    value === "order" ||
    value === "family" ||
    value === "genus" ||
    value === "species"
  );
}
