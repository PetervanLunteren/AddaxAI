/**
 * Sort selector shared across the Verify tabs.
 *
 * Each tab passes the modes it supports via `availableSorts`. Events and
 * Files use newest / oldest / random / cls_low; Observations uses
 * similarity / similarity_reverse / newest / oldest / cls_low. The
 * Random mode is seeded so pagination and modal navigation stay stable;
 * the seed lives in URL state and the Shuffle button regenerates it.
 *
 * Rendered inline in the verify toolbar with a small "Sort" label.
 */

import { Shuffle } from "lucide-react";

import { Button } from "../ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import type { VerifySort } from "../../api/types";

const RANDOM_SEED_MAX = 2 ** 31;

const SORT_LABELS: Record<VerifySort, string> = {
  similarity: "Similarity",
  similarity_reverse: "Similarity (outliers first)",
  newest: "Newest first",
  oldest: "Oldest first",
  random: "Random",
  cls_low: "Lowest confidence first",
  events: "By event",
};

function newSeed(): number {
  return Math.floor(Math.random() * RANDOM_SEED_MAX);
}

interface SortSelectorProps {
  sort: VerifySort;
  seed: number | null;
  onChange: (sort: VerifySort, seed: number | null) => void;
  /** Modes the host tab supports. Order in the dropdown follows this array. */
  availableSorts: readonly VerifySort[];
}

export function SortSelector({
  sort,
  seed,
  onChange,
  availableSorts,
}: SortSelectorProps) {
  const handleSortChange = (next: VerifySort) => {
    if (next === "random") {
      onChange("random", seed ?? newSeed());
    } else {
      onChange(next, null);
    }
  };

  return (
    <div className="flex items-center gap-1.5">
      <Select value={sort} onValueChange={(v) => handleSortChange(v as VerifySort)}>
        <SelectTrigger className="h-8 w-52 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {availableSorts.map((mode) => (
            <SelectItem key={mode} value={mode}>
              {SORT_LABELS[mode]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {sort === "random" && (
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8 shrink-0"
          title="Shuffle again"
          onClick={() => onChange("random", newSeed())}
        >
          <Shuffle className="h-3.5 w-3.5" />
        </Button>
      )}
    </div>
  );
}
