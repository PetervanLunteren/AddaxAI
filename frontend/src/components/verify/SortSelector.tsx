/**
 * Sort selector shared across the verify toolbars.
 *
 * Each host passes the modes it supports via `availableSorts` (the
 * Counts gallery uses newest / oldest / random, the Labels grid uses
 * similarity / events). The option texts are self-labeling ("Sort by
 * ..."), so the trigger needs no extra label or icon. The Random mode
 * is seeded so pagination and modal navigation stay stable; the seed
 * lives in URL state and the Shuffle button regenerates it.
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
  similarity: "Sort by similarity",
  newest: "Sort by newest first",
  oldest: "Sort by oldest first",
  random: "Sort in random order",
  events: "Sort by event",
  path: "Sort by folder",
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
        <SelectTrigger className="h-8 min-h-8 w-52 text-sm">
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
