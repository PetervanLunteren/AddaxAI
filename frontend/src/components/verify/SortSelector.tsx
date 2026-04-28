/**
 * Sort selector for the Events and Files verify tabs.
 *
 * Four modes: newest, oldest, random, cls_low. Random is seeded so
 * pagination and modal Next/Prev stay consistent; the seed lives in URL
 * state. The Shuffle button regenerates the seed.
 *
 * cls_low is hidden when the project has no classification model: the
 * sort key would be NULL for every row, making the option useless.
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

function newSeed(): number {
  return Math.floor(Math.random() * RANDOM_SEED_MAX);
}

interface SortSelectorProps {
  sort: VerifySort;
  seed: number | null;
  onChange: (sort: VerifySort, seed: number | null) => void;
  showClsLow: boolean;
}

export function SortSelector({
  sort,
  seed,
  onChange,
  showClsLow,
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
      <span className="text-xs text-muted-foreground">Sort</span>
      <Select value={sort} onValueChange={(v) => handleSortChange(v as VerifySort)}>
        <SelectTrigger className="h-8 w-44 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="newest">Newest first</SelectItem>
          <SelectItem value="oldest">Oldest first</SelectItem>
          <SelectItem value="random">Random</SelectItem>
          {showClsLow && (
            <SelectItem value="cls_low">Lowest confidence first</SelectItem>
          )}
        </SelectContent>
      </Select>
      {sort === "random" && (
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8"
          title="Shuffle again"
          onClick={() => onChange("random", newSeed())}
        >
          <Shuffle className="h-3.5 w-3.5" />
        </Button>
      )}
    </div>
  );
}
