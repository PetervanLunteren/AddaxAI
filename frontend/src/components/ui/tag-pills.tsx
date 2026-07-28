/**
 * Inline pill display for key:value tag dictionaries.
 *
 * Designed for table cells: renders up to `maxVisible` tags as small
 * shadcn `Badge variant="secondary"` pills, then a `+N` pill for the
 * overflow. The overflow pill has a hover tooltip listing the rest.
 *
 * Visual hierarchy inside each pill: key in muted text, value in
 * normal foreground, separated by a colon. Long values truncate
 * inside the pill so the overall row stays predictable.
 */

import { Badge } from "./badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./tooltip";

interface TagPillsProps {
  tags: Record<string, string> | null | undefined;
  /** Maximum visible pills before collapsing the rest into a +N pill. */
  maxVisible?: number;
}

export function TagPills({ tags, maxVisible = 2 }: TagPillsProps) {
  const entries = Object.entries(tags ?? {});

  if (entries.length === 0) {
    return <span className="text-muted-foreground">{"\u2014"}</span>;
  }

  const visible = entries.slice(0, maxVisible);
  const overflow = entries.slice(maxVisible);

  return (
    <div className="flex items-center gap-1 flex-wrap">
      {visible.map(([key, value]) => (
        <Badge
          key={key}
          variant="secondary"
          className="text-xs font-normal max-w-[180px] gap-1"
        >
          <span className="text-muted-foreground shrink-0">{key}:</span>
          <span className="truncate min-w-0">{value}</span>
        </Badge>
      ))}

      {overflow.length > 0 && (
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant="secondary"
                className="text-xs font-normal cursor-default"
              >
                +{overflow.length}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <div className="space-y-0.5 text-xs max-w-[280px]">
                {overflow.map(([k, v]) => (
                  <div key={k}>
                    <span className="text-muted-foreground">{k}:</span> {v}
                  </div>
                ))}
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}
