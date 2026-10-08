/**
 * One headline number: label, value, and a line of context.
 *
 * A number with nothing beside it cannot be judged, so `note` should
 * always say what the value is out of or what it compares to.
 *
 * Ported from AddaxAI Connect without its week-on-week delta and
 * sparkline: AddaxAI data arrives per SD card, so there is no "previous
 * week" to compare against.
 */

import { Card, CardContent } from "../ui/card";

interface StatTileProps {
  label: string;
  value: string;
  note?: string;
  loading?: boolean;
  /** Grid placement from the page. The tile does not choose its own width. */
  className?: string;
}

export function StatTile({
  label,
  value,
  note,
  loading,
  className = "",
}: StatTileProps) {
  return (
    <Card className={`h-full ${className}`}>
      <CardContent className="flex h-full flex-col gap-1 p-4">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        <p className="text-2xl font-bold tabular-nums">
          {loading ? "..." : value}
        </p>
        {note && !loading && (
          <p className="text-xs text-muted-foreground">{note}</p>
        )}
      </CardContent>
    </Card>
  );
}
