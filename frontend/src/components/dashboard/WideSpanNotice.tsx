/**
 * What a time chart shows instead of itself when the capture years span
 * more than it can draw (see lib/date-span.ts). Naming the first and last
 * year points the user at the bad date. Plain text, no link: the cards
 * that use it can sit inside a link themselves.
 */

import type { YearSpan } from "../../lib/date-span";
import { cn } from "../../lib/utils";

export function WideSpanNotice({ span, className }: { span: YearSpan; className?: string }) {
  return (
    <p className={cn("text-center text-sm text-muted-foreground", className)}>
      Capture dates run from {span.first} to {span.last}, too wide a span to
      chart. Check the capture dates on the Deployments page.
    </p>
  );
}
