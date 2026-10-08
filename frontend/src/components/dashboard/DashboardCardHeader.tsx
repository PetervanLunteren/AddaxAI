/**
 * The header every dashboard card wears: title, an optional (i) with the
 * method behind the numbers, a one-line caption, and optional controls on
 * the right. One component so the spacing between title and caption is
 * the same on every card; written out per card it drifted.
 *
 * When a card gets an (i) is a rule in FRONTEND_CONVENTIONS ("Card help").
 */

import type { ReactNode } from "react";

import { CardHeader, CardTitle } from "../ui/card";
import { DashboardAboutPopover } from "./DashboardAboutPopover";

interface DashboardCardHeaderProps {
  title: ReactNode;
  caption: ReactNode;
  /** Method text for the (i) popover; omit when the caption says it all. */
  info?: ReactNode;
  /** Small marker after the title, e.g. the missing-dates warning. */
  marker?: ReactNode;
  /** Controls on the right: selects, a refresh button. */
  actions?: ReactNode;
}

export function DashboardCardHeader({
  title,
  caption,
  info,
  marker,
  actions,
}: DashboardCardHeaderProps) {
  return (
    <CardHeader className="pb-2">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-1.5">
            <CardTitle className="text-lg">{title}</CardTitle>
            {marker}
            {info && <DashboardAboutPopover>{info}</DashboardAboutPopover>}
          </div>
          <p className="text-sm text-muted-foreground">{caption}</p>
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
    </CardHeader>
  );
}
