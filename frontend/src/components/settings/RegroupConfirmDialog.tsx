/**
 * Confirmation shown before an independence-interval change that would
 * regroup events. Regrouped events lose their confirmation and manual
 * counts (events keep them only when their exact grouping is unchanged),
 * so this is a type-to-confirm gate that spells out the loss with a real
 * example and the totals.
 *
 * Shared by the project Settings page and the folder-run "Refine results"
 * slideout. It is only shown when the interval changed AND real work is at
 * risk (see `fetchRegroupImpact` in lib/reprocessSettings).
 */

import {
  formatInterval,
  type RegroupImpact,
} from "../../lib/reprocessSettings";
import { Callout } from "../ui/callout";
import { TypeToConfirmDialog } from "../ui/type-to-confirm-dialog";

function plural(n: number, word: string): string {
  return n === 1 ? word : `${word}s`;
}

export function RegroupConfirmDialog({
  open,
  onOpenChange,
  impact,
  fromInterval,
  toInterval,
  onConfirm,
  isPending = false,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  impact: RegroupImpact;
  /** Independence interval in seconds, before and after the edit. */
  fromInterval: number;
  toInterval: number;
  onConfirm: () => void;
  isPending?: boolean;
}) {
  const { confirmed_at_risk, counts_at_risk, total_confirmed, example } = impact;
  const kept = total_confirmed - confirmed_at_risk;
  const verb = toInterval > fromInterval ? "merges" : "splits";

  const countSummary =
    example && example.observations.length > 0
      ? example.observations.map((o) => `${o.count} ${o.label}`).join(", ")
      : "counts";

  return (
    <TypeToConfirmDialog
      open={open}
      onOpenChange={onOpenChange}
      title="Change the independence interval?"
      description="A different interval regroups events that are close together in time. Confirmations and manual counts on events that regroup are reset."
      confirmWord="REGROUP"
      confirmLabel="Regroup and reprocess"
      pendingLabel="Applying..."
      onConfirm={onConfirm}
      isPending={isPending}
      variant="warning"
    >
      <Callout variant="warning">
        Changing the interval from {formatInterval(fromInterval)} to{" "}
        {formatInterval(toInterval)}{" "}
        {confirmed_at_risk > 0 ? (
          <>
            {verb} {confirmed_at_risk} confirmed{" "}
            {plural(confirmed_at_risk, "event")}, so they need re-confirming
            {counts_at_risk > 0 && (
              <>
                , and {counts_at_risk} manual {plural(counts_at_risk, "count")}{" "}
                reset to the AI's
              </>
            )}
            .{" "}
            {kept > 0 && (
              <>
                {kept} confirmed {plural(kept, "event")} keep the same
                boundaries and stay confirmed.
              </>
            )}
          </>
        ) : (
          <>
            regroups events, so {counts_at_risk} manual{" "}
            {plural(counts_at_risk, "count")} will reset to the AI's.
          </>
        )}
      </Callout>

      {example && (
        <Callout variant="info">
          For example, you confirmed {countSummary} in the event
          {example.time_range ? <> on {example.time_range}</> : null}. The new
          interval{" "}
          {example.maps_to <= 1
            ? "merges it with nearby events into one"
            : `splits it into ${example.maps_to} events`}
          , so its confirmed count can't carry over to the new grouping.
        </Callout>
      )}

      <p className="text-xs text-muted-foreground">
        Your verified labels are not affected, only confirmed counts on events
        that actually regroup.
      </p>
    </TypeToConfirmDialog>
  );
}
