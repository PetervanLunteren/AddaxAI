/**
 * Event count panel — the Counts-page count editor.
 *
 * The right-hand panel of EventDetailModal. Shows the event's species
 * and counts (effective_count = human override, else AI MaxN) and lets
 * the verifier adjust each count, reset to the AI value, add a species
 * the AI missed, and sign the event off. The count is event-scoped
 * (across all files), distinct from the per-file detection list.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Minus, Plus, RotateCcw, X } from "lucide-react";
import { toast } from "sonner";
import { eventsApi } from "../../api/events";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { getSpeciesColor } from "../../utils/species-colors";
import { LabelPicker } from "./LabelPicker";
import type { LabelOption } from "../../hooks/useLabelOptions";
import type { EventObservationItem } from "../../api/types";

interface EventCountPanelProps {
  eventId: string;
  projectId: string;
  observations: EventObservationItem[];
  confirmed: boolean;
  labelOptions: LabelOption[];
  labelOptionsLoading: boolean;
}

export function EventCountPanel({
  eventId,
  projectId,
  observations,
  confirmed,
  labelOptions,
  labelOptionsLoading,
}: EventCountPanelProps) {
  const queryClient = useQueryClient();
  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["event", eventId] });
    queryClient.invalidateQueries({ queryKey: ["events"] });
  };

  const setCount = useMutation({
    mutationFn: ({ obsId, count }: { obsId: string; count: number | null }) =>
      eventsApi.setObservationCount(eventId, obsId, count),
    onSuccess: invalidate,
    onError: (e: Error) =>
      toast.error("Could not set count", { description: e.message }),
  });
  const addSpecies = useMutation({
    mutationFn: (opt: LabelOption) =>
      eventsApi.addObservation(eventId, {
        category: opt.category,
        count: 1,
        label: opt.label ?? null,
      }),
    onSuccess: invalidate,
    onError: (e: Error) =>
      toast.error("Could not add species", { description: e.message }),
  });
  const removeObs = useMutation({
    mutationFn: (obsId: string) => eventsApi.deleteObservation(eventId, obsId),
    onSuccess: invalidate,
    onError: (e: Error) =>
      toast.error("Could not update", { description: e.message }),
  });
  const setConfirmed = useMutation({
    mutationFn: (v: boolean) => eventsApi.setConfirmed(eventId, v),
    onSuccess: invalidate,
    onError: (e: Error) =>
      toast.error("Could not update", { description: e.message }),
  });

  const busy =
    setCount.isPending || addSpecies.isPending || removeObs.isPending;

  return (
    <div className="mx-3 mt-3 rounded-lg border bg-muted/40 flex flex-col">
      <div className="flex items-center justify-between px-3 pt-3 pb-2">
        <div className="flex items-center gap-1.5">
          <h3 className="text-sm font-semibold">Counts</h3>
          <Badge variant="outline" className="text-xs">
            {observations.length}
          </Badge>
        </div>
        <Button
          size="sm"
          variant={confirmed ? "default" : "outline"}
          onClick={() => setConfirmed.mutate(!confirmed)}
          disabled={setConfirmed.isPending}
          className="h-7 gap-1.5"
        >
          <Check className="h-3.5 w-3.5" />
          {confirmed ? "Confirmed" : "Confirm"}
        </Button>
      </div>

      <div className="px-3 pb-3 space-y-1">
        {observations.map((obs) => {
          const name = resolveSpeciesName({
            common_name: obs.common_name,
            scientific_name: obs.scientific_name,
            label: obs.label,
            category: obs.category,
          });
          const colorKey = obs.label_taxonomy_id || obs.label || obs.category;
          const isOverridden = obs.effective_count !== obs.max_n;
          const isHumanOnly = obs.max_n === 0;
          return (
            <div
              key={obs.id}
              className="flex items-center justify-between gap-2 rounded border bg-white px-2 py-1.5 text-sm"
            >
              <span className="flex items-center gap-1.5 truncate">
                <span
                  className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm"
                  style={{ backgroundColor: getSpeciesColor(colorKey) }}
                />
                <span className="truncate">{name}</span>
                {isOverridden && !isHumanOnly && (
                  <span className="shrink-0 text-[10px] text-muted-foreground">
                    AI saw {obs.max_n}
                  </span>
                )}
              </span>
              <span className="flex shrink-0 items-center gap-1">
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  disabled={busy || obs.effective_count <= 0}
                  title="Decrease"
                  onClick={() =>
                    setCount.mutate({
                      obsId: obs.id,
                      count: Math.max(0, obs.effective_count - 1),
                    })
                  }
                >
                  <Minus className="h-3.5 w-3.5" />
                </Button>
                <span className="w-6 text-center tabular-nums">
                  {obs.effective_count}
                </span>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  disabled={busy}
                  title="Increase"
                  onClick={() =>
                    setCount.mutate({
                      obsId: obs.id,
                      count: obs.effective_count + 1,
                    })
                  }
                >
                  <Plus className="h-3.5 w-3.5" />
                </Button>
                {(isOverridden || isHumanOnly) && (
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-6 w-6 text-muted-foreground"
                    disabled={busy}
                    title={isHumanOnly ? "Remove species" : "Reset to AI count"}
                    onClick={() => removeObs.mutate(obs.id)}
                  >
                    {isHumanOnly ? (
                      <X className="h-3.5 w-3.5" />
                    ) : (
                      <RotateCcw className="h-3.5 w-3.5" />
                    )}
                  </Button>
                )}
              </span>
            </div>
          );
        })}

        {observations.length === 0 && (
          <div className="py-3 text-center text-xs text-muted-foreground">
            No species recorded
          </div>
        )}

        {/* Add a species the AI missed entirely. */}
        <div className="flex items-center gap-2 pt-1">
          <LabelPicker
            value={null}
            onSelect={(option) => addSpecies.mutate(option)}
            options={labelOptions}
            isLoading={labelOptionsLoading}
            projectId={projectId}
            triggerIcon={Plus}
            triggerTitle="Add a species the AI missed"
            hideDot
            hideLabel
          />
          <span className="text-xs text-muted-foreground">
            Add a species the AI missed
          </span>
        </div>
      </div>
    </div>
  );
}
