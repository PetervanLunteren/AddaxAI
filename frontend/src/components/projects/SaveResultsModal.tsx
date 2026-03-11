/**
 * Modal showing before/after statistics after a settings change.
 *
 * Two cards: Observations (detection counts) and Independent events.
 * Each card shows total before → after with percentage change,
 * plus a collapsible per-label breakdown.
 */

import { useEffect, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Card, CardContent } from "../ui/card";

// --- Types ---

export interface LabelCount {
  label: string;
  count: number;
}

export interface StatSnapshot {
  total: number;
  labels: LabelCount[];
}

export interface SaveResults {
  observations: { before: StatSnapshot; after: StatSnapshot };
  events: { before: StatSnapshot; after: StatSnapshot };
}

interface SaveResultsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  results: SaveResults;
}

// --- Helpers ---

function normalizeLabel(s: string): string {
  return s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="bg-muted px-1.5 py-0.5 rounded text-xs">{children}</code>
  );
}

function formatDelta(before: number, after: number): string {
  const delta = after - before;
  const sign = delta >= 0 ? "+" : "";
  return `${sign}${delta}`;
}

function SmallCode({ children }: { children: React.ReactNode }) {
  return (
    <code className="bg-muted px-1 py-0.5 rounded">{children}</code>
  );
}

/** Compute per-label diffs from two snapshots, sorted by absolute change. */
function computeLabelDiff(
  before: LabelCount[],
  after: LabelCount[],
): { label: string; before: number; after: number }[] {
  const beforeMap = new Map(before.map((s) => [s.label, s.count]));
  const afterMap = new Map(after.map((s) => [s.label, s.count]));
  const allLabels = new Set([...beforeMap.keys(), ...afterMap.keys()]);

  const diff: { label: string; before: number; after: number }[] = [];
  for (const lbl of allLabels) {
    const b = beforeMap.get(lbl) ?? 0;
    const a = afterMap.get(lbl) ?? 0;
    if (b !== a) diff.push({ label: lbl, before: b, after: a });
  }
  return diff.sort((a, b) => Math.abs(b.after - b.before) - Math.abs(a.after - a.before));
}

// --- Stat card ---

function StatCard({
  title,
  before,
  after,
  expanded,
  onToggle,
}: {
  title: string;
  before: StatSnapshot;
  after: StatSnapshot;
  expanded: boolean;
  onToggle: () => void;
}) {
  const diff = computeLabelDiff(before.labels, after.labels);

  return (
    <Card>
      <CardContent className="pt-4 pb-4">
        <p className="text-sm font-medium">{title}</p>

        <p className="text-sm text-muted-foreground mt-1">
          <Code>{before.total.toLocaleString()}</Code>
          {" \u2192 "}
          <Code>{after.total.toLocaleString()}</Code>
          {" "}<Code>({formatDelta(before.total, after.total)})</Code>
        </p>

        {diff.length > 0 && (
          <>
            <button
              type="button"
              onClick={onToggle}
              className="text-xs text-muted-foreground hover:underline flex items-center gap-1 mt-1"
            >
              {expanded ? (
                <ChevronUp className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
              {expanded ? "Hide" : "Show"} breakdown ({diff.length} labels
              changed)
            </button>

            {expanded && (
              <div className="mt-2 space-y-1 max-h-48 overflow-y-auto">
                {diff.map(({ label, before: b, after: a }) => (
                  <div
                    key={label}
                    className="flex justify-between items-center text-xs text-muted-foreground"
                  >
                    <span>{normalizeLabel(label)}</span>
                    <span className="tabular-nums">
                      <SmallCode>{b}</SmallCode>
                      {" \u2192 "}
                      <SmallCode>{a}</SmallCode>
                      {" "}<SmallCode>({formatDelta(b, a)})</SmallCode>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

// --- Component ---

export function SaveResultsModal({
  open,
  onOpenChange,
  results,
}: SaveResultsModalProps) {
  const [obsExpanded, setObsExpanded] = useState(false);
  const [eventsExpanded, setEventsExpanded] = useState(false);

  // Reset collapse state each time modal opens
  useEffect(() => {
    if (open) {
      setObsExpanded(false);
      setEventsExpanded(false);
    }
  }, [open]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Effect on statistics</DialogTitle>
        </DialogHeader>

        <div className="space-y-3 overflow-y-auto min-h-0">
          <StatCard
            title="Observations"
            before={results.observations.before}
            after={results.observations.after}
            expanded={obsExpanded}
            onToggle={() => setObsExpanded(!obsExpanded)}
          />
          <StatCard
            title="Independent events"
            before={results.events.before}
            after={results.events.after}
            expanded={eventsExpanded}
            onToggle={() => setEventsExpanded(!eventsExpanded)}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
