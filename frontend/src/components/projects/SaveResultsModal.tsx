/**
 * Modal showing before/after statistics after a settings change, named
 * for the two things users verify: Labels (detections per species) and
 * Counts (individuals per species). Each card lists the per-species
 * changes with a trailing "N other labels unchanged" line, or "No
 * change." when that card was untouched. Which cards show is set by the
 * `metrics` prop (folder-run refine shows Labels only).
 */

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Card, CardContent } from "../ui/card";
import {
  ALL_METRICS,
  METRIC_FIELD,
  METRIC_META,
  type SaveMetric,
} from "../../lib/saveMetrics";

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
  /** Per-label detection counts (the "Labels" card). */
  observations: { before: StatSnapshot; after: StatSnapshot };
  /** Per-species effective_count, human count where set (the "Counts" card). */
  independent_observations: { before: StatSnapshot; after: StatSnapshot };
}

/** Verified labels / confirmed counts that reprocessing leaves untouched.
 * Shown as footer lines so a small (or zero) diff still tells its story. */
export interface Protection {
  verifiedLabels: number;
  totalLabels: number;
  confirmedCounts: number;
  totalCounts: number;
}

interface SaveResultsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  results: SaveResults;
  /** Subset + order of cards to show. Defaults to all three. */
  metrics?: SaveMetric[];
  /** Verified/confirmed share, rendered as footer lines. */
  protection?: Protection | null;
}

// --- Helpers ---

function normalizeLabel(s: string): string {
  return s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
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
  subtitle,
  before,
  after,
}: {
  title: string;
  subtitle: string;
  before: StatSnapshot;
  after: StatSnapshot;
}) {
  const diff = computeLabelDiff(before.labels, after.labels);
  const totalLabels = new Set([
    ...before.labels.map((l) => l.label),
    ...after.labels.map((l) => l.label),
  ]).size;
  const unchangedCount = totalLabels - diff.length;

  return (
    <Card>
      <CardContent className="pt-4 pb-4">
        <p className="text-sm font-medium">{title}</p>
        <p className="text-xs text-muted-foreground">{subtitle}</p>

        {diff.length === 0 ? (
          <p className="text-xs text-muted-foreground italic mt-2">No change.</p>
        ) : (
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
            {unchangedCount > 0 && (
              <p className="text-xs text-muted-foreground italic pt-1">
                {unchangedCount} other label
                {unchangedCount === 1 ? "" : "s"} unchanged
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// --- Component ---

function pct(part: number, total: number): number {
  return total > 0 ? Math.round((part / total) * 100) : 0;
}

export function SaveResultsModal({
  open,
  onOpenChange,
  results,
  metrics = ALL_METRICS,
  protection,
}: SaveResultsModalProps) {
  const showLabelsProtection =
    !!protection && metrics.includes("labels") && protection.totalLabels > 0;
  const showCountsProtection =
    !!protection && metrics.includes("counts") && protection.totalCounts > 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Effect on statistics</DialogTitle>
        </DialogHeader>

        <div className="space-y-3 overflow-y-auto min-h-0">
          {metrics.map((metric) => {
            const { before, after } = results[METRIC_FIELD[metric]];
            return (
              <StatCard
                key={metric}
                title={METRIC_META[metric].title}
                subtitle={METRIC_META[metric].subtitle}
                before={before}
                after={after}
              />
            );
          })}
        </div>

        {(showLabelsProtection || showCountsProtection) && protection && (
          <div className="border-t pt-3 mt-1 space-y-0.5 text-xs text-muted-foreground">
            {showLabelsProtection && (
              <p>
                {pct(protection.verifiedLabels, protection.totalLabels)}% of
                labels are verified and were left unchanged.
              </p>
            )}
            {showCountsProtection && (
              <p>
                {pct(protection.confirmedCounts, protection.totalCounts)}% of
                counts are confirmed and were left unchanged.
              </p>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
