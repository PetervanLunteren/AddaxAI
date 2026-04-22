/**
 * Confusion matrix renderer.
 *
 * Rows = current (verified) class. Columns = original machine
 * prediction. Counts are raw; each cell's background intensity is
 * normalised against the row max, so the dominant prediction for each
 * true class stands out. Clicking a non-zero cell raises `onCellClick`
 * with both class names and the row's taxonomy UUID (when available)
 * so the parent can deep-link to the Verify grid.
 */

import { useMemo } from "react";
import { Loader2 } from "lucide-react";

import { matrixCellColor } from "../../lib/metric-colors";
import type { PerformanceResponse } from "../../api/performance";

interface ConfusionMatrixProps {
  data: PerformanceResponse | undefined;
  loading: boolean;
  onCellClick?: (args: {
    rowClass: string;
    rowTaxonomyId: string | null;
    colClass: string;
  }) => void;
}

export function ConfusionMatrix({ data, loading, onCellClick }: ConfusionMatrixProps) {
  const rowMaxes = useMemo(() => {
    if (!data) return [] as number[];
    return data.matrix.map((row) => row.reduce((m, n) => (n > m ? n : m), 0));
  }, [data]);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center rounded-lg border bg-card p-16">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">Computing matrix...</span>
      </div>
    );
  }

  if (!data) return null;

  if (data.classes.length === 0) {
    return (
      <EmptyState
        title="No verified detections to compare"
        body={
          <>
            Verify some detections on the Verify page; once there are machine-
            labelled detections that a user has confirmed or relabelled, they
            will show up here.
          </>
        }
      />
    );
  }

  if (data.grand_total === 0 && data.skipped_no_prediction > 0) {
    return (
      <EmptyState
        title="No stored predictions to compare"
        body={
          <>
            This project was analysed before the prediction-history columns
            existed, so the raw machine label is not preserved alongside the
            current one. Re-run analysis on the deployments you want to see
            here; new detections will populate this matrix automatically.
          </>
        }
      />
    );
  }

  const classes = data.classes;
  const displays = data.class_display_names;

  return (
    <div className="rounded-lg border bg-card">
      <div className="overflow-x-auto">
        <table className="text-sm border-collapse">
          <thead>
            <tr>
              <th
                className="sticky left-0 top-0 z-20 bg-card p-2 text-xs text-muted-foreground"
                style={{ minWidth: 120 }}
              >
                true \ predicted
              </th>
              {classes.map((col, j) => (
                <th
                  key={col}
                  title={displays[j]}
                  className="sticky top-0 z-10 bg-card px-2 py-2 text-xs font-medium text-muted-foreground whitespace-nowrap"
                >
                  <div className="flex justify-center">
                    <span className="inline-block max-w-[140px] truncate rotate-[-30deg] origin-bottom-left">
                      {displays[j]}
                    </span>
                  </div>
                </th>
              ))}
              <th
                className="sticky top-0 z-10 bg-card px-2 py-2 text-xs font-medium text-muted-foreground"
              >
                Σ
              </th>
            </tr>
          </thead>
          <tbody>
            {classes.map((row, i) => {
              const rowMax = rowMaxes[i];
              const rowOther = row === "other";
              return (
                <tr key={row}>
                  <th
                    className="sticky left-0 z-10 bg-card px-2 py-1.5 text-left text-xs font-medium text-muted-foreground whitespace-nowrap"
                    title={displays[i]}
                    style={{
                      borderLeft: rowOther ? "1px dashed var(--color-border)" : undefined,
                    }}
                  >
                    <span className="inline-block max-w-[160px] truncate">
                      {displays[i]}
                    </span>
                  </th>
                  {classes.map((col, j) => {
                    const count = data.matrix[i][j];
                    const intensity = rowMax > 0 ? count / rowMax : 0;
                    const style = matrixCellColor(intensity);
                    const isDiag = i === j;
                    const colOther = col === "other";
                    return (
                      <td
                        key={col}
                        title={`${displays[i]} → ${displays[j]}: ${count}`}
                        onClick={() =>
                          count > 0 &&
                          onCellClick?.({
                            rowClass: row,
                            rowTaxonomyId: data.class_taxonomy_ids[i],
                            colClass: col,
                          })
                        }
                        className="text-center tabular-nums px-2 py-1.5"
                        style={{
                          background: style.background,
                          color: style.color,
                          cursor: count > 0 && onCellClick ? "pointer" : "default",
                          outline: isDiag ? "1px solid var(--color-primary)" : undefined,
                          outlineOffset: isDiag ? "-1px" : undefined,
                          borderLeft: colOther ? "1px dashed var(--color-border)" : undefined,
                          borderTop: rowOther ? "1px dashed var(--color-border)" : undefined,
                        }}
                      >
                        {count === 0 ? "·" : count}
                      </td>
                    );
                  })}
                  <td className="px-2 py-1.5 text-right tabular-nums text-muted-foreground">
                    {data.row_totals[i]}
                  </td>
                </tr>
              );
            })}
            <tr className="border-t">
              <th className="sticky left-0 z-10 bg-card px-2 py-1.5 text-left text-xs font-medium text-muted-foreground">
                Σ
              </th>
              {classes.map((col, j) => (
                <td
                  key={col}
                  className="px-2 py-1.5 text-center tabular-nums text-muted-foreground"
                >
                  {data.col_totals[j]}
                </td>
              ))}
              <td className="px-2 py-1.5 text-right tabular-nums font-medium text-foreground">
                {data.grand_total}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface EmptyStateProps {
  title: string;
  body: React.ReactNode;
}

function EmptyState({ title, body }: EmptyStateProps) {
  return (
    <div className="rounded-lg border bg-card p-8 text-center space-y-2">
      <div className="text-sm font-medium text-foreground">{title}</div>
      <div className="text-sm text-muted-foreground max-w-xl mx-auto">{body}</div>
    </div>
  );
}
