/**
 * Confusion matrix renderer.
 *
 * Rows = current (verified) class. Columns = original machine
 * prediction. Cells show either raw counts (per-row max intensity) or
 * row / column ratios depending on `mode`.
 */

import { useMemo } from "react";
import { Info, Loader2 } from "lucide-react";

import { matrixCellColor } from "../../lib/metric-colors";
import type { PerformanceResponse } from "../../api/performance";
import type { MatrixMode } from "./PerformanceFilterBar";

/** Fixed side length of every data cell and column header, in px. */
const CELL_SIZE = 44;

interface ConfusionMatrixProps {
  data: PerformanceResponse | undefined;
  loading: boolean;
  /**
   * "counts" renders raw integers with per-row intensity.
   * "recall" renders count / row_total (rows sum to 100%; diagonal = recall).
   * "precision" renders count / col_total (cols sum to 100%; diagonal = precision).
   * In the normalised modes, intensity equals the displayed value because
   * every cell is already on the same [0, 1] scale.
   */
  mode?: MatrixMode;
}

const PCT = new Intl.NumberFormat("en", {
  style: "percent",
  maximumFractionDigits: 1,
});

export function ConfusionMatrix({
  data,
  loading,
  mode = "counts",
}: ConfusionMatrixProps) {
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
            {/* Axis title row: "Predicted" spans the column-label region */}
            <tr>
              <th className="bg-card" aria-hidden="true" />
              <th className="bg-card" aria-hidden="true" />
              <th
                colSpan={classes.length}
                className="sticky top-0 z-10 bg-card pb-1 text-center text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
              >
                Predicted (AI)
              </th>
              <th className="bg-card" aria-hidden="true" />
              <th className="w-full bg-card" aria-hidden="true" />
            </tr>
            <tr>
              <th className="bg-card" aria-hidden="true" />
              <th
                className="sticky left-0 top-0 z-20 bg-card"
                style={{ minWidth: 120 }}
                aria-hidden="true"
              />
              {classes.map((col, j) => (
                <th
                  key={col}
                  title={displays[j]}
                  className="sticky top-0 z-10 bg-card px-0 pt-2 pb-1 text-xs font-medium text-muted-foreground align-bottom"
                  style={{ width: CELL_SIZE, minWidth: CELL_SIZE }}
                >
                  <div className="flex items-end justify-center">
                    <div
                      className="whitespace-nowrap overflow-hidden text-ellipsis"
                      style={{
                        writingMode: "vertical-rl",
                        transform: "rotate(180deg)",
                        maxHeight: 130,
                      }}
                    >
                      {displays[j]}
                    </div>
                  </div>
                </th>
              ))}
              <th
                className="sticky top-0 z-10 bg-card pl-3 pr-2 py-2 text-xs font-medium text-muted-foreground align-bottom"
              >
                Σ
              </th>
              {/* spacer absorbs leftover width so Σ sits next to the matrix */}
              <th className="w-full" aria-hidden="true" />
            </tr>
          </thead>
          <tbody>
            {classes.map((row, i) => {
              const rowMax = rowMaxes[i];
              const rowOther = row === "other";
              return (
                <tr key={row}>
                  {i === 0 && (
                    <th
                      rowSpan={classes.length + 1}
                      className="align-middle px-1 text-muted-foreground"
                    >
                      <div
                        className="whitespace-nowrap text-center text-[11px] font-semibold uppercase tracking-wider"
                        style={{
                          writingMode: "vertical-rl",
                          transform: "rotate(180deg)",
                        }}
                      >
                        True (verified)
                      </div>
                    </th>
                  )}
                  <th
                    className="sticky left-0 z-10 bg-card pl-2 pr-3 py-1.5 text-right text-xs font-medium text-muted-foreground"
                    title={displays[i]}
                    style={{
                      borderLeft: rowOther ? "1px dashed var(--color-border)" : undefined,
                      maxWidth: 160,
                    }}
                  >
                    <span
                      className="inline-block whitespace-nowrap overflow-hidden text-ellipsis align-middle"
                      style={{ maxWidth: 150 }}
                    >
                      {displays[i]}
                    </span>
                  </th>
                  {classes.map((col, j) => {
                    const count = data.matrix[i][j];
                    const rowTotal = data.row_totals[i];
                    const colTotal = data.col_totals[j];
                    // In counts mode intensity is per-row max. In the
                    // normalised modes the cell value itself is in
                    // [0, 1] and IS the intensity — no separate
                    // denominator needed.
                    let ratio = 0;
                    if (mode === "recall" && rowTotal > 0) {
                      ratio = count / rowTotal;
                    } else if (mode === "precision" && colTotal > 0) {
                      ratio = count / colTotal;
                    }
                    const intensity =
                      mode === "counts"
                        ? rowMax > 0
                          ? count / rowMax
                          : 0
                        : ratio;
                    const style = matrixCellColor(intensity);
                    const isDiag = i === j;
                    const colOther = col === "other";
                    const displayValue =
                      count === 0
                        ? "·"
                        : mode === "counts"
                          ? count
                          : PCT.format(ratio);
                    const tooltipValue =
                      mode === "counts"
                        ? count
                        : `${count} (${PCT.format(ratio)} ${
                            mode === "recall" ? "of row" : "of column"
                          })`;
                    return (
                      <td
                        key={col}
                        title={`${displays[i]} → ${displays[j]}: ${tooltipValue}`}
                        className="text-center tabular-nums p-0"
                        style={{
                          width: CELL_SIZE,
                          minWidth: CELL_SIZE,
                          maxWidth: CELL_SIZE,
                          height: CELL_SIZE,
                          background: style.background,
                          color: style.color,
                          outline: isDiag ? "1px solid var(--color-primary)" : undefined,
                          outlineOffset: isDiag ? "-1px" : undefined,
                          borderLeft: colOther ? "1px dashed var(--color-border)" : undefined,
                          borderTop: rowOther ? "1px dashed var(--color-border)" : undefined,
                        }}
                      >
                        {displayValue}
                      </td>
                    );
                  })}
                  <td className="pl-3 pr-2 py-1.5 text-right tabular-nums text-muted-foreground">
                    {data.row_totals[i]}
                  </td>
                  <td className="w-full" aria-hidden="true" />
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
                  className="p-0 text-center tabular-nums text-muted-foreground"
                  style={{ width: CELL_SIZE, minWidth: CELL_SIZE, height: CELL_SIZE }}
                >
                  {data.col_totals[j]}
                </td>
              ))}
              <td className="pl-3 pr-2 py-1.5 text-right tabular-nums font-medium text-foreground">
                {data.grand_total}
              </td>
              <td className="w-full" aria-hidden="true" />
            </tr>
          </tbody>
        </table>
      </div>

      <MatrixFooter data={data} />
    </div>
  );
}

function MatrixFooter({ data }: { data: PerformanceResponse }) {
  const totalInScope =
    data.grand_total + data.skipped_unverified + data.skipped_no_prediction;
  if (totalInScope === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t px-4 py-3 text-xs text-muted-foreground">
      <Info className="h-3.5 w-3.5" />
      <span>
        Based on {data.grand_total.toLocaleString()} verified detection
        {data.grand_total === 1 ? "" : "s"} of {totalInScope.toLocaleString()}{" "}
        in the filtered range
      </span>
      {data.skipped_unverified > 0 && (
        <>
          <span aria-hidden="true">·</span>
          <span>
            {data.skipped_unverified.toLocaleString()} not yet verified
          </span>
        </>
      )}
      {data.skipped_no_prediction > 0 && (
        <>
          <span aria-hidden="true">·</span>
          <span>
            {data.skipped_no_prediction.toLocaleString()} excluded (no stored
            prediction; re-run analysis to include them)
          </span>
        </>
      )}
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
