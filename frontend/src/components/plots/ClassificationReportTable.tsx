/**
 * Classification report renderer.
 *
 * Per-class precision / recall / F1 / support, plus macro and weighted
 * averages, computed server-side from the same data as the confusion
 * matrix. The F1 column uses the diverging status palette; the other
 * metric columns stay plain.
 */

import { Info, Loader2 } from "lucide-react";

import { f1DivergingColor } from "../../lib/metric-colors";
import type { PerformanceResponse } from "../../api/performance";

interface ClassificationReportTableProps {
  data: PerformanceResponse | undefined;
  loading: boolean;
  onRowClick?: (args: {
    className: string;
    taxonomyId: string | null;
  }) => void;
}

function fmt(value: number | null): string {
  if (value === null) return "–";
  return value.toFixed(3);
}

export function ClassificationReportTable({
  data,
  loading,
  onRowClick,
}: ClassificationReportTableProps) {
  if (loading && !data) {
    return (
      <div className="flex items-center justify-center rounded-lg border bg-card p-16">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">
          Computing metrics...
        </span>
      </div>
    );
  }

  if (!data) return null;

  if (data.classes.length === 0 || data.grand_total === 0) {
    return (
      <div className="rounded-lg border bg-card p-8 text-center space-y-2">
        <div className="text-sm font-medium text-foreground">
          Nothing to score yet
        </div>
        <div className="text-sm text-muted-foreground max-w-xl mx-auto">
          Precision, recall, and F1 require verified detections. Verify some
          detections on the Verify page to populate this table.
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border bg-card">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b">
            <tr className="text-xs text-muted-foreground">
              <th className="text-left py-2 pl-4 pr-6 font-medium">Class</th>
              <th className="text-right py-2 px-6 font-medium">Support</th>
              <th className="text-right py-2 px-6 font-medium">Precision</th>
              <th className="text-right py-2 px-6 font-medium">Recall</th>
              <th className="text-right py-2 pl-6 pr-4 font-medium">F1</th>
            </tr>
          </thead>
          <tbody>
            {data.per_class.map((m, i) => {
              const f1Style = f1DivergingColor(m.f1);
              const isOther = m.class_name === "other";
              return (
                <tr
                  key={m.class_name}
                  onClick={() =>
                    onRowClick?.({
                      className: m.class_name,
                      taxonomyId: data.class_taxonomy_ids[i],
                    })
                  }
                  className="border-b hover:bg-accent/40"
                  style={{
                    cursor: onRowClick ? "pointer" : "default",
                    borderStyle: isOther ? "dashed" : undefined,
                  }}
                >
                  <td className="py-1.5 pl-4 pr-6 truncate max-w-[280px]">
                    {m.display_name}
                  </td>
                  <td className="py-1.5 px-6 text-right tabular-nums">
                    {m.support}
                  </td>
                  <td className="py-1.5 px-6 text-right tabular-nums">
                    {fmt(m.precision)}
                  </td>
                  <td className="py-1.5 px-6 text-right tabular-nums">
                    {fmt(m.recall)}
                  </td>
                  <td
                    className="py-1.5 pl-6 pr-4 text-right tabular-nums"
                    style={{
                      background: f1Style.background,
                      color: f1Style.color,
                    }}
                  >
                    {fmt(m.f1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot className="border-t-2">
            <SummaryRow
              label="Macro avg"
              precision={data.macro_precision}
              recall={data.macro_recall}
              f1={data.macro_f1}
            />
            <SummaryRow
              label="Weighted avg"
              precision={data.weighted_precision}
              recall={data.weighted_recall}
              f1={data.weighted_f1}
            />
          </tfoot>
        </table>
      </div>

      <ReportFooter data={data} />
    </div>
  );
}

function ReportFooter({ data }: { data: PerformanceResponse }) {
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

interface SummaryRowProps {
  label: string;
  precision: number | null;
  recall: number | null;
  f1: number | null;
}

function SummaryRow({ label, precision, recall, f1 }: SummaryRowProps) {
  const f1Style = f1DivergingColor(f1);
  return (
    <tr>
      <td className="py-1.5 pl-4 pr-6 font-medium whitespace-nowrap">{label}</td>
      <td />
      <td className="py-1.5 px-6 text-right tabular-nums font-medium">
        {fmt(precision)}
      </td>
      <td className="py-1.5 px-6 text-right tabular-nums font-medium">
        {fmt(recall)}
      </td>
      <td
        className="py-1.5 pl-6 pr-4 text-right tabular-nums font-medium"
        style={{ background: f1Style.background, color: f1Style.color }}
      >
        {fmt(f1)}
      </td>
    </tr>
  );
}
