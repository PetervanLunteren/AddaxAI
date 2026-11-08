/**
 * Calibration insight view.
 *
 * One row per AI label: how many detections the sample can draw from,
 * the answers so far, and the score from which 90, 95 and 99% of the
 * labels are right. Calibrate opens the "Is this a <label>?" dialog;
 * expanding a row shows the curve with its 95% band and the same
 * result in words, so the graph is not the only way to read it.
 */

import { Fragment, useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Loader2 } from "lucide-react";

import { calibrationApi, type CalibrationLabel } from "../api/calibration";
import { CalibrationCurve } from "../components/calibration/CalibrationCurve";
import { CalibrationDialog } from "../components/calibration/CalibrationDialog";
import { PlotExplainer } from "../components/plots/PlotExplainer";
import { Button } from "../components/ui/button";
import { curveSummary } from "../lib/calibration-summary";
import { resolveSpeciesName } from "../lib/species-name-mode";

const DASH = <span className="text-muted-foreground">–</span>;

function hasAnswers(row: CalibrationLabel): boolean {
  return row.right + row.wrong + row.unsure > 0;
}

/** The progress line under a label's name. Only for labels someone has started on. */
function status(row: CalibrationLabel): string | null {
  if (row.thresholds || !hasAnswers(row)) return null;
  if (row.no_error_bound !== null) {
    return `No errors in ${row.right} answers: at least ${Math.floor(row.no_error_bound * 100)}% right`;
  }
  const needed = [
    row.right_needed > 0 ? `${row.right_needed} more right` : null,
    row.wrong_needed > 0 ? `${row.wrong_needed} more wrong` : null,
  ].filter(Boolean);
  return `${needed.join(" and ")} needed for a curve`;
}

export function CalibrationPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [expanded, setExpanded] = useState<string | null>(null);
  const [calibrating, setCalibrating] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["calibration", "overview", projectId],
    queryFn: () => calibrationApi.get(projectId!),
    enabled: !!projectId,
  });

  if (!projectId) return null;
  const calibratingRow = data?.labels.find((r) => r.label === calibrating);

  return (
    <div className="min-h-screen">
      <header className="border-b bg-card/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Calibration</h1>
              <p className="text-sm text-muted-foreground">
                From which score the AI labels can be trusted, per label
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        {isLoading ? (
          <div className="flex items-center justify-center rounded-lg border bg-card p-16">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : !data || data.labels.length === 0 ? (
          <div className="rounded-lg border bg-card p-8 text-center text-sm text-muted-foreground">
            No AI labels to calibrate yet. Run a classification model on this project first.
          </div>
        ) : (
          <div className="overflow-x-auto rounded-lg border bg-card">
            <table className="w-full text-sm">
              <thead className="border-b">
                {/* Group labels carry a rule that spans exactly their
                    columns; the cell padding leaves a gap between the
                    two rules, so each group reads as one unit. */}
                <tr className="text-xs text-muted-foreground">
                  <th rowSpan={2} className="py-2 pl-4 pr-2 text-left align-bottom font-normal">
                    AI label
                  </th>
                  <th rowSpan={2} className="px-3 py-2 text-right align-bottom font-normal">
                    Detections
                  </th>
                  <th colSpan={3} className="px-3 pt-2 font-medium">
                    <span className="block border-b pb-1 text-center">Answers</span>
                  </th>
                  <th colSpan={3} className="px-3 pt-2 font-medium">
                    <span className="block border-b pb-1 text-center">
                      Trust labels from score
                    </span>
                  </th>
                  <th rowSpan={2} className="py-2 pl-3 pr-4" />
                </tr>
                <tr className="text-xs text-muted-foreground">
                  <th className="px-3 pb-2 text-right font-normal">Right</th>
                  <th className="px-3 pb-2 text-right font-normal">Wrong</th>
                  <th className="px-3 pb-2 text-right font-normal">Can't tell</th>
                  <th className="px-3 pb-2 text-right font-normal">90% right</th>
                  <th className="px-3 pb-2 text-right font-normal">95% right</th>
                  <th className="px-3 pb-2 text-right font-normal">99% right</th>
                </tr>
              </thead>
              <tbody>
                {data.labels.map((row) => {
                  const open = expanded === row.label;
                  const started = hasAnswers(row);
                  const note = status(row);
                  const name = resolveSpeciesName(row);
                  return (
                    <Fragment key={row.label}>
                      <tr className="border-b">
                        <td className="py-1.5 pl-4 pr-2">
                          <button
                            type="button"
                            className="flex items-center gap-1 text-left disabled:cursor-default"
                            onClick={() => setExpanded(open ? null : row.label)}
                            disabled={!row.curve}
                            aria-expanded={open}
                          >
                            {row.curve ? (
                              open ? (
                                <ChevronDown className="h-4 w-4" />
                              ) : (
                                <ChevronRight className="h-4 w-4" />
                              )
                            ) : (
                              <span className="w-4" />
                            )}
                            <span>{name}</span>
                          </button>
                          {note && (
                            <div className="pl-5 text-xs text-muted-foreground">{note}</div>
                          )}
                        </td>
                        <td className="px-3 py-1.5 text-right tabular-nums">
                          {row.detections.toLocaleString()}
                        </td>
                        {[row.right, row.wrong, row.unsure].map((n, i) => (
                          <td key={i} className="px-3 py-1.5 text-right tabular-nums">
                            {started ? n : DASH}
                          </td>
                        ))}
                        {[0, 1, 2].map((i) => {
                          const t = row.thresholds?.[i];
                          return (
                            <td key={i} className="px-3 py-1.5 text-right tabular-nums">
                              {!t ? DASH : t.score === null ? "not reached" : t.score.toFixed(2)}
                            </td>
                          );
                        })}
                        <td className="py-1.5 pl-3 pr-4 text-right">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setCalibrating(row.label)}
                            disabled={row.detections === 0}
                          >
                            {started ? "Continue" : "Calibrate"}
                          </Button>
                        </td>
                      </tr>
                      {open && row.curve && (
                        <tr className="border-b">
                          <td colSpan={9} className="px-4 py-3">
                            <CalibrationCurve row={row} />
                            <p className="mt-2 text-sm">{curveSummary(row, name)}</p>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
            {data.left_out > 0 && (
              <p className="border-t px-4 py-2 text-xs text-muted-foreground">
                {data.left_out} {data.left_out === 1 ? "label" : "labels"} with fewer than{" "}
                {data.min_detections} detections {data.left_out === 1 ? "is" : "are"} left
                out.
              </p>
            )}
          </div>
        )}

        <PlotExplainer
          plotKey="calibration"
          what={
            <>
              <p>
                One row per label the AI gave: how many detections carry it, the
                answers so far, and three scores. From the 90% score up, at least 90
                of 100 labels are right; the same for 95 and 99. "Not reached" means
                the curve does not get there within the scores you answered, so no
                score separates that label this reliably.
              </p>
              <p>
                Opening a row shows the curve: AI score from left to right, share of
                right labels from bottom to top. The shaded band is the 95%
                uncertainty and narrows with more answers. The text under it says the
                same in words. Counts and exports do not use any of this.
              </p>
            </>
          }
          how={
            <>
              <p>
                Calibrate asks "Is this a ...?" about detections the app picks evenly
                across the label's score range, so the sample is not tilted towards
                mistakes the way verified detections are. A curve is fitted once there
                are at least 10 right and 10 wrong answers, and is only drawn over the
                scores that were answered. Can't tell answers are stored but left out
                of the fit. Answers never verify or relabel a detection.
              </p>
              <p>
                The scores hold for this project with this model. Another survey,
                camera setup or model needs its own calibration. Labels with fewer than{" "}
                {data?.min_detections ?? 100} detections are left out: by the time a
                curve shows, nearly every one of them has been answered.
              </p>
            </>
          }
        />
      </main>

      {calibratingRow && (
        <CalibrationDialog
          projectId={projectId}
          row={calibratingRow}
          onClose={() => setCalibrating(null)}
          onShowResult={() => {
            setExpanded(calibratingRow.label);
            setCalibrating(null);
          }}
        />
      )}
    </div>
  );
}

export default CalibrationPage;
