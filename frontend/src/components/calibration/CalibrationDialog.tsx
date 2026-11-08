/**
 * "Is the AI label right?" One crop at a time from a batch the backend
 * draws across the label's score range. Y, N and U answer, ← goes back
 * one (answering again replaces the answer). Can't tell is stored but
 * left out of the fit. Answers never touch verification.
 *
 * The keys follow the Labels page: U is "unknown" there too, and ← / →
 * step through files there. Yes is deliberately not Enter, so that
 * confirming takes a real keypress and a held key cannot wave photos
 * through unseen. On the screen after a batch, Enter is the main
 * button.
 *
 * The AI score is deliberately not shown: seeing 0.98 next to a crop
 * nudges the answer, and the fit needs the person's own judgement.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, CheckCircle2, Loader2 } from "lucide-react";

import {
  calibrationApi,
  type CalibrationAnswerValue,
  type CalibrationLabel,
} from "../../api/calibration";
import { API_BASE_URL, ApiError } from "../../lib/api-client";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import {
  getContrastTextColor,
  getSpeciesColor,
  useSpeciesColorsVersion,
} from "../../utils/species-colors";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Kbd } from "../ui/kbd";
import { Progress } from "../ui/progress";

const KEYS: Record<string, CalibrationAnswerValue> = {
  y: "right",
  n: "wrong",
  u: "unsure",
};

const cropUrl = (detectionId: string) =>
  `${API_BASE_URL}/api/detections/${detectionId}/crop?size=800&aspect=1.333`;

function tally(row: CalibrationLabel): string {
  const parts = [`${row.right} right`, `${row.wrong} wrong`, `${row.unsure} can't tell`];
  return parts.join(" · ");
}

function stillNeeded(row: CalibrationLabel): string | null {
  if (row.right_needed === 0 && row.wrong_needed === 0) return null;
  const parts = [
    row.right_needed > 0 ? `${row.right_needed} more right` : null,
    row.wrong_needed > 0 ? `${row.wrong_needed} more wrong` : null,
  ].filter(Boolean);
  return `${parts.join(" and ")} for a curve`;
}

interface CalibrationDialogProps {
  projectId: string;
  row: CalibrationLabel;
  onClose: () => void;
  /** Close and open this label's curve on the page. */
  onShowResult: () => void;
}

export function CalibrationDialog({
  projectId,
  row,
  onClose,
  onShowResult,
}: CalibrationDialogProps) {
  const queryClient = useQueryClient();
  useSpeciesColorsVersion();
  // Bumped for every new batch so each one is a fresh draw.
  const [round, setRound] = useState(0);
  const [index, setIndex] = useState(0);

  const { data: batch, isLoading } = useQuery({
    queryKey: ["calibration", "sample", projectId, row.label, round],
    queryFn: () => calibrationApi.sample(projectId, row.label),
    staleTime: Infinity,
    gcTime: 0,
  });

  const answer = useMutation({
    mutationFn: (value: CalibrationAnswerValue) =>
      calibrationApi.answer(batch![index].detection_id, row.label, value),
    onSuccess: () => {
      setIndex((i) => i + 1);
      queryClient.invalidateQueries({ queryKey: ["calibration", "overview", projectId] });
    },
    onError: (error) => {
      // The detection left the sample after the batch was drawn (deleted,
      // reprocessed). Nothing to answer, so move on instead of retrying.
      if (error instanceof ApiError && error.status === 404) setIndex((i) => i + 1);
    },
  });

  const current = batch?.[index];
  const next = batch?.[index + 1];
  const batchDone = !isLoading && !current;
  const poolExhausted = batch !== undefined && batch.length === 0;
  // `row` is refetched after every answer, so this flips the moment the
  // minimum is reached.
  const curveReady = row.thresholds !== null;

  // Load the next crop while the person looks at this one.
  useEffect(() => {
    if (next) new Image().src = cropUrl(next.detection_id);
  }, [next]);

  const submit = useCallback(
    (value: CalibrationAnswerValue) => {
      if (current && !answer.isPending) answer.mutate(value);
    },
    [current, answer],
  );
  const back = useCallback(() => {
    if (!answer.isPending) setIndex((i) => Math.max(0, i - 1));
  }, [answer.isPending]);
  const nextBatch = useCallback(() => {
    setRound((r) => r + 1);
    setIndex(0);
  }, []);

  // The main button on the screen after a batch, also behind Enter:
  // show the curve once there is one, otherwise draw the next batch.
  const primary = useMemo(() => {
    if (!batchDone) return null;
    if (curveReady) return { label: "Show result", run: onShowResult };
    if (poolExhausted) return { label: "Close", run: onClose };
    return { label: "Next batch", run: nextBatch };
  }, [batchDone, curveReady, poolExhausted, onShowResult, onClose, nextBatch]);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // A held key repeats faster than the answer round-trips, which
      // would answer the next photo unseen.
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      if (e.key === "Enter" && primary) {
        e.preventDefault();
        primary.run();
        return;
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        back();
        return;
      }
      const value = KEYS[e.key.toLowerCase()];
      if (!value) return;
      e.preventDefault();
      submit(value);
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [submit, back, primary]);

  const name = resolveSpeciesName(row);
  const pillColor = getSpeciesColor(row.label);
  const needed = stillNeeded(row);

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Is the AI label right?</DialogTitle>
          <DialogDescription>
            To find the score from which this label can be trusted, the app needs right
            and wrong examples across all scores. Judge each photo on its own. Nothing
            here verifies or relabels a detection.
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">AI label</span>
          <span
            className="rounded-full px-3 py-1 text-sm font-medium"
            style={{ background: pillColor, color: getContrastTextColor(pillColor) }}
          >
            {name}
          </span>
        </div>

        {isLoading ? (
          <div className="flex aspect-[4/3] items-center justify-center rounded-lg bg-muted">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : current ? (
          <div className="space-y-3">
            <div className="aspect-[4/3] overflow-hidden rounded-lg bg-black">
              <img
                key={current.detection_id}
                src={cropUrl(current.detection_id)}
                alt={`Detection the AI labelled ${name}`}
                className="h-full w-full object-contain"
              />
            </div>
            <div className="flex items-center gap-3">
              <Progress value={(index / batch!.length) * 100} className="h-1.5 flex-1" />
              <span className="text-xs tabular-nums text-muted-foreground">
                {index + 1} / {batch!.length}
              </span>
            </div>
          </div>
        ) : (
          <div className="flex aspect-[4/3] flex-col items-center justify-center gap-2 rounded-lg bg-muted text-center">
            {poolExhausted ? (
              <p className="font-medium">Every {name} detection has an answer</p>
            ) : (
              <>
                <CheckCircle2 className="h-8 w-8 text-good-ink" />
                <p className="font-medium">
                  {curveReady ? "Enough answers for a curve" : "Batch done"}
                </p>
              </>
            )}
            <p className="text-sm text-muted-foreground">{tally(row)}</p>
            {needed && <p className="text-sm text-muted-foreground">{needed}</p>}
            {curveReady && !poolExhausted && (
              <p className="text-sm text-muted-foreground">
                More answers make the band around the curve narrower.
              </p>
            )}
          </div>
        )}

        {answer.isError && !(answer.error instanceof ApiError && answer.error.status === 404) && (
          <p className="text-sm font-medium text-destructive-ink">
            {(answer.error as Error).message}
          </p>
        )}

        {current ? (
          <>
            <div className="grid grid-cols-3 gap-2">
              <Button size="lg" onClick={() => submit("right")} disabled={answer.isPending}>
                Yes<Kbd onPrimary>Y</Kbd>
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={() => submit("wrong")}
                disabled={answer.isPending}
              >
                No<Kbd>N</Kbd>
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={() => submit("unsure")}
                disabled={answer.isPending}
              >
                Can't tell<Kbd>U</Kbd>
              </Button>
            </div>
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <Button
                variant="ghost"
                size="sm"
                onClick={back}
                disabled={index === 0 || answer.isPending}
              >
                <ArrowLeft className="mr-1 h-3.5 w-3.5" />
                Back<Kbd>←</Kbd>
              </Button>
              <span>
                {tally(row)}
                {needed && ` · ${needed}`}
              </span>
            </div>
          </>
        ) : (
          primary && (
            <DialogFooter>
              {primary.label !== "Close" && (
                <Button variant="outline" onClick={onClose}>
                  Close
                </Button>
              )}
              {curveReady && !poolExhausted && (
                <Button variant="outline" onClick={nextBatch}>
                  Next batch
                </Button>
              )}
              <Button onClick={primary.run}>
                {primary.label}
                <Kbd onPrimary>⏎</Kbd>
              </Button>
            </DialogFooter>
          )
        )}
      </DialogContent>
    </Dialog>
  );
}
