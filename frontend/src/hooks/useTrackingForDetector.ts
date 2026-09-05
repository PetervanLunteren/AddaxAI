/**
 * Choosing an underwater detector switches "Track animals across frames"
 * on and lifts the video frame rate to the tracker's default.
 *
 * One hook for the folder-run setup step and the project Settings page,
 * so the two forms react to a detector change the same way. It acts only
 * on a change the person makes in the form: the first detector it sees is
 * whatever the run, the project or the last-used settings seeded, and
 * someone who turned tracking off on such a detector keeps that choice
 * on the next visit. MegaDetector leaves both settings alone, in both
 * directions: switching back from SharkTrack does not turn tracking off,
 * because a camera trap user may have wanted it.
 */

import { useEffect, useRef } from "react";
import type { FieldValues, Path, PathValue, UseFormReturn } from "react-hook-form";

import type { ModelInfo } from "../api/types";

/** BoT-SORT's default sampling rate: SharkTrack's accuracy versus speed
 *  sweet spot (MOTA 0.77 at 3 fps). Lower rates track worse. */
export const TRACKING_DEFAULT_FPS = 3;

type TrackingFields = { video_tracking: boolean; video_fps: number };

export function useTrackingForDetector<T extends FieldValues & TrackingFields>(
  form: UseFormReturn<T>,
  detectionModelId: string | undefined,
  detectionModel: ModelInfo | undefined,
): void {
  const previousId = useRef<string | undefined>(undefined);
  useEffect(() => {
    const previous = previousId.current;
    previousId.current = detectionModelId;
    if (previous === undefined || previous === detectionModelId) return;
    if (!detectionModel?.tracking_recommended) return;
    form.setValue(
      "video_tracking" as Path<T>,
      true as PathValue<T, Path<T>>,
      { shouldDirty: true },
    );
    if (form.getValues("video_fps" as Path<T>) < TRACKING_DEFAULT_FPS) {
      form.setValue(
        "video_fps" as Path<T>,
        TRACKING_DEFAULT_FPS as PathValue<T, Path<T>>,
        { shouldDirty: true },
      );
    }
  }, [detectionModelId, detectionModel, form]);
}
