/**
 * Index route for `/folder-runs/:runId`.
 *
 * Looks up the persisted step on the run and redirects the user
 * there. Used when someone reopens a folder run by id (e.g. from the
 * recent work strip on the home screen). Renders nothing while the
 * lookup is in flight; the backend always returns a valid step.
 */

import { Navigate, useParams } from "react-router-dom";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunResumeIndex() {
  const { runId } = useParams<{ runId: string }>();
  const { run, isLoading } = useFolderRun();

  if (isLoading || !run) {
    // Render nothing while the run loads. The Outlet is gated on the
    // layout query; this just avoids a flicker through the wrong
    // step.
    return null;
  }
  // Guard against an unknown persisted step (the backend maps retired
  // slugs like counts / summary to labels, but an unexpected value
  // must never land the user on a dead route).
  const known = ["setup", "labels", "save"];
  const step = known.includes(run.step) ? run.step : "labels";
  return <Navigate to={`/folder-runs/${runId}/${step}`} replace />;
}
