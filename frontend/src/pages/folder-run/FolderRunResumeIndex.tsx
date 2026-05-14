/**
 * Index route for `/folder-runs/:runId`.
 *
 * Looks up the persisted step on the run and redirects the user
 * there. Used when someone reopens a folder run by id (e.g. from the
 * recent work strip on the home screen). Falls back to "folder" when
 * the lookup is still in flight or the run has no step set.
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
  return <Navigate to={`/folder-runs/${runId}/${run.step}`} replace />;
}
