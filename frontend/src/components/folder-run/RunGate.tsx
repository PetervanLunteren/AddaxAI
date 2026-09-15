/**
 * The guard every post-setup step starts with: no run id means the user
 * reached the URL without finishing setup, so send them to a new run;
 * a run still loading shows the loading card; otherwise render the step
 * with the loaded run and its id, both narrowed to non-null.
 *
 * Used as a render prop so the step body can read `run.project` and pass
 * `runId` on without re-checking either.
 */

import type { ReactNode } from "react";
import { Navigate } from "react-router-dom";
import { Card, CardContent } from "../ui/card";
import type { FolderRunResponse } from "../../api/folder-runs";
import { useFolderRun } from "../../pages/folder-run/FolderRunLayout";

export function RunGate({
  children,
}: {
  children: (run: FolderRunResponse, runId: string) => ReactNode;
}) {
  const { runId, run, isLoading } = useFolderRun();

  if (!runId) {
    return <Navigate to="/folder-runs/new" replace />;
  }

  if (isLoading || !run) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-sm text-muted-foreground">
          Loading run...
        </CardContent>
      </Card>
    );
  }

  return <>{children(run, runId)}</>;
}
