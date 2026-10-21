/**
 * First-run setup wizard.
 *
 * Single screen. Until env-addaxai-base is installed, the rest of the app
 * is gated (see SetupGate in App.tsx). The wizard is re-entrant: closing
 * and reopening mid-install leaves the user back here with a Resume
 * button, since the backend's env_manager is idempotent.
 *
 * It also asks the one question that has to be answered before anything
 * is downloaded: camera trap or underwater footage. That decides the
 * detector and the environments installed here, and it is the same
 * device-global preference the model pickers filter on afterwards
 * (lib/data-type.ts). It is pre-answered with camera trap and changeable
 * later, so the majority press one button and the rest flip one control.
 * One screen rather than a multi-step wizard: two settings with safe
 * defaults do not need steps, and every extra screen is one more place
 * to abandon a first launch.
 *
 * This screen is also where a user lands who switches data type later
 * and does not have that type's models yet: readiness is per data type,
 * so the gate sends them here, the question is already on their new
 * answer, and one button installs what is missing.
 *
 * Visual: the same treatment as the home screen, a blurred forest photo
 * behind a scrim, so the first screen a new user meets looks like the
 * app rather than like a blank page. The card itself is deliberately
 * unchanged: everything inside it (progress bar, error callout, buttons)
 * is a shared component built for a light surface, so putting it on
 * glass would mean a second, darker version of each one. Only the
 * shadow is deepened, because `shadow-sm` disappears over a photo.
 */

import { useEffect, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { setupApi } from "../api/setup";
import {
  DATA_TYPE_OPTIONS,
  type DataType,
  setDataType,
  useDataType,
} from "../lib/data-type";
import { Button } from "../components/ui/button";
import { Progress } from "../components/ui/progress";
import { SegmentedControl } from "../components/ui/segmented-control";
import { Callout } from "../components/ui/callout";
import { LogoPlate } from "../components/layout/LogoPlate";
import { PhotoBackdrop } from "../components/layout/PhotoBackdrop";
import { ContinueWithoutRevocationChecks } from "../components/setup/ContinueWithoutRevocationChecks";

const POLL_INTERVAL_MS = 1500;

/**
 * Photo frame shared by both returns below. The loading state uses it too,
 * so a slow backend does not show a white page that flips to a photo once
 * the first poll lands.
 */
function SetupFrame({ children }: { children: ReactNode }) {
  return (
    <div className="relative min-h-screen overflow-hidden">
      <PhotoBackdrop />
      <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-4 py-8">
        {children}
      </div>
    </div>
  );
}

export default function SetupPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Stored as the user flips it, so the poll below, the install and
  // every model list afterwards read one value. The store is the state:
  // `setDataType` notifies its subscribers, so this screen, the photo
  // behind it and the gate around it all move together, without a
  // reload and without a second copy of the answer.
  const dataType = useDataType();

  const { data: status } = useQuery({
    queryKey: ["setup-status", dataType],
    queryFn: () => setupApi.getStatus(dataType),
    refetchInterval: POLL_INTERVAL_MS,
  });

  const install = useMutation({
    mutationFn: () => setupApi.installEnv(dataType),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["setup-status"] });
    },
  });

  // Once setup is ready, leave the wizard.
  useEffect(() => {
    if (status?.ready) {
      navigate("/", { replace: true });
    }
  }, [status?.ready, navigate]);

  if (!status) {
    return (
      <SetupFrame>
        <p className="text-sm text-white/85 drop-shadow">Checking setup state...</p>
      </SetupFrame>
    );
  }

  const inProgress = status.install_in_progress;
  // Two error sources. A started install that fails reports through the
  // polled status. A refusal before the install starts (e.g. the
  // disk-space pre-flight 507) is only on the mutation itself; the
  // status endpoint never learns about it.
  const statusError = !inProgress ? status.error : null;
  // The mutation error is suppressed while an install runs. Two buttons
  // start one, neither disables on click, and the status takes up to a
  // poll to catch up, so an impatient second click lands a 409 here.
  // Without this the page shouts "Setup failed" over a live progress bar
  // and a user who believes it quits a perfectly healthy install.
  const errorText =
    statusError ?? (inProgress ? null : install.error?.message) ?? null;
  const hasError = errorText !== null;
  // Tagged by the backend, never guessed from the message text. Tied to
  // statusError so a pre-flight refusal (which carries no kind) cannot
  // leave a stale offer on screen from an earlier attempt.
  const showRevocationOptOut =
    statusError !== null && status.error_kind === "tls_revocation";
  // The button must surface whenever something is still missing, not just
  // when env is missing. Otherwise a half-complete state (env installed but
  // bundled models missing, common in dev mode) leaves the user with no
  // affordance and the page looks frozen.
  const showStartButton = !inProgress && !status.ready && !hasError;
  // Any finished piece on disk (env or models) means a previous attempt got
  // partway. Closing the app mid-setup is safe: installs are atomic and
  // finished downloads are kept, so we resume rather than start over.
  const hasPartialProgress =
    (status.env_installed || status.models_installed) && !status.ready;

  return (
    <SetupFrame>
      <LogoPlate className="mb-6" logoClassName="h-28" />
      <div
        className="w-full max-w-xl rounded-lg border bg-card p-8
          shadow-[0_20px_50px_-20px_rgba(0,0,0,0.55)]"
      >
        <h1 className="text-2xl font-bold tracking-tight">Initial setup</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          The AI models and their environment need to be installed before
          AddaxAI can analyse your photos and videos. This is a one-time
          download and can take 10 to 30 minutes depending on your internet
          connection.
        </p>

        {/* The one question, and only while there is a button to press:
            a resumed or running install must not show a control that
            would contradict what is already on disk. */}
        {showStartButton && (
          <div className="mt-6 space-y-2">
            <h2 className="text-sm font-medium">What will you be working on?</h2>
            <SegmentedControl
              options={DATA_TYPE_OPTIONS.map((o) => ({
                value: o.value,
                title: o.label,
                label: o.label,
              }))}
              value={dataType}
              onChange={(v) => setDataType(v as DataType)}
            />
            <p className="text-xs text-muted-foreground">
              This decides which AI models are installed and offered. You
              can change it later.
            </p>
          </div>
        )}

        <div className="mt-6 space-y-2 text-sm">
          <Row label="Analysis environment" ok={status.env_installed} />
          <Row label="Default models" ok={status.models_installed} />
        </div>

        {showStartButton && (
          <div className="mt-6">
            <Button
              onClick={() => install.mutate()}
              disabled={install.isPending}
              className="w-full"
            >
              {install.isPending
                ? "Starting..."
                : hasPartialProgress
                  ? "Resume setup"
                  : "Start setup"}
            </Button>
            <p className="mt-2 text-xs text-muted-foreground">
              {hasPartialProgress
                ? "Some pieces are already installed. Resuming picks up "
                  + "where the last attempt stopped; finished downloads "
                  + "are kept."
                : "An internet connection is required for this one-time setup."}
            </p>
          </div>
        )}

        {inProgress && (
          <div className="mt-6 space-y-3">
            <Progress value={status.progress_pct} />
            <p
              className="truncate text-sm text-muted-foreground"
              title={status.message || "Installing..."}
            >
              {status.message || "Installing..."}
            </p>
            <p className="text-xs text-muted-foreground">
              {Math.round(status.progress_pct)}% complete. You can close the
              app and resume later; finished downloads are kept.
            </p>
          </div>
        )}

        {hasError && (
          <div className="mt-6 space-y-3">
            <Callout variant="error" title="Setup failed">
              <span className="text-xs break-words whitespace-pre-line">
                {errorText}
              </span>
            </Callout>
            <Button onClick={() => install.mutate()} className="w-full">
              Try again
            </Button>
            {/* Only for the one failure plain retrying cannot fix. The
                backend stops sending this kind once the choice has been
                made, so the offer never repeats uselessly. */}
            {showRevocationOptOut && (
              <ContinueWithoutRevocationChecks
                onRetry={() => install.mutate()}
              />
            )}
          </div>
        )}

        {status.ready && (
          <div className="mt-6 text-sm text-[#0f6064]">
            Setup complete. Opening AddaxAI...
          </div>
        )}
      </div>
    </SetupFrame>
  );
}

interface RowProps {
  label: string;
  ok: boolean;
}

function Row({ label, ok }: RowProps) {
  return (
    <div className="flex items-center justify-between">
      <span>{label}</span>
      <span className={ok ? "text-[#0f6064]" : "text-muted-foreground"}>
        {ok ? "Ready" : "Not ready"}
      </span>
    </div>
  );
}
