/**
 * First-run setup wizard.
 *
 * Single screen. Until env-addaxai-base is installed, the rest of the app
 * is gated (see SetupGate in App.tsx). The wizard is re-entrant: closing
 * and reopening mid-install leaves the user back here with a Resume
 * button, since the backend's env_manager is idempotent.
 */

import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { setupApi } from "../api/setup";
import { Button } from "../components/ui/button";
import { Progress } from "../components/ui/progress";
import { Callout } from "../components/ui/callout";
import { ContinueWithoutRevocationChecks } from "../components/setup/ContinueWithoutRevocationChecks";

const POLL_INTERVAL_MS = 1500;

export default function SetupPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: status } = useQuery({
    queryKey: ["setup-status"],
    queryFn: setupApi.getStatus,
    refetchInterval: POLL_INTERVAL_MS,
  });

  const install = useMutation({
    mutationFn: setupApi.installEnv,
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
      <div className="min-h-screen flex items-center justify-center text-sm text-muted-foreground">
        Checking setup state...
      </div>
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
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-8">
      <img
        src="/branding/logo-wordmark.png"
        alt="AddaxAI"
        className="mb-6 h-28 w-auto"
      />
      <div className="w-full max-w-xl rounded-lg border bg-card-background p-8 shadow-sm">
        <h1 className="text-2xl font-bold tracking-tight">Initial setup</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          The AI models and their environment need to be installed before
          AddaxAI can analyse your photos and videos. This is a one-time
          download and can take 10 to 30 minutes depending on your internet
          connection.
        </p>

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
            <p className="mt-2 text-center text-xs text-muted-foreground">
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
            <p className="text-center text-xs text-muted-foreground">
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
    </div>
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
