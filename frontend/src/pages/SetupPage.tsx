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
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";

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
  const hasError = !!status.error && !inProgress;
  // The button must surface whenever something is still missing, not just
  // when env is missing. Otherwise a half-complete state (env installed but
  // bundled models missing, common in dev mode) leaves the user with no
  // affordance and the page looks frozen.
  const showStartButton = !inProgress && !status.ready && !hasError;
  const isRetry = status.env_installed && !status.models_installed;

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
          AddaxAI needs to install the analysis environment before you can
          process images. This is a one-time download of about 1.9 GB and
          can take 10 to 30 minutes depending on your internet connection.
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
                : isRetry
                  ? "Try again"
                  : "Start setup"}
            </Button>
            <p className="mt-2 text-center text-xs text-muted-foreground">
              {isRetry
                ? "Default models are still missing. Click try again to "
                  + "download them from HuggingFace."
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
              {Math.round(status.progress_pct)}% complete. Do not close
              the app.
            </p>
          </div>
        )}

        {hasError && (
          <div className="mt-6 space-y-3">
            <Alert variant="destructive">
              <AlertTitle>Setup failed</AlertTitle>
              <AlertDescription className="text-xs break-words">
                {status.error}
              </AlertDescription>
            </Alert>
            <Button onClick={() => install.mutate()} className="w-full">
              Try again
            </Button>
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
        {ok ? "Ready" : "Missing"}
      </span>
    </div>
  );
}
