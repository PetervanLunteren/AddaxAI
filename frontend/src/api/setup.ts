/**
 * First-run setup wizard API.
 *
 * Backend gates the wizard: until env-addaxai-base is installed, the rest
 * of the app is unreachable. Default model weights are already on disk by
 * the time the wizard runs (copied from the bundle by app/main.py
 * lifespan), so the wizard's only real job is to install the conda env.
 */

import { api } from "../lib/api-client";

export interface SetupStatus {
  ready: boolean;
  models_installed: boolean;
  env_installed: boolean;
  install_in_progress: boolean;
  progress_pct: number;
  message: string;
  error: string | null;
  /**
   * Names a failure the UI can offer a specific remedy for. Only
   * "tls_revocation" today, set when the environment build died because
   * Windows could not check certificate revocation and the user has not
   * already accepted skipping it. Null for every ordinary failure.
   */
  error_kind: string | null;
  user_data_dir: string;
}

/**
 * A legacy AddaxAI (v5 / v6) install found on this machine, plus the
 * progress of a removal if one is running. Presence and progress share
 * one payload so the dialog needs a single poll.
 */
export interface LegacyInstallStatus {
  found: boolean;
  version: string | null;
  /** Paths the app will delete. */
  removable_paths: string[];
  /** Paths found but needing admin rights, so the user deletes them. */
  manual_paths: string[];
  removal_in_progress: boolean;
  removal_error: string | null;
}

export const setupApi = {
  getStatus: () => api.get<SetupStatus>("/api/setup/status"),
  installEnv: () => api.post<{ status: string }>("/api/setup/install-env", {}),
  /** Wipe and rebuild specific environments (the env-drift "Update now"
   *  button). Same endpoint as installEnv, with force_envs set. */
  rebuildEnvs: (forceEnvs: string[]) =>
    api.post<{ status: string }>("/api/setup/install-env", {
      force_envs: forceEnvs,
    }),

  /**
   * Record that the user accepts building environments without a
   * certificate revocation check. Writes a marker file the backend reads
   * on every build, so one click covers the wizard, the drift rebuild
   * and model preparation alike. Starts nothing: the caller retries its
   * own build afterwards.
   */
  allowNoRevocationCheck: () =>
    api.post<{ status: string; marker_path: string }>(
      "/api/setup/allow-no-revocation-check",
      {},
    ),

  getLegacyInstall: () =>
    api.get<LegacyInstallStatus>("/api/setup/legacy-install"),
  removeLegacyInstall: () =>
    api.post<{ status: string }>("/api/setup/legacy-install/remove", {}),
};
